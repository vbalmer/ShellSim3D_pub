"""Inference with a trained model: NNPredictor (loads a model version once and predicts stresses sig and
stiffnesses D in physical units) and the symmetry wrappers (mid-plane, x <-> y). V. Balmer, 2026.

    pred = NNPredictor.from_folder(path, version)
    out  = pred.predict(X)                                  # {'sig': (n, 6), 'D': (n, 6, 6)} in [N, mm]
    out  = pred.predict(X, symmetry='midplane', one_pass=True)

X = [6 strains, geom] in physical units, as in the sampled data.
"""

import os
import glob
import copy
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.func import vmap, jacrev

from .architectures import FFNN, bsymm_swap_pairs, xy_swap_pairs
from .data_utils import (transform_data, scale_inputs, apply_t_norm, undo_t_norm, apply_fc_norm, undo_fc_norm)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')

SIG_TYPES = ['y-std'] * 6
Y_TYPES = ['y-std'] * 6 + ['y-st-stitched'] * 36       # [sig(6), D(36)]; transform_data needs this layout for D
OOM_BATCH_SIZE = 265                                     # jacobian batch size after an out-of-memory error


################################ loading a model version ################################

def get_inp_from_folder(path: str, version: int) -> dict:
    """Read the hyperparameters (inp.pkl) of model version path/v_<version>."""
    with open(os.path.join(path, f'v_{version}', 'inp.pkl'), 'rb') as handle:
        return pickle.load(handle)


def get_stats_from_folder(path: str, version: int) -> dict:
    """Read the normalisation statistics (stats.pkl) of model version path/v_<version>."""
    with open(os.path.join(path, f'v_{version}', 'stats.pkl'), 'rb') as handle:
        return pickle.load(handle)


def get_full_model_path(path: str, version: int) -> str:
    """Path of the best_trained_model_<epoch>.pt file in path/v_<version>."""
    matches = glob.glob(os.path.join(path, f'v_{version}', '**', 'best_trained_model_*.pt'), recursive=True)
    model_path = matches[0]
    epoch = Path(model_path).stem.replace('best_trained_model_', '')
    print(f'Testing model v_{version} with best trained model at epoch {epoch}')
    return model_path


def test_model_instance(inp: dict, path: str, version: int) -> FFNN:
    """Trained model of path/v_<version> on `device`, in eval mode."""
    model = FFNN(inp)
    model.load_state_dict(torch.load(get_full_model_path(path, version), map_location=device))
    model.to(device)
    model.eval()
    return model


################################ symmetries ################################

class Symmetry:
    """Symmetry enforced at inference: g(x) = 0.5 * [f(x) + out(f(inp(x)))], all in physical units."""

    def inp(self, X: np.ndarray) -> np.ndarray:
        """Transformed copy of the physical inputs X (n, 6 + n_geom)."""
        raise NotImplementedError

    def out_sig(self, sig: np.ndarray) -> np.ndarray:
        """Stresses (n, 6) of the transformed input mapped back."""
        raise NotImplementedError

    def out_D(self, D: np.ndarray) -> np.ndarray:
        """Tangents (n, 6, 6) of the transformed input mapped back."""
        raise NotImplementedError


class MidplaneSymmetry(Symmetry):
    """Mid-plane reflection: curvatures chi -> -chi and rho_top <-> rho_bot; the moments change sign, so D has
    the moment rows and the curvature columns negated."""

    def __init__(self, input_size: int):
        self.swap_pairs = bsymm_swap_pairs(input_size)

    def inp(self, X):
        Xr = X.copy()
        Xr[:, 3:6] = -Xr[:, 3:6]
        for i, j in self.swap_pairs:
            Xr[:, [i, j]] = Xr[:, [j, i]]
        return Xr

    def out_sig(self, sig):
        sig = sig.copy()
        sig[:, 3:6] = -sig[:, 3:6]
        return sig

    def out_D(self, D):
        D = D.copy()
        D[:, 3:6, :] = -D[:, 3:6, :]
        D[:, :, 3:6] = -D[:, :, 3:6]
        return D


class XYSymmetry(Symmetry):
    """x <-> y relabelling: eps_x <-> eps_y, chi_x <-> chi_y, rho_x <-> rho_y; n_x <-> n_y and m_x <-> m_y."""

    PERM = [1, 0, 2, 4, 3, 5]

    def __init__(self, input_size: int):
        self.swap_pairs = [(0, 1), (3, 4)] + xy_swap_pairs(input_size)

    def inp(self, X):
        Xr = X.copy()
        for i, j in self.swap_pairs:
            Xr[:, [i, j]] = Xr[:, [j, i]]
        return Xr

    def out_sig(self, sig):
        return sig[:, self.PERM]

    def out_D(self, D):
        return D[:, self.PERM, :][:, :, self.PERM]


SYMMETRIES = {'midplane': MidplaneSymmetry, 'xy': XYSymmetry}


################################ predictor ################################

class NNPredictor:
    """
    Trained model with its hyperparameters and normalisation statistics.

    Args:
        model   (FFNN): trained model (eval mode); kept on its device for the forward pass, a CPU copy is used
                        for the jacobian (D)
        inp     (dict): hyperparameters (input_size, batch_size, Sobolev)
        stats   (dict): normalisation statistics of the training data (incl. optional 't_norm' / 'fc_norm')
    """

    def __init__(self, model: FFNN, inp: dict, stats: dict):
        self.model = model
        self.inp = inp
        self.stats = stats
        self._model_cpu = None
        self.t_norm = stats.get('t_norm') if isinstance(stats, dict) else None
        self.fc_norm = stats.get('fc_norm') if isinstance(stats, dict) else None

    @classmethod
    def from_folder(cls, path: str, version: int):
        """Load model version path/v_<version> (best_trained_model_*.pt, inp.pkl, stats.pkl)."""
        inp = get_inp_from_folder(path, version)
        return cls(test_model_instance(inp, path, version), inp, get_stats_from_folder(path, version))

    @property
    def model_cpu(self):
        if self._model_cpu is None:
            p = next(self.model.parameters())
            self._model_cpu = self.model if p.device == cpu else copy.deepcopy(self.model).to(cpu)
        return self._model_cpu

    @property
    def batch_size(self):
        return self.inp.get('batch_size')

    # ---------------- normalisation (physical <-> network space) ----------------

    def _t_phys(self, X):
        return None if self.t_norm is None else X[:, self.t_norm['t_idx']]

    def _fc_phys(self, X):
        return None if self.fc_norm is None else X[:, self.fc_norm['fc_idx']]

    def normalise_X(self, X: np.ndarray) -> np.ndarray:
        """Physical inputs (n, 6 + n_geom) -> network inputs (t-scaling of the strains, then standardisation)."""
        X = scale_inputs(X, self.t_norm)
        return transform_data(X, self.stats, forward=True, type=['x-std'] * X.shape[1])

    def unnormalise_sig(self, sig_n: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Network stresses (n, 6) -> [N, mm]; X: physical inputs (for t and fc)."""
        sig = transform_data(sig_n, self.stats, forward=False, type=SIG_TYPES)
        return undo_fc_norm(undo_t_norm(sig, self._t_phys(X), self.stats), self._fc_phys(X), self.stats)

    def normalise_sig(self, sig: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Inverse of unnormalise_sig."""
        sig = apply_fc_norm(apply_t_norm(sig, self._t_phys(X), self.stats), self._fc_phys(X), self.stats)
        return transform_data(sig, self.stats, forward=True, type=SIG_TYPES)

    def unnormalise_D(self, D_n: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Network tangents (n, 6, 6) (jacobian w.r.t. the normalised strains) -> [N, mm]."""
        y = transform_data(self._pad_D(D_n), self.stats, forward=False, type=Y_TYPES)
        y = undo_fc_norm(undo_t_norm(y, self._t_phys(X), self.stats), self._fc_phys(X), self.stats)
        return y[:, 6:].reshape((-1, 6, 6))

    def normalise_D(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Inverse of unnormalise_D."""
        y = apply_fc_norm(apply_t_norm(self._pad_D(D), self._t_phys(X), self.stats), self._fc_phys(X), self.stats)
        return transform_data(y, self.stats, forward=True, type=Y_TYPES)[:, 6:].reshape((-1, 6, 6))

    @staticmethod
    def _pad_D(D):
        """(n, 6, 6) -> (n, 42) with zero stress columns: the [sig, D] layout of the t-factors and transform_data."""
        return np.concatenate((np.zeros((D.shape[0], 6)), D.reshape((-1, 36))), axis=1)

    # ---------------- network calls (network space in and out) ----------------

    def _sig_norm(self, Xn: torch.Tensor) -> np.ndarray:
        """Forward pass, batchwise on the model's device. Returns (n, out_size)."""
        dev = next(self.model.parameters()).device
        bs = self.batch_size or Xn.shape[0]
        with torch.no_grad():
            out = [self.model(xb.to(dev)) for xb in torch.split(Xn, bs)]
        return torch.cat(out, dim=0).cpu().numpy()

    def _jac(self, X: torch.Tensor) -> torch.Tensor:
        """Per-sample jacobian d out / d strains (n, out_size, 6) on the CPU; geometry columns not differentiated."""
        m = self.model_cpu
        f = lambda eps, geom: m(torch.cat([eps, geom]))
        return vmap(jacrev(f, argnums=0), randomness='different')(X[:, :6], X[:, 6:])

    def _D_norm(self, Xn: torch.Tensor, batched: bool = True) -> np.ndarray:
        """Jacobian, batchwise (inp['batch_size']; smaller batches after an out-of-memory error) or in one go."""
        Xn = Xn.to(cpu)
        if not batched:
            return self._jac(Xn).detach().numpy()
        bs = self.batch_size or Xn.shape[0]
        try:
            return np.concatenate([self._jac(xb).detach().numpy() for xb in torch.split(Xn, bs)], axis=0)
        except (torch.cuda.OutOfMemoryError, MemoryError):
            print(f'OOM in the jacobian, retrying with batch_size={OOM_BATCH_SIZE}')
            return np.concatenate([self._jac(xb).detach().numpy() for xb in torch.split(Xn, OOM_BATCH_SIZE)], axis=0)

    def _sig_D_norm(self, Xn: torch.Tensor):
        """Stresses and jacobian from one pass on the CPU (jacrev with has_aux)."""
        m = self.model_cpu
        Xn = Xn.to(cpu)

        def f(eps, geom):
            out = m(torch.cat([eps, geom]))
            return out, out

        J, out = vmap(jacrev(f, argnums=0, has_aux=True), randomness='different')(Xn[:, :6], Xn[:, 6:])
        return out.detach().numpy(), J.detach().numpy()

    # ---------------- prediction in physical units ----------------

    def _predict_plain(self, X, sig, D, one_pass, batched):
        Xn = torch.as_tensor(self.normalise_X(X), dtype=torch.float32)
        if one_pass and sig and D:
            sig_n, D_n = self._sig_D_norm(Xn)
        else:
            sig_n = self._sig_norm(Xn) if sig else None
            D_n = self._D_norm(Xn, batched) if D else None
        return {'sig': None if sig_n is None else self.unnormalise_sig(sig_n, X),
                'D': None if D_n is None else self.unnormalise_D(D_n, X)}

    def predict(self, X: np.ndarray, sig: bool = True, D: bool = True, symmetry=None, one_pass: bool = False,
                batched: bool = True) -> dict:
        """
        Stresses and tangents in [N, mm].

        Args:
            X        (np.arr):  physical inputs [6 strains, geom], shape (n, input_size)
            sig      (bool):    compute the stresses
            D        (bool):    compute the tangents (jacobian w.r.t. the strains; the expensive part)
            symmetry (str|Symmetry|None): None, 'midplane', 'xy' or a Symmetry; averages f(x) and the mapped
                                f(inp(x)) (two predictions)
            one_pass (bool):    sig and D from one CPU pass (needs sig and D); else forward pass on the model's
                                device and a separate jacobian pass
            batched  (bool):    jacobian batchwise (inp['batch_size']); False: whole input at once (small inputs,
                                e.g. in an FE solver)

        Returns:
            dict: 'sig' (n, 6) and 'D' (n, 6, 6); None for a part that is not computed
        """
        if isinstance(symmetry, str):
            symmetry = SYMMETRIES[symmetry](self.inp['input_size'])
        out = self._predict_plain(X, sig, D, one_pass, batched)
        if symmetry is None:
            return out

        out_r = self._predict_plain(symmetry.inp(X), sig, D, one_pass, batched)
        if sig:
            out['sig'] = 0.5 * (out['sig'] + symmetry.out_sig(out_r['sig']))
        if D:
            out['D'] = 0.5 * (out['D'] + symmetry.out_D(out_r['D']))
        return out
