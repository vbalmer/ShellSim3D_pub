"""Network, optimiser and losses for the Sobolev-trained surrogate of the layered RC shell section.

FFNN maps [eps(6), geom] -> sig(6); the tangent D = d sig / d eps is the network Jacobian (autograd), so the
Sobolev loss trains sig and D jointly. V. Balmer, 2026.
"""

import ast

import torch
from torch import nn
from torch import vmap


############################################### Loss classes ###############################################

class wbMSELoss(nn.Module):
    """Weighted-bending MSE. Up-weights the squared error of near-zero-target entries (|target| < band) by
    `weight`; everything else is plain MSE:
      * stress vector (n,6): bending moment columns `moments` (membrane columns untouched)
      * D tangent   (n,6,6): coupling blocks Dmb = D[membrane,moments] and Dbm = D[moments,membrane]
                             (Dm and Db blocks untouched)
    Called as criterion(prediction, target).
    """

    def __init__(self, weight=10.0, band=0.5, band_D=None, moments=slice(3, 6), membrane=slice(0, 3)):
        super().__init__()
        self.weight, self.band = weight, band
        self.band_D = band if band_D is None else band_D
        self.moments, self.membrane = moments, membrane

    def forward(self, y, y_hat):
        se = (y - y_hat) ** 2
        w = torch.ones_like(se)
        _apply_bending_weights(y, y_hat, w, self.weight, self.band, self.band_D, self.moments, self.membrane)
        return (w * se).mean()


def _apply_bending_weights(y, y_hat, w, weight, band, band_D, moments, membrane):
    """Set (in place) w = weight where the target y_hat of a moment / coupling entry is near zero (see wbMSELoss).
    Unknown layouts leave w untouched (plain MSE)."""
    m, b = moments, membrane
    mask = torch.zeros_like(w, dtype=torch.bool)
    if y.dim() == 2 and y.shape[1] >= 6:                          # stress vector
        mask[:, m] = y_hat[:, m].abs() < band
    elif y.dim() == 3 and y.shape[-1] >= 6 and y.shape[-2] >= 6:  # D tangent (n,6,6)
        mask[:, b, m] = y_hat[:, b, m].abs() < band_D             # Dmb = dn/dchi
        mask[:, m, b] = y_hat[:, m, b].abs() < band_D             # Dbm = dm/deps
    w[mask] = weight


loss_mapping = {
    'MSELoss': nn.MSELoss,
    'wbMSELoss': wbMSELoss,
}


def build_loss(loss_type, inp):
    """Instantiate loss_type ('MSELoss' or 'wbMSELoss'); weight / band from inp are forwarded to wbMSELoss."""
    cls = loss_mapping[loss_type]
    kwargs = {}
    if loss_type == 'wbMSELoss':
        if inp.get('weight') is not None: kwargs['weight'] = inp['weight']
        if inp.get('band')   is not None: kwargs['band']   = inp['band']
    return cls(**kwargs)


def bsymm_swap_pairs(input_size):
    """rho_top <-> rho_bot geom-column index pairs for the mid-plane reflection (used by symm_predict).

    Args:
        input_size (int): network input width in the layout
            [eps_x, eps_y, gamma_xy, chi_x, chi_y, chi_xy, t, rho_xtop, rho_xbot, rho_ytop, rho_ybot(, fc(, fs))].

    Returns:
        list[tuple[int,int]]: (7,8), (9,10) for 5, 6 or 7 geom columns; [] otherwise (no top/bottom split).
    """
    return [(7, 8), (9, 10)] if input_size - 6 in (5, 6, 7) else []


def xy_swap_pairs(input_size):
    """rho_x <-> rho_y geom-column index pairs for the x <-> y relabelling (used by xysymm_predict).

    Args:
        input_size (int): network input width (see bsymm_swap_pairs).

    Returns:
        list[tuple[int,int]]: (7,9), (8,10) for 5, 6 or 7 geom columns; (7,8) for [t, rho_x, rho_y]; [] otherwise.
    """
    n_geom = input_size - 6
    if n_geom in (5, 6, 7):
        return [(7, 9), (8, 10)]        # rho_xtop <-> rho_ytop, rho_xbot <-> rho_ybot
    if n_geom == 3:
        return [(7, 8)]                 # rho_x <-> rho_y
    return []


############################################### Architecture ###############################################

class FFNN(nn.Module):
    """Fully connected network: input_size -> hidden_layers -> out_size, with the activation from inp."""

    def __init__(self, inp: dict):
        super().__init__()
        self.inp = inp
        self.activation = getattr(nn, self.inp['activation'])
        self.hidden_layers = ast.literal_eval(self.inp['hidden_layers'])

        layers = []
        for i in range(len(self.hidden_layers) - 1):
            if self.inp['BatchNorm']:
                layers.append(nn.BatchNorm1d(self.hidden_layers[i]))
            layers.append(nn.Dropout(self.inp['dropout_rate']))
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(self.activation())

        self.layers = nn.Sequential(
            nn.Linear(self.inp['input_size'], self.hidden_layers[0]),
            self.activation(),
            *layers,
            nn.Linear(self.hidden_layers[-1], self.inp['out_size'])
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


############################################### Optimiser ###############################################

class Adam_LBFGS(torch.optim.Optimizer):
    """AdamW, switching to LBFGS after switch_step optimisation steps (switch_step = inf -> Adam only)."""

    def __init__(self,
                 inp,
                 params,
                 switch_step,
                 lbfgs_params=None,
                 adam_hyper={"betas": (0.9, 0.99)},
                 lbfgs_hyper={"lr": 1., "max_iter": 20, "history_size": 100}):
        self._params = list(params)
        self.switch_step = switch_step
        self.inp = inp
        self.adam = torch.optim.AdamW(self._params, lr=self.inp['learning_rate'], **adam_hyper)
        self.lbfgs = torch.optim.LBFGS(self._params if lbfgs_params is None else lbfgs_params, **lbfgs_hyper)
        defaults = {}
        super().__init__(self._params, defaults)
        self.state["step"] = 0
        self.state["using_lbfgs"] = False

    def step(self, closure):  # type: ignore[override]
        """closure() zeroes the gradients, computes the loss, calls backward and returns (loss, loss_logs)."""
        self.state["step"] += 1

        # Phase I: Adam
        if not self.state["using_lbfgs"]:
            loss, loss_logs = closure()
            self.adam.step()
            if self.state["step"] >= self.switch_step:
                print(f"[Adam_LBFGS] Switching to LBFGS at optimiser step {self.state['step']}")
                self.state["using_lbfgs"] = True
            return loss, loss_logs

        # Phase II: LBFGS (calls the closure several times internally)
        def lbfgs_closure():
            return closure()[0]
        loss = self.lbfgs.step(lbfgs_closure)
        return loss, closure(backward=False)[1]

    def using_lbfgs(self):
        return self.state["using_lbfgs"]


############################################### Sobolev loss ###############################################

class CustomLosses:
    """Sobolev loss: w1 * L(sig_pred, sig) + w2 * L(D_pred, D), with D_pred the Jacobian of the network w.r.t.
    its inputs (strain columns), all in normalised coordinates."""

    def __init__(self, inp: dict, ffnn: FFNN):
        self.ffnn = ffnn
        self.inp = inp
        self.criterion = build_loss(self.inp['loss_type'], self.inp)

    def Sobolev_CustomLoss(self, features, labels):
        if not self.inp['Sobolev']:
            raise SyntaxError("To access Sobolev_CustomLoss please set inp['Sobolev'] = True")

        loss1 = self.get_first_order_loss(features, labels)      # stresses
        loss2 = self.get_second_order_loss(features, labels)     # tangent stiffness
        return self.get_combined_loss(loss1, loss2)

    def get_first_order_loss(self, features, labels):
        features.requires_grad_(True)
        predictions = self.ffnn(features)
        target = labels[:, 0:self.inp['out_size']]
        return self.criterion(predictions, target)

    def get_second_order_loss(self, features, labels):
        J = vmap(torch.func.jacrev(self.ffnn.forward), randomness='different')(features)

        sz = self.inp['out_size']
        if len(J.shape) > 3:
            J = J.squeeze(1)
        # loss on the full 6x6 strain block of the Jacobian (geometry-input derivatives are not used)
        D_lbl = labels[:, sz:].reshape(-1, sz, sz)
        crit = self.criterion if self.inp['loss_type_D'] is None else build_loss(self.inp['loss_type_D'], self.inp)
        return crit(J[:, :sz, :sz], D_lbl)

    def get_combined_loss(self, loss1, loss2):
        if self.inp['w_s'] == 'max':
            loss = max(loss1, loss2)
        else:
            w_s = ast.literal_eval(self.inp['w_s'])
            loss = w_s[0] * loss1 + w_s[1] * loss2

        loss_logs = {"loss1": loss1,
                     "loss2": loss2,
                     "train_loss": loss}
        return loss, loss_logs
