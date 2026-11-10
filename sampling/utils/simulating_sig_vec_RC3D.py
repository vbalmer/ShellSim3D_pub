"""Vectorised layered-shell simulator: generalised strains -> generalised stresses and tangent stiffness.

For every sample the generalised strains [eps_x, eps_y, gamma_xy, chi_x, chi_y, 2 chi_xy] are mapped to layer
strains, the layer stresses are evaluated with the RC constitutive laws (constitutive_laws.py), integrated over the
thickness to the generalised stresses [n_x, n_y, n_xy, m_x, m_y, m_xy], and the 6x6 tangent stiffness D is obtained
by the complex-step method. Vectorised port of the existing non-vectorised layered shell model (A. Näsbom) V. Balmer, 2026.
"""

import os
import sys
import time

from .config import USE_GPU, to_numpy
from .constitutive_laws import ConstitutiveLaws
from .concrete_classes import mat_from_fc
from .steel_classes import mat_from_fs

if USE_GPU:
    import cupy as np
    import numpy as np_
else:
    import numpy as np
    np_ = np


def geom_column_names(constants):
    """Ordered names of the geometry columns appended after the 6 strains (columns 6, 7, ...).

    Order: 't' (if t_values is set), then per direction 'rho_?top'/'rho_?bot' when both value lists are set, or a
    single shared 'rho_?' when only one is set (none set -> direction absent), then 'fc' and 'fs' if their value lists
    are set. This is the layout produced by append_t_geom / sample_per_t + append_geom.

    Args:
        constants (dict): sampling constants.

    Returns:
        list[str]: e.g. ['t', 'rho_xtop', 'rho_xbot', 'rho_ytop', 'rho_ybot', 'fc', 'fs'].
    """
    names = []
    if constants.get('t_values') is not None:
        names.append('t')
    for d in ('rho_x', 'rho_y'):
        top = constants.get(f'{d}top_values') is not None
        bot = constants.get(f'{d}bot_values') is not None
        if top and bot:
            names += [f'{d}top', f'{d}bot']
        elif top or bot:
            names.append(d)
    if constants.get('fc_values') is not None:
        names.append('fc')
    if constants.get('fs_values') is not None:
        names.append('fs')
    return names


class SigSimulator:
    def __init__(self, constants):
        self.constants = constants


    def build_geom(self, eps_g_batch, geom_keys=None) -> dict:
        """
        Per-sample geometry from the columns beyond the 6 strains, expanded to the simulator dict.
        The column layout is given by geom_keys; the per-layer reinforcement pattern comes from constants.

        Args:
            eps_g_batch (np.arr): (n, 6+n_geom) strains plus the appended geom columns.
            geom_keys   (list):   names of the geom columns in order; None -> geom_column_names(constants).

        Returns:
            dict: {'t': (n,1)} and/or {'rho_x': (n,nl,1)}, {'rho_y': (n,nl,1)}, for an 'fc' column
                  {'mat': {'Ec', 'fcp', 'ec0', ...: (n,1,1)}} (interpolated via mat_from_fc), and for an 'fs' column
                  {'mat_s': {'fsy', 'fsu', 'Es', 'Esh': (n,1,1)}} (via mat_from_fs); empty if no geom columns.
        """
        if eps_g_batch.shape[1] <= 6:
            return {}
        if geom_keys is None:
            geom_keys = geom_column_names(self.constants)
        n_geom = eps_g_batch.shape[1] - 6
        if len(geom_keys) != n_geom:
            raise ValueError(f"build_geom: input has {n_geom} geom column(s) but geom_keys={geom_keys} "
                             f"lists {len(geom_keys)}. The geom vector and its layout must match.")

        geom = {}
        for i, key in enumerate(geom_keys):
            col = eps_g_batch[:, 6 + i]                                          # (n,) this geom column
            if key == 't':
                geom['t'] = col[:, None]                                          # (n, 1)
            elif key in ('rho_x', 'rho_y'):                                       # single shared column
                geom[key] = self._expand_rho(col, key, half=None)
            elif key in ('rho_xtop', 'rho_ytop'):                                 # upper-half column
                direction = key[:5]
                geom[direction] = geom.get(direction, 0) + self._expand_rho(col, direction, half='top')
            elif key in ('rho_xbot', 'rho_ybot'):                                 # lower-half column
                direction = key[:5]
                geom[direction] = geom.get(direction, 0) + self._expand_rho(col, direction, half='bot')
            elif key == 'fc':                                                     # concrete strength
                # only fc is stored; the fc-dependent material parameters are interpolated from dict_CC here
                mat = mat_from_fc(col[:, None, None], xp=np)                     # each (n, 1, 1)
                geom['mat'] = {k: v.astype(col.dtype) for k, v in mat.items()}
            elif key == 'fs':                                                     # steel yield strength
                # only fs is stored; fsy, fsu, Es, Esh are interpolated from dict_S here
                mat = mat_from_fs(col[:, None, None], xp=np)                     # each (n, 1, 1)
                geom['mat_s'] = {k: v.astype(col.dtype) for k, v in mat.items()}
            else:
                raise ValueError(f"build_geom: unknown geom key '{key}'.")
        return geom


    def _expand_rho(self, col, direction, half):
        """
        Expand a per-sample reinforcement-ratio column to per-layer (n, nl, 1) using the baseline
        pattern constants[direction] to place it: half='top' -> upper-half reinforced layers,
        half='bot' -> lower-half, half=None -> all reinforced layers (shared top == bottom).
        """
        pattern = np.array(self.constants[direction], dtype=float)               # (nl,) baseline ratios
        nl = pattern.shape[0]
        layers = np.arange(nl)
        reinforced = (pattern != 0)
        if half == 'top':
            mask = reinforced & (layers >= nl // 2)
        elif half == 'bot':
            mask = reinforced & (layers <  nl // 2)
        else:
            mask = reinforced
        return col[:, None, None] * mask.astype(pattern.dtype)[None, :, None]     # (n, nl, 1)


    def find_e_vec(self, eps_g, t = None) -> np.array:
        """
        Layer strains from the generalised strains.

        Args:
            eps_g   (np.arr): generalised strains (n_tot, 6)
            t       (scalar / (n,1)): thickness; per-sample if provided, else constant constants['t']
        Returns:
            e       (np.arr): layer strains (n_tot, nl, 3)

        """
        t0 = time.perf_counter()
        nl = self.constants['n_layer']                  # int
        if t is None:
            t = self.constants['t']                     # scalar thickness

        l = np.arange(nl)                               # (nl,)
        z = -t/2 + (2*l+1)*t/(2*nl)                     # (nl,) scalar t, or (n_tot, nl) per-sample t

        # layer strain = membrane strain - z * curvature   (S = [I3 | -z*I3])
        eps_mem = eps_g[:, :3]                          # (n_tot, 3)
        chi     = eps_g[:, 3:6]                         # (n_tot, 3)
        e = eps_mem[:, np.newaxis, :] - z[..., np.newaxis] * chi[:, np.newaxis, :]   # (n_tot, nl, 3)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated layer strains e in {t1/60:.2f} min.')
        return e


    def find_s_vec(self, e, mat_dict, cm_klij=3, geom = None) -> np.array:
        """
        Layer stresses. For cm_klij = 3 the constitutive law is evaluated three times, each with a complex-step
        perturbation of one strain component, so that the imaginary parts carry the layer tangent (see get_et).

        Args:
            e         (np.arr): layer strains (n_tot, nl, 3)
            mat_dict  (dict):   material parameters
            cm_klij   (int):    1 linear elastic, 3 nonlinear RC
            geom      (dict):   per-sample geometry overrides passed to ConstitutiveLaws (see build_geom)
        Returns:
            s         (np.arr): layer stresses (n_tot, nl, 3, 3) complex for cm_klij = 3, (n_tot, nl, 3) for 1.

        """
        t0 = time.perf_counter()

        if cm_klij == 3:
            # Reinforced Concrete Calculation
            s = np.zeros((e.shape[0], e.shape[1], 3, 3), dtype=np.complex64)

            e0 = e+np.array([0.0000000000000001j,0,0])
            material_law0 = ConstitutiveLaws(e0, self.constants, mat_dict, cm_klij=cm_klij, geom=geom)
            s[:,:,0,:] = material_law0.out().squeeze(-1)
            t1 =(time.perf_counter()-t0)
            print(f'Calculated 1/3 instance of layer stresses s in {t1/60:.2f} min.')

            e1 = e+np.array([0,0.0000000000000001j,0])
            material_law1 = ConstitutiveLaws(e1, self.constants, mat_dict, cm_klij=cm_klij, geom=geom)
            s[:,:,1,:] = material_law1.out().squeeze(-1)
            t2 =(time.perf_counter()-t0)
            print(f'Calculated 2/3 instance of layer stresses s in {t2/60:.2f} min.')

            e2 = e+np.array([0,0,0.0000000000000001j])
            material_law2 = ConstitutiveLaws(e2, self.constants, mat_dict, cm_klij=cm_klij, geom=geom)
            s[:,:,2,:] = material_law2.out().squeeze(-1)
            t3 =(time.perf_counter()-t0)
            print(f'Calculated 3/3 instance of layer stresses s in {t3/60:.2f} min.')

        elif cm_klij == 1:
            # Linear Elastic calculation
            material_law0 = ConstitutiveLaws(e, self.constants, mat_dict, cm_klij=cm_klij, geom=geom)
            s = material_law0.out().real
            t1 =(time.perf_counter()-t0)
            print(f'Calculated linear elastic layer stresses s in {t1/60:.2f} min.')

        return s


    def find_sh_vec(self, s, cm_klij = 3, t = None) -> np.array:
        """
        Generalised stresses by integrating the layer stresses over the thickness.

        Args:
            s   (np.arr): layer stresses (n_tot, nl, 3, 3)
            t   (scalar / (n,1)): thickness; per-sample if provided, else constant constants['t']
        Returns:
            sh  (np.arr): generalised stresses (n_tot, 6), real

        """
        t0 = time.perf_counter()
        n_tot = s.shape[0]
        nl = self.constants['n_layer']                  # int
        if t is None:
            t = self.constants['t']                     # scalar thickness
        l = np.arange(nl)                               # shape (nl,)
        z = -t/2+(2*l+1)*t/(2 * nl)                     # (nl,) scalar t, or (n_tot, nl) per-sample t

        if cm_klij == 3:
            s = s[:,:,0,:].real                         # shape (n_tot, nl, 3)
        elif cm_klij == 1:
            s = s.reshape(n_tot, nl, 3)                 # shape (n_tot, nl, 3)

        # generalised stress = (t/nl) * sum_layers([s ; -z*s])   (S = [I3 | -z*I3])
        sh_layer = np.concatenate([s, -z[..., np.newaxis] * s], axis=-1)   # (n_tot, nl, 6)
        sh = (t/nl) * sh_layer.sum(axis = 1)            # (n_tot, 6); (n,1)*(n,6) or scalar*(n,6)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated generalised stresses sh in {t1/60:.2f} min.')
        return sh


    def find_dh_vec(self, s, mat_dict, cm_klij = 3, t = None, geom = None) -> np.array:
        """
        Generalised tangent stiffness D = [[Dm, Dmb], [Dbm, Db]] by integrating the layer tangents.

        Args:
            s         (np.arr): layer stresses (n_tot, nl, 3, 3)
            mat_dict  (dict):   material parameters
            t         (scalar / (n,1)): thickness; per-sample if provided, else constant constants['t']
            geom      (dict):   per-sample geometry from build_geom; its 'mat' entry (per-sample fc) overrides
                                mat_dict['Ec'] in the linear-elastic (cm_klij = 1) stiffness
        Returns:
            dh        (np.arr): stiffness matrix entries (n_tot, 6, 6)

        """
        t0 = time.perf_counter()
        nl = self.constants['n_layer']
        if t is None:
            t = self.constants['t']
        l = np.arange(nl)                               # shape (nl,)
        z = -t / 2 + (2 * l + 1) * t / (2 * nl)         # (nl,) scalar t, or (n_tot, nl) per-sample t

        Dp = self.get_et(s, mat_dict, cm_klij = cm_klij, geom = geom)       # shape (n_tot, nl, 3, 3)

        # complex step fills one row per pushed strain (Dp[k, j] = dsig_j/deps_k) -> swap to row = stress,
        # Dp[j, k] = dsig_j/deps_k (matters where Dp is not symmetric: cracked)
        Dp = np.swapaxes(Dp, 2, 3)

        # z_ broadcastable with Dp; scale_ broadcastable with the (n_tot, 3, 3) sums
        z_ = z.reshape(z.shape[0], nl, 1, 1) if z.ndim == 2 else z.reshape(1, nl, 1, 1)
        scale = t / nl
        scale_ = scale.reshape(-1, 1, 1) if getattr(scale, 'ndim', 0) > 0 else scale

        Dmh     = np.sum((Dp)       , axis = 1)*scale_             # shape (n_tot, 3, 3)
        Dmbh    = np.sum((-z_*Dp)   , axis = 1)*scale_             # shape (n_tot, 3, 3)
        Dbh     = np.sum((z_**2*Dp) , axis = 1)*scale_             # shape (n_tot, 3, 3)

        De_1 = np.concatenate([Dmh, Dmbh], axis=2)          # (n_tot, 3, 6)
        De_2 = np.concatenate([Dmbh, Dbh], axis=2)          # (n_tot, 3, 6)
        De   = np.concatenate([De_1, De_2], axis=1)         # (n_tot, 6, 6)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated stiffness matrix D in {t1/60:.2f} min.')
        return De


    def get_et(self, s, mat_dict, cm_klij = 3, geom = None) -> np.array:
        """
        Per-layer tangent stiffness matrix.

        Args:
            s         (np.arr): layer stresses (n_tot, nl, 3, 3)
            mat_dict  (dict):   material parameters
            cm_klij   (int):    1 linear elastic, 3 nonlinear RC (complex-step tangent: imag(s) / h)
            geom      (dict):   per-sample geometry; geom['mat']['Ec'] (n,1,1) overrides mat_dict['Ec'] if present
        Returns:
            dp        (np.arr): stiffness matrix entries (n_tot, nl, 3, 3)
        """

        n_tot = s.shape[0]
        nl = self.constants['n_layer']

        if cm_klij == 1:
            v = self.constants['nu']
            ET_unit = 1 / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
            if geom and 'mat' in geom:
                E = geom['mat']['Ec'].reshape(n_tot, 1, 1, 1)                     # per-sample Ec
                dp = np.broadcast_to(E * ET_unit[np.newaxis, np.newaxis, :, :], (n_tot, nl, 3, 3))
            else:
                ET = mat_dict['Ec'] * ET_unit
                dp = np.broadcast_to(ET[np.newaxis, np.newaxis,:,:], (n_tot, nl, 3,3))

        elif cm_klij == 3:
            dp = s.imag / 1e-16                         # complex-step derivative, step h = 1e-16

        return dp


######################### batchwise calculations - wrapper #########################

class HiddenPrints:
    """Context manager that silences stdout."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def sig_simulation_batchwise(eps_g, simulatesig:SigSimulator, cm, mat_dict, n_batches = 500):
    """
    Batch-wise calculation of sig_g and D (bounds GPU memory).

    Args:
        eps_g           (np.arr)        : sampled strains + geom columns (ntot, 6 + n_geom), on the backend device
        simulatesig     (SigSimulator)  : simulator
        cm              (int)           : material model (1 = lin. el., 3 = RC)
        mat_dict        (dict)          : material properties
        n_batches       (int)           : number of batches

    Returns:
        sig_g           (np.arr)        : simulated generalised stresses (ntot, 6), numpy
        dh              (np.arr)        : simulated stiffness            (ntot, 6, 6), numpy

    """
    sig_g = np_.zeros((eps_g.shape[0], 6))
    dh = np_.zeros((eps_g.shape[0],6,6))
    n_batches = int(n_batches)
    batch_size = -(-eps_g.shape[0] // n_batches)   # ceil -> covers every row

    t00 = time.perf_counter()
    for i in range(n_batches):
        if i*batch_size >= eps_g.shape[0]:
            break                                   # ceil can leave trailing empty batches; stop early
        if i%10 != 0:
            with HiddenPrints():
                sig_g, dh,_ = single_batch_execution(i, batch_size, simulatesig, mat_dict, cm, dh, sig_g, eps_g)
        else:
            sig_g, dh,t0 = single_batch_execution(i, batch_size, simulatesig, mat_dict, cm, dh, sig_g, eps_g)
            print(f'Finished batch {i+1}/{n_batches} with batchsize = {batch_size} in {(time.perf_counter() - t0)/60:.2f} min.')

        if USE_GPU:
            np.get_default_memory_pool().free_all_blocks()   # release GPU blocks so the pool doesn't fragment

    print(f'Finished calculation of all {n_batches} batches in {(time.perf_counter()-t00)/60:.2f} min')
    return sig_g, dh


def single_batch_execution(i:int, batch_size: int, simulatesig:SigSimulator, mat_dict, cm, dh, sig_g, eps_g):
    """Simulate rows [i*batch_size, (i+1)*batch_size) of eps_g and write them into sig_g / dh."""
    t0 = time.perf_counter()

    start = i*batch_size
    end = min((i+1)*batch_size, eps_g.shape[0])     # clamp so the last batch covers the remainder
    eps_g_batch = eps_g[start:end,:]
    geom = simulatesig.build_geom(eps_g_batch)
    t = geom.get('t')

    e = simulatesig.find_e_vec(eps_g_batch[:, :6], t = t)                          # layer strains
    s = simulatesig.find_s_vec(e, mat_dict, cm_klij = cm, geom = geom)             # layer stresses
    sig_g_batch = simulatesig.find_sh_vec(s, cm_klij = cm, t = t)                  # generalised stresses
    dh_batch = simulatesig.find_dh_vec(s, mat_dict, cm_klij = cm, t = t, geom = geom)   # stiffness

    sig_g[start:end,:] = to_numpy(sig_g_batch).astype(sig_g.dtype)
    dh[start:end,:] = to_numpy(dh_batch).astype(dh.dtype)

    return sig_g, dh, t0
