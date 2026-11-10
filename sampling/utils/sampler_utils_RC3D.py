"""Sampling utilities for the 3D RC shell sampler: strain sampling, geometry / material draw, admissibility
filter, outlier removal and saving. See ../README.md for the method and the meaning of the constants.
V. Balmer, 2026.
"""

import os
import time

import h5py
import numpy as np
import pyDOE as doe

from .config import USE_GPU, to_numpy
from .concrete_classes import mat_from_fc
from .steel_classes import mat_from_fs
from .simulating_sig_vec_RC3D import geom_column_names

if USE_GPU:
    import cupy as xp
else:
    xp = np


STRAIN_NAMES = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy']


########################################## Setup ##########################################

def get_constant_sampling_params(constants: dict) -> tuple:
    """
    Copy of the sampling constants and the default material dict derived from them.

    Args:
        constants (dict): sampling constants (constant_sampling_params.constants).

    Returns:
        constants (dict): shallow copy of the input (the module-level dict is not mutated).
        mat_dict  (dict): material parameters for the scalar fc / fs (per-sample fc / fs geom columns override
                          these, see SigSimulator.build_geom), plus the concrete law switches cmcc / cmcs.
    """
    constants = dict(constants)
    mat_dict = mat_from_fc(constants['fc'])
    mat_dict.update(mat_from_fs(constants['fs']))
    mat_dict.update({'cmcc': constants.get('cmcc', 1), 'cmcs': constants.get('cmcs', 1)})
    return constants, mat_dict


def setup_sampler_dirs(constants: dict) -> tuple:
    """
    Apply the N_SAMPLES_3D environment override (if set) and resolve the output directory.

    The output directory is $DATA_DIR if set, else sampling/data next to the main script.

    Args:
        constants (dict): sampling constants.

    Returns:
        constants     (dict): same dict, with n_samples_3D updated if overridden.
        save_data_dir (str):  output directory (created if it does not exist).
    """
    if os.environ.get('N_SAMPLES_3D'):
        constants['n_samples_3D'] = int(float(os.environ['N_SAMPLES_3D']))
        print(f"[sampler] N_SAMPLES_3D override -> {constants['n_samples_3D']:,}")

    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    save_data_dir = os.environ.get('DATA_DIR') or default_dir
    os.makedirs(save_data_dir, exist_ok=True)
    print(f'[sampler] Saving data to: {save_data_dir}')

    return constants, save_data_dir


def initialise_wandb(constants, mat_dict, log_wandb, project = 'sampler_3DRC'):
    """Start a Weights & Biases run with the sampling constants as config (only if log_wandb)."""
    if log_wandb:
        import wandb
        wandb.init(config={**constants, **mat_dict}, project=project)


########################################## Sampling strains ##########################################

def resolve_combined_counts(constants, n_dims):
    """
    Pre-filter sample counts (log, uniform) for the combined samplers.

    Per-axis mode (n_samples_3D is None and n_per_axis_uni is set): n_per_axis_log**n_dims log points and
    n_per_axis_uni**n_dims uniform grid points. Otherwise n_samples_3D is split by p_samples_log (None -> half/half).

    Args:
        constants (dict): sampling constants.
        n_dims    (int):  number of sampled dimensions (6).

    Returns:
        (log_samples, uniform_samples) (tuple[int])
    """
    n_per_axis_uni = constants.get('n_per_axis_uni')
    n_per_axis_log = constants.get('n_per_axis_log')
    per_axis_mode = constants.get('n_samples_3D') is None and n_per_axis_uni is not None

    if per_axis_mode:
        if n_per_axis_log is None:
            n_per_axis_log = n_per_axis_uni
        log_samples     = int(n_per_axis_log ** n_dims)
        uniform_samples = int(n_per_axis_uni ** n_dims)
        print(f'Per-axis sampling: uniform {n_per_axis_uni}/axis -> {uniform_samples:,} pts, '
              f'log {n_per_axis_log}/axis -> {log_samples:,} pts (pre-filter, before force-zero nodes).')
    elif constants['p_samples_log'] is not None:
        log_samples     = int(constants['n_samples_3D'] * constants['p_samples_log'])
        uniform_samples = int(constants['n_samples_3D'] * (1 - constants['p_samples_log']))
    else:
        log_samples     = int(constants['n_samples_3D'] / 2)
        uniform_samples = log_samples

    return log_samples, uniform_samples


def min_section_t(constants):
    """Thinnest sampled section thickness: min(t_values) if set, else the scalar constants['t']."""
    return min(constants['t_values']) if constants.get('t_values') is not None else constants['t']


def sample_eps(sampler: str, constants: dict, sampler_type_log = 'lhs', zero_value_epsx = 0.5e-6) -> tuple:
    """
    Sample generalised strains [eps_x, eps_y, eps_xy, chi_x, chi_y, chi_xy].

    Args:
        sampler          (str):   'uniform_3D'          full 6D grid
                                  'uniform_3D_grouped'  6D grid built as product of two 3D grids (memory-efficient)
                                  'lhs'                 Latin hypercube in [min, max]
                                  'log'                 log-spaced magnitudes in [zero value, min_log/max_log]
                                  'combined_log_uniform', 'combined_log_lhs', 'combined_lhs_uniform'
        constants        (dict):  sampling constants.
        sampler_type_log (str):   how the exponents of the log block are sampled ('lhs' or 'uniform_3D_grouped').
        zero_value_epsx  (float): smallest membrane strain magnitude of the log block (curvature: divided by t/2).

    Returns:
        data    (np.arr): sampled strains (n, 6), rows ordered subset by subset.
        subsets (list):   [(name, n_rows), ...] row counts of the subsets in row order.
    """
    n_dims = len(STRAIN_NAMES)
    force_zero = constants.get('force_zero_axes', False)    # insert an exact 0 node in each uniform axis

    def _grid(mn, mx, n):
        t0 = time.perf_counter()
        d = samplers(STRAIN_NAMES, mn, mx, samples=n).uniform_multi_grouped(force_zero=force_zero)
        print(f'Sampled {d.shape[0]:,} uniform grid points in {(time.perf_counter()-t0)/60:.2f} min')
        return d

    def _lhs(n):
        t0 = time.perf_counter()
        d = samplers(STRAIN_NAMES, constants['min'], constants['max'], samples=n).lhs(criterion='c')
        print(f'Sampled {d.shape[0]:,} LHS points in {(time.perf_counter()-t0)/60:.2f} min')
        return d

    def _log(n):
        exponents, max_log_neg, max_log_pos = sample_exponents(constants, sampler_type_log, zero_value_epsx,
                                                               num_samples=n, force_zero=force_zero)
        return convert_log_data_to_eps(exponents, max_log_neg, max_log_pos)

    if sampler == 'uniform_3D':
        data = samplers(STRAIN_NAMES, constants['min'], constants['max'], samples=constants['n_samples_3D']).uniform_multi()
        subsets = [('uniform', data.shape[0])]

    elif sampler == 'uniform_3D_grouped':
        data = _grid(constants['min'], constants['max'], constants['n_samples_3D'])
        subsets = [('uniform', data.shape[0])]

    elif sampler == 'lhs':
        data = _lhs(constants['n_samples_3D'])
        subsets = [('lhs', data.shape[0])]

    elif sampler == 'log':
        data = _log(constants['n_samples_3D'])
        subsets = [('log', data.shape[0])]

    elif sampler == 'combined_lhs_uniform':
        half_samples = int(constants['n_samples_3D'] / 2)
        data_0 = _lhs(half_samples)
        data_1 = _grid(constants['min'], constants['max'], half_samples)
        data = np.concatenate((data_0, data_1), axis = 0)
        subsets = [('lhs', data_0.shape[0]), ('uniform', data_1.shape[0])]

    elif sampler in ('combined_log_uniform', 'combined_log_lhs'):
        log_samples, uniform_samples = resolve_combined_counts(constants, n_dims)
        data_0 = _log(log_samples)
        if sampler == 'combined_log_uniform':
            data_1 = _grid(constants['min'], constants['max'], uniform_samples)
        else:
            data_1 = _lhs(uniform_samples)
        data = np.concatenate((data_0, data_1), axis = 0)
        subsets = [('log', data_0.shape[0]), ('uniform', data_1.shape[0])]

        # optional second uniform grid with its own bounds (min_uni2 / max_uni2, n_per_axis_uni2 per axis)
        if sampler == 'combined_log_uniform' and constants.get('n_per_axis_uni2'):
            data_2 = _grid(constants['min_uni2'], constants['max_uni2'], int(constants['n_per_axis_uni2'] ** n_dims))
            data = np.concatenate((data, data_2), axis = 0)
            subsets.append(('uniform2', data_2.shape[0]))

    else:
        raise ValueError(f"Unknown sampler '{sampler}'.")

    # Clamp tiny nonzero values (they produce huge stiffnesses). The threshold tracks the log-sampler's curvature
    # floor (thinnest section), so it stays consistent when t_values / zero_value_epsx change.
    threshold = zero_value_epsx / (min_section_t(constants) / 2)
    data = filter_small_epsilon(data, threshold=threshold, force_zero=force_zero)

    return data, subsets


def convert_log_data_to_eps(data_, max_log_neg, max_log_pos):
    """
    Convert sampled exponents into strains 10**exponent with a sign.

    Dimensions where an exponent exceeds the negative (positive) bound only admit the other sign; elsewhere the sign
    is drawn at random.

    Args:
        data_        (np.arr): sampled exponents (n, 6).
        max_log_neg  (np.arr): log10 of |min_log| per dimension.
        max_log_pos  (np.arr): log10 of |max_log| per dimension.

    Returns:
        data (np.arr): strains (n, 6).
    """
    eps_no_sign = 10**(data_)

    # magnitudes inside both bounds get a random sign
    mask_sign = ~((data_ < max_log_pos) & (data_ > max_log_neg))
    sign_eps = np.random.randint(0,2, size = eps_no_sign.shape)
    sign_eps[sign_eps==0] = -1

    data = eps_no_sign.copy()
    data[mask_sign] = sign_eps[mask_sign]*eps_no_sign[mask_sign]

    return data


def sample_exponents(constants, sampler_type_log, zero_value_epsx, num_samples = None, force_zero = False):
    """
    Sample the base-10 exponents of the log block, between log10 of the zero value and log10 of max(|min_log|, |max_log|).

    Args:
        constants        (dict):  sampling constants (min_log, max_log, n_samples_3D).
        sampler_type_log (str):   'lhs' or 'uniform_3D_grouped'.
        zero_value_epsx  (float): smallest membrane strain magnitude; the curvature floor uses the thinnest section.
        num_samples      (int):   number of samples; None -> constants['n_samples_3D'].
        force_zero       (bool):  passed to the grouped uniform sampler.

    Returns:
        data, max_log_neg, max_log_pos (np.arr): exponents (n, 6) and the log10 bounds per dimension.
    """
    # thinnest section sets the smallest curvature floor (smallest t/2 -> largest min |chi|)
    t = min_section_t(constants)
    zero_vec = [zero_value_epsx]*3 + [zero_value_epsx/(t/2)]*3

    max_log_neg = np.log10([abs(x) for x in constants['min_log']])
    max_log_pos = np.log10([abs(x) for x in constants['max_log']])

    min_log = np.log10(zero_vec)
    max_log = np.maximum(max_log_neg, max_log_pos)

    n_samples = num_samples if num_samples is not None else constants['n_samples_3D']

    t0 = time.perf_counter()
    if sampler_type_log == 'lhs':
        data = samplers(STRAIN_NAMES, min_log, max_log, samples= n_samples).lhs(criterion = 'c')
    elif sampler_type_log == 'uniform_3D_grouped':
        data = samplers(STRAIN_NAMES, min_log, max_log, samples= n_samples).uniform_multi_grouped(force_zero=force_zero)
    else:
        raise ValueError("Choose either 'lhs' or 'uniform_3D_grouped' as sampler for log sampling.")
    print(f'Sampled {data.shape[0]:,} log exponents ({sampler_type_log}) in {(time.perf_counter()-t0)/60:.2f} min')

    return data, max_log_neg, max_log_pos


def filter_small_epsilon(data, threshold = 3e-9, force_zero = False):
    """
    Raise nonzero strain magnitudes below threshold to threshold (avoids extremely large stiffnesses).
    With force_zero, exact zeros (deliberate axis nodes) are kept.
    """
    mask = abs(data) < threshold
    if force_zero:
        mask = mask & (data != 0.0)
    print(f'Detected {np.sum(mask)} points below threshold of {threshold}. These values are set to {threshold} in the dataset.')
    data[mask] = threshold

    return data


class samplers:
    """Point samplers over the box [min, max] (one bound per dimension)."""

    def __init__(self, parnames, min, max, samples):
        self.parnames = parnames
        self.min = min
        self.max = max
        self.samples = samples

    def lhs(self, criterion):
        """
        Latin hypercube samples (pyDOE). The number of samples is rounded to n_i**dim with n_i = round(samples**(1/dim)).

        Args:
            criterion (str): pyDOE.lhs criterion (e.g. 'c' = center).

        Returns:
            points (np.arr): (n_i**dim, dim)
        """
        dim = len(self.min)
        n_i = int(np.round((self.samples)**(1/dim), 0))

        bounds = np.vstack((self.min, self.max)).T

        lhs = doe.lhs(dim, samples=n_i**dim, criterion=criterion)
        points = np.zeros((n_i**dim,dim))
        for i in range(dim):
            par_min, par_max = bounds[i]
            points[:,i] = np.array(lhs[:, i]) * (par_max - par_min) + par_min

        return points

    def uniform_multi(self):
        """Full tensor grid with n_i = round(samples**(1/dim)) points per axis."""
        n_dims = len(self.min)
        n_i = int(np.round((self.samples)**(1/n_dims), 0))

        axes = [np.linspace(self.min[i], self.max[i], n_i) for i in range(n_dims)]
        grids = np.meshgrid(*axes, indexing='ij')
        return np.column_stack([g.ravel() for g in grids])

    def uniform_multi_grouped(self, group_size=3, force_zero=False):
        """
        Tensor grid as the Cartesian product of two group_size-dimensional grids (membrane x curvature), built with
        repeat / tile instead of a full meshgrid. n_i = round(samples**(1/dim)) points per axis; with force_zero an
        exact 0 node is added to axes that do not contain one. Returns float32 (n, dim).
        """
        n_dims = len(self.min)
        n_groups = n_dims // group_size
        n_i = int(np.round((self.samples)**(1/n_dims), 0))

        def _axis(i):
            ax = np.linspace(self.min[i], self.max[i], n_i)
            # A zero-crossing grid node can land on a floating residual (e.g. 6.8e-21) instead of exact 0, which
            # filter_small_epsilon would then clamp to its threshold. Snap such residuals to exact 0; only append an
            # explicit zero node when the axis has none.
            ax[np.abs(ax) < 1e-20] = 0.0
            if force_zero and not np.any(ax == 0.0):
                ax = np.append(ax, 0.0)
            return np.unique(ax)

        group_grids = []
        for g in range(n_groups):
            axes = [_axis(i) for i in range(g * group_size, (g + 1) * group_size)]
            grids = np.meshgrid(*axes, indexing='ij')
            group_grids.append(np.column_stack([gr.ravel() for gr in grids]))

        g0, g1 = group_grids[0], group_grids[1]
        n0, n1 = len(g0), len(g1)

        points = np.empty((n0 * n1, n_dims), dtype=np.float32)
        points[:, :group_size] = np.repeat(g0, n1, axis=0)   # repeat each row of g0 n1 times
        points[:, group_size:] = np.tile(g1, (n0, 1))         # tile g1 n0 times

        return points


########################################## Thickness-scaled curvature bounds ##########################################

def scale_chi_bounds(constants, t, divisor=2.0):
    """Shallow copy of constants with the curvature bounds recomputed from the membrane bounds via
    chi = eps/(t/divisor): dims 3,4,5 of min/max/min_log/max_log (and min_uni2/max_uni2 if set) <- dims 0,1,2
    divided by (t/divisor).

    Args:
        constants (dict):  base sampling constants.
        t         (float): section thickness the curvatures are scaled to.
        divisor   (float): z = t/divisor (2.0 = extreme-fibre match: the outer fibre strain equals the membrane bound).

    Returns:
        dict: shallow copy with chi bounds scaled for t.
    """
    c = dict(constants)
    z = t / divisor
    for key in ('min', 'max', 'min_log', 'max_log', 'min_uni2', 'max_uni2'):
        if constants.get(key) is None:
            continue
        b = list(constants[key])
        for j in (3, 4, 5):                          # chi dim scaled from the matching membrane dim (j-3)
            b[j] = constants[key][j - 3] / z
        c[key] = b
    return c


def scale_chi_bounds_to_sampling_t(constants):
    """Single-pass mode (sample_per_t = False): scale the curvature bounds to the one thickness being sampled
    (t_values None -> constants['t']; single-entry t_values -> that value). With several t_values there is no unique
    thickness -> bounds unchanged with a warning (use sample_per_t = True instead).
    """
    t_values = constants.get('t_values')
    if t_values is None:
        t = constants['t']
    elif len(t_values) == 1:
        t = t_values[0]
    else:
        print('[sampler] scale_chi_bounds_to_sampling_t: multiple t_values in a single-pass run -> curvature '
              'bounds left unscaled (no unique t). Use sample_per_t=True to scale per thickness.')
        return constants
    c = scale_chi_bounds(constants, t)
    print(f'[sampler] Curvature bounds scaled to t={float(t):g} mm (chi = eps/(t/2)): '
          f'|chi_x| max = {abs(c["max"][3]):.3g}, |chi_x| log max = {abs(c["max_log"][3]):.3g} 1/mm.')
    return c


def scaled_chi_constants(constants, t):
    """Copy of constants for one thickness t: curvature bounds via scale_chi_bounds with divisor
    constants['chi_divisor'] (default 2.0), and t_values pinned to [t] so the near-zero curvature floor
    (min_section_t) is on this thickness' scale.
    """
    c = scale_chi_bounds(constants, t, divisor=constants.get('chi_divisor', 2.0))
    c['t_values'] = [t]
    c['t_probs']  = None
    return c


def sample_per_t(constants, sampler):
    """
    Per-thickness sampling (sample_per_t = True): run sample_eps once for each t in t_values, each with curvature
    bounds chi = eps/(t/chi_divisor) (see scaled_chi_constants), so every thickness gets curvatures on its own
    scale. Each block's t is appended as a column; rows are regrouped subset by subset (log, uniform, ...) across all
    t so that the subsets stay contiguous.

    Args:
        constants (dict): base sampling constants (t_values gives the thicknesses).
        sampler   (str):  sampler name passed to sample_eps.

    Returns:
        eps_t   (np.arr): (N, 7) = [eps(6), t]
        subsets (list):   [(name, n_total), ...] pre-filter row counts per subset in row order.
    """
    t_values = np.asarray(constants['t_values'], dtype=float)
    blocks, totals = {}, {}                                                # subset name -> per-t row blocks / total count

    for t in t_values:
        c_t = scaled_chi_constants(constants, t)
        eps, subsets_t = sample_eps(sampler=sampler, constants=c_t)
        eps = np.hstack([eps, np.full((eps.shape[0], 1), t, dtype=eps.dtype)])   # append this block's t column
        start = 0
        for name, n in subsets_t:
            blocks.setdefault(name, []).append(eps[start:start + n])
            totals[name] = totals.get(name, 0) + n
            start += n
        counts_str = ' + '.join(f'{name} {n:,}' for name, n in subsets_t)
        print(f'  t={t:g} mm: sampled {counts_str} rows (|chi| max = {abs(c_t["max"][3]):.3g}).')

    eps_t = np.vstack([b for name in blocks for b in blocks[name]])        # grouped by subset across all t
    subsets = [(name, totals[name]) for name in blocks]
    print(f'sample_per_t: {t_values.size} thicknesses -> {eps_t.shape[0]:,} rows '
          f'({" + ".join(f"{name} {n:,}" for name, n in subsets)}).')
    return eps_t, subsets


########################################## Geometry / material columns ##########################################

def append_geom(eps_g, constants, seed=None, include_t=True):
    """
    Append randomly drawn geometry / material columns to the strains -> eps_and_geom.

    Columns, in order (each only if its value list is set), see geom_column_names:
      t                       from t_values (weights t_probs), only if include_t
      rho_xtop, rho_xbot      from rho_xtop_values / rho_xbot_values (one shared 'rho_x' column if only one is set)
      rho_ytop, rho_ybot      likewise for y
      fc                      from fc_values (weights fc_probs)
      fs                      from fs_values (weights fs_probs)
    The fc- / fs-dependent material parameters are not stored but interpolated on the fly (mat_from_fc / mat_from_fs).

    Args:
        eps_g     (np.arr): strains (n, 6), or (n, 7) with the t column already attached (sample_per_t, include_t=False).
        constants (dict):   sampling constants.
        seed      (int):    seed of a dedicated RNG for the draws (reproducible independently of the strain sampling);
                            None -> global np.random stream.
        include_t (bool):   draw and append the t column.

    Returns:
        np.arr: eps_and_geom (n, 6 + n_geom).
    """
    n = eps_g.shape[0]
    cols = [eps_g]
    rng = np.random.default_rng(seed) if seed is not None else np.random

    def _draw(key, p=None):
        # p: optional per-value draw weights (need not sum to 1); None -> uniform
        vals = np.asarray(constants[key], dtype=eps_g.dtype)
        if p is not None:
            p = np.asarray(p, dtype=float)
            if p.shape[0] != vals.shape[0]:
                raise ValueError(f"probs length {p.shape[0]} != {key} length {vals.shape[0]}")
            p = p / p.sum()
        return rng.choice(vals, size=(n, 1), p=p)

    def _append_rho(direction):
        top_key, bot_key = f'{direction}top_values', f'{direction}bot_values'
        top_on = constants.get(top_key) is not None
        bot_on = constants.get(bot_key) is not None
        if top_on and bot_on:
            cols.extend([_draw(top_key), _draw(bot_key)]); return 'top/bot'
        if top_on or bot_on:
            cols.append(_draw(top_key if top_on else bot_key)); return 'shared'
        return 'off'

    t_on = include_t and constants.get('t_values') is not None
    if t_on:
        cols.append(_draw('t_values', p=constants.get('t_probs')))
    rho_x_mode = _append_rho('rho_x')
    rho_y_mode = _append_rho('rho_y')
    fc_on = constants.get('fc_values') is not None
    if fc_on:
        cols.append(_draw('fc_values', p=constants.get('fc_probs')))
    fs_on = constants.get('fs_values') is not None
    if fs_on:
        cols.append(_draw('fs_values', p=constants.get('fs_probs')))

    eps_and_geom = np.hstack(cols)
    print(f'Built eps_and_geom: width {eps_and_geom.shape[1]} '
          f'(t drawn: {t_on}, rho_x: {rho_x_mode}, rho_y: {rho_y_mode}, fc: {fc_on}, fs: {fs_on}).')
    return eps_and_geom


########################################## Admissibility filter ##########################################

def get_principal_strains(eps_x, eps_y, gamma_xy):
    """Principal strains (eps_1 >= eps_2) of a plane strain state."""
    eps_avg = 0.5 * (eps_x + eps_y)
    R = np.sqrt(((eps_x - eps_y) * 0.5) ** 2 + (gamma_xy * 0.5) ** 2)
    return eps_avg + R, eps_avg - R


def get_mask_strains_principal(eps_top, eps_bot, eps_2_min = -5e-3, eps_1_max = 55e-3):
    """
    Rows whose top and bottom layer satisfy eps_2 > eps_2_min and eps_1 < eps_1_max.

    Args:
        eps_top, eps_bot (np.arr): (eps_x, eps_y, gamma_xy) in the top / bottom layer, (n, 3).
        eps_2_min, eps_1_max (float | np.arr): limits, scalar or per row (n,).

    Returns:
        mask (np.arr): bool (n,)
    """
    eps_1_top, eps_2_top = get_principal_strains(eps_top[:, 0], eps_top[:, 1], eps_top[:, 2])
    eps_1_bot, eps_2_bot = get_principal_strains(eps_bot[:, 0], eps_bot[:, 1], eps_bot[:, 2])

    mask = ((eps_2_top > eps_2_min) & (eps_2_bot > eps_2_min) &
            (eps_1_top < eps_1_max) & (eps_1_bot < eps_1_max))

    print(f'Amount of points left after principal filtering: {np.sum(mask)}/{eps_top.shape[0]} = {np.sum(mask)/eps_top.shape[0]*100:.2f}%')
    return mask


def filter_3d_data(eps_g, constants, eps_2_min = None, eps_1_max = None):
    """
    Remove physically inadmissible samples: the principal strains in the outermost layers (top and bottom) must
    satisfy eps_2 > eps_2_min (concrete crushing) and eps_1 < eps_1_max (excessive tension).

    Args:
        eps_g      (np.arr): strains, optionally with geom columns (n, 6 + n_geom)
        constants  (dict):   sampling constants (n_layer, t, principal_filter_*; geom layout)
        eps_2_min  (float | np.arr): eps_2 lower limit, scalar or per row (n,); None -> principal_filter_eps_2_min
        eps_1_max  (float | np.arr): eps_1 upper limit, scalar or per row (n,); None -> principal_filter_eps_1

    Returns:
        eps_g_f (np.arr): surviving rows (same column layout as input)
        mask    (np.arr): per-row survival mask (n,)
    """
    eps_dev = xp.asarray(eps_g)
    geom_keys = geom_column_names(constants) if eps_g.shape[1] > 6 else []
    if len(geom_keys) != eps_g.shape[1] - 6:
        raise ValueError(f"filter_3d_data: input has {eps_g.shape[1] - 6} geom column(s) but the constants "
                         f"imply geom_keys={geom_keys}. The geom vector and its layout must match.")
    if 't' in geom_keys:
        t = eps_dev[:, 6 + geom_keys.index('t')][:, None]  # (n, 1) per-sample thickness
    else:
        t = constants['t']                                  # scalar thickness

    # only the extreme layers are needed (l = 0 bottom, l = nl-1 top)
    nl = constants['n_layer']
    z_bot = -t/2 + (2*0        + 1) * t/(2*nl)
    z_top = -t/2 + (2*(nl-1)   + 1) * t/(2*nl)
    eps_mem = eps_dev[:, :3]
    chi     = eps_dev[:, 3:6]
    eps_bot = to_numpy(eps_mem - z_bot * chi)                # (n, 3)
    eps_top = to_numpy(eps_mem - z_top * chi)                # (n, 3)

    if eps_2_min is None:
        eps_2_min = constants['principal_filter_eps_2_min']
    if eps_1_max is None:
        eps_1_max = constants['principal_filter_eps_1']
    mask = get_mask_strains_principal(eps_top, eps_bot, eps_2_min = eps_2_min, eps_1_max = eps_1_max)

    return eps_g[mask], mask


def filter_3d_data_batchwise(eps_g, constants, n_batches = 2, eps_2_min = None, eps_1_max = None):
    """
    filter_3d_data in n_batches row chunks (bounds GPU memory). Row order is preserved, so the result equals a
    single filter_3d_data call.

    Returns:
        eps_g_f (np.arr), mask (np.arr): as filter_3d_data.
    """
    n = eps_g.shape[0]
    edges = [(i * n) // n_batches for i in range(n_batches + 1)]
    eps_list, mask_list = [], []

    for i in range(n_batches):
        start, end = edges[i], edges[i + 1]
        if start == end:
            continue
        t0 = time.perf_counter()
        lim = {k: (v[start:end] if isinstance(v, np.ndarray) else v)       # per-row limits -> this chunk's slice
               for k, v in (('eps_2_min', eps_2_min), ('eps_1_max', eps_1_max))}
        eps_f_b, mask_b = filter_3d_data(eps_g[start:end], constants, **lim)
        eps_list.append(eps_f_b)
        mask_list.append(mask_b)
        if USE_GPU:
            xp.get_default_memory_pool().free_all_blocks()
        print(f'Finished filtering batch {i+1}/{n_batches} (rows {start}:{end}) in {time.perf_counter()-t0:.2f} sec.')

    mask = np.concatenate(mask_list, axis = 0) if mask_list else np.zeros(0, dtype = bool)
    return np.concatenate(eps_list, axis = 0), mask


def per_subset_principal_limits(subsets, sampling_type, constants):
    """
    Per-row principal-filter limits: rows of the 'uniform2' subset get principal_filter_eps_2_min_uni2 /
    principal_filter_eps_1_uni2, all other rows the main limits.

    Args:
        subsets       (list):          [(name, count), ...] in row order.
        sampling_type (np.arr | None): per-row subset index.
        constants     (dict):          sampling constants.

    Returns:
        (eps_2_min, eps_1_max): per-row arrays (n,), or (None, None) -> filter_3d_data uses the main limits.
    """
    e2_u2 = constants.get('principal_filter_eps_2_min_uni2')
    e1_u2 = constants.get('principal_filter_eps_1_uni2')
    names = [name for name, _ in subsets]
    if (e2_u2 is None and e1_u2 is None) or 'uniform2' not in names:
        return None, None
    if sampling_type is None:
        print('[filter] WARNING: subset counts do not cover every row -> uniform2 principal limits NOT applied '
              '(main limits used for all rows).')
        return None, None

    is_u2 = sampling_type == names.index('uniform2')
    e2_main, e1_main = constants['principal_filter_eps_2_min'], constants['principal_filter_eps_1']
    eps_2_min = np.where(is_u2, e2_main if e2_u2 is None else e2_u2, e2_main)
    eps_1_max = np.where(is_u2, e1_main if e1_u2 is None else e1_u2, e1_main)
    print(f'[filter] uniform2 principal limits: eps_2_min = {eps_2_min[is_u2][0]:.2e}, eps_1_max = {eps_1_max[is_u2][0]:.2e} '
          f'(other subsets: {e2_main:.2e}, {e1_main:.2e})')
    return eps_2_min, eps_1_max


def prefilter_with_subset_report(eps_and_geom, subsets, constants, n_batches = 4):
    """
    Apply the admissibility filter and report how many rows of each sampling subset survive.

    Args:
        eps_and_geom (np.arr): strains + geom columns, rows ordered subset by subset.
        subsets      (list):   [(name, count), ...] row counts per subset in row order; per-subset reporting is
                               skipped if they do not cover every row.
        constants    (dict):   sampling constants.
        n_batches    (int):    number of row chunks filtered one at a time.

    Returns:
        eps_and_geom_f  (np.arr): surviving rows.
        sampling_type_f (np.arr): subset index of each surviving row (indexes `subsets`), or None.
    """
    n_pre = eps_and_geom.shape[0]
    subset_ok = bool(subsets) and sum(c for _, c in subsets) == n_pre
    sampling_type = (np.concatenate([np.full(c, i, dtype = np.int32) for i, (_, c) in enumerate(subsets)])
                     if subset_ok else None)

    eps_2_min, eps_1_max = per_subset_principal_limits(subsets, sampling_type, constants)

    eps_and_geom_f, mask = filter_3d_data_batchwise(eps_and_geom, constants, n_batches = n_batches,
                                                    eps_2_min = eps_2_min, eps_1_max = eps_1_max)

    if not subset_ok:
        return eps_and_geom_f, None

    sampling_type_f = sampling_type[mask]
    surv = np.bincount(sampling_type_f, minlength = len(subsets))
    print('Points left per subset after filtering:')
    for i, (name, c) in enumerate(subsets):
        print(f'  {name:8s}: {surv[i]:,}/{c:,} ({surv[i]/c*100 if c else 0.0:.2f}%)')

    return eps_and_geom_f, sampling_type_f


########################################## Outliers in D ##########################################

def find_outlier_d(dh: np.array, eps_g: np.array, factor: float = 1.5) -> tuple:
    """
    Interquartile-range outlier detection on the stiffness entries: a sample is an outlier if any of its 36 entries
    lies outside [q1 - factor*iqr, q3 + factor*iqr] (quartiles over the dataset, per entry).

    Args:
        dh     (np.arr): (n, 6, 6) stiffness data.
        eps_g  (np.arr): (n, ...) corresponding strains.
        factor (float):  IQR factor.

    Returns:
        outliers      (np.arr): D of the outlier samples.
        eps_g_outlier (np.arr): strains of the outlier samples.
        mask_1d       (np.arr): bool (n,), True for outliers.
    """
    q1 = np.quantile(dh, 0.25, axis = 0)
    q3 = np.quantile(dh, 0.75, axis = 0)
    iqr = q3-q1

    outlier_mask = (dh < q1-(factor*iqr)) | (dh > q3+(factor*iqr))

    mask_1d = outlier_mask.any(axis=(1, 2))
    print(f'Found {np.sum(mask_1d)} outliers for D in dataset with N = {dh.shape[0]/1e3:.2f}*1e3 points ({np.sum(mask_1d)/(dh.shape[0])*100:.3f}%)')

    return dh[mask_1d], eps_g[mask_1d], mask_1d


########################################## Saving ##########################################

def save_3D_data(data_: np.array, save_dir: str, filename: str):
    """Write data_ as float32 dataset `filename` to <save_dir>/output_<filename>.h5."""
    t0 = time.perf_counter()
    data_ = to_numpy(data_)
    with h5py.File(os.path.join(save_dir, f'output_{filename}.h5'), 'w') as f:
        f.create_dataset(filename, data = data_, dtype='float32')
    print(f'time save_data {filename}: {(time.perf_counter()-t0)/60:.2f}min')
