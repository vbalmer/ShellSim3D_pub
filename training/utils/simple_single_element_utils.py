"""Simplified single-element test: NLFEA only, stress-controlled (iterative), with vector-valued start and end
stresses (all six components can be ramped simultaneously). V. Balmer, 2026.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from .single_element_utils import (SAMPLING_CONSTANTS, get_constant_sampling_params, default_full_geom_keys,
                                   calculate_strain_profile, calculate_sig_D_NLFEA, SIG_LABELS, EPS_LABELS)


def simple_single_element_test(min_, max_, geom, num_samples = 100, small_value = 1e-20,
                               save_path = None, full_geom_keys = None):
    """
    Stress-controlled single-element test (NLFEA only).

    Samples num_samples generalised-stress vectors on the straight line from min_ to max_, solves for the strain
    profile of each target stress (calculate_strain_profile) and evaluates the NLFEA stresses and stiffnesses at
    these strains. Components that are zero are set to small_value.

    Args:
        min_, max_      (list)  start / end target generalised stress [n_x, n_y, n_xy, m_x, m_y, m_xy]
                                [N/mm, Nmm/mm], e.g. [0, 0, 0, 0, 0, 0], [0, 0, 0, 100_000, 0, 150_000].
        geom            (list)  geometry vector, e.g. [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs].
        num_samples     (int)   number of stress vectors between min_ and max_.
        small_value     (float) value replacing exact zeros in the sampled stress vectors.
        save_path       (str)   folder to save the figure in; None -> not saved.
        full_geom_keys  (list)  names of the geom entries, in order; None -> inferred from len(geom).

    Returns:
        results         (dict)  {'sig_target' (n, 6), 'eps' (n, 6), 'sig' (n, 6), 'D' (n, 6, 6)}
    """

    min_, max_ = np.asarray(min_, dtype = float), np.asarray(max_, dtype = float)
    if min_.shape != (6,) or max_.shape != (6,):
        raise ValueError(f'min_ and max_ must both have 6 entries [n_x, n_y, n_xy, m_x, m_y, m_xy], got {min_} and {max_}.')

    # swept components (min != max) define the x-axes of the plot
    idx_swept = [k for k in range(6) if min_[k] != max_[k]]
    if len(idx_swept) == 0:
        raise ValueError('min_ and max_ are identical; at least one stress component must be swept.')

    constants, mat_dict = get_constant_sampling_params(SAMPLING_CONSTANTS)
    if full_geom_keys is None:
        full_geom_keys = default_full_geom_keys(len(geom))

    # Step 1: target stress vectors on the line min_ -> max_, zeros replaced by small_value
    sig_target = np.linspace(min_, max_, num_samples)
    sig_target[sig_target == 0] = small_value
    print(f'Sampled {num_samples} target sig vectors from {min_.tolist()} to {max_.tolist()}.')

    # Step 2: Newton-Raphson solve for the strain profile of each target stress
    eps = calculate_strain_profile(sig_target, constants, mat_dict, geom = geom, geom_keys = full_geom_keys)
    X = np.concatenate((eps, np.tile(np.array(geom), (num_samples, 1))), axis = 1)
    print('Determined strain profiles for target sig.')

    # Step 3: NLFEA stresses and stiffnesses at the solved strains
    sig_D_NLFEA = calculate_sig_D_NLFEA(X, constants, mat_dict, cm = 3, geom_keys = full_geom_keys)
    res = np.max(np.abs(sig_D_NLFEA['sig'] - sig_target), axis = 0)
    print('Calculated NLFEA values. Max |sig_NLFEA - sig_target| per component: ' + ', '.join(f'{r:.3g}' for r in res))

    # Step 4: plot
    plot_simple_single_element_test(eps, sig_D_NLFEA, idx_swept, min_, max_, save_path)

    return {'sig_target': sig_target, 'eps': eps, 'sig': sig_D_NLFEA['sig'], 'D': sig_D_NLFEA['D']}


def plot_simple_single_element_test(eps, sig_D_NLFEA, idx_swept, min_, max_, save_path = None):
    """6 rows (one per stress component i); for every swept component j one column pair: sig_i and D_ij vs eps_j."""

    ncols = 2 * len(idx_swept)
    fig, axs = plt.subplots(6, ncols, figsize = [ncols * 7, 6 * 7], squeeze = False)

    for c, j in enumerate(idx_swept):
        for i in range(6):
            ax_sig, ax_D = axs[i, 2 * c], axs[i, 2 * c + 1]
            ax_sig.plot(eps[:, j], sig_D_NLFEA['sig'][:, i], color = 'black', label = 'NLFEA')
            ax_D.plot(eps[:, j], sig_D_NLFEA['D'][:, i, j], color = 'black', label = 'NLFEA')
            ax_sig.set_xlabel(EPS_LABELS[j])
            ax_D.set_xlabel(EPS_LABELS[j])
            ax_sig.set_ylabel(SIG_LABELS[i])
            ax_D.set_ylabel(f'D_{i+1}{j+1}')

    fmt = lambda v: '[' + ', '.join(f'{x:g}' for x in v) + ']'
    fig.suptitle(f'Stress-controlled single-element test (NLFEA): sig from {fmt(min_)} to {fmt(max_)}')
    fig.tight_layout(rect = [0, 0, 1, 0.98])

    if save_path is not None:
        os.makedirs(save_path, exist_ok = True)
        filename = 'simple_stress_path_[' + ','.join(map(str, idx_swept)) + '].png'
        fig.savefig(os.path.join(save_path, filename))
        print('Saved stress path figure ', filename, ' at ', save_path)

    return
