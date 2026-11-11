"""Single-element test: compares the trained network (NN) with the layered NLFEA section model (and optionally the
linear-elastic model, LFEA) along a strain or stress path of one shell element. V. Balmer, 2026.

Modes (single_element_test):
    plain sweep           one strain component idx_eps[0] is swept from min_ to max_, all others ~0.
    bending_iterative     a moment (idx_eps = [3], [4] or [5]) is swept with all other stresses zero; the strain
                          profile is found by Newton-Raphson iteration (calculate_strain_profile).
    tension_iterative     same for a membrane force (idx_eps = [0], [1] or [2]); switched on automatically for a
                          membrane sweep of an asymmetrically reinforced section.
    shear_iterative       pure in-plane shear n_xy (idx_eps = [2]); also plots the D symmetry diagnostic.
    synclastic_iterative  m_x = m_y swept (idx_eps = [3, 4]).
    nm_iterative          n swept with the coupled moment m = n * nm_ratio (idx_eps = [0] or [1], optionally
                          with the partner curvature, e.g. [0, 3]).
In the iterative modes min_ / max_ are generalised stresses [N/mm] or [Nmm/mm], otherwise strains [-] or [1/mm].
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from sampling.constant_sampling_params import constants as SAMPLING_CONSTANTS
from sampling.utils.sampler_utils_RC3D import get_constant_sampling_params
from sampling.utils.simulating_sig_vec_RC3D import SigSimulator, HiddenPrints

from .test_utils import predict_D, predict_sig, predict_sig_D, get_inp_from_folder, test_model_instance, get_stats_from_folder


SIG_LABELS = np.array(['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy'])
EPS_LABELS = np.array(['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy'])


def single_element_test(idx_eps, geom, model_path, min_, max_, save_path = None, plot_LFEA = False,
                        multirow = True, NN_comp = None, NN_comp2 = None, xlim = None,
                        bending_iterative = False, tension_iterative = False, shear_iterative = False,
                        synclastic_iterative = False, nm_iterative = False, nm_ratio = 50.0,
                        include_membrane_strains = False, symm_predict = False, xysymm_predict = False,
                        full_geom_keys = None, geom_keys_main = None, geom_keys_comp = None, geom_keys_comp2 = None,
                        num_samples = 100, return_sig_D = False):
    """
    Single-element test along a strain or stress path (see the module docstring for the modes).

    Computes sig and D with the NLFEA section model (and LFEA if plot_LFEA) and with the NN, and plots them
    against the swept (and partner) strain components.

    Args:
        idx_eps         (list):     swept component, e.g. [0] (eps_x / n_x) or [3, 4] (synclastic_iterative)
        geom            (list):     geometry vector, e.g. [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs]
        model_path      (list):     [path, version] of the trained model (folder path/v_<version>)
        min_, max_      (float):    range of the swept strain (plain sweep) or stress (iterative modes)
        save_path       (str):      folder to save the figures in; None -> not saved
        plot_LFEA       (bool):     also plot the linear-elastic (uncracked) section response
        multirow        (bool):     True: one row per stress component (6 rows); False: only the swept component
        NN_comp         (list):     [path, version] of a second model to compare; None -> no comparison
        NN_comp2        (list):     [path, version] of a third model to compare; None -> no comparison
        xlim            (tuple):    x-axis limits of the first two columns (display only)
        bending_iterative, tension_iterative, shear_iterative, synclastic_iterative, nm_iterative (bool):
                                    stress-controlled modes (mutually exclusive), see the module docstring;
                                    all False -> plain strain sweep
        nm_ratio        (float):    nm_iterative: m = n * nm_ratio [mm]
        include_membrane_strains (bool): synclastic_iterative: add two column pairs vs the developed eps_x, eps_y
        symm_predict    (bool):     predict with the mid-plane symmetry wrapper g(x) = 0.5 [f(x) + R f(R x)]
        xysymm_predict  (bool):     predict with the x <-> y relabelling wrapper g(x) = 0.5 [f(x) + P f(P x)]
        full_geom_keys  (list):     names of the geom entries in order; None -> inferred from len(geom)
        geom_keys_main, geom_keys_comp, geom_keys_comp2 (list):
                                    geom inputs of the main / comparison models (subset of full_geom_keys);
                                    None -> inferred from the model input size
        num_samples     (int):      number of points along the path
        return_sig_D    (bool):     also return the NLFEA and NN results

    Returns:
        None, or if return_sig_D:
        sig_D_NLFEA     (dict):     NLFEA results {'sig': (n, 6), 'D': (n, 6, 6)}
        sig_D_NN        (dict):     NN results (see predict_sig_D_NN_wrapper)
        Figures stress_path_<idx>(_multirow).png (and D_symmetry_diff_v<version>.png for shear_iterative) in save_path.
    """

    if bending_iterative and idx_eps[0] not in (3, 4, 5):
        raise ValueError(f'bending_iterative expects idx_eps to be a curvature index (3 -> m_x/chi_x, '
                         f'4 -> m_y/chi_y, 5-> m_xy/chi_xy), got idx_eps={idx_eps}.')

    if tension_iterative and idx_eps[0] not in (0, 1, 2):
        raise ValueError(f'tension_iterative expects idx_eps to be a membrane index (0 -> n_x/eps_x, '
                         f'1 -> n_y/eps_y, 2 -> n_xy/eps_xy), got idx_eps={idx_eps}.')

    if shear_iterative and idx_eps[0] != 2:
        raise ValueError(f'shear_iterative expects the in-plane shear index idx_eps = [2] '
                         f'(n_xy/eps_xy), got idx_eps={idx_eps}.')

    if synclastic_iterative and (len(idx_eps) < 2 or sorted(idx_eps[:2]) != [3, 4]):
        raise ValueError('synclastic_iterative requires idx_eps = [3, 4] (chi_x and chi_y).')

    if include_membrane_strains and not synclastic_iterative:
        raise ValueError('include_membrane_strains is only implemented for synclastic_iterative = True.')

    if nm_iterative and idx_eps[0] not in (0, 1):
        raise ValueError(f'nm_iterative expects idx_eps[0] to be 0 (N_x/eps_x) or 1 (N_y/eps_y), '
                         f'got idx_eps={idx_eps}.')
    if nm_iterative and len(idx_eps) > 1 and idx_eps[1] not in (3, 4):
        raise ValueError(f'nm_iterative: idx_eps[1] must be 3 (chi_x) or 4 (chi_y) when specified, '
                         f'got idx_eps={idx_eps}.')

    if sum([bool(tension_iterative), bool(bending_iterative), bool(shear_iterative),
            bool(synclastic_iterative), bool(nm_iterative)]) > 1:
        raise ValueError('tension_iterative, bending_iterative, shear_iterative, synclastic_iterative '
                         'and nm_iterative are mutually exclusive; enable only one.')

    constants, mat_dict = get_constant_sampling_params(SAMPLING_CONSTANTS)

    # Step 0: name the full geom vector; each model then selects its own geom columns by name
    if full_geom_keys is None:
        full_geom_keys = default_full_geom_keys(len(geom))
    n_geom_full = len(full_geom_keys)

    if len(geom) < n_geom_full:
        raise ValueError(f"geom={geom} provides {len(geom)} value(s) but full_geom_keys={full_geom_keys} "
                         f"names {n_geom_full} (in sampler column order).")
    geom = list(geom[:n_geom_full])

    with HiddenPrints():
        model_paths = [model_path] + [c for c in (NN_comp, NN_comp2) if c is not None]
        n_geom_max = max(get_inp_from_folder(mp[0], mp[1])['input_size'] - 6 for mp in model_paths)
    if n_geom_full < n_geom_max:
        raise ValueError(f"geom={geom} provides {n_geom_full} geom value(s) but a model needs {n_geom_max}. "
                         f"Pass the full geom vector (e.g. [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs]).")

    # Step 0b: a prescribed membrane strain on an asymmetric section induces moments -> iterate on pure tension
    plain = not (bending_iterative or tension_iterative or shear_iterative or synclastic_iterative or nm_iterative)
    if plain and idx_eps[0] in (0, 1, 2) and geom_is_asymmetric(geom, full_geom_keys):
        tension_iterative, plain = True, False
        print('[tension_iterative] Asymmetric reinforcement detected (rho_top != rho_bot); auto-enabling '
              'the pure-tension iteration. NOTE: min_/max_ are now interpreted as target generalised '
              'force n [N/mm], not strain.')

    # Step 1: strain path
    if plain:
        inp_vector = sample_idx_eps(idx_eps, min_, max_, geom, num_samples = num_samples)
        print('Sampled eps values.')
    else:
        if bending_iterative or tension_iterative or shear_iterative:
            inp_vector = sample_idx_sig(idx_eps, min_, max_, num_samples = num_samples)
            print('Sampled target sig values.')
        elif nm_iterative:
            inp_vector = sample_idx_sig([idx_eps[0]], min_, max_, num_samples = num_samples)
            # coupled moment in the same direction: n_x (col 0) -> m_x (col 3), n_y (col 1) -> m_y (col 4)
            inp_vector['sig_target'][:, idx_eps[0] + 3] = inp_vector['sig_target'][:, idx_eps[0]] * nm_ratio
            xy = ['x', 'y'][idx_eps[0]]
            print(f'Sampled target sig values (nm_iterative: M_{xy} = N_{xy} x {nm_ratio}, all other = 0).')
        elif synclastic_iterative:
            inp_vector = sample_idx_sig([3], min_, max_, num_samples = num_samples)
            inp_vector['sig_target'][:, 4] = inp_vector['sig_target'][:, 3]          # m_y = m_x
            print('Sampled target sig values (synclastic iterative: M_x = M_y, all other = 0).')

        # Newton-Raphson solve for the strain profile of each target stress (with the actual section)
        strain = calculate_strain_profile(inp_vector['sig_target'], constants, mat_dict,
                                          geom = geom, geom_keys = full_geom_keys)
        inp_vector['X_test'] = np.concatenate((strain, np.tile(np.array(geom), (strain.shape[0], 1))), axis = 1) \
            if len(geom) > 0 else strain
        inp_vector['y_test'] = np.zeros_like(inp_vector['X_test'])
        print('Determined strain profiles for target sig.')

    # Step 2a: NLFEA
    sig_D_NLFEA = calculate_sig_D_NLFEA(inp_vector['X_test'], constants, mat_dict, cm = 3, geom_keys = full_geom_keys)
    print('Calculated NLFEA values')

    # Step 2b: LFEA (only if plotted)
    sig_D_linel = calculate_sig_D_NLFEA(inp_vector['X_test'], constants, mat_dict, cm = 1, plot_LFEA = plot_LFEA,
                                        geom_keys = full_geom_keys)

    # Step 2c: NN (geom columns selected per model, by name)
    sig_D_NN = predict_sig_D_NN_wrapper(inp_vector, model_path, NN_comp, NN_comp2 = NN_comp2,
                                        full_geom_keys = full_geom_keys, geom_keys_main = geom_keys_main,
                                        geom_keys_comp = geom_keys_comp, geom_keys_comp2 = geom_keys_comp2,
                                        symm_predict = symm_predict, xysymm_predict = xysymm_predict)
    print('Calculated NN values')

    # Step 3: plot
    modes = dict(bending_iterative = bending_iterative, tension_iterative = tension_iterative,
                 shear_iterative = shear_iterative, synclastic_iterative = synclastic_iterative,
                 nm_iterative = nm_iterative, include_membrane_strains = include_membrane_strains)
    plot_single_element_test(idx_eps, inp_vector['X_test'], sig_D_NLFEA, sig_D_linel, sig_D_NN,
                             multirow, NN_comp, model_path, save_path, plot_LFEA, NN_comp2 = NN_comp2,
                             xlim = xlim, nm_ratio = nm_ratio, **modes)

    # Step 3b: pure-shear symmetry diagnostic (D_11 - D_22 and D_21 - D_12 vs gamma_xy)
    if shear_iterative:
        plot_D_symmetry_diff(inp_vector['X_test'], sig_D_NLFEA, sig_D_NN,
                             model_path, NN_comp, NN_comp2, save_path, xlim = xlim)

    if return_sig_D:
        return sig_D_NLFEA, sig_D_NN
    return


################################ strain / stress paths ################################

def sample_idx_eps(idx_eps, min_idx_eps, max_idx_eps, geom, num_samples = 100, small_value = 1e-20):
    """
    Strain path: component idx_eps[0] swept from min to max, all other strains set to small_value (~0).

    Args:
        idx_eps         (list):     swept strain component (single entry), e.g. [0] for eps_x
        min_idx_eps     (float):    start value of the swept strain [-] or curvature [1/mm]
        max_idx_eps     (float):    end value of the swept strain
        geom            (list):     geometry vector appended to every row (empty -> strains only)
        num_samples     (int):      number of points along the path
        small_value     (float):    value of the non-swept strains

    Returns:
        test_data       (dict):     {'X_test': (num_samples, 6+len(geom)), 'y_test': zeros of the same shape (dummy)}
    """
    if len(idx_eps) != 1:
        raise ValueError(f'The plain strain sweep varies a single component; got idx_eps={idx_eps}.')

    eps_vec = small_value*np.ones((num_samples, 6))
    eps_vec[:, idx_eps[0]] = np.linspace(min_idx_eps, max_idx_eps, num_samples)

    if len(geom) > 0:
        X_test = np.concatenate((eps_vec, np.tile(np.array(geom), (num_samples, 1))), axis = 1)
    else:
        X_test = eps_vec

    return {'X_test': X_test, 'y_test': np.zeros_like(X_test)}


def sample_idx_sig(sig_idx, min_idx_sig, max_idx_sig, num_samples = 100):
    """
    Target generalised stress: component sig_idx[0] swept from min to max, all other components zero.

    Args:
        sig_idx         (list):     swept stress component (first entry used), e.g. [3] for m_x
        min_idx_sig     (float):    start value of the swept stress [N/mm] or [Nmm/mm]
        max_idx_sig     (float):    end value of the swept stress
        num_samples     (int):      number of points along the path

    Returns:
        test_data       (dict):     {'sig_target': (num_samples, 6)}
    """
    sig_vec = np.zeros((num_samples, 6))
    sig_vec[:, sig_idx[0]] = np.linspace(min_idx_sig, max_idx_sig, num_samples)
    return {'sig_target': sig_vec}


################################ NLFEA ################################

def calculate_sig_D_NLFEA(eps_g: np.ndarray, constants: dict, mat_dict: dict, cm: int, plot_LFEA: bool = False,
                          geom_keys = None):
    """
    Generalised stresses and tangent stiffness from the layered section model (the same simulator as the sampler).

    Args:
        eps_g           (np.arr): (n, 6+n_geom) strains plus geometry columns
        constants       (dict):   sampling constants (layer layout, reinforcement pattern, ...)
        mat_dict        (dict):   default material parameters
        cm              (int):    1: linear elastic, 3: nonlinear (cracked RC)
        plot_LFEA       (bool):   for cm = 1: compute only if True (else returns None entries)
        geom_keys       (list):   names of the geom columns of eps_g; None -> sampler layout

    Returns:
        sig_D_NLFEA     (dict):   {'sig': (n, 6), 'D': (n, 6, 6)}; {'sig': None, 'D': None} if cm = 1 and not plot_LFEA
    """
    if not (plot_LFEA or cm == 3):
        return {'sig': None, 'D': None}

    with HiddenPrints():
        simulatesig = SigSimulator(constants)
        geom = simulatesig.build_geom(eps_g, geom_keys = geom_keys)
        t = geom.get('t')

        e = simulatesig.find_e_vec(eps_g[:, :6], t = t)                               # layer strains
        s = simulatesig.find_s_vec(e, mat_dict, cm_klij = cm, geom = geom)              # layer stresses
        sig_g = simulatesig.find_sh_vec(s, cm_klij = cm, t = t)                         # generalised stresses
        dh = simulatesig.find_dh_vec(s, mat_dict, cm_klij = cm, t = t, geom = geom)     # tangent stiffness

    return {'sig': sig_g, 'D': dh}


def calculate_strain_profile(sig_target, constants, mat_dict, max_iter = 100, tol = 1e-6,
                             remove_nonconverged = False, geom = None, geom_keys = None):
    """
    Newton-Raphson solve for the generalised strains that produce a target generalised stress, with the NLFEA
    section model. Vectorised over all samples: strain += D^-1 (sig_target - sig(strain)).

    Args:
        sig_target          (np.arr): target generalised stresses (n, 6)
        constants           (dict):   sampling constants
        mat_dict            (dict):   default material parameters
        max_iter            (int):    maximum number of iterations
        tol                 (float):  tolerance on the energy-like residual ratio |r . delta| / |r0 . delta0|
        remove_nonconverged (bool):   drop the rows that have not converged
        geom                (list):   geometry vector appended to the strains (e.g. asymmetric reinforcement)
        geom_keys           (list):   names of the geom entries

    Returns:
        strain              (np.arr): (n', 6) solved generalised strains (n' = n unless rows are removed)
    """
    t0 = time.perf_counter()
    n = sig_target.shape[0]

    geom_tile = (np.tile(np.asarray(geom, dtype = float), (n, 1))
                 if geom is not None and len(geom) > 0 else None)

    # Linear-elastic first guess: at zero strain the nonlinear tangent is ill-defined (principal directions 0/0)
    strain = np.zeros((n, 6))
    X0 = strain if geom_tile is None else np.concatenate((strain, geom_tile), axis = 1)
    D_elastic = calculate_sig_D_NLFEA(X0, constants, mat_dict, cm = 1, plot_LFEA = True,
                                      geom_keys = geom_keys)['D']
    try:
        strain0 = np.linalg.solve(D_elastic, sig_target)
        if np.all(np.isfinite(strain0)):
            strain = strain0
        else:
            print('[strain profile iteration] elastic seed produced non-finite strains; '
                  'falling back to the zero seed.')
    except np.linalg.LinAlgError:
        print('[strain profile iteration] elastic seed stiffness singular; falling back to the zero seed.')

    first_norm = None
    for num_iter in range(1, max_iter + 1):
        X = strain if geom_tile is None else np.concatenate((strain, geom_tile), axis = 1)
        sig_D = calculate_sig_D_NLFEA(X, constants, mat_dict, cm = 3, geom_keys = geom_keys)

        residual = sig_target - sig_D['sig']                                 # (n, 6)
        delta_strain = np.linalg.solve(sig_D['D'], residual)                 # (n, 6)
        strain = strain + delta_strain

        current_norm = np.abs(np.sum(residual * delta_strain, axis = 1))     # (n,)
        if first_norm is None:
            first_norm = np.where(current_norm == 0, 1.0, current_norm)
        row_ratio = current_norm / first_norm
        n_exceed = int(np.count_nonzero(row_ratio >= tol))
        if num_iter > 1 and n_exceed == 0:
            break

    status = 'converged' if n_exceed == 0 else 'not converged'
    print(f'[strain profile iteration] Newton-Raphson {status} after {num_iter} iters: '
          f'{n_exceed}/{n} rows exceed tol ({tol:.1e}).')

    if remove_nonconverged and n_exceed > 0:
        strain = strain[row_ratio < tol]
        print(f'[strain profile iteration] Removed {n_exceed} non-converged rows '
              f'({n_exceed/n*100:.3f} %), {strain.shape[0]} rows kept.')

    t_elapsed = time.perf_counter() - t0
    print(f'[strain profile iteration] calculate_strain_profile done in {t_elapsed/60:.2f}min ({t_elapsed:.2f}s)')

    return strain


################################ NN ################################

def predict_sig_D_NN(inp_vector: dict, model_path: list, full_geom_keys = None, model_geom_keys = None,
                     symm_predict = False, xysymm_predict = False):
    """
    sig and D predicted by one trained model.

    Args:
        inp_vector      (dict):  X_test = 6 strains + full geom (in full_geom_keys order), y_test (dummy)
        model_path      (list):  [path, version] of the trained model (folder path/v_<version>)
        full_geom_keys  (list):  names of the geom columns of inp_vector; None -> inferred
        model_geom_keys (list):  geom inputs of this model (subset of full_geom_keys); None -> inferred
        symm_predict    (bool):  predict with the mid-plane symmetry wrapper
        xysymm_predict  (bool):  predict with the x <-> y relabelling wrapper

    Returns:
        sig_D_NN        (dict):  {'sig': {'pred': (n, 6)}, 'D': {'pred': (n, 6, 6)}}, predictions in physical units.
                                 sig and D come from one NN pass per input set (predict_sig_D, CPU); only with
                                 xysymm_predict (no one-pass variant) are the separate predict_sig / predict_D
                                 passes used, whose full outputs (incl. labels, pred_norm) are then returned.
    """
    with HiddenPrints():
        inp = get_inp_from_folder(model_path[0], model_path[1])
        test_model = test_model_instance(inp, model_path[0], model_path[1])
        stats = get_stats_from_folder(model_path[0], model_path[1])

        inp_vector = match_inp_vector_to_model(inp_vector, inp['input_size'], model_path,
                                               full_geom_keys = full_geom_keys, model_geom_keys = model_geom_keys)

        if xysymm_predict:
            # no one-pass variant of the x<->y wrapper: separate sig / D passes
            sig_NN = predict_sig(test_model, inp, inp_vector, stats, sobolev = inp['Sobolev'], xysymm_predict = True)
            De_NN = predict_D(test_model, inp, inp_vector, stats, sobolev = inp['Sobolev'], xysymm_predict = True)
            return {'sig': sig_NN, 'D': De_NN}

        sig_D = predict_sig_D(test_model, inp, inp_vector, stats, sobolev = inp['Sobolev'],
                              symm_predict = symm_predict)

    return {'sig': {'pred': sig_D['sig'][:, :6]}, 'D': {'pred': sig_D['D']}}


def predict_sig_D_NN_wrapper(inp_vector: dict, model_path: list, NN_comp: list, NN_comp2: list = None,
                             full_geom_keys = None, geom_keys_main = None, geom_keys_comp = None, geom_keys_comp2 = None,
                             symm_predict = False, xysymm_predict = False):
    """
    sig and D of the main model and, if set, of one or two comparison models.

    Args:
        inp_vector      (dict):  X_test = 6 strains + full geom (in full_geom_keys order), y_test (dummy)
        model_path      (list):  [path, version] of the main model
        NN_comp         (list):  [path, version] of the first comparison model; None or version None -> skipped
        NN_comp2        (list):  [path, version] of the second comparison model; None or version None -> skipped
        full_geom_keys  (list):  names of the geom columns of inp_vector; None -> inferred
        geom_keys_main, geom_keys_comp, geom_keys_comp2 (list):
                                 geom inputs of the main / comparison models; None -> inferred
        symm_predict    (bool):  predict with the mid-plane symmetry wrapper
        xysymm_predict  (bool):  predict with the x <-> y relabelling wrapper

    Returns:
        sig_D_NN (dict): {'sig': {'pred': ...}, 'D': {'pred': ...}}; 'pred' is a single array without comparison
                         models, else a list [main, comp1(, comp2)].
    """
    comps = [(c, gk) for c, gk in ((NN_comp, geom_keys_comp), (NN_comp2, geom_keys_comp2))
             if c is not None and c[1] is not None]

    if not comps:
        return predict_sig_D_NN(inp_vector, model_path, full_geom_keys = full_geom_keys,
                                model_geom_keys = geom_keys_main,
                                symm_predict = symm_predict, xysymm_predict = xysymm_predict)

    sig_D_NN = {'sig': {'pred': []}, 'D': {'pred': []}}
    models = [([model_path[0], model_path[1]], geom_keys_main)] + [([c[0], c[1]], gk) for c, gk in comps]
    for mp, mgk in models:
        sig_D_NN_ = predict_sig_D_NN(inp_vector, mp, full_geom_keys = full_geom_keys, model_geom_keys = mgk,
                                     symm_predict = symm_predict, xysymm_predict = xysymm_predict)
        sig_D_NN['sig']['pred'].append(sig_D_NN_['sig']['pred'])
        sig_D_NN['D']['pred'].append(sig_D_NN_['D']['pred'])
    return sig_D_NN


def geom_is_asymmetric(geom, geom_keys, tol = 1e-12):
    """
    Check for asymmetric reinforcement (top != bottom).

    Args:
        geom        (list):   geometry values in geom_keys order
        geom_keys   (list):   names of the geom entries; None -> False
        tol         (float):  equality tolerance of the reinforcement ratios

    Returns:
        bool: True if rho_xtop != rho_xbot or rho_ytop != rho_ybot (False without a top/bottom split)
    """
    if geom_keys is None:
        return False
    kv = {k: geom[i] for i, k in enumerate(geom_keys) if i < len(geom)}
    def differs(a, b):
        return a in kv and b in kv and abs(float(kv[a]) - float(kv[b])) > tol
    return differs('rho_xtop', 'rho_xbot') or differs('rho_ytop', 'rho_ybot')


def default_full_geom_keys(n_geom: int):
    """
    Geom column names in the sampler order, inferred from the number of columns. Other subsets than the ones
    below cannot be inferred from the count; pass the keys explicitly.

    Args:
        n_geom      (int):  number of geom columns (0, 1, 2, 3, 5, 6 or 7)

    Returns:
        list: 7 -> ['t', 'rho_xtop', 'rho_xbot', 'rho_ytop', 'rho_ybot', 'fc', 'fs'], 6 -> without 'fs',
              5 -> without 'fc', 'fs'; 3 -> ['t', 'rho_x', 'rho_y'], 2 -> ['t', 'rho_y'], 1 -> ['rho_y'], 0 -> []
    """
    canonical = ['t', 'rho_xtop', 'rho_xbot', 'rho_ytop', 'rho_ybot']
    named = {7: canonical + ['fc', 'fs'], 6: canonical + ['fc'], 5: canonical,
             3: ['t', 'rho_x', 'rho_y'], 2: ['t', 'rho_y'], 1: ['rho_y'], 0: []}
    if n_geom not in named:
        raise ValueError(f'Cannot name {n_geom} geom columns; only {canonical} are defined. Pass geom_keys explicitly.')
    return named[n_geom]


def match_inp_vector_to_model(inp_vector: dict, input_size: int, model_path: list,
                              full_geom_keys = None, model_geom_keys = None):
    """
    Slice X_test / y_test to the inputs of one model, selecting the geom columns by name (rho_x / rho_y map to
    the top layer of the top/bottom scheme).

    Args:
        inp_vector      (dict):  X_test = 6 strains + full geom (in full_geom_keys order), y_test
        input_size      (int):   input width of the model (6 strains + its geom columns)
        model_path      (list):  [path, version] of the model (for error messages)
        full_geom_keys  (list):  names of the geom columns of inp_vector; None -> inferred
        model_geom_keys (list):  geom inputs of this model; None -> inferred from input_size

    Returns:
        matched         (dict):  copy of inp_vector with X_test = [strains, this model's geom columns] and y_test[:, :6]
    """
    xdim = inp_vector['X_test'].shape[1]
    if input_size > xdim:
        raise ValueError(f"Model {model_path} expects input_size={input_size} but inp_vector only "
                         f"has {xdim} columns (6 strains + geom). Pass enough geometrical parameters.")

    n_geom_full = xdim - 6
    n_geom_model = input_size - 6
    if full_geom_keys is None:
        full_geom_keys = default_full_geom_keys(n_geom_full)
    if len(full_geom_keys) != n_geom_full:
        raise ValueError(f"full_geom_keys={full_geom_keys} does not match the {n_geom_full} geom "
                         f"columns in inp_vector.")

    key_to_col = {k: 6 + i for i, k in enumerate(full_geom_keys)}
    if model_geom_keys is None:
        model_geom_keys = default_full_geom_keys(n_geom_model)
    if len(model_geom_keys) != n_geom_model:
        raise ValueError(f"Model {model_path} input_size implies {n_geom_model} geom column(s), but "
                         f"model_geom_keys={model_geom_keys} lists {len(model_geom_keys)}.")

    geom_aliases = {'rho_x': ['rho_xtop', 'rho_xbot'], 'rho_y': ['rho_ytop', 'rho_ybot']}

    def resolve_geom_col(k):
        if k in key_to_col:
            return key_to_col[k]
        for alt in geom_aliases.get(k, []):
            if alt in key_to_col:
                return key_to_col[alt]
        return None

    resolved = {k: resolve_geom_col(k) for k in model_geom_keys}
    missing = [k for k, c in resolved.items() if c is None]
    if missing:
        raise ValueError(f"Model {model_path} needs geom {missing}, not present in the shared geom "
                         f"vector {full_geom_keys} (nor via aliases {geom_aliases}). Add them to geom / geom_keys.")

    cols = list(range(6)) + [resolved[k] for k in model_geom_keys]
    matched = dict(inp_vector)
    matched['X_test'] = inp_vector['X_test'][:, cols]
    matched['y_test'] = inp_vector['y_test'][:, :6]
    return matched


################################ plotting ################################

def partner_indices(idx_eps, bending_iterative = False, tension_iterative = False, shear_iterative = False,
                    synclastic_iterative = False, nm_iterative = False, include_membrane_strains = False):
    """
    Strain indices of the extra column pair(s) of the plot (sig and D vs a partner strain).

    Args:
        idx_eps     (list):  swept component(s)
        bending_iterative, tension_iterative, shear_iterative, synclastic_iterative, nm_iterative,
        include_membrane_strains (bool): mode flags of single_element_test

    Returns:
        list: partner strain indices
            synclastic_iterative (+ include_membrane_strains) -> [idx_eps[1]] (+ [0, 1])
            bending_iterative     -> [idx_eps[0] - 3] (coupled eps); torsion idx_eps = [5] -> [0, 1]
            shear_iterative       -> [0, 1] (developed eps_x, eps_y)
            nm_iterative          -> [idx_eps[1]] if given, else [idx_eps[0] + 3]
            tension_iterative     -> [idx_eps[0] + 3] (compensating chi)
            plain sweep           -> []
    """
    if synclastic_iterative:
        return [idx_eps[1], 0, 1] if include_membrane_strains else [idx_eps[1]]
    if bending_iterative:
        return [0, 1] if idx_eps[0] == 5 else [idx_eps[0] - 3]
    if shear_iterative:
        return [0, 1]
    if nm_iterative:
        return [idx_eps[1]] if len(idx_eps) > 1 else [idx_eps[0] + 3]
    if tension_iterative:
        return [idx_eps[0] + 3]
    return []


def plot_single_element_test(idx_eps: list, eps_data: np.ndarray, sig_D_NLFEA: dict, sig_D_linel: dict, sig_D_NN: dict,
                             multirow: bool, NN_comp: list, model_path: list, save_path: str, plot_LFEA: bool,
                             NN_comp2: list = None, xlim: tuple = None, nm_ratio: float = 50.0, **modes):
    """
    Plot sig_i and D_ij vs the swept strain (columns 1-2) and vs the partner strain(s) of the mode
    (further column pairs).

    Args:
        idx_eps     (list):    swept component(s)
        eps_data    (np.arr):  strains (and geom) along the path, shape (n, 6+n_geom)
        sig_D_NLFEA (dict):    NLFEA results {'sig': (n, 6), 'D': (n, 6, 6)}
        sig_D_linel (dict):    LFEA results (entries None if not plotted)
        sig_D_NN    (dict):    NN results (see predict_sig_D_NN_wrapper)
        multirow    (bool):    one row per stress component (True) or only the swept component
        NN_comp     (list):    [path, version] of the first comparison model or None
        model_path  (list):    [path, version] of the main model (legend label)
        save_path   (str):     folder to save the figure in; None -> not saved
        plot_LFEA   (bool):    also plot the LFEA curves
        NN_comp2    (list):    [path, version] of the second comparison model or None
        xlim        (tuple):   x-axis limits of the first two columns
        nm_ratio    (float):   nm_iterative coupling ratio (title)
        **modes     (bool):    mode flags of single_element_test (see partner_indices)

    Returns:
        None; figure saved via save_single_el_plot
    """
    nrows = sig_D_NLFEA['sig'].shape[1] if multirow else 1
    idx_partner = partner_indices(idx_eps, **modes)
    ncols = 2 + 2 * len(idx_partner)
    fig, axs = plt.subplots(nrows, ncols, figsize = [ncols*7, nrows*7], squeeze = False)

    for i in range(nrows):
        j = i if multirow else idx_eps[0]
        plot_comparison(axs[i, 0], idx_eps, eps_data, sig_D_NLFEA['sig'], sig_D_linel['sig'], sig_D_NN['sig']['pred'],
                        j, NN_comp, model_path, plot_LFEA, NN_comp2 = NN_comp2, xlim = xlim)
        plot_comparison(axs[i, 1], idx_eps, eps_data, sig_D_NLFEA['D'], sig_D_linel['D'], sig_D_NN['D']['pred'],
                        j, NN_comp, model_path, plot_LFEA, NN_comp2 = NN_comp2, xlim = xlim)

        for p, ip in enumerate(idx_partner):
            c = 2 + 2 * p
            plot_comparison(axs[i, c], [ip], eps_data, sig_D_NLFEA['sig'], sig_D_linel['sig'], sig_D_NN['sig']['pred'],
                            j, NN_comp, model_path, plot_LFEA, NN_comp2 = NN_comp2)
            plot_comparison(axs[i, c+1], [ip], eps_data, sig_D_NLFEA['D'], sig_D_linel['D'], sig_D_NN['D']['pred'],
                            j, NN_comp, model_path, plot_LFEA, NN_comp2 = NN_comp2)

    figure_formatting_single_el(fig, axs, idx_eps, multirow, idx_partner, nm_ratio = nm_ratio, **modes)
    save_single_el_plot(fig, save_path, multirow, idx_eps)
    return


def plot_D_symmetry_diff(eps_data, sig_D_NLFEA, sig_D_NN, model_path, NN_comp, NN_comp2, save_path, xlim = None):
    """
    Pure-shear symmetry diagnostic vs gamma_xy: D_11 - D_22 (x/y asymmetry, 0 for a symmetric section) and
    D_21 - D_12 (reciprocity, 0 for NLFEA). NLFEA in black, NN curve(s) as in plot_comparison.

    Args:
        eps_data    (np.arr):  strains (and geom) along the path, shape (n, 6+n_geom); x-axis gamma_xy = column 2
        sig_D_NLFEA (dict):    NLFEA results, 'D' of shape (n, 6, 6)
        sig_D_NN    (dict):    NN results (see predict_sig_D_NN_wrapper)
        model_path  (list):    [path, version] of the main model
        NN_comp     (list):    [path, version] of the first comparison model or None
        NN_comp2    (list):    [path, version] of the second comparison model or None
        save_path   (str):     folder to save the figure in; None -> not saved
        xlim        (tuple):   x-axis limits

    Returns:
        None; figure saved as D_symmetry_diff_v<version>.png in save_path
    """
    gamma = eps_data[:, 2]
    order = np.argsort(gamma)
    x = gamma[order]

    NLFEA_D = sig_D_NLFEA['D']
    NN_pred = sig_D_NN['D']['pred']

    NN_KWARGS   = dict(color = 'dodgerblue', linestyle = '--', label = 'NN' + str(model_path[1]))
    COMP_STYLES = [dict(color = 'coral', linestyle = '--'), dict(color = 'mediumseagreen', linestyle = '-.')]
    comps = [c for c in (NN_comp, NN_comp2) if c is not None and c[1] is not None]
    if isinstance(NN_pred, list):
        nn_curves = [(NN_pred[0], NN_KWARGS)]
        nn_curves += [(NN_pred[k + 1], dict(**COMP_STYLES[k], label = 'NN' + str(c[1]))) for k, c in enumerate(comps)]
    else:
        nn_curves = [(NN_pred, NN_KWARGS)]

    fig, axs = plt.subplots(1, 2, figsize = [14, 6], squeeze = True)

    specs = [
        (axs[0], (0, 0), (1, 1), 'D_11 - D_22', 'diagonal x/y asymmetry (0 if symmetric)'),
        (axs[1], (1, 0), (0, 1), 'D_21 - D_12', 'tangent reciprocity D_ij - D_ji (0 for NLFEA)'),
    ]
    for ax, (ai, aj), (bi, bj), ylab, subtitle in specs:
        ax.plot(x, (NLFEA_D[:, ai, aj] - NLFEA_D[:, bi, bj])[order], color = 'black', label = 'NLFEA')
        for pred, kw in nn_curves:
            ax.plot(x, (pred[:, ai, aj] - pred[:, bi, bj])[order], **kw)
        ax.axhline(0, color = 'grey', lw = 0.6, zorder = 0)
        ax.set_xlabel('gamma_xy')
        ax.set_ylabel(ylab)
        ax.set_title(subtitle)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend()

    fig.suptitle(f'Pure-shear D symmetry diagnostic (v_{model_path[1]})')
    fig.tight_layout(rect = [0, 0, 1, 0.96])

    if save_path is not None:
        os.makedirs(save_path, exist_ok = True)
        filename = f'D_symmetry_diff_v{model_path[1]}.png'
        fig.savefig(os.path.join(save_path, filename))
        print('Saved D symmetry diagnostic ', filename, ' at ', save_path)
    return


def plot_comparison(axs, idx_eps, eps_data, NLFEA_data, linel_data, NN_data, i, NN_comp, model_path, plot_LFEA,
                    NN_comp2 = None, xlim = None):
    """
    Plot component i (sig_i, or D_i,idx_eps[0] for stiffness data) of NLFEA, LFEA and NN vs eps[:, idx_eps[0]].

    Args:
        axs         (Axes):    subplot
        idx_eps     (list):    strain component on the x-axis (first entry)
        eps_data    (np.arr):  strains (and geom) along the path, shape (n, 6+n_geom)
        NLFEA_data  (np.arr):  NLFEA stresses (n, 6) or stiffnesses (n, 6, 6)
        linel_data  (np.arr):  LFEA data of the same shape (only used if plot_LFEA)
        NN_data     (np.arr|list): NN data of the same shape, or a list [main, comp1(, comp2)] with comparison models
        i           (int):     stress component (row of D)
        NN_comp     (list):    [path, version] of the first comparison model or None
        model_path  (list):    [path, version] of the main model (legend label)
        plot_LFEA   (bool):    also plot the LFEA curve
        NN_comp2    (list):    [path, version] of the second comparison model or None
        xlim        (tuple):   x-axis limits

    Returns:
        None
    """

    x_vec = eps_data[:, idx_eps[0]]

    NLFEA_KWARGS = dict(color = 'black', label = 'NLFEA')
    LFEA_KWARGS = dict(color = 'lightgrey', linestyle = ':', label = 'LFEA')
    NN_KWARGS = dict(color = 'dodgerblue', linestyle = '--', label = 'NN'+str(model_path[1]))
    COMP_STYLES = [dict(color = 'coral', linestyle = '--'), dict(color = 'mediumseagreen', linestyle = '-.')]
    comp_kwargs = [dict(**COMP_STYLES[k], label = 'NN'+str(c[1]))
                   for k, c in enumerate((NN_comp, NN_comp2))
                   if c is not None and c[1] is not None]
    comparing = len(comp_kwargs) > 0

    if len(NLFEA_data.shape) > 2:     # stiffness
        sel = lambda d: d[:, i, idx_eps[0]]
    else:                             # stresses
        sel = lambda d: d[:, i]

    axs.plot(x_vec, sel(NLFEA_data), **NLFEA_KWARGS)
    if plot_LFEA:
        axs.plot(x_vec, sel(linel_data), **LFEA_KWARGS)
    if not comparing:
        axs.plot(x_vec, sel(NN_data), **NN_KWARGS)
    else:
        axs.plot(x_vec, sel(NN_data[0]), **NN_KWARGS)
        for k, kw in enumerate(comp_kwargs):
            axs.plot(x_vec, sel(NN_data[k+1]), **kw)

    if xlim is not None:
        axs.set_xlim(xlim)
    return


def figure_formatting_single_el(fig, axs, idx_eps, multirow, idx_partner, bending_iterative = False,
                                tension_iterative = False, shear_iterative = False, synclastic_iterative = False,
                                nm_iterative = False, nm_ratio = 50.0, include_membrane_strains = False):
    """
    Axis labels, title and legend of the single-element plot.

    Args:
        fig         (Figure):  figure
        axs         (np.arr):  subplots, shape (nrows, ncols)
        idx_eps     (list):    swept component(s)
        multirow    (bool):    one row per stress component (True) or only the swept component
        idx_partner (list):    partner strain indices of the extra column pairs (see partner_indices)
        bending_iterative, tension_iterative, shear_iterative, synclastic_iterative, nm_iterative,
        include_membrane_strains (bool): mode flags of single_element_test (title)
        nm_ratio    (float):   nm_iterative coupling ratio (title)

    Returns:
        None
    """

    D_labels = np.array([[f'D_{i+1}{j+1}' for j in range(6)] for i in range(6)])

    rows = range(axs.shape[0]) if multirow else [0]
    for r in rows:
        i = r if multirow else idx_eps[0]
        axs[r, 0].set_xlabel(EPS_LABELS[idx_eps[0]])
        axs[r, 1].set_xlabel(EPS_LABELS[idx_eps[0]])
        axs[r, 0].set_ylabel(SIG_LABELS[i])
        axs[r, 1].set_ylabel(D_labels[i, idx_eps[0]])
        for p, ip in enumerate(idx_partner):
            c = 2 + 2 * p
            axs[r, c].set_xlabel(EPS_LABELS[ip])
            axs[r, c+1].set_xlabel(EPS_LABELS[ip])
            axs[r, c].set_ylabel(SIG_LABELS[i])
            axs[r, c+1].set_ylabel(D_labels[i, ip])

    partner_str = ', '.join(EPS_LABELS[ip] for ip in idx_partner)
    if synclastic_iterative and include_membrane_strains:
        fig.suptitle(f'Synclastic-iterative bending: M_x = M_y swept, {EPS_LABELS[idx_eps[0]]} = {EPS_LABELS[idx_eps[1]]} solved; '
                     f'cols 1-2 vs {EPS_LABELS[idx_eps[0]]}, cols 3-4 vs {EPS_LABELS[idx_eps[1]]}, '
                     f'cols 5-8 vs developed eps_x, eps_y')
    elif synclastic_iterative:
        fig.suptitle(f'Synclastic-iterative bending: M_x = M_y swept, {EPS_LABELS[idx_eps[0]]} = {partner_str} solved; '
                     f'cols 1-2 vs {EPS_LABELS[idx_eps[0]]}, cols 3-4 vs {partner_str}')
    elif tension_iterative:
        fig.suptitle(f'Pure-tension iteration: {EPS_LABELS[idx_eps[0]]} (moments driven to 0); '
                     f'cols 3-4 vs compensating {partner_str}')
    elif shear_iterative:
        fig.suptitle(f'Pure-shear iteration: {EPS_LABELS[idx_eps[0]]} (n_x, n_y and moments driven to 0); '
                     f'cols 3-6 vs developed {partner_str}')
    elif bending_iterative and idx_eps[0] == 5:
        fig.suptitle(f'Pure-torsion iteration: {EPS_LABELS[idx_eps[0]]} (all other stresses driven to 0); '
                     f'cols 3-6 vs developed {partner_str}')
    elif bending_iterative:
        fig.suptitle(f'Pure-bending iteration: {EPS_LABELS[idx_eps[0]]} (all other stresses driven to 0); '
                     f'cols 3-4 vs coupled {partner_str}')
    elif nm_iterative:
        xy = ['x', 'y'][idx_eps[0]]
        fig.suptitle(f'Combined N+M iteration: N_{xy} swept, M_{xy} = N_{xy} x {nm_ratio:.4g} [Nmm/mm]; '
                     f'cols 3-4 vs {partner_str}')
    else:
        fig.suptitle(f'Variation of {EPS_LABELS[idx_eps[0]]}')

    # legend without duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    plt.legend(seen.values(), seen.keys())
    return


def save_single_el_plot(fig, save_path, multirow, idx_eps):
    """
    Save the single-element figure.

    Args:
        fig         (Figure):  figure
        save_path   (str):     output folder (created if needed); None -> not saved
        multirow    (bool):    adds the suffix '_multirow' to the file name
        idx_eps     (list):    swept component(s), used in the file name

    Returns:
        None; figure saved as save_path/stress_path_<idx>(_multirow).png
    """
    if save_path is None:
        return
    os.makedirs(save_path, exist_ok = True)
    idx_label = str(idx_eps[0]) if len(idx_eps) == 1 else '[' + ','.join(map(str, idx_eps)) + ']'
    filename = 'stress_path_' + idx_label + ('_multirow.png' if multirow else '')
    fig.savefig(os.path.join(save_path, filename))
    print('Saved stress path figure ', filename, ' at ', save_path)
    return
