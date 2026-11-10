"""Generate a dataset of generalised strains, stresses and tangent stiffnesses of layered reinforced concrete shells.

Pipeline: sample strains -> draw geometry / material per sample -> remove inadmissible strain states -> simulate
stresses and stiffness (layered RC model) -> remove stiffness outliers -> save (HDF5) -> diagnostic plots.

Run from this folder:  python sampler_analytical_RC3D.py
Settings: flags below and constant_sampling_params.py; see README.md.
V. Balmer, 2026.
"""

import os

import numpy as np

USE_GPU         = os.environ.get('USE_GPU', '1').lower() in ('1', 'true', 'yes')   # cupy on the GPU, else numpy on the CPU

from utils import config
config.USE_GPU = USE_GPU                    # must be set before the other utils modules are imported

from constant_sampling_params import constants as CONSTANTS
from utils.sampler_utils_RC3D import (append_geom, find_outlier_d, get_constant_sampling_params, initialise_wandb,
                                      prefilter_with_subset_report, sample_eps, sample_per_t, save_3D_data,
                                      scale_chi_bounds_to_sampling_t, setup_sampler_dirs)
from utils.simulating_sig_vec_RC3D import SigSimulator, geom_column_names, sig_simulation_batchwise
from utils.plotting import imshow_D_filtered, plot_filtered_stiffness, plot_geom_distributions

if USE_GPU:
    import cupy as xp
else:
    xp = np


FILTER_DATA    = True                      # remove strain states outside the principal-strain limits
REMOVE_OUTLIERS = False                      # remove samples with extreme stiffness entries (IQR criterion)
OUTLIER_FACTOR  = 80                        # IQR factor of the outlier criterion
SAVE_D          = True                      # also save the stiffness matrices
PLOT            = True                      # write diagnostic plots to ./plots
SIM_BATCH_SIZE  = 1_000_000                 # max rows per simulation batch (GPU memory); lower if out of memory
FILTER_BATCHES  = 10                        # number of chunks for the admissibility filter (GPU memory)
SEED            = 42                        # RNG seed for reproducible sampling
LOG_WANDB       = os.environ.get('LOG_WANDB', '0').lower() in ('1', 'true', 'yes')   # log run to Weights & Biases


def main():
    np.random.seed(SEED)

    ############################ 0 - Settings ############################
    constants, mat_dict = get_constant_sampling_params(CONSTANTS)
    constants, save_data_dir = setup_sampler_dirs(constants)
    sampling_type = constants['sampling_type']
    cm = constants['cm']
    per_t = constants.get('sample_per_t', False)
    if not per_t:
        # single pass: tie the curvature bounds to the (single) sampled thickness, chi = eps/(t/2)
        constants = scale_chi_bounds_to_sampling_t(constants)

    initialise_wandb(constants, mat_dict, LOG_WANDB)
    simulatesig = SigSimulator(constants)
    save_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(save_plot_path, exist_ok=True)

    ############################ 1 - Sample strains, draw geometry / material ############################
    if per_t:
        eps, subsets = sample_per_t(constants, sampling_type)      # (N, 7): strains + t column
    else:
        eps, subsets = sample_eps(sampler=sampling_type, constants=constants)
    eps[:, 2] = eps[:, 2] * 2          # eps_xy -> gamma_xy
    eps[:, 5] = eps[:, 5] * 2          # chi_xy -> 2*chi_xy
    eps_and_geom = append_geom(eps, constants, seed=SEED, include_t=not per_t)

    ############################ 2 - Admissibility filter ############################
    subset_idx = None                        # per-row index into subsets (log, uniform, ...), set by the filter
    if FILTER_DATA:
        eps_and_geom, subset_idx = prefilter_with_subset_report(eps_and_geom, subsets, constants,
                                                                   n_batches = FILTER_BATCHES)

    ############################ 3 - Simulate stresses and stiffness ############################
    n_sub = max(1, -(-eps_and_geom.shape[0] // SIM_BATCH_SIZE))
    sig_g, D = sig_simulation_batchwise(xp.asarray(eps_and_geom), simulatesig, cm, mat_dict, n_batches=n_sub)

    ############################ 4 - Remove stiffness outliers ############################
    if REMOVE_OUTLIERS:
        _, _, mask_outliers = find_outlier_d(D, eps_and_geom, OUTLIER_FACTOR)
        n_total = D.shape[0]
        eps_and_geom, sig_g, D = eps_and_geom[~mask_outliers], sig_g[~mask_outliers], D[~mask_outliers]
        if subset_idx is not None:
            subset_idx = subset_idx[~mask_outliers]
        print(f'Removed {np.sum(mask_outliers)} outliers from dataset ({np.sum(mask_outliers)/n_total*100:.3f} %)')

    ############################ 5 - Save ############################
    save_3D_data(eps_and_geom[:, :6], save_data_dir, filename='eps_g')
    if eps_and_geom.shape[1] > 6:
        save_3D_data(eps_and_geom[:, 6:], save_data_dir, filename='geom')
    save_3D_data(sig_g, save_data_dir, filename='sig_g')
    if SAVE_D:
        save_3D_data(D, save_data_dir, filename='D')

    ############################ 6 - Diagnostic plots ############################
    if PLOT:
        plot_filtered_stiffness(eps_and_geom[:, :6], D, 0, save_plot_path, remove_outliers = REMOVE_OUTLIERS)
        imshow_D_filtered(eps_and_geom[:, :6], D, 0, save_plot_path)
        if eps_and_geom.shape[1] > 6:
            # post-filter distributions of the drawn geometry / material columns
            plot_geom_distributions(eps_and_geom, geom_column_names(constants), save_plot_path,
                                    sampling_type=subset_idx, sampling_type_names=[name for name, _ in subsets])

    if LOG_WANDB:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
