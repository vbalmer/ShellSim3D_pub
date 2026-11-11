"""Train the Sobolev surrogate on the sampled dataset (see README.md). V. Balmer, 2026.

Default run: trains on the example dataset in sampling/data with the hyperparameters in config_inp.py and writes
the model to training/logs/v_<n> (including the diagonal test plots in v_<n>/plots).
"""

import os

from config_inp import inp as INP
from utils.train_utils import (set_torch_params, setup_dirs, setup_hyperparams, main_train,
                               save_inp, save_stats, save_test_data)
from utils.data_utils import (get_data, add_geom_data, split_data, plot_split_data, get_stats, t_norm_from_inp,
                              fc_norm_from_inp, get_normalised_data, plot_norm_data, data_to_torch)
from utils.test_utils import test_NN_model


_on = lambda name, default: os.environ.get(name, default).lower() in ('1', 'true', 'yes')

SAVE_FOLDER = True                                  # copy the trained model to a new version folder logs/v_<n>
SOBOLEV     = _on('SOBOLEV', '1')                   # train on stresses and stiffnesses (Sobolev loss)
PLOT_DATA   = False                                 # 3D scatter plots of the split / normalised data (training/plots)
DATA_FRAC   = float(os.environ.get('DATA_FRAC', 1.0))   # < 1: keep this fraction of train / eval after the split (test set unchanged)
RUN_TEST    = _on('RUN_TEST', '1')                  # diagonal test plots after training
LOG_WANDB   = _on('LOG_WANDB', '0')                 # log the training to Weights & Biases


def main():
    set_torch_params()                              # seed python, numpy, torch

    path_data, MODEL_DIR, LOGS_DIR = setup_dirs()
    inp = setup_hyperparams(INP, SOBOLEV)
    inp['data_frac'] = DATA_FRAC                    # recorded in inp.pkl
    GEOM_SIZE = inp['input_size'] - inp['out_size']

    ############################ 0 - Read data      ############################
    # data stays in the original units (N, mm)
    features, labels_concat = get_data(path_data)
    features = add_geom_data(path_data, features, GEOM_SIZE)

    ############################ 1 - Train-Eval-Test Split ############################
    train_eval_test_data = split_data(features, labels_concat, test_size=0.1, eval_size=0.2)
    if DATA_FRAC < 1.0:                             # split is already shuffled -> first n rows are a random subset
        for key in ('train', 'eval'):
            n_keep = int(DATA_FRAC * train_eval_test_data[f'X_{key}'].shape[0])
            train_eval_test_data[f'X_{key}'] = train_eval_test_data[f'X_{key}'][:n_keep]
            train_eval_test_data[f'y_{key}'] = train_eval_test_data[f'y_{key}'][:n_keep]
        print(f'DATA_FRAC={DATA_FRAC}: train {train_eval_test_data["X_train"].shape[0]/1e6:.2f}*1e6, '
              f'eval {train_eval_test_data["X_eval"].shape[0]/1e6:.2f}*1e6, test unchanged.')
    plot_split_data(train_eval_test_data, plot=PLOT_DATA)

    ############################ 2 - Normalisation  ############################
    t_norm  = t_norm_from_inp(inp)                  # t-normalisation config (None -> off)
    fc_norm = fc_norm_from_inp(inp)                 # fc-normalisation config (None -> off)
    stats   = get_stats(train_eval_test_data, t_norm=t_norm, fc_norm=fc_norm)
    if t_norm is not None:
        stats['t_norm'] = t_norm                    # saved with the model for inference
        print(f"[t_norm] ON: powers={t_norm['powers']}, input_powers={t_norm['input_powers']}, t_idx={t_norm['t_idx']}")
    if fc_norm is not None:
        stats['fc_norm'] = fc_norm                  # saved with the model for inference
        print(f"[fc_norm] ON: sig and D divided by fc (fc_idx={fc_norm['fc_idx']})")
    norm_data = get_normalised_data(train_eval_test_data, stats, SOBOLEV)
    plot_norm_data(norm_data, plot=PLOT_DATA)
    torch_data = data_to_torch(norm_data)

    ############################ 3 - Save config    ############################
    save_inp(inp, save_path=MODEL_DIR)
    save_stats(stats, save_path=MODEL_DIR)
    save_test_data(train_eval_test_data, save_path=MODEL_DIR)

    ############################ 4 - Train          ############################
    version = main_train(torch_data, inp, save_path=MODEL_DIR, logs_dir=LOGS_DIR,
                         save_folder=SAVE_FOLDER, stats=stats, log_wandb=LOG_WANDB)

    ############################ 5 - Test           ############################
    if RUN_TEST and SAVE_FOLDER:
        test_data = {'X_test': train_eval_test_data['X_test'],
                     'y_test': train_eval_test_data['y_test']}
        test_NN_model(test_data, stats, save_path=LOGS_DIR, version=version)


if __name__ == '__main__':
    main()
