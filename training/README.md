# Training of the surrogate model for reinforced concrete shell elements

This folder trains the data-driven shell model: a neural network that maps the **generalised strains** and the
**geometry / material** of a layered reinforced concrete (RC) shell section to the **generalised stresses**. The
**tangent stiffness** is the Jacobian of the network and is trained together with the stresses (Sobolev training).
The training data comes from [`../sampling`](../sampling/README.md).

## Contents

1. [Method](#1-method)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
   - 3.1 [Training run](#31-training-run)
     - [Settings](#settings-config_inppy)
     - [Pipeline flags](#pipeline-flags-trainpy)
   - 3.2 [Single-element test](#32-single-element-test-single_element_testpy)
   - 3.3 [Simple single-element test](#33-simple-single-element-test-simple_single_element_testpy)
4. [Output](#4-output)
   - 4.1 [Trained model](#41-trained-model)
   - 4.2 [Single-element tests](#42-single-element-tests)
5. [Code structure](#5-code-structure)
6. [Notes](#6-notes)

## 1. Method

1. **Data**: strains `[eps_x, eps_y, gamma_xy, chi_x, chi_y, 2 chi_xy]`, geometry / material
   `[t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs]` (network input, 13 columns), stresses
   `[n_x, n_y, n_xy, m_x, m_y, m_xy]` and stiffness `D = d sig / d eps` (labels), split into train / eval / test
   (70 / 20 / 10 %, fixed seed).
2. **Normalisation** (statistics of the training set):
   - *t-normalisation*: stresses are divided by `t^a` (`a = 1` for forces, `2` for moments) and the curvature inputs
     are multiplied by `t`, so thin and thick sections have similar ranges;
   - *fc-normalisation*: stresses and stiffnesses are divided by `fc`;
   - then all inputs and stresses are standardised. `D` is scaled with `std(sig_j) / std(eps_k)`, so the Jacobian of
     the normalised network is the normalised stiffness.
3. **Network**: fully connected (default 5 x 1024, ELU).
4. **Sobolev loss**: `w1 * L(sig_NN, sig) + w2 * L(D_NN, D)` with `D_NN` the Jacobian of the network w.r.t. the strain
   inputs (`torch.func.jacrev`). `L` is the MSE or the weighted MSE `wbMSELoss`, which up-weights near-zero moments
   and membrane-bending coupling entries `D_mb`, `D_bm` (they are small but matter for the structural response).
5. **Optimisation**: AdamW with a step learning-rate schedule (optional switch to LBFGS and early stopping). The
   model with the lowest validation loss is kept.
6. **Test**: diagonal plots (prediction vs label) of stresses and stiffnesses on the test set.
7. **Single-element tests** for specific loading situations: the network is compared with the section model (NLFEA)
   along the load path of one shell element, e.g. pure bending, pure tension, pure shear, synclastic bending
   (`m_x = m_y`) or combined normal force and bending. For a stress-controlled path the strains are found with a
   Newton-Raphson iteration on the section model, and all stresses and stiffness entries are plotted along the path
   (see [3.2](#32-single-element-test-single_element_testpy) and [3.3](#33-simple-single-element-test-simple_single_element_testpy)).

## 2. Installation

Use the conda environment in the repository root (see [`../sampling/README.md`](../sampling/README.md#2-installation)).
Training needs `torch`, `numpy`, `h5py`, `scikit-learn` and `matplotlib`; a CUDA GPU is used if available. The
single-element tests also use the sampling code (`../sampling`). `wandb` is optional. The diagonal test plots use
LaTeX text rendering (`text.usetex`), so a LaTeX installation is needed for them (or set `RUN_TEST=0`).

## 3. Usage

```bash
cd training

# training
python train.py

# single-element test (NN vs NLFEA)
python single_element_test.py

# simple single-element test (NLFEA only)
python simple_single_element_test.py
```
### 3.1 Training run

```
cd training
python train.py
```

- **Default run**: trains on the example dataset in `sampling/data` with the settings of the provided model
  (see [4.1](#41-trained-model)). With the small example dataset the result is only a smoke test; for a usable
  model, sample a training dataset (see the sampling README) and point `DATA_DIR` to it.
- **Settings**: hyperparameters in [`config_inp.py`](config_inp.py) (see [Settings](#settings-config_inppy)),
  pipeline flags at the top of [`train.py`](train.py) (see [Pipeline flags](#pipeline-flags-trainpy)).
- **Environment variables**:

  | Variable | Effect |
  | --- | --- |
  | `DATA_DIR` | folder with the sampled HDF5 files (default: `sampling/data`) |
  | `MODEL_DIR` | working folder of the current run (default: `training/checkpoints`) |
  | `LOGS_DIR` | folder of the versioned results `v_<n>` (default: `training/logs`) |
  | `NUM_EPOCHS`, `BATCH_SIZE`, `HIDDEN_LAYERS`, `EARLY_STOP_STEPS`, `EARLY_STOP_RTOL` | override the corresponding entries of `config_inp.py` |
  | `SOBOLEV=0` | train on the stresses only |
  | `DATA_FRAC` | use only this fraction of the train / eval sets (test set unchanged) |
  | `RUN_TEST=0` | skip the diagonal test plots |
  | `LOG_WANDB=1` | log the training to Weights & Biases (project `ShellSim3D`) |

- **Memory**: the whole dataset is held in memory (as float64 during normalisation), so large datasets need a
  large-memory workstation.

#### Settings (`config_inp.py`)

| Group | Key | Meaning |
| --- | --- | --- |
| **Network** | `input_size` / `out_size` | 6 strains + number of geometry columns (13 for the full sampler output) / 6 stresses |
|  | `hidden_layers` | layer widths, e.g. `str([1024]*5)` |
|  | `activation` | name of a `torch.nn` activation, e.g. `ELU` |
|  | `dropout_rate`, `BatchNorm` | regularisation (off by default) |
| **Optimisation** | `batch_size` | mini-batch size (`None` → full batch) |
|  | `num_epochs`, `learning_rate` | number of epochs, initial learning rate |
|  | `lr_scheduler` | `standard`: decay x0.99 every 100 steps; `plateau`: reduce on validation plateau |
|  | `switch_step_percentage` | fraction of the optimisation steps after which AdamW switches to LBFGS (`1` → never) |
|  | `early_stop_steps` / `early_stop_rtol` | stop if the epoch-mean validation loss has not improved by `rtol` for this many steps (`None` → off) |
| **Loss** | `loss_type` | `MSELoss` or `wbMSELoss` |
|  | `loss_type_D` | loss of the stiffness term (`None` → same as `loss_type`) |
|  | `weight` / `band` | `wbMSELoss`: up-weight factor for normalised targets with magnitude below `band` |
|  | `w_s` | Sobolev weights `[w1, w2]` of stresses and stiffness, or `'max'` |
| **Normalisation** | `t_normalise`, `t_norm_powers`, `t_norm_input_powers`, `t_idx` | t-normalisation: output powers, input (strain) powers and column of `t` |
|  | `fc_normalise`, `fc_idx` | fc-normalisation and column of `fc` |

#### Pipeline flags (`train.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `SAVE_FOLDER` | `True` | copy the results to a new version folder `logs/v_<n>` |
| `SOBOLEV` | `True` (env `SOBOLEV`) | Sobolev training (stresses and stiffnesses) |
| `PLOT_DATA` | `False` | 3D scatter plots of the split and normalised data (`training/plots`) |
| `DATA_FRAC` | `1.0` (env `DATA_FRAC`) | fraction of the train / eval sets used |
| `RUN_TEST` | `True` (env `RUN_TEST`) | diagonal test plots after training |
| `LOG_WANDB` | `False` (env `LOG_WANDB`) | Weights & Biases logging |

### 3.2 Single-element test (`single_element_test.py`)

```
cd training
python single_element_test.py
```

Compares a trained network with the layered section model (NLFEA, the simulator of the sampler) along a path of one
shell element and plots all stresses `sig_i` and stiffness entries `D_ij` along the path. The default run is pure
bending `m_x` of an asymmetrically reinforced 500 mm section with the provided model v_156. The settings are at the
top of the script:

| Setting | Meaning |
| --- | --- |
| `geom` | `[t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs]` |
| `model_path`, `model_version` | folder with `v_<version>` (`trained_models` or `logs`) and version |
| `NN_comp`, `NN_comp2` | up to two further models `[path, version]` to compare (`None` → off) |
| `idx_eps`, `min_`, `max_` | swept component and range |
| `bending_iterative` | moment `m_x`, `m_y` or `m_xy` (`idx_eps = [3]`, `[4]`, `[5]`) swept, all other stresses zero |
| `tension_iterative` | membrane force (`[0]`, `[1]`, `[2]`) swept, all other stresses zero; switched on automatically for membrane sweeps of asymmetric sections |
| `shear_iterative` | pure shear `n_xy` (`[2]`); also plots `D_11 - D_22` and `D_21 - D_12` |
| `synclastic_iterative` | `m_x = m_y` swept (`[3, 4]`); `include_membrane_strains` adds columns vs `eps_x`, `eps_y` |
| `nm_iterative`, `nm_ratio` | `n` swept with `m = n * nm_ratio` (`[0]` or `[1]`) |
| (all of the above `False`) | plain sweep of strain `idx_eps[0]`, all other strains zero |
| `symm_predict` / `xysymm_predict` | predict with the mid-plane symmetry / x-y relabelling wrapper (average of the network and its mirrored evaluation) |
| `plot_LFEA` | also plot the linear-elastic response |
| `multirow`, `xlim` | all six rows or only the swept component; x-axis zoom |

In the iterative modes the range is a generalised stress (`n` in N/mm, `m` in Nmm/mm) and the strain path is found
with a Newton-Raphson iteration on the section model; otherwise the range is a strain (curvature in 1/mm). Figures
go to `training/plots_single_el`.

### 3.3 Simple single-element test (`simple_single_element_test.py`)

```
cd training
python simple_single_element_test.py
```

NLFEA only (no network): all six stress components are ramped linearly from `min_` to `max_` (e.g. `m_x` and `m_xy`
together), the strain path is solved iteratively, and `sig_i` and `D_ij` are plotted against every swept component.

## 4. Output

A run of `train.py` writes its working files to `MODEL_DIR` (`training/checkpoints`) and copies them at the end to a
new version folder `LOGS_DIR/v_<n>` (`training/logs/v_1`, `v_2`, ...):

| File | Content |
| --- | --- |
| `best_trained_model__<epoch>.pt` | weights with the lowest validation loss (used for testing / inference) |
| `last_trained_model.pt` | weights after the last epoch |
| `inp.pkl` | hyperparameters |
| `stats.pkl` | normalisation statistics (needed for inference) |
| `test_data.pkl` | test set in physical units (`X_test`, `y_test`) |
| `plots/` | diagonal test plots `diagonal_match_{sig,D}_{t,u}.png` (normalised / physical units) |

`training/checkpoints`, `training/logs`, `training/plots` and `training/plots_single_el` are not tracked by git.

### 4.1 Trained model

`trained_models/v_156` is the model used in the paper (`best_trained_model__43.pt`, `inp.pkl`, `stats.pkl`; the test
set is not included). It was trained on a large sampled dataset with the defaults of `config_inp.py` (`wbMSELoss`,
weight 20, band 0.5, Sobolev weights `[0.9, 0.1]`, t- and fc-normalisation, 5 x 1024 ELU, 80 epochs, batch size 4096).
To use it elsewhere, load it with `NNPredictor` from `utils/predictor.py`. It reads the model, `inp.pkl` and
`stats.pkl` once and predicts in physical units (run from `training/`):

```python
from utils.predictor import NNPredictor

pred = NNPredictor.from_folder('trained_models', 156)
out = pred.predict(X)                          # X: (n, 13) = [6 strains, 7 geometry params], as in the sampled data
sig, D = out['sig'], out['D']                  # (n, 6) and (n, 6, 6) in [N, mm]
out = pred.predict(X, symmetry='midplane')     # mid-plane symmetric prediction ('xy': x <-> y symmetric)
```

The functions `predict_sig` / `predict_D` / `predict_sig_D` in `utils/test_utils.py` are thin wrappers around
`NNPredictor`.

### 4.2 Single-element tests

Both scripts save their figures to `training/plots_single_el` (created if needed):

| Script | File | Content |
| --- | --- | --- |
| `single_element_test.py` | `stress_path_<idx>_multirow.png` (`stress_path_<idx>.png` with `multirow = False`) | one row per stress component `sig_i` (or only the swept one); column pairs `sig_i` and `D_ij` vs the swept strain and vs the partner strain(s) of the mode (e.g. the coupled `eps_x` for bending). Curves: NLFEA (black), LFEA (grey, dotted, if `plot_LFEA`), main model (blue, dashed), comparison models (orange, green). `<idx>` is the swept component, e.g. `3` or `[3,4]` |
|  | `D_symmetry_diff_v<version>.png` | `shear_iterative` only: `D_11 - D_22` and `D_21 - D_12` vs `gamma_xy` |
| `simple_single_element_test.py` | `simple_stress_path_[<idx>].png` | NLFEA only: six rows (`sig_i`), one column pair `sig_i`, `D_ij` per swept stress component `<idx>` (e.g. `[3,5]` for `m_x`, `m_xy`) |

`simple_single_element_test.py` also keeps the results in the variable `results` (`sig_target`, `eps`, `sig`, `D`)
and prints the largest deviation between the reached and the target stresses. `single_element_test` returns the
NLFEA and NN results when called with `return_sig_D = True`.

## 5. Code structure

| File | Content |
| --- | --- |
| `train.py` | main script (training pipeline) |
| `config_inp.py` | hyperparameters |
| `single_element_test.py`, `simple_single_element_test.py` | single-element test scripts |
| `utils/data_utils.py` | reading the data, split, normalisation |
| `utils/architectures.py` | network, losses (incl. Sobolev loss), optimiser |
| `utils/train_utils.py` | training loop, directories, saving |
| `utils/predictor.py` | `NNPredictor`: loading a model version, prediction of stresses / stiffnesses in physical units (incl. mid-plane and x <-> y symmetry) |
| `utils/test_utils.py` | diagonal test plots on the test set, prediction wrappers around `NNPredictor` |
| `utils/single_element_utils.py` | single-element test, NLFEA evaluation and strain-path iteration |
| `utils/simple_single_element_utils.py` | simple single-element test |
| `trained_models/v_156` | trained model of the paper |

## 6. Notes

- **Reproducibility**: python, numpy and torch are seeded (`SEED = 42`) and cuDNN runs deterministically, so a rerun
  on the same machine gives the same model.
- **GPU**: training uses the first CUDA device if available; the stiffness prediction in testing runs on the CPU
  (`torch.func.jacrev` per sample).
- **Units**: N and mm throughout (stresses in N/mm and Nmm/mm, `fc`, `fs` in MPa).
