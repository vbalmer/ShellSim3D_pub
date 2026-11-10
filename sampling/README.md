# Sampler for reinforced concrete shell sections

This folder generates the dataset used to train the data-driven shell model: pairs of **generalised strains** and
**generalised stresses**, together with the **tangent stiffness**, for layered reinforced concrete (RC) shell
sections with varying thickness, reinforcement and material strengths.

## Contents

1. [Method](#1-method)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
   - 3.1 [Settings](#31-settings-constant_sampling_paramspy)
   - 3.2 [Pipeline flags](#32-pipeline-flags-sampler_analytical_rc3dpy)
4. [Output](#4-output)
   - 4.1 [Example dataset](#41-example-dataset)
5. [Code structure](#5-code-structure)
6. [Notes](#6-notes)

## 1. Method

1. **Sample generalised strains** `[eps_x, eps_y, gamma_xy, chi_x, chi_y, 2 chi_xy]`, by default as a combination of
   - a **uniform grid** over `[min, max]` (`n_per_axis_uni` points per axis, optionally with an exact zero node), and
   - a **log block**: magnitudes log-spaced between a small zero value and `min_log` / `max_log`, with random sign,
     which refines the region of small strains (cracking).

   With `sample_per_t = True` the sampling is repeated for every thickness `t` in `t_values`, and the curvature
   bounds are derived from the membrane bounds as `chi = eps / (t / chi_divisor)`, so that every thickness covers
   curvatures on its own scale.
2. **Draw geometry and material per sample**: thickness `t`, reinforcement ratios of the top and bottom layers in x
   and y, concrete strength `fc` and steel yield strength `fs`, each uniformly (or with weights) from the given lists.
   They are stored as additional input columns.
3. **Admissibility filter**: samples whose outermost layers exceed the principal-strain limits
   (`eps_2 > principal_filter_eps_2_min`, `eps_1 < principal_filter_eps_1`) are removed.
4. **Simulation**: the section is divided into `n_layer` layers. Each layer is evaluated with a plane-stress RC
   model with rotating cracks: concrete in compression with softening (fib Model Code / CSFM, or Kaufmann 1998),
   reinforcement with the tension chord model and a bilinear bare-steel law. Layer stresses are integrated over the
   thickness to `[n_x, n_y, n_xy, m_x, m_y, m_xy]`. The tangent stiffness `D` is computed with the complex-step method.
5. **Outlier removal** (optional, `REMOVE_OUTLIERS`): samples with extreme stiffness entries (interquartile-range
   criterion, factor `OUTLIER_FACTOR`) are removed.
6. **Output**: HDF5 files plus diagnostic plots.

## 2. Installation

Use the conda environment in the repository root:

```
conda env create -f ../ShellSim_vb_0_gpu.yml
conda activate ShellSim_vb_0_gpu
```

The sampler needs `numpy`, `h5py`, `matplotlib`, `pyDOE` and, for GPU runs, `cupy` (CUDA 11.x). `wandb` is
optional.

## 3. Usage

```
cd sampling
python sampler_analytical_RC3D.py
```

- **Settings**: the sampling ranges, geometry and materials are in [`constant_sampling_params.py`](constant_sampling_params.py)
  (see [3.1](#31-settings-constant_sampling_paramspy)). The pipeline flags are at the top of
  [`sampler_analytical_RC3D.py`](sampler_analytical_RC3D.py) (see [3.2](#32-pipeline-flags-sampler_analytical_rc3dpy)).
- **Environment variables**:

  | Variable | Effect |
  | --- | --- |
  | `DATA_DIR` | output directory (default: `sampling/data`) |
  | `USE_GPU=0` | run on the CPU with numpy (only for small runs; see [Notes](#6-notes)) |
  | `N_SAMPLES_3D` | set `n_samples_3D` (switches from per-axis mode to a total sample count) |
  | `LOG_WANDB=1` | log the run configuration to Weights & Biases (project `sampler_3DRC`) |

- **Dataset size**: the default settings (`n_per_axis_uni = 2`, `n_per_axis_log = 5`) produce the small example
  dataset in `sampling/data` (see below) within about a minute. For a training dataset, increase the points per
  axis, e.g. `n_per_axis_uni = 13`, `n_per_axis_log = 8` (about 38 M samples before filtering, about 8 M after,
  1.7 GB of output), set `REMOVE_OUTLIERS = True`, and write to a directory outside the repository via `DATA_DIR`.
- **Memory**: the whole dataset is held in memory, so large runs need a large-memory workstation. The simulation runs
  on the GPU in batches of `SIM_BATCH_SIZE` rows; lower it if the GPU runs out of memory.

### 3.1 Settings (`constant_sampling_params.py`)

Units: lengths [mm], stresses [MPa], strains [-], curvatures [1/mm]. Bounds are given per strain component in the
order `[eps_x, eps_y, eps_xy, chi_x, chi_y, chi_xy]` (before the shear components are doubled).

| Group | Key | Meaning |
| --- | --- | --- |
| **Method** | `sampling_type` | strain sampler: `combined_log_uniform` (default), `combined_log_lhs`, `combined_lhs_uniform`, `uniform_3D_grouped`, `uniform_3D`, `lhs`, `log` |
|  | `cm` | material model: 3 = nonlinear reinforced concrete (default), 1 = linear elastic |
| **Number of samples** | `n_samples_3D` | total number of strain samples; `None` → per-axis mode |
|  | `n_per_axis_uni` | per-axis mode: uniform grid points per axis (`n**6` samples per thickness, more with the zero node) |
|  | `n_per_axis_log` | per-axis mode: log block with `n**6` samples per thickness |
|  | `p_samples_log` | if `n_samples_3D` is set: fraction of log samples (`None` → half) |
| **Strain bounds** | `min` / `max` | bounds of the uniform grid (and LHS) |
|  | `min_log` / `max_log` | largest negative / positive magnitudes of the log block |
|  | `force_zero_axes` | add an exact zero node to every uniform axis (pure axis states, e.g. pure bending) |
|  | `n_per_axis_uni2`, `min_uni2` / `max_uni2` | optional second uniform grid with its own bounds (`combined_log_uniform` only); `None` → off |
|  | `sample_per_t` | sample once per thickness with thickness-scaled curvature bounds (recommended); `False` → single pass with random `t` per sample |
|  | `chi_divisor` | curvature bounds `chi = eps / (t / chi_divisor)`; `2.0` matches the membrane bounds at the outer fibre |
| **Admissibility filter** | `principal_filter_eps_2_min` | minimum principal strain allowed in the top and bottom layer |
|  | `principal_filter_eps_1` | maximum principal strain allowed in the top and bottom layer |
|  | `principal_filter_eps_2_min_uni2` / `principal_filter_eps_1_uni2` | the same for the second uniform grid (`None` → main limits) |
| **Cross section** | `n_layer` | number of layers |
|  | `t` / `t_values` / `t_probs` | thickness: constant `t`, or drawn per sample from `t_values` with optional weights `t_probs` |
|  | `rho_x`, `rho_y` | baseline reinforcement pattern per layer (bottom layer first); nonzero entries mark the reinforced layers |
|  | `rho_xtop_values`, `rho_xbot_values`, `rho_ytop_values`, `rho_ybot_values` | ratios drawn per sample for the reinforced layers in the upper / lower half. If only one of top / bot is set, one shared value is drawn for the whole section; if both are `None`, the baseline pattern is used |
|  | `D` | rebar diameter |
| **Materials** | `fc` / `fc_values` / `fc_probs` | concrete cylinder strength: constant, or drawn per sample. The other concrete parameters (`Ec`, `fct`, bond stresses, ...) are interpolated from the table in `utils/concrete_classes.py` |
|  | `fs` / `fs_values` / `fs_probs` | steel yield strength: constant, or drawn per sample. `fsu = 1.08 fsy`, `Es`, `Esh` from `utils/steel_classes.py` |
|  | `cmcc` | concrete in compression: 1 = fib Model Code / CSFM, 2 = Kaufmann (1998) |
|  | `cmcs` | compression softening: 1 = CSFM, 2 = Kaufmann (1998) |
|  | `nu` | Poisson's ratio |

With `sample_per_t` (or a single thickness), only the membrane entries of the bounds matter; the curvature entries
are derived from them.

### 3.2 Pipeline flags (`sampler_analytical_RC3D.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `FILTER_DATA` | `True` | apply the admissibility filter |
| `REMOVE_OUTLIERS` / `OUTLIER_FACTOR` | `False` / `80` | stiffness outlier removal and its IQR factor; only use for large sampled data amounts (> 1M) |
| `SAVE_D` | `True` | save the stiffness matrices |
| `PLOT` | `True` | write diagnostic plots to `sampling/plots` |
| `SIM_BATCH_SIZE` | `1_000_000` | rows per simulation batch |
| `FILTER_BATCHES` | `10` | chunks for the admissibility filter |
| `SEED` | `42` | seed of the strain sampling and of the geometry / material draw |

## 4. Output

All files are written to `DATA_DIR` as float32 HDF5 datasets (dataset name = file name without `output_`).
Row `i` of all files belongs to the same sample.

| File | Shape | Columns | Units |
| --- | --- | --- | --- |
| `output_eps_g.h5` | (n, 6) | `eps_x, eps_y, gamma_xy, chi_x, chi_y, 2 chi_xy` | [-], [1/mm] |
| `output_geom.h5` | (n, n_geom) | `t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs` (only the drawn ones) | [mm], [-], [MPa] |
| `output_sig_g.h5` | (n, 6) | `n_x, n_y, n_xy, m_x, m_y, m_xy` | [N/mm], [Nmm/mm] |
| `output_D.h5` | (n, 6, 6) | `D[i, j] = d sig_i / d eps_j`, blocks `[[D_m, D_mb], [D_bm, D_b]]` | consistent with the above |

`gamma_xy` and `2 chi_xy` are the engineering shear strain and twist, i.e. the work-conjugate pairs of `n_xy` and
`m_xy`. `rho_?top` applies to the reinforced layers in the upper half of the section, `rho_?bot` to those in the
lower half (see `rho_x` / `rho_y` in [3.1](#31-settings-constant_sampling_paramspy)).

Diagnostic plots (`sampling/plots`): all 36 stiffness entries along `eps_x` for samples with all other strains ≈ 0
(`filtered_dataset_D.png`, `matrix_imshow.png`), and the post-filter distribution of every drawn geometry /
material column (`<name>_distribution.png`).

### 4.1 Example dataset

`sampling/data` contains a small example dataset (29,848 samples, 6.3 MB) produced by running the script with its
default settings on the GPU (`SEED = 42`, no outlier removal). It is meant for testing the downstream code, not for training.

## 5. Code structure

| File | Content |
| --- | --- |
| `sampler_analytical_RC3D.py` | main script (pipeline) |
| `constant_sampling_params.py` | settings |
| `utils/sampler_utils_RC3D.py` | strain samplers, geometry / material draw, filter, outlier removal, saving |
| `utils/simulating_sig_vec_RC3D.py` | layered section simulator (strains → stresses, stiffness) |
| `utils/constitutive_laws.py` | RC layer material model |
| `utils/concrete_classes.py`, `utils/steel_classes.py` | material parameter tables |
| `utils/defcplx_np.py` | complex array type for the complex-step derivative |
| `utils/plotting.py` | diagnostic plots |
| `utils/config.py` | CPU / GPU backend switch |

## 6. Notes

- **Reproducibility**: with a fixed `SEED` and the same backend, the sampler produces identical datasets.
- **CPU vs. GPU**: stresses agree between the backends to float32 precision. At degenerate strain states (e.g. all
  strains exactly zero, or `eps_x = eps_y` with `gamma_xy = 0`), the principal direction of the rotating-crack model
  is not unique, so the complex-step tangent depends on round-off and can differ between numpy and cupy (including
  sign changes of off-diagonal entries). The datasets were generated on the GPU.
- The simulator uses complex64 arithmetic for the complex-step derivative, so stresses are accurate to float32
  precision.
