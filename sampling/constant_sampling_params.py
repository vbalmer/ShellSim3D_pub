"""Sampling settings for sampler_analytical_RC3D.py. See README.md (section 'Settings') for details.

Units: lengths [mm], stresses [MPa], strains [-], curvatures [1/mm].
Strain order: [eps_x, eps_y, eps_xy, chi_x, chi_y, chi_xy] (eps_xy / chi_xy are doubled to gamma_xy / 2 chi_xy
after sampling).
"""

constants = {
    # ---------------------------------------------------------------- method
    'sampling_type': 'combined_log_uniform',    # strain sampler, see README.md for the available samplers
    'cm': 3,                                # material model: 3 = nonlinear reinforced concrete, 1 = linear elastic

    # ---------------------------------------------------------------- number of samples
    'n_samples_3D': None,                   # total strain samples; None -> per-axis mode (n_per_axis_* below)
    # defaults give the small example dataset; training dataset: e.g. 13 / 8 (see README.md, 'Dataset size')
    'n_per_axis_uni': 2,                    # uniform grid: points per axis (n**6 per thickness)
    'n_per_axis_log': 5,                    # log block: n**6 points per thickness
    'p_samples_log': 0.2,                   # only if n_samples_3D is set: fraction of log samples (None -> 0.5)

    # ---------------------------------------------------------------- strain bounds
    'min':     [-3e-3]*2 + [-4e-3] + [-0.06e-3]*2 + [-0.08e-3],      # uniform grid / LHS lower bounds
    'max':     [5e-3]*2  + [4e-3]  + [0.10e-3]*2  + [0.08e-3],       # uniform grid / LHS upper bounds
    'min_log': [-3e-3]*2 + [-50e-3] + [-0.06e-3]*2 + [-0.8e-3],      # log block: largest negative magnitudes
    'max_log': [50e-3]*2 + [50e-3]  + [1e-3]*2     + [0.8e-3],       # log block: largest positive magnitudes
    'force_zero_axes': True,                # add an exact 0 node to each uniform axis (pure axis states)

    # optional second uniform grid with its own bounds (combined_log_uniform only); None -> off
    'n_per_axis_uni2': None,
    'min_uni2': [-4e-3]*2 + [-5e-3] + [-0.06e-3]*2 + [-0.08e-3],
    'max_uni2': [6e-3]*2  + [5e-3]  + [0.10e-3]*2  + [0.08e-3],

    # ---------------------------------------------------------------- curvature scaling with thickness
    'sample_per_t': True,                   # sample once per t in t_values with chi bounds = eps bounds / (t/chi_divisor)
    'chi_divisor': 2.0,                     # 2.0: curvature bounds match the membrane bounds at the outer fibre

    # ---------------------------------------------------------------- admissibility filter (outer layers)
    'principal_filter_eps_2_min': -3e-3,    # keep rows with eps_2 > this in the top and bottom layer
    'principal_filter_eps_1': 50e-3,        # keep rows with eps_1 < this in the top and bottom layer
    'principal_filter_eps_2_min_uni2': -4e-3,   # same limits for the second uniform grid; None -> main limits
    'principal_filter_eps_1_uni2': 50e-3,

    # ---------------------------------------------------------------- cross section
    'n_layer': 20,                          # number of layers over the thickness
    't': 300,                               # [mm] thickness if t_values is None
    't_values': [100, 200, 300, 400, 500],  # [mm] thicknesses drawn per sample; None -> constant t
    't_probs': None,                        # draw weights for t_values (need not sum to 1); None -> uniform

    # reinforcement: baseline pattern per layer (nonzero = reinforced layer, bottom layer first)
    'rho_x': [0.025]*4 + [0]*12 + [0.025]*4,
    'rho_y': [0.025]*4 + [0]*12 + [0.025]*4,
    # ratios drawn per sample for the reinforced layers in the upper (top) / lower (bot) half;
    # only one of top/bot set -> one shared value; both None -> baseline pattern
    'rho_xtop_values': [0.005, 0.015, 0.025, 0.035, 0.045],
    'rho_xbot_values': [0.005, 0.015, 0.025, 0.035, 0.045],
    'rho_ytop_values': [0.005, 0.015, 0.025, 0.035, 0.045],
    'rho_ybot_values': [0.005, 0.015, 0.025, 0.035, 0.045],
    'D': 16,                                # [mm] rebar diameter

    # ---------------------------------------------------------------- materials
    'fc': 30,                               # [MPa] concrete strength if fc_values is None
    'fc_values': [16, 25, 35, 45, 50],      # [MPa] drawn per sample; None -> constant fc
    'fc_probs': None,                       # draw weights for fc_values; None -> uniform
    'fs': 435,                              # [MPa] steel yield strength if fs_values is None
    'fs_values': [400, 450, 500, 550, 600], # [MPa] drawn per sample; None -> constant fs
    'fs_probs': None,                       # draw weights for fs_values; None -> uniform
    'cmcc': 1,                              # concrete in compression: 1 = fib Model Code / CSFM, 2 = Kaufmann (1998)
    'cmcs': 1,                              # compression softening:   1 = CSFM,                  2 = Kaufmann (1998)
    'nu': 0,                                # Poisson's ratio
}
