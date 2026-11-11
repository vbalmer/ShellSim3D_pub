"""Settings of run_load_path_table.py: case studies (plates under in-plane / bending / combined loading) x
cross-sections, NN vs NLFEA, summarised in one table.

Default: example run of case 3D-8 (pure bending in x) with the isotropic section. Full table of the paper: cases = 'all',
geoms = 'all' (30 load paths).
"""

# trained model (training/trained_models/v_<model_version>)
model_dir     = 'training/trained_models'      # relative to the repository root
model_version = 156

# solver
numit             = 12          # max. Newton-Raphson iterations per load level
res_tol           = 5           # convergence: max. residual [N] < res_tol
# per-case override of res_tol. 2D-2C: the first load level is a large step from the linear elastic solution; the
# cracked tangent of the first iteration is indefinite, so the residual can stall just above 5 and the next level would
# then restart from the linear elastic solution
res_tol_case      = {('2D-6C', 'iso'): 15, ('2D-2C', 'iso'): 10, ('2D-2C', 'aniso'): 10}
symm_predict      = True        # mid-plane symmetry wrapper of the NN (exact for mid-plane symmetric sections)
gauss_order_nn    = 2           # Gauss points per direction, NN run
gauss_order_nlfea = 2           # Gauss points per direction, NLFEA run
nlfea_control     = False       # NN run: also evaluate the NLFEA at every NN iteration (comparison arrays, slower)
opt_time          = False       # NN stress and tangent from one pass, one B^T D B per element (same results up to
                                # round-off)
conv_plots        = True        # save a convergence plot of every iteration

# cross-sections: [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs] in [mm], [-], [MPa]
GEOMS = {
    'iso':   dict(label='isotropic',   geom=[300, 0.025, 0.025, 0.025, 0.025, 30, 450]),
    'aniso': dict(label='anisotropic', geom=[500, 0.006, 0.0375, 0.0375, 0.006, 25, 400]),
}

PLATE = dict(L=6_000, B=6_000, ms=600)       # plate dimensions and mesh size [mm]

# 'all', or lists of case ids / section keys, e.g. ['3D-1', '2D-2C'] / ['aniso']
cases = ['3D-8']
geoms = ['iso']

# True: run all selected (case, section) pairs; False: only rebuild the table from the latest existing run folders;
# list of (case, section) pairs: re-run only these, e.g. [('3D-6', 'aniso')]
run = True

# default load levels = sign * capacity * fractions (rounded to 3 significant digits)
fractions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.88, 0.94, 0.97, 1.0]

# approximate capacities in load-level units (n [N/mm] for 3D-1 ... 3D-5, m [Nmm/mm] otherwise)
capacity = {
    ('3D-1',  'iso'): 1_410,   ('3D-1',  'aniso'): 1_820,
    ('3D-2',  'iso'): 1_410,   ('3D-2',  'aniso'): 600,
    ('3D-3',  'iso'): 10_200,  ('3D-3',  'aniso'): 10_200,
    ('3D-4',  'iso'): 1_410,   ('3D-4',  'aniso'): 600,
    ('3D-5',  'iso'): 10_200,  ('3D-5',  'aniso'): 10_200,
    ('3D-6',  'iso'): 175e3,   ('3D-6',  'aniso'): 195e3,
    ('3D-7',  'iso'): 175e3,   ('3D-7',  'aniso'): 30e3,
    ('3D-8',  'iso'): 175e3,   ('3D-8',  'aniso'): 620e3,
    ('2D-1C', 'iso'): 175e3,   ('2D-1C', 'aniso'): 120e3,
    ('2D-2C', 'iso'): 175e3,   ('2D-2C', 'aniso'): 30e3,
    ('2D-3C', 'iso'): 115e3,   ('2D-3C', 'aniso'): 90e3,
    ('2D-4C', 'iso'): 95e3,    ('2D-4C', 'aniso'): 120e3,
    ('2D-5C', 'iso'): 70e3,    ('2D-5C', 'aniso'): 35e3,
    ('2D-6C', 'iso'): 60e3,    ('2D-6C', 'aniso'): 38e3,
    ('2D-7C', 'iso'): 210e3,   ('2D-7C', 'aniso'): 275e3,
}

# explicit load levels, override capacity * fractions for a (case, section) pair
load_levels = {
    ('3D-1',  'aniso'): [100, 250, 400, 550, 700, 800, 880, 940, 970, 1000],
    ('3D-8',  'aniso'): [100e3, 200e3, 300e3, 400e3, 500e3, 550e3, 600e3, 620e3, 640e3, 650e3, 660e3],
    ('2D-2C', 'aniso'): [25e3, 50e3, 75e3, 90e3, 100e3, 110e3, 120e3, 130e3, 140e3, 148e3, 155e3],
    ('3D-7',  'aniso'): [25e3, 50e3, 75e3, 90e3, 100e3, 110e3, 115e3, 120e3, 125e3, 130e3, 140e3, 148e3],
    ('2D-6C', 'iso'):   [6e3, 15e3, 24e3, 33e3, 42e3, 48e3, 52.8e3, 54e3, 56.4e3, 58.2e3, 60e3],
    ('2D-5C', 'aniso'): [8.75e3, 14e3, 19.2e3, 24.5e3, 28e3, 30.8e3, 32.9e3, 34e3, 35e3, 38e3, 41e3, 44e3],
}

# a load level counts as converged if the stress resultants at the plate-centre element match the applied (uniform)
# field within m_tol (relative to its largest component); the table uses the levels converged in both NN and NLFEA
m_tol = 0.02

# table cut-off: only levels with |load| <= table_max (load-level units) are used, without re-running
table_max = {}

# output folders, relative to deploying/
save_path     = 'plots_load_path_table'        # table (xlsx / csv), figures, logs
data_out_root = 'data_out'                     # run data, one folder per load path
