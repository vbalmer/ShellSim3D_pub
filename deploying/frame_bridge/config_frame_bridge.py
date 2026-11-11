"""Settings of run_frame_bridge.py: load path of the frame bridge (FB), NN vs NLFEA. Geometry and reinforcement of the
bridge: frame_bridge_geom.py."""

# trained model (training/trained_models/v_<model_version>)
model_dir     = 'training/trained_models'      # relative to the repository root
model_version = 156

# material (all regions)
fc = 16.3           # concrete strength [MPa]
fs = 440            # steel yield strength [MPa]

# load levels: area load q on the middle of the top slab [N/mm2]
load_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38]

# solver
numit             = 12          # max. Newton-Raphson iterations per load level
res_tol           = 5           # convergence: max. residual [N] < res_tol
symm_predict      = True        # mid-plane symmetry wrapper of the NN
gauss_order_nn    = 2           # Gauss points per direction, NN run
gauss_order_nlfea = 2           # Gauss points per direction, NLFEA run
nlfea_control     = False       # NN run: also evaluate the NLFEA at every NN iteration (comparison arrays, slower)
opt_time          = False       # NN stress and tangent from one pass, one B^T D B per element
conv_plots        = True        # save a convergence plot of every iteration

# output folders, relative to deploying/
save_path     = 'plots_frame_bridge'           # figures
data_out_root = 'data_out'                     # run data
