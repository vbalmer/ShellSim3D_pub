"""Frame bridge (FB): load path of the frame bridge (two walls and a top slab, area load on the slab) with the
NN-based FEA and the NLFEA. V. Balmer, 2026.

Settings: config_frame_bridge.py; geometry and reinforcement: frame_bridge_geom.py (both in this folder).

Output (relative to deploying/):
    <data_out_root>/data_<timestamp>_FB_v<model>/       run data (mat_res per load level, convergence plots)
    <save_path>/FB_v<model>_<timestamp>/                load-deformation figure (roof centre) and residual per load level

Run: python run_frame_bridge.py (from deploying/frame_bridge/)
"""

import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))       # deploying/frame_bridge
DEPLOY = os.path.dirname(HERE)                          # deploying (fea_utils, run_utils, output folders)
ROOT = os.path.dirname(DEPLOY)                          # repository root (trained model)
if DEPLOY not in sys.path:
    sys.path.append(DEPLOY)

import config_frame_bridge as cfg
from frame_bridge_geom import fb_mat_dict, ROOF_CENTRE
from fea_utils.nn_prediction import load_NN_model
from run_utils.load_path_utils import (solver_settings, run_load_path, run_folder, get_loadsteps,
                                      print_residual_summary, load_mat_res, nearest_node_element, print_timings)
from run_utils.load_path_plots import plot_load_path


def main():
    t_start = time.time()
    folder = run_folder(os.path.join(DEPLOY, cfg.data_out_root), 'FB', cfg.model_version)
    fig_dir = os.path.join(DEPLOY, cfg.save_path,
                           f"FB_v{cfg.model_version}_{datetime.now().strftime('%Y%m%d_%H%M')}")

    nn_model = load_NN_model(os.path.join(ROOT, cfg.model_dir), cfg.model_version)

    # 1 - load path with the NN and the NLFEA
    settings = solver_settings(cfg)
    timings = run_load_path(fb_mat_dict(cfg.fc, cfg.fs), cfg.load_levels, folder, settings, nn_model)

    # 2 - iterations and remaining residual per load level (with the absolute res_tol the bridge typically uses all
    #     numit iterations; the residual relative to the applied load shows how well equilibrium is reached)
    steps = get_loadsteps(folder)
    print_residual_summary(folder, steps)

    # 3 - results at the roof centre: tracked node / element
    node, el = nearest_node_element(load_mat_res(folder, steps[0], 'norm'), ROOF_CENTRE)

    # 4 - figures (roof centre, all load levels)
    plot_load_path(folder, fig_dir, track=dict(node=node, el=el), n_levels=len(steps),
                   ylabel='Area load q [N/mm2]', tag='FB')

    print(f"\nRun data: {folder}\nFigures:  {fig_dir}")
    print_timings({'FB': timings})
    print(f'\nTotal time: {(time.time() - t_start) / 60:.1f} min')


if __name__ == '__main__':
    main()
