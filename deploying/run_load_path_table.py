"""Load-path table: NN-based FEA vs NLFEA for plate case studies (in-plane, bending and combined loading) and
cross-sections, summarised in one table. V. Balmer, 2026.

Per (case, section): the load path is run with the NN and the NLFEA (run_utils/load_path_utils.run_load_path), the
results at the plate centre are compared over the load levels converged in both, and figures are saved.
Settings: config_load_path_table.py; helpers: run_utils/load_path_utils.py (LoadPathTable), run_utils/load_path_plots.py.

Output:
    <save_path>/Load_path_table_v<model>_<timestamp>.xlsx   sheet 'Table' (layout of the paper) + sheet 'Data'
    <save_path>/Load_path_table_v<model>_<timestamp>.csv    = sheet 'Data'
    <save_path>/<case>_<section>/                           load-deformation and residual-per-load-level figures
    <save_path>/_logs/table_log_<timestamp>.log             solver output
    <data_out_root>/data_<timestamp>_<case>_<section>_v<model>/   run data (mat_res per load level)

Run: python run_load_path_table.py (from deploying/)
"""

import os
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

import config_load_path_table as cfg
from fea_utils.nn_prediction import load_NN_model
from run_utils.load_path_utils import (LoadPathTable, run_load_path, get_loadsteps, print_residual_summary, log_to,
                                      print_timings)
from run_utils.load_path_plots import plot_load_path


# ── case matrix (fixed: case studies of the paper) ────────────────────────────────────────────────────────────────
# scenario = boundary conditions / load case of Mesh_gmsh_vb, sign = sign of the load levels, kind = the load level is
# a membrane force ('n', N/mm) or a moment ('m', Nmm/mm). 9 / 109 include a fixed transverse compression (180 kN per
# edge) besides the tension / compression; 208 / 209 / 210 add membrane forces = load / 100 to the moments.
_CASES = [
    ('3D-1',  'Base cases',     'Pure shear',                                               8,   +1, 'n'),
    ('3D-2',  'Base cases',     'Pure tension in y-direction',                              109, +1, 'n'),
    ('3D-3',  'Base cases',     'Pure compression in y-direction',                          109, -1, 'n'),
    ('3D-4',  'Base cases',     'Pure tension in x-direction',                              9,   +1, 'n'),
    ('3D-5',  'Base cases',     'Pure compression in x-direction',                          9,   -1, 'n'),
    ('3D-6',  'Base cases',     'Pure torsion',                                             203, +1, 'm'),
    ('3D-7',  'Base cases',     'Pure positive bending in y-direction',                     201, +1, 'm'),
    ('3D-8',  'Base cases',     'Pure positive bending in x-direction',                     202, +1, 'm'),
    ('2D-1C', 'Combined cases', 'Anticlastic bending (x- and y-direction)',                 204, +1, 'm'),
    ('2D-2C', 'Combined cases', 'Synclastic bending (x- and y-direction)',                  205, +1, 'm'),
    ('2D-3C', 'Combined cases', 'Anticlastic bending (x- and y-direction) + torsion',       206, +1, 'm'),
    ('2D-4C', 'Combined cases', 'Synclastic bending (x- and y-direction) + torsion',        207, +1, 'm'),
    ('2D-5C', 'Combined cases', 'Anticlastic bending (x- and y-direction) + torsion + in-plane', 208, +1, 'm'),
    ('2D-6C', 'Combined cases', 'Synclastic bending (x- and y-direction) + torsion + in-plane',  209, +1, 'm'),
    ('2D-7C', 'Combined cases', 'Torsion and normal forces',                                210, +1, 'm'),
]
CASES = [dict(zip(('id', 'group', 'desc', 'scenario', 'sign', 'kind'), c)) for c in _CASES]
GROUP_ORDER = ['Base cases', 'Combined cases']


def main():
    t_start = time.time()
    table = LoadPathTable(cfg, CASES, GROUP_ORDER)
    cases, gkeys = table.selected()
    save_path = os.path.join(HERE, cfg.save_path)
    data_out_root = os.path.join(HERE, cfg.data_out_root)
    os.makedirs(data_out_root, exist_ok=True)
    os.makedirs(os.path.join(save_path, '_logs'), exist_ok=True)
    log_path = os.path.join(save_path, '_logs', datetime.now().strftime('table_log_%Y%m%d_%H%M.log'))
    log_f = open(log_path, 'w')
    print(f'[log] solver output is written to {log_path}')

    # trained model (loaded once, only if anything is run)
    any_run = any(table.to_run(c, g) for c in cases for g in gkeys)
    nn_model = load_NN_model(os.path.join(ROOT, cfg.model_dir), cfg.model_version) if any_run else None

    records, timings = {}, {}
    for case in cases:
        for g in gkeys:
            rid = table.run_id(case, g)
            records[(case['id'], g)] = None

            # 1 - load path with the NN and the NLFEA (or the latest existing run)
            folder = table.find_latest_run_folder(data_out_root, case, g)
            if table.to_run(case, g):
                levels = table.load_levels(case, g)
                print(f"\n=== {rid}  ({case['desc']}, {cfg.GEOMS[g]['label']}): {len(levels)} load levels "
                      f'{levels[0]:g} .. {levels[-1]:g}')
                settings = table.solver_settings(case, g)
                try:
                    run_dir = table.run_folder(case, g, data_out_root)
                    with log_to(log_f):
                        timings[rid] = run_load_path(table.mat_dict(case, g), levels, run_dir, settings, nn_model)
                    folder = run_dir
                except Exception:
                    timings[rid] = None
                    print(f'[{rid}] run FAILED, using the latest existing run folder ({folder}).')
                    traceback.print_exc()
            if folder is None:
                print(f'[{rid}] no run folder found, left empty in the table.')
                continue

            try:
                # 2 - iterations and remaining residual per load level
                steps = get_loadsteps(folder)
                print_residual_summary(folder, steps)

                # 3 - results at the plate centre: tracked node / element, levels converged in both, RMSE
                res = table.read_run(folder, cfg.GEOMS[g]['geom'][0])
                rec = records[(case['id'], g)] = table.table_record(res, case, g, folder)

                # 4 - figures (plate centre, levels converged in both)
                fig_dir = os.path.join(save_path, rid)
                with log_to(log_f):
                    plot_load_path(folder, fig_dir, track=dict(node=rec['node'], el=rec['el']),
                                   n_levels=rec['n_matched'] if rec['n_matched'] >= 2 else None,
                                   ylabel=table.load_label(case), tag=rid)
                print(f'[{rid}] figures saved to {fig_dir}')
            except Exception:
                print(f'[{rid}] evaluation FAILED ({folder}).')
                traceback.print_exc()

    # 5 - table of all (case, section) pairs
    xlsx_path = os.path.join(save_path, datetime.now().strftime(
        f'Load_path_table_v{cfg.model_version}_%Y%m%d_%H%M.xlsx'))
    csv_path = table.write_table(records, cases, gkeys, xlsx_path)
    print(f'\n[table] Saved: {xlsx_path}\n[table] Saved: {csv_path}')

    if timings:
        print_timings(timings)
    print(f'\nTotal time: {(time.time() - t_start) / 60:.1f} min')
    log_f.close()


if __name__ == '__main__':
    main()
