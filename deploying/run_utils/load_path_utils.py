"""Load paths: running a load path with the NN and the NLFEA, run folders, reading the results back, the residual
summary and the load-path table of the plate case studies. Shared by run_load_path_table.py and
frame_bridge/run_frame_bridge.py. V. Balmer, 2026.

Output of a load path (out_dir):
    <out_dir>/<load level>/mat_res_NN.pkl      NN run (stresses and tangents from the NN)
    <out_dir>/<load level>/mat_res_norm.pkl    NLFEA run
    <out_dir>/<load level>/conv_plots/         convergence plots per iteration (conv_plt_<i>NN / NLFEA.png)
"""

import os
import csv
import glob
import time
import pickle
import contextlib
from datetime import datetime

import numpy as np

from fea_utils.Main_vb_vec import main_solver                 # fea_utils first: adds the repository root to sys.path
from fea_utils.Mesh_gmsh_vb import SCENARIO_FB, FS_DEFAULT
from sampling.utils.concrete_classes import mat_from_fc
from sampling.utils.steel_classes import mat_from_fs


################################## input ##################################

def extend_material_parameters(mat: dict) -> dict:
    """
    Add the steel (fsy, fsu, Es, Esh from fs) and concrete parameters (Ec, tb0, tb1, ect, ec0, fcp, fct from fc),
    interpolated as in the sampling (sampling/utils/steel_classes.py, concrete_classes.py), and the bar diameter D
    (default 16 mm). The frame bridge may override fc / fs per region (resolved per element in Mesh_gmsh_vb).

    Args:
        mat     (dict): model input with 'fc' [MPa] and optionally 'fs' [MPa] (default 435), 'D' [mm]

    Returns:
        dict: mat with the material parameters added (modified in place)
    """
    fs = float(np.asarray(mat.get('fs', FS_DEFAULT)).reshape(-1)[0])
    mat_s = mat_from_fs(fs)
    D = float(np.asarray(mat.get('D', 16)).reshape(-1)[0])
    mat.update({'fs': np.array([fs]), 'fsy': mat_s['fsy'], 'fsu': mat_s['fsu'],
                'Es': mat_s['Es'], 'Esh': mat_s['Esh'], 'D': D})

    if 'fc' not in mat:
        raise KeyError("The model input needs the concrete strength 'fc' [MPa].")
    mat_c = mat_from_fc(float(np.asarray(mat['fc']).reshape(-1)[0]))
    mat['E_1'] = mat_c['Ec']
    for key in ('tb0', 'tb1', 'ect', 'ec0', 'fcp', 'fct'):
        mat[key] = mat_c[key]
    return mat


################################## run ##################################

def solver_settings(cfg, res_tol: float = None) -> dict:
    """
    Solver settings of a load path from a config module (config_load_path_table.py / config_frame_bridge.py).

    Args:
        cfg     (module): settings with numit, res_tol, symm_predict, gauss_order_nn, gauss_order_nlfea,
                          nlfea_control, opt_time, conv_plots
        res_tol (float):  override of cfg.res_tol (e.g. per case); None: cfg.res_tol

    Returns:
        dict: settings for run_load_path
    """
    return {'numit': cfg.numit, 'res_tol': cfg.res_tol if res_tol is None else res_tol,
            'symm_predict': cfg.symm_predict, 'gauss_order_nn': cfg.gauss_order_nn,
            'gauss_order_nlfea': cfg.gauss_order_nlfea, 'nlfea_control': cfg.nlfea_control,
            'opt_time': cfg.opt_time, 'conv_plots': cfg.conv_plots}


@contextlib.contextmanager
def log_to(log_f):
    """Redirect stdout / stderr (solver output) to the open file log_f; None: no redirection."""
    if log_f is None:
        yield
        return
    with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        try:
            yield
        finally:
            log_f.flush()


def print_timings(timings: dict):
    """
    Run times per load path.

    Args:
        timings (dict): {name: {'t_nn', 't_nlfea'} [s] from run_load_path, or None for a failed run}
    """
    print('\n' + '=' * 52 + '\nTIMING SUMMARY (per load path)')
    for name, t in timings.items():
        if t is None:
            print(f'  {name:<12} FAILED')
            continue
        t_nn = f"{t['t_nn'] / 60:6.1f}" if t['t_nn'] is not None else '     -'
        print(f"  {name:<12} NN {t_nn} min, NLFEA {t['t_nlfea'] / 60:6.1f} min")
    print('=' * 52)


def run_load_path(mat: dict, levels: list, out_dir: str, settings: dict, nn_model=None) -> dict:
    """
    Run one load path with the NN (if nn_model is given) and with the NLFEA.

    Args:
        mat         (dict):         model input (plate_mat_dict / frame_bridge.fb_mat_dict)
        levels      (list):         load levels: edge load [N/mm] or moment [Nmm/mm] (plates), area load [N/mm2]
                                    (frame bridge)
        out_dir     (str):          output folder (one subfolder per load level)
        settings    (dict):         numit, res_tol, symm_predict, gauss_order_nn, gauss_order_nlfea, nlfea_control,
                                    opt_time, conv_plots
        nn_model    (NNPredictor):  trained model; None: NLFEA only

    Returns:
        dict: 't_nn', 't_nlfea' run times [s] (t_nn None without NN)
    """
    mat = dict(mat)
    scenario = int(np.array(mat['s']).flatten()[0])
    levels = np.asarray(levels, dtype=float)
    mat['F'] = levels * mat['L'] * mat['B'] if scenario == SCENARIO_FB else mat['L'] * levels
    mat['F_N'] = np.array([0])
    mat = extend_material_parameters(mat)
    conv_plots = settings.get('conv_plots', True)

    t_nn = None
    if nn_model is not None:
        NN_hybrid = {'predict_sig': True, 'predict_D': True, 'model_dim': 'THREEDIM',
                     'numit': settings['numit'], 'res_tol': settings['res_tol'],
                     'symm_predict': settings['symm_predict'], 'gauss_order': settings['gauss_order_nn'],
                     'nlfea_control': settings['nlfea_control'], 'opt_time': settings['opt_time']}
        t0 = time.perf_counter()
        main_solver(dict(mat), NN_hybrid, out_dir, nn_model=nn_model, conv_plots=conv_plots)
        t_nn = time.perf_counter() - t0

    NN_hybrid = {'predict_sig': False, 'predict_D': False, 'numit': settings['numit'], 'res_tol': settings['res_tol'],
                 'gauss_order': settings['gauss_order_nlfea'], 'opt_time': settings['opt_time']}
    t0 = time.perf_counter()
    main_solver(dict(mat), NN_hybrid, out_dir, conv_plots=conv_plots)
    t_nlfea = time.perf_counter() - t0

    print('\n' + '=' * 52)
    print('TIMING SUMMARY (entire load path)')
    if t_nn is not None:
        print(f'  NN pipeline : {t_nn / 60:.2f} min ({t_nn:.1f} s)')
    print(f'  NLFEA       : {t_nlfea / 60:.2f} min ({t_nlfea:.1f} s)')
    print('=' * 52 + '\n')
    return {'t_nn': t_nn, 't_nlfea': t_nlfea}


################################## run folders ##################################

def run_folder(data_out_root: str, name: str, model_version) -> str:
    """
    New run folder <data_out_root>/data_<timestamp>_<name>_v<model_version> (created).

    Args:
        data_out_root   (str):      root folder of the run data
        name            (str):      name of the load path, e.g. '3D-8_iso' or 'FB'
        model_version   (int|str):  version of the trained model

    Returns:
        str: path of the run folder
    """
    folder = os.path.join(data_out_root, datetime.now().strftime(f'data_%Y%m%d_%H%M_{name}_v{model_version}'))
    os.makedirs(folder, exist_ok=True)
    return folder


def find_latest_run_folder(data_out_root: str, name: str, model_version):
    """Most recent run folder of a load path (timestamped names sort chronologically), or None (see run_folder)."""
    matches = sorted(glob.glob(os.path.join(data_out_root, f'data_*_{name}_v{model_version}')))
    return matches[-1] if matches else None


################################## reading ##################################

def get_loadsteps(folder: str) -> np.ndarray:
    """Load levels of a load path (numeric subfolder names), sorted by magnitude."""
    names = [n for n in os.listdir(folder) if os.path.isdir(os.path.join(folder, n))]
    steps = np.array(names, dtype=np.float32)
    return steps[np.argsort(np.abs(steps))]


def level_folder(folder: str, level: float) -> str:
    """Subfolder of a load level ('%g' formatting, as written by the solver)."""
    return os.path.join(folder, '%g' % level)


def get_mat_res_path(level_dir: str, tag: str) -> str:
    """
    Result file of one load level.

    Args:
        level_dir   (str): load-level folder
        tag         (str): 'NN' or 'norm' (NLFEA); 'NN' also matches mat_res_NN_sig / _D.pkl

    Returns:
        str: path of mat_res_<tag>*.pkl
    """
    exact = os.path.join(level_dir, 'mat_res_' + tag + '.pkl')
    if os.path.exists(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(level_dir, 'mat_res_' + tag + '*.pkl')))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No 'mat_res_{tag}*.pkl' file found in {level_dir}")


def load_mat_res(folder: str, level: float, tag: str) -> dict:
    """mat_res of one load level ('NN' or 'norm')."""
    with open(get_mat_res_path(level_folder(folder, level), tag), 'rb') as fh:
        return pickle.load(fh)


def displ_fields(mat_res: dict) -> np.ndarray:
    """Nodal displacements / rotations (n_nodes, 6): [ux, uy, uz, thx, thy, thz] in [mm], [rad]."""
    return np.column_stack([np.ravel(mat_res[k]) for k in ['ux', 'uy', 'uz', 'thx', 'thy', 'thz']])


def eps_fields(mat_res: dict) -> np.ndarray:
    """Generalised strains at the element centres (n_elements, 6): [eps_x, eps_y, gamma_xy, chi_x, chi_y, 2 chi_xy]."""
    eps_g = np.asarray(mat_res['eps_g'])
    return eps_g[:, 0, :6] if eps_g.ndim == 3 else eps_g[:, 0, 0, :6]


def sig_fields(mat_res: dict) -> np.ndarray:
    """Stress resultants at the element centres (n_elements, 6): [n_x, n_y, n_xy, m_x, m_y, m_xy]."""
    sig_g = np.asarray(mat_res['sig_g'])
    return sig_g[:, 0, :6] if sig_g.ndim == 3 else sig_g[:, 0, 0, :6]


def nearest_node_element(mat_res: dict, xyz) -> tuple:
    """Node and element (centre) closest to the point xyz [mm]. Returns (node, element) indices."""
    xyz = np.asarray(xyz, dtype=float)
    node = int(np.argmin(np.linalg.norm(np.asarray(mat_res['NODESG']) - xyz, axis=1)))
    el = int(np.argmin(np.linalg.norm(np.asarray(mat_res['COORD_c'][0]) - xyz, axis=1)))
    return node, el


def n_iterations(mat_res: dict) -> int:
    """
    Number of equilibrium iterations of a load level (numit: the solver stopped at the maximum number of iterations;
    it may still have reached res_tol in the last one).
    The load-path table uses a different, structural criterion instead (LoadPathTable.read_run).
    """
    return len(np.atleast_1d(mat_res['convrm']))


def residual_summary(folder: str, tag: str) -> tuple:
    """
    Per load level of a run: number of iterations and the residual of the last iteration relative to the applied
    load, separately for forces (translational DOFs) and moments (rotational DOFs), sums of |.| over all nodes.

    Args:
        folder  (str): load-path folder
        tag     (str): 'NN' or 'norm' (NLFEA)

    Returns:
        tuple: n_iter (n_levels,), residual forces / applied forces (n_levels,), residual moments / applied moments
               (n_levels,); nan if no forces / moments are applied
    """
    n_it, ratio_f, ratio_m = [], [], []
    for s in get_loadsteps(folder):
        res = load_mat_res(folder, s, tag)
        fe = np.abs(np.asarray(res['fe']).flatten())
        f_app = fe[0::6].sum() + fe[1::6].sum() + fe[2::6].sum()
        m_app = fe[3::6].sum() + fe[4::6].sum() + fe[5::6].sum()
        n_it.append(n_iterations(res))
        ratio_f.append(abs(np.atleast_1d(res['convrf'])[-1]) / f_app if f_app > 0 else np.nan)
        ratio_m.append(abs(np.atleast_1d(res['convrm'])[-1]) / m_app if m_app > 0 else np.nan)
    return np.array(n_it), np.array(ratio_f), np.array(ratio_m)


def print_residual_summary(folder: str, steps=None):
    """
    Print the iterations and the remaining residual relative to the applied load per load level (forces and moments
    separately, '-' if none applied), for the NLFEA and the NN run (see residual_summary).

    Args:
        folder  (str):    load-path folder
        steps   (np.arr): load levels (default: get_loadsteps(folder))
    """
    steps = get_loadsteps(folder) if steps is None else steps
    fmt = lambda r: f'{"-":>8}' if np.isnan(r) else f'{r:8.1e}'
    print('\nload level                   ' + ' '.join(f'{s:8g}' for s in steps))
    for key, tag in (('NLFEA', 'norm'), ('NN', 'NN')):
        try:
            n_it, ratio_f, ratio_m = residual_summary(folder, tag)
        except FileNotFoundError:
            continue                                    # no NN run
        print(f'{key:<6} iterations           ' + ' '.join(f'{n:8d}' for n in n_it))
        print(f'{key:<6} res. forces / load   ' + ' '.join(fmt(r) for r in ratio_f))
        print(f'{key:<6} res. moments / load  ' + ' '.join(fmt(r) for r in ratio_m))

################################## load-path table (plate case studies) ##################################

# ── table format ──────────────────────────────────────────────────────────────────────────────────────────────────
LOAD_UNIT = {'n': 'kN/m', 'm': 'kNm/m'}
LOAD_SCALE = {'n': 1.0, 'm': 1e-3}             # load level [N/mm | Nmm/mm] -> [kN/m | kNm/m]

# 9 values per run: eps_x, eps_y, gamma_xy [‰], chi_x, chi_y, chi_xy [mrad/m], u_x, u_y, u_z [mm]
COMP_KEYS = ['eps_x', 'eps_y', 'gamma_xy', 'chi_x', 'chi_y', 'chi_xy', 'u_x', 'u_y', 'u_z']
# table blocks: (title, unit, components as (symbol, subscript))
BLOCKS = [('Strains | Maximum strain reached', '‰ or mrad/m',
           [('ε', 'x'), ('ε', 'y'), ('γ', 'xy'), ('χ', 'x'), ('χ', 'y'), ('χ', 'xy')]),
          ('Displacements | Maximum displacement reached', 'mm',
           [('u', 'x'), ('u', 'y'), ('u', 'z')])]


def plate_mat_dict(scenario: int, geom: list, plate: dict) -> dict:
    """
    Model input of a plate case (scenarios 8, 9, 109, 201 - 210).

    Args:
        scenario    (int):  boundary conditions / load case, see Mesh_gmsh_vb
        geom        (list): cross-section [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs] in [mm], [-], [MPa]
        plate       (dict): L, B (plate dimensions [mm]) and ms (mesh size [mm])

    Returns:
        dict: model input, every value as a length-1 array
    """
    t, rxt, rxb, ryt, ryb, fc, fs = geom
    d = dict(L=plate['L'], B=plate['B'], H=0, fc=fc, E_1=0, E_2=0, ms=plate['ms'], s=scenario,
             t_1=t, t_2=0, nl=20, nu_1=0, nu_2=0, mat=3,
             rho_x_top=rxt, rho_x_bot=rxb, rho_y_top=ryt, rho_y_bot=ryb, fs=fs)
    return {k: np.array([v]) for k, v in d.items()}


class LoadPathTable:
    """
    Load-path table of plate case studies x cross-sections, NN vs NLFEA.

    Args:
        cfg         (module):   settings (config_load_path_table.py)
        cases       (list):     case matrix, dicts with id, group, desc, scenario (Mesh_gmsh_vb), sign (of the load
                                levels), kind ('n': membrane force [N/mm], 'm': moment [Nmm/mm])
        group_order (list):     order of the case groups in the table
    """

    def __init__(self, cfg, cases, group_order):
        self.cfg = cfg
        self.cases = cases
        self.group_order = group_order

    def selected(self):
        """Cases and section keys selected in the settings (cfg.cases, cfg.geoms). Returns (cases, gkeys)."""
        cfg = self.cfg
        cases = [c for c in self.cases if cfg.cases == 'all' or c['id'] in cfg.cases]
        gkeys = [g for g in cfg.GEOMS if cfg.geoms == 'all' or g in cfg.geoms]
        return cases, gkeys

    def to_run(self, case, gkey):
        """True if the load path of a (case, section) pair is (re-)run (cfg.run), False: latest existing run."""
        run = self.cfg.run
        return run is True or (isinstance(run, (list, tuple)) and (case['id'], gkey) in run)

    ################################## running ##################################

    @staticmethod
    def run_id(case, gkey):
        """Name of a (case, section) pair, e.g. '3D-8_iso'."""
        return f"{case['id']}_{gkey}"

    def mat_dict(self, case, gkey):
        """Model input of a (case, section) pair (plate of cfg.PLATE, section of cfg.GEOMS)."""
        return plate_mat_dict(case['scenario'], self.cfg.GEOMS[gkey]['geom'], self.cfg.PLATE)

    @staticmethod
    def load_label(case):
        """Label of the load axis of a case."""
        return f"Load level [{'N/mm' if case['kind'] == 'n' else 'Nmm/mm'}]"

    def load_levels(self, case, gkey):
        """Load levels of a (case, section) pair: explicit levels of the config, else sign * capacity * fractions."""
        cfg = self.cfg
        key = (case['id'], gkey)
        if key in cfg.load_levels:
            levels = list(cfg.load_levels[key])
        else:
            # 3 significant digits: load-level folders are read back as '%g' % level
            levels = sorted({float(f'{case["sign"] * cfg.capacity[key] * f:.3g}') for f in cfg.fractions}, key=abs)
        if max(abs(v) for v in levels) >= 1e6:
            raise ValueError(f'{self.run_id(case, gkey)}: load levels >= 1e6 are not read back correctly '
                             '(%g folder names)')
        return levels

    def solver_settings(self, case, gkey):
        """Solver settings of a (case, section) pair: the settings of the config with the per-case res_tol."""
        return solver_settings(self.cfg, res_tol=self.cfg.res_tol_case.get((case['id'], gkey)))

    def run_folder(self, case, gkey, data_out_root):
        """New run folder <data_out_root>/data_<timestamp>_<case>_<section>_v<model> (created, see run_folder)."""
        return run_folder(data_out_root, self.run_id(case, gkey), self.cfg.model_version)

    def find_latest_run_folder(self, data_out_root, case, gkey):
        """Most recent run folder of a (case, section) pair, or None (see find_latest_run_folder)."""
        return find_latest_run_folder(data_out_root, self.run_id(case, gkey), self.cfg.model_version)

    ################################## reading ##################################

    def read_run(self, folder, t):
        """
        Signed eps / chi at the plate-centre element and u at the plate-centre node (homogeneous field), per load
        level, for NLFEA and NN, plus the convergence flag per level.

        Converged: all cases apply uniform edge loads to a statically determinate plate, so the stress resultants at
        the centre element are affine in the load level, s = a + b * load (a = fixed loads, e.g. the transverse
        compression of 9 / 109). a, b are taken from the first NLFEA level (lowest load, always converged) and the
        other NLFEA level that the most levels agree with. A level is converged if max|s - s_applied| <= m_tol *
        max|s_applied|, compared as extreme-fibre stresses n / t and 6 m / t^2 (the solver residual of n_iterations /
        residual_summary is not a usable criterion here). n_ok = number of leading levels converged in both.

        Args:
            folder  (str):      run folder
            t       (float):    section thickness [mm]

        Returns:
            dict: load, NLFEA / NN {eps, sig, u, ok}, n_ok, el, node
        """
        cfg = self.cfg
        steps = get_loadsteps(folder)
        if len(steps) < 2:
            raise ValueError(f'{folder}: fewer than 2 load levels')
        out = {'load': steps}
        centre = (cfg.PLATE['L'] / 2, cfg.PLATE['B'] / 2, 0.0)
        el = node = None
        for tag, key in (('norm', 'NLFEA'), ('NN', 'NN')):
            eps, sig, u = [], [], []
            for s in steps:
                res = load_mat_res(folder, s, tag)
                if el is None:                     # first NLFEA level: plate-centre node and element
                    node, el = nearest_node_element(res, centre)
                eps.append(eps_fields(res)[el])
                sig.append(sig_fields(res)[el])
                u.append(displ_fields(res)[node, :3])
            out[key] = {'eps': np.array(eps), 'sig': np.array(sig), 'u': np.array(u)}

        scale = np.r_[np.full(3, 1 / t), np.full(3, 6 / t ** 2)]           # n / t, 6 m / t^2
        S = out['NLFEA']['sig'] * scale

        def ok_flags(sig, s_app):
            return np.abs(sig - s_app).max(axis=1) / np.abs(s_app).max(axis=1) <= cfg.m_tol

        best_n, s_app = -1, None
        for j in range(1, len(steps)):
            b = (S[j] - S[0]) / (steps[j] - steps[0])
            cand = (S[0] - b * steps[0])[None, :] + steps[:, None] * b[None, :]
            n_fit = int(ok_flags(S, cand).sum())
            if n_fit > best_n:                     # ties: keep the earlier level
                best_n, s_app = n_fit, cand
        for key in ('NLFEA', 'NN'):
            out[key]['ok'] = ok_flags(out[key]['sig'] * scale, s_app)
        both = out['NLFEA']['ok'] & out['NN']['ok']
        out['n_ok'] = len(steps) if both.all() else int(np.argmin(both))
        out['el'], out['node'] = el, node
        return out

    @staticmethod
    def metrics(res, case):
        """RMSE(NN, NLFEA) and max |NLFEA| per component over the leading levels converged in both, in table units."""
        def table_units(d):                        # eps [‰], chi [mrad/m] (eps_g holds 2 chi_xy -> halved), u [mm]
            return np.c_[d['eps'][:, :3] * 1e3, d['eps'][:, 3:6] * 1e6 * np.array([1, 1, 0.5]), d['u']]

        n = res['n_ok']
        nl, nn = table_units(res['NLFEA'])[:n], table_units(res['NN'])[:n]
        nan9 = np.full(9, np.nan)
        return {'n_levels': len(res['load']), 'n_ok_nlfea': int(res['NLFEA']['ok'].sum()),
                'n_ok_nn': int(res['NN']['ok'].sum()), 'n_matched': n, 'el': res['el'], 'node': res['node'],
                'max_load': res['load'][n - 1] * LOAD_SCALE[case['kind']] if n else np.nan,
                'rmse': np.sqrt(np.mean((nn - nl) ** 2, axis=0)) if n else nan9,
                'max': np.abs(nl).max(axis=0) if n else nan9}

    def table_record(self, res, case, gkey, folder):
        """
        Table entry of a (case, section) pair from the results at the plate centre: applies the cut-off cfg.table_max
        (only levels with |load| <= table_max count), computes the table values (metrics), adds the run folder and
        prints a summary line.

        Args:
            res     (dict): results of read_run
            case    (dict): case
            gkey    (str):  section key
            folder  (str):  run folder

        Returns:
            dict: metrics + 'folder'
        """
        l_max = self.cfg.table_max.get((case['id'], gkey))
        if l_max is not None:
            res['n_ok'] = min(res['n_ok'], int((np.abs(res['load']) <= l_max).sum()))
        rec = self.metrics(res, case)
        rec['folder'] = folder
        print(f"[{self.run_id(case, gkey)}] converged levels NLFEA {rec['n_ok_nlfea']}/{rec['n_levels']}, "
              f"NN {rec['n_ok_nn']}/{rec['n_levels']}, used {rec['n_matched']}, "
              f"max load {rec['max_load']:g} {LOAD_UNIT[case['kind']]}  ({folder})")
        return rec

    ################################## writing ##################################

    def geom_label(self, gkey, sep='\n'):
        """Label and section vector of a section key, e.g. 'isotropic [300, 0.025, ...]'."""
        g = self.cfg.GEOMS[gkey]
        return g['label'] + sep + '[' + ', '.join(f'{v:g}' for v in g['geom']) + ']'

    def write_table(self, records, cases, gkeys, xlsx_path):
        """
        Excel table (sheet 'Table': layout of the paper, sheet 'Data': one row per run) and the same data as CSV.

        Args:
            records     (dict): {(case id, section key): metrics + 'folder', or None}
            cases       (list): cases in the table
            gkeys       (list): section keys in the table
            xlsx_path   (str):  path of the Excel file (the CSV is written next to it)

        Returns:
            str: path of the CSV file
        """
        # imported here: only the table needs openpyxl
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, Side
        from openpyxl.cell.rich_text import CellRichText, TextBlock
        from openpyxl.cell.text import InlineFont
        from openpyxl.utils import get_column_letter

        GEOMS = self.cfg.GEOMS
        wb = Workbook()
        ws = wb.active
        ws.title = 'Table'
        font = Font(name='Times New Roman', size=10)
        thin, none = Side(style='thin'), Side()
        center = Alignment(horizontal='center', vertical='center', wrap_text=True)
        nG = len(gkeys)

        def put(r, c, v, f=font, fmt=None, align=center, r2=None, c2=None):
            if r2 is not None or c2 is not None:
                ws.merge_cells(start_row=r, start_column=c, end_row=r2 or r, end_column=c2 or c)
            cell = ws.cell(r, c, v)
            cell.font, cell.alignment = f, align
            if fmt:
                cell.number_format = fmt
            return cell

        def rich(*parts):
            # parts: (text, style), style '' / 'i' (italic) / 'sub' (subscript)
            return CellRichText(*[TextBlock(InlineFont(rFont='Times New Roman', sz=10, i=s == 'i',
                                                       vertAlign='subscript' if s == 'sub' else None), t)
                                  for t, s in parts])

        def num(v):
            return '–' if v is None or not np.isfinite(v) else float(v)

        # header (rows 1-3)
        put(1, 1, 'Case Study No.', r2=3)
        c_load = 2
        put(1, c_load, 'Max. Load NN =\nMax. Load NLFEA\n[kN/m] or [kNm/m]', c2=c_load + nG - 1)
        nm_label = rich(*[p for d in ('n', 'm') for p in
                          ((d, 'i'), ('x', 'sub'), (' = ', ''), (d, 'i'), ('y', 'sub'), (' = ', ''), (d, 'i'),
                           ('xy', 'sub'), ('\n' if d == 'n' else '', ''))])
        for i, g in enumerate(gkeys):
            put(2, c_load + i, GEOMS[g]['label'].capitalize())
            put(3, c_load + i, nm_label)
        c_block, offs = [], []
        c, off = c_load + nG, 0
        for title, unit, comps in BLOCKS:
            c_block.append(c)
            offs.append(off)
            w = len(comps)
            put(1, c, rich(('RMSE', 'i'), (f' {title}\n[{unit}]', '')), c2=c + w * nG - 1)
            for i, g in enumerate(gkeys):
                put(2, c + w * i, GEOMS[g]['label'].capitalize(), c2=c + w * i + w - 1)
                for j, (sym, sub) in enumerate(comps):
                    put(3, c + w * i + j, rich((sym, 'i' if sym == 'u' else ''), (sub, 'sub')))
            c += w * nG
            off += w
        n_cols = c - 1

        # data: two rows per case (RMSE, max reached), a thin empty row between the case groups
        r = 4
        top_rows, sep_rows = set(), set()
        for group in self.group_order:
            gcases = [cs for cs in cases if cs['group'] == group]
            if not gcases:
                continue
            if r > 4:
                sep_rows.add(r)
                ws.row_dimensions[r].height = 6
                r += 1
            for case in gcases:
                top_rows.add(r)
                put(r, 1, case['id'], r2=r + 1)
                for i, g in enumerate(gkeys):
                    rec = records.get((case['id'], g))
                    put(r, c_load + i, num(rec['max_load']) if rec else '–', fmt='0', r2=r + 1)
                    for (_, _, comps), c0, off in zip(BLOCKS, c_block, offs):
                        w = len(comps)
                        vals_rmse = rec['rmse'][off:off + w] if rec else [None] * w
                        vals_max = rec['max'][off:off + w] if rec else [None] * w
                        for j in range(w):
                            put(r, c0 + w * i + j, num(vals_rmse[j]), fmt='0.00')
                            put(r + 1, c0 + w * i + j, num(vals_max[j]), fmt='0.00')
                r += 2

        # borders: thin grid, no line between the RMSE and max row of a case, separator row without inner lines
        for row in ws.iter_rows(min_row=1, max_row=r - 1, min_col=1, max_col=n_cols):
            for cell in row:
                if cell.row in sep_rows:
                    cell.border = Border(top=thin, bottom=thin)
                else:
                    cell.border = Border(left=thin, right=thin,
                                         top=none if cell.row - 1 in top_rows else thin,
                                         bottom=none if cell.row in top_rows else thin)

        for c in range(1, n_cols + 1):
            ws.column_dimensions[get_column_letter(c)].width = 6.5
        ws.column_dimensions['A'].width = 10
        for i in range(nG):
            ws.column_dimensions[get_column_letter(c_load + i)].width = 14
        ws.row_dimensions[1].height = 42
        ws.row_dimensions[3].height = 28
        ws.freeze_panes = ws.cell(4, c_load)

        # long format: one row per (case, section), also written as CSV
        header = ['case', 'group', 'description', 'scenario', 'geom', 'geom_vector', 'folder', 'load_unit',
                  'max_load', 'n_levels', 'n_ok_NLFEA', 'n_ok_NN', 'n_matched', 'centre_el'] + \
                 [f'rmse_{k}' for k in COMP_KEYS] + [f'max_{k}' for k in COMP_KEYS]
        rows = []
        for case in cases:
            for g in gkeys:
                rec = records.get((case['id'], g))
                base = [case['id'], case['group'], case['desc'], case['scenario'], g, self.geom_label(g, sep=' ')]
                if rec is None:
                    rows.append(base + [None] * (len(header) - len(base)))
                else:
                    rows.append(base + [rec['folder'], LOAD_UNIT[case['kind']], rec['max_load'], rec['n_levels'],
                                        rec['n_ok_nlfea'], rec['n_ok_nn'], rec['n_matched'], rec['el']]
                                + [float(v) for v in rec['rmse']] + [float(v) for v in rec['max']])
        wd = wb.create_sheet('Data')
        wd.append(header)
        for row in rows:
            wd.append([None if isinstance(v, float) and not np.isfinite(v) else v for v in row])
        wb.save(xlsx_path)

        csv_path = os.path.splitext(xlsx_path)[0] + '.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        return csv_path
