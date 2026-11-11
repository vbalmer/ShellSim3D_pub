"""Figures of a load path (NN vs NLFEA): load-deformation curves at a tracked node / element and the residual per
load level. V. Balmer, 2026."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .load_path_utils import (get_loadsteps, load_mat_res, displ_fields, eps_fields, n_iterations, level_folder,
                              get_mat_res_path)

U_LABELS = ['u_x', 'u_y', 'u_z', 'th_x', 'th_y', 'th_z']
EPS_LABELS = ['eps_x', 'eps_y', 'gamma_xy', 'chi_x', 'chi_y', '2*chi_xy']


def _nn_available(folder, load_steps):
    try:
        get_mat_res_path(level_folder(folder, load_steps[0]), 'NN')
        return True
    except FileNotFoundError:
        return False


def plot_load_path(folder: str, fig_dir: str, track: dict, n_levels=None, ylabel: str = 'Load level', tag: str = ''):
    """
    All figures of a load path (NN vs NLFEA): load-deformation curves at the tracked node / element
    (Load-Deformation_2x2_<tag>.png) and residual per load level (residual_loadlevel_<tag>.png).

    Args:
        folder      (str):  load-path folder
        fig_dir     (str):  folder of the figures
        track       (dict): node, el (indices of the tracked node / element)
        n_levels    (int):  load-deformation curves of the first n_levels load levels; None: all but the last one
        ylabel      (str):  label of the load axis
        tag         (str):  suffix of the file names
    """
    suffix = '_' + tag if tag else ''
    plot_load_path_2x2(folder, fig_dir, track, n_levels=n_levels, ylabel=ylabel,
                       filename=f'Load-Deformation_2x2{suffix}.png')
    residual_loadlevel_plot(folder, fig_dir, tag=tag)
    plt.close('all')


def tracked_values(folder: str, track: dict, tag: str) -> dict:
    """
    |u| at the tracked node and |eps| at the tracked element for every load level.

    Args:
        folder  (str):  load-path folder
        track   (dict): node, el (indices)
        tag     (str):  'NN' or 'norm' (NLFEA)

    Returns:
        dict: 'u' (n_levels, 6) [mm, rad], 'eps' (n_levels, 6) [permille, mrad/mm], 'n_iter' (n_levels,)
    """
    load_steps = get_loadsteps(folder)
    u, eps, n_it = [], [], []
    for level in load_steps:
        res = load_mat_res(folder, level, tag)
        u.append(np.abs(displ_fields(res)[track['node']]))
        eps.append(np.abs(eps_fields(res)[track['el']]) * 1e3)
        n_it.append(n_iterations(res))
    return {'u': np.array(u), 'eps': np.array(eps), 'n_iter': np.array(n_it)}


def plot_load_path_2x2(folder: str, save_path: str, track: dict, n_levels=None, ylabel: str = 'Load level',
                       filename: str = 'Load-Deformation_2x2.png'):
    """
    Load-deformation curves NN vs NLFEA at one node (displacements | rotations, top row) and one element (strains |
    curvatures, middle row), and the mesh with the two points (bottom row).

    Args:
        folder      (str):  load-path folder
        save_path   (str):  folder of the figure
        track       (dict): node, el (indices of the tracked node / element)
        n_levels    (int):  plot the first n_levels load levels; None: all but the last one
        ylabel      (str):  label of the load axis
        filename    (str):  file name of the figure
    """
    load_steps = get_loadsteps(folder)
    n = len(load_steps) - 1 if n_levels is None else n_levels
    nn_present = _nn_available(folder, load_steps)
    vals = {'NLFEA': tracked_values(folder, track, 'norm')}
    if nn_present:
        vals['NN'] = tracked_values(folder, track, 'NN')

    fig, ax = plt.subplots(3, 2, figsize=[16, 15])
    loads = np.abs(load_steps[:n])
    for row, (key, labels, xl) in enumerate([('u', U_LABELS, ('Displacements [mm]', 'Rotations [rad]')),
                                             ('eps', EPS_LABELS, ('Strains [‰]', 'Curvatures [mrad/mm]'))]):
        name = f" @node {track['node']}" if key == 'u' else f" @el {track['el']}"
        for col, dims in enumerate([[0, 1, 2], [3, 4, 5]]):
            cmap = [plt.cm.Blues, plt.cm.RdPu][col]
            for k, j in enumerate(dims):
                color = cmap(np.linspace(0.4, 0.85, 3)[k])
                ax[row, col].plot(vals['NLFEA'][key][:n, j], loads, color=color, marker='o',
                                  label='NLFEA, ' + labels[j] + name)
                if nn_present:
                    ax[row, col].plot(vals['NN'][key][:n, j], loads, color=color, linestyle='--', marker='x',
                                      label='NN, ' + labels[j] + name)
            ax[row, col].set_xlabel(xl[col])
            ax[row, col].legend()
        ax[row, 0].set_ylabel(ylabel)

    _plot_point_mesh(fig, ax[2], load_mat_res(folder, load_steps[0], 'norm'), track)
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename))
    plt.close(fig)
    print(f'Saved figure {filename} at {save_path}')


def _plot_point_mesh(fig, ax_row, mat_res, track):
    """Undeformed mesh (3D if the structure is not flat) with the tracked node / element, and their coordinates."""
    nodes = np.asarray(mat_res['NODESG'])
    elements = np.asarray(mat_res['ELEMENTS']).astype(int)
    centroids = np.asarray(mat_res['COORD_c'][0])
    flat = np.ptp(nodes[:, 2]) < 1e-6 * max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1.0)

    ax_mesh = ax_row[0]
    if not flat:
        spec = ax_mesh.get_subplotspec()
        ax_mesh.remove()
        ax_mesh = fig.add_subplot(spec, projection='3d')

    lines = np.vstack([np.vstack([nodes[e[e < 10**5]], nodes[e[0]], [np.nan] * 3]) for e in elements])
    pts = [('node', track['node'], nodes[track['node']], 'o'), ('el', track['el'], centroids[track['el']], 's')]
    for name, i, xyz, marker in pts:
        if flat:
            ax_mesh.scatter(xyz[0], xyz[1], color='red', marker=marker, s=40, zorder=3)
            ax_mesh.annotate(f'{name} {i}', (xyz[0], xyz[1]), xytext=(4, 4), textcoords='offset points',
                             color='red', fontsize=7)
        else:
            ax_mesh.scatter(xyz[0], xyz[1], xyz[2], color='red', marker=marker, s=40, depthshade=False)
            ax_mesh.text(xyz[0], xyz[1], xyz[2], f' {name} {i}', color='red', fontsize=7)
    if flat:
        ax_mesh.plot(lines[:, 0], lines[:, 1], color='grey', linewidth=0.5, zorder=1)
        ax_mesh.set_aspect('equal', 'box')
    else:
        ax_mesh.plot(lines[:, 0], lines[:, 1], lines[:, 2], color='grey', linewidth=0.5)
        ax_mesh.set_box_aspect([np.ptp(nodes[:, 0]) or 1.0, np.ptp(nodes[:, 1]) or 1.0, np.ptp(nodes[:, 2]) or 1.0])
        ax_mesh.set_zlabel('z [mm]')
    ax_mesh.set_xlabel('x [mm]')
    ax_mesh.set_ylabel('y [mm]')
    ax_mesh.set_title('Tracked points (o = node for u / th, square = element for eps / chi)', fontsize=9)

    ax_list = ax_row[1]
    ax_list.axis('off')
    text = [f'{name:>5} {i:5d} ({xyz[0]:6.0f},{xyz[1]:6.0f},{xyz[2]:6.0f})' for name, i, xyz, _ in pts]
    ax_list.text(0, 1, '\n'.join(text), family='monospace', va='top', fontsize=8, transform=ax_list.transAxes)
    ax_list.set_title('Tracked points [x, y, z in mm]', fontsize=9)


def residual_loadlevel_plot(folder: str, save_path: str, tag: str = ''):
    """
    Residual forces and moments of the last iteration per load level, NLFEA (solid) and NN (dashed). Top: absolute,
    bottom: relative to the applied forces / moments (fallback if none applied: relative to the maximum residual of
    the load level). Grey bars: number of iterations per load level (light: NLFEA, dark: NN).
    Saved as residual_loadlevel_<tag>.png.

    Args:
        folder      (str):  load-path folder
        save_path   (str):  folder of the figure
        tag         (str):  suffix of the file name
    """
    load_levels = get_loadsteps(folder).astype(float)
    have_nn = _nn_available(folder, load_levels)

    def collect(tag_res):
        rm_last, rf_last, rm_ratio, rf_ratio, n_iter = [], [], [], [], []
        for level in load_levels:
            res = load_mat_res(folder, level, tag_res)
            convrm = np.abs(np.atleast_1d(res['convrm']))
            convrf = np.abs(np.atleast_1d(res['convrf']))
            rm_last.append(convrm[-1])
            rf_last.append(convrf[-1])
            fe = np.abs(np.asarray(res['fe']).flatten())
            fa_sum = fe[0::6].sum() + fe[1::6].sum() + fe[2::6].sum()     # applied forces
            ma_sum = fe[3::6].sum() + fe[4::6].sum() + fe[5::6].sum()     # applied moments
            rf_ratio.append(convrf[-1] / fa_sum if fa_sum > 0 else (convrf[-1] / np.max(convrf) if np.max(convrf) > 0 else 0.0))
            rm_ratio.append(convrm[-1] / ma_sum if ma_sum > 0 else (convrm[-1] / np.max(convrm) if np.max(convrm) > 0 else 0.0))
            n_iter.append(len(convrm))
        return [np.array(v) for v in (rm_last, rf_last, rm_ratio, rf_ratio, n_iter)]

    data = {'NLFEA': collect('norm')}
    if have_nn:
        data['NN'] = collect('NN')

    def iteration_bars(ax):
        ax_bg = ax.twinx()
        ax_bg.set_zorder(ax.get_zorder() - 1)
        ax_bg.set_yticks([])
        for spine in ax_bg.spines.values():
            spine.set_visible(False)
        w = 0.35 * (np.min(np.diff(np.sort(load_levels))) if len(load_levels) > 1 else 1.0)
        ax_bg.set_ylim(0, max(max(np.max(d[4]) for d in data.values()), 1) / 0.3)
        for (key, d), shade, off in zip(data.items(), ('0.8', '0.6'), (-w / 2 if have_nn else 0.0, w / 2)):
            for x, c in zip(load_levels, d[4]):
                ax_bg.bar(x + off, c, width=w, color=shade, zorder=0)
                ax_bg.text(x + off, c, str(int(c)), ha='center', va='bottom', fontsize=7, color='0.4')

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(os.path.basename(os.path.normpath(folder)))
    for a in (ax_top, ax_bot):
        iteration_bars(a)
        a.patch.set_visible(False)
    bar_handles = [Patch(facecolor='0.8', label='Iterations (NLFEA)')] + \
                  ([Patch(facecolor='0.6', label='Iterations (NN)')] if have_nn else [])

    for ax, (i_m, i_f), title, ylabels in [
            (ax_top, (0, 1), 'Residual of the last iteration step per load level',
             ('$\\Sigma$ residual moments [Nmm]', '$\\Sigma$ residual forces [N]')),
            (ax_bot, (2, 3), 'Residual of the last iteration step relative to the applied load per load level',
             ('Residual moments:\n last it. / applied moments [-]', 'Residual forces:\n last it. / applied forces [-]'))]:
        ax_f = ax.twinx()
        handles = []
        for key, d in data.items():
            ls = '-' if key == 'NLFEA' else '--'
            handles.append(ax.plot(load_levels, d[i_m], color='tab:blue', linestyle=ls, marker='o',
                                   label=f'Residual moments ({key})')[0])
            handles.append(ax_f.plot(load_levels, d[i_f], color='tab:red', linestyle=ls, marker='s',
                                     label=f'Residual forces ({key})')[0])
        ax.set_yscale('log')
        ax_f.set_yscale('log')
        ax.set_ylabel(ylabels[0], color='tab:blue')
        ax_f.set_ylabel(ylabels[1], color='tab:red')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax_f.tick_params(axis='y', labelcolor='tab:red')
        ax.set_title(title)
        ax.grid(True)
        ax.legend(handles=handles + bar_handles, loc='best')
    ax_bot.set_xlabel('Load level')
    if len(load_levels) > 0 and np.all(load_levels < 0):
        ax_bot.invert_xaxis()

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    figname = 'residual_loadlevel' + ('_' + tag if tag else '') + '.png'
    fig.savefig(os.path.join(save_path, figname), dpi=200)
    plt.close(fig)
    print('saved residual-load-level plot to', os.path.join(save_path, figname))
