"""Diagnostic plots written by the sampler (stiffness along one strain axis, distributions of the drawn
geometry / material values). V. Balmer, 2026.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .sampler_utils_RC3D import find_outlier_d


########################################## Stiffness along one strain axis ##########################################

def get_mask_strain(data_eps, idx_eps, tol = (0.5e-3, 0.5e-3, 1e-5, 5e-7, 5e-7, 5e-7)):
    """
    Rows where all strains except idx_eps are ~0 (|eps_k| < tol[k]).

    Args:
        data_eps (np.arr):     strains (n, 6).
        idx_eps  (int | list): strain index (line, e.g. 0 = pure eps_x) or indices (plane, e.g. [0, 1]) allowed to vary.
        tol      (tuple):      per-dimension tolerance.

    Returns:
        data_f_eps (np.arr), mask (np.arr): selected rows and the boolean mask.
    """
    tol = np.array(tol)
    idx_eps = np.atleast_1d(idx_eps)
    cols = ~np.isin(np.arange(data_eps.shape[1]), idx_eps)
    mask = np.all(np.abs(data_eps[:,cols])<tol[cols], axis =1)

    print(f'After filtering data for plotting D, {mask.sum()} datapoints are left.')
    if mask.sum() < 1:
        raise UserWarning('No datapoints found in given range. Please change the filtering tolerance.')

    return data_eps[mask], mask


def sort_data(data_f_eps, data_f_D, idx_eps):
    """Sort strains and stiffnesses in ascending order of strain idx_eps."""
    order = np.argsort(data_f_eps[:, idx_eps])
    return data_f_eps[order], data_f_D[order]


def deduplicate_by_eps(data_s_eps, data_s_D, idx_eps, decimals=6):
    """Keep the first row per (rounded) value of strain idx_eps."""
    x = np.round(data_s_eps[:, idx_eps], decimals=decimals)
    unique_eps, unique_D = [], []
    for val in np.unique(x):
        group_mask = x == val
        unique_eps.append(data_s_eps[group_mask][0])
        unique_D.append(data_s_D[group_mask][0])
    return np.array(unique_eps), np.array(unique_D)


def plot_filtered_stiffness(data_eps, data_D, idx_eps, save_path, remove_outliers: bool = False):
    """
    Plot all 36 stiffness entries along strain idx_eps for the samples where all other strains are ~0
    (filtered_dataset_D.png). If remove_outliers is False, IQR outliers are highlighted.

    Args:
        data_eps        (np.arr): strains (n, 6).
        data_D          (np.arr): stiffness (n, 6, 6).
        idx_eps         (int):    strain index on the x axis.
        save_path       (str):    output directory (None -> not saved).
        remove_outliers (bool):   whether outliers were already removed from the data.
    """
    data_f_eps, mask = get_mask_strain(data_eps, idx_eps)
    data_s_eps, data_s_D = sort_data(data_f_eps, data_D[mask], idx_eps)
    data_d_eps, data_d_D = deduplicate_by_eps(data_s_eps, data_s_D, idx_eps)

    if not remove_outliers:
        data_d_D_outlier, data_d_eps_outlier, _ = find_outlier_d(data_d_D, data_d_eps, 1000)
    else:
        data_d_D_outlier, data_d_eps_outlier = None, None

    plot_data_stiffness(data_d_eps, data_d_D, data_d_D_outlier, data_d_eps_outlier, idx_eps, save_path)


def plot_data_stiffness(data_s_eps, data_s_D, data_d_D_outlier, data_d_eps_outlier, idx_eps, save_path):
    """6x6 grid of D_ij over strain idx_eps (outliers in coral)."""
    fig, axs = plt.subplots(6,6, figsize = [30,20])

    for i in range(6):
        for j in range(6):
            axs[i,j].plot(data_s_eps[:,idx_eps], data_s_D[:,i,j], marker = 'o')
            if data_d_eps_outlier is not None:
                axs[i,j].plot(data_d_eps_outlier[:,idx_eps], data_d_D_outlier[:,i,j], marker = 'o', color = 'coral')

    figure_formatting_D(axs, idx_eps)

    if save_path is not None:
        filename = 'filtered_dataset_D.png'
        fig.savefig(os.path.join(save_path, filename))
        print(f'Saved {filename} to {save_path}')
    plt.close(fig)


def figure_formatting_D(axs, idx_eps):
    """Axis labels of the 6x6 stiffness grid."""
    names_D = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$'],
                        ['$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$'],
                        ['$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$'],
                        ['$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$'],
                        ['$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$'],
                        ['$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$'],
                        ])
    names_eps = np.array(['$\\varepsilon_x$', '$\\varepsilon_y$', '$\\gamma_{xy}$', '$\\chi_x$', '$\\chi_y$', '$\\chi_{xy}$'])

    for i in range(6):
        for j in range(6):
            axs[i,j].set_ylabel(names_D[i,j])
            axs[i,j].set_xlabel(names_eps[idx_eps])


def imshow_D_filtered(data_eps, data_D, idx_eps, save_path):
    """Heat map of the 36 stiffness entries (columns) of the samples along strain idx_eps (rows, sorted)."""
    data_f_eps, mask = get_mask_strain(data_eps, idx_eps)
    _, data_s_D = sort_data(data_f_eps, data_D[mask], idx_eps)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_s_D.reshape((-1,36)), aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    if save_path is not None:
        filename = 'matrix_imshow.png'
        fig.savefig(os.path.join(save_path, filename))
        print(f'Saved {filename} to {save_path}')
    plt.close(fig)


########################################## Distributions ##########################################

def plot_value_distribution(values, out_dir, filename='distribution.png', title='distribution', xlabel='value',
                            sampling_type=None, sampling_type_names=None):
    """Bar chart of a discrete-valued column's distribution (counts + % per unique value).

    Used to check that the post-filter distribution of a drawn parameter (t, rho, fc, fs) is roughly uniform.

    Args:
        values              (np.arr): 1D array of values (one per sample).
        out_dir             (str):    directory to save the png into (created if missing).
        filename            (str):    output file name.
        title               (str):    plot title.
        xlabel              (str):    x-axis label.
        sampling_type       (np.arr): optional per-value subset index (same length as values); if given, each
                                      printed row also shows the log / uniform split of that value.
        sampling_type_names (list):   subset index -> name (e.g. ['log', 'uniform', ...]).

    Returns:
        str: full path of the saved figure.
    """
    values = np.asarray(values)
    vals, cnt = np.unique(values, return_counts=True)
    pct = 100.0 * cnt / cnt.sum()
    print(f'=== {xlabel} distribution ({cnt.sum()/1e6:.2f}M total) ===')
    for v, c, p in zip(vals, cnt, pct):
        line = f'{v:10.4g}  n={c:>10}  {p:5.1f}%'
        if sampling_type is not None:
            m = values == v
            for name in ('log', 'uniform'):
                if name in sampling_type_names:
                    code = sampling_type_names.index(name)
                    line += f'  {name} {100.0 * np.sum(sampling_type[m] == code) / c:5.1f}%'
        print(line)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([f'{v:g}' for v in vals], cnt, color='steelblue', alpha=0.85)
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{p:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel('count'); ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'saved {path}')
    return path


def plot_geom_distributions(eps_and_geom, geom_names, save_path, sampling_type=None, sampling_type_names=None):
    """Distribution bar charts of every drawn geometry / material column (t, rho_*, fc, fs) of the final dataset.

    Args:
        eps_and_geom (np.arr): (n, 6 + n_geom) strains + geom columns.
        geom_names   (list):   names of the geom columns (geom_column_names).
        save_path    (str):    output directory.
        sampling_type, sampling_type_names: see plot_value_distribution.
    """
    labels = {'t': 't [mm]', 'fc': 'fc [MPa]', 'fs': 'fs [MPa]'}
    for i, name in enumerate(geom_names):
        plot_value_distribution(eps_and_geom[:, 6 + i], save_path,
                                filename=f'{name}_distribution.png', xlabel=labels.get(name, name),
                                title=f'surviving {name} distribution (post-filter)',
                                sampling_type=sampling_type, sampling_type_names=sampling_type_names)
