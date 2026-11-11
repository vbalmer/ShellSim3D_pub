"""Data handling for training: reading the sampled HDF5 files, train/eval/test split, normalisation
(optional t- and fc-normalisation, then standardisation) and conversion to torch. V. Balmer, 2026.
"""

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch


SEED = 42
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')   # training/plots

################################## Reading data ##################################


def read_h5_file(save_data_path, filename, n_every: int) -> np.ndarray:
    """
    Read dataset `filename` from save_data_path/output_<filename>.h5.

    Args:
        save_data_path  (str):  folder containing the HDF5 files
        filename        (str):  dataset name, e.g. 'eps_g', 'sig_g', 'D' or 'geom'
        n_every         (int):  read only every n_every-th row

    Returns:
        data          (np.arr): stored array (rows subsampled by n_every)
    """
    path = os.path.join(save_data_path, 'output_' + filename + '.h5')
    with h5py.File(path, 'r') as f:
        data = f[filename][::n_every, :]
    return data


def get_data(path_data: str) -> tuple:
    """
    Read the sampled strains, stresses and stiffnesses (output_eps_g.h5, output_sig_g.h5, output_D.h5).

    Args:
        path_data       (str):  folder containing the sampled data

    Returns:
        features      (np.arr): generalised strains, shape (ntot, 6)
        labels_concat (np.arr): generalised stresses and flattened stiffnesses, shape (ntot, 6+36)
    """

    features = read_h5_file(path_data, 'eps_g', n_every=1)
    labels = read_h5_file(path_data, 'sig_g', n_every=1)
    labels_diff = read_h5_file(path_data, 'D', n_every=1)
    labels_concat = np.concatenate((labels, labels_diff.reshape((-1, labels.shape[1]**2))), axis=1)

    print(f'Total amount of datapoints: {features.shape[0]/1e6:.2f}*1e6.')

    return features, labels_concat


def add_geom_data(path_data: str, features: np.ndarray, GEOM_SIZE: int) -> np.ndarray:
    """
    Append the per-sample geometry / material columns (output_geom.h5) to the strains.

    Args:
        path_data   (str)   : folder containing output_geom.h5
        features    (np.arr): strains, shape (ntot, 6)
        GEOM_SIZE   (int)   : number of geometry columns expected by the model (input_size - out_size)

    Returns:
        features    (np.arr): shape (ntot, 6+GEOM_SIZE)
    """
    if GEOM_SIZE == 0:
        return features

    geom = read_h5_file(path_data, 'geom', n_every=1)
    if geom.shape[1] != GEOM_SIZE:
        raise ValueError(f'geom data in {path_data} has {geom.shape[1]} columns but GEOM_SIZE = {GEOM_SIZE} '
                         f'(input_size - out_size). Set input_size to 6 + {geom.shape[1]} '
                         f'(e.g. 13 for [t, rho x4, fc, fs]).')
    features = np.concatenate((features, geom), axis=1)
    print(f'Added n = {GEOM_SIZE} geometrical input parameters to the dataset')

    return features


################################## Split data ##################################

def split_data(features, labels, test_size=0.1, eval_size=0.2):
    """
    Split the data into train, eval and test sets (fractions of the full dataset, fixed seed).

    Args:
        features    (np.arr): inputs, shape (ntot, 6+GEOM_SIZE)
        labels      (np.arr): labels, shape (ntot, 6+36)
        test_size   (float):  fraction of the full dataset used for testing
        eval_size   (float):  fraction of the full dataset used for evaluation (validation)

    Returns:
        train_eval_test_data (dict): X_train, X_eval, X_test, y_train, y_eval, y_test
    """

    test_size_2 = eval_size / (1 - test_size)

    X_aux, X_test, y_aux, y_test = train_test_split(features, labels, test_size=test_size, random_state=SEED)
    X_train, X_eval, y_train, y_eval = train_test_split(X_aux, y_aux, test_size=test_size_2, random_state=SEED)

    train_eval_test_data = {
        'X_train': X_train,
        'X_eval': X_eval,
        'X_test': X_test,
        'y_train': y_train,
        'y_eval': y_eval,
        'y_test': y_test,
    }

    return train_eval_test_data


def plot_split_data(data: dict, plot: bool):
    """
    3D scatter plots of the strains and stresses of the train, eval and test sets.

    Args:
        data        (dict): split dataset (X_*, y_*), physical units
        plot        (bool): create the plots (False -> no-op)

    Returns:
        None; figures saved to training/plots
    """
    if plot:
        plot_3D_data(data['X_train'][:, :6], 'train_scatter_eps_g')
        plot_3D_data(data['X_eval'][:, :6], 'eval_scatter_eps_g')
        plot_3D_data(data['X_test'][:, :6], 'test_scatter_eps_g')
        plot_3D_data(data['y_train'][:, :6], 'train_scatter_sig_g')
        plot_3D_data(data['y_eval'][:, :6], 'eval_scatter_sig_g')
        plot_3D_data(data['y_test'][:, :6], 'test_scatter_sig_g')
    return


################################## Normalisation ##################################

def get_stats(data, t_norm=None, fc_norm=None):
    """
    Statistics (mean, std, ...) of all subsets (train, eval, test). Only the train statistics are used for the
    normalisation.

    Args:
        data        (dict): split dataset (X_*, y_*).
        t_norm      (dict): if given ({'powers', 'input_powers', 't_idx'}), the y statistics are computed on the
                            t-divided outputs y / t^(a_j[+b_k]) and the X statistics on the t-scaled strains
                            (scale_inputs), i.e. on what the normalisation later sees. None -> plain stats.
        fc_norm     (dict): if given ({'fc_idx'}), the y statistics are additionally computed on y / fc.

    Returns:
        stats       (dict): {'stats_X_train': {...}, 'stats_y_train': {...}, ...}
    """

    stats = {}
    for key in data.keys():
        d = data[key]
        x_key = str(key).replace('y', 'X', 1)
        if t_norm is not None:
            if str(key).startswith('y'):                             # divide outputs by t^(a_j[+b_k])
                if x_key in data:
                    d = d / _t_factor(data[x_key][:, t_norm['t_idx']], d.shape[1],
                                      t_norm['powers'], t_norm.get('input_powers'))
            elif str(key).startswith('X'):                           # scale the strain inputs by t^b
                d = scale_inputs(d, t_norm)
        if fc_norm is not None and str(key).startswith('y') and x_key in data:
            d = d / np.asarray(data[x_key][:, fc_norm['fc_idx']], dtype=float).reshape(-1, 1)   # divide outputs by fc
        stats['stats_' + key] = statistics(d)

    return stats


def t_norm_power_vector(powers, n_cols: int, input_powers=None) -> np.ndarray:
    """
    Per-output-column t-power for y = [sig(6), D(36)]: sig_j -> a_j, D[j,k] -> a_j + b_k.

    Args:
        powers       (list): output powers a (6,), e.g. [1, 1, 1, 2, 2, 2]
        n_cols       (int):  number of columns of y (6: sig only, 42: sig and D)
        input_powers (list): strain input powers b (6,); None / all-zero -> output-only scheme

    Returns:
        p          (np.arr): t-power per column, shape (n_cols,)
    """
    a = np.asarray(powers, dtype=float)                                                   # (6,) output powers
    b = np.zeros(6) if input_powers is None else np.asarray(input_powers, dtype=float)    # (6,) input powers
    p_D = (a[:, None] + b[None, :]).ravel()                                               # (36,) row-major
    p = np.concatenate([a, p_D])                                                          # (42,)
    return p[:n_cols]


def _t_factor(t_phys, n_cols: int, powers, input_powers=None) -> np.ndarray:
    """
    Per-sample divisor t^power of the outputs (see t_norm_power_vector).

    Args:
        t_phys       (np.arr): physical thickness per sample, shape (n,)
        n_cols       (int):    number of columns of y
        powers       (list):   output powers (6,)
        input_powers (list):   strain input powers (6,) or None

    Returns:
        factor     (np.arr):   t^power, shape (n, n_cols)
    """
    p = t_norm_power_vector(powers, n_cols, input_powers)
    t = np.asarray(t_phys, dtype=float).reshape(-1, 1)
    return np.power(t, p.reshape(1, -1))


def scale_inputs(X: np.ndarray, t_norm) -> np.ndarray:
    """
    Input side of the t-normalisation: multiply the 6 strain columns of X by t^input_powers (t from column
    t_idx of X); geometry columns are unchanged. With input_powers [0,0,0,1,1,1] the curvatures (chi ~ 1/t)
    become O(1).

    Args:
        X           (np.arr): inputs [eps(6), geom], shape (n, 6+n_geom), physical units
        t_norm      (dict):   {'input_powers', 't_idx', ...} (see t_norm_from_inp); None -> no-op

    Returns:
        Xs          (np.arr): scaled copy of X (X itself if t_norm is None or input_powers is all-zero)
    """
    if t_norm is None:
        return X
    b = np.asarray(t_norm.get('input_powers', np.zeros(6)), dtype=float)
    if not np.any(b):
        return X
    t = np.asarray(X[:, t_norm['t_idx']], dtype=float)
    Xs = X.copy()
    Xs[:, :6] = X[:, :6] * (t[:, None] ** b[None, :])
    return Xs


def t_norm_from_inp(inp: dict):
    """
    t-normalisation config from the hyperparameters. Stored in stats (saved with the model), so training and
    inference use the same scheme.

    Args:
        inp         (dict): hyperparameters (t_normalise, t_norm_powers, t_norm_input_powers, t_idx)

    Returns:
        t_norm      (dict|None): {'powers', 'input_powers', 't_idx'} if inp['t_normalise'], else None
    """
    if not inp.get('t_normalise'):
        return None
    return {'powers':       list(inp.get('t_norm_powers',       [1, 1, 1, 2, 2, 2])),
            'input_powers': list(inp.get('t_norm_input_powers', [0, 0, 0, 1, 1, 1])),
            't_idx':        int(inp.get('t_idx', 6))}


def apply_t_norm(y: np.ndarray, t_phys, stats: dict) -> np.ndarray:
    """
    Divide the physical outputs by t^power (before the standardisation).

    Args:
        y           (np.arr): outputs [sig(6)(, D(36))], shape (n, 6) or (n, 42)
        t_phys      (np.arr): physical thickness per sample, shape (n,); None -> no-op
        stats       (dict):   statistics with the 't_norm' block; without it -> no-op

    Returns:
        y_t         (np.arr): t-divided outputs, same shape as y
    """
    tn = stats.get('t_norm') if isinstance(stats, dict) else None
    if tn is None or t_phys is None:
        return y
    return y / _t_factor(t_phys, y.shape[1], tn['powers'], tn.get('input_powers')).astype(y.dtype)


def undo_t_norm(y: np.ndarray, t_phys, stats: dict) -> np.ndarray:
    """
    Inverse of apply_t_norm: multiply the outputs by t^power.

    Args:
        y           (np.arr): t-divided outputs, shape (n, 6) or (n, 42)
        t_phys      (np.arr): physical thickness per sample, shape (n,); None -> no-op
        stats       (dict):   statistics with the 't_norm' block; without it -> no-op

    Returns:
        y_phys      (np.arr): outputs in physical units, same shape as y
    """
    tn = stats.get('t_norm') if isinstance(stats, dict) else None
    if tn is None or t_phys is None:
        return y
    return y * _t_factor(t_phys, y.shape[1], tn['powers'], tn.get('input_powers')).astype(y.dtype)


def fc_norm_from_inp(inp: dict):
    """
    fc-normalisation config from the hyperparameters. Stored in stats (saved with the model).

    Args:
        inp         (dict): hyperparameters (fc_normalise, fc_idx; fc_idx is the column of fc in the input vector,
                            11 for [eps(6), t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs])

    Returns:
        fc_norm     (dict|None): {'fc_idx'} if inp['fc_normalise'], else None
    """
    if not inp.get('fc_normalise'):
        return None
    return {'fc_idx': int(inp.get('fc_idx', 11))}


def apply_fc_norm(y: np.ndarray, fc_phys, stats: dict) -> np.ndarray:
    """
    Divide the outputs (all stresses and D entries) by the per-sample fc (before the standardisation).

    Args:
        y           (np.arr): outputs, shape (n, 6) or (n, 42)
        fc_phys     (np.arr): concrete strength per sample, shape (n,); None -> no-op
        stats       (dict):   statistics with the 'fc_norm' block; without it -> no-op

    Returns:
        y_fc        (np.arr): fc-divided outputs, same shape as y
    """
    fn = stats.get('fc_norm') if isinstance(stats, dict) else None
    if fn is None or fc_phys is None:
        return y
    return y / np.asarray(fc_phys, dtype=float).reshape(-1, 1).astype(y.dtype)


def undo_fc_norm(y: np.ndarray, fc_phys, stats: dict) -> np.ndarray:
    """
    Inverse of apply_fc_norm: multiply the outputs by fc.

    Args:
        y           (np.arr): fc-divided outputs, shape (n, 6) or (n, 42)
        fc_phys     (np.arr): concrete strength per sample, shape (n,); None -> no-op
        stats       (dict):   statistics with the 'fc_norm' block; without it -> no-op

    Returns:
        y_phys      (np.arr): outputs multiplied by fc, same shape as y
    """
    fn = stats.get('fc_norm') if isinstance(stats, dict) else None
    if fn is None or fc_phys is None:
        return y
    return y * np.asarray(fc_phys, dtype=float).reshape(-1, 1).astype(y.dtype)


def get_normalised_data(data: dict, stats: dict, sobolev: bool) -> dict:
    """
    Normalise all subsets in data with the train statistics.

    Inputs: (t-scaling, then) standardisation. Stresses: (t- and fc-division, then) standardisation. D: scaled
    with std(sig_j) / std(eps_k) ('stitched'), consistent with the Jacobian of the normalised network.

    Args:
        data        (dict): dataset with X_* and y_* arrays.
        stats       (dict): statistics (see get_stats); only the train statistics are used.
        sobolev     (bool): unused (kept for the call signature); D columns are normalised whenever present.

    Returns:
        normalised_data (dict): keys with suffix '_t' (e.g. 'X_train_t').
    """

    xdim = data['X_test'].shape[1]         # 6+GEOM_SIZE
    ydim = data['y_test'].shape[1]         # 6 or 6+36
    norm_type_x = ['x-std']*xdim
    norm_type_y = ['y-std']*6+['y-st-stitched']*36 if ydim > 6 else ['y-std']*ydim

    tn = stats.get('t_norm') if isinstance(stats, dict) else None    # optional t-normalisation
    fn = stats.get('fc_norm') if isinstance(stats, dict) else None   # optional fc-normalisation

    normalised_data = {}
    for key in data.keys():
        if 'X' in key:
            X = scale_inputs(data[key], tn) if tn is not None else data[key]
            normalised_data[key+'_t'] = transform_data(X, stats, forward=True, type=norm_type_x)
        elif 'y' in key:
            y = data[key]
            x_key = key.replace('y', 'X', 1)
            if tn is not None and x_key in data:
                y = apply_t_norm(y, data[x_key][:, tn['t_idx']], stats)
            if fn is not None and x_key in data:
                y = apply_fc_norm(y, data[x_key][:, fn['fc_idx']], stats)
            normalised_data[key+'_t'] = transform_data(y, stats, forward=True, type=norm_type_y)

    return normalised_data


def statistics(data: np.ndarray):
    """
    Column-wise statistics of one data array.

    Args:
        data        (np.arr): shape (n, ncols)

    Returns:
        stats       (dict):   'mean', 'std', 'log_mean', 'log_std' (of log(data + |min| + 1e-8)), 'max', 'min',
                              'q_5', 'q_95'; each of shape (ncols,)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    q_95 = np.percentile(data, 95, axis=0)
    q_5 = np.percentile(data, 5, axis=0)

    delta = 1e-8
    log_mean = np.mean(np.log(data+np.abs(min)+delta), axis=0)
    log_std = np.std(np.log(data+np.abs(min)+delta), axis=0)

    stats = {
        'mean': mean,
        'std': std,
        'log_mean': log_mean,
        'log_std': log_std,
        'max': max,
        'min': min,
        'q_5': q_5,
        'q_95': q_95
    }
    return stats


def transform_data(data: np.ndarray, stats_: dict, forward: bool, type: list):
    """
    Forward: standardise the data. Backward: transform standardised data back to the original scale.

    Args:
        data            (np.arr):   data, shape (n, ncols)
        stats_          (dict):     statistics (only stats_X_train / stats_y_train are used)
        forward         (bool):     True: forward transformation, False: backward transformation
        type            (list):     per column: 'x-std' / 'y-std' (standardisation with the X / y stats) or
                                    'y-st-stitched' (D entries, scaled with std(sig_j) / std(eps_k))

    Returns:
        new_data       (np.arr):    transformed data
    """

    data_stdnorm = np.zeros((data.shape))
    data_nonorm = np.zeros((data.shape))
    new_data = np.zeros((data.shape))
    new_data_ = np.zeros((data.shape))
    np_data = data.copy()
    np_data_ = data.copy()

    # for the stitched transformation of D: D_coeff[j,k] = std(sig_j) / std(eps_k)
    if 'y-st-stitched' in type:
        D_coeff_sz = int(np.sqrt(type.count('y-st-stitched')))
        D_coeff = np.zeros((D_coeff_sz, D_coeff_sz))
        for j in range(D_coeff_sz):
            for k in range(D_coeff_sz):
                D_coeff[j, k] = stats_['stats_y_train']['std'][j]/stats_['stats_X_train']['std'][k]
        D_coeff_ = D_coeff.reshape((1, D_coeff_sz*D_coeff_sz))

    if forward:
        for i in range(data.shape[1]):
            if 'x' in type[i]:
                stats = stats_['stats_X_train']
            elif 'y' in type[i]:
                stats = stats_['stats_y_train']
            else:
                raise RuntimeError('Please define an appropriate transformation type including the variable x or y to be transformed.')

            if 'std' in type[i]:
                # standard normal distribution
                if stats['std'][i] == 0:
                    stats['std'][i] = 1.0
                data_stdnorm[:, i] = (np_data[:, i]-stats['mean'][i]*np.ones(np_data.shape[0]))/(stats['std'][i]*np.ones(np_data.shape[0]))
                new_data[:, i] = data_stdnorm[:, i]

            elif 'st-stitched' in type[i]:
                # D entries; assumes y = [sig(D_coeff_sz), D(D_coeff_sz*D_coeff_sz)]
                np_data_[:, i] = np_data[:, i]
                data_stdnorm[:, i] = np.divide(1, D_coeff_[0, i-D_coeff_sz])*np_data_[:, i]
                new_data[:, i] = data_stdnorm[:, i]

    elif not forward:
        for i in range(data.shape[1]):
            if 'x' in type[i]:
                stats = stats_['stats_X_train']
            elif 'y' in type[i]:
                stats = stats_['stats_y_train']
            else:
                raise RuntimeError('Please define an appropriate transformation type including the variable x or y to be transformed.')

            if 'std' in type[i]:
                if stats['std'][i] == 0:
                    stats['std'][i] = 1.0
                data_nonorm[:, i] = np_data[:, i]*stats['std'][i]*np.ones(np_data.shape[0])+stats['mean'][i]*np.ones(np_data.shape[0])
                new_data[:, i] = data_nonorm[:, i]

            elif 'st-stitched' in type[i]:
                data_nonorm[:, i] = D_coeff_[0, i-D_coeff_sz]*np_data[:, i]
                new_data_[:, i] = data_nonorm[:, i]
                new_data[:, i] = new_data_[:, i]

    return new_data


def plot_norm_data(data: dict, plot: bool):
    """
    3D scatter plots of the normalised training strains and stresses.

    Args:
        data        (dict): normalised dataset (see get_normalised_data), uses X_train_t and y_train_t
        plot        (bool): create the plots (False -> no-op)

    Returns:
        None; figures saved to training/plots
    """
    if plot:
        plot_3D_data(data['X_train_t'][:, :6], 'train_norm_scatter_eps_g')
        plot_3D_data(data['y_train_t'][:, :6], 'train_norm_scatter_sig_g')
    return


def data_to_torch(data: dict) -> dict:
    """
    Convert the numpy arrays to float32 torch tensors.

    Args:
        data        (dict): numpy arrays, e.g. the output of get_normalised_data ('X_train_t', ...)

    Returns:
        data_torch  (dict): torch tensors; keys get an extra suffix 't' (e.g. 'X_train_tt')
    """

    data_torch = {}
    for key in data.keys():
        data_torch[key+'t'] = torch.from_numpy(data[key])
        data_torch[key+'t'] = data_torch[key+'t'].type(torch.float32)

    return data_torch


################################## Plotting ##################################

def plot_3D_data(data, filename, n_every: int = 1):
    """
    3D scatter of the membrane and bending triplets of a dataset (strains or stresses), saved to training/plots.

    Args:
        data       (np.arr): shape (ntot, 6)
        filename   (str):    figure name; must contain 'eps' or 'sig' (axis labels)
        n_every    (int):    unused (kept for the call signature)

    Returns:
        None; figure saved as training/plots/<filename>.png
    """
    fig = plt.figure(figsize=(14, 7))

    for i in range(2):
        x = data[:, i*3]
        y = data[:, i*3+1]
        z = data[:, i*3+2]

        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.scatter(x, y, z, s=2, alpha=0.1)
        ax.set_title(f'$N$ = {len(x)/1e6:.2f}*1e6', y=0.95)
        figure_formatting(ax, i, filename)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, filename + ".png"))
    print(f'Saved {filename} to {PLOT_DIR}{os.sep}{filename}.png')

    return


def figure_formatting(ax, i, filename):
    """
    Axis labels of the 3D scatter plots.

    Args:
        ax          (Axes3D): subplot
        i           (int):    0: membrane triplet (eps_x, eps_y, gamma_xy / n_x, n_y, n_xy), 1: bending triplet
        filename    (str):    figure name; 'eps' -> strain labels, 'sig' -> stress labels

    Returns:
        None
    """
    if 'eps' in filename:
        if i == 0:
            ax.set_xlabel('eps_x')
            ax.set_ylabel('eps_y')
            ax.set_zlabel('gamma_xy')
        elif i == 1:
            ax.set_xlabel('chi_x')
            ax.set_ylabel('chi_y')
            ax.set_zlabel('chi_xy')

    elif 'sig' in filename:
        if i == 0:
            ax.set_xlabel('n_x')
            ax.set_ylabel('n_y')
            ax.set_zlabel('n_xy')
        elif i == 1:
            ax.set_xlabel('m_x')
            ax.set_ylabel('m_y')
            ax.set_zlabel('m_xy')
