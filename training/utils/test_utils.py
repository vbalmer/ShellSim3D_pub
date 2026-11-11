"""Testing of trained models: diagonal plots (prediction vs label) on the test set, and the prediction
functions predict_sig / predict_D / predict_sig_D / predict_NN_symm / predict_NN_xysymm (thin wrappers around
NNPredictor in predictor.py). V. Balmer, 2026.
"""

import os
import re
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText

from .architectures import FFNN
from .data_utils import get_normalised_data
from .predictor import (NNPredictor, device, get_inp_from_folder, get_stats_from_folder, get_full_model_path,
                        test_model_instance)


################################ main test function ################################

def test_NN_model(test_data: dict, stats: dict, save_path: str, version: int,
                  plot_path = None, thickness_split = False, rhoy_split = False):
    """
    Main testing function: diagonal plots (prediction vs label) of stresses and stiffnesses on the test set.
    inp and the trained model are read from the model's version folder.

    Args:
        test_data       (dict):       test data in physical units (X_test, y_test)
        stats           (dict):       statistics (only the train statistics are used for the normalisation)
        save_path       (str):        folder containing the version folders v_<n>
        version         (int):        version of the trained model; None -> latest version in save_path
        plot_path       (str):        output folder of the figures; None -> save_path/v_<version>/plots
        thickness_split (float|bool): if set, only points of this thickness [mm] are coloured (rest gray)
        rhoy_split      (float|bool): if set, only points with this rho_y [-] are coloured (rest gray)

    Returns:
        diagonal plots (png) in plot_path
    """

    ## 0 - Create test model instance
    if version is None:
        version = get_latest_version(save_path)
    if plot_path is None:
        plot_path = os.path.join(save_path, f'v_{version}', 'plots')
    os.makedirs(plot_path, exist_ok=True)
    inp = get_inp_from_folder(save_path, version)
    test_model = test_model_instance(inp, save_path, version)

    ## 1 - Predict stresses
    plot_data_sig = predict_sig(test_model, inp, test_data, stats, sobolev=inp['Sobolev'])

    ## 2 - Predict stiffnesses
    plot_data_D = predict_D(test_model, inp, test_data, stats, sobolev=inp['Sobolev'])

    ## 2b - split masks (highlight only points of the given thickness / rho_y, rest gray)
    splits = build_splits(test_data, thickness_split, rhoy_split)

    ## 3+4 - Create diagonal plots for stresses and stiffnesses, once per split
    for mask, suffix in splits:
        make_diagonal_plots(plot_data_sig, plot_data_D, stats, plot_path, mask, suffix)

    return


def build_thickness_mask(test_data: dict, thickness_split):
    """Boolean mask selecting the test points of a given thickness, for the thickness-split diagonal plots.

    Args:
        test_data      (dict):       PHYSICAL test data; X_test = [eps(6), t, rho...] so t is column 6.
        thickness_split (float|bool): thickness [mm] to highlight; False -> no split (all points coloured).

    Returns:
        tuple(np.ndarray|None, str): (mask, suffix). mask is the (ntest,) bool array of selected points
            (None if split off / column absent / no match); suffix is a filename tag like '_t100' ('' if None).
    """
    if not thickness_split or test_data['X_test'].shape[1] <= 6:
        return None, ''
    t_pts = test_data['X_test'][:, 6]                     # physical thickness (geom col 0, index 6)
    m = np.isclose(t_pts, float(thickness_split))
    if not m.any():
        print(f'[thickness_split] no points with t={thickness_split} mm found; plotting all coloured.')
        return None, ''
    print(f'[thickness_split] highlighting t={int(thickness_split)} mm: {m.sum()}/{len(m)} points')
    return m, f'_t{int(thickness_split)}'


def build_rhoy_mask(test_data: dict, rhoy_split):
    """Boolean mask selecting the test points of a given y-reinforcement ratio, for the rho_y-split diagonal plots.

    rho_y is taken as rho_ybot (column 10) for the full layouts [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot(, fc(, fs))]
    (5-, 6- and 7-geom; fc / fs are appended after rho), else as the LAST geom column (X_test[:, -1]):
    [t, rho_x, rho_y] (3-geom) and the single-geom rho-only model (1-geom, geom col 6). If that
    column is actually t (a thickness-only model), rhoy_split (~0.01-0.04) simply matches nothing
    and we fall back to all-coloured with a printed message.

    Args:
        test_data  (dict):       PHYSICAL test data; X_test = [eps(6), geom...] with rho_y last.
        rhoy_split (float|bool): rho_y [-] to highlight; False -> no split (all points coloured).

    Returns:
        tuple(np.ndarray|None, str): (mask, suffix). mask is the (ntest,) bool array of selected points
            (None if split off / no geom column / no match); suffix is a filename tag like '_rhoy0.025' ('' if None).
    """
    if not rhoy_split:
        return None, ''
    if test_data['X_test'].shape[1] <= 6:                 # no geom columns at all (6-input model)
        print('[rhoy_split] model has no geom columns; plotting all coloured.')
        return None, ''
    rho_col = 10 if test_data['X_test'].shape[1] >= 6 + 5 else -1     # full layouts: rho_ybot (fc / fs follow)
    rho_pts = test_data['X_test'][:, rho_col]             # physical rho_y
    m = np.isclose(rho_pts, float(rhoy_split))
    if not m.any():
        vals = np.unique(np.round(rho_pts, 6))
        print(f'[rhoy_split] no points with rho_y={rhoy_split} found (last geom col values: {vals}); '
              f'plotting all coloured.')
        return None, ''
    print(f'[rhoy_split] highlighting rho_y={rhoy_split}: {m.sum()}/{len(m)} points')
    return m, f'_rhoy{rhoy_split:g}'


def build_splits(test_data: dict, thickness_split, rhoy_split):
    """Collect the (mask, suffix) pairs for the diagonal plots: one set of plots per active split.

    Args:
        test_data      (dict):       PHYSICAL test data.
        thickness_split (float|bool): thickness [mm] to highlight; False -> off.
        rhoy_split      (float|bool): y-reinforcement ratio rho_y [-] to highlight; False -> off.

    Returns:
        list[tuple[np.ndarray|None, str]]: (mask, suffix) pairs. Base (None, '') if no split is active;
            duplicate suffixes (e.g. both splits fell back to base) are dropped to avoid overwriting figures.
    """
    candidates = []
    if thickness_split:
        candidates.append(build_thickness_mask(test_data, thickness_split))
    if rhoy_split:
        candidates.append(build_rhoy_mask(test_data, rhoy_split))
    if not candidates:
        candidates.append((None, ''))

    splits, seen = [], set()
    for mask, suffix in candidates:
        if suffix in seen:
            continue
        seen.add(suffix)
        splits.append((mask, suffix))
    return splits


def make_diagonal_plots(plot_data_sig: dict, plot_data_D: dict, stats: dict,
                        plot_path: str, mask, suffix: str):
    """Create the four diagonal plots (stresses & stiffnesses, normalised & transformed) for one split.

    Args:
        plot_data_sig (dict): predicted/label stresses (keys 'labels'/'pred'/'labels_norm'/'pred_norm').
        plot_data_D   (dict): predicted/label stiffnesses (same keys).
        stats         (dict): train statistics used for the RSE colouring / denormalisation.
        plot_path      (str): output folder for the figures.
        mask (np.ndarray|None): points to colour (rest gray); None -> all coloured.
        suffix         (str): filename tag identifying the split (e.g. '_t100', '_rhoy0.025', '').

    Returns:
        list[str]: the four generated figure filenames.
    """
    return [
        # stresses
        multiple_diagonal_plots(plot_data_sig['labels_norm'], plot_data_sig['pred_norm'],
                                stats, 'rse', plot_path, transf = 't', mask = mask, suffix = suffix),
        multiple_diagonal_plots(plot_data_sig['labels'], plot_data_sig['pred'],
                                stats, 'rse', plot_path, transf = 'u', mask = mask, suffix = suffix),
        # stiffnesses
        multiple_diagonal_plots_D(plot_data_D['labels_norm'], plot_data_D['pred_norm'],
                                  stats, 'rse', plot_path, transf = 't', mask = mask, suffix = suffix),
        multiple_diagonal_plots_D(plot_data_D['labels'], plot_data_D['pred'],
                                  stats, 'rse', plot_path, transf = 'u', mask = mask, suffix = suffix),
    ]


################################ auxiliary functions for testing ################################

def get_latest_version(path: str):
    """Highest n of the folders v_<n> in path (None if there are none)."""
    versions = [int(m.group(1)) for name in os.listdir(path) if (m := re.match(r'v_(\d+)$', name))]
    if not versions:
        return None
    print(f'No version number entered in input. Testing latest available version {max(versions)}')
    return max(versions)

def get_testdata_from_folder(path:str, version: int):
    """
    get test data from given folder

    Args:
        path    (str): location where test data and model is saved
        version (int): version number of to be tested model
    """

    full_path = os.path.join(path, 'v_'+str(version))
    with open(os.path.join(full_path, 'test_data.pkl'),'rb') as handle:
        test_data = pickle.load(handle)

    return test_data


################################ prediction (thin wrappers around NNPredictor) ################################
# Kept for the existing callers (single_element_utils, FE deployment code); new code can use NNPredictor directly.

def _symmetry(symm_predict: bool, xysymm_predict: bool):
    return 'midplane' if symm_predict else ('xy' if xysymm_predict else None)


def predict_sig(test_model:FFNN, inp:dict, test_data:dict, stats:dict, sobolev:bool, symm_predict:bool=False, xysymm_predict:bool=False):
    """
    Stresses of the trained model on a data set, with labels.

    Args:
        test_model  (FFNN):     Trained model in eval mode
        inp         (dict):     Hyperparameters of trained model
        test_data   (dict):     X_test (physical inputs) and y_test (labels [sig(6), D(36)] or dummy)
        stats       (dict):     Statistics of the training data (normalisation)
        sobolev     (bool):     unused (kept for the call signature)
        symm_predict, xysymm_predict (bool): mid-plane / x <-> y symmetry wrapper (see NNPredictor.predict)

    Returns:
        plot_data   (dict):     'pred', 'pred_norm', 'labels', 'labels_norm', each (n, 6)
    """
    P = NNPredictor(test_model, inp, stats)
    X = test_data['X_test']
    pred = P.predict(X, D=False, symmetry=_symmetry(symm_predict, xysymm_predict))['sig']
    return {
        'pred': pred,
        'pred_norm': P.normalise_sig(pred, X),
        'labels': test_data['y_test'][:, :6],
        'labels_norm': get_normalised_data(test_data, stats, sobolev)['y_test_t'][:, :6],
    }


def predict_D(test_model: FFNN, inp: dict, test_data: dict, stats: dict, sobolev:bool, non_batchwise:bool=False, symm_predict:bool=False, xysymm_predict:bool=False,
              strain_jac_only:bool=False):
    """
    Stiffnesses (jacobian of the stresses w.r.t. the strains) of the trained model on a data set, with labels.

    Args:
        test_model     (FFNN): Trained model
        inp            (dict): Hyperparameters
        test_data      (dict): X_test (physical inputs) and y_test (labels [sig(6), D(36)] or dummy)
        stats          (dict): Statistics of the training data (normalisation)
        sobolev        (bool): True, if test_model was trained including sobolev losses (warning otherwise).
        non_batchwise  (bool): jacobian on the full input in one pass (FE deployment); default batchwise.
        symm_predict, xysymm_predict (bool): mid-plane / x <-> y symmetry wrapper (see NNPredictor.predict)
        strain_jac_only (bool): unused (the jacobian is always taken w.r.t. the 6 strains only); kept for the
                               call signature.
    Returns:
        plot_data   (dict): 'pred', 'pred_norm', 'labels', 'labels_norm', each (n, 6, 6)
    """
    if not sobolev:
        print('This model was not trained with Sobolev losses. Predictions may differ significantly from labels in dataset.')

    P = NNPredictor(test_model, inp, stats)
    X = test_data['X_test']
    pred = P.predict(X, sig=False, symmetry=_symmetry(symm_predict, xysymm_predict), batched=not non_batchwise)['D']
    return {
        'pred': pred,
        'pred_norm': P.normalise_D(pred, X),
        'labels': test_data['y_test'][:, 6:].reshape((-1, 6, 6)),
        'labels_norm': get_normalised_data(test_data, stats, sobolev)['y_test_t'][:, 6:].reshape((-1, 6, 6)),
    }


def predict_sig_D(test_model: FFNN, inp: dict, test_data: dict, stats: dict, sobolev: bool, symm_predict: bool = False,
                  xysymm_predict: bool = False):
    """
    Stress and tangent in [N, mm] from one NN pass per input set (CPU), no labels.

    Returns:
        dict: 'sig' (n, 6), 'D' (n, 6, 6)
    """
    return NNPredictor(test_model, inp, stats).predict(test_data['X_test'], one_pass=True,
                                                       symmetry=_symmetry(symm_predict, xysymm_predict))


def _symm_wrapper(symmetry, test_model, inp, test_data, stats, compute_sig, compute_D, one_pass):
    P = NNPredictor(test_model, inp, stats)
    X = test_data['X_test']
    if one_pass:
        compute_sig = compute_D = True
    out = P.predict(X, sig=compute_sig, D=compute_D, symmetry=symmetry, one_pass=one_pass)
    sig_n = None if out['sig'] is None else torch.as_tensor(P.normalise_sig(out['sig'], X), dtype=torch.float32)
    D_n = None if out['D'] is None else P.normalise_D(out['D'], X)
    return sig_n, D_n


def predict_NN_symm(test_model, inp, test_data, stats, sobolev, compute_sig=True, compute_D=True, strain_jac_only=False,
                    one_pass=False):
    """Mid-plane symmetry wrapper g(x) = 0.5*[f(x) + R_out f(R_in x)]; see NNPredictor.predict(symmetry='midplane').

    Returns:
        tuple: (sig_g_norm torch (n, 6) | None, D_g_norm np (n, 6, 6) | None), NORMALISED.
    """
    return _symm_wrapper('midplane', test_model, inp, test_data, stats, compute_sig, compute_D, one_pass)


def predict_NN_xysymm(test_model, inp, test_data, stats, sobolev, compute_sig=True, compute_D=True, one_pass=False):
    """x <-> y relabelling-symmetry wrapper g(x) = 0.5*[f(x) + P_out f(P_in x)]; see NNPredictor.predict(symmetry='xy').

    Returns:
        tuple: (sig_g_norm torch (n, 6) | None, D_g_norm np (n, 6, 6) | None), NORMALISED.
    """
    return _symm_wrapper('xy', test_model, inp, test_data, stats, compute_sig, compute_D, one_pass)


################################ auxiliary functions for plotting stresses ################################

def multiple_diagonal_plots(Y: np.array, preds: np.array, stats: dict, error_type: str, save_path: str, transf:str,
                            mask = None, suffix = ''):
    """
    Creates diagonal plots to show results of testing.

    Args:
        Y       (np.arr):   test labels (shape: ntest, 6)
        preds   (np.arr):   test predictions (shape: ntest, 6)
        stats     (dict):   statistical values utilised in error calculation (only train values)
        error_type (str):   'rse', 'nrse', ...
        save_path  (str):   location where plot will be stored.
        transf    (dict):   't': plotting normalised data, 'u': plotting original units data
        mask    (np.arr):   optional (ntest,) bool; if given, only these points are RSE-coloured, rest gray.
        suffix     (str):   filename tag appended to the saved plot (e.g. '_t100').

    Returns:
        diagonal plots as png in save_path
    """


    # Calculate relevant errors
    errors = calculate_errors(Y, preds, stats, transf, id = 'sig')

    # Define figures
    fig, ax = plt.subplots(2, 3, figsize=[15.5, 10], dpi=100)
    fig.subplots_adjust(wspace=0.5)

    # Plot figures
    errors_ = {key: arr.reshape(-1, 2, 3) for key, arr in errors.items()}
    scatters = plot_main_figure(ax, Y.reshape((-1,2,3)), preds.reshape((-1,2,3)),
                                errors_, error_type, mask = mask)

    # Formatting figures
    figure_formatting(ax, transf, Y.reshape((-1,2,3)), preds.reshape((-1,2,3)))
    add_description_box(ax, errors_)
    add_colorbars(fig, ax, scatters, transf, error_type)
    for ax_ in ax.ravel():
        ax_.set_aspect('equal', 'box')
        ax_.axis('square')

    # Save figure
    return save_diag_figure(save_path, transf, id = 'sig', suffix = suffix)




def figure_formatting(ax, transf:str, Y:np.array, pred:np.array):
    """
    Sets up formatting and labels for axes

    Args: 
        ax      ():         axes of figure
        transf  (str):      type to plot (t: normalised or u: in original units)
        Y       (np.arr):   labels (for limit calculation), shape: (ntest, 2, 3)
        pred    (np.arr):   predictions (for limit calculation), shape (ntest, 2, 3)
    
    """
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })
    
    if transf == 'u':
        plotname = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
                            [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
                            [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$'],
                          [r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$'],  
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$']])
    elif transf == 't':
        plotname = np.array([[r'$n_{x,norm}$', r'$n_{y,norm}$', r'$n_{xy,norm}$'],
                        [r'$m_{x,norm}$', r'$m_{y,norm}$', r'$m_{xy,norm}$'],
                        [r'$v_{x,norm}$', r'$v_{y,norm}$', r'$v_{y,norm}$']])
        plotname_p = np.array([[r'$\tilde{n}_{x,norm}$', r'$\tilde{n}_{y,norm}$', r'$\tilde{n}_{xy,norm}$'],
                        [r'$\tilde{m}_{x,norm}$', r'$\tilde{m}_{y,norm}$', r'$\tilde{m}_{xy,norm}$'],
                        [r'$\tilde{v}_{x,norm}$', r'$\tilde{v}_{y,norm}$', r'$\tilde{v}_{y,norm}$']])
        units = np.array([['$[-]$', '$[-]$', '$[-]$'],
                          ['$[-]$', '$[-]$', '$[-]$'],  
                          ['$[-]$', '$[-]$', '$[-]$']])

    for i in range(2):
        for j in range(3):
            ax[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
            ax[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
            ax[i,j].set_xlim([np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), 
                              np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])])
            ax[i,j].set_ylim([np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), 
                              np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])])
            ax[i,j].grid(True, which='major', color='#666666', linestyle='-')
    
    return 

def calculate_errors(Y, predictions, stats, transf, id = 'sig'):
    if id == 'sig':
        num_cols = 6
        num_cols_plt = 6
    elif id == 'De-NLRC': 
        idx = 6
        num_cols_plt = 36

    ### Transform the units of the statistics which relate back to the train set to the units desired in the diagonal plot
    if transf == 't':
        # for transformed units.
        if id == 'sig' or id == 'eps':
            mean_train_ = 0*np.ones((1,num_cols))
            q_5_train_ = -1.645*np.ones((1,num_cols))
            q_95_train_ = 1.645*np.ones((1,num_cols))

            mean_train = mean_train_
            q_5_train = q_5_train_
            q_95_train = q_95_train_

        elif id == 'De-NLRC':
            mean_train_ = 0*np.ones((1,idx, idx))
            q_5_train_ = -1.645*np.ones((1,idx, idx))
            q_95_train_ = 1.645*np.ones((1,idx, idx))

            mean_train = mean_train_.reshape((-1, idx*idx))
            q_5_train = q_5_train_.reshape((-1, idx*idx))
            q_95_train = q_95_train_.reshape((-1, idx*idx))

    elif transf == 'u':
        # the statistics are already in the units N, mm
        if id == 'sig' or id == 'eps':
            mean_train_ = stats['stats_y_train']['mean'][0:num_cols].reshape((1,num_cols))
            q_5_train_ = stats['stats_y_train']['q_5'][0:num_cols].reshape((1,num_cols))
            q_95_train_ = stats['stats_y_train']['q_95'][0:num_cols].reshape((1,num_cols))
        
            mean_train = mean_train_
            q_5_train = q_5_train_
            q_95_train = q_95_train_

        if id == 'De-NLRC':
            # for nonlinear version of De (i.e. Dmb is not zero)
            # if predicting sigma with network and D with derivatives
            mean_train_ = stats['stats_y_train']['mean']
            q_5_train_ = stats['stats_y_train']['q_5']
            q_95_train_ = stats['stats_y_train']['q_95']

            mean_train = mean_train_[idx:].reshape((-1,idx*idx))
            q_5_train = q_5_train_[idx:].reshape((-1,idx*idx))
            q_95_train = q_95_train_[idx:].reshape((-1,idx*idx))


    ### Calculate errors
    r_squared2 = np.zeros((1,num_cols_plt))
    rse_max=np.zeros((1,num_cols_plt))
    n_5p, n_10p, rmse, aux_, aux__, rrmse = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rrse_max, nrse_max, nrmse, log_max, mean_log_err = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rse, nrse, log_err = np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt))
    # Delta_max_i, Delta_max_max, Delta_max_mean = np.zeros((Y.shape[0], num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    for i in range(num_cols_plt):
        Y_col = Y[:, i].flatten()
        pred_col = predictions[:,i].flatten()
        r_squared2[:,i] = np.corrcoef(Y_col, pred_col)[0, 1]**2
        rse[:,i] = np.sqrt((pred_col-Y_col)**2)
        rse_max[:,i] = np.max(rse[:,i])
        rmse[:,i] = np.sqrt(np.mean((pred_col - Y_col) ** 2))
        aux_[:,i] = np.sqrt(np.mean((mean_train[:,i]*np.ones(Y_col.shape) - Y_col) ** 2))
        if aux_[:,i].any() == 0:
            aux_[aux_[:,i] == 0] = 1
        rrmse[:,i] = np.divide(rmse[:,i],aux_[:,i])
        rrse_max[0,i] = np.max(np.divide(np.sqrt((pred_col-Y_col)**2), aux_[:,i]))

        # Calculate normalised RMSE
        aux__[:,i] = q_95_train[:,i]-q_5_train[:,i]
        if aux__[:,i].any() == 0:
            aux__[aux__[:,i] == 0] = 1
        nrse[:,i] = np.divide(np.sqrt((pred_col-Y_col)**2), aux__[:,i])*100
        nrmse[:,i] = np.divide(rmse[:,i], aux__[:,i])
        nrse_max[:,i] = np.max(nrse[:,i]/100)


    errors = {
        'rse': rse,
        'rse_max': rse_max,
        'rmse': rmse,
        'nrmse': nrmse,
        'nrse': nrse,
        'nrse_max': nrse_max,
        'rrmse': rrmse,
        'rrse_max': rrse_max,
        'r_squared2': r_squared2,
    }


    return errors

def plot_main_figure(ax, Y, pred, errors, error_type, norms_ = None, mask = None):
    """
    actual scatter plotting

    Args:
        ax      ():         axes of figure
        Y       (np.arr):   labels, shape: (ntest, nrows, ncols)
        pred    (np.arr):   predictions, shape: (ntest, nrows, ncols)
        errors  (np.arr):   errors, shape: (ntest, nrows, ncols)
        error_type (str):   type of error to plot as colorscale
        norms_    (list):   colors for colorbar in stiffness plot. Is none for sig-plot.
        mask    (np.arr):   optional (ntest,) bool; if given, only these points are RSE-coloured (and set
                            the colour scale), the rest are drawn gray. None -> all points coloured.

    """

    nrows = Y.shape[1]
    ncols = Y.shape[2]

    # colour scale from the selected points only (falls back to all points if no mask)
    err_sel = errors[error_type][mask] if mask is not None else errors[error_type]
    if norms_ is None:
        norms = [mcolors.Normalize(vmin=np.min(err_sel[:, i, :]),
                                vmax=np.max(err_sel[:, i, :]))
                                    for i in range(nrows)]
    else:
        norms = norms_.copy()

    scatters = []

    for i in range(nrows):
         for j in range(ncols):
            norm = norms[i] if norms_ is None else norms[i // 3 * 2 + j // 3]
            if mask is not None:
                # non-selected thicknesses in gray, selected coloured by RSE (and drawn on top)
                ax[i,j].scatter(Y[~mask,i,j], pred[~mask,i,j], marker = 'o', s = 20,
                                c = 'lightgray', linestyle='None', alpha = 0.3)
                scatter = ax[i,j].scatter(Y[mask,i,j], pred[mask,i,j], marker = 'o', s = 20,
                                          c = errors[error_type][mask, i,j], cmap = 'plasma', linestyle='None',
                                          alpha = 0.4, norm = norm)
            else:
                scatter = ax[i,j].scatter(Y[:,i,j], pred[:,i,j], marker = 'o', s = 20,
                                          c = errors[error_type][:, i,j], cmap = 'plasma', linestyle='None',
                                          alpha = 0.4, norm = norm)
            scatters.append(scatter)

            # add diagonal line to show where points should be.
            ax[i,j].plot([np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])],
                         [np.min([np.min(Y[:,i,j]), np.min(pred[:,i,j])]), np.max([np.max(Y[:,i,j]), np.max(pred[:,i,j])])],
                            color='white', linestyle='--', linewidth = 1)


    return scatters
            
def add_description_box(axs, errors):
    """
    Adds error description box to every subplot

    ax      ():         axes of plot
    errors  (np.arr):   errors, shape (ntest, nrows, ncols)
    """

    nrows = errors['r_squared2'].shape[1]
    ncols = errors['r_squared2'].shape[2]

    for i in range(nrows):
        for j in range(ncols):
            at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,i,j][0], precision=2) + '$ \n' +
                            '$RMSE = ' + np.array2string(errors['rmse'][:,i,j][0], precision=2) + '$ \n' +
                            '$rRMSE = ' + np.array2string(errors['rrmse'][:,i,j][0]*100, precision=0) + '\% $ \n' +
                            '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,i,j][0]*100, precision=0) + '\% $ \n'+
                            '$nRMSE = ' + np.array2string(errors['nrmse'][:,i,j][0]*100, precision=0) + '\% $  \n'+
                            '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,i,j][0]*100, precision=0) + '\% $',
                            prop=dict(size=10), frameon=True,loc='upper left')
            at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
            axs[i,j].add_artist(at)

    return

def add_colorbars(fig, ax, scatters, transf, error_type):
    """
    add colorbars.
    """

    for i in range(2):
        if error_type == 'rse': 
            if i == 0:
                if transf == 't':
                    name = 'RSE \: [-]'
                elif transf == 'u':
                    name = 'RSE \: [N/mm]'
            if i == 1:
                if transf == 't':
                    name = 'RSE \: [-]'
                elif transf == 'u':
                    name = 'RSE \: [Nmm/mm]'
        elif error_type == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=ax[i,:], orientation='vertical', label=f'Row {i+1} $'+name+'$')
        cbar.set_label('$'+name+'$')

    return

def save_diag_figure(save_path, transf, filename = 'multi_diag_scatter', id = None, suffix = ''):
    filename = 'diagonal_match_'+id+'_'+transf+suffix+'.png'
    save_location = os.path.join(save_path, filename)
    plt.savefig(save_location, dpi=100, bbox_inches='tight')
    print(f'Saved {filename} to {save_path}')

    return filename

################################ auxiliary functions for plotting stiffnesses ################################

def multiple_diagonal_plots_D(D_true: np.array, D_pred: np.array, stats: dict, error_type: str, save_path: str, transf: str,
                              mask = None, suffix = ''):
    """
    Creates diagonal plots to show results of testing for stiffness

    Args:
        D_true  (np.arr):   test labels (shape: ntest, 6, 6)
        D_pred  (np.arr):   test predictions (shape: ntest, 6, 6)
        stats     (dict):   statistical values utilised in error calculation (only train values)
        error_type (str):   'rse', 'nrse', ...
        save_path  (str):   location where plot will be stored.
        transf    (dict):   't': plotting normalised data, 'u': plotting original units data
        mask    (np.arr):   optional (ntest,) bool; if given, only these points are RSE-coloured, rest gray.
        suffix     (str):   filename tag appended to the saved plot (e.g. '_t100').

    Returns:
        diagonal plots as png in save_path
    """

    # Calculate relevant errors
    errors = calculate_errors(D_true.reshape((-1, 36)), D_pred.reshape((-1,36)),
                              stats, transf, id = 'De-NLRC')


    # Define figures
    fig, ax = plt.subplots(6, 6, figsize=[25, 25], dpi=100)
    fig.subplots_adjust(wspace=0.4)

    errors_ = {key: arr.reshape(-1, 6, 6) for key, arr in errors.items()}
    norms = add_colorbars_D(fig, ax, errors_, transf, error_type, mask = mask)

    # Plot figures
    scatters = plot_main_figure(ax, D_true, D_pred, errors_, error_type, norms, mask = mask)

    # Formatting figures
    figure_formatting_D(ax, transf, D_true, D_pred)
    add_description_box(ax, errors_)
    for ax_ in ax.ravel():
        ax_.set_aspect('equal', 'box')
        ax_.axis('square')

    # Save figure
    return save_diag_figure(save_path, transf, id = 'D', suffix = suffix)



def figure_formatting_D(ax, transf: str, D_true:np.array, D_pred:np.array):
    """
    Sets up formatting and labels for axes

    Args: 
        ax      ():         axes of figure
        transf  (str):      type to plot (t: normalised or u: in original units)
        D_true  (np.arr):   labels (for limit calculation), shape: (ntest, 2, 3)
        D_pred  (np.arr):   predictions (for limit calculation), shape (ntest, 2, 3)
    
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })
    
    
    plotname = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$'],
                        ['$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$'],
                        ['$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$'],
                        ['$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$'],
                        ['$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$'],
                        ['$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$'],
                        ])
    plotname_p = np.array([[r'$\tilde{D}_{m,11}$', r'$\tilde{D}_{m,12}$', r'$\tilde{D}_{m,13}$', r'$\tilde{D}_{mb,11}$', r'$\tilde{D}_{mb,12}$', r'$\tilde{D}_{mb,13}$'],
                        [r'$\tilde{D}_{m,21}$', r'$\tilde{D}_{m,22}$', r'$\tilde{D}_{m,23}$', r'$\tilde{D}_{mb,21}$', r'$\tilde{D}_{mb,22}$', r'$\tilde{D}_{mb,23}$'],
                        [r'$\tilde{D}_{m,31}$', r'$\tilde{D}_{m,32}$', r'$\tilde{D}_{m,33}$', r'$\tilde{D}_{mb,31}$', r'$\tilde{D}_{mb,32}$', r'$\tilde{D}_{mb,33}$'],
                        [r'$\tilde{D}_{bm,11}$', r'$\tilde{D}_{bm,12}$', r'$\tilde{D}_{bm,13}$', r'$\tilde{D}_{b,11}$', r'$\tilde{D}_{b,12}$', r'$\tilde{D}_{b,13}$'],
                        [r'$\tilde{D}_{bm,21}$', r'$\tilde{D}_{bm,22}$', r'$\tilde{D}_{bm,23}$', r'$\tilde{D}_{b,21}$', r'$\tilde{D}_{b,22}$', r'$\tilde{D}_{b,23}$'],
                        [r'$\tilde{D}_{bm,31}$', r'$\tilde{D}_{bm,32}$', r'$\tilde{D}_{bm,33}$', r'$\tilde{D}_{b,31}$', r'$\tilde{D}_{b,32}$', r'$\tilde{D}_{b,33}$'],
                        ])

    if transf == 't':
        units = np.array([['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,
                          ['$[-]$']*6,                          
                          ])
    elif transf =='u':
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          ])

    
    for i in range(6):
        for j in range(6):
            ax[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j], labelpad = 2)
            ax[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j], labelpad = 2)
            ax[i,j].set_xlim([np.min([np.min(D_true[:,i,j]), np.min(D_pred[:,i,j])]), 
                              np.max([np.max(D_true[:,i,j]), np.max(D_pred[:,i,j])])])
            ax[i,j].set_ylim([np.min([np.min(D_true[:,i,j]), np.min(D_pred[:,i,j])]), 
                              np.max([np.max(D_true[:,i,j]), np.max(D_pred[:,i,j])])])
            ax[i,j].grid(True, which='major', color='#666666', linestyle='-')

    return

def add_colorbars_D(fig, axs, errors: dict, transf: str, error_type: str, mask = None):
    """
    Create four colorbars corresponding to the 4 submatrices of D (Dm, Dmb, Dbm and Db)

    fig         ():     Figure
    axs         ():     Set of axes
    errors      (dict): Contains all calculated errors
    transf      (str):  Type of plot (t: transformed units, u: original units)
    error_type  (str):  e.g. 'rse'
    mask     (np.arr):  optional (ntest,) bool; if given, colour scale is set from these points only.

    """

    # Define blocks for which colorbars are visualised.
    block_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    err = errors[error_type][mask] if mask is not None else errors[error_type]
    norms = []

    for bx, by in block_positions:
        block_errors = np.concatenate([
            err[:, b0, b1].flatten()
            for b0 in range(bx, bx + 3) for b1 in range(by, by + 3)
            ])
        norms.append(mcolors.Normalize(vmin=np.min(block_errors), vmax=np.max(block_errors)))



    # Define 3x3 blocks for colorbars
    if transf == 'u':
        colorbar_labels = {"rse": ["RSE [N/mm]", "RSE [N]", "RSE [N]", "RSE [Nmm]"], "nrse":  ["nRSE [%]", "nRSE [%]", "nRSE [%]", "nRSE [%]"]}
    else: 
        colorbar_labels = {"rse": ["RSE [-]", "RSE [-]", "RSE [-]", "RSE [-]"], "nrse":  ["nRSE [%]", "nRSE [%]", "nRSE [%]", "nRSE [%]"]}
    colorbar_label = colorbar_labels.get(error_type, "Error Scale")

    for norm, (bx, by), label in zip(norms, block_positions, colorbar_label):
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[bx:bx+3, by:by+3], orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(label, fontsize=12)

    return norms


