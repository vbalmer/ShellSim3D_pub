"""Helpers of the solver (Main_vb_vec.py): convergence measures per iteration, convergence plots and the arrays of the
intermediate solutions. V. Balmer, 2026."""

import os

import numpy as np
import matplotlib.pyplot as plt


################################## convergence plots ##################################

_CONV_HIST = {}             # convergence history of the current load level, reset at iteration 0


def plot_convergence(i, mat_un_thn, numit, mat_e_r, save_dir, NN = 'NN'):
    """
    Convergence plot after iteration i: relative strain and displacement / rotation changes (percentiles) and the
    residual forces / moments of all iterations of the current load level so far. Saved as conv_plt_<i><NN>.png.

    Args:
        i           (int):  iteration (0 resets the history)
        mat_un_thn  (dict): displacement / rotation convergence values (convergence_values_un_thn)
        numit       (int):  maximum number of iterations (x-axis)
        mat_e_r     (dict): strain and residual convergence values (convergence_values_eps_r)
        save_dir    (str):  folder of the figure
        NN          (str):  tag of the run ('NN' or 'NLFEA')
    """
    rele = mat_e_r['rele']
    convrf = mat_e_r['convrf']
    convrm = mat_e_r['convrm']
    relthn = mat_un_thn['relthn']
    relun = mat_un_thn['relun']

    h = _CONV_HIST
    new = {'it_steps': i + 2,
           'conve50': np.percentile(abs(rele), 50), 'conve90': np.percentile(abs(rele), 90),
           'conve99': np.percentile(abs(rele), 99),
           'convun50': np.percentile(abs(relun), 50), 'convun90': np.percentile(abs(relun), 90),
           'convun99': np.percentile(abs(relun), 99), 'convunmed': np.median(abs(relun)),
           'convthn50': np.percentile(abs(relthn), 50), 'convthn90': np.percentile(abs(relthn), 90),
           'convthn99': np.percentile(abs(relthn), 99),
           'convthnmed': np.median(abs(relun)) if i == 0 else np.median(abs(relthn))}
    for key, val in new.items():
        h[key] = [val] if i == 0 else np.append(h[key], val)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    it_steps = h['it_steps']

    if abs(min(min(h['conve50']), 0.001)) < 10**12 and max(h['conve50']) > 0 and max(h['conve50']) < 10**12:
        ax1.axis([2, numit + 1, min(min(h['conve50']), 0.001), max(h['conve99'])])
    ax1.set_yscale("log")
    ax1.set_title(r"$\delta$" + r"$\epsilon$$_{klij}$" + "/" + r"$\epsilon$$_{klij}$")
    ax1.plot(it_steps, h['conve50'], 'k')
    ax1.plot(it_steps, h['conve90'], 'b')
    ax1.plot(it_steps, h['conve99'], 'g')
    ax1.legend(["$P$$_{50}$", "$P$$_{90}$", "$P$$_{99}$"], loc=3)
    ax1.grid(True)

    if abs(min(min(h['convun50']), 0.001)) < 10**12 and max(h['convun50']) > 0 and max(h['convun50']) < 10**12:
        ax2.axis([2, numit + 1, min(min(h['convun50']), 0.001), max(h['convun99'])])
    ax2.set_yscale("log")
    ax2.set_title(r"$\delta$" + "$u$$_{n}$" + "/" + "$u$$_{n}$")
    ax2.plot(it_steps, h['convun50'], 'k')
    ax2.plot(it_steps, h['convthn50'], 'k--')
    ax2.plot(it_steps, h['convun90'], 'b')
    ax2.plot(it_steps, h['convunmed'], 'm')
    ax2.plot(it_steps, h['convthn90'], 'b--')
    ax2.plot(it_steps, h['convun99'], 'g')
    ax2.plot(it_steps, h['convthn99'], 'g--')
    ax2.plot(it_steps, h['convthnmed'], 'm--')
    ax2.legend(['Displacements', 'Rotations'], loc=3)
    ax2.grid(True)

    if abs(min(min(convrf), 0.001)) < 10**12 and max(convrm) > 0 and max(convrm) < 10**12:
        ax3.axis([2, numit + 1, min(min(convrf), 0.001), max(abs(convrm))])
    ax3.set_yscale("log")
    ax3.set_title("$Residual$")
    ax3.plot(it_steps, convrf, 'r')
    ax3.plot(it_steps, convrm, 'm')
    ax3.legend(["$R$$_{F}$", "$R$$_{M}$"], loc=3)
    ax3.grid(True)

    ax1.set_ylim(1e-15, 10)
    ax2.set_ylim(1e-15, 10)
    ax3.set_ylim(1e-5, 1e8)

    plt.subplots_adjust(hspace=0.5)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'conv_plt_' + str(i) + NN))
    plt.close(fig)


################################## convergence values ##################################

def un_thn(u):
    """
    Split the global displacement vector (6 DOF per node) into displacements and rotations.

    Args:
        u   (np.arr): global displacement vector, shape (6 * n_nodes, 1)

    Returns:
        tuple: un (3 * n_nodes,) displacements, thn (3 * n_nodes,) rotations
    """
    un = np.array([])
    thn = np.array([])
    count = 0
    for j in range(len(u)):
        if count < 2.5:
            un = np.append(un,u[j],axis=0)
        else:
            thn = np.append(thn,u[j],axis=0)
        count += 1
        if count > 5.5:
            count = 0
    return un,thn


def convergence_values_un_thn(u, unold, thnold):
    """
    Relative change of the nodal displacements / rotations between two iterations. Only values with
    |un| > max|un| / 1000 (resp. thn) are tracked, the others are set to zero.

    Args:
        u       (np.arr): global displacement vector of the current iteration
        unold   (np.arr): displacements of the previous iteration
        thnold  (np.arr): rotations of the previous iteration

    Returns:
        dict: un, thn, diffun, relun, diffthn, relthn and their percentiles / medians (convun50, ..., convthn_med)
    """
    [un, thn] = un_thn(u)
    diffun = np.zeros_like(un)
    relun = np.zeros_like(un)
    maxun = np.max(abs(un)) / 1000
    diffun[np.where(abs(un) > maxun)] = np.ndarray.flatten(unold[np.where(abs(un) > maxun)]) - np.ndarray.flatten(
        un[np.where(abs(un) > maxun)])
    relun[np.where(abs(un) > maxun)] = np.divide(diffun[np.where(abs(un) > maxun)],
                                                np.ndarray.flatten(un[np.where(abs(un) > maxun)]))
    diffun[np.where(abs(un) < maxun)] = 0
    relun[np.where(abs(un) < maxun)] = 0

    diffthn = np.zeros_like(un)
    relthn = np.zeros_like(un)
    maxthn = np.max(abs(thn)) / 1000
    diffthn[np.where(abs(thn) > maxthn)] = np.ndarray.flatten(thnold[np.where(abs(thn) > maxthn)]) - np.ndarray.flatten(
        thn[np.where(abs(thn) > maxthn)])
    relthn[np.where(abs(thn) > maxthn)] = np.divide(diffthn[np.where(abs(thn) > maxthn)],
                                                    np.ndarray.flatten(thn[np.where(abs(thn) > maxthn)]))
    diffthn[np.where(abs(thn) < maxthn)] = 0
    relthn[np.where(abs(thn) < maxthn)] = 0

    return {
        'un': un,
        'thn': thn,
        'diffun': diffun,
        'relun': relun,
        'diffthn': diffthn,
        'relthn': relthn,
        'convun50': np.percentile(abs(relun), 50),
        'convun90': np.percentile(abs(relun), 90),
        'convun99': np.percentile(abs(relun), 99),
        'convthn50': np.percentile(abs(relthn), 50),
        'convthn90': np.percentile(abs(relthn), 90),
        'convthn99': np.percentile(abs(relthn), 99),
        'convun_med': np.median(abs(relun)),
        'convthn_med': np.median(abs(relthn)),
    }


def convergence_values_eps_r(e_conv, eold, r, mat_convergence_eps_r, i):
    """
    Relative change of the layer strains between two iterations (only |e| > max|e| / 1000) and the sums of the
    absolute residual forces / moments, appended to the values of the previous iterations.

    Args:
        e_conv                  (np.arr): layer strains of the current iteration (triangle sentinels set to 0)
        eold                    (np.arr): layer strains of the previous iteration
        r                       (np.arr): residual vector (constrained DOFs set to 0)
        mat_convergence_eps_r   (dict):   values of the previous iterations (None for i = 0)
        i                       (int):    iteration

    Returns:
        dict: e, diffe, rele, convrf / convrm (sum of |residual| forces / moments per iteration), convr_comp
              (per component n_x ... m_xy, shape (6, n_iter)), conve50 / 90 / 99
    """
    e_conv[e_conv<-99999] = 0
    maxe = np.max(abs(e_conv))/1000
    diffe = np.ndarray.flatten(eold[np.where(abs(e_conv)>maxe)])-np.ndarray.flatten(e_conv[np.where(abs(e_conv)>maxe)])
    rele = np.divide(diffe,np.ndarray.flatten(e_conv[np.where(abs(e_conv)>maxe)]))

    # per-component residuals (nx, ny, nxy, mx, my, mxy), shape (6, n_iter)
    res_comp_i = np.array([np.sum(np.abs(r[k::6])) for k in range(6)]).reshape((6, 1))

    if i ==0:
        convrf = sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6]))
        convrm = sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6]))
        convr_comp = res_comp_i
    else:
        convrm_ = mat_convergence_eps_r['convrm']
        convrf_ = mat_convergence_eps_r['convrf']
        convrf = np.append(convrf_, sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6])))
        convrm = np.append(convrm_, sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6])))
        convr_comp = np.append(mat_convergence_eps_r['convr_comp'], res_comp_i, axis=1)

    return {
        'e': e_conv,
        'diffe': diffe,
        'rele': rele,
        'convrf': convrf,
        'convrm': convrm,
        'convr_comp': convr_comp,
        'conve50': np.percentile(abs(rele), 50),
        'conve90': np.percentile(abs(rele), 90),
        'conve99': np.percentile(abs(rele), 99),
    }


################################## intermediate solutions ##################################

def store_intermediate_solutions(numit, NN_hybrid, sh, sh_, eh, eh_, u, u_, De_tot, De_tot_):
    """
    Arrays for the solutions of all iterations of a load level, row 0 = linear elastic solution.

    Args:
        numit       (int):      maximum number of iterations
        NN_hybrid   (dict):     solver settings (which values the NN predicts)
        sh, sh_     (np.arr):   generalised stresses (run / NLFEA control)
        eh, eh_     (np.arr):   generalised strains (run / NLFEA control)
        u, u_       (np.arr):   displacements (run / NLFEA control)
        De_tot, De_tot_ (np.arr): tangents (run / NLFEA control)

    Returns:
        tuple: sh_cum, sh_cum_, eh_cum, eh_cum_, u_cum, u_cum_, De_cum, De_cum_, fi_cum, fi_cum_ (None for the
               control arrays that are not used)
    """
    sh_cum = np.zeros((numit+1, *sh.shape))
    eh_cum = np.zeros((numit+1, *eh.shape))
    u_cum = np.zeros((numit+1, *u.shape))
    fi_cum = np.zeros((numit+1, *u.shape))
    De_cum = np.zeros((numit+1, *De_tot.shape))
    if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
        sh_cum_ = np.zeros((numit+1, *sh_.shape))
        eh_cum_ = np.zeros((numit+1, *eh_.shape))
        fi_cum_ = np.zeros((numit+1, *u.shape))
        De_cum_ = None
        u_cum_ = None
    elif not NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
        eh_cum_ = None
        sh_cum_ = None
        u_cum_ = None
        fi_cum_ = None
        De_cum_ = None
    elif not NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        u_cum_ = np.zeros((numit+1, *u.shape))
        fi_cum_ = np.zeros((numit+1, *u.shape))
        De_cum_ = np.zeros((numit+1, *De_tot_.shape))
        eh_cum_ = np.zeros((numit+1, *eh.shape))
        sh_cum_ = None
    elif NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        sh_cum_ = np.zeros((numit+1, *sh_.shape))
        eh_cum_ = np.zeros((numit+1, *eh_.shape))
        u_cum_ = np.zeros((numit+1, *u.shape))
        fi_cum_ = np.zeros((numit+1, *u.shape))
        De_cum_ = np.zeros((numit+1, *De_tot_.shape))

    sh_cum[0,:,:,:,:] = sh
    eh_cum[0,:,:,:,:] = eh
    u_cum[0,:,:,] = u
    De_cum[0,:,:,:,:,:] = De_tot
    if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
        sh_cum_[0,:,:,:,:] = sh_
        eh_cum_[0,:,:,:,:] = eh_
    if not NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        u_cum_[0,:,:,] = u_
        De_cum_[0,:,:,:,:,:] = De_tot_
        eh_cum_[0,:,:,:,:] = eh_
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        sh_cum_[0,:,:,:,:] = sh_
        eh_cum_[0,:,:,:,:] = eh_
        u_cum_[0,:,:,] = u_
        De_cum_[0,:,:,:,:,:] = De_tot_

    return sh_cum, sh_cum_, eh_cum, eh_cum_, u_cum, u_cum_, De_cum, De_cum_, fi_cum, fi_cum_
