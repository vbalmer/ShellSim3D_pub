"""Nonlinear solver (vectorised): Newton-Raphson iteration over the load levels, either with the NLFEA (layered
constitutive laws) or with the NN (stresses and / or tangents predicted by the trained model). V. Balmer, 2026.

Each load level starts from the converged state of the previous one (or from the linear elastic solution if the
previous level did not converge) and is saved to <out_dir>/<load level>/mat_res_<tag>.pkl, tag = 'NN' (NN stresses
and tangents) or 'norm' (NLFEA), together with the convergence plots (<out_dir>/<load level>/conv_plots).
"""

import os
import time
import pickle

import numpy as np

from .Mesh_gmsh_vb import input_definition, SCENARIO_FB
from .fem_vb import fem_func, time_K, time_B, time_Kinv, time_sh, time_eh
from .Postprocess_vb import post_func
from .main_utils_vb import (plot_convergence, un_thn, convergence_values_un_thn, convergence_values_eps_r,
                            store_intermediate_solutions)


def result_tag(NN_hybrid):
    """File tag of a run: 'NN' (stresses and tangents from the NN), 'NN_sig', 'NN_D' or 'norm' (NLFEA)."""
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
        return 'NN'
    if NN_hybrid['predict_sig']:
        return 'NN_sig'
    if NN_hybrid['predict_D']:
        return 'NN_D'
    return 'norm'


def load_level_label(mat, F):
    """
    Load level entered by the user, recovered from the applied force: F = level * L * B for the frame bridge (area
    load), F = level * L otherwise. Integral levels are returned as int (folder names such as '141', '25000').
    """
    scenario = int(np.array(mat["s"]).flatten()[0])
    load_norm = mat["L"] * mat["B"] if scenario == SCENARIO_FB else mat["L"]
    lvl = float(np.ravel(np.asarray(F) / load_norm)[0])
    return int(round(lvl)) if abs(lvl - round(lvl)) < 1e-9 else round(lvl, 6)


def main_solver(mat: dict, NN_hybrid: dict, out_dir: str, nn_model=None, conv_plots: bool = True):
    '''
    Solve all load levels of mat['F'] with the NLFEA or the NN.

    Args:
        mat         (dict):         model input (geometry, scenario, load levels 'F', material), see load_path_utils
        NN_hybrid   (dict):         solver settings: predict_sig, predict_D (NN or NLFEA), numit, res_tol, gauss_order,
                                    symm_predict, nlfea_control, opt_time, model_dim
        out_dir     (str):          output folder, one subfolder per load level
        nn_model    (NNPredictor):  trained model (only for the NN run)
        conv_plots  (bool):         save the convergence plot of every iteration

    Returns:
        dict: mat_res of the last load level (also saved per load level)
    '''
    start_main = time.time()
    opt_time = NN_hybrid.get('opt_time', False)
    tag = result_tag(NN_hybrid)
    run_name = 'NN' if tag != 'norm' else 'NLFEA'

    # -------------------------------------------------------------------------------------------------------------------- #
    # 1 Geometry definition
    # -------------------------------------------------------------------------------------------------------------------- #
    print("-------------------------------------------------------")
    print("1 Geometry definition")
    print("-------------------------------------------------------")

    MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln = input_definition(mat, NN_hybrid)
    if MATK['cm'][0] != 3 or it_type != 1:
        raise NotImplementedError('Only nonlinear RC (mat = 3) with the tangent stiffness iteration is implemented.')
    nn_model = nn_model if tag != 'norm' else None
    fem_func0 = fem_func(MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln, nn_model)
    fem_func1 = fem_func(MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln, nn_model)

    # -------------------------------------------------------------------------------------------------------------------- #
    # 2 Initialisation
    # -------------------------------------------------------------------------------------------------------------------- #
    print("-------------------------------------------------------")
    print("2 Initialisation")
    print("-------------------------------------------------------")

    # 2.1 External forces, residual strains (none), forces from internal stresses (none) and constrained DOFs
    print("2.1 Assembly of force vector and condensed DOFs")
    fe = fem_func0.f_assemble(0, Load_el, Load_n)
    B = fem_func0.find_b(go=gauss_order)
    [e0, ex0, ey0, gxy0, e10, e30, th0] = fem_func0.find_e0_vec(go=gauss_order)
    [e0c, ex0c, ey0c, gxy0c, e10c, e30c, th0c] = fem_func0.find_e0_vec(go=1)
    s0 = fem_func0.find_s0_vec(go=gauss_order)
    [sh0,f0] = fem_func0.f0_assemble(B,s0,1)
    cDOF,cVAL = fem_func0.c_dof()

    # 2.2 Linear elastic solution as the start of the iteration
    e, eold, eh, eh_, s, s_prev, sh, sh_, r, u, u_, unold, thnold, De_tot, De_tot_ = linear_elastic_solution(fem_func0, fem_func1, B, fe, f0, e0, cDOF, cVAL, NN_hybrid, gauss_order)

    numit = NN_hybrid['numit']
    sh_cum, sh_cum_, eh_cum, eh_cum_, u_cum, u_cum_, De_cum, De_cum_, fi_cum, fi_cum_ = store_intermediate_solutions(numit, NN_hybrid, sh, sh_, eh, eh_, u, u_, De_tot, De_tot_)

    # NLFEA control in NN mode: per iteration, also solve the "true" NLFEA system (u_, De_tot_) and evaluate the
    # NLFEA stresses (s, sh_) at the NN state -- only for the comparison arrays u_cum_, sh_cum_, eh_cum_, De_cum_.
    # Off: these are skipped (saved as None); the strain convergence measure uses the layer strains e of the NN
    # displacement field.
    nn_mode = NN_hybrid['predict_sig'] or NN_hybrid['predict_D']
    nlfea_control = NN_hybrid.get('nlfea_control', True) or not nn_mode
    if not nlfea_control:
        sh_cum_, eh_cum_, u_cum_, De_cum_ = None, None, None, None
    r_cum = np.zeros_like(u_cum)   # residual vector per iteration

    # -------------------------------------------------------------------------------------------------------------------- #
    # 3 Nonlinear solution: tangent stiffness iteration per load level
    # -------------------------------------------------------------------------------------------------------------------- #
    print("-------------------------------------------------------")
    print("3 Solution")
    print("-------------------------------------------------------")
    print("3.2 Tangent Stiffness Iteration")

    F_all = np.atleast_1d(mat['F'])
    mat_convergence_eps_r = None
    for j in range(len(Load_n)):
        level = load_level_label(mat, F_all[j])
        level_dir = os.path.join(out_dir, str(level))
        print('----------------------------------------------------------------------------')
        print(f'Started iteration for load step {j}, with load level {level}')
        fe = fem_func0.f_assemble(j, Load_el, Load_n)
        r_cum = np.zeros_like(u_cum)   # fresh per load step
        for i in range(numit):
            start_time_iteration = time.perf_counter()

            # 3.2.1 Solution with the tangent stiffness of the current state
            fi = fem_func0.find_fi(B, sh)
            fi_ = fi.copy()
            if not NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
                fi_cum[i,:,:] = fi
            elif NN_hybrid['predict_D']:
                fi_cum[i,:,:] = fi
                fi_cum_[i,:,:] = fi_

            if NN_hybrid['predict_D']:
                du, De_tot=fem_func0.solve_sys_nn_num_vec(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],eh,sh, e, s, NN_hybrid['model_dim'], scenario = mat["s"], symm_predict=NN_hybrid.get('symm_predict', False), opt_time=opt_time)
                u -= du
                if nlfea_control:
                    du_, De_tot_=fem_func1.solve_sys_vec(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s)
                    u_ -= du_
            else:
                du, De_tot=fem_func0.solve_sys_vec(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s, opt_time=opt_time)
                u -= du
                u_, De_tot_ = u.copy(), De_tot.copy()      # no prediction of D --> u is real u, D is real D

            if NN_hybrid['predict_sig']:
                eh = fem_func0.find_eh_vec(B,u, gauss_order)
                sh = fem_func0.find_sh_nn_vec(eh, gauss_order, NN_hybrid['model_dim'], symm_predict=NN_hybrid.get('symm_predict', False), opt_time=opt_time)
                if nlfea_control:
                    # "real" strains and stresses (NLFEA control at the NN state)
                    eh_ = fem_func1.find_eh_vec(B,u, gauss_order)
                    [e, ex, ey, gxy, e1, e3, th] = fem_func1.find_e_vec(e0, eh_, gauss_order)
                    s = fem_func1.find_s_vec(e,s_prev,gauss_order)
                    sh_ = fem_func1.find_sh_vec(s, gauss_order)
                else:
                    # layer strains of the NN state only (for the strain convergence measure), no NLFEA stresses
                    [e, ex, ey, gxy, e1, e3, th] = fem_func1.find_e_vec(e0, eh, gauss_order)
            else:
                eh = fem_func0.find_eh_vec(B,u, gauss_order)
                eh_ = fem_func0.find_eh_vec(B,u_, gauss_order)
                [e, ex, ey, gxy, e1, e3, th] = fem_func0.find_e_vec(e0, eh, gauss_order)
                s = fem_func0.find_s_vec(e,s_prev,gauss_order)
                sh = fem_func0.find_sh_vec(s, gauss_order)

            # 3.2.2 Convergence control: residual vector r = fi - fe (without the constrained DOFs: rcond)
            r = np.add(fi, -(fe+f0))
            rcond = fem_func0.v_stat_con(r, cDOF,np.zeros_like(cVAL))
            print(" - Iteration step " + str(i) + " complete, maximum residual = " + str(np.round(np.max(abs(rcond)),1)))

            # 3.2.3 Values of this iteration
            sh_cum[i+1,:,:,:,:] = sh
            eh_cum[i+1,:,:,:,:] = eh
            u_cum[i+1,:,:,] = u
            De_cum[i+1,:,:,:,:,:] = De_tot
            if not nlfea_control:
                pass                                                        # no NLFEA control arrays
            elif NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
                sh_cum_[i+1,:,:,:,:] = sh_
                eh_cum_[i+1,:,:,:,:] = eh_
            elif NN_hybrid['predict_D'] and not NN_hybrid['predict_sig']:
                u_cum_[i+1,:,:] = u_
                De_cum_[i+1,:,:,:,:,:] = De_tot_
                eh_cum_[i+1,:,:,:,:] = eh_
            elif NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
                u_cum_[i+1,:,:] = u_
                De_cum_[i+1,:,:,:,:,:] = De_tot_
                eh_cum_[i+1,:,:,:,:] = eh_
                sh_cum_[i+1,:,:,:,:] = sh_

            mat_conv = None if i == 0 else mat_convergence_eps_r
            r[cDOF.astype(int)] = 0
            mat_convergence_eps_r = convergence_values_eps_r(e_conv=e, eold=eold, r=r, mat_convergence_eps_r= mat_conv, i=i)
            r_cum[i+1,:,:] = r.reshape(r_cum[i+1].shape)
            mat_convergence_un_thn_nl = convergence_values_un_thn(u, unold, thnold)

            print(" - Iteration step " + str(i) + " complete, sum of residual forces = " + str(np.round(abs(mat_convergence_eps_r['convrf'][-1]), 1)))
            print(" - Iteration step " + str(i) + " complete, sum of residual moments = " + str(np.round(abs(mat_convergence_eps_r['convrm'][-1]), 1)))

            if conv_plots:
                plot_convergence(i, mat_convergence_un_thn_nl, numit, mat_convergence_eps_r,
                                 os.path.join(level_dir, 'conv_plots'), run_name)

            # values of this iteration as the old values for the next one
            if i < numit-1:
                eold = e
                [un, thn] = un_thn(u)
                unold = un
                thnold = thn
            s_prev = s

            # converged: max. residual < res_tol, also in the last iteration (then the next load level continues from
            # this state; only a level that does not converge restarts the next one from the linear elastic solution)
            res_tol = NN_hybrid.get('res_tol', 5)  # max-residual convergence threshold [N/mm]
            converged = np.round(np.max(abs(rcond)),1) < res_tol
            if converged or i == numit-1:
                if converged:
                    print(f'Stopped iteration after {i} steps, because the results already converged.')
                else:
                    print(f'Stopped iteration after {i} steps, because maximum iteration steps were reached. Solution did not converge for this load step.')
                mat_res = save_data_inter_loop(NN_hybrid, eold, fem_func0, B, u, gauss_order, e0c, sh0, s_prev, MATK, COORD, GEOMK, ELS, GEOMA, BC, MASK, NODESG,
                                                e0, e, s, fi, fe, f0, cDOF, start_main, mat_convergence_un_thn_nl, na, mat,
                                                De_tot, u_cum, u_cum_, sh_cum, sh_cum_, eh_cum, eh_cum_, De_cum, De_cum_,
                                                mat_convergence_eps_r=mat_convergence_eps_r, r_cum=r_cum)
                os.makedirs(level_dir, exist_ok=True)
                with open(os.path.join(level_dir, f'mat_res_{tag}.pkl'), 'wb') as f:
                    pickle.dump(mat_res, f)
                if not converged:
                    print('Redoing lin.el. calculation to start next load step afresh.')
                    e, eold, eh, eh_, s, s_prev, sh, sh_, r, u, u_, unold, thnold, De_tot, De_tot_ = linear_elastic_solution(fem_func0, fem_func1, B, fe, f0, e0, cDOF, cVAL, NN_hybrid, gauss_order)
                print(f'Time required for one iteration step: {(time.perf_counter()-start_time_iteration)/60:.2f} min.')
                break

            print(f'Time required for one iteration step: {(time.perf_counter()-start_time_iteration)/60:.2f} min.')

    print('Finished :)')
    return mat_res


def linear_elastic_solution(fem_func0, fem_func1, B, fe, f0, e0, cDOF, cVAL, NN_hybrid, gauss_order):
    """
    Linear elastic solution (start of the iteration) and the corresponding strains / stresses of the run.

    Returns:
        tuple: e, eold, eh, eh_, s, s_prev, sh, sh_, r, u, u_, unold, thnold, De_tot, De_tot_
    """
    print("2.2 Solution for Linear Elasticity")

    # The first (initialising) solve is always based on the linear elastic stiffness.
    u, De_tot = fem_func0.solve_0(B,fe+f0,np.zeros_like(e0),cDOF,cVAL, go=gauss_order)
    u_, De_tot_ = u.copy(), De_tot.copy()
    [un, thn] = un_thn(u)
    unold = un
    thnold = thn

    if NN_hybrid['predict_sig']:
        # stresses of the NN
        eh = fem_func0.find_eh_vec(B,u,gauss_order)
        sh = fem_func0.find_sh_nn_vec(eh, gauss_order, NN_hybrid['model_dim'], symm_predict=NN_hybrid.get('symm_predict', False), opt_time=NN_hybrid.get('opt_time', False))
        print('Gauss order for NN-run is: ', gauss_order)
        r = u
        # layer strains / stresses (strain convergence measure, NLFEA control)
        eh_ = fem_func1.find_eh_vec(B,u,gauss_order)
        [e,ex,ey,gxy,e1,e3,th] = fem_func1.find_e_vec(e0,eh_,gauss_order)
        eold = e
        if NN_hybrid.get('opt_time', False) and not NN_hybrid.get('nlfea_control', True):
            # NLFEA check stresses skipped: s / s_prev / sh_ are only consumed by the NLFEA control
            s, s_prev, sh_ = None, None, np.zeros_like(sh)
        else:
            s_lin = fem_func1.find_s_vec(e,0,gauss_order,True)
            s_prev = s_lin
            s = fem_func1.find_s_vec(e,s_prev,gauss_order)
            sh_ = fem_func1.find_sh_vec(s, gauss_order)

    else:
        # NLFEA: strains and stresses layer-wise
        eh = fem_func0.find_eh_vec(B,u,gauss_order)
        eh_ = fem_func0.find_eh_vec(B,u_,gauss_order)
        [e,ex,ey,gxy,e1,e3,th] = fem_func0.find_e_vec(e0,eh,gauss_order)
        s_lin = fem_func0.find_s_vec(e,0,gauss_order,True)
        s_prev = s_lin
        s = fem_func0.find_s_vec(e,s_prev,gauss_order)
        sh = fem_func0.find_sh_vec(s, gauss_order)
        sh_ = sh.copy()
        if NN_hybrid['predict_D']:
            print('Gauss order for NN-run is: ', gauss_order)
        else:
            print('Gauss order for NLFEA-run is: ', gauss_order)
        eold = e
        r = u

    return e, eold, eh, eh_, s, s_prev, sh, sh_, r, u, u_, unold, thnold, De_tot, De_tot_


def save_data_inter_loop(NN_hybrid, eold, fem_func0, B, u, gauss_order, e0c, sh0, s_prev, MATK, COORD, GEOMK, ELS, GEOMA, BC, MASK, NODESG,
                         e0, e, s, fi, fe, f0, cDOF, start_main, mat_convergence_un_thn_nl, na, mat,
                         De_tot, u_cum, u_cum_, sh_cum, sh_cum_, eh_cum, eh_cum_, De_cum, De_cum_,
                         mat_convergence_eps_r=None, r_cum=None):
    """
    Results of a load level: displacements, generalised strains / stresses at the element centres, steel stresses
    (NLFEA), postprocessed fields (POST), the intermediate solutions of all iterations and the model input.

    Returns:
        dict: mat_res (keys BC, COORD_c, COORD, ELEMENTS, MATK, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na,
              sig_g, eps_g, ux, uy, uz, thx, thy, thz, De_tot, u_cum(_), sh_cum(_), eh_cum(_), De_cum(_), ssx, ssy,
              convrf, convrm, convr_comp, r_cum and the entries of mat)
    """
    # 3.3 Relative change of the layer strains in the last iteration
    diffe = eold - e
    rele = np.divide(diffe,e)

    # 3.4 Displacements, generalised strains / stresses at the element centres
    ux = u[0::6]
    uy = u[1::6]
    uz = u[2::6]
    thx = u[3::6]
    thy = u[4::6]
    thz = u[5::6]
    eh = fem_func0.find_eh_vec(B,u, gauss_order)

    Bc = fem_func0.find_b(1)
    ehc = fem_func0.find_eh_vec(Bc,u,1)

    if NN_hybrid['predict_sig']:
        sh = fem_func0.find_sh_nn_vec(ehc,1, NN_hybrid['model_dim'], symm_predict=NN_hybrid.get('symm_predict', False), opt_time=NN_hybrid.get('opt_time', False))+sh0
    else:
        # same vectorised material law as the solver loop, so the saved sig_g / POST stress resultants are
        # consistent with the equilibrium state
        ec = fem_func0.find_e_vec(e0c,ehc,1)[0]
        sc = fem_func0.find_s_vec(ec,s_prev,1)
        sh = fem_func0.find_sh_vec(sc,1)+sh0

    if NN_hybrid['predict_sig'] or NN_hybrid['predict_D']:
        # NN run: no layer strains / steel stresses
        [e, ex, ey, gxy, e1, e3, th] = [0, 0, 0, 0, 0, 0, 0]
        ssx = np.zeros_like(eh)
        ssy = np.zeros_like(eh)
        spx = np.zeros_like(eh)
        spy = np.zeros_like(eh)
    else:
        [e, ex, ey, gxy, e1, e3, th] = fem_func0.find_e_vec(np.zeros_like(e0), eh, gauss_order)
        [ssx, ssy, spx, spy] = fem_func0.find_ss_vec(e, MATK["cm"])

    Nx = sh[:, :, :, 0]
    Ny = sh[:, :, :, 1]
    Nxy = sh[:, :, :, 2]
    Mx = sh[:, :, :, 3]
    My = sh[:, :, :, 4]
    Mxy = sh[:, :, :, 5]
    Qx = sh[:, :, :, 6]
    Qy = sh[:, :, :, 7]

    # 3.5 Displacements, applied loads and reactions
    print("2.4 Solution complete")
    r = np.add(fi, -(fe+f0))

    print(" - Maximum displacements:")
    print("   - ux_max = " + str(np.round(float(max(np.abs(ux))),3)))
    print("   - uy_max = " + str(np.round(float(max(np.abs(uy))),3)))
    print("   - uz_max = " + str(np.round(float(max(np.abs(uz))),3)))
    print(" - Sum of applied forces:")
    print("   - Fx = " + str(np.round(float(sum(fe[0::6])),3)))
    print("   - Fy = " + str(np.round(float(sum(fe[1::6])),3)))
    print("   - Fz = " + str(np.round(float(sum(fe[2::6])),3)))
    print(" - Sum of reaction forces:")
    print("   - Rx = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==0].astype(int)])),1)))
    print("   - Ry = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==1].astype(int)])),1)))
    print("   - Rz = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==2].astype(int)])),1)))

    # 4.1 Time management
    print("total time used: ", time.time()-start_main)
    print("time used in stiffness matrix calculation", time_K.time_spent)
    print("time used in B-matrix calculation", time_B.time_spent)
    print("time used in stiffness matrix inversion", time_Kinv.time_spent)
    print("time used in calculation of sh", time_sh.time_spent)
    print("time used in calculation of eh", time_eh.time_spent)

    # 4.2 Postprocessing
    print("-------------------------------------------------------")
    print("3 Postprocess Data")
    print("-------------------------------------------------------")
    post_func0 = post_func(COORD, GEOMK, ELS, GEOMA, NN_hybrid)
    relun, relthn = mat_convergence_un_thn_nl['relun'], mat_convergence_un_thn_nl['relthn']
    POST = post_func0.post(ux,uy,uz,r,thx,thy,thz,Nx,Ny,Nxy,Mx,My,Mxy,Qx,Qy,ex,ey,gxy,e3,e1,th,ssx,ssy,spx,spy,rele,relun,relthn)

    # 4.3 Results
    mat_res = {
        'BC': BC,
        'COORD_c': COORD["c"],
        'COORD': COORD,
        'ELEMENTS': ELS[0],
        'MATK': MATK,
        'fe': fe,
        'GEOMA': GEOMA,
        'GEOMK': GEOMK,
        'MASK': MASK,
        'NODESG': NODESG,
        'POST': POST,
        'gauss_order': gauss_order,
        'na': na,
        'sig_g': sh,
        'eps_g': ehc,
        'ux': ux,
        'uy': uy,
        'uz': uz,
        'thx': thx,
        'thy': thy,
        'thz': thz,
        'De_tot': De_tot,
        'u_cum': u_cum,
        'u_cum_': u_cum_,
        'sh_cum': sh_cum,
        'sh_cum_': sh_cum_,
        'eh_cum': eh_cum,
        'eh_cum_': eh_cum_,
        'De_cum': De_cum,
        'De_cum_': De_cum_,
        'ssx': ssx,
        'ssy': ssy,
        }

    if mat_convergence_eps_r is not None:
        mat_res['convrf'] = mat_convergence_eps_r['convrf']
        mat_res['convrm'] = mat_convergence_eps_r['convrm']
        mat_res['convr_comp'] = mat_convergence_eps_r['convr_comp']
    if r_cum is not None:
        mat_res['r_cum'] = r_cum
    mat_res.update(mat)

    return mat_res
