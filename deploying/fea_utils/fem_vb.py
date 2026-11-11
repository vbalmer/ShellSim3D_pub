"""Layered shell finite elements (MITC-type quads / triangles, 6 DOF per node) for the NLFEA and the NN-based FEA.

Vectorised implementation: the layer stresses come from the vectorised constitutive laws of the sampling
(sampling/utils/constitutive_laws.py, complex step for the tangent), the layer strains and the thickness integration
of the stresses / tangents from the section model of the sampling (SigSimulator in
sampling/utils/simulating_sig_vec_RC3D.py; the transverse shear is added here), the generalised stresses / tangents
of the NN from the trained model (nn_prediction.py). Based on the shell code of A. Näsbom. V. Balmer, 2026.

Generalised strains eh / stresses sh per Gauss point: [eps_x, eps_y, gamma_xy, chi_x, chi_y, chi_xy, gamma_xz,
gamma_yz] / [n_x, n_y, n_xy, m_x, m_y, m_xy, q_x, q_y].
"""

import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from sampling.utils.constitutive_laws import ConstitutiveLaws
from sampling.utils.simulating_sig_vec_RC3D import SigSimulator
from .gauss_points import gauss_points
from .nn_prediction import make_NN_prediction, n_geom_from_model


"""---------------------------------------------- Time Measurement---------------------------------------------------"""
class gettime:
    def _updatetime(self,update=True,delta_t=0):
        if update:
            self.time_spent += delta_t
        else:
            self.time_spent = 0
time_K = gettime()
time_K._updatetime(update=False)
time_B = gettime()
time_B._updatetime(update=False)
time_Kinv = gettime()
time_Kinv._updatetime(update=False)
time_sh = gettime()
time_sh._updatetime(update=False)
time_eh = gettime()
time_eh._updatetime(update=False)


class fem_func():
    """
    Args:
        MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln:
                    mesh, material and load definition from Mesh_gmsh_vb.input_definition
        nn_model    (NNPredictor | None): trained model for the NN-based FEA (None: NLFEA only)
    """
    def __init__(self, MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln, nn_model=None):
        self.MATK = MATK
        self.NODESG = NODESG
        self.ELS = ELS
        self.COORD = COORD
        self.GEOMA = GEOMA
        self.GEOMK = GEOMK
        self.MASK = MASK
        self.na = na
        self.BC = BC
        self.gauss_order = gauss_order
        self.it_type = it_type
        self.Load_el = Load_el
        self.Load_n = Load_n
        self.copln = copln
        self.ELEMENTS = ELS[0]
        self.nn_model = nn_model
        self._cache = {}            # batched element data (mesh and B-matrices are constant during a run)

    """------------------------------------------------ Integration------------------------------------------------------"""

    def gauss_points(self, nn, go):
        """Gauss points and weights in one dimension (see fea_utils/gauss_points.py)."""
        return gauss_points(nn, go)


    """----------------------------------------- Batched element data ---------------------------------------------------"""

    ELEMENT_TYPES = (4, 3)          # quadrilaterals (24 DOF), triangles (18 DOF)

    def _element_dofs(self, n_nodes):
        """
        Elements with n_nodes nodes and their global DOF indices (6 per node, in the element node order).

        Args:
            n_nodes (int): 4 (quadrilaterals) or 3 (triangles)

        Returns:
            tuple: els (E,) element indices, dofs (E, 6 * n_nodes) global DOFs
        """
        key = ('dofs', n_nodes)
        if key not in self._cache:
            els = np.where(np.asarray(self.ELS[4]) == n_nodes)[0]
            nodes = np.asarray(self.ELEMENTS)[els, :n_nodes].astype(int)
            dofs = (nodes[:, :, None] * 6 + np.arange(6)[None, None, :]).reshape(len(els), 6 * n_nodes)
            self._cache[key] = (els, dofs)
        return self._cache[key]

    def _b_batch(self, B, n_nodes, go, kind='r'):
        """
        Stacked B-matrices [Bm; Bb; Bs] and integration weights of all elements with n_nodes nodes. Triangles have no
        Gauss point (1, 1): its B-matrix and weight are zero.

        Args:
            B       (dict): B-matrix container of find_b
            n_nodes (int):  4 or 3
            go      (int):  Gauss order (Gauss points 0 ... go-1 of B are used)
            kind    (str):  'r' (rotated to global coordinates) or 'nr' (local)

        Returns:
            tuple: B_batch (E, go, go, 8, 6 * n_nodes), weight (E, go, go) = w_i * w_j * Jdet
        """
        key = ('B', id(B), n_nodes, go, kind)
        hit = self._cache.get(key)
        if hit is None or hit[0] is not B:
            els, _ = self._element_dofs(n_nodes)
            ndof = 6 * n_nodes
            B_batch = np.zeros((len(els), go, go, 8, ndof))
            Jdet = np.zeros((len(els), go, go))
            for idx, k in enumerate(els):       # one-off gather from the nested dicts of find_b
                for i in range(go):
                    for j in range(go):
                        if n_nodes == 3 and i == 1 and j == 1:
                            continue
                        B_batch[idx, i, j] = np.vstack([B["Bm"][kind][k][i][j], B["Bb"][kind][k][i][j],
                                                        B["Bs"][kind][k][i][j]])
                        Jdet[idx, i, j] = B["Jdet"][k][i][j]
            _, w = self.gauss_points(n_nodes, go)
            w_grid = np.outer(w, w).copy()
            if n_nodes == 3 and go >= 2:
                w_grid[1, 1] = 0.0
            hit = (B, B_batch, w_grid[None, :, :] * Jdet)
            self._cache[key] = hit
        return hit[1], hit[2]

    def _drilling_batch(self, B, n_nodes, go):
        """
        Stiffness term for the rotational DOF about the element normal (coplanar elements, avoids the singularity of
        the drilling DOF): iscopl * A_k * t_k * 33600 * Tkr * 1e-8 per element.

        Returns:
            np.arr: (E, 6 * n_nodes, 6 * n_nodes)
        """
        key = ('drill', id(B), n_nodes, go)
        hit = self._cache.get(key)
        if hit is None or hit[0] is not B:
            els, _ = self._element_dofs(n_nodes)
            ndof = 6 * n_nodes
            _, weight = self._b_batch(B, n_nodes, go)
            A_k = weight.sum(axis=(1, 2))
            drill = np.zeros((len(els), ndof, ndof))
            for idx, k in enumerate(els):
                iscoplk = 1 if (self.ELEMENTS[k].any() in self.copln) else 0
                if iscoplk:
                    Tkr = self.rotLG(k)[1][:ndof, :ndof]
                    drill[idx] = iscoplk * A_k[idx] * self.GEOMK["t"][k] * 33600 * Tkr * 1e-8
            hit = (B, drill)
            self._cache[key] = hit
        return hit[1]

    def _rot_batch(self, n_nodes):
        """Rotations Tk (local -> global) of all elements with n_nodes nodes, shape (E, 6 * n_nodes, 6 * n_nodes)."""
        key = ('Tk', n_nodes)
        if key not in self._cache:
            els, _ = self._element_dofs(n_nodes)
            ndof = 6 * n_nodes
            self._cache[key] = np.array([self.rotLG(k)[0][:ndof, :ndof] for k in els]).reshape(len(els), ndof, ndof)
        return self._cache[key]

    def _section(self):
        """
        Section model of the sampling (SigSimulator) for the layer strains and the thickness integration. All elements
        must have the same number of layers; nu is set if it is the same in all elements (linear elastic tangent).
        """
        if 'section' not in self._cache:
            nlk = np.unique(np.asarray(self.GEOMK["nlk"], dtype=int))
            if nlk.size != 1:
                raise NotImplementedError(f'All elements need the same number of layers, got {nlk.tolist()}.')
            vc = np.unique(np.asarray(self.MATK["vc"], dtype=float))
            constants = {'n_layer': int(nlk[0]), 'nu': float(vc[0]) if vc.size == 1 else None}
            self._cache['section'] = SigSimulator(constants, verbose=False)
        return self._cache['section']

    def _samples(self, go):
        """
        Thickness per sample for the section model, samples = (element, Gauss point i, j) in the order of
        x.reshape(nel * go * go, ...) of an array (nel, go, go, ...).

        Returns:
            np.arr: t, shape (nel * go * go, 1)
        """
        return np.repeat(np.asarray(self.GEOMK["t"], dtype=float), go * go)[:, None]

    def _assemble_matrix(self, Ke_by_type):
        """Global matrix from the element matrices {n_nodes: (E, ndof, ndof)} (scatter with the element DOF map)."""
        n = 6 * len(self.COORD["n"][0][:, 0])
        K = np.zeros((n, n))
        for n_nodes, Ke in Ke_by_type.items():
            _, dofs = self._element_dofs(n_nodes)
            np.add.at(K, (dofs[:, :, None], dofs[:, None, :]), Ke)
        return K

    def _assemble_vector(self, fe_by_type):
        """Global column vector (6 * n_nodes, 1) from the element vectors {n_nodes: (E, ndof)}."""
        v = np.zeros(6 * len(self.COORD["n"][0][:, 0]))
        for n_nodes, fe in fe_by_type.items():
            _, dofs = self._element_dofs(n_nodes)
            np.add.at(v, dofs.ravel(), fe.ravel())
        return v[:, None]


    """----------------------------------------------- Material Laws-----------------------------------------------------"""


    """------------------------------------------------ B_Matrices-------------------------------------------------------"""

    def jacobi(self, k, i, j, go):
        """ -------------------------------------- Get Jacobian of Element  ------------------------------------------------
                ----------------------------------------------- INPUT: --------------------------------------------------
                - k: element number
                - i,j: Gauss point numbers in xi - and eta space
                - go: gauss order
                ---------------------------------------------- OUTPUT: --------------------------------------------------
                - xi,eta: Gauss points
                - J: Jacobian
                - J_inv: Inverse of jacobian
                - J_det: Determinant of jacobian
        -----------------------------------------------------------------------------------------------------------------"""
        # 1 Values of Importance

        # 1.1 Nodes of element k
        e_k = self.ELEMENTS[k, :]
        e_k = e_k[e_k<10**5]

        # 1.2 Area of element k
        a_k = self.GEOMK["ak"][k]

        # 1.3 Local coordinates of nodes of element k
        NODESL = self.COORD["n"][2][a_k]
        v = np.array(NODESL[e_k])

        # -----------------------------------------------------------------------------------------------------------------#
        # 2 Jacobian for quadrilaterals
        # -----------------------------------------------------------------------------------------------------------------#
        order = 1
        if len(e_k) == 4:
            if order == 1:

                # 2.1 Gauss points and weights
                gp, w = self.gauss_points(4,go)
                xi = gp[j]
                eta = gp[i]

                # 2.2 Derivatives of Shape Functions in xi-eta-space
                N1xi = float(- (1 - eta) / 4)
                N2xi = float((1 - eta) / 4)
                N3xi = float((1 + eta) / 4)
                N4xi = float(- (1 + eta) / 4)
                N1eta = float(- (1 - xi) / 4)
                N2eta = float(- (1 + xi) / 4)
                N3eta = float((1 + xi) / 4)
                N4eta = float((1 - xi) / 4)

                # 2.3 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])

                # 2.4 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)
        # -----------------------------------------------------------------------------------------------------------------#
        # 3 Jacobian of triangles
        # -----------------------------------------------------------------------------------------------------------------#
        else:
            if order == 1:

                # 3.1 Coordinates of element nodes: delete entry 100'001
                v = np.array(NODESL[self.ELEMENTS[k][0:3]])

                # 3.2 Gauss Points
                gp, w = self.gauss_points(3, go)
                xi = gp[j]
                eta = gp[i]

                # 3.3 Derivatives Shape Functions in xi-eta space
                N1xi = -1
                N2xi = 1
                N3xi = 0
                N1eta = -1
                N2eta = 0
                N3eta = 1

                # 3.4 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])

                # 3.5 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)

        return xi,eta,J,J_inv,J_det


    def b_kij(self, k, i, j, go,rot=True):
        """ ---------------------------------------- Create B-Matrices  -------------------------------------------------
            ----------------------------------------------- INPUT: ------------------------------------------------------
            - k: element number
            - i,j: Gauss point numbers in xi - and eta space
            - go: gauss order
            - rot: True, if B-Matrices shall be rotated in global coordinate system
            ---------------------------------------------- OUTPUT: ------------------------------------------------------
            - B_m: Membrane strain matrix
            - B_b: Bending strain matrix
            - B_s: Transverse shear strain matrix.
                - For quadrilaterals: locking-free element with prescribed transverse shear strain profile
                - For triangles: locking-free element with reduced integration of shear terms
            - J_det: Determinant of Jacobian
        -----------------------------------------------------------------------------------------------------------------"""

        # 0 Initiate Time Measurement
        start = time.time()

        # 1 Values of Importance

        # 1.1 Nodes of element k
        e_k = self.ELEMENTS[k, :]
        e_k = e_k[e_k<10**5]

        # 1.2 Area of element k
        a_k = self.GEOMK["ak"][k]

        # 1.3 Local coordinates of nodes of element k
        NODESL = self.COORD["n"][2][a_k]
        v = np.array(NODESL[e_k])

        # -----------------------------------------------------------------------------------------------------------------#
        # 2 Integration of quadrilaterals
        # -----------------------------------------------------------------------------------------------------------------#
        order = 1
        if len(e_k) == 4:
            if order == 1:
                # print(go)
                # 2.1 Integration of Bending and Membrane Terms

                # 2.1.1 Gauss points and weights
                gp, w = self.gauss_points(4,go)
                xi = gp[j]
                eta = gp[i]

                # 2.1.2 Shape Functions in xi-eta-space
                N1xe = 1/4*(1-xi)*(1-eta)/1
                N2xe = 1/4*(1+xi)*(1-eta)/1
                N3xe = 1/4*(1+xi)*(1+eta)/1
                N4xe = 1/4*(1-xi)*(1+eta)/1

                # 2.1.3 Derivatives of Shape Functions in xi-eta-space
                N1xi = float(- (1 - eta) / 4)
                N2xi = float((1 - eta) / 4)
                N3xi = float((1 + eta) / 4)
                N4xi = float(- (1 + eta) / 4)
                N1eta = float(- (1 - xi) / 4)
                N2eta = float(- (1 + xi) / 4)
                N3eta = float((1 + xi) / 4)
                N4eta = float((1 - xi) / 4)
                N1xi_eta = np.array([[N1xi], [N1eta]])
                N2xi_eta = np.array([[N2xi], [N2eta]])
                N3xi_eta = np.array([[N3xi], [N3eta]])
                N4xi_eta = np.array([[N4xi], [N4eta]])

                # 2.1.4 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])

                # 2.1.5 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)

                # 2.1.6 Derivatives of Shape Functions in x-y-Space
                N1x_y = np.matmul(J_inv, N1xi_eta)
                N2x_y = np.matmul(J_inv, N2xi_eta)
                N3x_y = np.matmul(J_inv, N3xi_eta)
                N4x_y = np.matmul(J_inv, N4xi_eta)
                N1x = float(N1x_y[0])
                N1y = float(N1x_y[1])
                N2x = float(N2x_y[0])
                N2y = float(N2x_y[1])
                N3x = float(N3x_y[0])
                N3y = float(N3x_y[1])
                N4x = float(N4x_y[0])
                N4y = float(N4x_y[1])

                # 2.1.7 Strain matrices in bending and membrane, transverse shear strain matrix for full integration
                #       without prescribed shear transverse shear strain (with locking)
                #       - Rotated global coordinates with Tk
                Tk = self.rotLG(k)[0]
                B_sf = np.array([[0, 0, N1x, -N1xe, 0, 0,0, 0, N2x, -N2xe, 0, 0,0, 0, N3x, -N3xe, 0, 0,0, 0, N4x, -N4xe, 0,0],
                                [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0, 0, 0, N4y, 0, -N4xe, 0]])
                B_b = np.array([[0,0,0, N1x,0,0,0, 0, 0, N2x, 0,0,0,0,0, N3x, 0,0, 0,0,0, N4x, 0,0],
                                [0,0,0,0, N1y, 0,0, 0,0,0, N2y, 0,0, 0,0,0, N3y, 0,0, 0,0,0, N4y,0],
                                [0,0,0,N1y, N1x,0,0,0,0,N2y,N2x,0,0,0,0,N3y,N3x,0,0,0,0,N4y,N4x,0]])
                B_m = np.array([[N1x,0,0,0, 0, 0, N2x,0, 0,0,0,0, N3x,0, 0, 0,0,0, N4x,0, 0,0,0,0],
                                [0, N1y,0, 0, 0,0,0, N2y,0, 0, 0,0,0, N3y,0, 0, 0,0,0, N4y,0,0, 0,0],
                                [N1y, N1x,0, 0,0,0,N2y,N2x,0,0,0,0,N3y,N3x,0,0,0,0,N4y,N4x,0,0,0,0]])

                # 2.2 Integration of Shear Terms: Assumed Transverse Shear Element
                Js = np.zeros((4,2,2))
                for ijs in range(4):

                    # 2.2.1 Integration points in edge centers
                    xis = np.array([0, 1, 0, -1])[ijs]
                    etas = np.array([-1, 0, 1, 0])[ijs]

                    # 2.2.2 Shape Functions in xi-etas-space
                    N1s = 1 / 4 * (1 - xis) * (1 - etas)/1
                    N2s = 1 / 4 * (1 + xis) * (1 - etas)/1
                    N3s = 1 / 4 * (1 + xis) * (1 + etas)/1
                    N4s = 1 / 4 * (1 - xis) * (1 + etas)/1

                    # 2.2.3 Derivatives of Shape Functions in xis-etas-space
                    N1xis = float(- (1 - etas) / 4)
                    N2xis = float((1 - etas) / 4)
                    N3xis = float((1 + etas) / 4)
                    N4xis = float(- (1 + etas) / 4)
                    N1etas = float(- (1 - xis) / 4)
                    N2etas = float(- (1 + xis) / 4)
                    N3etas = float((1 + xis) / 4)
                    N4etas = float((1 - xis) / 4)
                    N1xis_etas = np.array([[N1xis], [N1etas]])
                    N2xis_etas = np.array([[N2xis], [N2etas]])
                    N3xis_etas = np.array([[N3xis], [N3etas]])
                    N4xis_etas = np.array([[N4xis], [N4etas]])

                    # 2.1.4 Gradient Matrix
                    Grad_Mats = np.array([[N1xis, N2xis, N3xis, N4xis], [N1etas, N2etas, N3etas, N4etas]])

                    # 2.1.5 Jacobian
                    Js[ijs][:][:] = np.matmul(Grad_Mats, v)
                    Js_inv = np.linalg.inv(Js[ijs][:][:])
                    Js_det = np.linalg.det(Js[ijs][:][:])

                    # 2.2.4 Derivatives of Shape Functions in x-y-Space
                    N1x_y = np.matmul(Js_inv, N1xis_etas)
                    N2x_y = np.matmul(Js_inv, N2xis_etas)
                    N3x_y = np.matmul(Js_inv, N3xis_etas)
                    N4x_y = np.matmul(Js_inv, N4xis_etas)
                    N1x = float(N1x_y[0])
                    N1y = float(N1x_y[1])
                    N2x = float(N2x_y[0])
                    N2y = float(N2x_y[1])
                    N3x = float(N3x_y[0])
                    N3y = float(N3x_y[1])
                    N4x = float(N4x_y[0])
                    N4y = float(N4x_y[1])

                    # 2.2.5 Bijs-Matrix
                    B_ijs = np.array([[0, 0, N1x, -N1s, 0, 0, 0, 0, N2x, -N2s, 0, 0, 0, 0, N3x, -N3s, 0, 0, 0, 0, N4x, -N4s, 0, 0],
                                    [0, 0, N1y, 0, -N1s, 0,0, 0, N2y, 0, -N2s, 0,0, 0, N3y, 0, -N3s, 0,0, 0, N4y, 0, -N4s, 0]
                                    ])

                    # 2.2.6 Assembly to B-bar Matrix
                    if ijs == 0:
                        B_bar = B_ijs
                    else:
                        B_bar = np.append(B_bar,B_ijs,axis=0)

                # 2.2.7 Auxiliary Matrices for prescribed transverse shear strain field

                Cmat = np.array([[Js[0][0, 0], Js[0][0, 1], 0, 0, 0, 0, 0, 0],
                                [Js[0][1, 0], Js[0][1, 1], 0, 0, 0, 0, 0, 0],
                                [0, 0, Js[1][0, 0], Js[1][0, 1], 0, 0, 0, 0],
                                [0, 0, Js[1][1, 0], Js[1][1, 1], 0, 0, 0, 0],
                                [0, 0, 0, 0, Js[2][0, 0], Js[2][0, 1], 0, 0],
                                [0, 0, 0, 0, Js[2][1, 0], Js[2][1, 1], 0, 0],
                                [0, 0, 0, 0, 0, 0, Js[3][0, 0], Js[3][0, 1]],
                                [0, 0, 0, 0, 0, 0, Js[3][1, 0], Js[3][1, 1]],
                                ])
                Tmat = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1]])
                Pmat = np.array([[1, -1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, -1]])
                Amat = np.array([[1, gp[i], 0, 0], [0, 0, 1, gp[j]]])

                # 2.2.8 B_s-Matrix
                B_sats = np.matmul(J_inv,np.matmul(Amat,np.matmul(np.linalg.inv(Pmat),np.matmul(Tmat,np.matmul(Cmat,B_bar)))))

                # 2.3 Integration of Shear Terms: reduced integration
                # 2.3.1 Gauss points and weights
                # gp, w = gauss_points(4, 1)
                # xi = gp[0]
                # eta = gp[0]
                #
                # # 2.3.2 Shape Functions in xi-eta-space
                # N1xe = 1 / 4 * (1 - xi) * (1 - eta)
                # N2xe = 1 / 4 * (1 + xi) * (1 - eta)
                # N3xe = 1 / 4 * (1 + xi) * (1 + eta)
                # N4xe = 1 / 4 * (1 - xi) * (1 + eta)
                #
                # # 2.3.3 Derivatives of Shape Functions in xi-eta-space
                # N1xi = float(- (1 - eta) / 4)
                # N2xi = float((1 - eta) / 4)
                # N3xi = float((1 + eta) / 4)
                # N4xi = float(- (1 + eta) / 4)
                # N1eta = float(- (1 - xi) / 4)
                # N2eta = float(- (1 + xi) / 4)
                # N3eta = float((1 + xi) / 4)
                # N4eta = float((1 - xi) / 4)
                # N1xi_eta = np.array([[N1xi], [N1eta]])
                # N2xi_eta = np.array([[N2xi], [N2eta]])
                # N3xi_eta = np.array([[N3xi], [N3eta]])
                # N4xi_eta = np.array([[N4xi], [N4eta]])
                #
                # # 2.3.4 Gradient Matrix
                # Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])
                #
                # # 2.3.5 Jacobian
                # J = np.matmul(Grad_Mat, v)
                # J_inv = np.linalg.inv(J)
                # J_det = np.linalg.det(J)
                #
                # # 2.3.6 Derivatives of Shape Functions in x-y-Space
                # N1x_y = np.matmul(J_inv, N1xi_eta)
                # N2x_y = np.matmul(J_inv, N2xi_eta)
                # N3x_y = np.matmul(J_inv, N3xi_eta)
                # N4x_y = np.matmul(J_inv, N4xi_eta)
                # N1x = float(N1x_y[0])
                # N1y = float(N1x_y[1])
                # N2x = float(N2x_y[0])
                # N2y = float(N2x_y[1])
                # N3x = float(N3x_y[0])
                # N3y = float(N3x_y[1])
                # N4x = float(N4x_y[0])
                # N4y = float(N4x_y[1])
                #
                # # 2.1.7 Shear Strain matrix with reduced integration
                # #       - Rotated global coordinates with Tk
                # Tk = rotLG(k)[0]
                # B_sred = np.array(
                #     [[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0, 0, 0, N4x, -N4xe, 0, 0],
                #      [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0, 0, 0, N4y, 0, -N4xe, 0]])

                # Return wished for Shear matrix
                # B_s = B_sf
                B_s = B_sats
                # B_s = B_sred
        # -----------------------------------------------------------------------------------------------------------------#
        # 3 Integration of triangles
        # -----------------------------------------------------------------------------------------------------------------#
        else:
            if order == 1:
                # 3.1 Integration of Bending and Membrane Terms

                # 3.1.1 Coordinates of element nodes: delete entry 100'001
                v = np.array(NODESL[self.ELEMENTS[k][0:3]])

                # 3.1.2 Gauss Points
                gp, w = self.gauss_points(3, go)
                xi = gp[j]
                eta = gp[i]

                # 3.1.3 Shape Functions in xi-eta space
                N1xe = 1 - xi - eta
                N2xe = xi
                N3xe = eta

                # 3.1.4 Derivatives Shape Functions in xi-eta space
                N1xi = -1
                N2xi = 1
                N3xi = 0
                N1eta = -1
                N2eta = 0
                N3eta = 1

                N1xi_eta = np.array([[N1xi], [N1eta]])
                N2xi_eta = np.array([[N2xi], [N2eta]])
                N3xi_eta = np.array([[N3xi], [N3eta]])

                # 3.1.5 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])

                # 3.1.6 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)

                # 3.1.7 Derivatives of Shape Functions in x-y-Space
                N1x_y = np.matmul(J_inv, N1xi_eta)
                N2x_y = np.matmul(J_inv, N2xi_eta)
                N3x_y = np.matmul(J_inv, N3xi_eta)
                N1x = float(N1x_y[0])
                N1y = float(N1x_y[1])
                N2x = float(N2x_y[0])
                N2y = float(N2x_y[1])
                N3x = float(N3x_y[0])
                N3y = float(N3x_y[1])

                # 3.1.7 Strain matrices in bending and membrane, transverse shear strain matrix for full integration
                #       without prescribed shear transverse shear strain (with locking)
                #       - Rotated global coordinates with Tk
                B_sf = np.array([[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0],
                                [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0]])
                B_b = np.array([[0, 0, 0, N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0],
                                [0, 0, 0, 0, N1y, 0, 0, 0, 0, 0, N2y, 0, 0, 0, 0, 0, N3y, 0],
                                [0, 0, 0, N1y, N1x, 0, 0, 0, 0, N2y, N2x, 0, 0, 0, 0, N3y, N3x, 0]])
                B_m = np.array([[N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0, 0, 0, 0],
                                [0, N1y, 0, 0, 0, 0, 0, N2y, 0, 0, 0, 0, 0, N3y, 0, 0, 0, 0],
                                [N1y, N1x, 0, 0, 0, 0, N2y, N2x, 0, 0, 0, 0, N3y, N3x, 0, 0, 0, 0]])

                # 3.2 Integration of Shear Terms: Assumed Transverse Shear Element: constant! only dep. on a1 and a2
                Js = np.zeros((2, 2, 2))
                xisall = np.array([0.5,0])
                etasall = np.array([0, 0.5])
                for ijs in range(2):

                    # 3.2.1 Integration points in edge centers
                    xis = xisall[ijs]
                    etas = etasall[ijs]

                    # 3.2.2 Shape Functions in xi-etas-space
                    N1s = 1 - xis - etas
                    N2s = xis
                    N3s = etas

                    # 3.2.3 Derivatives of Shape Functions in xis-etas-space
                    N1xis = -1
                    N2xis = 1
                    N3xis = 0
                    N1etas = -1
                    N2etas = 0
                    N3etas = 1

                    N1xis_etas = np.array([[N1xis], [N1etas]])
                    N2xis_etas = np.array([[N2xis], [N2etas]])
                    N3xis_etas = np.array([[N3xis], [N3etas]])

                    # 3.1.4 Gradient Matrix
                    Grad_Mats = np.array([[N1xis, N2xis, N3xis], [N1etas, N2etas, N3etas]])

                    # 3.1.5 Jacobian
                    Js[ijs][:][:] = np.matmul(Grad_Mats, v)
                    Js_inv = np.linalg.inv(Js[ijs][:][:])
                    Js_det = np.linalg.det(Js[ijs][:][:])

                    # 3.2.4 Derivatives of Shape Functions in x-y-Space
                    N1x_y = np.matmul(Js_inv, N1xis_etas)
                    N2x_y = np.matmul(Js_inv, N2xis_etas)
                    N3x_y = np.matmul(Js_inv, N3xis_etas)
                    N1x = float(N1x_y[0])
                    N1y = float(N1x_y[1])
                    N2x = float(N2x_y[0])
                    N2y = float(N2x_y[1])
                    N3x = float(N3x_y[0])
                    N3y = float(N3x_y[1])


                    # 3.2.5 Bijs-Matrix
                    B_ijs = np.array(
                        [[0, 0, N1x, -N1s, 0, 0, 0, 0, N2x, -N2s, 0, 0, 0, 0, N3x, -N3s, 0, 0],
                        [0, 0, N1y, 0, -N1s, 0, 0, 0, N2y, 0, -N2s, 0, 0, 0, N3y, 0, -N3s, 0]
                        ])

                    # 3.2.6 Assembly to B-bar Matrix
                    if ijs == 0:
                        B_bar = B_ijs
                    else:
                        B_bar = np.append(B_bar, B_ijs, axis=0)

                # 3.2.7 Auxiliary Matrices for prescribed transverse shear strain field

                Cmat = np.array([[Js[0][0, 0], Js[0][0, 1], 0, 0],
                                [Js[0][1, 0], Js[0][1, 1], 0, 0],
                                [0, 0, Js[1][0, 0], Js[1][0, 1]],
                                [0, 0, Js[1][1, 0], Js[1][1, 1]]])
                Tmat = np.array([[1, 0, 0,0], [0,0, 0, 1]])
                Pmat = np.array([[1, 0,], [0,1]])
                Amat = np.array([[1, 0], [0, 1]])

                # 3.2.8 B_s-Matrix
                Tk = self.rotLG(k)[0]
                Tk = Tk[0:18,0:18]
                B_s = np.matmul(J_inv,
                                np.matmul(Amat, np.matmul(np.linalg.inv(Pmat), np.matmul(Tmat, np.matmul(Cmat, B_bar)))))

                # 3.2 Integration of Shear Terms: Reduced Integration
                # 3.2.1 Gauss Points
                # xi = 1/3
                # eta = 1/3
                #
                # # 3.2.2 Shape Functions in xi-eta space
                # N1xe = 1 - xi - eta
                # N2xe = xi
                # N3xe = eta
                #
                # # 3.2.3 Derivatives Shape Functions in xi-eta space
                # N1xi = -1
                # N2xi = 1
                # N3xi = 0
                # N1eta = -1
                # N2eta = 0
                # N3eta = 1
                # N1xi_eta = np.array([[N1xi], [N1eta]])
                # N2xi_eta = np.array([[N2xi], [N2eta]])
                # N3xi_eta = np.array([[N3xi], [N3eta]])
                #
                # # 3.2.4 Gradient Matrix
                # Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])
                #
                # # 3.2.5 Jacobian
                # J = np.matmul(Grad_Mat, v)
                # J_inv = np.linalg.inv(J)
                # J_det = np.linalg.det(J)
                #
                # # 3.2.6 Derivatives of Shape Functions in x-y-Space
                # N1x_y = np.matmul(J_inv, N1xi_eta)
                # N2x_y = np.matmul(J_inv, N2xi_eta)
                # N3x_y = np.matmul(J_inv, N3xi_eta)
                # N1x = float(N1x_y[0])
                # N1y = float(N1x_y[1])
                # N2x = float(N2x_y[0])
                # N2y = float(N2x_y[1])
                # N3x = float(N3x_y[0])
                # N3y = float(N3x_y[1])
                #
                # # 3.2.7 Strain matrix in transverse shear with reduced integration
                # Tk = rotLG(k)[0]
                # Tk = Tk[0:18,0:18]
                # B_s = np.array([[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0],
                #                  [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0]])


        end = time.time()
        time_B._updatetime(delta_t=end - start)
        # 4 Return B-Matrices in Global or Local Coordinate System
        if rot:
            return B_m@Tk,B_b@Tk,B_s@Tk,J_det
        else:
            return B_m,B_b,B_s,J_det


    def find_b(self,go):
        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        nk = len(self.ELEMENTS[:, 0])
        Bmr = {}
        Bbr = {}
        Bsr = {}
        Bmnr = {}
        Bbnr = {}
        Bsnr = {}
        J = {}
        for k in range(0, nk):
            Bmr[k] = {}
            Bbr[k] = {}
            Bsr[k] = {}
            Bmnr[k] = {}
            Bbnr[k] = {}
            Bsnr[k] = {}
            J[k] = {}
            gp, w = self.gauss_points(self.ELS[4][k], go)
            n_k = self.find_nodes(k)
            n_k = n_k[n_k < 100000]
            for i in range(len(gp)):
                Bmr[k][i] = {}
                Bbr[k][i] = {}
                Bsr[k][i] = {}
                Bmnr[k][i] = {}
                Bbnr[k][i] = {}
                Bsnr[k][i] = {}
                J[k][i] = {}
                for j in range(len(gp)):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        continue
                    [Bmr[k][i][j], Bbr[k][i][j], Bsr[k][i][j], J[k][i][j]] = self.b_kij(k, i, j, go, rot=True)
                    [Bmnr[k][i][j], Bbnr[k][i][j], Bsnr[k][i][j], J[k][i][j]] = self.b_kij(k, i, j, go, rot=False)
        B = {}
        B["Bm"] = {}
        B["Bm"]["r"] = Bmr
        B["Bm"]["nr"] = Bmnr
        B["Bb"] = {}
        B["Bb"]["r"] = Bbr
        B["Bb"]["nr"] = Bbnr
        B["Bs"] = {}
        B["Bs"]["r"] = Bsr
        B["Bs"]["nr"] = Bsnr
        B["Jdet"] = J
        return B

    """---------------------------------------- Stiffness Matrix Calculation---------------------------------------------"""

    def rotLG(self,k):

        Tk1 = self.GEOMK["Tk1"][k]
        Tk2 = self.GEOMK["Tk2"][k]
        Tk3 = self.GEOMK["Tk3"][k]

        temp1 = np.append(Tk1, np.zeros((3, 3)), axis=1)
        temp2 = np.append(np.zeros((3, 3)), Tk2, axis=1)
        temp3 = np.append(temp1, temp2, axis=0)
        temp4 = np.append(temp3, np.zeros((6, 18)), axis=1)
        temp5 = np.append(np.append(np.zeros((6, 6)), temp3, axis=1), np.zeros((6, 12)), axis=1)
        temp6 = np.append(np.append(np.zeros((6, 12)), temp3, axis=1), np.zeros((6, 6)), axis=1)
        temp7 = np.append(np.zeros((6, 18)), temp3, axis=1)
        Tk = np.append(np.append(temp4, temp5, axis=0), np.append(temp6, temp7, axis=0), axis=0)

        temp1 = np.append(np.zeros((3, 3)), np.zeros((3, 3)), axis=1)
        temp2 = np.append(np.zeros((3, 3)),Tk3, axis=1)
        temp3 = np.append(temp1,temp2, axis=0)
        temp4 = np.append(temp3,np.zeros((6,6)),axis = 1)
        temp5 = np.append(np.zeros((6,6)),temp3,axis = 1)
        temp6 = np.append(temp4,temp5,axis=0)
        temp7 = np.append(temp6,np.zeros((12,12)),axis=1)
        temp8 = np.append(np.zeros((12, 12)),temp6, axis=1)
        Tkr = np.append(temp7,temp8,axis=0)

        return Tk,Tkr
        # return np.identity(24),np.identity(24)


    @staticmethod
    def ke_btdb(weight, B_batch, De_g):
        """
        Element stiffness as one weighted B^T D B product on the full 8x8 tangent (opt_time). Equals the
        block-wise einsums of k_k_vec / k_k_nn_num_vec (Kme + Kbe + Kse + Kmbe + Kbme), as De holds the
        same off-diagonal blocks and zero membrane/bending-shear coupling.

        Args:
            weight  (np.arr): Gauss weight * Jdet, shape (E, go, go).
            B_batch (np.arr): stacked [Bm; Bb; Bs], shape (E, go, go, 8, ndof).
            De_g    (np.arr): 8x8 tangent, shape (E, go, go, 8, 8).

        Returns:
            Ke_batch (np.arr): shape (E, ndof, ndof).
        """
        BtDB = np.swapaxes(B_batch, -1, -2) @ De_g @ B_batch                # (E, go, go, ndof, ndof)
        return np.einsum('eij,eijab->eab', weight, BtDB)

    def k_k_vec(self, B, e, s, cmk, go, opt_time=False):
        """
        Element stiffness matrices of all elements (NLFEA): layer integration of the layer tangents (Dmh, Dmbh, Dbh,
        Dsh) and Gauss-point integration of B^T D B as batched einsums, grouped by element type, plus the stiffness
        of the drilling DOF of coplanar elements.

        Args:
            B    (dict): B-matrix container (uses B["Bm"]["r"], B["Bb"]["r"], B["Bs"]["r"], B["Jdet"]).
            e    (np.arr): strains from find_e_vec, shape (nel, nlk, go, go, 5) (unused, the tangent comes from s).
            s    (np.arr): stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5) (unused for cm == 1).
            cmk  (array-like): cm per element, shape (nel,): 1 (linear elastic, initial stiffness) or 3 (nonlinear RC).
            go   (int):  Gauss order.
            opt_time (bool): one B^T D B product per element (ke_btdb) instead of the block-wise einsums.

        Returns:
            Ke_by_type (dict): {n_nodes: Ke (E, ndof, ndof)} per element type (quads: 24, tris: 18 DOF), elements in
                               the order of _element_dofs.
            De         (np.arr): 8x8 tangent per (k, i, j), shape (nel, go, go, 8, 8).
        """
        nel = len(self.ELEMENTS[:, 0])
        cm_arr = np.unique(np.asarray(cmk))
        if cm_arr.size != 1 or cm_arr[0] not in (1, 3):
            raise NotImplementedError(f"k_k_vec: one cm (1 or 3) for all elements, got {cm_arr.tolist()}.")
        cm = int(cm_arr[0])

        # 1-3 Layer tangents and thickness integration (section model of the sampling) ------------------------
        #     cm = 3: complex-step layer tangents from s; cm = 1: linear elastic, E = Ec, nu = vc of the elements
        sec = self._section()
        nlk = sec.constants['n_layer']
        s_in = np.transpose(s[..., :3, :3], (0, 2, 3, 1, 4, 5)).reshape(nel * go * go, nlk, 3, 3)
        geom = None
        if cm == 1:
            if sec.constants['nu'] is None:
                raise NotImplementedError('k_k_vec: the linear elastic tangent needs the same nu in all elements.')
            Ec = np.asarray(self.MATK["Ec"], dtype=float)
            geom = {'mat': {'Ec': np.repeat(Ec, go * go)[:, None, None]}}
        D6 = sec.find_dh_vec(s_in, {}, cm_klij=cm, t=self._samples(go), geom=geom).reshape(nel, go, go, 6, 6)
        Dmh, Dmbh, Dbh = D6[..., 0:3, 0:3], D6[..., 0:3, 3:6], D6[..., 3:6, 3:6]

        # transverse shear (linear elastic): Dsh = 5/6 * G * t * I, G = Ec / (2 (1 + vc))
        Ec = np.asarray(self.MATK["Ec"], dtype=float)
        vc = np.asarray(self.MATK["vc"], dtype=float)
        t_arr = np.asarray(self.GEOMK["t"], dtype=float)
        Dsh_kl = (5.0 / 6.0) * (Ec / (2.0 * (1.0 + vc)) * t_arr)[:, None, None] * np.eye(2)     # (nel, 2, 2)
        Dsh_full = np.broadcast_to(Dsh_kl[:, None, None, :, :], (nel, go, go, 2, 2))

        # 4 Assemble 8x8 De block per (k, i, j) ------------------------------------------------
        # Block layout: [[Dmh, Dmbh, 0], [Dmbh, Dbh, 0], [0, 0, Dsh]] (as find_dh_vec of the sampling).
        # Note: the off-diagonal block is Dmbh (not Dmbh^T) on BOTH sides, and K gets both coupling
        # terms separately in step 5 (Kmbe + Kbme), since Dmbh is non-symmetric for cracked layers.
        De = np.zeros((nel, go, go, 8, 8))
        De[..., 0:3, 0:3] = Dmh
        De[..., 0:3, 3:6] = Dmbh
        De[..., 3:6, 0:3] = Dmbh
        De[..., 3:6, 3:6] = Dbh
        De[..., 6:8, 6:8] = Dsh_full

        # 5 Per-element-type Ke via batched block-wise einsums --------------------------------
        Ke_by_type = {}
        for n_nodes in self.ELEMENT_TYPES:
            els, _ = self._element_dofs(n_nodes)
            if len(els) == 0:
                continue
            B_batch, weight = self._b_batch(B, n_nodes, go)                 # (E, go, go, 8, ndof), (E, go, go)

            # Block-wise sub-B matrices
            Bm_b = B_batch[..., 0:3, :]                                     # (E, go, go, 3, ndof)
            Bb_b = B_batch[..., 3:6, :]                                     # (E, go, go, 3, ndof)
            Bs_b = B_batch[..., 6:8, :]                                     # (E, go, go, 2, ndof)

            if opt_time:
                Ke_batch = self.ke_btdb(weight, B_batch, De[els])
            else:
                Dmh_g  = Dmh[els]
                Dmbh_g = Dmbh[els]
                Dbh_g  = Dbh[els]
                Dsh_g  = Dsh_full[els]

                # K_block[e, a, b] = sum_{i, j, p, q} weight * B^T D B
                Kme_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmh_g,  Bm_b)
                Kbe_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bb_b, Dbh_g,  Bb_b)
                Kse_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bs_b, Dsh_g,  Bs_b)
                Kmbe_batch = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmbh_g, Bb_b)
                # Bb^T Dmbh Bm (not Kmbe^T = Bb^T Dmbh^T Bm): the cracked layer tangent is non-symmetric
                Kbme_batch = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bb_b, Dmbh_g, Bm_b)

                Ke_batch = Kme_batch + Kbe_batch + Kse_batch + Kmbe_batch + Kbme_batch   # (E, ndof, ndof)

            # stiffness of the drilling DOF (coplanar elements)
            Ke_by_type[n_nodes] = Ke_batch + self._drilling_batch(B, n_nodes, go)

        return Ke_by_type, De


    def append_geom_cols(self, strains, k_valid, model):
        """
        Append per-element geometry columns so the NN input width matches the model's inp['input_size'];
        column order [eps(6), t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, (fc, (fs))] matches the sampler's append_geom,
        fc = per-element MATK['fcp'], fs = per-element MATK['fsyx'] (= steel yield strength; x == y enforced in find_s_vec)
        (legacy widths [eps(6), t, rho_x, rho_y] and [eps(6), t, rho_y] still supported).

        Args:
            strains (np.arr): (n_valid, 6) generalised strains for the valid samples.
            k_valid (np.arr): (n_valid,) element index per valid sample (k_idx[valid]).
            model   (NNPredictor): trained model; its inp['input_size'] sets the width.

        Returns:
            np.arr: (n_valid, 6 + n_geom) input in raw units, so the model's stored stats normalise the geom columns as in sampling.
        """
        n_geom = n_geom_from_model(model)
        if n_geom == 0:
            return strains
        # per-element per-layer rho arrays: layer 0 = bottom, layer -1 = top (baseline single-value: bot == top)
        t_col        = np.asarray(self.GEOMK['t'], dtype=strains.dtype)[k_valid].reshape(-1, 1)
        rhox_bot_col = np.asarray([r[0]  for r in self.GEOMK['rhox']], dtype=strains.dtype)[k_valid].reshape(-1, 1)
        rhox_top_col = np.asarray([r[-1] for r in self.GEOMK['rhox']], dtype=strains.dtype)[k_valid].reshape(-1, 1)
        rhoy_bot_col = np.asarray([r[0]  for r in self.GEOMK['rhoy']], dtype=strains.dtype)[k_valid].reshape(-1, 1)
        rhoy_top_col = np.asarray([r[-1] for r in self.GEOMK['rhoy']], dtype=strains.dtype)[k_valid].reshape(-1, 1)
        # legacy single-value geometry columns (n_geom 1/2/3) default to the top reinforcement layer
        rhox_col     = rhox_top_col
        rhoy_col     = rhoy_top_col
        if n_geom == 1:
            return np.concatenate((strains, rhoy_col), axis=1)
        if n_geom == 2:
            return np.concatenate((strains, t_col, rhoy_col), axis=1)
        if n_geom == 3:
            return np.concatenate((strains, t_col, rhox_col, rhoy_col), axis=1)
        if n_geom == 5:
            return np.concatenate((strains, t_col, rhox_top_col, rhox_bot_col, rhoy_top_col, rhoy_bot_col), axis=1)
        if n_geom in (6, 7):
            fc_col = np.asarray(self.MATK['fcp'], dtype=strains.dtype)[k_valid].reshape(-1, 1)   # per-element fc
            cols = [strains, t_col, rhox_top_col, rhox_bot_col, rhoy_top_col, rhoy_bot_col, fc_col]
            if n_geom == 7:
                cols.append(np.asarray(self.MATK['fsyx'], dtype=strains.dtype)[k_valid].reshape(-1, 1))   # per-element fs
            return np.concatenate(cols, axis=1)
        raise ValueError(f"NN input_size implies {n_geom} geometry columns (input_size={n_geom + 6}); only 6, 7, 8, 9, 11, 12, 13 supported.")


    def k_k_nn_num_vec(self, B, e, s, eh, sh, cmk, go, model_dim, scenario, symm_predict=False, opt_time=False):
        """
        Vectorised drop-in for k_k_nn_num. Implemented for model_dim == 'THREEDIM' only.

        For each (k, i, j):
            * Predict the upper-left 6x6 block of De from the NN using eh as input.
            * Out-of-plane shear block Dsh (2x2), linear elastic as in k_k_vec (5/6 * G * t).
            * Combine: De[0:6, 0:6] = D_NN; De[6:8, 6:8] = Dsh; off-block zeros.
            * Per-element Ke assembled block-by-block matching k_k_nn_num's original
              formula: Ke = Kme + Kbe + Kse + Kmbe + Kbme + drilling, with
                  Kmbe = w*Jdet * Bm^T Dmbh Bb
                  Kbme = w*Jdet * Bm^T Dbmh Bb 

        Args:
            B    (dict):   B-matrix container (uses ["Bm"]["r"], ["Bb"]["r"], ["Bs"]["r"], ["Jdet"]).
            e    (np.arr): strains from find_e_vec, shape (nel, nlk, go, go, 5).
            s    (np.arr): stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5).
            eh   (np.arr): generalised strains, shape (nel, go, go, 8) -- NN input.
            sh   (np.arr): generalised stresses, shape (nel, go, go, 8). Accepted for
                           signature parity; not consumed (matches k_k_nn_num).
            cmk  (array-like): per-element cm. Restricted to all cm == 3.
            go   (int): Gauss order.
            model_dim (str): only 'THREEDIM' implemented.
            scenario  (int): boundary scenario code (unused).
            symm_predict (bool): mid-plane symmetry wrapper of the NN.
            opt_time  (bool): NN tangent from find_sh_nn_vec's one-pass cache if the strains match (else NN
                              jacobian w.r.t. the strains only), one B^T D B product per element (ke_btdb).

        Returns:
            Ke_by_type (dict): {n_nodes: Ke (E, ndof, ndof)} per element type (see k_k_vec).
            De         (np.arr): per-(k, i, j) 8x8 tangent, shape (nel, go, go, 8, 8).

        """
        if model_dim != 'THREEDIM':
            raise NotImplementedError(
                f"k_k_nn_num_vec: only model_dim='THREEDIM' implemented; got {model_dim!r}"
            )

        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        cm_arr = np.asarray(cmk)
        if not (cm_arr == 3).all():
            raise NotImplementedError("k_k_nn_num_vec: only cm == 3 supported.")

        # 1 Numerical Dsh (out-of-plane shear) -- analytical per-layer + layer integration ----
        # Per-layer Ds_kl = 5/6 * G * I2 with G = Ec/(2*(1+vc)).
        Ec = np.asarray(self.MATK["Ec"], dtype=float)                        # (nel,)
        vc = np.asarray(self.MATK["vc"], dtype=float)                        # (nel,)
        G  = Ec / (2.0 * (1.0 + vc))                                          # (nel,)
        I2 = np.eye(2)
        Ds = (5.0 / 6.0) * G[:, None, None, None] * np.broadcast_to(I2, (nel, nlk, 2, 2))
        Ds = np.ascontiguousarray(Ds)                                         # (nel, nlk, 2, 2)

        t_arr   = np.asarray(self.GEOMK["t"],   dtype=float)
        nlk_arr = np.asarray(self.GEOMK["nlk"], dtype=int)
        l_idx   = np.arange(nlk)
        dz = np.broadcast_to(t_arr[:, None] / nlk_arr[:, None], (nel, nlk)).copy()
        layer_mask = l_idx[None, :] < nlk_arr[:, None]
        dz = dz * layer_mask

        Dsh_kl   = np.sum(Ds * dz[:, :, None, None], axis=1)                 # (nel, 2, 2)
        Dsh_full = np.broadcast_to(Dsh_kl[:, None, None, :, :],
                                   (nel, go, go, 2, 2))                       # (nel, go, go, 2, 2)

        # 2 NN prediction for 6x6 D matrix ----------------------------------------------------
        n_tot = nel * go * go
        k_idx = np.repeat(np.arange(nel), go * go)
        ij    = np.tile(np.arange(go * go), nel)
        i_idx = ij // go
        j_idx = ij %  go
        types_arr = np.asarray(self.ELS[4])
        sentinel  = (types_arr[k_idx] == 3) & (i_idx == 1) & (j_idx == 1)
        valid     = ~sentinel

        eh_flat  = eh.reshape(n_tot, 8)
        input_nn = eh_flat[valid, :6]                                         # (n_valid, 6)

        chosen_model = self.nn_model
        input_nn = self.append_geom_cols(input_nn, k_idx[valid], chosen_model)   # (n_valid, 6 + n_geom)
        cache = getattr(self, '_D_cache', None) if opt_time else None
        if cache is not None and cache[0].shape == input_nn.shape and np.array_equal(cache[0], input_nn):
            D_pred_valid = cache[1].copy()                                    # from find_sh_nn_vec's one-pass call
        else:
            mat_NN = make_NN_prediction(input_nn, predict='D', model=chosen_model, non_batchwise=True, symm_predict=symm_predict)
            D_pred_valid = np.asarray(mat_NN['D'])                            # (n_valid, 6, 6)

        # Scatter to full (n_tot, 6, 6) and reshape to (nel, go, go, 6, 6).
        D_pred_full = np.zeros((n_tot, 6, 6))
        D_pred_full[valid] = D_pred_valid
        D_pred_struct = D_pred_full.reshape(nel, go, go, 6, 6)

        # 3 Combine 8x8 De -- NN for upper-left 6x6, numerical Dsh for [6:8, 6:8] -----------
        De = np.zeros((nel, go, go, 8, 8))
        De[..., 0:6, 0:6] = D_pred_struct
        De[..., 6:8, 6:8] = Dsh_full

        # 4 Per-element-type Ke via batched block-wise einsums -------------------------------
        Ke_by_type = {}
        for n_nodes in self.ELEMENT_TYPES:
            els, _ = self._element_dofs(n_nodes)
            if len(els) == 0:
                continue
            B_batch, weight = self._b_batch(B, n_nodes, go)

            Bm_b = B_batch[..., 0:3, :]
            Bb_b = B_batch[..., 3:6, :]
            Bs_b = B_batch[..., 6:8, :]

            De_g   = De[els]
            if opt_time:
                Ke_batch = self.ke_btdb(weight, B_batch, De_g)
            else:
                Dmh_g  = De_g[..., 0:3, 0:3]
                Dmbh_g = De_g[..., 0:3, 3:6]
                Dbmh_g = De_g[..., 3:6, 0:3]
                Dbh_g  = De_g[..., 3:6, 3:6]
                Dsh_g  = De_g[..., 6:8, 6:8]

                # Block einsums matching k_k_nn_num's exact assembly:
                Kme_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmh_g,  Bm_b)
                Kbe_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bb_b, Dbh_g,  Bb_b)
                Kse_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bs_b, Dsh_g,  Bs_b)
                Kmbe_batch = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmbh_g, Bb_b)
                # Note: Kbme uses Bm^T ... Bb (not Bb^T ... Bm), matching the original k_k_nn_num.
                Kbme_batch = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bb_b, Dbmh_g, Bm_b)

                Ke_batch = Kme_batch + Kbe_batch + Kse_batch + Kmbe_batch + Kbme_batch

            # stiffness of the drilling DOF (coplanar elements), as in k_k_vec
            Ke_by_type[n_nodes] = Ke_batch + self._drilling_batch(B, n_nodes, go)

        return Ke_by_type, De


    def k_glob_vec(self, B, e, s, cmk, go, opt_time=False):
        """
        Global stiffness matrix of the NLFEA (cm == 3) or the linear elastic initial stiffness (cm == 1): element
        matrices from k_k_vec, scattered into K with the element DOF map.

        Args:
            B    (dict):  B-matrix container (forwarded to k_k_vec).
            e    (np.arr): strains from find_e_vec, shape (nel, nlk, go, go, 5).
            s    (np.arr): stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5).
            cmk  (array-like): cm per element, shape (nel,): 1 or 3.
            go   (int): Gauss order.

        Returns:
            K     (np.arr): global stiffness, shape (6*nn, 6*nn).
            D_tot (np.arr): per-(k, i, j) 8x8 tangent, shape (nel, go, go, 8, 8).
        """
        start = time.time()

        Ke_by_type, D_tot = self.k_k_vec(B, e, s, cmk, go, opt_time=opt_time)
        K = self._assemble_matrix(Ke_by_type)

        end = time.time()
        time_K._updatetime(delta_t=end - start)
        return K, D_tot


    def k_glob_nn_num_vec(self, B, e, s, eh, sh, cmk, go, model_dim, scenario, symm_predict=False, opt_time=False):
        """
        Global stiffness matrix with the NN tangent: element matrices from k_k_nn_num_vec, scattered into K with the
        element DOF map.

        Restrictions:
            * model_dim == 'THREEDIM' only (inherited from k_k_nn_num_vec).
            * cm == 3 only.

        Args:
            B    (dict):  B-matrix container (forwarded to k_k_nn_num_vec).
            e, s, eh, sh: standard FEM tensors.
            cmk  (array-like): per-element cm.
            go   (int):   Gauss order.
            model_dim (str), scenario (int): forwarded to k_k_nn_num_vec.

        Returns:
            K     (np.arr): global stiffness, shape (6*nn, 6*nn).
            D_tot (np.arr): per-(k, i, j) 8x8 tangent, shape (nel, go, go, 8, 8).
        """
        start = time.time()

        Ke_by_type, D_tot = self.k_k_nn_num_vec(B, e, s, eh, sh, cmk, go, model_dim, scenario,
                                                symm_predict=symm_predict, opt_time=opt_time)
        K = self._assemble_matrix(Ke_by_type)

        end = time.time()
        time_K._updatetime(delta_t=end - start)
        return K, D_tot


    """-------------------------------------------- Static Condensation--------------------------------------------------"""


    def c_dof(self):
        """ --------------------------------- Create Vector of condensed DOFs -------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - Boundary conditions from Input file
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - cDOF --> Vector of DOFs to be condensed
        -----------------------------------------------------------------------------------------------------------------"""
        numb = len(self.BC[:,0])
        cDOF = np.array([])
        cVAL = np.array([])
        " ----------------------------------- condensed DOFs from Boundary Conditions -------------------------------------"
        for i in range(numb):
            xmin = self.BC[i,0]
            xmax = self.BC[i,1]
            ymin = self.BC[i,2]
            ymax = self.BC[i,3]
            zmin = self.BC[i,4]
            zmax = self.BC[i,5]
            nodes_i = self.find_node_range(xmin,xmax,ymin,ymax,zmin,zmax)
            if self.BC[i,6]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6)
                cVAL = np.append(cVAL, self.BC[i,6]*np.ones_like(nodes_i))
            if self.BC[i,7]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 1)
                cVAL = np.append(cVAL, self.BC[i, 7]*np.ones_like(nodes_i))
            if self.BC[i, 8]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 2)
                cVAL = np.append(cVAL, self.BC[i, 8]*np.ones_like(nodes_i))
            if self.BC[i, 9]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 3)
                cVAL = np.append(cVAL, self.BC[i, 9]*np.ones_like(nodes_i))
            if self.BC[i, 10]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 4)
                cVAL = np.append(cVAL, self.BC[i, 10]*np.ones_like(nodes_i))
            if self.BC[i, 11]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 5)
                cVAL = np.append(cVAL, self.BC[i, 11]*np.ones_like(nodes_i))
        " ------------------------------------------------ Sort and return ------------------------------------------------"
        indeces = np.argsort(cDOF)
        cDOF = cDOF[indeces]
        cVAL = cVAL[indeces]
        _,induni = np.unique(cDOF,return_index=True)
        cDOF = cDOF[induni]
        cVAL = cVAL[induni]
        # cDOF = np.sort(cDOF)
        # cDOF = np.unique(cDOF)
        cDOF= cDOF.astype(int)
        return cDOF,cVAL


    def v_stat_con(self, v, cDOF, cVAL):
        """ -------------------------------------- Static condensation --------------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - v         --> (Force) Vector to be condensed
            - cond_DOF  --> Vector of DOFs to be condensed
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - v         --> Condensed (force) vector
        -----------------------------------------------------------------------------------------------------------------"""
        # v = np.delete(v, cDOF, 0)
        for i in range(len(cDOF)):
            v[cDOF[i]] = cVAL[i]
        return v


    def m_stat_con(self, M, cDOF):
        """ -------------------------------------- Static condensation --------------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - M         --> (Stiffness) Matrix to be condensed
            - cDOF      --> Vector of DOFs to be condensed
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - M         --> Condensed (stiffness) matrix
        -----------------------------------------------------------------------------------------------------------------"""
        # M = np.delete(M, cDOF, 0)
        # M = np.delete(M, cDOF, 1)
        for i in range(len(cDOF)):
            M[cDOF[i],:] = np.zeros_like(M[cDOF[i],:])
            M[cDOF[i],cDOF[i]] = 1
        return M


    """------------------------------------------------- Assembly--------------------------------------------------------"""


    def f_assemble(self, load_step, Load_el, Load_n):
        """ --------------------------------- Create Vector of applied forces -------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - Applied force conditions from Input file
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - f_e --> vector of applied outer forces per DOF
        -----------------------------------------------------------------------------------------------------------------"""

        " -------------------------------------------- Element Loads ------------------------------------------------------"

        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        f_e = np.zeros((1,6*nn))
        nlel = len(Load_el[load_step][:, 0])
        for i in range(nlel):
            xmin = Load_el[load_step][i, 0]
            xmax = Load_el[load_step][i, 1]
            ymin = Load_el[load_step][i, 2]
            ymax = Load_el[load_step][i, 3]
            zmin = Load_el[load_step][i,4]
            zmax = Load_el[load_step][i,5]
            dir = int(Load_el[load_step][i, 6])

            els_i = self.find_el_range(xmin,xmax,ymin,ymax,zmin,zmax)

            q = Load_el[load_step][i, 7]
            q_e = np.zeros((6, 1))
            q_e[dir-1][0] = q

            for k in range(len(els_i)):
                el = els_i[k]
                nodes = self.find_nodes(el)
                nodes = nodes[nodes<10**5]
                J_det = 0
                q_e1 = np.zeros((6, 1))
                q_e2 = np.zeros((6, 1))
                q_e3 = np.zeros((6, 1))
                q_e4 = np.zeros((6, 1))
                for ii in range(2):
                    for jj in range(2):
                        if self.ELS[4][k] == 4:
                            J_det_ij = self.jacobi(el,ii,jj,2)[4]
                            J_det += J_det_ij
                            xi = self.gauss_points(4,2)[0][jj]
                            eta = self.gauss_points(4,2)[0][ii]
                            wxi = self.gauss_points(4,2)[1][jj]
                            weta = self.gauss_points(4,2)[1][ii]

                            n1 = 1 / 4 * (1 - xi) * (1 - eta)
                            n2 = 1 / 4 * (1 + xi) * (1 - eta)
                            n3 = 1 / 4 * (1 + xi) * (1 + eta)
                            n4 = 1 / 4 * (1 - xi) * (1 + eta)
                        else:
                            if jj == 1 and ii == 1:
                                continue
                            else:
                                J_det_ij = self.jacobi(el, ii, jj, 2)[4]
                                J_det += J_det_ij
                                xi = self.gauss_points(3,2)[0][jj]
                                eta = self.gauss_points(3,2)[0][ii]
                                wxi = self.gauss_points(3,2)[1][jj]
                                weta = self.gauss_points(3,2)[1][ii]

                                n1 = 1 - xi - eta
                                n2 = xi
                                n3 = eta
                                n4 = 0

                        N1 = np.zeros((6,6))
                        np.fill_diagonal(N1,n1)
                        N2 = np.zeros((6,6))
                        np.fill_diagonal(N2,n2)
                        N3 = np.zeros((6,6))
                        np.fill_diagonal(N3,n3)
                        N4 = np.zeros((6,6))
                        np.fill_diagonal(N4,n4)

                        q_e1 += N1@q_e*J_det_ij*wxi*weta
                        q_e2 += N2@q_e*J_det_ij*wxi*weta
                        q_e3 += N3@q_e*J_det_ij*wxi*weta
                        q_e4 += N4@q_e*J_det_ij*wxi*weta
                        q_e_all = [q_e1,q_e2,q_e3,q_e4]
                for n in range(len(nodes)):
                    node = nodes[n]
                    for vecit in range(6):
                        f_e[0][node*6+vecit] += q_e_all[n][vecit]
        " --------------------------------------------- Nodal Loads -------------------------------------------------------"
        nln = len(Load_n[load_step][:, 0])
        for i in range(nln):
            xmin = Load_n[load_step][i, 0]
            xmax = Load_n[load_step][i, 1]
            ymin = Load_n[load_step][i, 2]
            ymax = Load_n[load_step][i, 3]
            zmin = Load_n[load_step][i, 4]
            zmax = Load_n[load_step][i, 5]
            dir = Load_n[load_step][i, 6]

            nodes_i = self.find_node_range(xmin,xmax,ymin,ymax,zmin,zmax)
            # if load is not acting in any existing node: create equivalent force and moment in closest node
            if nodes_i.size == 0:
                ms = self.MATK['ms']
                step = ms/10
                count = 1
                while nodes_i.size == 0:
                    nodes_i = self.find_node_range(xmin-count*step,xmax+count*step,ymin-count*step,ymax+count*step,zmin-count*step,zmax+count*step)
                    count += 1
                if nodes_i.size > 1:
                    nodes_i = np.array([nodes_i[0]])
                coordsi = NODESG[nodes_i[0],:]
                diffcordsi = np.array([(xmin+xmax)/2, (ymin+ymax)/2,(zmin+zmax)/2])-coordsi
            else:
                diffcordsi = np.array([0,0,0])
            load_i = Load_n[load_step][i,7]
            for j in range(len(nodes_i)):
                node_j = nodes_i[j]
                coord = NODESG[node_j,:]
                node = self.find_node(coord)
                if dir == 1:
                    DOF = node*6
                elif dir == 2:
                    DOF = node*6+1
                elif dir == 3:
                    DOF = node*6+2
                elif dir == 4:
                    DOF = node*6+3
                elif dir == 5:
                    DOF = node*6+4
                elif dir == 6:
                    DOF = node*6+5
                f_e[0,DOF] = f_e[0,DOF] + load_i
            # Add bending moment for equivalent nodal force (only for offset translational loads)
                if dir in (1, 2, 3) and np.any(diffcordsi):
                    fii = np.zeros(3)
                    fii[int(dir)-1] = load_i
                    moments = np.cross(diffcordsi, fii)
                    for im in range(0, 3):
                        f_e[0, int(node*6+3+im)] += moments[im]
        f_e = np.transpose(f_e)
        return f_e


    def f0_assemble(self, B,s0,go):
        """
        External force vector caused by the internal (shrinkage) stresses: f_0 = - sum w_i w_j Jdet B^T sh0 over all
        elements and Gauss points, with sh0 the internal layer stresses s0 integrated over the thickness.

        Args:
            B   (dict):   B-matrix container of find_b (Gauss points 0 ... go-1 are used)
            s0  (np.arr): internal layer stresses from find_s0_vec, shape (nel, nlk, >= go, >= go, 3)
            go  (int):    Gauss order of the integration

        Returns:
            tuple: sh0 (nel, go, go, 8) internal stresses integrated over the thickness, f_0 (6 * n_nodes, 1) external
                   force vector per DOF
        """
        sh0 = self.find_sh0_vec(s0[:, :, :go, :go])
        if go >= 2:
            sh0[np.asarray(self.ELS[4]) == 3, 1, 1, :] = 0.0                 # triangles: no Gauss point (1, 1)
        f0_by_type = {}
        for n_nodes in self.ELEMENT_TYPES:
            els, _ = self._element_dofs(n_nodes)
            if len(els) == 0:
                continue
            B_batch, weight = self._b_batch(B, n_nodes, go)
            f0_by_type[n_nodes] = -np.einsum('eij,eijpa,eijp->ea', weight, B_batch, sh0[els])
        return sh0, self._assemble_vector(f0_by_type)
    

    """-------------------------------------------- Auxiliary Functions--------------------------------------------------"""

    def find_node(self, coord):
        """ -------------------------------- Find node number for given coordinates -------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - coord --> coordinates of regarded node
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - nr    --> node number of regarded node
        -----------------------------------------------------------------------------------------------------------------"""
        NODESG = self.COORD["n"][0]
        for j in range(len(NODESG[:, 0])):
            diff = abs(np.add(coord, -NODESG[j, :]))
            if max(diff) < pow(10, -10):
                nr = j
        return nr


    def find_nodes(self, el):
        """ ------------------------------------ Nodes connected to a given Element ----------------------------------------
                --------------------------------------------    INPUT: -----------------------------------------------------
                - el --> element number
                --------------------------------------------- OUTPUT:-------------------------------------------------------
                - nodes    --> vector of nodes connected to given element
        -----------------------------------------------------------------------------------------------------------------"""
        nodes = self.ELEMENTS[int(el)]
        return nodes


    def find_node_range(self, xmin, xmax, ymin, ymax, zmin, zmax):
        NODESG = self.COORD["n"][0]
        nodesx = NODESG[:,0]
        nodesy = NODESG[:,1]
        nodesz = NODESG[:,2]
        ind1 = np.array(np.where(nodesx<=xmax)).ravel()
        ind2 = np.array(np.where(nodesx>=xmin)).ravel()
        indx = np.intersect1d(ind1,ind2)

        ind1 = np.array(np.where(nodesy <= ymax)).ravel()
        ind2 = np.array(np.where(nodesy >= ymin)).ravel()
        indy = np.intersect1d(ind1, ind2)

        ind1 = np.array(np.where(nodesz <= zmax)).ravel()
        ind2 = np.array(np.where(nodesz >= zmin)).ravel()
        indz = np.intersect1d(ind1, ind2)

        indxy = np.intersect1d(indx,indy)
        ind = np.intersect1d(indxy,indz)

        return ind


    def find_el_range(self, xmin,xmax,ymin,ymax,zmin,zmax):
        centx = self.COORD["c"][0][:,0]
        centy = self.COORD["c"][0][:,1]
        centz = self.COORD["c"][0][:,2]

        ind1 = np.array(np.where(centx<=xmax)).ravel()
        ind2 = np.array(np.where(centx>=xmin)).ravel()
        indx = np.intersect1d(ind1,ind2)

        ind1 = np.array(np.where(centy <= ymax)).ravel()
        ind2 = np.array(np.where(centy >= ymin)).ravel()
        indy = np.intersect1d(ind1, ind2)

        ind1 = np.array(np.where(centz <= zmax)).ravel()
        ind2 = np.array(np.where(centz >= zmin)).ravel()
        indz = np.intersect1d(ind1, ind2)

        indxy  = np.intersect1d(indx,indy)
        ind = np.intersect1d(indxy,indz)
        return ind


    def find_fi(self, B,sh):
        """
        Internal nodal forces fint = sum w_i w_j Jdet B^T sh over all elements and Gauss points.

        Args:
            B   (dict):   B-matrix container of find_b (Gauss order self.gauss_order)
            sh  (np.arr): generalised stresses, shape (nel, go, go, 8)

        Returns:
            np.arr: internal forces, shape (6 * n_nodes, 1)
        """
        go = self.gauss_order
        fi_by_type = {}
        for n_nodes in self.ELEMENT_TYPES:
            els, _ = self._element_dofs(n_nodes)
            if len(els) == 0:
                continue
            B_batch, weight = self._b_batch(B, n_nodes, go)
            # triangles: the sentinel at Gauss point (1, 1) has zero weight and B, it is masked to avoid 0 * 1e5 terms
            sh_g = np.where(weight[..., None] != 0.0, np.asarray(sh, dtype=float)[els], 0.0)
            fi_by_type[n_nodes] = np.einsum('eij,eijpa,eijp->ea', weight, B_batch, sh_g)
        return self._assemble_vector(fi_by_type)


    """----------------------------------------------- Find Strains -----------------------------------------------------"""


    def find_eh_vec(self, B, u, go):
        """
        Vectorised version of find_eh.

        Args:
            B   (dict): B-matrix container (uses B["Bm"]["nr"], B["Bb"]["nr"], B["Bs"]["nr"])
            u   (np.arr): global displacement vector
            go  (int): Gauss order

        Returns:
            eh  (np.arr): generalised strains at Gauss points, shape (nel, go, go, 8)

        Strategy: group elements by type (4-node quads / 3-node tris), assemble per-element local
        displacement u_k, rotation Tk, and stacked B matrix B_kij = vstack(Bm, Bb, Bs) for each
        Gauss point. Then `eh_kij = B_kij @ Tk @ u_k` is evaluated as two einsum contractions.
        Triangle elements get the -1e5 sentinel at the (i=1, j=1) Gauss slot, matching find_eh.
        """
        start = time.time()

        nel = len(self.ELEMENTS[:, 0])
        eh = np.zeros((nel, go, go, 8))
        types = np.asarray(self.ELS[4])

        u_flat = np.asarray(u).ravel()
        for n_nodes in self.ELEMENT_TYPES:
            els, dofs = self._element_dofs(n_nodes)
            if len(els) == 0:
                continue

            u_e   = u_flat[dofs]                                            # element displacements (E, ndof)
            u_rot = np.einsum('eij,ej->ei', self._rot_batch(n_nodes), u_e)  # in local coordinates (E, ndof)

            # stacked local B matrices (E, go, go, 8, ndof); triangles: zero at Gauss point (1, 1), sentinel below
            B_batch, _ = self._b_batch(B, n_nodes, go, kind='nr')

            # eh_kij = B_kij @ u_rot : (E, go, go, 8)
            eh_e = np.einsum('eijkl,el->eijk', B_batch, u_rot)
            eh[els] = eh_e

        # Triangle sentinel at (i=1, j=1); only reachable when go >= 2.
        if go >= 2:
            tri_mask = (types == 3)
            eh[tri_mask, 1, 1, :] = -10**5

        end = time.time()
        time_eh._updatetime(delta_t=end - start)
        return eh


    def find_sh_vec(self, s, go):
        """
        Generalised stresses from the layer stresses (thickness integration of the section model of the sampling,
        SigSimulator.find_sh_vec): [n, m] = (t / nl) * sum_l [s_l, -z_l * s_l], plus the transverse shear forces
        [q_x, q_y] = (t / nl) * sum_l [txz_l, tyz_l].

        Args:
            s   (np.arr): layer stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5); only the unperturbed real
                          part s[..., 0, :].real is used (the complex-step perturbations are needed for D only)
            go  (int):    Gauss order

        Returns:
            sh  (np.arr): generalised stresses, shape (nel, go, go, 8); triangles: sentinel -1e5 at Gauss point (1, 1)
        """
        nel = len(self.ELEMENTS[:, 0])
        sec = self._section()
        nlk = sec.constants['n_layer']
        t_s = self._samples(go)                                           # (n, 1)

        s_flat = np.transpose(s, (0, 2, 3, 1, 4, 5)).reshape(nel * go * go, nlk, 3, 5)
        sh = np.zeros((nel * go * go, 8))
        sh[:, :6] = sec.find_sh_vec(s_flat[..., :3], cm_klij=3, t=t_s)   # in-plane: n, m
        sh[:, 6:8] = (t_s / nlk) * s_flat[:, :, 0, 3:5].real.sum(axis=1)  # transverse shear: q_x, q_y
        sh = sh.reshape(nel, go, go, 8)

        # Triangle sentinel at (i=1, j=1) for go >= 2
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            sh[tri_mask, 1, 1, :] = -10**5
        return sh


    def find_sh_nn_vec(self, eh, go, model_dim, symm_predict=False, opt_time=False):
        """
        Generalised stresses from the NN (self.nn_model), one batched NN call for all Gauss points.

        Args:
            eh             (np.arr): generalised strains from find_eh_vec, shape (nel, go, go, 8)
            go             (int):    Gauss order
            model_dim:               accepted for signature parity (unused)
            opt_time       (bool):   stress and tangent from one CPU pass; tangent cached in self._D_cache
                                     for k_k_nn_num_vec.

        Returns:
            sh         (np.arr): generalised stresses, shape (nel, go, go, 8). The NN predicts
                                 the first 6 components (membrane + bending); the last 2
                                 (qx, qy) stay zero, matching find_sh_nn. Triangle (i=1, j=1)
                                 slots set to -1e5 for go >= 2.

        Restriction: all elements must have cm == 3 (nonlinear RC).
        """
        start = time.time()

        nel = len(self.ELEMENTS[:, 0])
        sh  = np.zeros((nel, go, go, 8))

        cm_arr = np.asarray(self.MATK["cm"])
        if (cm_arr != 3).any():
            raise NotImplementedError('find_sh_nn_vec: only cm == 3 (nonlinear RC) is supported.')

        # Sample-to-element mapping (n = k*go*go + i*go + j)
        n_tot = nel * go * go
        k_idx = np.repeat(np.arange(nel), go * go)
        ij    = np.tile(np.arange(go * go), nel)
        i_idx = ij // go
        j_idx = ij %  go

        # Triangle (i=1, j=1) sentinel positions
        types    = np.asarray(self.ELS[4])
        sentinel = (types[k_idx] == 3) & (i_idx == 1) & (j_idx == 1)        # (n_tot,)
        valid    = ~sentinel

        # Flatten eh: (nel, go, go, 8) -> (n_tot, 8)
        eh_flat = eh.reshape(n_tot, 8)

        chosen_model = self.nn_model

        # Single batched NN call on valid samples (cm == 3 path: first 6 components only).
        input_nn = self.append_geom_cols(eh_flat[valid, :6], k_idx[valid], chosen_model)   # (n_valid, 6 + n_geom)
        if opt_time:
            # stress and tangent from one CPU pass; the tangent is cached for the next k_k_nn_num_vec call at the
            # same strains (same NN version for sig and D, see load_path_run_utils.get_paths)
            mat_NN = make_NN_prediction(input_nn, predict='sig_D', model=chosen_model, symm_predict=symm_predict)
            self._D_cache = (input_nn.copy(), np.asarray(mat_NN['D']))
        else:
            mat_NN = make_NN_prediction(input_nn, predict='sig', model=chosen_model, symm_predict=symm_predict)
        sig_pred = np.asarray(mat_NN['sig'])                                # (n_valid, 6)

        # Scatter predictions back into the full (n_tot, 6) layout, then reshape to structured.
        sig_full = np.zeros((n_tot, 6))
        sig_full[valid] = sig_pred
        sh[..., :6] = sig_full.reshape(nel, go, go, 6)

        # Out-of-plane shear forces (qx, qy) -- analytical, matching ConstitutiveLaws.sigma_shear:
        #   txz_layer = 5/6 * G * gxz,   tyz_layer = 5/6 * G * gyz   (G = Ec / (2*(1+vc)))
        # gxz, gyz are constant through thickness (no z-dependence in find_e_klij's S), so the
        # thickness integral is just (5/6 * G * gxz) * t for qx (and analogously for qy).
        Ec    = np.asarray(self.MATK["Ec"], dtype=float)                # (nel,)
        vc    = np.asarray(self.MATK["vc"], dtype=float)                # (nel,)
        G     = Ec / (2.0 * (1.0 + vc))                                  # (nel,)
        t_arr = np.asarray(self.GEOMK["t"], dtype=float)                 # (nel,)
        coeff = (5.0 / 6.0) * G * t_arr                                  # (nel,) = (5/6) * G * t
        sh[..., 6] = coeff[:, None, None] * eh[..., 6]                   # qx
        sh[..., 7] = coeff[:, None, None] * eh[..., 7]                   # qy

        # Triangle (i=1, j=1) sentinel for go >= 2 (overwrites all 8 components, including
        # the qx, qy we just wrote -- matches find_sh_nn's behaviour).
        if go >= 2:
            tri_mask = (types == 3)
            sh[tri_mask, 1, 1, :] = -10**5

        end = time.time()
        time_sh._updatetime(delta_t=end - start)
        return sh


    def find_e_klij_vec(self, eh):
        """
        Layer strains from the generalised strains (section model of the sampling, SigSimulator.find_e_vec):
        [ex, ey, gxy] = membrane strain - z * curvature, plus the transverse shear strains [gxz, gyz] (constant over
        the thickness). No initial strains e0 (see find_e_vec).

        Args:
            eh  (np.arr): generalised strains, shape (nel, go, go, 8)

        Returns:
            e_klij  (np.arr): layer strains [ex, ey, gxy, gxz, gyz], shape (nel, nlk, go, go, 5)
        """
        nel, go = eh.shape[0], eh.shape[1]
        sec = self._section()
        nlk = sec.constants['n_layer']

        eh_flat = eh.reshape(nel * go * go, 8)
        e_in = sec.find_e_vec(eh_flat[:, :6], t=self._samples(go))                 # (n, nlk, 3)
        e_sh = np.broadcast_to(eh_flat[:, None, 6:8], (nel * go * go, nlk, 2))     # (n, nlk, 2)
        e = np.concatenate([e_in, e_sh], axis=-1).reshape(nel, go, go, nlk, 5)
        return np.ascontiguousarray(np.transpose(e, (0, 3, 1, 2, 4)))              # (nel, nlk, go, go, 5)


    def find_e_vec(self, e0, eh, go):
        """
        Vectorised version of find_e. Drop-in replacement: same args / return as find_e.
        Works for any Gauss order; in particular both go = 1 and go = 2.

        Args:
            e0  (np.arr): initial strains, shape (nel, nlk, go, go, 5)
            eh  (np.arr): generalised strains, shape (nel, go, go, 8)
            go  (int):    Gauss order

        Returns:
            e, ex, ey, gxy, e1, e3, th — same shapes / semantics as find_e

        Delegates the S @ eh contraction to find_e_klij_vec, subtracts e0, and computes the
        principal strain components in bulk. Triangle (i=1, j=1) sentinel is applied via
        boolean masking, matching find_e.
        """
        # t0 = time.perf_counter()
        e_klij_all = self.find_e_klij_vec(eh)                                # (nel, nlk, go, go, 5)
        e = e_klij_all - e0

        ex  = e[..., 0]
        ey  = e[..., 1]
        gxy = e[..., 2]

        # Vectorised e_principal
        r  = 0.5 * np.sqrt((ex - ey) ** 2 + gxy ** 2)
        m  = 0.5 * (ex + ey)
        e1 = m + r
        e3 = m - r
        with np.errstate(divide='ignore', invalid='ignore'):
            th_nz = np.arctan(gxy / (2.0 * (e1 - ex)))
        th_z = np.where(ex > ey, np.pi / 2,
               np.where(ex < ey, 0.0, np.pi / 4))
        th   = np.where(np.abs(gxy) > 0, th_nz, th_z)

        # Triangle sentinel at (i=1, j=1); only reachable when go >= 2.
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            SENT = -10**5
            e  [tri_mask, :, 1, 1, :] = SENT
            ex [tri_mask, :, 1, 1]    = SENT
            ey [tri_mask, :, 1, 1]    = SENT
            gxy[tri_mask, :, 1, 1]    = SENT
            e1 [tri_mask, :, 1, 1]    = SENT
            e3 [tri_mask, :, 1, 1]    = SENT
            th [tri_mask, :, 1, 1]    = SENT

        # t1 = (time.perf_counter() - t0)
        # print(f'Calculated strains e, principals in {t1/60:.2f} min.')
        return e, ex, ey, gxy, e1, e3, th


    def find_e0_vec(self, go):
        """
        Vectorised version of find_e0.

        Args:
            go  (int): Gauss order

        Returns:
            e   (np.arr): initial generalised strains, shape (nel, nlk, go, go, 5)
            ex  (np.arr): in-plane strain x-component,  shape (nel, nlk, go, go)
            ey  (np.arr): in-plane strain y-component,  shape (nel, nlk, go, go)
            gxy (np.arr): in-plane shear strain,        shape (nel, nlk, go, go)
            e1  (np.arr): max principal strain,         shape (nel, nlk, go, go)
            e3  (np.arr): min principal strain,         shape (nel, nlk, go, go)
            th  (np.arr): principal direction angle,    shape (nel, nlk, go, go)

        In the original find_e0, e_klij = [1,1,0,0,0]*ecs[k] is independent of layer l and Gauss point (i,j),
        so ex == ey == ecs[k] and gxy == 0 throughout. Under those conditions e_principal collapses to
        e1 = e3 = ecs[k] and th = pi/4, which lets us skip per-element calls entirely. Triangle elements get
        the -1e5 sentinel at the (i=1, j=1) Gauss slot, matching the original.
        """
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        ecs = np.asarray(self.MATK["ecs"], dtype=float)                          # (nel,)
        base = np.array([1.0, 1.0, 0.0, 0.0, 0.0])                                # (5,)

        e   = np.broadcast_to(ecs[:, None, None, None, None] * base,
                              (nel, nlk, go, go, 5)).copy()
        ex  = np.broadcast_to(ecs[:, None, None, None],
                              (nel, nlk, go, go)).copy()
        ey  = ex.copy()
        gxy = np.zeros((nel, nlk, go, go))

        e1 = ex.copy()
        e3 = ex.copy()
        th = np.full((nel, nlk, go, go), np.pi / 4)

        # Triangle sentinel at (i=1, j=1); only reachable when go >= 2.
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            SENT = -10**5
            e  [tri_mask, :, 1, 1, :] = SENT
            ex [tri_mask, :, 1, 1]    = SENT
            ey [tri_mask, :, 1, 1]    = SENT
            gxy[tri_mask, :, 1, 1]    = SENT
            e1 [tri_mask, :, 1, 1]    = SENT
            e3 [tri_mask, :, 1, 1]    = SENT
            th [tri_mask, :, 1, 1]    = SENT

        return e, ex, ey, gxy, e1, e3, th


    def find_sh0_vec(self, s0):
        """
        Internal layer stresses integrated over the thickness: sh0 = sum_l S_l^T [s0_l, 0, 0] dz with
        S_l^T [sx, sy, txy, 0, 0] = [sx, sy, txy, -z_l sx, -z_l sy, -z_l txy, 0, 0].

        Args:
            s0  (np.arr): internal layer stresses [sx, sy, txy], shape (nel, nlk, go, go, 3)

        Returns:
            np.arr: sh0, shape (nel, go, go, 8)
        """
        nel, nlk = s0.shape[0], s0.shape[1]
        t_arr   = np.asarray(self.GEOMK["t"],   dtype=float)                # (nel,)
        nlk_arr = np.asarray(self.GEOMK["nlk"], dtype=int)                  # (nel,)
        l_idx   = np.arange(nlk)
        z  = -t_arr[:, None] / 2 + (2 * l_idx[None, :] + 1) * t_arr[:, None] / (2 * nlk_arr[:, None])   # (nel, nlk)
        dz = (t_arr / nlk_arr)[:, None] * (l_idx[None, :] < nlk_arr[:, None])                             # (nel, nlk)
        s0w = s0 * dz[:, :, None, None, None]                                                           # (nel, nlk, go, go, 3)
        sh0 = np.zeros(s0.shape[:1] + s0.shape[2:4] + (8,))
        sh0[..., 0:3] = s0w.sum(axis=1)
        sh0[..., 3:6] = (-z[:, :, None, None, None] * s0w).sum(axis=1)
        return sh0

    """---------------------------------------------- Find Stresses -----------------------------------------------------"""


    def find_s_vec(self, e, s_prev, go, dolinel=False):
        """
        Vectorised drop-in for find_s. Same signature as find_s. Per-Gauss-point computation
        is preserved (every (k, i, j) becomes one sample in a flat batch of size nel*go*go
        passed to ConstitutiveLaws), then reshaped back to the structured layout matching e.

        Output:
            s.shape = (nel, nlk, go, go, 3, 5)        # complex128
                axes 0..3  : same (nel, nlk, go, go) prefix as e
                axis 4     : complex-step perturbation index (mirrors find_s's last axis)
                axis 5     : stress component (sx, sy, txy, txz, tyz)

        find_s's object-array entries (instances with .sx, .sy, .txy, .txz, .tyz attributes)
        become numerical entries indexed by axis 5. Imaginary parts of axis 4 carry the
        complex-step derivative d s / d(ex, ey, gxy) for the in-plane components; out-of-plane
        shear (txz, tyz) is linear in (gxz, gyz) and is not perturbed.

        Branching on dolinel:
            * dolinel = False (cm_klij = 3): 3 complex-step perturbations -> s[..., p, :] for p in 0..2.
            * dolinel = True  (cm_klij = 1): single linear-elastic evaluation -> s[..., 0, :] only.

        Mesh constraints (asserted upfront):
            * nu is uniform across elements (single MATK value); the bar diameter GEOMK['dx'] == GEOMK['dy']
              is uniform across elements and layers.
            * Concrete parameters (Ec, fct, fcp, tb0, tb1, ec0) MAY vary per element (per-region fc,
              scenario-300 sub-panels): if non-uniform they are passed per-sample via `geom['mat']`.
            * Steel parameters (fsy, fsu, Es, Esh) MAY vary per element (per-region fs, scenario-300
              sub-panels): if non-uniform they are passed per-sample via `geom['mat_s']`.
            * Reinforcement ratios GEOMK['rhox'], GEOMK['rhoy'] MAY vary per element (scenario-300
              sub-panels): if non-uniform they are passed per-sample to ConstitutiveLaws via `geom`.
            * Steel material is the same in x and y within each element (Esx == Esy, fsyx == fsyy, ...).
            * cm_k in {1, 3} (ConstitutiveLaws).
        Element-0 values are then used for ConstitutiveLaws. Mixed meshes need the masked-op
        rewrite in constitutive_laws.py; raise NotImplementedError on attempt.

        s_prev is accepted for signature parity with find_s; ConstitutiveLaws does not consume
        it for cm_klij in {1, 3} (no fixed-crack history), so it is otherwise unused.
        """
        # t0 = time.perf_counter()

        # 0 Setup ------------------------------------------------------------------------------
        nel   = len(self.ELEMENTS[:, 0])
        nlk   = max(self.GEOMK["nlk"])
        n_tot = nel * go * go

        # 1 Uniform-mesh assertions ------------------------------------------------------------
        def _uniform_MATK(key):
            arr = np.asarray(self.MATK[key]).reshape(-1)
            if arr.size > 1 and not np.all(arr == arr[0]):
                raise NotImplementedError(
                    f"find_s_vec: MATK[{key!r}] varies across elements; ConstitutiveLaws "
                    f"requires uniform material. Use find_s, or rewrite the masked ops in "
                    f"constitutive_laws.py to broadcast (n_tot, nl, 1) per-sample params."
                )
            return float(arr[0])

        def _uniform_GEOMK_layer(key):
            arr = np.asarray(self.GEOMK[key])
            if arr.ndim == 2 and arr.shape[0] > 1 and not np.all(arr == arr[0:1, :]):
                raise NotImplementedError(
                    f"find_s_vec: GEOMK[{key!r}] varies across elements; not supported."
                )
            return arr[0, :] if arr.ndim == 2 else arr

        def _bar_diameter():
            # bar diameter for the TCM (GEOMK dx / dy, as GEOM[2] / GEOM[3] in find_s); not MATK['Dmax'] (max. grain
            # size, aggregate interlock only). ConstitutiveLaws has a single D -> uniform and dx == dy required.
            dx, dy = np.asarray(self.GEOMK["dx"], dtype=float), np.asarray(self.GEOMK["dy"], dtype=float)
            if not (np.all(dx == dx.flat[0]) and np.array_equal(dx, dy)):
                raise NotImplementedError("find_s_vec: ConstitutiveLaws needs a single bar diameter "
                                          "(GEOMK['dx'] == GEOMK['dy'], uniform over elements and layers).")
            return float(dx.flat[0])

        cm_arr = np.asarray(self.MATK["cm"])
        if not np.isin(cm_arr, [1, 3]).all():
            raise NotImplementedError("find_s_vec: only cm in {1, 3} (ConstitutiveLaws).")

        # 2 Decide cm_klij (linear elastic override via dolinel) ------------------------------
        cm_klij = 1 if dolinel else int(cm_arr.reshape(-1)[0])
        if cm_klij not in (1, 3):
            raise NotImplementedError(
                f"find_s_vec: cm_klij={cm_klij} not supported by ConstitutiveLaws."
            )

        # 3 Build mat_dict and constants from element 0 ---------------------------------------
        # ConstitutiveLaws stores a single steel material (Es, Esh, fsy, fsu) regardless of
        # direction (see constitutive_laws.py:43-48). Anisotropic reinforcement *ratios*
        # (rhox != rhoy) are fine and handled separately; what we forbid here is anisotropic
        # steel *material* properties within an element (they MAY vary between elements).
        steel_keys = {'Es': ("Esx", "Esy"), 'Esh': ("Eshx", "Eshy"),
                      'fsy': ("fsyx", "fsyy"), 'fsu': ("fsux", "fsuy")}
        steel_k = {}
        for label, (x_key, y_key) in steel_keys.items():
            x_arr = np.asarray(self.MATK[x_key], dtype=float).reshape(-1)     # (nel,)
            y_arr = np.asarray(self.MATK[y_key], dtype=float).reshape(-1)
            if not np.array_equal(x_arr, y_arr):
                raise NotImplementedError(
                    f"find_s_vec: anisotropic steel material ({x_key} != {y_key}) not supported "
                    f"by ConstitutiveLaws -- it collapses both directions to a single {label!r} "
                    f"value. Note: anisotropic reinforcement *ratios* (rhox != rhoy) ARE "
                    f"supported and handled separately."
                )
            steel_k[label] = x_arr
        steel_uniform = all(np.all(v == v[0]) for v in steel_k.values())

        # Concrete parameters MAY vary per element (per-region fc, scenario 300): uniform meshes keep the
        # scalar path; otherwise element-0 values go into mat_dict and the per-element values are passed
        # per-sample via geom['mat'] below (ConstitutiveLaws broadcasts / masks them).
        conc_keys = ('Ec', 'fct', 'fcp', 'tb0', 'tb1', 'ec0')
        conc_k = {key: np.asarray(self.MATK[key], dtype=float).reshape(-1) for key in conc_keys}   # (nel,)
        conc_uniform = all(np.all(v == v[0]) for v in conc_k.values())
        mat_dict = {
            'ect': 0.0,                                                  # matches MAT[5] = 0*fct/Ec in find_s
            'ec0': float(conc_k['ec0'][0]),
            'Ec':  float(conc_k['Ec'][0]),
            'fct': float(conc_k['fct'][0]),
            'fcp': float(conc_k['fcp'][0]),
            'tb0': float(conc_k['tb0'][0]),
            'tb1': float(conc_k['tb1'][0]),
            'Es':  float(steel_k['Es'][0]),
            'Esh': float(steel_k['Esh'][0]),
            'fsy': float(steel_k['fsy'][0]),
            'fsu': float(steel_k['fsu'][0]),
            'ts':  bool(self.MATK.get('ts', True)),                     # False: CMM- (no tension stiffening)
            'cmcc': int(self.MATK.get('cmcc', 1)),                      # concrete law: 1 fib, 2 Kaufmann (1998)
            'cmcs': int(self.MATK.get('cmcs', 1)),                      # softening: 1 CSFM, 2 Kaufmann (1998)
        }
        # Reinforcement: uniform meshes keep the existing constants path unchanged. Only when rho
        # varies across elements (scenario-300 sub-panels) do we pass per-element rho via `geom`.
        rhox_layers = np.asarray(self.GEOMK['rhox'], dtype=float)               # (nel, nlk)
        rhoy_layers = np.asarray(self.GEOMK['rhoy'], dtype=float)               # (nel, nlk)
        rho_uniform = (np.all(rhox_layers == rhox_layers[0:1]) and
                       np.all(rhoy_layers == rhoy_layers[0:1]))
        constants = {
            'nu':      _uniform_MATK("vc"),
            'n_layer': nlk,
            'rho_x':   _uniform_GEOMK_layer("rhox") if rho_uniform else rhox_layers[0],
            'rho_y':   _uniform_GEOMK_layer("rhoy") if rho_uniform else rhoy_layers[0],
            'D':       _bar_diameter(),
        }

        # 4 Flatten e: (nel, nlk, go, go, 5) -> (n_tot, nlk, 5) -------------------------------
        # All 5 strain components are kept; ConstitutiveLaws uses the first 3 for the in-plane
        # constitutive law and the last 2 (gxz, gyz) for the analytical out-of-plane shear.
        # Sample order n = k*go*go + i*go + j.
        e_in   = np.transpose(e, (0, 2, 3, 1, 4))                            # (nel, go, go, nlk, 5)
        e_flat = e_in.reshape(n_tot, nlk, 5)                                  # (n_tot, nlk, 5)

        # Identify triangle (i=1, j=1) sentinel samples: find_e_vec writes -1e5 there. We must
        # exclude them from ConstitutiveLaws -- otherwise the garbage strain feeds into divides
        # and emits "invalid value encountered in divide" warnings. The scalar pipeline skips
        # those samples via an if-branch in find_s, so it never sees them either.
        if go >= 2:
            k_idx    = np.repeat(np.arange(nel), go * go)
            ij       = np.tile(np.arange(go * go), nel)
            i_idx    = ij // go
            j_idx    = ij %  go
            types    = np.asarray(self.ELS[4])
            sentinel = (types[k_idx] == 3) & (i_idx == 1) & (j_idx == 1)
        else:
            sentinel = np.zeros(n_tot, dtype=bool)
        valid   = ~sentinel
        e_valid = e_flat[valid]                                               # (n_valid, nlk, 5)

        # Non-uniform rho only: expand per-element per-layer rho to per-sample (n_valid, nlk, 1)
        # and pass via `geom`. Sample order is n = k*go*go + i*go + j, so k = n // (go*go).
        # Uniform meshes keep geom_rho = None -> ConstitutiveLaws uses the constants path as before.
        geom_rho = None
        k_of_n   = np.repeat(np.arange(nel), go * go)
        if not rho_uniform:
            geom_rho = {'rho_x': rhox_layers[k_of_n][valid][:, :, None],
                        'rho_y': rhoy_layers[k_of_n][valid][:, :, None]}
        # Non-uniform concrete only (per-element fc): per-sample (n_valid, 1, 1) material via geom['mat'].
        if not conc_uniform:
            geom_rho = geom_rho or {}
            k_valid  = k_of_n[valid]
            geom_rho['mat'] = {key: conc_k[key][k_valid][:, None, None] for key in conc_keys}
            geom_rho['mat']['ect'] = np.zeros_like(geom_rho['mat']['Ec'])
        # Non-uniform steel only (per-element fs): per-sample (n_valid, 1, 1) steel via geom['mat_s'].
        if not steel_uniform:
            geom_rho = geom_rho or {}
            k_valid  = k_of_n[valid]
            geom_rho['mat_s'] = {key: steel_k[key][k_valid][:, None, None] for key in steel_keys}

        # 5 Per-Gauss-point evaluation via ConstitutiveLaws -----------------------------------
        s_flat = np.zeros((n_tot, nlk, 3, 5), dtype=np.complex128)
        EPS_J = 1e-16j

        # Suppress masked-out divide-by-zero / 0/0 warnings inherent to vectorised constitutive
        # code: ConstitutiveLaws uses `np.where(cond, a/b, fallback)` patterns where numpy
        # evaluates a/b even on the False branch (and the result is discarded). The scalar
        # `stress` class avoids this by using Python if/else, so the unvectorised pipeline
        # doesn't emit the warning.
        with np.errstate(invalid='ignore', divide='ignore'):
            if cm_klij == 3:
                # 3 complex-step perturbations -> imag(s[..., p, :]) is d s / d e_p for p in {0,1,2}
                # Only the first 3 strain components (ex, ey, gxy) are perturbed -- matches find_s.
                for p, pert in enumerate([(EPS_J, 0, 0, 0, 0),
                                           (0, EPS_J, 0, 0, 0),
                                           (0, 0, EPS_J, 0, 0)]):
                    e_p = e_valid + np.array(pert, dtype=np.complex128)
                    cl  = ConstitutiveLaws(e_p, constants, mat_dict, cm_klij=cm_klij, geom=geom_rho,
                                           dtype=np.complex128, verbose=False)
                    s_flat[valid, :, p, :] = cl.out(shear_out=True).squeeze(-1)   # (n_valid, nlk, 5)
            else:  # cm_klij == 1, linear elastic: single eval, only slot 0
                cl = ConstitutiveLaws(e_valid, constants, mat_dict, cm_klij=cm_klij, geom=geom_rho,
                                      dtype=np.complex128, verbose=False)
                s_flat[valid, :, 0, :] = cl.out(shear_out=True).squeeze(-1).astype(np.complex128)

        # 6 Reshape back to structured (nel, nlk, go, go, 3, 5) -------------------------------
        s = s_flat.reshape(nel, go, go, nlk, 3, 5)
        s = np.transpose(s, (0, 3, 1, 2, 4, 5))                              # (nel, nlk, go, go, 3, 5)
        s = np.ascontiguousarray(s)

        # 7 Triangle sentinel at (i=1, j=1) — only reachable for go >= 2 ----------------------
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            s[tri_mask, :, 1, 1, :, :] = -10**5

        # t1 = (time.perf_counter() - t0)
        # print(f'Calculated layer stresses s in {t1/60:.2f} min.')
        return s


    def find_s0_vec(self, go):
        """
        Vectorised version of find_s0.

        Args:
            go  (int): Gauss order

        Returns:
            s0  (np.arr): residual element stresses [s_x, s_y, t_xy], shape (nel, nlk, go, go, 3)

        find_s0_klij(k, l, i, j) only depends on k via -Ec[k]*ecs[k]/(1+vc[k]^2) * [1,1,0], so we
        build that scalar coefficient once per element and broadcast over (l, i, j). The original
        find_s0 has no triangle-Gauss-point sentinel, so none is applied here either.
        """
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        Ec  = np.asarray(self.MATK["Ec"],  dtype=float)
        ecs = np.asarray(self.MATK["ecs"], dtype=float)
        vc  = np.asarray(self.MATK["vc"],  dtype=float)

        coeff = -Ec * ecs / (1.0 + vc ** 2)                       # (nel,)
        base  = np.array([1.0, 1.0, 0.0])                         # (3,)

        s0 = np.broadcast_to(coeff[:, None, None, None, None] * base,
                             (nel, nlk, go, go, 3)).copy()

        return s0


    def find_ss_vec(self, e, cmk):
        """
        Vectorised drop-in for find_ss. Signature differs: takes strain `e` (steel stress is
        a function of strain, not of the sigma stresses carried by the vec `s` array).

        Args:
            e    (np.arr): strain from find_e_vec, shape (nel, nlk, go, go, 5)
            cmk  (array-like): per-element cm identifier, shape (nel,)

        Returns:
            ssx, ssy, spx, spy : shape (nel, nlk, go, go). spx, spy are sentinel everywhere
            (CFRP is not implemented in ConstitutiveLaws, matching its "skipped" comment).

        Constitutive: bilinear steel law -- matches find_ss exactly in compression and
        elastic tension. For cracked-tension where the scalar pipeline uses ssr (TCM), this
        returns the bare bilinear value. ssx, ssy are post-processing outputs (the FEM solve
        does not consume them), so the simplification is generally fine.

        Sentinel positions (filled with -1e-5, matching find_ss):
            * cm < 1.5 (linear elastic, no reinforcement law)
            * triangle (i=1, j=1) for go >= 2
        """
        nel = e.shape[0]
        nlk = e.shape[1]
        go  = e.shape[2]

        SENT = -1e-5
        ssx = np.full((nel, nlk, go, go), SENT)
        ssy = np.full((nel, nlk, go, go), SENT)
        spx = np.full((nel, nlk, go, go), SENT)
        spy = np.full((nel, nlk, go, go), SENT)

        cm_arr   = np.asarray(cmk)
        valid_cm = (cm_arr >= 1.5)

        if valid_cm.any():
            # Per-element steel (per-region fs, scenario 300), broadcast over (nlk, go, go); x == y (see find_s_vec).
            Es  = np.asarray(self.MATK["Esx"],  dtype=float).reshape(-1)[:, None, None, None]   # (nel, 1, 1, 1)
            Esh = np.asarray(self.MATK["Eshx"], dtype=float).reshape(-1)[:, None, None, None]
            fsy = np.asarray(self.MATK["fsyx"], dtype=float).reshape(-1)[:, None, None, None]
            esy = fsy / Es

            def _ss_bilin(strain):
                return np.where(
                    strain >= esy,
                    fsy + Esh * (strain - esy),
                    np.where(strain <= -esy,
                             -fsy + Esh * (strain + esy),
                             strain * Es),
                )

            ssx_calc = _ss_bilin(e[..., 0])
            ssy_calc = _ss_bilin(e[..., 1])

            vmask = np.broadcast_to(valid_cm[:, None, None, None], (nel, nlk, go, go))
            ssx = np.where(vmask, ssx_calc, ssx)
            ssy = np.where(vmask, ssy_calc, ssy)

        # Triangle (i=1, j=1) sentinel for go >= 2
        if go >= 2:
            types    = np.asarray(self.ELS[4])
            tri_mask = (types == 3)
            ssx[tri_mask, :, 1, 1] = SENT
            ssy[tri_mask, :, 1, 1] = SENT
            spx[tri_mask, :, 1, 1] = SENT
            spy[tri_mask, :, 1, 1] = SENT

        return ssx, ssy, spx, spy


    """--------------------------------------------- Solve & Control ----------------------------------------------------"""
    def solve_sys_vec(self, B, fe, cDOF, cVAL, cmk, e, s, opt_time=False):
        """
        Solve the system with the NLFEA tangent stiffness (k_glob_vec; opt_time: one B^T D B product per element,
        see k_k_vec).

        Returns:
            tuple: u (displacement increment), D_tot (per-Gauss-point 8x8 tangents)
        """
        K, D_tot = self.k_glob_vec(B, e, s, cmk, self.gauss_order, opt_time=opt_time)

        Kcond  = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        start = time.time()
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        end = time.time()
        time_Kinv._updatetime(delta_t=end - start)

        return u, D_tot


    def solve_sys_nn_num_vec(self, B, fe, cDOF, cVAL, cmk, eh, sh, e, s, model_dim, scenario,
                             symm_predict=False, opt_time=False):
        """
        Solve the system with the NN tangent (k_glob_nn_num_vec, opt_time: see k_k_nn_num_vec).

        Restrictions (inherited from k_k_nn_num_vec):
            * model_dim == 'THREEDIM' only.
            * cm == 3 only.

        Returns:
            tuple: u (displacement increment), D_tot (per-Gauss-point 8x8 tangents)
        """
        K, D_tot = self.k_glob_nn_num_vec(B, e, s, eh, sh, cmk, self.gauss_order,
                                          model_dim, scenario,
                                          symm_predict=symm_predict, opt_time=opt_time)

        Kcond  = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        start = time.time()
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        end = time.time()
        time_Kinv._updatetime(delta_t=end - start)

        return u, D_tot


    def solve_0(self, B,fe,e, cDOF,cVAL, go):
        """
        Linear elastic solution (start of the iteration): initial stiffness with the linear elastic layer tangent
        (cm = 1, E = Ec, nu = vc of every element, see k_k_vec).

        Args:
            B           (dict):   B-matrix container of find_b
            fe          (np.arr): force vector, shape (6 * n_nodes, 1)
            e           (np.arr): layer strains (zeros), shape (nel, nlk, go, go, 5)
            cDOF, cVAL  (np.arr): constrained DOFs and their values
            go          (int):    Gauss order

        Returns:
            tuple: u (displacements of the free DOFs), D_tot (linear elastic 8x8 tangents, (nel, go, go, 8, 8))
        """
        nel = len(self.ELEMENTS[:, 0])
        s_lin = np.zeros(np.shape(e)[:4] + (3, 5))          # layer stresses are not needed for the elastic tangent
        K, D_tot = self.k_glob_vec(B, e, s_lin, np.ones(nel, dtype=int), go)
        Kcond = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        return u, D_tot

