"""Vectorised layer constitutive laws for reinforced concrete (in-plane stresses per layer).

Vectorised port of the layered RC material model of Stresses_mixreinf.py (A. Näsbom), V. Balmer, 2026.
Only the material options used by the sampler are implemented:
    cm_klij   = 1  linear elastic (used for the elastic reference), 3  nonlinear reinforced concrete
    cmcc      = 1  concrete in compression acc. to fib Model Code / CSFM, 2  parabola acc. to Kaufmann (1998)
    cmcs      = 1  compression softening acc. to CSFM,                    2  acc. to Kaufmann (1998)
    steel        tension chord model (TCM) with a bilinear bare-steel law
Cracks are rotating (crack direction = principal direction). cmcc / cmcs are read from mat_dict (default 1).
"""

from .config import USE_GPU
from .defcplx_np import cplx

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class ConstitutiveLaws():
    def __init__(self, e, constants, mat_dict, cm_klij = 3, geom = None):

        # General
        self.cm_klij = cm_klij          # material type: 1 - lin.el., 3 - nonlinear RC
        self.e = e*cplx(np.ones_like(e),np.zeros_like(e))   # complex layer strains, shape: (n_tot, nl, 3)

        # Material parameters
        self.ect = mat_dict['ect']
        self.ec0 = mat_dict['ec0']
        self.Ec = mat_dict['Ec']
        self.fct = mat_dict['fct']
        self.fcp = mat_dict['fcp']
        self.tb0 = mat_dict['tb0']
        self.tb1 = mat_dict['tb1']
        self.cmcc = int(mat_dict.get('cmcc', 1))     # concrete in compression: 1 - fib / CSFM, 2 - Kaufmann (1998)
        self.cmcs = int(mat_dict.get('cmcs', 1))     # compression softening:   1 - CSFM, 2 - Kaufmann (1998)
        if self.cmcc not in (1, 2) or self.cmcs not in (1, 2):
            raise NotImplementedError(f'cmcc = {self.cmcc}, cmcs = {self.cmcs}: only 1 and 2 are implemented.')

        self.Esx = mat_dict['Es']
        self.Esy = mat_dict['Es']
        self.Es = mat_dict['Es']
        self.Esh = mat_dict['Esh']
        self.fsy = mat_dict['fsy']
        self.fsu = mat_dict['fsu']

        self.v = constants['nu']
        self.nl = constants['n_layer']

        # Geometrical parameters
        ntot = self.e.shape[0]
        self.rho_x = np.broadcast_to(np.array(constants['rho_x'])[np.newaxis,:,np.newaxis], (ntot, self.nl, 1))
        self.rho_y = np.broadcast_to(np.array(constants['rho_y'])[np.newaxis,:,np.newaxis], (ntot, self.nl, 1))
        self.D     = constants['D']

        # per-sample geometry overrides (already-built per-layer arrays)
        geom = geom or {}
        if 'rho_x' in geom:
            self.rho_x = geom['rho_x']
        if 'rho_y' in geom:
            self.rho_y = geom['rho_y']

        # per-sample concrete parameters (from a per-sample fc column, interpolated in build_geom): (n,1,1) ->
        # broadcast to (ntot, nl, 1) so they can be masked like the strains. Scalars from mat_dict otherwise.
        if 'mat' in geom:
            for key in ('ect', 'ec0', 'Ec', 'fct', 'fcp', 'tb0', 'tb1'):
                setattr(self, key, np.broadcast_to(geom['mat'][key], (ntot, self.nl, 1)))

        # per-sample steel parameters (from a per-sample fs column, interpolated in build_geom), same broadcasting
        if 'mat_s' in geom:
            for key in ('fsy', 'fsu', 'Es', 'Esh'):
                setattr(self, key, np.broadcast_to(geom['mat_s'][key], (ntot, self.nl, 1)))
            self.Esx = self.Esy = self.Es

        # Other constants
        self.tole = 1e-9                # tolerance for the submodel / principal-direction switches
        self.ff = 0.01                  # residual concrete stiffness in cracked tension (numerical stability)
        self.ecsx = 0.0                 # shrinkage strain in x (not considered)
        self.ecsy = 0.0                 # shrinkage strain in y (not considered)
        self.lbd = 0.67                 # crack spacing factor lambda (srm = lambda * sr0)

    def principal(self, ex, ey, gxy):
        """
        Vectorised principal strains and direction.
        Args:
            ex, ey, gxy:  shape (n_tot, nl, 3)
        Returns:
            e1, e3:       shape (n_tot, nl, 3)
            th:           shape (n_tot, nl, 3)
            submodel:     shape (n_tot, nl, 3), int (1, 2, or 3)
        """
        # 1 Mohr's Circle
        re = 0.5 * np.sqrt((ex - ey)**2 + gxy**2)
        me = 0.5 * (ex + ey)
        e1 = me + re
        e3 = me - re

        # 2 Principal direction th
        # Default: atan(gxy / (2*(e1-ex)))
        denom = 2 * (e1 - ex)
        th = np.arctan(np.where(np.abs(denom) > 1e-10, gxy / denom, 0.0))

        # Override where e1 ≈ ex or e1 ≈ ey (axis-aligned frame); near_ey added for x/y symmetry.
        near_ex = np.abs(e1 - ex) < 1e-10
        near_ey = np.abs(e1 - ey) < 1e-10
        th = np.where(near_ex & (ex > ey),  np.pi/2 - 1e-10, th)
        th = np.where(near_ex & (ex < ey),  1e-10,            th)
        th = np.where(near_ey & (ey > ex),  1e-10,            th)
        th = np.where(near_ey & (ey < ex),  np.pi/2 - 1e-10, th)
        th = np.where(near_ex & (ex == ey), np.pi/4,          th)

        # 3 Submodel
        tole = self.tole
        threshold = self.ect * 0 + tole         
        submodel = np.where(
            (e1 > threshold) & (e3 > threshold), 3,
            np.where(
                (e1 <= threshold) & (e3 <= threshold), 1,
                2
            )
        ).astype(int)

        # 4 Concrete strains
        thc = th                                 # rotating cracks: thc = th
        ec3 = e3
        ec1 = np.where(e1 < tole, e1, 0.0)

        ecx = np.where(ec3 < 0, ec1 + (ec3 - ec1) * np.cos(thc)**2, 0.0)
        ecy = np.where(ec3 < 0, ec1 + (ec3 - ec1) * np.sin(thc)**2, 0.0)

        return e1, e3, th, submodel, ecx, ecy, ec1, ec3

    def _m(self, val, mask):
        """Material parameter at the masked locations: val[mask] for a per-sample (ntot, nl, 1) array
        (per-sample fc / fs), the scalar unchanged otherwise."""
        return val[mask] if getattr(val, 'ndim', 0) > 0 else val

    def _steel(self, mask):
        """Steel parameters (fsy, fsu, Es, Esh) at the masked locations (per-sample fs) or the scalars."""
        return tuple(self._m(getattr(self, key), mask) for key in ('fsy', 'fsu', 'Es', 'Esh'))

    ''' -------------------------------------CONSTITUTIVE LAWS-------------------------------------'''

    def fcs(self, e1):
        """ ---------------------- Calculation concrete strength with softening as a function of e1---------------------
        ----------------------------------------    INPUT (self.): -------------------------------------------------
        - fcp:          Cylinder compressive strength of concrete, int
        - e1:           Principal tensile strain, shape (ntot, nl, 1)

        - cmcs = 1:     softening according to CSFM (fib Eq.(7.3- 40) /SIA modified)
        - cmcs = 2:     softening according to Kaufmann (1998): fc = min(fcp^(2/3) / (0.4 + 30 max(e1, 0)), fcp)
        ----------------------------------------------- OUTPUT:-----------------------------------------------------
        - fc:           Concrete strength, shape (ntot, nl, 1)
        -------------------------------------------------------------------------------------------------------------"""

        if self.cmcs == 2:
            e1p = np.where(e1 > 0, e1, 0.0)
            fc = self.fcp ** (2 / 3) / (0.4 + 30 * e1p)
            return np.where(fc > self.fcp, self.fcp * np.ones_like(fc), fc)

        # Precompute scalar thresholds
        elim1 = 0.2 * self.ec0 / 0.34
        elim2 = 0.22 / 35.75
        kc2 = 1 / (1.2 + 55 * elim2)

        # Allocate kc array
        kc = np.empty_like(e1)

        # Masks for each branch
        mask_neg   = e1 < 0
        mask_low   = ~mask_neg & (e1 <= elim1)
        mask_high  = ~mask_neg & (e1 >= elim2)
        mask_mid   = ~mask_neg & ~mask_low & ~mask_high

        kc[mask_neg]  = 1.0
        kc[mask_low]  = 1 + (self._m(elim1, mask_low) - e1[mask_low]) * 0.001
        kc[mask_high] = 1 / (1.2 + 55 * e1[mask_high])
        elim1_mid = self._m(elim1, mask_mid)
        kc[mask_mid]  = 1 - (1 - kc2) * (e1[mask_mid] - elim1_mid) / (elim2 - elim1_mid)

        fc = kc * self.fcp

        return fc

    def sc(self, e, e1t):
        """ Concrete stress for the chosen compression law (cmcc = 1: fib / CSFM, 2: Kaufmann (1998)) """
        return self.sc_kfm(e, e1t) if self.cmcc == 2 else self.sc_fib(e, e1t)

    def sc_kfm(self, e, e1t):
        """ --------------------------------- Calculation concrete compressive stress-------------------------------
                Vectorised sc_kfm of Stresses_mixreinf (cmcc_klij = 2)
                ----------------------------------------    INPUT (self.): ---------------------------------------------
                - e:            Normal strain in regarded direction (shape: ntot, nl, 1)
                - e1t:          Principal tensile strain of the total strain state (shape: ntot, nl, 1)
                ----------------------------------------------- OUTPUT:-------------------------------------------------
                - sc3:          Concrete compressive stress according to Kaufmann (1998)
                                - e < 0, |e| < alpha*ec0:   sc3 = fc (e^2 + 2 e ec0) / ec0^2 (parabola, ec0 not softened)
                                - e <= -alpha*ec0:          linear with the slope of the parabola at alpha*ec0
                                - 0 < e < ect:              sc3 = Ec*e linear elastic in tension
                                - e >= ect:                 sc3 = "0" (ff*e for numerical stability)
            ---------------------------------------------------------------------------------------------------------"""
        fc = self.fcs(e1t)
        ec0 = self.ec0 * np.ones_like(fc)
        alpha = 0.99

        sc3 = np.empty_like(e)

        # Masks
        mask_comp       = e < 0
        mask_inner      = mask_comp & (np.abs(e) < alpha * ec0)
        mask_outer      = mask_comp & ~mask_inner
        mask_elastic    = ~mask_comp & (e < self.ect)
        mask_tension    = ~mask_comp & ~mask_elastic

        # Compressive inner (parabolic branch)
        e_in, ec0_in = e[mask_inner], ec0[mask_inner]
        sc3[mask_inner] = fc[mask_inner] * (e_in ** 2 + 2 * e_in * ec0_in) / ec0_in ** 2

        # Compressive outer (linear continuation)
        fc_out, ec0_out = fc[mask_outer], ec0[mask_outer]
        sc3[mask_outer] = (fc_out * (alpha ** 2 - 2 * alpha)
                           + fc_out * (2 * alpha - 2) / ec0_out * (-e[mask_outer] - alpha * ec0_out))

        # Elastic (tension near zero)
        sc3[mask_elastic] = self._m(self.Ec, mask_elastic) * e[mask_elastic]

        # Tension softening
        sc3[mask_tension] = self.ff * e[mask_tension]

        return sc3

    def sc_fib(self, e, e1t):
        """ --------------------------------- Calculation concrete compressive stress-------------------------------
                ----------------------------------------    INPUT (self.): ---------------------------------------------
                - e:            Normal strain in regarded direction (shape: ntot, nl, 1)
                - e1t:          Principal tensile strain of the total strain state (shape: ntot, nl, 1)
                ----------------------------------------------- OUTPUT:-------------------------------------------------
                - sc3:          Concrete compressive stress according to fib model code
                                - e < 0:         sc3 = f(e) according to compression parabola
                                - 0 < e < ect:   sc3 = Ec*e linear elastic in tension
                                - e > ect:       sc3 = "0" (ff*e for numerical stability)
            ---------------------------------------------------------------------------------------------------------"""
        # softened strength and the matching strain at peak, per sample (ntot, nl, 1)
        fc = self.fcs(e1t)
        ec0 = self.ec0 * fc / self.fcp
        k = self.Ec / (fc / ec0)
        k = np.where(k < 1, 1.0, k)     # k cannot be smaller than 1! 1 means linear elastic

        sc3 = np.empty_like(e)

        # Masks
        mask_comp       = e < 0
        mask_inner      = mask_comp & (np.abs(e) < ec0)
        mask_outer      = mask_comp & ~mask_inner
        mask_elastic    = ~mask_comp & (e <= self.ect)
        mask_tension    = ~mask_comp & ~mask_elastic

        # Compressive inner (parabolic branch)
        eta = -e[mask_inner] / ec0[mask_inner]
        k_in = k[mask_inner]
        sc3[mask_inner] = -fc[mask_inner] * (k_in * eta - eta ** 2) / (1 + (k_in - 2) * eta)

        # Compressive outer (post-peak branch)
        sc3[mask_outer] = -fc[mask_outer] + (e[mask_outer] + ec0[mask_outer]) / 1000

        # Elastic (tension/compression near zero)
        sc3[mask_elastic] = self._m(self.Ec, mask_elastic) * e[mask_elastic]

        # Tension softening
        sc3[mask_tension] = self.ff * e[mask_tension]

        return sc3
    
    def ssr(self, e, ec, ecs, srm, rho, mask):
        """ ------------------------------------------- Define Output---------------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - e:            strain at integration point in one layer, just at locations where there is tension-tension and rho != 0
                            (shape: n_tot, nl*mask, 3)
            - ec:           output from principal (principal concrete strains?), just at locations where there is tension-tension and rho != 0
                            (shape: n_tot, nl*mask, 1)
            - ecs:          shrinkage strain (currently zero)
            - sr:           crack spacing
            - rho:          rho only at given locations in element and for one dimension (shape: n_tot, nl*mask, 1)
            - mask:         location mask the other inputs were sliced with (shape: n_tot, nl, 1); used to slice
                            the per-sample concrete (tb0, tb1, Ec) and steel (fsy, fsu, Es, Esh) parameters consistently
            NOTE: Only for cms_klij = 1 (bilinear + TCM)
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - ss:           steel stressses
        -------------------------------------------------------------------------------------------------------------"""

        tb0, tb1, Ec = self._m(self.tb0, mask), self._m(self.tb1, mask), self._m(self.Ec, mask)
        st = self._steel(mask)
        ssr = self.ssr_tcm_bilin(e+ecs, ec+ecs, srm, rho, tb0, tb1, Ec, st)
        # ssr shape: (n_tot, nl*mask, 1)

        # catch yielding in compression (bare steel law)
        comp_yield_mask = ssr < -st[0]
        if np.any(comp_yield_mask):
            ssr_comp = self.ss_bilin(e, st)
            ssr = np.where(comp_yield_mask, ssr_comp, ssr)

        return ssr

    def ss_bilin(self, e, st):
        """ ------------------------ Calculation of bare reinforcing steel stress: bilinear-----------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - e:            Strains only at given layers and for one dimension (ntot, nl*masks, 1)
            - st:           steel parameters (fsy, fsu, Es, Esh) at the same locations (scalars or shaped like e),
                            see _steel
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - ss:           Steel stress
        -------------------------------------------------------------------------------------------------------------"""
        fsy, _, Es, Esh = st
        esy = fsy / Es

        ss = np.empty_like(e)

        mask_elastic = np.abs(e) <= esy
        mask_tension = ~mask_elastic & (e > 0)
        mask_compress = ~mask_elastic & (e < 0)

        ss[mask_elastic]  = e[mask_elastic] * self._m(Es, mask_elastic)
        ss[mask_tension]  = self._m(fsy, mask_tension) + self._m(Esh, mask_tension) * (e[mask_tension] - self._m(esy, mask_tension))
        ss[mask_compress] = -self._m(fsy, mask_compress) + self._m(Esh, mask_compress) * (e[mask_compress] + self._m(esy, mask_compress))

        return ss

    def ssr_seelhofer(self, e, ec, srm, rho, tb0, tb1, Ec, st):
        """ --------------------- Calculate Steel Stress with Seelhofer for non-stabilised crack -----------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e: normal strain
                - ec: axial concrete compressive strain from mohr's circle of concrete strains
                - srm: Crack spacing: srm = lambda * sr0
                - rho:          rho only at given locations in element and for one dimension (shape: n_tot, nl*mask, 1)
                - tb0, tb1, Ec: bond stresses / concrete modulus at the same locations (scalar or shaped like rho)
                - st:           steel parameters (fsy, fsu, Es, Esh) at the same locations (scalar or shaped like rho)
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ssr: steel/CFRP stress at the crack according to Seelhofer
            ---------------------------------------------------------------------------------------------------------"""
        fsy, _, Es, Esh = st

        # 2 Elastic crack element
        a = 4 * tb0 / (Es * self.D)
        b = 4 * tb0 / (Ec * self.D) * rho / (1 - rho) * srm
        c = srm * (ec - e)
        x1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        ssr = (ec + 4 * tb0 / (Ec * self.D) * rho / (1 - rho) * x1 + 4 * tb0 / (Es * self.D) * x1) * Es

        # 3 Elastic-Plastic crack element
        mask_ep = (x1 < srm / 2) & (ssr > fsy)

        if np.any(mask_ep):
            ec_ep = ec.copy()
            fsy_ep, Es_ep = self._m(fsy, mask_ep), self._m(Es, mask_ep)
            ec_ep[mask_ep] = ec[mask_ep] + (fsy_ep / Es_ep - ec[mask_ep]) * Es_ep * rho[mask_ep] / (1 - rho[mask_ep]) / self._m(Ec, mask_ep)

            x1_ep = (fsy - ec_ep * Es) * self.D / (4 * tb0)

            c_ep = -e * srm + 2 * ec_ep * (srm / 2 - x1_ep) + (fsy / Es + ec_ep) * x1_ep
            b_ep = 2 * (fsy / Es - ec_ep)
            a_ep = 4 * tb1 / (self.D * Esh)
            discriminant = np.where(mask_ep,b_ep**2 - 4 * a_ep * c_ep,0.0)
            x2_ep = (-b_ep + np.sqrt(discriminant)) / (2 * a_ep)

            ssr_ep = ec_ep * Es + x1_ep * 4 * tb0 / self.D + x2_ep * 4 * tb1 / self.D

            x1[mask_ep]  = x1_ep[mask_ep]
            ssr[mask_ep] = ssr_ep[mask_ep]


        return x1, ssr

    def ssr_tcm_bilin(self, e, ec, srm, rho, tb0, tb1, Ec, st):
        """ ------------------------------------ Calculate Steel Stress with the TCM -----------------------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e:        normal strain (just at masked locations, shape: (ntot, nl*mask, 1))
                - ec:       axial concrete compressive strain from mohr's circle of concrete strains, shape: (ntot, nl*mask, 1)
                - srm:      Crack spacing: srm = lambda * sr0, shape: (ntot, nl*mask, 1)
                - rho:          rho only at given locations in element and for one dimension (shape: n_tot, nl*mask, 1)
                - tb0, tb1, Ec: bond stresses / concrete modulus at the same locations (scalar or shaped like rho)
                - st:           steel parameters (fsy, fsu, Es, Esh) at the same locations (scalar or shaped like rho)
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ss_out:   steel stress at the crack according to TCM, shape: (ntot, nl*mask, 1)
            ---------------------------------------------------------------------------------------------------------"""
        fsy, fsu, Es, Esh = st

        # 1 Seelhofer
        x1, ssr = self.ssr_seelhofer(e, ec, srm, rho, tb0, tb1, Ec, st)

        # 2. TCM
        mask_tcm = x1 >= srm / 2
        ss_out = np.empty_like(e)
        ss_out[~mask_tcm] = ssr[~mask_tcm]

        # 2.1 Bare steel stress
        st_naked = self.ss_bilin(e, st)

        # 2.2 Steel stress for fully elastic crack element
        s1 = st_naked + tb0 * srm / self.D

        # 2.3 Steel stress for fully plastic element
        s3 = fsy + Esh * (e - fsy / Es) + tb1 * srm / self.D

        # 2.4 Masks for stress regimes (only where mask_tcm is active)
        mask_elastic  = mask_tcm & (s1 <= fsy)
        mask_plastic  = mask_tcm & (s1 > fsy) & (s3 - (2 * tb1 * srm / self.D) >= fsy)
        mask_partial  = mask_tcm & ~mask_elastic & ~mask_plastic

        # 2.4.1 Fully elastic
        ss_out[mask_elastic] = s1[mask_elastic]

        # 2.4.2 Fully plastic
        ss_out[mask_plastic] = s3[mask_plastic]

        # 2.4.3 Partially elastic
        s2 = (fsy - Es * e) * tb1 * srm / self.D * (tb0 / tb1 - Es / Esh)
        s2 = s2 + Es / Esh * tb0 * tb1 * srm ** 2 / self.D ** 2
        discriminant = np.where(mask_partial,s2,0.0)
        s2 = tb0 * srm / self.D - np.sqrt(discriminant)                # to avoid warning in output.
        s2 = fsy + 2 * s2 / (tb0 / tb1 - Es / Esh)
        ss_out[mask_partial] = s2[mask_partial]

        # 2.5 If stress > ultimate stress, assign ultimate stress
        mask_ultimate = ss_out.real > fsu
        ss_out[mask_ultimate] = self._m(fsu, mask_ultimate) + 100 * e[mask_ultimate]

        return ss_out

    def sr0_vc(self, th):
        """ --------------------------------- Calculate diagonal crack spacing------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - th: Principal direction (shape: ntot, nl, 1)
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - srx, sry: Crack spacing in reinforcement direction (shape: ntot, nl, 1)
        -------------------------------------------------------------------------------------------------------------"""

        # 0.1 Manipulate (mirror) negative angles
        th = np.where(th < 0, -th, th)

        # 1 Initial Assumption
        # 1.1 in x
        mask_rhox = self.rho_x > 1e-9
        denom_x = np.where(mask_rhox, 2 * self.tb0 / self.D * self.rho_x, 1.0)  # safe dummy in false branch, to avoid divide by zero warning.
        srx0_mask = np.where(mask_rhox, self.fct * (1 - self.rho_x) / denom_x, 1e3)
        srxmax = np.where(mask_rhox, self.D / (2 * self.tb0) * self.fsy + self.D / (2 * self.tb1) * (self.fsu - self.fsy), 1e3)
        srx0 = np.minimum(srx0_mask, srxmax)

        # 1.2 in y
        mask_rhoy = self.rho_y > 1e-9
        denom_y = np.where(mask_rhoy, 2 * self.tb0 / self.D * self.rho_y, 1.0)  # safe dummy in false branch, to avoid divide by zero warning.
        sry0_mask = np.where(mask_rhoy, self.fct * (1 - self.rho_y) / denom_y, 1e3)
        srymax = np.where(mask_rhoy, self.D / (2 * self.tb0) * self.fsy + self.D / (2 * self.tb1) * (self.fsu - self.fsy), 1e3)
        sry0 = np.minimum(sry0_mask, srymax)

        # 2 Actual crack spacing as a function of lambda
        sr = self.lbd / ((np.sin(th) / srx0) + (np.cos(th) / sry0))

        # 3 Recalculate spacings in reinforcement directions
        # 3.1 x-direction
        srx_ = np.where(
            np.abs(np.sin(th)) > 0,
            sr / np.sin(th),
            sr / 1e-10
        )
        srx = np.minimum(srx_, srxmax)

        # 3.2 y-direction
        sry_ = np.where(
            np.abs(np.cos(th)) > 0,
            sr / np.cos(th),
            sr / 1e-10
        )
        sry = np.minimum(sry_, srymax)

        srx0_lbd = srx0 * self.lbd * np.ones_like(th)
        sry0_lbd = sry0 * self.lbd * np.ones_like(th)

        return srx, sry, srx0_lbd, sry0_lbd


    ''' -------------------------------------COMBINED IN-PLANE STATES-------------------------------------'''

    def sigma_cart_1(self):
        '''
        Calculate linear elastic stresses
        '''

        ex = self.e[:,:,0:1]
        ey = self.e[:,:,1:2]
        gxy = self.e[:,:,2:3]

        E = self.Ec
        v = self.v

        sx_x = E/(1-v**2)*ex
        sx_y = E/(1-v**2)*v*ey
        sx_xy = np.zeros_like(self.e[:,:,0:1])
        sy_x = E/(1-v**2)*v*ex
        sy_y = E/(1-v**2)*ey
        sy_xy = np.zeros_like(self.e[:,:,0:1])
        txy_x = np.zeros_like(self.e[:,:,0:1])
        txy_y = np.zeros_like(self.e[:,:,0:1])
        txy_xy = E/(1-v**2)*(1-v)/2*gxy

        sx = sx_x + sx_y + sx_xy
        sy = sy_x + sy_y + sy_xy
        txy = txy_x + txy_y + txy_xy

        return sx, sy, txy

    def sigma_cart_31(self, mask):
        """
        in-plane stresses for compression - compression

        Args:
            self.e:     Strains (n_tot, nl, 3)
            mask:       Location where submodel = 1

        Returns:
            sx, sy, txy Stresses only at masked locations.
        """

        e = self.e
        e1t,_,th,_,_,_,ec1,ec3 = self.principal(e[:,:,0:1], e[:,:,1:2], e[:,:,2:3])
        
        # 2 Steel Contribution
        ssx = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhox = (self.rho_x != 0)
        ssx[mask&mask_rhox] = self.ss_bilin(e[:,:,0:1][mask&mask_rhox], self._steel(mask&mask_rhox))

        ssy = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhoy = (self.rho_y != 0)
        ssy[mask&mask_rhoy] = self.ss_bilin(e[:,:,1:2][mask&mask_rhoy], self._steel(mask&mask_rhoy))

        # 3 CFRP Contribution (skipped)

        # 4 Concrete Constitutive Matrix
        # 4.1 Principal Concrete stresses
        sc1 = self.sc(ec1, e1t)
        sc3 = self.sc(ec3, e1t)

        # 4.2 Concrete secant stiffness matrix
        #     If concrete in 1- and 3- directions: assign secant stiffness
        #     Else: assign "zero" (not possible in compression-compression case)
        scx = sc3 * np.cos(th) ** 2 + sc1 * np.sin(th) ** 2
        scy = sc3 * np.sin(th) ** 2 + sc1 * np.cos(th) ** 2
        txy = (sc1 - sc3) * np.sin(th) * np.cos(th)

        # 5 Assemble resulting stresses
        # 5.1 Stress assembly
        sx = 1/(1-self.v**2)*(scx[mask] + self.v*scy[mask])*(1-self.rho_x[mask]) + self.rho_x[mask]*ssx[mask]
        sy = 1/(1-self.v**2)*(scy[mask] + self.v*scx[mask])*(1-self.rho_y[mask]) + self.rho_y[mask]*ssy[mask]

        return sx, sy, txy[mask]

    def sigma_cart_32(self, mask):
        """
        in-plane stresses for tension - compression (cracked, concrete in compression)

        Args:
            self.e:     Strains (n_tot, nl, 3)
            mask:       Location where submodel = 2

        Returns:
            sx, sy, txy Stresses only at masked locations.
        """

        e = self.e
        e1t,_,th,_,ecx,ecy,ec1,ec3 = self.principal(e[:,:,0:1], e[:,:,1:2], e[:,:,2:3])

        srx, sry, _, _ = self.sr0_vc(th)

        # 2 Steel Contribution
        ssx = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhox = (self.rho_x != 0)
        mask_e = e[:,:,0:1] > ecx
        mask_1_x = mask & mask_rhox & mask_e
        ssx[mask_1_x] = self.ssr(self.e[:,:,0:1][mask_1_x], ecx[mask_1_x], self.ecsx,
                                srx[mask_1_x], self.rho_x[mask_1_x], mask_1_x)
        mask_2_x = mask & mask_rhox & ~mask_e
        ssx[mask_2_x] = self.ss_bilin(self.e[:,:,0:1][mask_2_x], self._steel(mask_2_x))

        ssy = np.zeros((e.shape[0], e.shape[1],1), dtype=np.complex64)
        mask_rhoy = (self.rho_y != 0)
        mask_e_y = e[:,:,1:2] > ecy
        mask_1_y = mask & mask_rhoy & mask_e_y
        ssy[mask_1_y] = self.ssr(self.e[:,:,1:2][mask_1_y], ecy[mask_1_y], self.ecsy,
                                sry[mask_1_y], self.rho_y[mask_1_y], mask_1_y)
        mask_2_y = mask & mask_rhoy & ~mask_e_y
        ssy[mask_2_y] = self.ss_bilin(self.e[:,:,1:2][mask_2_y], self._steel(mask_2_y))

        # 3 CFRP Contribution (skipped)

        # 4 Concrete Constitutive Matrix
        # 4.1 Principal Concrete stresses
        sc1 = self.sc(ec1, e1t)
        sc3 = self.sc(ec3, e1t)

        # 4.2 Concrete secant stiffness matrix
        scx = sc3 * np.cos(th) ** 2 + sc1 * np.sin(th) ** 2
        scy = sc3 * np.sin(th) ** 2 + sc1 * np.cos(th) ** 2
        txy = (sc1 - sc3) * np.sin(th) * np.cos(th)

        # 5 Assemble resulting stresses
        sx = 1/(1-self.v**2)*(scx[mask] + self.v*scy[mask]) + self.rho_x[mask]*ssx[mask]
        sy = 1/(1-self.v**2)*(scy[mask] + self.v*scx[mask]) + self.rho_y[mask]*ssy[mask]

        return sx, sy, txy[mask]

    def sigma_cart_33(self, mask):
        """
        in-plane stresses for tension - tension

        Args:
            self.e:     Strains (n_tot, nl, 3)
            mask:       Location where submodel = 3

        Returns:
            sx, sy, txy Stresses only at masked locations.
        """

        e = self.e
        _,_,th,_,ecx,ecy,_,_ = self.principal(e[:,:,0:1], e[:,:,1:2], e[:,:,2:3])      
        
        # 1 Crack spacing
        _, _, srx0_lbd, sry0_lbd = self.sr0_vc(th)

        # 2 Steel contribution
        ssx = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhox = (self.rho_x != 0)
        ssx[mask&mask_rhox] = self.ssr(self.e[:,:,0:1][mask&mask_rhox], ecx[mask&mask_rhox], self.ecsx,
                                       srx0_lbd[mask&mask_rhox], self.rho_x[mask&mask_rhox], mask&mask_rhox)

        ssy = np.zeros((self.e.shape[0], self.e.shape[1],1), dtype = np.complex64)
        mask_rhoy = (self.rho_y != 0)
        ssy[mask&mask_rhoy] = self.ssr(self.e[:,:,1:2][mask&mask_rhoy], ecy[mask&mask_rhoy], self.ecsy,
                                       sry0_lbd[mask&mask_rhoy], self.rho_y[mask&mask_rhoy], mask&mask_rhoy)

        # 3 - CFRP contribution (skipped)

        # 4 - Concrete contribution
        ex = e[:,:,0:1][mask]                   # only use strains in those layers, where given submodel applies.
        ey = e[:,:,1:2][mask]
        gxy = e[:,:,2:3][mask]
        scx = self.ff*ex
        scy = self.ff*ey

        # 5 - Resulting stresses
        sx = 1/(1-self.v**2)*(scx + self.v*scy) + self.rho_x[mask]*ssx[mask]
        sy = 1/(1-self.v**2)*(scy + self.v*scx) + self.rho_y[mask]*ssy[mask]
        txy = self.ff/2*gxy

        return sx, sy, txy

    ''' -------------------------------------OUTPUT (STRESSES)-------------------------------------'''

    def out(self):
        """ ------------------------------------------- Define Output---------------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - self.e:       [ex_klij,ey_klij,gxy_klij] strain state in integration point
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - s:            [sx_klij,sy_klij,txy_klij] stress state in integration point
        -------------------------------------------------------------------------------------------------------------"""

        # 1 Assign Strain Values
        ex = self.e[:,:,0:1]
        ey = self.e[:,:,1:2]
        gxy= self.e[:,:,2:3]

        # 2 Calculate Stresses based on given constitutive model and strain state
        #   submodel 1: compression - compression, 2: tension - compression, 3: tension - tension
        _, _, _, submodel,_,_,_,_ = self.principal(ex, ey, gxy)

        if self.cm_klij == 1:
            sx, sy, txy = self.sigma_cart_1()
        elif self.cm_klij == 3:
            n_tot = self.e.shape[0]
            mask3 = (submodel == 3)
            mask1 = (submodel == 1)
            mask2 = ~mask3 & ~mask1

            sx  = np.zeros((n_tot, self.nl, 1), dtype = np.complex64)
            sy  = np.zeros((n_tot, self.nl, 1), dtype = np.complex64)
            txy = np.zeros((n_tot, self.nl, 1), dtype = np.complex64)

            sx[mask3], sy[mask3], txy[mask3] = self.sigma_cart_33(mask3)
            sx[mask1], sy[mask1], txy[mask1] = self.sigma_cart_31(mask1)
            sx[mask2], sy[mask2], txy[mask2] = self.sigma_cart_32(mask2)

        # 3 Numerical floor on the shear stiffness (txy.imag = complex-step tangent); keeps the 6x6 tangent
        #   non-singular at zero shear strain
        ff = self.Ec/100
        mask_floor = abs(txy.imag) < abs(gxy.imag)*ff/2
        print(f'Amount of points that trigger shear stiffness tolerance: {mask_floor.sum()}/{mask_floor.size} = {mask_floor.sum()/mask_floor.size*100:.2f}%')
        txy = np.where(mask_floor, cplx(txy.real, abs(gxy.imag)*ff/2), txy)

        s = np.stack((sx, sy, txy), axis = 2)

        return s
