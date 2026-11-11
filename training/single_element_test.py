"""Single-element test: NN vs NLFEA along a strain or stress path of one shell element (see README.md). V. Balmer, 2026.

Default run: pure bending m_x (bending_iterative) of an asymmetrically reinforced 500 mm section, predicted with the
trained model trained_models/v_156; the figure is saved to training/plots_single_el.
"""

import os

from utils.single_element_utils import single_element_test

HERE = os.path.dirname(os.path.abspath(__file__))

# geometry [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs]  [mm, -, -, -, -, MPa, MPa]
geom = [500, 0.006, 0.0375, 0.0375, 0.006, 50, 500]

model_path = os.path.join(HERE, 'trained_models')   # folder containing v_<version>; e.g. os.path.join(HERE, 'logs') for own runs
model_version = 156
NN_comp = None                                      # e.g. [os.path.join(HERE, 'logs'), 1] to compare a second model
NN_comp2 = None                                     # e.g. [os.path.join(HERE, 'logs'), 2] to compare a third model

save_path = os.path.join(HERE, 'plots_single_el')

# Path: swept component idx_eps and range [min_, max_]. In the iterative modes the range is a generalised stress
# (n [N/mm] or m [Nmm/mm]), in the plain sweep a strain [-] or curvature [1/mm].
#   plain sweep:          idx_eps = [0], min_, max_ = -3e-3, 5e-3     (all modes False)
#   bending_iterative:    idx_eps = [3] (m_x), [4] (m_y) or [5] (m_xy)
#   tension_iterative:    idx_eps = [0], [1] or [2]                   (auto for membrane sweeps of asymmetric sections)
#   shear_iterative:      idx_eps = [2]
#   synclastic_iterative: idx_eps = [3, 4]                            (m_x = m_y)
#   nm_iterative:         idx_eps = [0] or [1], m = n * nm_ratio
idx_eps = [3]
min_, max_ = -150_000, 850_000

single_element_test(idx_eps, geom = geom, model_path = [model_path, model_version],
                    min_ = min_, max_ = max_, save_path = save_path,
                    plot_LFEA = False,                   # also plot the linear-elastic response
                    multirow = True,                     # all 6 stress components (rows)
                    NN_comp = NN_comp, NN_comp2 = NN_comp2,
                    xlim = None,                         # e.g. (2e-3, 5e-3): zoom the x-axis of the first two columns
                    bending_iterative = True,
                    tension_iterative = False,
                    shear_iterative = False,
                    synclastic_iterative = False,
                    nm_iterative = False, nm_ratio = -50.0,
                    include_membrane_strains = False,    # synclastic_iterative only: + columns vs eps_x, eps_y
                    symm_predict = True,                 # mid-plane symmetry wrapper g(x) = 0.5 [f(x) + R f(R x)]
                    xysymm_predict = False)              # x <-> y relabelling wrapper g(x) = 0.5 [f(x) + P f(P x)]
