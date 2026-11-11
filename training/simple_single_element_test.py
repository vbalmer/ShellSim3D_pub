"""Simplified single-element test: NLFEA only, stress-controlled, all six stress components can be ramped at once
(see README.md). V. Balmer, 2026.

Default run: m_x and m_xy ramped from 0 to [100'000, 150'000] Nmm/mm for a symmetric 350 mm section; the figure is
saved to training/plots_single_el.
"""

import os

from utils.simple_single_element_utils import simple_single_element_test

HERE = os.path.dirname(os.path.abspath(__file__))

geom = [350, 0.018, 0.018, 0.018, 0.018, 45, 514]                   # [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs]
save_path = os.path.join(HERE, 'plots_single_el')

min_, max_ = [0, 0, 0, 0, 0, 0], [0, 0, 0, 100_000, 0, 150_000]     # [n_x, n_y, n_xy, m_x, m_y, m_xy], zeros -> small_value

results = simple_single_element_test(min_, max_, geom, num_samples = 100, save_path = save_path)
