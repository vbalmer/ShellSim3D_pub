"""Frame bridge (FB): model input of the frame bridge case study. V. Balmer, 2026.

Geometry: two walls (area 1 at y = 0, area 3 at y = B) and the top slab (area 2 at z = H), L = 10 m (span direction
x), B = 5.4 m, H = 2.8 m; walls t_11 = 400 mm, slab t_12 = 450 mm; walls clamped at their base. Load: area load q
[N/mm2] on the middle of the slab (L/4 ... 3L/4, B/4 ... 3B/4), applied as nodal forces.

Each area is split into three sub-panels with their own reinforcement (keys rho{x|y}_{top|bot}_{area}{panel}):
walls by height at H/3 and H/2 (panel 1 = bottom), slab by the y-span at B/8 and 7B/8 (panel 1 = at y = 0). The
same scheme with fc_{area}{panel} / fs_{area}{panel} sets the concrete / steel strength per region; regions without
a key use the global fc / fs.
"""

import numpy as np

from fea_utils.Mesh_gmsh_vb import SCENARIO_FB

L, B, H = 10_000, 5_400, 2_800         # [mm]
ROOF_CENTRE = (L / 2, B / 2, H)         # tracked point of the load-deformation figure

# per-region reinforcement ratios [-] (area 1 / 3: walls, area 2: slab; panel 1 ... 3)
RHO_REGIONS = dict(
    rhox_top_11=0.0070, rhox_top_12=0.0070, rhox_top_13=0.0070,
    rhox_bot_11=0.0070, rhox_bot_12=0.0070, rhox_bot_13=0.0070,
    rhoy_top_11=0.0126, rhoy_top_12=0.0096, rhoy_top_13=0.0283,
    rhoy_bot_11=0.0126, rhoy_bot_12=0.0096, rhoy_bot_13=0.0096,

    rhox_top_21=0.0065, rhox_top_22=0.0065, rhox_top_23=0.0065,
    rhox_bot_21=0.0085, rhox_bot_22=0.0085, rhox_bot_23=0.0085,
    rhoy_top_21=0.0250, rhoy_top_22=0.0085, rhoy_top_23=0.0250,
    rhoy_bot_21=0.0250, rhoy_bot_22=0.0250, rhoy_bot_23=0.0250,

    rhox_top_31=0.0070, rhox_top_32=0.0070, rhox_top_33=0.0070,
    rhox_bot_31=0.0070, rhox_bot_32=0.0070, rhox_bot_33=0.0070,
    rhoy_top_31=0.0126, rhoy_top_32=0.0096, rhoy_top_33=0.0283,
    rhoy_bot_31=0.0126, rhoy_bot_32=0.0096, rhoy_bot_33=0.0096,
)


def fb_mat_dict(fc: float = 30.0, fs: float = None, ms: float = 600) -> dict:
    """
    Model input of the frame bridge.

    Args:
        fc  (float): concrete strength [MPa] (all regions)
        fs  (float): steel yield strength [MPa] (all regions; None: 435)
        ms  (float): mesh size [mm]

    Returns:
        dict: model input, every value as a length-1 array
    """
    d = dict(L=L, B=B, H=H, fc=fc, E_1=0, E_2=0, ms=ms, s=SCENARIO_FB,
             t_1=400, t_11=400, t_12=450, t_2=0, nl=20, nu_1=0, nu_2=0, mat=3,
             rho_x_top=0.03, rho_x_bot=0.03, rho_y_top=0.03, rho_y_bot=0.03,     # defaults of regions without a key
             **RHO_REGIONS)
    if fs is not None:
        d['fs'] = fs
    return {k: np.array([v]) for k, v in d.items()}
