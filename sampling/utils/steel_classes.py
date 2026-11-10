"""Reinforcing-steel parameters as a function of the yield strength fsy (analogous to concrete_classes.py for fc).

Only the bilinear bare-steel law + tension chord model is used, which needs fsy, fsu, Es, Esh. There is no rupture
strain in that law (stresses beyond fsu are capped as fsu + 100*e), so eps_su is not a model parameter.
"""

import numpy


dict_S = {
    'fsy': [400, 450, 500, 550, 600],                                                   # [MPa] yield strength (lookup key)
    'fsu': [432, 486, 540, 594, 648],                                                   # [MPa] ultimate strength = 1.08 fsy
    'Es':  [205e3, 205e3, 205e3, 205e3, 205e3],                                         # [MPa] elastic modulus
    'Esh': [9.4e3, 9.4e3, 9.4e3, 9.4e3, 9.4e3],                                         # [MPa] hardening modulus
}

# steel parameters derived from fs (everything in dict_S, including fsy itself)
FS_DEPENDENT_KEYS = ('fsy', 'fsu', 'Es', 'Esh')


def mat_from_fs(fs, xp = numpy):
    """
    Steel material parameters for (continuous) steel yield strength(s) fs, linearly interpolated between the rows
    of dict_S (keyed by 'fsy'). Values outside [min, max] of dict_S['fsy'] are clamped to the first / last row
    (with a warning).

    Args:
        fs  (float | array): steel yield strength(s) fsy [MPa]; any shape (e.g. scalar or (n, 1, 1)).
        xp  (module):        array module (numpy or cupy) the arrays are built with.

    Returns:
        dict: {key: value} for FS_DEPENDENT_KEYS, each with the shape of fs (a python float for scalar fs).
    """
    fs_tab = xp.asarray(dict_S['fsy'], dtype = float)
    fs_arr = xp.asarray(fs, dtype = float)
    if bool(xp.any(fs_arr < fs_tab[0])) or bool(xp.any(fs_arr > fs_tab[-1])):
        print(f'[mat_from_fs] WARNING: fs outside [{float(fs_tab[0])}, {float(fs_tab[-1])}] MPa -> clamped to the '
              f'nearest steel row (no extrapolation).')
    flat = fs_arr.ravel()
    out = {}
    for key in FS_DEPENDENT_KEYS:
        vals = xp.interp(flat, fs_tab, xp.asarray(dict_S[key], dtype = float)).reshape(fs_arr.shape)
        out[key] = float(vals) if fs_arr.ndim == 0 else vals
    return out
