"""Concrete material parameters per concrete class, and their interpolation for a continuous fc."""

import numpy


dict_CC = {
    'CC': [-2, -1, 0, 1, 2, 3, 4, 5],                                                   # [-] just an identifier
    'Ec': [28.8e3, 30.3e3, 32e3, 33.6e3, 35e3, 36.3e3, 37.5e3, 39e3],                   # [MPa]
    'tb0': [3.8, 4.4, 5.2, 5.8, 6.4, 7.0, 7.6, 8.2],                                    # [MPa]
    'tb1': [1.9, 2.2, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1],                                    # [MPa]
    'ect': [0.07e-3, 0.07e-3, 0.08e-3, 0.09e-3, 0.09e-3, 0.1e-3, 0.1e-3, 0.11e-3],      # [-]
    'ec0': [2.1e-3, 2.2e-3, 2.2e-3, 2.3e-3, 2.3e-3, 2.4e-3, 2.4e-3, 2.5e-3],            # [-]
    'fcp': [16, 20, 25, 30, 35, 40, 45, 50],	                                        # [MPa]
    'fct': [1.9, 2.2, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1],                                    # [MPa]
}

# concrete parameters derived from fc (everything in dict_CC except the identifier)
FC_DEPENDENT_KEYS = ('Ec', 'tb0', 'tb1', 'ect', 'ec0', 'fcp', 'fct')


def mat_from_fc(fc, xp = numpy):
    """
    Concrete material parameters for (continuous) concrete strength(s) fc, linearly interpolated between the
    concrete classes in dict_CC (keyed by 'fcp'). Values outside [min, max] of dict_CC['fcp'] are clamped to
    the first / last class (with a warning).

    Args:
        fc  (float | array): concrete cylinder strength(s) [MPa]; any shape (e.g. scalar or (n, 1, 1)).
        xp  (module):        array module (numpy or cupy) the arrays are built with.

    Returns:
        dict: {key: value} for FC_DEPENDENT_KEYS, each with the shape of fc (a python float for scalar fc).
    """
    fc_tab = xp.asarray(dict_CC['fcp'], dtype = float)
    fc_arr = xp.asarray(fc, dtype = float)
    if bool(xp.any(fc_arr < fc_tab[0])) or bool(xp.any(fc_arr > fc_tab[-1])):
        print(f'[mat_from_fc] WARNING: fc outside [{float(fc_tab[0])}, {float(fc_tab[-1])}] MPa -> clamped to the '
              f'nearest concrete class (no extrapolation).')
    flat = fc_arr.ravel()
    out = {}
    for key in FC_DEPENDENT_KEYS:
        vals = xp.interp(flat, fc_tab, xp.asarray(dict_CC[key], dtype = float)).reshape(fc_arr.shape)
        out[key] = float(vals) if fc_arr.ndim == 0 else vals
    return out
