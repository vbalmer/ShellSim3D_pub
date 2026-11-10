"""Complex-valued array type used for the complex-step derivative of the constitutive laws.

The tangent stiffness is obtained by perturbing the strains with a tiny imaginary part (complex-step method), so
all stresses are carried as complex arrays. `cplx` is a thin ndarray subclass that keeps the complex dtype through
arithmetic and compares on the real part only, so the branch logic of the material laws acts on the real strains.
"""

from .config import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class cplx(np.ndarray):
    """Complex array: cplx(data) or cplx(real, imag). Arithmetic stays complex, comparisons use the real part."""

    def __new__(cls, data, imag=None):
        if imag is not None:
            data = np.asarray(data, dtype=complex) + 1j * np.asarray(imag, dtype=complex)
        else:
            data = np.asarray(data, dtype=complex)
        return data.view(cls)

    def __array_finalize__(self, obj):
        pass

    # arithmetic: keep the cplx type
    def __add__(self, x):        return np.ndarray.__add__(self, x).view(cplx)
    def __radd__(self, x):       return np.ndarray.__radd__(self, x).view(cplx)
    def __sub__(self, x):        return np.ndarray.__sub__(self, x).view(cplx)
    def __rsub__(self, x):       return np.ndarray.__rsub__(self, x).view(cplx)
    def __mul__(self, x):        return np.ndarray.__mul__(self, x).view(cplx)
    def __rmul__(self, x):       return np.ndarray.__rmul__(self, x).view(cplx)
    def __truediv__(self, x):    return np.ndarray.__truediv__(self, x).view(cplx)
    def __rtruediv__(self, x):   return np.ndarray.__rtruediv__(self, x).view(cplx)
    def __pow__(self, x):        return np.ndarray.__pow__(self, x).view(cplx)
    def __rpow__(self, x):       return np.ndarray.__rpow__(self, x).view(cplx)

    # comparisons on the real part
    def __lt__(self, x):  return self.view(np.ndarray).real < np.asarray(x).view(np.ndarray).real
    def __le__(self, x):  return self.view(np.ndarray).real <= np.asarray(x).view(np.ndarray).real
    def __gt__(self, x):  return self.view(np.ndarray).real > np.asarray(x).view(np.ndarray).real
    def __ge__(self, x):  return self.view(np.ndarray).real >= np.asarray(x).view(np.ndarray).real
    def __eq__(self, x):  return self.view(np.ndarray).real == np.asarray(x).view(np.ndarray).real
    def __ne__(self, x):  return self.view(np.ndarray).real != np.asarray(x).view(np.ndarray).real

    def __repr__(self):
        if self.ndim == 0:
            v = self.item()
            return f'cplx({v.real!r}, {v.imag!r})'
        return f'cplx({np.array2string(np.array(self))})'
