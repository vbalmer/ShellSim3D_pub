"""Global backend switch.

USE_GPU selects the array backend (cupy on the GPU, numpy on the CPU) for all utils modules. It is read once,
when the modules are imported, so set it *before* importing anything else from utils:

    from utils import config
    config.USE_GPU = True
    from utils.sampler_utils_RC3D import ...
"""

USE_GPU = False


def to_numpy(a):
    """Return `a` as a numpy array (copies cupy arrays from the GPU, passes numpy arrays through)."""
    import numpy
    return a.get() if hasattr(a, 'get') and not isinstance(a, numpy.ndarray) else numpy.asarray(a)
