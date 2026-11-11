"""Gauss points and weights for quadrilateral and triangular shell elements."""

from math import sqrt

import numpy as np


def gauss_points(nn, go):
    """Gauss points and weights in one dimension.

    Args:
        nn  (int): number of element nodes (4: quadrilateral, 3: triangle)
        go  (int): Gauss order (quadrilaterals: any order >= 1; triangles: 1 or 2)

    Returns:
        tuple: xi (np.arr), Gauss points in one dimension; w (np.arr), Gauss weights in one dimension
    """
    # 0 Quadrilaterals: Gauss-Legendre on [-1, 1]
    if nn == 4:
        return np.polynomial.legendre.leggauss(go)

    # 1 Triangles
    if nn == 3:
        if go == 1:
            return np.array([1/3]), np.array([1/sqrt(2)])
        if go == 2:
            return np.array([1/6, 2/3]), np.array([1/sqrt(6), 1/sqrt(6)])
        raise ValueError(f'Gauss order {go} not implemented for triangles (use 1 or 2)')

    raise ValueError(f'Gauss points not implemented for elements with {nn} nodes (use 3 or 4)')
