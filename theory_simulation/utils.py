import math
import numpy as np
from warnings import warn


def fsolve_bounded_monotonic(func, bounds, tol=1e-5, max_iter=50):
    assert max_iter > 0 and tol > 0
    bounds = (float(bounds[0]), float(bounds[1]))

    # Check if the function is monotonically increasing or decreasing
    ymin, ymax = func(bounds[0]), func(bounds[1])
    increasing = ymax > ymin

    # Return bounds if best solution, and warn if necessary
    if math.fabs(ymin) < tol:
        return bounds[0]
    elif math.fabs(ymax) < tol:
        return bounds[1]
    elif ymin > 0 and increasing or ymin < 0 and not increasing:
        warn(f'Lower bound is too large. Decrease it to obtain convergence.')
        return bounds[0]
    elif ymax > 0 and not increasing or ymax < 0 and increasing:
        warn(f'Upper bound is too small. Increase it to obtain convergence.')
        return bounds[1]

    # Do a brute-force binary search to find the solution
    x = None
    error = np.inf
    i = 0
    while error > tol and i < max_iter:
        x = (bounds[0] + bounds[1]) / 2
        y = func(x)
        error = math.fabs(y)
        if increasing and y < 0 or not increasing and y > 0:
            bounds = (x, bounds[1])
        else:
            bounds = (bounds[0], x)
        i += 1

    if i == max_iter:
        warn(f'Max iteration reached without convergence. Error = {error}')

    return x


def project_onto_subspace(x: np.ndarray, normal: np.ndarray):
    assert x.ndim == 2 and \
           normal.ndim == 1 and \
           x.shape[1] == normal.shape[0]
    normal = normal / np.linalg.norm(normal)
    normal = normal.reshape(-1, 1)
    return x - x @ normal * normal.T


def get_ed(eigvals: np.ndarray) -> float:
    return eigvals.sum() ** 2 / (eigvals ** 2).sum()


def interpolate_rotations(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    """
Interpolation between n-dimensional rotation matrices (orthonormal basis sets).
Method is from https://www.researchgate.net/publication/334230650_Robust_Rotation_Interpolation_Based_on_SOn_Geodesic_Distance
    :param start: Source rotation matrix (B in paper)
    :param end: Destination matrix being interpolated towards (A in paper)
    :param t: interpolation amount (0 to 1)
    :return: interpolated matrix
    """
    assert 0 <= t <= 1
    d = t * end + (1 - t) * start
    u, s, vh = np.linalg.svd(d)
    s[:-1] = 1
    s[-1] = np.linalg.det(u @ vh.T)
    r = u @ np.diag(s) @ vh
    return r
