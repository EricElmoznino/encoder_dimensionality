from tqdm import tqdm
import numpy as np
from numpy.random import multivariate_normal
import pandas as pd
from numpy.typing import ArrayLike
from theory_simulation.regression import regression_performance
from theory_simulation.utils import fsolve_bounded_monotonic


def run_simulation(dim_eco: int = 20, dim_exp: int = 3, ed_eco: float = 8,
                   dims_model: ArrayLike = np.linspace(1, 20, 10),
                   n_repeats: int = 10) -> pd.DataFrame:
    dims_model = np.round(dims_model).astype(int).tolist()

    results = pd.DataFrame()
    for dim_model in tqdm(dims_model):
        for _ in range(n_repeats):
            samples_model, samples_exp, ed_model, ed_exp = \
                make_dataset(dim_eco, dim_exp, dim_model, ed_eco)
            r2 = regression_performance(samples_model, samples_exp)
            results = results.append({'dim_eco': dim_eco, 'dim_exp': dim_exp,
                                      'dim_model': dim_model, 'ed_eco': ed_eco,
                                      'ed_model': ed_model, 'ed_exp': ed_exp,
                                      'r2': r2}, ignore_index=True)

    results = results.astype({'dim_eco': int, 'dim_exp': int, 'dim_model': int})

    return results


def validate_dims(func):
    """Decorator that checks to see if dimension parameters are valid and consistent."""
    def wrap(dim_eco: int, dim_exp: int, dim_model: int, ed_eco: float,
             *args, **kwargs):
        assert dim_eco >= 1 and dim_exp >= 1 and dim_model >= 1
        assert dim_eco >= dim_exp and dim_eco >= dim_model
        assert dim_eco >= ed_eco
        return func(dim_eco, dim_exp, dim_model, ed_eco, *args, **kwargs)
    return wrap


def eigvecs_for_ed(ed, ambient):
    # Create eigenspectrum from power-law decay, and
    # find the decay exponent that gives the correct
    # effective dimensionality
    def func(alpha):
        alpha = float(alpha)
        i = np.arange(1, ambient + 1)
        numerator = (i ** -alpha).sum() ** 2
        denominator = (i ** (-2 * alpha)).sum()
        return numerator / denominator - ed
    alpha = fsolve_bounded_monotonic(func, bounds=(0, 5))

    # Construct the eigenvalues according to the decay
    # rate and a total variance equal to the ambient dimensionality
    eigvals = np.arange(1, ambient + 1) ** -alpha
    eigvals = ambient / eigvals.sum() * eigvals

    eigvecs = np.diag(np.sqrt(eigvals))
    return eigvecs


@validate_dims
def make_dataset(dim_eco: int, dim_exp: int, dim_model: int, ed_eco: float,
                 n_samples: int = 1000) -> (np.array, np.array, float, float):
    # Create ecological manifold eigenvectors/values
    eigvecs_eco = eigvecs_for_ed(ed_eco, dim_eco)

    # Create projection matrices
    eco_to_exp, _, ed_exp = sample_subspace_mvnormal(eigvecs_eco, dim_exp)
    eco_to_model, _, ed_model = sample_subspace_mvnormal(eigvecs_eco, dim_model)

    # Create dataset
    samples_eco = np.random.multivariate_normal(mean=np.zeros(dim_eco),
                                                cov=eigvecs_eco.T @ eigvecs_eco,
                                                size=n_samples)
    samples_exp = samples_eco @ eco_to_exp.T
    samples_model = samples_eco @ eco_to_model.T

    return samples_model, samples_exp, ed_model, ed_exp


def sample_subspace_mvnormal(eigvecs: np.ndarray, ndims: int) -> (np.ndarray, np.ndarray, float):
    """
Generate a random lower-dimensional subspace, with basis vectors
sampled according to probability under a multivariate normal distribution
    :param eigvecs: row eigenvectors of the parent space scaled by sqrt(eigen values)
    :param ndims: dimensionality of the sampled subspace
    :returns: Geometry of the resulting sampling subspace.
        a) Row vectors of sampled orthonormal basis.
        b) Row eigenvectors scaled by sqrt(eigen values).
        c) Effective dimensionality.
    """
    ambient_dim = eigvecs.shape[0]
    basis = []
    cov = eigvecs.T @ eigvecs
    for i in range(ndims):
        dim = multivariate_normal(mean=np.zeros(ambient_dim), cov=cov)
        dim = dim / np.linalg.norm(dim)
        basis.append(dim)
        subspace = project_onto_subspace(np.eye(ambient_dim), dim)
        cov = subspace @ cov @ subspace.T
    basis = np.stack(basis)

    cov = basis @ (eigvecs.T @ eigvecs) @ basis.T
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvecs = eigvecs.T * np.sqrt(eigvals).reshape(-1, 1)

    ed = eigvals.sum() ** 2 / (eigvals ** 2).sum()

    return basis, eigvecs, ed


def project_onto_subspace(x: np.ndarray, normal: np.ndarray):
    assert x.ndim == 2 and \
           normal.ndim == 1 and \
           x.shape[1] == normal.shape[0]
    normal = normal / np.linalg.norm(normal)
    normal = normal.reshape(-1, 1)
    return x - x @ normal * normal.T
