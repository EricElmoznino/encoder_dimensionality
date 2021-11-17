import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import ortho_group
from sklearn.metrics import r2_score
from typing import Optional
from theory_simulation.regression import regression_performance
from theory_simulation.utils import fsolve_bounded_monotonic, project_onto_subspace


class EDSimulation:

    def __init__(self, ambient, n_samples=1000, resolution=.1):
        self.ambient = ambient
        self.n_samples = n_samples
        self.resolution = resolution

        # Natural image manifold
        self.nat_init = False
        self.nat_eigvecs = None
        self.nat_eigvals = None
        self.nat_ed = None

        # Ecological manifold
        self.eco_init = False
        self.eco_eigvecs = None
        self.eco_eigvals = None
        self.eco_ed = None
        self.eco_alignment = None
        self.eco_alignment_strength = None

        # Model manifold
        self.model_init = False
        self.model_eigvecs = None
        self.model_eigvals = None
        self.model_ed = None
        self.model_alignment = None
        self.model_alignment_strength = None

        # Data manifold
        self.data_init = False
        self.data_eigvecs = None
        self.data_eigvals = None
        self.data_ed = None
        self.data_alignment = None
        self.data_alignment_strength = None

        # Empirical samples
        self.data_samples = None
        self.neural_samples = None
        self.model_samples = None
        self.neural_noise_ceiling = None

    def make_nat_manifold(self, ed):
        self.nat_eigvecs, self.nat_eigvals = sample_subspace(ambient=self.ambient, ed=ed)
        self.nat_ed = ed

        self.data_init = self.eco_init = self.model_init = False
        self.data_samples = self.neural_samples = self.model_samples = self.neural_noise_ceiling = None
        self.nat_init = True

    def make_eco_manifold(self, ed, alignment=None, alignment_strength=None):
        assert alignment in [None, 'nat']
        assert self.nat_init

        alignment_eigvecs = None
        if alignment == 'nat':
            alignment_eigvecs = self.nat_eigvecs

        self.eco_eigvecs, self.eco_eigvals = sample_subspace(ambient=self.ambient, ed=ed,
                                                             alignment_eigvecs=alignment_eigvecs,
                                                             alignment_strength=alignment_strength)
        self.eco_ed = ed
        self.eco_alignment = alignment
        self.eco_alignment_strength = alignment_strength

        self.data_init = self.model_init = False
        self.data_samples = self.neural_samples = self.model_samples = self.neural_noise_ceiling = None
        self.eco_init = True

    def make_model_manifold(self, ed, alignment=None, alignment_strength=None):
        assert alignment in [None, 'nat', 'eco']
        assert self.nat_init and self.eco_init

        alignment_eigvecs = None
        if alignment == 'nat':
            alignment_eigvecs = self.nat_eigvecs
        elif alignment == 'eco':
            alignment_eigvecs = self.eco_eigvecs

        self.model_eigvecs, self.model_eigvals = sample_subspace(ambient=self.ambient, ed=ed,
                                                                 alignment_eigvecs=alignment_eigvecs,
                                                                 alignment_strength=alignment_strength)
        self.model_ed = ed
        self.model_alignment = alignment
        self.model_alignment_strength = alignment_strength

        self.model_samples = None
        self.model_init = True

    def make_data_manifold(self, ed, alignment=None, alignment_strength=None):
        assert alignment in [None, 'nat', 'eco']
        assert self.nat_init and self.eco_init

        alignment_eigvecs = None
        if alignment == 'nat':
            alignment_eigvecs = self.nat_eigvecs
        elif alignment == 'eco':
            alignment_eigvecs = self.eco_eigvecs

        self.data_eigvecs, self.data_eigvals = sample_subspace(ambient=self.ambient, ed=ed,
                                                               alignment_eigvecs=alignment_eigvecs,
                                                               alignment_strength=alignment_strength)
        self.data_ed = ed
        self.data_alignment = alignment
        self.data_alignment_strength = alignment_strength

        self.data_samples = self.neural_samples = self.model_samples = self.neural_noise_ceiling = None
        self.data_init = True

    def sample_data(self):
        assert self.data_init
        cov = self.data_eigvecs @ np.diag(self.data_eigvals) @ self.data_eigvecs.T
        self.data_samples = np.random.multivariate_normal(mean=np.zeros(self.ambient), cov=cov,
                                                          size=self.n_samples)
        self.neural_samples = self.model_samples = self.neural_noise_ceiling = None

    def sample_neural(self):
        assert self.eco_init and self.data_samples is not None
        projection_basis = self.eco_eigvecs * np.sqrt(self.eco_eigvals).reshape(1, -1)
        signal = self.data_samples @ projection_basis
        noise = np.random.normal(scale=self.resolution, size=(self.n_samples, self.ambient))
        self.neural_samples = (signal + noise) @ ortho_group.rvs(self.ambient)
        self.neural_noise_ceiling = r2_score(signal + noise, signal, multioutput='variance_weighted')

    def sample_model(self):
        assert self.model_init and self.data_samples is not None
        projection_basis = self.model_eigvecs * np.sqrt(self.model_eigvals).reshape(1, -1)
        signal = self.data_samples @ projection_basis
        noise = np.random.normal(scale=self.resolution, size=(self.n_samples, self.ambient))
        self.model_samples = (signal + noise) @ ortho_group.rvs(self.ambient)

    def sample(self, remake_manifolds=True):
        if remake_manifolds:
            assert self.nat_init and self.eco_init and self.model_init and self.data_init
            self.make_nat_manifold(self.nat_ed)
            self.make_eco_manifold(self.eco_ed, self.eco_alignment, self.eco_alignment_strength)
            self.make_model_manifold(self.model_ed, self.model_alignment, self.model_alignment_strength)
            self.make_data_manifold(self.data_ed, self.data_alignment, self.data_alignment_strength)
        self.sample_data()
        self.sample_neural()
        self.sample_model()

    def encoding_performance(self, return_state=True):
        assert self.neural_samples is not None and self.model_samples is not None
        r2 = regression_performance(self.model_samples, self.neural_samples)
        r2_ceiled = r2 / self.neural_noise_ceiling
        if not return_state:
            return r2, r2_ceiled
        return {'r2': r2, 'r2_ceiled': r2_ceiled,
                'nat_ed': self.nat_ed, 'eco_ed': self.eco_ed, 'model_ed': self.model_ed, 'data_ed': self.data_ed,
                'eco_alignment': self.eco_alignment, 'eco_alignment_strength': self.eco_alignment_strength,
                'model_alignment': self.model_alignment, 'model_alignment_strength': self.model_alignment_strength,
                'data_alignment': self.data_alignment, 'data_alignment_strength': self.data_alignment_strength,
                'ambient': self.ambient, 'n_samples': self.n_samples, 'resolution': self.resolution}


def sample_subspace(ambient: int, ed: float, max_var: float = 1,
                    alignment_eigvecs: Optional[np.ndarray] = None,
                    alignment_strength: float = None) -> (np.ndarray, np.ndarray):
    """
Samples a multivariate normal manifold within an ambient space. The sampled manifold
may have eigenvectors that preferentially align with those of another manifold.
    :param ambient: Dimensionality of the ambient space.
    :param ed: Desired effective dimensionality of the resulting subspace.
    :param max_var: Variance along the top PC of the resulting subspace.
    :param alignment_eigvecs: Optional eigenvectors for the subspace to preferentially align to.
    :param alignment_strength: Strength of alignment pressure.
        0 will have no alignment pressure,
        1 will lead to alignment pressure that perfectly aligns all eigenvectors,
        -1 will lead to misalignment pressure that perfectly misaligns all eigenvectors
    :returns: Eigenvectors and eigenvalues of the resulting subspace.
    """
    assert 1 <= ed <= ambient
    assert alignment_eigvecs is None and alignment_strength is None or \
           alignment_eigvecs is not None and alignment_strength is not None

    eigvals = eigvals_for_ed(ambient, ed, max_var)

    # eigvecs = ortho_group.rvs(ambient)
    # if alignment_eigvecs is not None:
    #     eigvecs = interpolate_rotations(eigvecs, alignment_eigvecs, alignment_strength)

    if alignment_eigvecs is None or alignment_strength == 0:
        eigvecs = ortho_group.rvs(ambient)
        return eigvecs, eigvals
    elif alignment_strength == 1:
        return alignment_eigvecs, eigvals
    elif alignment_strength == -1:
        return alignment_eigvecs[:, ::-1], eigvals

    alignment_eigvals = np.linspace(0, 1, ambient)
    if alignment_strength < 0:
        alignment_eigvals -= 1
    alignment_eigvals = np.exp(-(alignment_strength * 20 * alignment_eigvals))

    eigvecs = []
    L = alignment_eigvecs * np.sqrt(alignment_eigvals).reshape(1, -1)
    for _ in range(ambient):
        dim = L @ np.random.normal(size=ambient)
        dim = dim / np.linalg.norm(dim)
        eigvecs.append(dim)
        L = project_onto_subspace(L.T, dim).T
    eigvecs = np.stack(eigvecs, axis=1)

    return eigvecs, eigvals


def eigvals_for_ed(ambient, ed, max_var):
    # Create eigenspectrum from power-law decay, and
    # find the decay exponent that gives the correct
    # effective dimensionality
    def func(alpha):
        alpha = float(alpha)
        i = np.arange(1, ambient + 1)
        numerator = (i ** -alpha).sum() ** 2
        denominator = (i ** (-2 * alpha)).sum()
        return numerator / denominator - ed

    alpha = fsolve_bounded_monotonic(func, bounds=(0, 10))

    # Construct the eigenvalues according to the decay
    # rate and initial value
    eigvals = np.arange(1, ambient + 1) ** -alpha
    eigvals = max_var * eigvals

    return eigvals
