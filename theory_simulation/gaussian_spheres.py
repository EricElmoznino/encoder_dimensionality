from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Tuple, Optional
from numpy.typing import ArrayLike


def run_simulation(dim_nat: int = 30, dim_eco: int = 10, dim_exp: int = 3,
                   max_shared: Optional[int] = None,
                   dims_model: ArrayLike = np.linspace(1, 20, 20),
                   n_repeats: int = 10, method: str = 'proj') -> pd.DataFrame:
    assert method in ['proj', 'pick']
    make_dataset = make_dataset_proj if method == 'proj' else make_dataset_pick
    dims_model = np.round(dims_model).astype(int).tolist()

    results = pd.DataFrame()
    for dim_model in tqdm(dims_model):
        dim_shared_eco_model = min([dim_model, dim_eco] +
                                   ([] if max_shared is None else [max_shared]))
        for _ in range(n_repeats):
            samples_model, samples_exp = make_dataset(dim_nat, dim_eco, dim_exp,
                                                      dim_model, dim_shared_eco_model)
            r2 = regression_performance(samples_model, samples_exp)
            results = results.append({'dim_nat': dim_nat, 'dim_eco': dim_eco, 'dim_exp': dim_exp,
                                      'dim_model': dim_model, 'dim_shared_eco_model': dim_shared_eco_model,
                                      'r2': r2}, ignore_index=True)

    results = results.astype({'dim_nat': int, 'dim_eco': int, 'dim_exp': int,
                              'dim_model': int, 'dim_shared_eco_model': int})

    return results


def validate_dims(func):
    """Decorator that checks to see if dimension parameters are valid and consistent."""

    def wrap(dim_nat: int, dim_eco: int, dim_exp: int,
             dim_model: int, dim_shared_eco_model: int,
             *args, **kwargs):
        assert dim_nat > 1 and dim_eco >= 1 and dim_exp >= 1 and dim_model >= 1 and dim_shared_eco_model >= 0
        assert dim_nat >= dim_eco >= dim_exp and dim_nat >= dim_model
        assert dim_nat >= (dim_eco + dim_model - dim_shared_eco_model)
        assert dim_eco >= dim_shared_eco_model and dim_model >= dim_shared_eco_model
        return func(dim_nat, dim_eco, dim_exp, dim_model, dim_shared_eco_model, *args, **kwargs)

    return wrap


@validate_dims
def make_dataset_proj(dim_nat: int, dim_eco: int, dim_exp: int,
                      dim_model: int, dim_shared_eco_model: int,
                      n_samples: int = 1000) -> Tuple[np.array, np.array]:
    # Create projection matrices
    nat_basis = ortho_group.rvs(dim_nat)
    nat_to_eco = nat_basis[:, :dim_eco]
    eco_to_exp = ortho_group.rvs(dim_eco)[:, :dim_exp]
    eco_to_model_shared = ortho_group.rvs(dim_eco)[:, :dim_shared_eco_model]
    nat_to_model_unshared = nat_basis[:, dim_eco:dim_eco + dim_model - dim_shared_eco_model]

    # Create dataset
    samples_nat = np.random.normal(size=(n_samples, dim_nat))
    samples_eco = samples_nat @ nat_to_eco
    samples_exp = samples_eco @ eco_to_exp
    samples_model = np.concatenate([samples_eco @ eco_to_model_shared,
                                    samples_nat @ nat_to_model_unshared],
                                   axis=1)

    return samples_model, samples_exp


@validate_dims
def make_dataset_pick(dim_nat: int, dim_eco: int, dim_exp: int,
                      dim_model: int, dim_shared_eco_model: int,
                      n_samples: int = 1000) -> Tuple[np.array, np.array]:
    # Select dimension subsets
    nat_to_eco = np.random.choice(range(dim_nat), dim_eco, replace=False)
    eco_to_exp = np.random.choice(range(dim_eco), dim_exp, replace=False)
    eco_to_model_shared = np.random.choice(range(dim_eco), dim_shared_eco_model, replace=False)
    nat_to_model_unshared = np.random.choice([i for i in range(dim_nat) if i not in nat_to_eco],
                                             dim_model - dim_shared_eco_model, replace=False)

    # Create dataset
    samples_nat = np.random.normal(size=(n_samples, dim_nat))
    samples_eco = samples_nat[:, nat_to_eco]
    samples_exp = samples_eco[:, eco_to_exp]
    samples_model = np.concatenate([samples_eco[:, eco_to_model_shared],
                                    samples_nat[:, nat_to_model_unshared]],
                                   axis=1)

    return samples_model, samples_exp


def regression_performance(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2
