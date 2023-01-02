"""Iterative Imputing data imputation technique."""


from typing import List, Tuple

import numpy as np

import torch
from torch import Tensor
from torch.nn import MSELoss

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import f1_score


@torch.no_grad()
def iterative_imputing(
    data: Tensor, noisy_data: Tensor, indices: Tensor, device: str, seed: int = 42,
    estimator_name: str = "tree", maxiter: int = 10
) -> Tuple[List[Tensor], List[float], List[float]]:
    """Sample missing pixels using iterative imputing.

    Parameters
    ----------
    data : torch.Tensor
        True data.

    noisy_data : torch.Tensor
        Initial tensor containing missing values.

    indices : torch.Tensor
        Indices of missing values.

    device : str
        Device of the model and data.

    seed : int, default=42
        Seed to use everywhere for reproducibility.

    estimator_name : {"tree", "ridge"}, default="tree"
        If "tree", will use a random forest regressor as estimator. If "ridge", will use a bayesian
        ridge linear regressor as estimator.

    maxiter : int, default=10
        Number of iterative imputing iterations.

    Returns
    -------
    x_list : list of torch.Tensor
        All sampled tensors.

    loss_list : list of float
        All losses.

    f1_list : list of float
        All f1-scores.

    Raises
    ------
    ValueError
        If **estimator_name** is not supported.
    """
    # Iterative Imputer
    if estimator_name == "tree":
        estimator = ExtraTreesRegressor(random_state=seed)
    elif estimator_name == "ridge":
        estimator = BayesianRidge()
    else:
        err_msg = f"Unsupported estimator name {estimator_name}."
        err_msg += """Choose from "tree" and "ridge"."""
        raise ValueError(err_msg)
    ite_imp = IterativeImputer(estimator=estimator, max_iter=maxiter, random_state=seed, verbose=2)

    # Evaluation metrics
    criterion = MSELoss()
    loss_list = []

    f1_data = data.detach().clone().cpu().numpy()
    f1_data[f1_data < 0.5] = 0
    f1_data[f1_data > 0.5] = 1
    f1_data = f1_data.astype(int)
    f1_list = []

    # Initialize algorithm
    x_nan = noisy_data.detach().clone().cpu().numpy()
    indices_nan = indices.detach().clone().cpu().numpy()
    x_nan[indices_nan] = np.nan
    x_nan[0, np.all(np.isnan(x_nan), axis=0)] = 0  # does not work when whole column is nan
    x_list = torch.zeros((1, *noisy_data.shape))

    # Run algorithm
    decoded = ite_imp.fit_transform(x_nan)
    x = torch.Tensor(decoded).to(device)

    # Log x
    x_list[0] = x.detach().clone()

    # Log L2 loss
    loss = criterion(x, data)
    loss_list.append(loss.data.item())

    # Log F1-score
    f1_x = x.detach().clone().cpu().numpy()
    f1_x[f1_x < 0.5] = 0
    f1_x[f1_x > 0.5] = 1
    f1_x = f1_x.astype(int)
    f1_list.append(f1_score(f1_data.flatten(), f1_x.flatten()))

    return x_list, loss_list, f1_list
