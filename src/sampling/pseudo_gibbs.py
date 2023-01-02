"""Pseudo-Gibbs sampling technique for data imputation.

References
----------
.. [1] Diedrik P. Kingma and Max Welling. *Auto-Encoding Variational Bayes.* (Available at:
    https://arxiv.org/abs/1312.6114)
"""


from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import MSELoss

from sklearn.metrics import f1_score

from tqdm import tqdm


@torch.no_grad()
def pseudo_gibbs_sampling(
    model: Module, data: Tensor, noisy_data: Tensor, indices: Tensor, device: str,
    gibbs_iter: int = 2000
) -> Tuple[List[Tensor], List[float], List[float]]:
    """Sample missing pixels via pseudo-Gibbs sampling.

    In order to use this function with your own Variational Auto Encoder, it will need to implement
    *encode* which maps data onto mean and logarithm and variance, *sample* which samples from
    normal distribution and *decode* which brings samples back to data.

    Parameters
    ----------
    model : torch.Module
        A Variational Auto Encoder instance.

    data : torch.Tensor
        True data.

    noisy_data : torch.Tensor
        Initial tensor containing missing values.

    indices : torch.Tensor
        Indices of missing values.

    device : str
        Device of the model and data.

    gibbs_iter : int, default=2000
        Number of pseudo-Gibbs sampling iterations.

    Returns
    -------
    x_list : list of torch.Tensor
        All sampled tensors except burn-in.

    loss_list : list of float
        All losses including burn-in.

    f1_list : list of float
        All f1-scores including burn-in.
    """
    # Evaluation metrics
    criterion = MSELoss()
    loss_list = []

    f1_data = data.clone().cpu().numpy()
    f1_data[f1_data < 0.5] = 0
    f1_data[f1_data > 0.5] = 1
    f1_data = f1_data.astype(int)
    f1_list = []

    # Initialize algorithm
    x = noisy_data.clone()  # x_0
    x_list = torch.zeros((gibbs_iter, *x.shape)).to(device)

    for tt in tqdm(range(gibbs_iter)):

        # Sample encoding based on missing values
        mean, logvar = model.encode(x)
        samples = model.sample(mean, logvar)

        # Sample missing values based on encoding
        decoded = model.decode(samples)

        x[indices] = decoded[indices]

        # Log x
        x_list[tt] = x.clone()

        # Log L2 loss
        loss = criterion(x, data)
        loss_list.append(loss.data.item())

        # Log F1-score
        f1_x = x.clone().cpu().numpy()
        f1_x[f1_x < 0.5] = 0
        f1_x[f1_x > 0.5] = 1
        f1_x = f1_x.astype(int)
        f1_list.append(f1_score(f1_data.flatten(), f1_x.flatten()))

    return x_list, loss_list, f1_list
