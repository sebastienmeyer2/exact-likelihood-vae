"""Metropolis-Hasting within Gibbs technique for data imputation.

References
----------
.. [1] Pierre-Alexandre Mattei and Jes Frellsen. *Leveraging the Exact Likelihood of Deep Latent
    Variable Models.* (Available at: https://arxiv.org/abs/1802.04826)
"""


from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.nn import Module
from torch.nn import MSELoss

from sklearn.metrics import f1_score

from tqdm import tqdm


from sampling.pseudo_gibbs import pseudo_gibbs_sampling


@torch.no_grad()
def metropolis_hastings_w_gibbs_sampling(
    model: Module, data: Tensor, noisy_data: Tensor, indices: Tensor, device: str,
    mhwg_iter: int = 2000, mhwg_gibbs: int = 10
) -> Tuple[List[Tensor], List[float], List[float]]:
    """Sample missing pixels via Metropolis-Hastings within Gibbs sampling.

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

    mhwg_iter : int, default=2000
        Number of Metropolis-Hastings within Gibbs iterations.

    mhwg_gibbs: int, default=10
        Number of pseudo-Gibbs sampling during the initialization of the Metropolis-Hastings within
        Gibbs sampling.

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

    f1_data = data.detach().clone().cpu().numpy()
    f1_data[f1_data < 0.5] = 0
    f1_data[f1_data > 0.5] = 1
    f1_data = f1_data.astype(int)
    f1_list = []

    # The authors advise to start with a few iterations of pseudo-Gibbs sampling
    if mhwg_gibbs > 0:
        noisy_data_list, loss_list, f1_list = pseudo_gibbs_sampling(
            model, data, noisy_data, indices, device, gibbs_iter=mhwg_gibbs
        )
        noisy_data = noisy_data_list.mean(dim=0)

    # Initialize algorithm
    x = noisy_data.detach().clone()  # x_0

    mean, logvar = model.encode(x)
    z = model.sample(mean, logvar)  # z_0
    decoded_z = model.decode(z)  # decoded from z_0

    x_list = torch.zeros((mhwg_gibbs+mhwg_iter, *x.shape)).to(device)
    if mhwg_gibbs > 0:
        x_list[:mhwg_gibbs] = noisy_data_list

    # Initialize standard normal
    nb_images = z.shape[0]
    encoding_dim = z.shape[-1]

    loc = torch.zeros(encoding_dim).to(device)
    covariance = torch.eye(encoding_dim).to(device)
    norm = MultivariateNormal(loc, covariance)

    for tt in tqdm(range(mhwg_iter)):

        # New proposal z_hat and corresponding decoded from z_hat
        mean, logvar = model.encode(x)
        z_hat = model.sample(mean, logvar)

        decoded_z_hat = model.decode(z_hat)

        # Acceptance-Rejection for z_hat
        log_prior_z_hat = norm.log_prob(z_hat)
        log_prior_z = norm.log_prob(z)

        loc_z_hat = mean
        covariance_z_hat = torch.stack(
            [logvar[i].exp() * torch.eye(encoding_dim).to(device) for i in range(nb_images)],
            dim=0
        )
        norm_z_hat = MultivariateNormal(loc_z_hat, covariance_z_hat)

        log_psi_z = norm_z_hat.log_prob(z)
        log_psi_z_hat = norm_z_hat.log_prob(z_hat)

        log_phi_z = F.binary_cross_entropy(decoded_z, x)
        log_phi_z_hat = F.binary_cross_entropy(decoded_z_hat, x)

        num = log_prior_z_hat + log_phi_z_hat + log_psi_z
        denom = log_prior_z + log_phi_z + log_psi_z_hat
        log_rho = num - denom

        # Sample uniform for each image
        uniform = torch.zeros(nb_images).uniform_().to(device)
        accepted_images = uniform < log_rho

        # Update images where uniform passes
        empty_indices = torch.zeros_like(indices)
        accepted_indices = indices.detach().clone()
        accepted_indices[~accepted_images] = empty_indices[~accepted_images]
        x[accepted_indices] = decoded_z_hat[accepted_indices].detach().clone()  # x

        z[accepted_images] = z_hat[accepted_images].detach().clone()  # z
        decoded_z = model.decode(z)  # decoded from z

        # Log x
        x_list[tt] = x.detach().clone()

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
