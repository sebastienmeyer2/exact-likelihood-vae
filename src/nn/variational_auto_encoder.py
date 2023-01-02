"""Variational Auto Encoder architecture.

References
----------
.. [1] Diedrik P. Kingma and Max Welling. *Auto-Encoding Variational Bayes.* (Available at:
    https://arxiv.org/abs/1312.6114)
"""


from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Linear, ReLU, Sigmoid


class Encoder(Module):
    """Encoder architecture of the Variational Auto Encoder."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, encoding_dim: int = 32):
        """Initialize the Encoder architecture.

        Parameters
        ----------
        input_dim : int, default=784
            Input dimension.

        hidden_dim : int, default=128
            Hidden dimension.

        encoding_dim : int, default=32
            Encoding dimension.
        """
        super().__init__()

        self.model = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

        self.fc_mean = Linear(hidden_dim, encoding_dim)
        self.fc_logvar = Linear(hidden_dim, encoding_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        mean : torch.Tensor
            Encoded mean.

        logvar : torch.Tensor
            Encoded logarithm of the variance.
        """
        x = self.model(x)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class Decoder(Module):
    """Decoder architecture of the Variational Auto Encoder."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, encoding_dim: int = 32):
        """Initialize the Decoder architecture.

        Parameters
        ----------
        input_dim : int, default=784
            Input dimension.

        hidden_dim : int, default=128
            Hidden dimension.

        encoding_dim : int, default=32
            Encoding dimension.
        """
        super().__init__()

        self.model = Sequential(
            Linear(encoding_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, input_dim),
            Sigmoid()  # output values between 0 and 1
        )

    def forward(self, encoded: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        encoded : torch.Tensor
            Encoded data.

        Returns
        -------
        decoded : torch.Tensor
            Decoded data.
        """
        decoded = self.model(encoded)

        return decoded


class VariationalAutoEncoder(Module):
    """Variational Auto Encoder architecture."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, encoding_dim: int = 32):
        """Initialize the Variational Auto Encoder architecture.

        Parameters
        ----------
        input_dim : int, default=784
            Input dimension.

        hidden_dim : int, default=128
            Hidden dimension.

        encoding_dim : int, default=32
            Encoding dimension.
        """
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim
        )
        self.decoder = Decoder(
            input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encoding pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        tuple of torch.Tensor
            Encoded mean and logarithm of variance.
        """
        return self.encoder(x)

    def sample(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Sampling pass.

        Parameters
        ----------
        mean : torch.Tensor
            Encoded mean.

        logvar : torch.Tensor
            Encoded logarithm of the variance.

        Returns
        -------
        samples : torch.Tensor
            Sampled data.
        """
        normal = torch.zeros_like(mean).normal_()
        std = torch.exp(logvar/2)

        samples = mean + std * normal

        return samples

    def encode_sample(self, x: Tensor) -> Tensor:
        """Encoding and sampling pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Sampled data.
        """
        mean, logvar = self.encode(x)
        samples = self.sample(mean, logvar)

        return samples

    def decode(self, encoded: Tensor) -> Tensor:
        """Decoding pass.

        Parameters
        ----------
        encoded : torch.Tensor
            Encoded data.

        Returns
        -------
        torch.Tensor
            Decoded data.
        """
        return self.decoder(encoded)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        decoded : torch.Tensor
            Decoded data.
        """
        mean, logvar = self.encode(x)
        samples = self.sample(mean, logvar)
        decoded = self.decode(samples)

        return decoded, mean, logvar


class VAELoss(Module):
    """Variational Auto Encoder loss function."""

    def __init__(self):
        """Initialize the Variational Auto Encoder loss function."""
        super().__init__()

    def forward(self, output: Tuple[Tensor, Tensor, Tensor], target: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        output : tuple of torch.Tensor
            Decoded data as well as encoded mean and logarithm of variance.

        target : torch.Tensor
            True data.

        Returns
        -------
        loss : torch.Tensor
            Value of the loss for the Variational Auto Encoded, which is comprised of a binary
            cross entropy (reconstruction) part and a Kullback-Leibler divergence part.
        """
        samples, mean, logvar = output

        ae_loss = F.binary_cross_entropy(samples, target, reduction="sum")
        kl_div = - 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum()
        loss = ae_loss + kl_div

        return loss
