"""Gather utilitary functions for randomness control and parameters checking."""


from typing import Tuple

import argparse

import random
import numpy as np

import torch
from torch.nn import Module
from torch.nn import BCELoss, MSELoss
from torch.optim import Optimizer
from torch.optim import Adam
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


from nn import AutoEncoder, VariationalAutoEncoder, VAELoss


def str2bool(v: str) -> bool:
    """An easy way to handle boolean options.

    Parameters
    ----------
    v : str
        Argument value.

    Returns
    -------
    str2bool(v) : bool
        Corresponding boolean value, if it exists.

    Raises
    ------
    argparse.ArgumentTypeError
        If the entry cannot be converted to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int = 42):
    """Fix seed for current run.

    Parameters
    ----------
    seed : int, default=42
        Seed to use everywhere for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)  # pandas seed is numpy seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_model(
    model_name: str = "ae", input_dim: int = 784, hidden_dim: int = 128, encoding_dim: int = 32
) -> Module:
    """Initialize a model based on its name.

    Parameters
    ----------
    model_name : {"ae", "vae"}
        Name of the model following project usage.

    input_dim : int, default=784
        Input dimension.

    hidden_dim : int, default=128
        Hidden dimension.

    encoding_dim : int, default=32
        Encoding dimension.

    Returns
    -------
    model : torch.nn.Module
        An Auto Encoder to be trained and evaluated.

    Raises
    ------
    ValueError
        If the **model_name** is unsupported.
    """
    model: Module

    if model_name == "ae":
        model = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim)
    elif model_name == "vae":
        model = VariationalAutoEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim
        )
    else:
        err_msg = f"""Unknown model {model_name}. Choose from: "ae" and "vae"."""
        raise ValueError(err_msg)

    return model


def init_criterion(model_name: str = "ae") -> Module:
    """Initialize a loss function based on the considered dataset.

    Parameters
    ----------
    model_name : {"ae", "vae"}
        Name of the model following project usage.

    Returns
    -------
    criterion : torch.nn.Module
        A loss function. Classical Auto Encoder is associated with the binary cross entropy loss,
        while loss function for Variational Auto Encoder is a combination of binary cross entropy
        and Kullback-Leibler divergence.
    """
    criterion: Module

    if model_name == "ae":
        criterion = BCELoss(reduction="sum")
    elif model_name == "vae":
        criterion = VAELoss()
    else:  # default loss
        criterion = MSELoss(reduction="sum")

    return criterion


def init_optim(
    model: Module, optim_name: str = "adam", lr: float = 1e-3, weight_decay: float = 0.0
) -> Optimizer:
    """Initialize an optimizer based on its name.

    Parameters
    ----------
    optim_name : {"adam"}
        Name of the optimizer following project usage.

    lr : float, default=0.001
        Learning rate for the optimizer.

    weight_decay : float, default=0.0
        Weight decay for the optimizer.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        An optimizer to train the model with.
    """
    optimizer: Optimizer

    if optim_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        err_msg = f"""Unknown optimizer {optim_name}. Choose from: "adam"."""
        raise ValueError(err_msg)

    return optimizer


def init_data(data_dir: str = "data") -> Tuple[Dataset, Dataset]:
    """Initialize MNIST dataset.

    Parameters
    ----------
    data_dir : str, default="data"
        Data directory.

    Returns
    -------
    train_set : torch.utils.data.Dataset
        The training set.

    test_set : torch.utils.data.Dataset
        The test set.
    """
    transform = Compose([
        ToTensor()
    ])

    train_set = MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = MNIST(data_dir, train=False, download=True, transform=transform)

    return train_set, test_set


def convert_name(name: str = "ae") -> str:
    """Convert model or sampling name to header.

    Parameters
    ----------
    name : str, default="ae"
        Name of the model or sampling.

    Returns
    -------
    header : str
        Corresponding header.
    """
    header: str

    if name == "ae":
        header = "Auto Encoder"
    elif name == "vae":
        header = "Variational Auto Encoder"
    elif name == "simple":
        header = "Single pass"
    elif name == "gibbs":
        header = "Pseudo-Gibbs sampling"
    elif name == "mhwg":
        header = "Metropolis-Hastings within Gibbs sampling"
    elif name == "ite_tree":
        header = "Trees Iterative Imputing"
    elif name == "ite_ridge":
        header = "Ridge Iterative Imputing"
    elif name == "mar":
        header = "Missing-at-random"
    elif name == "half":
        header = "Missing upper half"
    else:
        header = name

    return header
