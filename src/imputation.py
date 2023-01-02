"""Main file to use Variational Auto Encoder for data imputation."""


import os

from typing import List

import argparse

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


from sampling import (
    single_pass_sampling, pseudo_gibbs_sampling, metropolis_hastings_w_gibbs_sampling,
    iterative_imputing
)
from utils import set_seed, init_model, init_data, convert_name


def to_img(x: Tensor) -> ndarray:
    """Convert linear auto-encoder output back to image of MNIST format.

    Parameters
    ----------
    x : torch.Tensor
        Linear tensor.

    Returns
    -------
    x : numpy.ndarray
        Converted tensor to image.
    """
    x = x.cpu().data.numpy()
    x = 0.5 * (x + 1)
    x = np.clip(x, 0, 1)
    x = x.reshape([-1, 28, 28])

    return x


def plot_imputation(
    seed: int = 42,
    action: str = "mar",
    input_dim: int = 784,
    hidden_dim: int = 128,
    encoding_dim: int = 32,
    data_dir: str = "data",
    model_dir: str = "models",
    nb_images: int = 10,
    noise: float = 0.2,
    sampling_names: List[str] = ["gibbs", "mhwg"],
    gibbs_iter: int = 2000,
    mhwg_iter: int = 2000,
    mhwg_gibbs: int = 10,
    result_dir: str = "results"
):
    """Plot missing-at-random imputation from the test set.

    Parameters
    ----------
    seed : int, default=42
        Seed to use everywhere for reproducibility.

    action : {"mar", "half"}, default="mar"
        If "mar", pixels are set as missing at random, following **noise** argument. If "half",
        upper half of the image is set as missing.

    input_dim : int, default=784
        Input dimension.

    hidden_dim : int, default=128
        Hidden dimension.

    encoding_dim : int, default=32
        Encoding dimension.

    data_dir : str, default="data"
        Data directory.

    model_dir : str, default="models"
        Checkpoints directory.

    nb_images : int, default=10
        Number of images to plot.

    noise : float, default=0.2
        Proportion of missing pixels in the noisy images.

    sampling_names : list of str, default=["gibbs", "mhwg"]
        Name of sampling procedures following project usage.

    gibbs_iter : int, default=2000
        Number of pseudo-Gibbs sampling iterations.

    mhwg_iter : int, default=2000
        Number of Metropolis-Hastings within Gibbs iterations.

    mhwg_gibbs: int, default=10
        Number of pseudo-Gibbs sampling during the initialization of the Metropolis-Hastings within
        Gibbs sampling.

    result_dir : str, default="results"
        Plots directory.

    Raises
    ------
    ValueError
        If the **model_dir** does not exist or if **sampling_name** is not supported.
    """
    # Fix randomness
    set_seed(seed)

    # Find the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}.")

    # Load data
    _, test_set = init_data(data_dir=data_dir)

    test_loader = DataLoader(test_set, batch_size=nb_images, shuffle=True)

    # Read batch
    data, _ = next(iter(test_loader))  # take one batch of images

    data = data.view([-1, input_dim])
    data.requires_grad = False
    data = data.to(device)

    # Convert original data to images
    orig_imgs = to_img(data)

    # Create figure for L2 loss
    fig_loss = plt.figure(figsize=(10, 6), constrained_layout=True)
    ax_loss = fig_loss.gca()

    ax_loss.set_xlabel("Sampling iteration")
    ax_loss.set_ylabel(r"$L2$ loss")
    ax_loss.set_title("Evolution of loss with different sampling")

    # Create figure for F1-score
    fig_f1 = plt.figure(figsize=(10, 6), constrained_layout=True)
    ax_f1 = fig_f1.gca()

    ax_f1.set_xlabel("Sampling iteration")
    ax_f1.set_ylabel(r"$F1$-score")
    ax_f1.set_title("Evolution of F1-score with different sampling")

    # Create figure for images
    nrows = len(sampling_names) + 1 + int(noise > 0.0)
    ncols = nb_images

    fig_img = plt.figure(figsize=(ncols, nrows), constrained_layout=True)
    subfigs_img = fig_img.subfigures(nrows=nrows)

    fontsize = nb_images
    fig_img.suptitle(f"Results for {convert_name(action)} imputation")

    # Plot original images
    subfigs_img[0].suptitle("Original images", fontsize=fontsize)
    axs_img = subfigs_img[0].subplots(ncols=ncols, sharex=True, sharey=True)

    for i in range(ncols):

        axs_img[i].imshow(orig_imgs[i], cmap="gray")
        axs_img[i].axis("off")

    # Configure some indices as missing pixels
    if action == "mar":

        indices = torch.zeros_like(data).uniform_() < noise

    elif action == "half":

        indices = torch.ones_like(data).bool()
        indices[:, int(noise*input_dim):] = False

    # Initialize missing pixels to random values (we use gaussian)
    noisy_data = data.clone()
    normal_data = torch.zeros_like(noisy_data).normal_(mean=0.5, std=0.25).abs().clip(min=0, max=1)
    noisy_data[indices] = normal_data[indices]

    # In case of pure reconstruction, indices are complete image
    if noise <= 0.0:

        indices = torch.ones_like(data).bool()

    # Convert missing-at-random data to images
    noisy_imgs = to_img(noisy_data)

    # Plot noisy images
    if noise > 0.0:

        if action == "mar":

            header = f"{100*noise: .0f}% of missing pixels at random"

        elif action == "half":

            header = f"{100*noise: .0f}% of missing pixels in the upper half"

        subfigs_img[1].suptitle(header, fontsize=fontsize)
        axs_img = subfigs_img[1].subplots(ncols=ncols)

        for i in range(ncols):

            axs_img[i].imshow(noisy_imgs[i], cmap="gray")
            axs_img[i].axis("off")

    # Load the model
    model = init_model(
        model_name="vae", input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim
    )

    if not os.path.exists(model_dir):
        raise ValueError("Model directory does not exist. Maybe train a model first?")

    model_filename = f"{model_dir}/vae.pth"
    model.load_state_dict(torch.load(model_filename))

    model.to(device)

    model.eval()

    for k, sampling_name in enumerate(sampling_names):

        # Run sampling procedure
        burn_in = 0

        if sampling_name == "simple":

            decoded_list, loss_list, f1_list = single_pass_sampling(
                model, data, noisy_data, indices, device
            )

        elif sampling_name == "gibbs":

            decoded_list, loss_list, f1_list = pseudo_gibbs_sampling(
                model, data, noisy_data, indices, device, gibbs_iter=gibbs_iter
            )
            burn_in = gibbs_iter // 10

        elif sampling_name == "mhwg":

            decoded_list, loss_list, f1_list = metropolis_hastings_w_gibbs_sampling(
                model, data, noisy_data, indices, device, mhwg_iter=mhwg_iter,
                mhwg_gibbs=mhwg_gibbs
            )
            burn_in = mhwg_iter // 10

        elif sampling_name == "ite_tree":

            decoded_list, loss_list, f1_list = iterative_imputing(
                data, noisy_data, indices, device, seed=seed, estimator_name="tree"
            )

        elif sampling_name == "ite_ridge":

            decoded_list, loss_list, f1_list = iterative_imputing(
                data, noisy_data, indices, device, seed=seed, estimator_name="ridge"
            )

        else:

            err_msg = f"Unsupported sampling procedure {sampling_name}."
            err_msg += """Choose from "simple", "gibbs", "mhwg", "ite_tree" and "ite_ridge"."""
            raise ValueError(err_msg)

        decoded = decoded_list[min(burn_in, len(decoded_list)):].mean(dim=0)
        loss = np.mean(loss_list[min(burn_in, len(decoded_list)):])
        f1 = np.mean(f1_list[min(burn_in, len(decoded_list)):])

        # Legends
        header = convert_name(sampling_name)

        # Plot loss
        nb_losses = len(loss_list)
        ax_loss.plot(range(nb_losses)[::10], loss_list[::10], label=f"{header}")

        ax_loss.legend(loc="upper right")

        # Plot F1-score
        nb_f1s = len(f1_list)
        ax_f1.plot(range(nb_f1s)[::10], f1_list[::10], label=f"{header}")

        ax_f1.legend(loc="lower right")

        # Convert to images
        decoded_imgs = to_img(decoded)

        # Plot decoded images
        subfigs_img[k+1+int(noise > 0.0)].suptitle(
            f"{header} (L2 loss: {loss: .4f} - F1-score: {f1:.2f})",
            fontsize=fontsize
        )
        axs_img = subfigs_img[k+1+int(noise > 0.0)].subplots(ncols=nb_images)

        for i in range(ncols):

            axs_img[i].imshow(decoded_imgs[i], cmap="gray")
            axs_img[i].axis("off")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fig_filename = f"{result_dir}/{action}_{noise: .2f}"
    for sampling_name in sampling_names:
        fig_filename += f"_{sampling_name}"

    if noise > 0.0:

        # Save figure of loss
        fig_loss_filename = fig_filename + "_loss.png"
        fig_loss.savefig(fig_loss_filename, facecolor="white")

        # Save figure of F1-score
        fig_f1_filename = fig_filename + "_f1.png"
        fig_f1.savefig(fig_f1_filename, facecolor="white")

    # Save figure of images
    fig_img_filename = fig_filename + "_img.png"
    fig_img.savefig(fig_img_filename, facecolor="white")

    print(f"""Figures saved under {fig_filename}_[loss,f1,img].png.""")


if __name__ == "__main__":

    # Command lines
    PARSER_DESC = "Main file to use Variational Auto Encoder for data imputation."
    PARSER = argparse.ArgumentParser(description=PARSER_DESC)

    # Seed
    PARSER.add_argument(
        "--seed",
        default=42,
        type=int,
        help="""
             Seed to use everywhere for reproducibility. Default: 42.
             """
    )

    # Action
    PARSER.add_argument(
        "--action",
        default="mar",
        type=str,
        choices=["mar", "half"],
        help="""
             Action to perform. If "mar", pixels are set as missing at random. If "half", upper
             half of the image is set as missing. Proportion of missing pixels is set using the
             **noise** argument.
             Default: "mar".
             """
    )

    # Model
    PARSER.add_argument(
        "--input-dim",
        default=784,
        type=int,
        help="""
             Dimension of the input images (after flattening). Default: 784.
             """
    )

    PARSER.add_argument(
        "--hidden-dim",
        default=128,
        type=int,
        help="""
             Hidden dimension in the neural networks. Default: 128.
             """
    )

    PARSER.add_argument(
        "--encoding-dim",
        default=32,
        type=int,
        help="""
             Dimension of the encoded images. Default: 32.
             """
    )

    PARSER.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="""
             Data directory. Default: "data".
             """
    )

    PARSER.add_argument(
        "--model-dir",
        default="models",
        type=str,
        help="""
             Checkpoints directory. Default: "models".
             """
    )

    # Imputation
    PARSER.add_argument(
        "--nb-images",
        default=10,
        type=int,
        help="""
             Number of images to plot. Default: 10.
             """
    )

    PARSER.add_argument(
        "--noise",
        default=0.2,
        type=float,
        help="""
             Proportion of pixels set to zero in the noisy images. Default: 0.2.
             """
    )

    PARSER.add_argument(
        "--sampling-names",
        default=["gibbs", "mhwg"],
        nargs="*",
        help="""
             Choose sampling names. Available sampling procedures: "simple", "gibbs", "mhwg",
             "ite_tree" and "ite_ridge".
             """
    )

    PARSER.add_argument(
        "--gibbs-iter",
        default=2000,
        type=int,
        help="""
             Number of pseudo-Gibbs sampling iterations. Default: 2000.
             """
    )

    PARSER.add_argument(
        "--mhwg-iter",
        default=2000,
        type=int,
        help="""
             Number of Metropolis-Hastings within Gibbs sampling iterations. Default: 2000.
             """
    )

    PARSER.add_argument(
        "--mhwg-gibbs",
        default=10,
        type=int,
        help="""
             Number of pseudo-Gibbs sampling iterations during the initialization of the
             Metropolis-Hastings within Gibbs sampling. Default: 5.
             """
    )

    PARSER.add_argument(
        "--result-dir",
        default="results",
        type=str,
        help="""
             Plots directory. Default: "results".
             """
    )

    # End of command lines
    ARGS = PARSER.parse_args()

    plot_imputation(
        seed=ARGS.seed,
        action=ARGS.action,
        input_dim=ARGS.input_dim,
        hidden_dim=ARGS.hidden_dim,
        encoding_dim=ARGS.encoding_dim,
        data_dir=ARGS.data_dir,
        model_dir=ARGS.model_dir,
        nb_images=ARGS.nb_images,
        noise=ARGS.noise,
        sampling_names=ARGS.sampling_names,
        gibbs_iter=ARGS.gibbs_iter,
        mhwg_iter=ARGS.mhwg_iter,
        mhwg_gibbs=ARGS.mhwg_gibbs,
        result_dir=ARGS.result_dir
    )
