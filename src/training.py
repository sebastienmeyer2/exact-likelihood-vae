"""Main file to train Auto Encoders."""


import os

from typing import List, Tuple

import argparse

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm


from utils import str2bool, set_seed, init_model, init_criterion, init_optim, init_data


def train(
    seed: int = 42,
    model_name: str = "vae",
    input_dim: int = 784,
    hidden_dim: int = 128,
    encoding_dim: int = 32,
    optim_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.,
    data_dir: str = "data",
    batch_size: int = 256,
    epochs: int = 10,
    noise: float = 0.,
    save: bool = True,
    model_dir: str = "models"
) -> Tuple[List[float], List[float]]:
    """Train an auto encoder.

    Parameters
    ----------
    seed : int, default=42
        Seed to use everywhere for reproducibility.

    model_name : {"ae", "vae"}, default="vae"
        Name of the model following project usage.

    input_dim : int, default=784
        Input dimension.

    hidden_dim : int, default=128
        Hidden dimension.

    encoding_dim : int, default=32
        Encoding dimension.

    optim_name : {"adam"}
        Name of the optimizer following project usage.

    lr : float, default=0.001
        Learning rate for the optimizer.

    weight_decay : float, default=0.0
        Weight decay for the optimizer.

    data_dir : str, default="data"
        Data directory.

    batch_size : int, default=256
        Batch size.

    epochs : int, default=10
        Number of training epochs.

    noise : float, default=0.2
        Proportion of pixels set to zero in the noisy images.

    save : bool, default=True
        If True, will save the checkpoint of the model in the **model_dir** directory.

    model_dir : str, default="models"
        Checkpoints directory.

    Returns
    -------
    train_loss_list : list of float
        List of training losses.

    test_loss_list : list of float
        List of test losses.
    """
    # Fix randomness
    set_seed(seed)

    # Find the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}.")

    # Create the model
    model = init_model(
        model_name=model_name, input_dim=input_dim, hidden_dim=hidden_dim,
        encoding_dim=encoding_dim
    )

    model.to(device)

    criterion = init_criterion(model_name=model_name)

    optimizer = init_optim(model, optim_name=optim_name, lr=lr, weight_decay=weight_decay)

    # Load data
    train_set, test_set = init_data(data_dir=data_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    # Log lists
    train_loss_list = []
    train_loss = 0.

    test_loss_list = []
    test_loss = 0.

    pbar = tqdm(range(epochs), desc=f"Epoch {0}/{epochs}")

    for epoch in range(epochs):

        model.train()

        train_loss = 0.

        for _, (data, _) in enumerate(train_loader):

            data = data.view([-1, input_dim]).to(device)

            # Add noise
            if noise > 0.:

                indices = torch.zeros_like(data).uniform_() < noise
                noisy_data = data.clone().to(device)
                noisy_data[indices] = 0

            # Set zero gradients
            optimizer.zero_grad()

            # Forward pass
            if noise > 0.:
                output = model(noisy_data)
            else:
                output = model(data)

            loss = criterion(output, data)

            # Backward pass
            loss.backward()

            optimizer.step()

            # Update loss
            train_loss += loss.data.item()

        pbar_msg = []

        # Log training metrics
        train_loss /= n_train
        train_loss_list.append(train_loss)

        pbar_msg.append(f"Epoch {epoch+1}/{epochs}")
        pbar_msg.append(f"; Train loss: {train_loss: .4f}")

        model.eval()

        # Evaluation
        test_loss = 0.

        for _, (data, _) in enumerate(test_loader):

            data = data.view([-1, input_dim]).to(device)

            # Forward pass
            output = model(data)

            test_loss += criterion(output, data).data.item()

        test_loss /= n_test
        test_loss_list.append(test_loss)

        pbar_msg.append(f"; Test loss: {test_loss: .4f}")

        # Update progress bar
        pbar_desc = "".join(pbar_msg)
        pbar.set_description(pbar_desc, refresh=False)
        pbar.update()

    pbar.close()

    # Save model
    if save:

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filename = f"{model_dir}/{model_name}.pth"
        torch.save(model.state_dict(), model_filename)

    return train_loss_list, test_loss_list


if __name__ == "__main__":

    # Command lines
    PARSER_DESC = "Main file to train Auto Encoders."
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

    # Model
    PARSER.add_argument(
        "--model-name",
        default="vae",
        type=str,
        choices=["ae", "vae"],
        help="""
             Name of the model following project usage. Available models: "ae" and "vae".
             Default: "vae".
             """
    )

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
             Hidden dimension in the Auto Encoder. Default: 128.
             """
    )

    PARSER.add_argument(
        "--encoding-dim",
        default=32,
        type=int,
        help="""
             Dimension of the embedding space. Default: 32.
             """
    )

    # Optimizer
    PARSER.add_argument(
        "--optim-name",
        default="adam",
        type=str,
        choices=["adam"],
        help="""
             Name of the optimizer following project usage. Available optimizers: "adam".
             Default: "adam".
             """
    )

    PARSER.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="""
             Learning rate for the optimizer. Default: 0.001.
             """
    )

    PARSER.add_argument(
        "--weight-decay",
        default=0.,
        type=float,
        help="""
             Weight decay for the optimizer. Default: 0.0.
             """
    )

    # Training
    PARSER.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="""
             Data directory. Default: "data".
             """
    )

    PARSER.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="""
             Training batch size. Default: 256.
             """
    )

    PARSER.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="""
             Number of training epochs. Default: 10.
             """
    )

    PARSER.add_argument(
        "--noise",
        default=0.0,
        type=float,
        help="""
             Proportion of pixels set to zero during training. Default: 0.0.
             """
    )

    # Saving
    PARSER.add_argument(
        "--save",
        default="True",
        type=str,
        help="""
             If True, will save the last checkpoint of the model. Default: True.
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

    # End of command lines
    ARGS = PARSER.parse_args()

    # Run the gridsearch
    train(
            seed=ARGS.seed,
            model_name=ARGS.model_name,
            input_dim=ARGS.input_dim,
            hidden_dim=ARGS.hidden_dim,
            encoding_dim=ARGS.encoding_dim,
            optim_name=ARGS.optim_name,
            lr=ARGS.lr,
            weight_decay=ARGS.weight_decay,
            data_dir=ARGS.data_dir,
            batch_size=ARGS.batch_size,
            epochs=ARGS.epochs,
            noise=ARGS.noise,
            save=str2bool(ARGS.save),
            model_dir=ARGS.model_dir,
    )
