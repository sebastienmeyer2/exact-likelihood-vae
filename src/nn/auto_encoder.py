"""Auto Encoder architecture."""


from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Linear, ReLU, Sigmoid


class Encoder(Module):
    """Encoder architecture of the Auto Encoder."""

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
            ReLU(),
            Linear(hidden_dim, encoding_dim),
            ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        encoded : torch.Tensor
            Encoded data.
        """
        encoded = self.model(x)

        return encoded


class Decoder(Module):
    """Decoder architecture of the Auto Encoder."""

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


class AutoEncoder(Module):
    """Auto Encoder architecture."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, encoding_dim: int = 32):
        """Initialize the Auto Encoder architecture.

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

    def encode(self, x: Tensor) -> Tensor:
        """Encoding pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Encoded data.
        """
        return self.encoder(x)

    def encode_sample(self, x: Tensor) -> Tensor:
        """Encoding and sampling pass.

        For the Auto Encoder, this is equivalent to encoding pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Sampled data.
        """
        return self.encode(x)

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

    def forward(self, x: Tensor) -> Tensor:
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
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        return decoded
