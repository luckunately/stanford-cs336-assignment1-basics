from math import ceil
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.mytorch.linear import Linear


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = 0,
        w1_weight: Float[Tensor, " d_ff d_model"] | None = None,
        w2_weight: Float[Tensor, " d_model d_ff"] | None = None,
        w3_weight: Float[Tensor, " d_ff d_model"] | None = None,
    ):
        """
        Initializes the SwiGLU activation function with two linear layers.
        linear1: torch.nn.Module First linear layer
        linear2: torch.nn.Module Second linear layer
        """
        super().__init__()
        if d_ff == 0:
            d_ff = ceil(8 * d_model / 3 / 64) * 64  # Default expansion factor of 8/3
        self.linear1 = Linear(d_model, d_ff, w1_weight)
        self.linear2 = Linear(d_ff, d_model, w2_weight)
        self.linear3 = Linear(d_model, d_ff, w3_weight)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SwiGLU activation function.
        x: torch.Tensor Input tensor of shape (..., in_features)
        Returns: torch.Tensor Output tensor of shape (..., out_features)
        """
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))


class SiLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SiLU activation function.
        x: torch.Tensor Input tensor of any shape
        Returns: torch.Tensor Output tensor of the same shape as input
        """
        return x * torch.sigmoid(x)
