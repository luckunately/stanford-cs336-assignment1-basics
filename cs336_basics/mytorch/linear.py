import torch


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super(Linear, self).__init__()
        if weight is not None:
            assert weight.shape == (
                out_features,
                in_features,
            ), "Weight shape does not match specified in_features and out_features"
            self.weight = torch.nn.Parameter(weight.to(device=device, dtype=dtype))
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
            torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer.
        x: torch.Tensor Input tensor of shape (..., in_features)
        Returns: torch.Tensor Output tensor of shape (..., out_features)
        """
        return torch.matmul(x, self.weight.t())
