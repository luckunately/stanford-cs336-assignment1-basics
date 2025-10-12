import torch


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        weight: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initializes the RMSNorm layer.
        d_model: int Dimension of the input features
        eps: float = 1e-8 Small value to avoid division by zero
        weight: torch.Tensor | None = None Predefined weights for the normalization
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        if weight is not None:
            assert weight.shape == (
                d_model,
            ), "Weight shape does not match specified d_model"
            self.weight = weight
        else:
            self.weight = torch.nn.Parameter(
                torch.ones(d_model, device=device, dtype=dtype)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.
        x: torch.Tensor Input tensor of shape (..., d_model)
        Returns: torch.Tensor Output tensor of shape (..., d_model)
        """
        assert (
            x.shape[-1] == self.weight.shape[0]
        ), "Input feature dimension does not match d_model"
        in_data_type = x.dtype
        x = x.to(dtype=torch.float32)  # for numerical stability
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return (x_normalized * self.weight).to(dtype=in_data_type)
