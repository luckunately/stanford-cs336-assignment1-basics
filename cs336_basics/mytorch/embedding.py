import torch
from cs336_basics.mytorch.linear import Linear


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initializes the embedding layer.
        num_embeddings: int Size of the vocabulary (number of unique tokens)
        embedding_dim: int Dimension of the embedding vectors
        weight: torch.Tensor | None = None Predefined weights for the embeddings
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        if weight is not None:
            assert weight.shape == (
                num_embeddings,
                embedding_dim,
            ), "Weight shape does not match specified num_embeddings and embedding_dim"
            self.embeddings = weight.to(device=device, dtype=dtype)
        else:
            self.embeddings = torch.empty(
                (num_embeddings, embedding_dim), device=device, dtype=dtype
            )
            torch.nn.init.trunc_normal_(
                self.embeddings, mean=0.0, std=1.0, a=-3.0, b=3.0
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding layer.
        input: torch.Tensor Input tensor of shape (...), containing token indices
        Returns: torch.Tensor Output tensor of shape (..., embedding_dim), containing the corresponding embeddings
        """
        assert input.dtype == torch.long, "Input tensor must be of type torch.long"
        return self.embeddings[input]
