import torch
from einops import rearrange, repeat
from torch import Tensor
from jaxtyping import Float, Int

class RoPE(torch.nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Initializes the RoPE (Rotary Positional Embedding) layer.
        d_k: int Embedding dimension size for the query or key tensor.
        theta: float Base frequency for the sinusoidal embeddings
        max_seq_len: int Maximum sequence length to precompute embeddings for
        device: torch.device | None = None Device to store the parameters on
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        position = torch.arange(max_seq_len).float()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        sinusoid_inp = torch.einsum("i , j -> i j", position, inv_freq) # (max_seq_len, d_k/2)
        cos_cached = torch.cos(sinusoid_inp)
        sin_cached = torch.sin(sinusoid_inp)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]) -> torch.Tensor:
        """
        Forward pass through the RoPE layer.
        x: torch.Tensor Input tensor of shape (..., seq_len, dim)
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns: torch.Tensor Output tensor of the same shape as input
        see https://krasserm.github.io/2022/12/13/rotary-position-embedding/ for implementation details
        """
        seq_len = x.shape[-2]
        assert seq_len <= self.cos_cached.shape[0], "Input sequence length exceeds maximum sequence length"
        assert x.shape[-1] == self.cos_cached.shape[1] * 2, "Input feature dimension must be twice d_k"
        # get cos and sin for the specific token positions
        cos = self.cos_cached[token_positions] # (batch_size, seq_len, d_k/2)
        sin = self.sin_cached[token_positions] # (batch_size, seq_len, d_k/2)
        cos = repeat((cos), '... d_k -> ... (d_k r)', r=2) 
        sin = repeat((sin), '... d_k -> ... (d_k r)', r=2) 
        x1, x2 = x[..., ::2], x[..., 1::2] # split last dim into even and odd indices
        x_rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x) # stack elements in last dim as pairs
        return x * cos.unsqueeze(0) + x_rotated * sin.unsqueeze(0)
