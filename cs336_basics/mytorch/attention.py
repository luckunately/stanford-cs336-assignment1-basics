from math import ceil
import re
from numpy import dtype
from tests.conftest import mask
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.mytorch.activition import Softmax
from cs336_basics.mytorch.RoPE import RoPE
from cs336_basics.mytorch.linear import Linear
from einops import rearrange, repeat


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Computes the scaled dot-product attention.
    Q: Float[Tensor, " ... queries d_k"] Query tensor
    K: Float[Tensor, " ... keys d_k"] Key tensor
    V: Float[Tensor, " ... values d_v"] Value tensor
    mask: Bool[Tensor, " ... queries keys"] | None = None Optional mask tensor to prevent attention to certain positions
    Returns: Float[Tensor, " ... queries d_v"] Output tensor after applying attention
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=Q.dtype)
    )
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = Softmax(dim=-1)(scores)
    return torch.matmul(attn_weights, V)


class MultiHeadAttention_with_RoPE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
    ):
        """
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model
        self.d_v = d_model
        assert q_proj_weight.shape == (
            self.d_k,
            d_model,
        ), f"q_proj_weight shape mismatch"
        assert k_proj_weight.shape == (
            self.d_k,
            d_model,
        ), "k_proj_weight shape mismatch"
        assert v_proj_weight.shape == (
            self.d_v,
            d_model,
        ), "v_proj_weight shape mismatch"
        assert o_proj_weight.shape == (
            d_model,
            self.d_v,
        ), "o_proj_weight shape mismatch"
        self.wq = Linear(self.d_k, d_model, weight=q_proj_weight)
        self.wk = Linear(self.d_k, d_model, weight=k_proj_weight)
        self.wv = Linear(self.d_v, d_model, weight=v_proj_weight)
        self.wo = Linear(d_model, self.d_v, weight=o_proj_weight)
        self.rope = RoPE(self.d_k // num_heads, theta, max_seq_len)
        
    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
        casual_mask: bool = True) -> Float[Tensor, " ... sequence_length d_out"]:
        """
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens
        Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """

        # Linear projections
        Q = rearrange(self.wq(in_features), " ... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(self.wk(in_features), " ... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(self.wv(in_features), " ... s (h d) -> ... h s d", h=self.num_heads)
        
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        if casual_mask:
            mask = torch.tril(
                torch.ones(
                    (Q.shape[-2], K.shape[-2]),
                    dtype=torch.bool,
                    device=Q.device,
                )
            )
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
        else:
            mask = None
        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = rearrange(attn_output, " ... h s d -> ... s (h d)")

        output = self.wo(attn_output)

        return output

class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_k d_in"] | None = None,
        k_proj_weight: Float[Tensor, " d_k d_in"] | None = None,
        v_proj_weight: Float[Tensor, " d_v d_in"] | None = None,
        o_proj_weight: Float[Tensor, " d_model d_v"] | None = None,
    ):
        """
        Initializes the multi-head attention layer.
        d_model: int Dimension of the input features
        num_heads: int Number of attention heads
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_v d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model
        self.d_v = d_model
        assert q_proj_weight is None or q_proj_weight.shape == (
            self.d_k,
            d_model,
        ), f"q_proj_weight shape mismatch"
        assert k_proj_weight is None or k_proj_weight.shape == (
            self.d_k,
            d_model,
        ), "k_proj_weight shape mismatch"
        assert v_proj_weight is None or v_proj_weight.shape == (
            self.d_v,
            d_model,
        ), "v_proj_weight shape mismatch"
        assert o_proj_weight is None or o_proj_weight.shape == (
            d_model,
            self.d_v,
        ), "o_proj_weight shape mismatch"
        self.wq = Linear(self.d_k, d_model, weight=q_proj_weight)
        self.wk = Linear(self.d_k, d_model, weight=k_proj_weight)
        self.wv = Linear(self.d_v, d_model, weight=v_proj_weight)
        self.wo = Linear(d_model, self.d_v, weight=o_proj_weight)

    def forward(
        self,
        query: Float[Tensor, " batch_size seq_len d_model"],
        key: Float[Tensor, " batch_size seq_len d_model"],
        value: Float[Tensor, " batch_size seq_len d_model"],
        casual_mask: bool = True,
    ) -> Float[Tensor, " batch_size seq_len d_model"]:
        """
        Forward pass for the multi-head attention layer.
        query: Float[Tensor, "batch_size seq_len d_model"] Query tensor
        key: Float[Tensor, "batch_size seq_len d_model"] Key tensor
        value: Float[Tensor, "batch_size seq_len d_model"] Value tensor
        mask: Bool[Tensor, "batch_size seq_len seq_len"] | None = None Optional mask tensor to prevent attention to certain positions
        Returns: Float[Tensor, "batch_size seq_len d_model"] Output tensor after applying multi-head attention
        """

        # Linear projections
        Q = rearrange(self.wq(query), " ... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(self.wk(key), " ... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(self.wv(value), " ... s (h d) -> ... h s d", h=self.num_heads)

        if casual_mask:
            mask = torch.tril(
                torch.ones(
                    (query.shape[1], key.shape[1]),
                    dtype=torch.bool,
                    device=query.device,
                )
            )
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
        else:
            mask = None
        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = rearrange(attn_output, " ... h s d -> ... s (h d)")

        output = self.wo(attn_output)

        return output
