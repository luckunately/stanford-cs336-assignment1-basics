from curses.ascii import SO
import token
import torch
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.mytorch.attention import MultiHeadAttention_with_RoPE
from cs336_basics.mytorch.activition import SwiGLU, Softmax
from cs336_basics.mytorch.RMSNorm import RMSNorm
from cs336_basics.mytorch.linear import Linear
from cs336_basics.mytorch.embedding import Embedding


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor] | None = None,
    ):
        """
        Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            weight=weights["token_embeddings.weight"] if weights else None,
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    weights=(
                        {
                            "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                            "ln2.weight": weights[f"layers.{i}.ln2.weight"],
                            "attn.q_proj.weight": weights[
                                f"layers.{i}.attn.q_proj.weight"
                            ],
                            "attn.k_proj.weight": weights[
                                f"layers.{i}.attn.k_proj.weight"
                            ],
                            "attn.v_proj.weight": weights[
                                f"layers.{i}.attn.v_proj.weight"
                            ],
                            "attn.output_proj.weight": weights[
                                f"layers.{i}.attn.output_proj.weight"
                            ],
                            "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                            "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                            "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
                        }
                        if weights
                        else None
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model, weight=weights["ln_final.weight"] if weights else None
        )

        self.lm_head = Linear(
            d_model,
            vocab_size,
            weight=weights["lm_head.weight"] if weights else None,
        )

        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        x: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        """
        x (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.
        Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
        """
        x = self.token_embeddings(x)  # (batch_size, seq_len, d_model)
        token_positions = torch.arange(x.shape[-2], device=x.device).unsqueeze(0)
         # (1, seq_len)
        for block in self.transformer_blocks:
            x = block(x, token_positions)  # (batch_size, seq_len, d_model)
        x = self.ln_final(x)  # (batch_size, seq_len, d_model)
        x = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        # x = self.softmax(x)  # (batch_size, seq_len, vocab_size)
        return x  # (batch_size, seq_len, vocab_size)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor] | None = None,
    ):
        """
        This transformer block uses RoPE.

        Args:
            d_model (int): The dimensionality of the Transformer block input.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            weights (dict[str, Tensor]):
                State dict of our reference implementation.
                The keys of this dictionary are:
                - `attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is (d_model, d_model).
                - `ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights

        self.layer_norm1 = RMSNorm(
            d_model, weight=weights["ln1.weight"] if weights else None
        )
        self.layer_norm2 = RMSNorm(
            d_model, weight=weights["ln2.weight"] if weights else None
        )

        self.mha_rope = MultiHeadAttention_with_RoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            q_proj_weight=weights["attn.q_proj.weight"] if weights else None,
            k_proj_weight=weights["attn.k_proj.weight"] if weights else None,
            v_proj_weight=weights["attn.v_proj.weight"] if weights else None,
            o_proj_weight=weights["attn.output_proj.weight"] if weights else None,
        )

        self.swiglu = SwiGLU(
            d_model,
            d_ff,
            w1_weight=weights["ffn.w1.weight"] if weights else None,
            w2_weight=weights["ffn.w2.weight"] if weights else None,
            w3_weight=weights["ffn.w3.weight"] if weights else None,
        )

    def forward(
        self,
        x: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " batch sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        x = x + self.mha_rope(
            self.layer_norm1(x), token_positions
        )  # (batch, seq_len, d_model)
        x = x + self.swiglu(self.layer_norm2(x))  # (batch, seq_len, d_model)
        return x
