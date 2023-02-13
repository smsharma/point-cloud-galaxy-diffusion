import jax
import jax.numpy as np
from flax import linen as nn


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention. Uses pre-LN configuration (LN within residual stream), which seems to work much better than post-LN."""

    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, x, y, mask=None):

        mask = None if mask is None else mask[..., None, :, :]

        # Multi-head attention
        if x is y:  # Self-attention
            x_sa = nn.LayerNorm()(x)  # pre-LN
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, x_sa, mask)
        else:  # Cross-attention
            x_sa, y_sa = nn.LayerNorm()(x), nn.LayerNorm()(y)
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, y_sa, mask)

        # Add into residual stream
        x += x_sa

        # MLP
        x_mlp = nn.LayerNorm()(x)  # pre-LN
        x_mlp = nn.gelu(nn.Dense(self.d_mlp)(x_mlp))
        x_mlp = nn.Dense(self.d_model)(x_mlp)

        # Add into residual stream
        x += x_mlp

        return x


class PoolingByMultiHeadAttention(nn.Module):
    """PMA block from the Set Transformer paper."""

    n_seed_vectors: int
    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, z, mask=None):
        seed_vectors = self.param(
            "seed_vectors",
            nn.linear.default_embed_init,
            (self.n_seed_vectors, z.shape[-1]),
        )
        seed_vectors = np.broadcast_to(seed_vectors, z.shape[:-2] + seed_vectors.shape)
        mask = None if mask is None else mask[..., None, :]
        return MultiHeadAttentionBlock(
            n_heads=self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp
        )(seed_vectors, z, mask)


class Transformer(nn.Module):
    """Simple decoder-only transformer for set modeling.
    Attributes:
      n_input: The number of input (and output) features.
      d_model: The dimension of the model embedding space.
      d_mlp: The dimension of the multi-layer perceptron (MLP) used in the feed-forward network.
      n_layers: Number of transformer layers.
      n_heads: The number of attention heads.
      induced_attention: Whether to use induced attention.
      n_inducing_points: The number of inducing points for induced attention.
    """

    n_input: int
    d_model: int = 128
    d_mlp: int = 512
    n_layers: int = 4
    n_heads: int = 4
    induced_attention: bool = False
    n_inducing_points: int = 32

    @nn.compact
    def __call__(self, x: np.ndarray, conditioning: np.ndarray = None, mask=None):

        # Input embedding
        x = nn.Dense(int(self.d_model))(x)  # (batch, seq_len, d_model)

        # Add conditioning
        if conditioning is not None:
            conditioning = nn.Dense(int(self.d_model))(conditioning)  # (batch, d_model)
            x += conditioning[:, None, :]  # (batch, seq_len, d_model)

        # Transformer layers
        for _ in range(self.n_layers):

            if not self.induced_attention:  # Vanilla self-attention
                mask_attn = (
                    None if mask is None else mask[..., None] * mask[..., None, :]
                )
                x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp
                )(x, x, mask_attn)
            else:  # Induced attention from set transformer paper
                h = PoolingByMultiHeadAttention(
                    self.n_inducing_points,
                    self.n_heads,
                    d_model=self.d_model,
                    d_mlp=self.d_mlp,
                )(x, mask)
                mask_attn = None if mask is None else mask[..., None]
                x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp
                )(x, h, mask_attn)

        # Final LN as in pre-LN configuration
        x = nn.LayerNorm()(x)

        # Unembed; zero init kernel to propagate zero residual initially before training
        x = nn.Dense(self.n_input, kernel_init=jax.nn.initializers.zeros)(x)

        return x
