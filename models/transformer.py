import jax
import jax.numpy as jnp
from flax import linen as nn


class MultiHeadAttentionBlock(nn.Module):
    n_heads: int
    d_head: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, x, y, mask=None):

        mask = None if mask is None else mask[..., None, :, :]

        # Multi-head attention
        x_mhsa = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=self.d_model // self.n_heads, out_features=self.d_model)(x, y, mask)

        # Add into residual stream and norm
        x = nn.LayerNorm()(x + x_mhsa)

        # MLP
        x_mlp = nn.gelu(nn.Dense(self.d_mlp)(x))
        x_mlp = nn.Dense(self.d_model)(x_mlp)

        # Add into residual stream and norm
        x = nn.LayerNorm()(x + x_mlp)

        return x


class PoolingByMultiHeadAttention(nn.Module):
    n_seed_vectors: int
    n_heads: int
    d_head: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, z, mask=None):
        seed_vectors = self.param("seed_vectors", nn.linear.default_embed_init, (self.n_seed_vectors, z.shape[-1]))
        seed_vectors = jnp.broadcast_to(seed_vectors, z.shape[:-2] + seed_vectors.shape)
        mask = None if mask is None else mask[..., None, :]
        return MultiHeadAttentionBlock(n_heads=1, d_head=self.d_model, d_model=self.d_model, d_mlp=64)(seed_vectors, z, mask)


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
    def __call__(self, x: jnp.ndarray, conditioning: jnp.ndarray = None, mask=None):

        # Sequence length
        batch, seq_length = x.shape[0], x.shape[1]

        # Input embedding
        x = nn.Dense(int(self.d_model))(x)  # (batch, seq_len, d_model)
        if conditioning is not None:
            conditioning = nn.Dense(int(self.d_model))(conditioning)  # (batch, d_model)

        # Add conditioning to each element of set
        if conditioning is not None:
            x += conditioning[:, None, :]  # (batch, seq_len, d_model)

        # Transformer layers
        for _ in range(self.n_layers):

            if not self.induced_attention:
                mask_attn = None if mask is None else mask[..., None] * mask[..., None, :]
                x = MultiHeadAttentionBlock(n_heads=self.n_heads, d_head=self.d_model // self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp)(x, x, mask_attn)
            else:
                h = PoolingByMultiHeadAttention(self.n_inducing_points, self.n_heads, d_head=self.d_model // self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp)(x, mask)
                mask_attn = None if mask is None else mask[..., None]
                x = MultiHeadAttentionBlock(n_heads=self.n_heads, d_head=self.d_model // self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp)(x, h, mask_attn)

        # Unembed; zero init kernel to propagate zero residual initially before training
        x = nn.Dense(self.n_input, kernel_init=jax.nn.initializers.zeros)(x)

        return x
