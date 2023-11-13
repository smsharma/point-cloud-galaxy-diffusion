import jax
import jax.numpy as np
from flax import linen as nn


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm; generate scale and shift parameters from conditioning context."""

    @nn.compact
    def __call__(self, x, conditioning):
        # Compute scale and shift parameters from conditioning context
        scale_and_shift = nn.gelu(nn.Dense(2 * x.shape[-1])(conditioning))
        scale, shift = np.split(scale_and_shift, 2, axis=-1)

        # Apply layer norm
        # Don't use bias or scale since these will be learnable through the conditioning context
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)

        # Apply scale and shift
        # Apple same scale, shift to all elements in sequence
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]

        return x


class MultiHeadSelfAttentionBlock(nn.Module):
    """Multi-head attention. Uses pre-LN configuration (LN within residual stream), which seems to work much better than post-LN."""

    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, x, conditioning, mask=None):
        mask = None if mask is None else mask[..., None, :, :]

        # Multi-head attention
        x_sa = AdaLayerNorm()(x, conditioning)  # pre-LN
        x_sa = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(x_sa, x_sa, mask)

        # Add into residual stream
        x += x_sa

        # MLP
        x_mlp = AdaLayerNorm()(x, conditioning)  # pre-LN
        x_mlp = nn.gelu(nn.Dense(self.d_mlp)(x_mlp))
        x_mlp = nn.Dense(self.d_model)(x_mlp)

        # Add into residual stream
        x += x_mlp

        return x


class Transformer(nn.Module):
    """Transformer with Adaptive Layer Norm for conditioning.
    See https://arxiv.org/abs/2212.09748, and a reference PyTorch implementation at
    https://github.com/adelacvg/diff-vits/blob/master/unet1d/attention.py#L320


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

    @nn.compact
    def __call__(self, x: np.ndarray, conditioning: np.ndarray = None, mask=None):
        # Input embedding
        x = nn.Dense(int(self.d_model))(x)  # (batch, seq_len, d_model)

        # Transformer layers
        for _ in range(self.n_layers):
            mask_attn = None if mask is None else mask[..., None] * mask[..., None, :]
            x = MultiHeadSelfAttentionBlock(
                n_heads=self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp
            )(x, conditioning, mask_attn)

        # Final LN as in pre-LN configuration
        x = AdaLayerNorm()(x, conditioning)

        # Unembed; zero init kernel to propagate zero residual initially before training
        x = nn.Dense(self.n_input, kernel_init=jax.nn.initializers.zeros)(x)
        return x
