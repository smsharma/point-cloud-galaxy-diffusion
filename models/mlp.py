import jax
import jax.numpy as np
import flax.linen as nn
import tensorflow_probability.substrates.jax as tfp

from typing import Sequence, Callable

tfd = tfp.distributions


class MLP(nn.Module):
    """A simple MLP."""

    feature_sizes: Sequence[int]
    activation: Callable[[np.ndarray], np.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, x):
        for features in self.feature_sizes[:-1]:
            x = nn.Dense(features)(x)
            x = self.activation(x)

        # No activation on final layer
        x = nn.Dense(self.feature_sizes[-1])(x)
        return x


class ResNet(nn.Module):
    """As residual MLP with conditioning."""

    n_layers: int = 4
    d_hidden: int = 512
    activation: Callable[[np.ndarray], np.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, x, cond=None):

        d_input = x.shape[-1]

        # Project conditioning context to hidden dimension
        # NOTE: Changed this recently; previously a new dense projection was used at each layer
        z_context = nn.Dense(self.d_hidden, use_bias=False)(cond[:, None, :])

        z = x
        for _ in range(self.n_layers):
            h = self.activation(nn.LayerNorm()(z))
            h = nn.Dense(self.d_hidden)(h)
            if cond is not None:
                # Add context vector
                h += z_context
            h = self.activation(nn.LayerNorm()(h))
            h = nn.Dense(d_input, kernel_init=jax.nn.initializers.zeros)(h)
            z = z + h  # Residual connection
        return z


class MLPEncoder(nn.Module):
    """An element-wise encoder."""

    d_hidden: int = 32
    n_layers: int = 3
    d_embedding: int = 8

    @nn.compact
    def __call__(self, x, cond=None, mask=None):
        x = nn.Dense(self.d_embedding)(x)  # Project to embedding size
        x = ResNet(n_layers=self.n_layers, d_hidden=self.d_hidden)(x, cond=cond)
        return x


class MLPDecoder(nn.Module):
    """An element-wise decoder."""

    d_output: int = 3
    noise_scale: float = 1.0e-3
    d_hidden: int = 32
    n_layers: int = 3

    @nn.compact
    def __call__(self, z, cond=None, mask=None):
        z = ResNet(n_layers=self.n_layers, d_hidden=self.d_hidden)(z, cond=cond)
        z = nn.Dense(self.d_output)(z)  # Project to output size
        return tfd.Normal(loc=z, scale=self.noise_scale)
