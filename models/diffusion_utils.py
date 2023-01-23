import jax
import jax.numpy as np
import flax.linen as nn


class NoiseSchedule_Scalar(nn.Module):
    gamma_min: float = -6.0
    gamma_max: float = 6.0

    def setup(self):
        init_bias = self.gamma_max
        init_scale = self.gamma_min - self.gamma_max
        self.w = self.param("w", nn.initializers.constant(init_scale), (1,))
        self.b = self.param("b", nn.initializers.constant(init_bias), (1,))

    @nn.compact
    def __call__(self, t):
        return self.b + -abs(self.w) * t


class NoiseSchedule_FixedLinear(nn.Module):
    gamma_min: float = -6.0
    gamma_max: float = 6.0

    @nn.compact
    def __call__(self, t):
        return self.gamma_max + (self.gamma_min - self.gamma_max) * t


def gamma(ts, gamma_min=-6, gamma_max=6):
    return gamma_max + (gamma_min - gamma_max) * ts


def sigma2(gamma):
    return jax.nn.sigmoid(-gamma)


def alpha(gamma):
    return np.sqrt(1 - sigma2(gamma))


def variance_preserving_map(x, gamma, eps):
    a = alpha(gamma)
    var = sigma2(gamma)

    x_shape = x.shape

    x = x.reshape(x.shape[0], -1)
    eps = eps.reshape(eps.shape[0], -1)

    noise_augmented = a * x + np.sqrt(var) * eps

    return noise_augmented.reshape(x_shape)


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=np.float32):
    """Build sinusoidal embeddings (from Fairseq)."""

    assert len(timesteps.shape) == 1
    timesteps *= 1000

    half_dim = embedding_dim // 2
    emb = np.log(10_000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
