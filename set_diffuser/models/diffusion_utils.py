import jax
import jax.numpy as np
import flax.linen as nn


class NoiseScheduleScalar(nn.Module):
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


class NoiseScheduleFixedLinear(nn.Module):
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
    if embedding_dim % 2 == 1:  # Zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def loss_vdm(params, model, rng, x, conditioning=None, mask=None, beta=1.0):
    """Compute the loss for a VDM model, sum of diffusion, latent, and reconstruction losses, appropriately masked."""
    loss_diff, loss_klz, loss_recon = model.apply(
        params, x, conditioning, mask, rngs={"sample": rng}
    )

    if mask is None:
        mask = np.ones(x.shape[:-1])

    loss_batch = (
        ((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) / beta
        + (loss_recon * mask[:, :, None]).sum((-1, -2))
    ) / mask.sum(-1)
    return loss_batch.mean()


def generate(vdm, params, rng, shape, conditioning=None, mask=None):
    """Generate samples from a VDM model."""

    # Generate latents
    rng, spl = jax.random.split(rng)
    zt = jax.random.normal(spl, shape + (vdm.d_embedding,))

    def body_fn(i, z_t):
        return vdm.apply(
            params,
            rng,
            i,
            vdm.timesteps,
            z_t,
            conditioning,
            mask=mask,
            method=vdm.sample_step,
        )

    z0 = jax.lax.fori_loop(lower=0, upper=vdm.timesteps, body_fun=body_fn, init_val=zt)

    g0 = vdm.apply(params, 0.0, method=vdm.gammat)
    var0 = sigma2(g0)
    z0_rescaled = z0 / np.sqrt(1.0 - var0)
    x = vdm.apply(params, z0_rescaled, conditioning, method=vdm.decode)
    return vdm.destandarize_features(x=x)
