import jax
import jax.numpy as np
import flax.linen as nn


class NoiseScheduleScalar(nn.Module):
    """A noise schedule that returns a pre-defined scalar for each timestep."""

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
    """A noise schedule that returns a fixed linear function of the timestep."""

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


def loss_vdm(params, model, rng, x, conditioning, mask, beta=1.0):
    """Compute the loss for a VDM model."""
    loss_diff, loss_klz, loss_recon = model.apply(params, x, conditioning, mask, rngs={"sample": rng, "uncond": rng})
    loss_batch = (((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) / beta + (loss_recon * mask[:, :, None]).sum((-1, -2))) / mask.sum(-1)
    return loss_batch.mean()


def generate(vdm, params, rng, shape, conditioning, mask=None, guidance_weight=0.0):
    """Generate samples from a VDM model."""

    # Generate latents
    rng, spl = jax.random.split(rng)
    zt = jax.random.normal(spl, shape + (vdm.d_embedding,))

    def body_fn(i, z_t):
        return vdm.apply(params, rng, i, vdm.timesteps, z_t, conditioning, mask=mask, guidance_weight=guidance_weight, method=vdm.sample_step)

    z0 = jax.lax.fori_loop(lower=0, upper=vdm.timesteps, body_fun=body_fn, init_val=zt)

    g0 = vdm.apply(params, 0.0, method=vdm.gammat)
    var0 = sigma2(g0)
    z0_rescaled = z0 / np.sqrt(1.0 - var0)
    return vdm.apply(params, z0_rescaled, conditioning, method=vdm.decode)


def elbo(vdm, params, rng, x, conditioning, mask):
    rng, spl = jax.random.split(rng)
    cond = vdm.apply(params, conditioning, method=vdm.embed)
    f = vdm.apply(params, x, conditioning, method=vdm.encode)
    loss_recon = vdm.apply(params, x, f, conditioning, rngs={"sample": rng}, method=vdm.recon_loss)
    loss_klz = vdm.apply(params, f, method=vdm.latent_loss)

    def body_fun(i, val):
        loss, rng = val
        rng, spl = jax.random.split(rng)
        new_loss = vdm.apply(params, np.array([i / vdm.timesteps]), f, cond, mask, rngs={"sample": spl}, method=vdm.diffusion_loss)
        return (loss + (new_loss * mask[:, :, None]).sum((-1, -2)) / vdm.timesteps, rng)

    loss_diff, rng = jax.lax.fori_loop(0, vdm.timesteps, body_fun, (np.zeros(x.shape[0]), rng))

    return ((loss_recon * mask[:, :, None]).sum((-1, -2)) + (loss_klz * mask[:, :, None]).sum((-1, -2)) + loss_diff) / mask.sum(-1)
