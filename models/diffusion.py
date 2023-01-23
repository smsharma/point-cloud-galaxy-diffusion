import jax
import flax.linen as nn
import jax.numpy as np
from jax import jit, vmap, grad
import tensorflow_probability.substrates.jax as tfp

from functools import partial

from models.diffusion_utils import variance_preserving_map, alpha, sigma2, gamma, get_timestep_embedding
from models.diffusion_utils import NoiseSchedule_Scalar, NoiseSchedule_FixedLinear
from models.transformer import Transformer

tfd = tfp.distributions


class ResNet(nn.Module):
    hidden_size: int
    n_layers: int = 1
    middle_size: int = 512

    @nn.compact
    def __call__(self, x, cond=None):
        assert x.shape[-1] == self.hidden_size, "Input must be hidden size."
        z = x
        for i in range(self.n_layers):
            h = nn.gelu(nn.LayerNorm()(z))
            h = nn.Dense(self.middle_size)(h)
            if cond is not None:
                h += nn.Dense(self.middle_size, use_bias=False)(cond[:, None, :])
            h = nn.gelu(nn.LayerNorm()(h))
            h = nn.Dense(self.hidden_size, kernel_init=jax.nn.initializers.zeros)(h)
            z = z + h
        return z


class Encoder(nn.Module):
    hidden_size: int = 256
    n_layers: int = 3
    z_dim: int = 128

    @nn.compact
    def __call__(self, ims, cond=None):
        # x = 2 * ims.astype('float32') - 1.0
        # x = einops.rearrange(x, '... x y d -> ... (x y d)')
        x = nn.Dense(self.hidden_size)(ims)
        x = ResNet(self.hidden_size, self.n_layers)(x, cond=cond)
        params = nn.Dense(self.z_dim)(ims)
        # params = ims
        return params


class Decoder(nn.Module):
    hidden_size: int = 512
    n_layers: int = 3

    @nn.compact
    def __call__(self, z, cond=None):
        z = nn.Dense(self.hidden_size)(z)
        z = ResNet(self.hidden_size, self.n_layers)(z, cond=cond)
        logits = nn.Dense(3)(z)
        # logits = z
        # logits = einops.rearrange(logits, '... (x y d) -> ... x y d', x=28, y=28, d=1)
        # return tfd.Independent(tfd.Bernoulli(logits=logits), 3)
        # log_std = self.param('log_std', nn.initializers.constant(np.log(0.0001)), (1,))
        # return tfd.Normal(loc=logits, scale=np.exp(log_std))
        return tfd.Normal(loc=logits, scale=1.0e-3)


class ScoreNet(nn.Module):
    embedding_dim: int = 128
    n_layers: int = 10

    @nn.compact
    def __call__(self, z, g_t, conditioning, mask, dim_t_emd=32):
        n_embd = self.embedding_dim

        t = g_t
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(z.shape[0])  # ensure t is a vector
        temb = get_timestep_embedding(t, dim_t_emd)
        cond = np.concatenate([temb, conditioning], axis=1)
        # cond = np.concatenate([t[:, None], conditioning], axis=1)
        cond = nn.gelu(nn.Dense(features=n_embd * 4, name="dense0")(cond))
        cond = nn.gelu(nn.Dense(features=n_embd * 4, name="dense1")(cond))
        cond = nn.Dense(n_embd)(cond)

        h = nn.Dense(n_embd)(z)
        # h = jax.vmap(ResNet(n_embd, self.n_layers))(h, cond)

        h = jax.vmap(Transformer(n_input=n_embd))(h, cond, mask)
        # h = jax.vmap(Transformer(n_input=n_embd))(h[:, None, :], cond[:, :])
        # h = h[:, 0, :]

        return z + h


class VDM(nn.Module):
    timesteps: int = 1000
    gamma_min: float = -3.0  # -13.3
    gamma_max: float = 3.0  # 5.0
    embedding_dim: int = 256
    antithetic_time_sampling: bool = True
    layers: int = 32

    def setup(self):
        # self.gamma = partial(gamma, gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        # self.gamma = NoiseSchedule_FixedLinear(gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        self.gamma = NoiseSchedule_Scalar(gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        self.score_model = ScoreNet(n_layers=self.layers, embedding_dim=self.embedding_dim)
        self.encoder = Encoder(z_dim=self.embedding_dim)
        self.decoder = Decoder()

    def gammat(self, t):
        return self.gamma(t)

    def recon_loss(self, x, f, cond):
        """The reconstruction loss measures the gap in the first step.

        We measure the gap from encoding the image to z_0 and back again."""
        # ## Reconsturction loss 2
        g_0 = self.gamma(0.0)
        eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        loss_recon = -self.decoder(z_0_rescaled, cond).log_prob(x)
        return loss_recon

    def latent_loss(self, f):
        """The latent loss measures the gap in the last step, this is the KL
        divergence between the final sample from the forward process and starting
        distribution for the reverse process, here taken to be a N(0,1)."""
        # KL z1 with N(0,1) prior
        g_1 = self.gamma(1.0)
        var_1 = sigma2(g_1)
        mean1_sqr = (1.0 - var_1) * np.square(f)
        loss_klz = 0.5 * (mean1_sqr + var_1 - np.log(var_1) - 1.0)
        return loss_klz

    def diffusion_loss(self, t, f, cond, mask):
        # sample z_t
        g_t = self.gamma(t)
        eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_t = variance_preserving_map(f, g_t[:, None], eps)
        # compute predicted noise
        eps_hat = self.score_model(z_t, g_t, cond, mask)
        # compute MSE of predicted noise
        loss_diff_mse = np.square(eps - eps_hat)

        # loss for finite depth T, i.e. discrete time
        T = self.timesteps
        s = t - (1.0 / T)
        g_s = self.gamma(s)
        loss_diff = 0.5 * T * np.expm1(g_s - g_t)[:, None, None] * loss_diff_mse
        return loss_diff

    def __call__(self, images, conditioning, mask=None, sample_shape=()):

        x = images
        n_batch = images.shape[0]

        cond = conditioning

        # 1. RECONSTRUCTION LOSS
        # add noise and reconstruct
        f = self.encoder(x, cond)
        loss_recon = self.recon_loss(x, f, cond)

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        loss_klz = self.latent_loss(f)

        # 3. DIFFUSION LOSS
        # sample time steps
        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = np.mod(t0 + np.arange(0.0, 1.0, step=1.0 / n_batch), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(n_batch,))

        # discretize time steps if we're working with discrete time
        T = self.timesteps
        t = np.ceil(t * T) / T

        loss_diff = self.diffusion_loss(t, f, cond, mask)

        # End of diffusion loss computation
        return (loss_diff, loss_klz, loss_recon)

    def embed(self, conditioning):
        return conditioning

    def encode(self, ims, conditioning=None):
        cond = conditioning
        return self.encoder(ims, cond)

    def decode(self, z0, conditioning=None):
        cond = conditioning
        return self.decoder(z0, cond)

    def sample_step(self, rng, i, T, z_t, conditioning, mask=None, guidance_weight=0.0):
        rng_body = jax.random.fold_in(rng, i)
        eps = jax.random.normal(rng_body, z_t.shape)
        t = (T - i) / T
        s = (T - i - 1) / T

        g_s = self.gamma(s)
        g_t = self.gamma(t)

        cond = conditioning

        eps_hat_cond = self.score_model(z_t, g_t * np.ones((z_t.shape[0],), z_t.dtype), cond, mask)

        eps_hat_uncond = self.score_model(z_t, g_t * np.ones((z_t.shape[0],), z_t.dtype), cond * 0.0, mask)

        eps_hat = (1.0 + guidance_weight) * eps_hat_cond - guidance_weight * eps_hat_uncond

        a = nn.sigmoid(g_s)
        b = nn.sigmoid(g_t)
        c = -np.expm1(g_t - g_s)
        sigma_t = np.sqrt(sigma2(g_t))
        z_s = np.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + np.sqrt((1.0 - a) * c) * eps

        return z_s
