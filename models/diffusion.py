import dataclasses

import jax
import flax.linen as nn
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp

from models.diffusion_utils import variance_preserving_map, alpha, sigma2, get_timestep_embedding
from models.diffusion_utils import NoiseScheduleScalar, NoiseScheduleFixedLinear
from models.transformer import Transformer

tfd = tfp.distributions


class ResNet(nn.Module):
    input_size: int
    n_layers: int = 1
    hidden_size: int = 512

    @nn.compact
    def __call__(self, x, cond=None):
        assert x.shape[-1] == self.input_size, "Input size mis-specified."
        z = x
        for _ in range(self.n_layers):
            h = nn.gelu(nn.LayerNorm()(z))
            h = nn.Dense(self.hidden_size)(h)
            if cond is not None:
                h += nn.Dense(self.hidden_size, use_bias=False)(cond[:, None, :])  # Project context to hidden size and add
            h = nn.gelu(nn.LayerNorm()(h))
            h = nn.Dense(self.input_size, kernel_init=jax.nn.initializers.zeros)(h)
            z = z + h  # Residual connection
        return z


class Encoder(nn.Module):
    hidden_size: int = 32
    n_layers: int = 3
    embedding_dim: int = 8
    latent_diffusion: bool = False

    @nn.compact
    def __call__(self, x, cond=None):
        if self.latent_diffusion:
            x = nn.Dense(self.hidden_size)(x)
            x = ResNet(input_size=self.hidden_size, n_layers=self.n_layers, hidden_size=int(4 * self.hidden_size))(x, cond=cond)
            x = nn.Dense(self.embedding_dim)(x)
        return x


class Decoder(nn.Module):
    hidden_size: int = 32
    n_layers: int = 3
    output_dim: int = 3
    scale: float = 1.0e-3
    latent_diffusion: bool = False

    @nn.compact
    def __call__(self, z, cond=None):
        if self.latent_diffusion:
            z = nn.Dense(self.hidden_size)(z)
            z = ResNet(input_size=self.hidden_size, n_layers=self.n_layers, hidden_size=int(4 * self.hidden_size))(z, cond=cond)
            z = nn.Dense(self.output_dim)(z)
        return tfd.Normal(loc=z, scale=self.scale)


class ScoreNet(nn.Module):
    embedding_dim: int = 128
    dim_t_embed: int = 32
    transformer_dict: dict = dataclasses.field(default_factory=lambda: {"d_model": 256, "d_mlp": 512, "n_layers": 4, "n_heads": 4, "flash_attention": True})

    @nn.compact
    def __call__(self, z, t, conditioning, mask):

        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(z.shape[0])  # Ensure t is a vector

        temb = get_timestep_embedding(t, self.dim_t_embed)  # Timestep embeddings
        cond = np.concatenate([temb, conditioning], axis=1)  # Concatenate with conditioning context

        # Pass context through a small MLP before passing into transformer
        cond = nn.gelu(nn.Dense(features=self.embedding_dim * 4)(cond))
        cond = nn.gelu(nn.Dense(features=self.embedding_dim * 4)(cond))
        cond = nn.Dense(self.embedding_dim)(cond)

        h = nn.Dense(self.embedding_dim)(z)  # Embed input before passing into transformer
        h = Transformer(n_input=self.embedding_dim, **self.transformer_dict)(h, cond, mask)

        return z + h


class VariationalDiffusionModel(nn.Module):
    timesteps: int = 1000
    gamma_min: float = -3.0
    gamma_max: float = 3.0
    embedding_dim: int = 8
    encoding_hidden_dim: int = 256
    antithetic_time_sampling: bool = True
    n_layers: int = 4
    noise_schedule: str = "learned_linear"  # "learned_linear" or "scalar"
    feature_dim: int = 3
    output_noise_scale: float = 1.0e-3
    latent_diffusion: bool = False
    dim_t_embed: int = 32
    transformer_dict: dict = dataclasses.field(default_factory=lambda: {"d_model": 256, "d_mlp": 512, "n_layers": 4, "n_heads": 4, "flash_attention": True})

    def setup(self):

        if self.noise_schedule == "learned_linear":
            self.gamma = NoiseScheduleFixedLinear(gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        elif self.noise_schedule == "scalar":
            self.gamma = NoiseScheduleScalar(gamma_min=self.gamma_min, gamma_max=self.gamma_max)

        if self.latent_diffusion:
            embedding_dim = self.embedding_dim
        else:
            embedding_dim = self.feature_dim

        self.score_model = ScoreNet(dim_t_embed=self.dim_t_embed, embedding_dim=embedding_dim, transformer_dict=self.transformer_dict)
        self.encoder = Encoder(hidden_size=self.encoding_hidden_dim, n_layers=self.n_layers, embedding_dim=embedding_dim, latent_diffusion=self.latent_diffusion)
        self.decoder = Decoder(hidden_size=self.encoding_hidden_dim, n_layers=self.n_layers, output_dim=self.feature_dim, scale=self.output_noise_scale, latent_diffusion=self.latent_diffusion)

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

    def sample_step(self, rng, i, T, z_t, conditioning, mask=None):  # , guidance_weight=0.0):
        rng_body = jax.random.fold_in(rng, i)
        eps = jax.random.normal(rng_body, z_t.shape)
        t = (T - i) / T
        s = (T - i - 1) / T

        g_s = self.gamma(s)
        g_t = self.gamma(t)

        cond = conditioning

        eps_hat_cond = self.score_model(z_t, g_t * np.ones((z_t.shape[0],), z_t.dtype), cond, mask)

        # eps_hat_uncond = self.score_model(z_t, g_t * np.ones((z_t.shape[0],), z_t.dtype), cond * 0.0, mask)

        eps_hat = eps_hat_cond  # (1.0 + guidance_weight) * eps_hat_cond - guidance_weight * eps_hat_uncond

        a = nn.sigmoid(g_s)
        b = nn.sigmoid(g_t)
        c = -np.expm1(g_t - g_s)
        sigma_t = np.sqrt(sigma2(g_t))
        z_s = np.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + np.sqrt((1.0 - a) * c) * eps

        return z_s
