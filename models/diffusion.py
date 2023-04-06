import dataclasses

from absl import logging

import jax
import flax.linen as nn
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp

from models.diffusion_utils import variance_preserving_map, alpha, sigma2
from models.diffusion_utils import NoiseScheduleScalar, NoiseScheduleFixedLinear
from models.scores import TransformerScoreNet, GraphScoreNet, EquivariantTransformerNet, EGNNScoreNet, NEQUIPScoreNet
from models.mlp import MLPEncoder, MLPDecoder

tfd = tfp.distributions


class VariationalDiffusionModel(nn.Module):
    """Variational Diffusion Model (VDM), adapted from https://github.com/google-research/vdm

    Attributes:
      d_feature: Number of features per set element.
      timesteps: Number of diffusion steps.
      gamma_min: Minimum log-SNR in the noise schedule (init if learned).
      gamma_max: Maximum log-SNR in the noise schedule (init if learned).
      antithetic_time_sampling: Antithetic time sampling to reduce variance.
      noise_schedule: Noise schedule; "learned_linear" or "scalar".
      noise_scale: Std of Normal noise model.
      d_t_embedding: Dimensions the timesteps are embedded to.
      score: Score function; "transformer", "graph", or "equivariant".
      score_dict: Dict of score arguments (see scores.py docstrings).
      n_classes: Number of classes in data. If >0, the first element of the conditioning vector is assumed to be integer class.
      embed_context: Whether to embed the conditioning context.
      use_encdec: Whether to use an encoder-decoder for latent diffusion.
    """

    d_feature: int = 3
    timesteps: int = 1000
    gamma_min: float = -8.0
    gamma_max: float = 14.0
    antithetic_time_sampling: bool = True
    noise_schedule: str = "learned_linear"  # "learned_linear" or "scalar"
    noise_scale: float = 1.0e-3
    d_t_embedding: int = 32
    score: str = "transformer"  # "transformer", "graph", "equivariant"
    score_dict: dict = dataclasses.field(default_factory=lambda: {"d_model": 256, "d_mlp": 512, "n_layers": 4, "n_heads": 4})
    encoder_dict: dict = dataclasses.field(default_factory=lambda: {"d_embedding": 12, "d_hidden": 256, "n_layers": 4})
    decoder_dict: dict = dataclasses.field(default_factory=lambda: {"d_hidden": 256, "n_layers": 4})
    n_classes: int = 0
    embed_context: bool = False
    d_context_embedding: int = 32
    use_encdec: bool = True

    def setup(self):

        # Noise schedule for diffusion
        if self.noise_schedule == "learned_linear":
            self.gamma = NoiseScheduleFixedLinear(gamma_min=self.gamma_min, gamma_max=self.gamma_max)
        elif self.noise_schedule == "scalar":
            self.gamma = NoiseScheduleScalar(gamma_min=self.gamma_min, gamma_max=self.gamma_max)

        # Score model specification
        if self.score == "transformer":
            self.score_model = TransformerScoreNet(d_t_embedding=self.d_t_embedding, score_dict=self.score_dict)
        elif self.score == "graph":
            self.score_model = GraphScoreNet(d_t_embedding=self.d_t_embedding, score_dict=self.score_dict)
        elif self.score == "egnn":
            self.score_model = EGNNScoreNet(d_t_embedding=self.d_t_embedding, score_dict=self.score_dict)
        elif self.score == "nequip":
            self.score_model = NEQUIPScoreNet(d_t_embedding=self.d_t_embedding, score_dict=self.score_dict)
        elif self.score == "equivariant":
            self.score_model = EquivariantTransformerNet(d_t_embedding=self.d_t_embedding, score_dict=self.score_dict)

        # Optional encoder/decoder for latent diffusion
        if self.use_encdec:
            self.encoder = MLPEncoder(**self.encoder_dict)
            self.decoder = MLPDecoder(d_output=self.d_feature, noise_scale=self.noise_scale, **self.decoder_dict)

        # Embedding for class and context
        if self.n_classes > 0:
            self.embedding_class = nn.Embed(self.n_classes, self.d_context_embedding)
        self.embedding_context = nn.Dense(self.d_context_embedding)

    def gammat(self, t):
        return self.gamma(t)

    def recon_loss(self, x, f, cond):
        """The reconstruction loss measures the gap in the first step.
        We measure the gap from encoding the image to z_0 and back again.
        """
        g_0 = self.gamma(0.0)
        eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        loss_recon = -self.decode(z_0_rescaled, cond).log_prob(x)
        return loss_recon

    def latent_loss(self, f):
        """The latent loss measures the gap in the last step, this is the KL
        divergence between the final sample from the forward process and starting
        distribution for the reverse process, here taken to be a N(0,1).
        """
        g_1 = self.gamma(1.0)
        var_1 = sigma2(g_1)
        mean1_sqr = (1.0 - var_1) * np.square(f)
        loss_klz = 0.5 * (mean1_sqr + var_1 - np.log(var_1) - 1.0)
        return loss_klz

    def diffusion_loss(self, t, f, cond, mask):
        """The diffusion loss measures the gap in the intermediate steps."""

        # Sample z_t
        g_t = self.gamma(t)
        eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_t = variance_preserving_map(f, g_t[:, None], eps)

        eps_hat = self.score_model(z_t, g_t, cond, mask)  # Compute predicted noise

        loss_diff_mse = np.square(eps - eps_hat)  # Compute MSE of predicted noise

        T = self.timesteps

        # NOTE: retain dimension here so that mask can be applied later (hence dummy dims)
        # NOTE: opposite sign convention to official VDM repo!
        if T == 0:
            # Loss for infinite depth T, i.e. continuous time
            _, g_t_grad = jax.jvp(self.gamma, (t,), (np.ones_like(t),))
            loss_diff = -0.5 * g_t_grad[:, None, None] * loss_diff_mse
        else:
            # Loss for finite depth T, i.e. discrete time
            s = t - (1.0 / T)
            g_s = self.gamma(s)
            loss_diff = 0.5 * T * np.expm1(g_s - g_t)[:, None, None] * loss_diff_mse

        return loss_diff

    def __call__(self, x, conditioning=None, mask=None):

        d_batch = x.shape[0]

        # 1. Reconstruction loss
        # Add noise and reconstruct
        f = self.encode(x, conditioning)
        loss_recon = self.recon_loss(x, f, conditioning)

        # 2. Latent loss
        # KL z1 with N(0,1) prior
        loss_klz = self.latent_loss(f)

        # 3. Diffusion loss
        # Sample time steps
        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = np.mod(t0 + np.arange(0.0, 1.0, step=1.0 / d_batch), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(d_batch,))

        # Discretize time steps if we're working with discrete time
        T = self.timesteps
        if T > 0:
            t = np.ceil(t * T) / T

        cond = self.embed(conditioning)
        loss_diff = self.diffusion_loss(t, f, cond, mask)

        return (loss_diff, loss_klz, loss_recon)

    def embed(self, conditioning):
        """Embed the conditioning vector, optionally including embedding a class assumed to be the first element of the context vector."""
        if not self.embed_context:
            return conditioning
        else:
            if self.n_classes > 0 and conditioning.shape[-1] > 1:  # If both classes and conditioning
                classes, conditioning = conditioning[..., 0].astype(np.int32), conditioning[..., 1:]
                class_embedding, context_embedding = self.embedding_class(classes), self.embedding_context(conditioning)
                return class_embedding + context_embedding
            elif self.n_classes > 0 and conditioning.shape[-1] == 1:  # If no conditioning but classes
                classes = conditioning[..., 0].astype(np.int32)
                class_embedding = self.embedding_class(classes)
                return class_embedding
            elif self.n_classes == 0 and conditioning is not None:  # If no classes but conditioning
                context_embedding = self.embedding_context(conditioning)
                return context_embedding
            else:  # If no conditioning
                return None

    def encode(self, x, conditioning=None, mask=None):
        """Encode an image x."""

        # Encode if using encoder-decoder; otherwise just return data sample
        if self.use_encdec:
            if conditioning is not None:
                cond = self.embed(conditioning)
            else:
                cond = None
            return self.encoder(x, cond, mask)
        else:
            return x

    def decode(self, z0, conditioning=None, mask=None):
        """Decode a latent sample z0."""

        # Decode if using encoder-decoder; otherwise just return last latent distribution
        if self.use_encdec:
            if conditioning is not None:
                cond = self.embed(conditioning)
            else:
                cond = None
            return self.decoder(z0, cond, mask)
        else:
            return tfd.Normal(loc=z0, scale=self.noise_scale)

    def sample_step(self, rng, i, T, z_t, conditioning=None, mask=None):
        """Sample a single step of the diffusion process."""
        rng_body = jax.random.fold_in(rng, i)
        eps = jax.random.normal(rng_body, z_t.shape)
        t = (T - i) / T
        s = (T - i - 1) / T

        g_s = self.gamma(s)
        g_t = self.gamma(t)

        cond = self.embed(conditioning)

        eps_hat_cond = self.score_model(z_t, g_t * np.ones((z_t.shape[0],), z_t.dtype), cond, mask)

        a = nn.sigmoid(g_s)
        b = nn.sigmoid(g_t)
        c = -np.expm1(g_t - g_s)
        sigma_t = np.sqrt(sigma2(g_t))
        z_s = np.sqrt(a / b) * (z_t - sigma_t * c * eps_hat_cond) + np.sqrt((1.0 - a) * c) * eps

        return z_s
