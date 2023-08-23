import jax
import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
from functools import partial


def elbo(vdm, params, rng, x, conditioning, mask, steps=20, unroll_loop=True):
    rng, spl = jax.random.split(rng)
    cond = vdm.apply(params, conditioning, method=vdm.embed)
    f = vdm.apply(params, x, conditioning, method=vdm.encode)
    loss_recon = vdm.apply(params, x, f, conditioning, rngs={"sample": rng}, method=vdm.recon_loss)
    loss_klz = vdm.apply(params, f, method=vdm.latent_loss)

    if not unroll_loop:

        def body_fun(i, val):
            loss, rng = val
            rng, spl = jax.random.split(rng)
            new_loss = vdm.apply(params, np.array([i / steps]), f, cond, mask, rngs={"sample": spl}, method=vdm.diffusion_loss)
            return (loss + (new_loss * mask[..., None]).sum((-1, -2)) / steps, rng)

        loss_diff, rng = jax.lax.fori_loop(0, steps, body_fun, (np.zeros(x.shape[0]), rng))

    else:
        loss_diff, rng = (np.zeros(x.shape[0]), rng)

        for i in range(steps):
            rng, spl = jax.random.split(rng)
            new_loss = vdm.apply(params, np.array([i / steps]), f, cond, mask, rngs={"sample": spl}, method=vdm.diffusion_loss)
            loss_diff = loss_diff + (new_loss * mask[..., None]).sum((-1, -2)) / steps

    return (loss_recon * mask[..., None]).sum((-1, -2)) + (loss_klz * mask[..., None]).sum((-1, -2)) + loss_diff


def likelihood(params, theta, x, rng, vdm, n_samples=1, steps=4):
    x = np.repeat(np.array([x]), n_samples, 0)
    theta = np.repeat(np.array([theta]), n_samples, 0)
    return -jax.vmap(elbo, in_axes=(None, None, None, 1, 1, 1, None, None))(vdm, params, rng, x, theta, np.ones_like(x[..., 0]), steps, True).mean()


def loss_fn_discriminative(params, x, theta, model, rng, n_samples, steps):
    """Loss function for the discriminative model.
    p(theta | x) = p(x | theta) * p(theta) / p(x)"""

    rng, spl = jax.random.split(rng)

    log_prior = tfp.distributions.Uniform(low=(0.1, 0.6), high=(0.5, 1.0)).log_prob(theta).sum()
    log_like = likelihood(params, theta, x, spl, model, n_samples, steps)
    log_ev = likelihood(params, np.zeros_like(theta), x, rng, model, n_samples, steps)
    return -(log_like + log_prior - log_ev)


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5))
def train_step(state, batch, rng, model, n_samples, steps):
    """Train for a single step."""
    x, conditioning, mask = batch
    loss, grads = jax.value_and_grad(loss_fn_discriminative)(state.params, x, conditioning, model, rng, n_samples, steps)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics
