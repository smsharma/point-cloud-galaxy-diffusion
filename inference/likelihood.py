import sys

sys.path.append("../")
from functools import partial
import jax
import jax.numpy as np


def elbo(vdm, params, rng, x, conditioning, mask, steps=2, unroll_loop=False):
    rng, spl = jax.random.split(rng)
    cond = vdm.apply(params, conditioning, method=vdm.embed)
    f = vdm.apply(params, x, conditioning, method=vdm.encode)
    loss_recon = vdm.apply(
        params, x, f, conditioning, rngs={"sample": rng}, method=vdm.recon_loss
    )
    loss_klz = vdm.apply(params, f, method=vdm.latent_loss)

    if not unroll_loop:

        def body_fun(i, val):
            loss, rng = val
            rng, spl = jax.random.split(rng)
            new_loss = vdm.apply(
                params,
                np.array([i / steps]),
                f,
                cond,
                mask,
                rngs={"sample": spl},
                method=vdm.diffusion_loss,
            )
            return (loss + (new_loss * mask[..., None]).sum((-1, -2)) / steps, rng)

        loss_diff, rng = jax.lax.fori_loop(
            0, steps, body_fun, (np.zeros(x.shape[0]), rng)
        )

    else:
        loss_diff, rng = (np.zeros(x.shape[0]), rng)

        for i in range(steps):
            rng, spl = jax.random.split(rng)
            new_loss = vdm.apply(
                params,
                np.array([i / steps]),
                f,
                cond,
                mask,
                rngs={"sample": spl},
                method=vdm.diffusion_loss,
            )
            loss_diff = loss_diff + (new_loss * mask[..., None]).sum((-1, -2)) / steps

    return (
        (loss_recon * mask[..., None]).sum((-1, -2))
        + (loss_klz * mask[..., None]).sum((-1, -2))
        + loss_diff
    )


@partial(jax.jit, static_argnums=(0, 5, 6))
def likelihood(vdm, rng, restored_state_params, x_test, params, steps=6, n_samples=2):
    x_test = np.repeat(np.array([x_test]), n_samples, 0)
    theta_test = np.repeat(np.array([params]), n_samples, 0)
    return -elbo(
        vdm=vdm,
        params=restored_state_params,
        rng=rng,
        x=x_test,
        conditioning=theta_test,
        mask=np.ones_like(x_test[..., 0]),
        steps=steps,
    ).mean()
