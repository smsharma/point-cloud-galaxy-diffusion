import jax
import jax.numpy as np
import flax
from ml_collections import ConfigDict

from functools import partial


def create_input_iter(ds):
    """Create an input iterator that prefetches to device."""

    def _prepare(xs):
        def _f(x):
            x = x._numpy()
            return x

        return jax.tree_util.tree_map(_f, xs)

    it = map(_prepare, ds)
    it = flax.jax_utils.prefetch_to_device(it, 2)
    return it


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def train_step(state, batch, rng, model, loss_fn, unconditional_dropout=False, p_uncond=0.0):
    """Train for a single step."""
    x, conditioning, mask = batch

    # Unconditional dropout rng
    rng, rng_uncond = jax.random.split(rng)

    # Set a fraction p_uncond of conditioning vectors to zero if unconditional_dropout is True
    if conditioning is not None and unconditional_dropout:
        random_nums = jax.random.uniform(rng_uncond, conditioning.shape[:1]).reshape(-1, 1)
        conditioning = np.where(random_nums < p_uncond, np.zeros_like(conditioning), conditioning)

    loss, grads = jax.value_and_grad(loss_fn)(state.params, model, rng, x, conditioning, mask)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics


def param_count(pytree):
    """Count the number of parameters in a pytree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(pytree))


def to_wandb_config(d: ConfigDict, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
