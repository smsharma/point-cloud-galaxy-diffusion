import jax
import jax.numpy as np
import flax
import optax
from ml_collections import ConfigDict

from functools import partial
from typing import Any


@flax.struct.dataclass
class StateStore:
    """A simple state store for training."""

    params: np.ndarray
    state: Any
    rng: Any
    step: int = 0


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


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(1, 2, 4))
def train_step(store, loss_fn, model, batch, opt):
    """Train for a single step."""
    rng, spl = jax.random.split(store.rng)
    x, conditioning, mask = batch
    loss, grads = jax.value_and_grad(loss_fn)(store.params, model, spl, x, conditioning, mask)
    grads = jax.lax.pmean(grads, "batch")
    updates, state = opt.update(grads, store.state, store.params)
    params = optax.apply_updates(store.params, updates)

    metrics = {'loss':jax.lax.pmean(loss, "batch")}

    return store.replace(params=params, state=state, rng=rng, step=store.step + 1), metrics


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
