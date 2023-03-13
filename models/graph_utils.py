import jax
import jax.numpy as np

import jraph

from functools import partial


@partial(jax.jit, static_argnums=(1,))
def nearest_neighbors(x, k, mask=None):
    """The shittiest implementation of nearest neighbours with masking in the world"""

    if mask is None:
        mask = np.ones((x.shape[0],), dtype=np.int32)

    n_nodes = x.shape[0]

    distance_matrix = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)

    distance_matrix = np.where(mask[:, None], distance_matrix, np.inf)
    distance_matrix = np.where(mask[None, :], distance_matrix, np.inf)

    indices = np.argsort(distance_matrix, axis=-1)[:, :k]

    sources = indices[:, 0].repeat(k)
    targets = indices.reshape(n_nodes * (k))

    return (sources, targets)


def add_graphs_tuples(graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(nodes=graphs.nodes + other_graphs.nodes)
