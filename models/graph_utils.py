import jax
import jax.numpy as np
import jraph
from jax_md import space, partition

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


@partial(jax.jit, static_argnums=(1,))
def nearest_neighbors_ann(x, k):
    """Algorithm from https://arxiv.org/abs/2206.14286. NOTE: Does not support masking."""

    dots = np.einsum("ik,jk->ij", x, x)
    db_half_norm = np.linalg.norm(x, axis=1) ** 2 / 2.0
    dists = db_half_norm - dots
    dist, neighbours = jax.lax.approx_min_k(dists, k=k, recall_target=0.95)
    sources = np.arange(x.shape[0]).repeat(k)
    targets = neighbours.reshape(x.shape[0] * (k))
    return (sources, targets)


def add_graphs_tuples(graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(nodes=graphs.nodes + other_graphs.nodes)


class RadiusSearch:
    """Jittable radius graph"""

    def __init__(self, box_size, cutoff, boundary_cond="free", capacity_multiplier=1.5):

        self.box_size = np.array(box_size)

        if boundary_cond == "free":
            self.displacement_fn, _ = space.free()
        elif boundary_cond == "periodic":
            self.displacement_fn, _ = space.periodic(self.box_size)
        else:
            raise NotImplementedError

        self.disp = jax.vmap(self.displacement_fn)
        self.dist = jax.vmap(space.metric(self.displacement_fn))
        self.cutoff = cutoff
        self.neighbor_list_fn = partition.neighbor_list(self.displacement_fn, self.box_size, cutoff, format=partition.Sparse, dr_threshold=cutoff / 6.0, mask_self=False, capacity_multiplier=capacity_multiplier)

        self.neighbor_list_fn_jit = jax.jit(self.neighbor_list_fn)
        self.neighbor_dist_jit = self.displacement_fn

        # Each time number of neighbours buffer overflows, reallocate
        self.n_times_reallocated = 0

    def init_neighbor_lst(self, pos):
        """Allocate initial neighbour list."""
        pos = np.mod(pos, self.box_size)
        nbr = self.neighbor_list_fn.allocate(pos)
        return nbr

    def update_neighbor_lst(self, pos, nbr):
        """Update neighbour list. If buffer overflows, reallocate (re-jit)."""
        pos = np.mod(pos, self.box_size)
        nbr_update = jax.vmap(self.neighbor_list_fn_jit.update, in_axes=(0, None))(pos, nbr)

        # If buffer overflows, update capacity of neighbours.
        # NOTE: This reallocation strategy might be more efficient: https://github.com/jax-md/jax-md/issues/192#issuecomment-1114002995
        if np.any(nbr_update.did_buffer_overflow):
            nbr = self.neighbor_list_fn.allocate(pos[0], extra_capacity=2**self.n_times_reallocated)
            self.n_times_reallocated += 1

        return nbr_update, nbr
