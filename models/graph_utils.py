import jax
import jax.numpy as np
import jraph
from jax_md import space, partition

from functools import partial


def apply_pbc(dr, cell):
    """Compute the distance between x and y accounting for periodic boundary conditions."""
    return dr - np.round(dr.dot(np.linalg.inv(cell))).dot(cell)


@partial(jax.jit, static_argnums=(1,))
def nearest_neighbors(x, k, boxsize: float = None, unit_cell=None, mask=None):
    """
    The shittiest implementation of nearest neighbours with masking in the world.
    Now with periodic boundary conditions!
    """

    if mask is None:
        mask = np.ones((x.shape[0],), dtype=np.int32)

    n_nodes = x.shape[0]

    # Compute the vector difference between positions accounting for PBC
    dr = x[:, None, :] - x[None, :, :]
    if boxsize is not None:
        dr = apply_pbc(
            dr=dr,
            cell=boxsize * unit_cell,
        )
    # Calculate the distance matrix accounting for PBC
    distance_matrix = np.sum(dr**2, axis=-1)

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


def rotation_matrix(angle_deg, axis):
    """Return the rotation matrix associated with counterclockwise rotation of `angle_deg` degrees around the given axis."""
    angle_rad = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)

    a = np.cos(angle_rad / 2)
    b, c, d = -axis * np.sin(angle_rad / 2)

    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)], [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)], [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def rotate_representation(data, angle_deg, axis):
    """Rotate `data` by `angle_deg` degrees around `axis`."""
    rot_mat = rotation_matrix(angle_deg, axis)

    if data.shape[-1] == 3:
        return np.matmul(rot_mat, data.T).T
    else:
        positions = data[:, :3]
        velocities = data[:, 3:6]
        scalars = data[:, 6:]

        rotated_positions = np.matmul(rot_mat, positions.T).T
        rotated_velocities = np.matmul(rot_mat, velocities.T).T

        return np.concatenate([rotated_positions, rotated_velocities, scalars], axis=1)
