import jax
import jax.numpy as np
import jraph
from jax_md import space, partition

from functools import partial

def wrap_positions_to_periodic_box(positions: np.array, cell_matrix: np.array)->np.array:
    """
    Apply periodic boundary conditions to a set of positions.

    Args:
        positions (np.array): An array of shape (N, 3) containing the particle positions.
        cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

    Returns:
        numpy.ndarray: An array of shape (N, 3) containing the wrapped particle positions.
    """
    inv_cell_matrix = np.linalg.inv(cell_matrix)
    fractional_positions = np.matmul(positions, inv_cell_matrix)
    fractional_positions = np.mod(fractional_positions, 1.0)
    return np.matmul(fractional_positions, cell_matrix)

def apply_pbc(dr: np.array, cell: np.array) -> np.array:
    """Apply periodic boundary conditions to a displacement vector, dr, given a cell.

    Args:
        dr (np.array): An array of shape (N,3) containing the displacement vector
        cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

    Returns:
        np.array: displacement vector with periodic boundary conditions applied
    """
    return dr - np.round(dr.dot(np.linalg.inv(cell))).dot(cell)


@partial(jax.jit, static_argnums=(1,))
def nearest_neighbors(
    x: np.array,
    k: int,
    boxsize: float = None,
    unit_cell: np.array = None,
    mask: np.array = None,
):
    """Returns the nearest neighbors of each node in x.

    Args:
        x (np.array): positions of nodes
        k (int): number of nearest neighbors to find
        boxsize (float, optional): size of box if perdioc boundary conditions. Defaults to None.
        unit_cell (np.array, optional): unit cell for applying periodic boundary conditions. Defaults to None.
        mask (np.array, optional): node mask. Defaults to None.

    Returns:
        sources, targets: pairs of neighbors
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


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
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
        self.neighbor_list_fn = partition.neighbor_list(
            self.displacement_fn,
            self.box_size,
            cutoff,
            format=partition.Sparse,
            dr_threshold=cutoff / 6.0,
            mask_self=False,
            capacity_multiplier=capacity_multiplier,
        )

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
        nbr_update = jax.vmap(self.neighbor_list_fn_jit.update, in_axes=(0, None))(
            pos, nbr
        )

        # If buffer overflows, update capacity of neighbours.
        # NOTE: This reallocation strategy might be more efficient: https://github.com/jax-md/jax-md/issues/192#issuecomment-1114002995
        if np.any(nbr_update.did_buffer_overflow):
            nbr = self.neighbor_list_fn.allocate(
                pos[0], extra_capacity=2**self.n_times_reallocated
            )
            self.n_times_reallocated += 1

        return nbr_update, nbr


def rotation_matrix(angle_deg, axis):
    """Return the rotation matrix associated with counterclockwise rotation of `angle_deg` degrees around the given axis."""
    angle_rad = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)

    a = np.cos(angle_rad / 2)
    b, c, d = -axis * np.sin(angle_rad / 2)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def rotate_representation(data, angle_deg, axis):
    """Rotate `data` by `angle_deg` degrees around `axis`."""
    rot_mat = rotation_matrix(angle_deg, axis)

    positions = data[:, :3]
    velocities = data[:, 3:6]
    scalars = data[:, 6:]

    rotated_positions = np.matmul(rot_mat, positions.T).T
    rotated_velocities = np.matmul(rot_mat, velocities.T).T

    return np.concatenate([rotated_positions, rotated_velocities, scalars], axis=1)
