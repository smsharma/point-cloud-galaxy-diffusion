import jax
import jax.numpy as np
import jraph

# from jax_md import space, partition

from functools import partial


@partial(jax.jit, static_argnums=(1,))
def nearest_neighbors(
    x: np.array,
    k: int,
    mask: np.array = None,
):
    """Returns the nearest neighbors of each node in x.

    Args:
        x (np.array): positions of nodes
        k (int): number of nearest neighbors to find
        mask (np.array, optional): node mask. Defaults to None.

    Returns:
        sources, targets: pairs of neighbors
    """
    if mask is None:
        mask = np.ones((x.shape[0],), dtype=np.int32)

    n_nodes = x.shape[0]
    # Compute the vector difference between positions
    dr = x[:, None, :] - x[None, :, :]

    # Calculate the distance matrix
    distance_matrix = np.sum(dr**2, axis=-1)

    distance_matrix = np.where(mask[:, None], distance_matrix, np.inf)
    distance_matrix = np.where(mask[None, :], distance_matrix, np.inf)

    indices = np.argsort(distance_matrix, axis=-1)[:, :k]

    sources = indices[:, 0].repeat(k)
    targets = indices.reshape(n_nodes * (k))

    return (sources, targets, dr[sources, targets])


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
    if data.shape[1] == 3:
        return np.matmul(rot_mat, data.T).T
    positions = data[:, :3]
    velocities = data[:, 3:6]
    scalars = data[:, 6:]

    rotated_positions = np.matmul(rot_mat, positions.T).T
    rotated_velocities = np.matmul(rot_mat, velocities.T).T
    return np.concatenate([rotated_positions, rotated_velocities, scalars], axis=1)


def replicate_box(
    features,
    box_size,
    n_pos_dim=3,
):
    n_replications = np.array([1, 1, 1])
    indices = np.indices(2 * n_replications + 1).reshape(3, -1).T
    displacements = indices * box_size - n_replications * box_size
    positions = features[:, :n_pos_dim]
    replicated_positions = positions[:, np.newaxis, :] + displacements[np.newaxis, :, :]
    replicated_features = np.repeat(features, indices.shape[0], axis=0)
    unfolded_positions = replicated_positions.reshape(-1, 3)
    replicated_features = replicated_features.at[:, :n_pos_dim].set(unfolded_positions)
    return replicated_features


def get_rotated_box(features, rotation_axis, rotation_angle, n_pos_dim=3, box_size: float = 1000.0):
    unfolded_features = replicate_box(features, box_size)
    rotated_features = rotate_representation(
        unfolded_features,
        rotation_angle,
        rotation_axis,
    )
    mask_in_box = np.all(
        (rotated_features[:, :n_pos_dim] >= 0) & (rotated_features[:, :n_pos_dim] <= box_size),
        axis=1,
    )
    return rotated_features[mask_in_box]
