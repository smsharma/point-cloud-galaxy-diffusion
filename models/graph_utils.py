from functools import partial

import jax
import jax.numpy as np
import flax.linen as nn

import jaxkdtree

EPS = 1e-5


class PairNorm(nn.Module):
    """PairNorm normalization layer from https://arxiv.org/abs/1909.12223."""

    @nn.compact
    def __call__(self, features, rescale_factor=1.0):
        # Center features by subtracting mean
        feature_sum = np.sum(features, axis=0)
        feature_centered = features - feature_sum / features.shape[0]

        # L2 norm per node across features
        feature_l2 = np.sqrt(np.sum(np.square(feature_centered), axis=1, keepdims=True))

        # Sum L2 norms across nodes
        feature_l2_sum = np.sum(feature_l2, keepdims=True)

        # Mean L2 norm
        feature_l2_sqrt_mean = np.sqrt(feature_l2_sum / features.shape[0])

        # Divide centered by L2 norm per node and multiply by mean L2 norm
        features_normalized = feature_centered / feature_l2 * feature_l2_sqrt_mean * rescale_factor

        return features_normalized


class Identity(nn.Module):
    """Module that applies the identity function, ignoring any additional args."""

    @nn.compact
    def __call__(self, x, **args):
        return x


def fourier_features(x, num_encodings=8, include_self=True):
    """Add Fourier features to a set of coordinates

    Args:
        x (jnp.array): Coordinates
        num_encodings (int, optional): Number of Fourier feature encodings. Defaults to 16.
        include_self (bool, optional): Whether to include original coordinates in output. Defaults to True.

    Returns:
        jnp.array: Fourier features of input coordinates
    """

    dtype, orig_x = x.dtype, x
    scales = 2 ** np.arange(num_encodings, dtype=dtype)
    x = x / scales
    x = np.concatenate([np.sin(x), np.cos(x)], axis=-1)
    x = np.concatenate((x, orig_x), axis=-1) if include_self else x
    return x


def apply_pbc(dr: np.array, cell: np.array) -> np.array:
    """Apply periodic boundary conditions to a displacement vector, dr, given a cell.

    Args:
        dr (np.array): An array of shape (N,3) containing the displacement vector
        cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

    Returns:
        np.array: displacement vector with periodic boundary conditions applied
    """
    return dr - np.round(dr.dot(np.linalg.inv(cell))).dot(cell)


@partial(jax.jit, static_argnums=(1, 4))
def nearest_neighbors(
    x: np.array,
    k: int,
    mask: np.array = None,
    cell: np.array = None,
    pbc: bool = False,
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

    mask = np.bool_(mask)

    n_nodes = x.shape[0]

    # Compute the vector difference between positions
    dr = (x[:, None, :] - x[None, :, :]) + EPS
    if pbc:
        dr = apply_pbc(
            dr=dr,
            cell=cell,
        )

    # Calculate the distance matrix
    distance_matrix = np.sum(dr**2, axis=-1)

    # Apply the mask to distance matrix
    distance_matrix = np.where(mask[:, None] & mask[None, :], distance_matrix, np.inf)

    # Get indices of nearest neighbors
    indices = np.argsort(distance_matrix, axis=-1)[:, :k]

    # Create sources and targets arrays
    sources = np.repeat(np.arange(n_nodes), k)
    targets = indices.ravel()

    # return sources, targets, distance_matrix[sources, targets]
    return sources, targets, dr[sources, targets]


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


@partial(jax.jit, static_argnums=(1, 2))
def nearest_neighbors_kd(x, k, max_radius=2000.0):
    # Implementation of nearest neighbors op
    res = jaxkdtree.kNN(x, k, max_radius)
    sources = np.repeat(np.arange(x.shape[0]), k)
    targets = res.reshape(-1)
    dr = x[sources] - x[targets]
    distances = np.sum(dr**2, axis=-1)

    return sources, targets, distances


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
