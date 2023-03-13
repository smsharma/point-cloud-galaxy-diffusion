from typing import List, Optional
import jax.numpy as np
from jax import vmap, jit
from scipy.spatial import cKDTree


@jit
def interp1d(x_final: np.array, x_original: np.array, y: np.array) -> np.array:
    """Interpolate 1d array to obtain funciton y(x)

    Args:
        x_final (np.array): output x
        x_original (np.array): input x
        y (np.array): function evaluated at x_original

    Returns:
        np.array: function ```y(x)''' evaluated at x_final
    """
    return np.interp(x_final, x_original, y)


@jit
def interp1d_vectorized(
    x_final: np.array, x_original: np.array, y: np.array
) -> np.array:
    """Vectorized version of interp1d, where x_original is a 2d array

    Args:
        x_final (np.array): output x
        x_original (np.array): input x
        y (np.array): function evaluated at x_original

    Returns:
        np.array: function ```y(x)''' evaluated at x_final
    """
    return vmap(interp1d, in_axes=(None, 0, None), out_axes=0)(
        x_final,
        x_original,
        y,
    )


def get_cdf(query_volumes: np.array, sample_volumes: np.array) -> np.array:
    """Compute the CDF of ```sample_volumes''' and return it at ```query_volumes'''.
    This function will interpolate the measured CDF to evaluate it at query points.

    Args:
        query_volumes (np.array): volumes for which the CDF is evaluated
        sample_volumes (np.array): samples of volumes

    Returns:
        np.array: CDF evaluated at query_volumes
    """
    sorted_distances = np.sort(sample_volumes, axis=0)
    p = np.arange(len(sample_volumes)) + 1
    return interp1d_vectorized(
        query_volumes,
        sorted_distances.T,
        p / p[-1],
    )


def get_volume(
    distance: np.array,
    dim: int,
) -> np.array:
    """Convert distances into volumes

    Args:
        distance (np.array): array of distances
        dim (int): dimensionality

    Returns:
        np.array: volumes
    """
    volume_prefactor = [2, np.pi, 4 * np.pi / 3][dim - 1]
    return volume_prefactor * distance**dim


def get_volume_knn(
    pos: np.array, random_pos: np.array, k: List[int], boxsize: float, n_threads: int
) -> np.array:
    """Get the volumes determined by the distance to the k-nearest neighbors of random_pos

    Args:
        pos (np.array): tracer positions.
        random_pos (np.array): random positions filling the simulations volume.
        k (List[int]): list of k nearest neighbors.
        boxsize (float): size of the simulation box.
        n_threads (int): number of threads.

    Returns:
        np.array: volume determined by the k-nearest neighbors of random_pos
    """
    dim = pos.shape[-1]
    k = np.atleast_1d(k)
    tree = cKDTree(pos, boxsize=boxsize)
    knn_distance, _ = tree.query(random_pos, k=k, workers=n_threads)
    return get_volume(
        knn_distance,
        dim,
    )


def get_CDFkNN(
    r_bins: np.array,
    pos: np.array,
    random_pos: np.array,
    boxsize: float,
    k: Optional[List[int]] = [
        1,
    ],
    n_threads: Optional[int] = 1,
) -> np.array:
    """Compute the CDF of the volume determined by the k-nearest neighbors of random_pos,
    evaluated at r_bins.

    Args:
        r_bins (np.array): distances at which to evaluate the CDF.
        pos (np.array): tracer positions.
        random_pos (np.array): random positions filling the simulations volume.
        boxsize (float): size of the simulation box.
        k (Optional[List[int]], optional): list of k nearest neighbors. Defaults to [1,].
        n_threads (int): number of threads.

    Returns:
        np.array: cdf
    """
    volume_for_k = get_volume_knn(
        pos=pos, random_pos=random_pos, boxsize=boxsize, k=k, n_threads=n_threads
    )
    dim = pos.shape[-1]
    binned_volumes = [2, np.pi, 4 * np.pi / 3][dim - 1] * r_bins**dim
    return get_cdf(query_volumes=binned_volumes, sample_volumes=volume_for_k)


def cdf2peaked_cdf(cdf: np.array) -> np.array:
    """Convert a CDF into a peaked CDF

    Args:
        cdf (np.array): cdf.

    Returns:
        np.array: peaked cdf
    """
    return np.where(
        cdf < 0.5,
        cdf,
        1 - cdf,
    )