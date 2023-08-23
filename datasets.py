import tensorflow as tf
import time
import jax
import jax.numpy as np
import numpy as vnp
from pathlib import Path
import pandas as pd
from absl import logging
from models.graph_utils import get_rotated_box

EPS = 1e-7


def make_dataloader(x, conditioning, mask, batch_size, seed, shuffle=True):
    n_train = len(x)

    train_ds = tf.data.Dataset.from_tensor_slices((x, conditioning, mask))
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()

    batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

    for _batch_size in reversed(batch_dims):
        train_ds = train_ds.batch(_batch_size, drop_remainder=False)

    if shuffle:
        train_ds = train_ds.shuffle(n_train, seed=seed)
    return train_ds


def get_halo_data(data_dir, n_features, n_particles, split: str = "train"):
    x = np.load(data_dir / f"{split}_halos.npy")
    conditioning = np.array(pd.read_csv(data_dir / f"{split}_cosmology.csv").values)
    if n_features == 7:
        x = x.at[:, :, -1].set(np.log10(x[:, :, -1]))  # Use log10(mass)
    x = x[:, :n_particles, :n_features]
    return x, conditioning


def get_nbody_data(
    n_features,
    n_particles,
    split: str = "train",
):
    DATA_DIR = Path("/n/holystore01/LABS/iaifi_lab/Lab/set-diffuser-data/")
    x, conditioning = get_halo_data(data_dir=DATA_DIR, n_features=n_features, n_particles=n_particles, split=split)
    if split == "train":
        x_train = x
    else:
        x_train, _ = get_halo_data(
            data_dir=DATA_DIR,
            n_features=n_features,
            n_particles=n_particles,
            split="train",
        )
    # Standardize per-feature (over datasets and particles)
    x_mean = x_train.mean(axis=(0, 1))
    x_std = x_train.std(axis=(0, 1))
    norm_dict = {"mean": x_mean, "std": x_std}
    conditioning = conditioning[:, [0, -1]]  # Select only omega_m and sigma_8
    mask = np.ones((x.shape[0], n_particles))  # No mask
    x = (x - x_mean + EPS) / (x_std + EPS)
    # Finalize
    return x, mask, conditioning, norm_dict


def nbody_dataset(
    n_features,
    n_particles,
    batch_size,
    seed,
    split: str = "train",
    shuffle: bool = True,
):
    x, mask, conditioning, norm_dict = get_nbody_data(
        n_features,
        n_particles,
        split=split,
    )
    ds = make_dataloader(
        x,
        conditioning,
        mask,
        batch_size,
        seed,
        shuffle=shuffle,
    )
    return ds, norm_dict


def load_data(dataset, n_features, n_particles, batch_size, seed, shuffle, split, **kwargs):
    if dataset == "nbody":
        train_ds, norm_dict = nbody_dataset(
            n_features,
            n_particles,
            batch_size,
            seed,
            shuffle=shuffle,
            split=split,
            **kwargs,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return train_ds, norm_dict


def augment_with_translations(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    rng, _ = jax.random.split(rng)
    x = x * norm_dict["std"] + norm_dict["mean"]

    # Draw N random translations
    translations = jax.random.uniform(rng, minval=-box_size / 2, maxval=box_size / 2, shape=(*x.shape[:2], 3))
    x = x.at[..., :n_pos_dim].set((x[..., :n_pos_dim] + translations[..., None, :]) % box_size)
    x = (x - norm_dict["mean"]) / norm_dict["std"]
    return x, conditioning, mask


def random_symmetry_matrix(key):
    # 8 possible sign combinations for reflections
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    # 6 permutations for axis swapping
    perms = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])

    # Randomly select one sign combination and one permutation
    sign = signs[jax.random.randint(key, (), 0, 8)]
    perm = perms[jax.random.randint(key, (), 0, 6)]

    # Combine them to form the random symmetry matrix
    matrix = np.eye(3)[perm] * sign
    return matrix


def augment_with_symmetries(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    rng, _ = jax.random.split(rng)
    # Rotations and reflections that respect boundary conditions
    matrix = random_symmetry_matrix(rng)
    x = x.at[..., :n_pos_dim].set(np.dot(x[..., :n_pos_dim], matrix.T))
    if x.shape[-1] > n_pos_dim:
        # Rotate velocities too
        x = x.at[..., n_pos_dim : n_pos_dim + 3].set(np.dot(x[..., n_pos_dim : n_pos_dim + 3], matrix.T))
    return x, conditioning, mask


def augment_data(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    rotations: bool = True,
    translations: bool = True,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    if rotations:
        x, conditioning, mask = augment_with_symmetries(
            x=x,
            mask=mask,
            conditioning=conditioning,
            rng=rng,
            norm_dict=norm_dict,
            n_pos_dim=n_pos_dim,
            box_size=box_size,
        )
    if translations:
        x, conditioning, mask = augment_with_translations(
            x=x,
            mask=mask,
            conditioning=conditioning,
            rng=rng,
            norm_dict=norm_dict,
            n_pos_dim=n_pos_dim,
            box_size=box_size,
        )
    return x, conditioning, mask
