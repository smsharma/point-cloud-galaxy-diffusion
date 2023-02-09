import tensorflow as tf
import jax
import jax.numpy as np
import pandas as pd
from absl import logging

try:
    from jetnet.datasets import JetNet
except ImportError:
    logging.info("JetNet dataset not available.")

EPS = 1e-7


def make_dataloader(x, conditioning, mask, batch_size, seed):

    n_train = len(x)

    train_ds = tf.data.Dataset.from_tensor_slices((x, conditioning, mask))
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()

    batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

    for _batch_size in reversed(batch_dims):
        train_ds = train_ds.batch(_batch_size, drop_remainder=False)

    train_ds = train_ds.shuffle(n_train, seed=seed)

    return train_ds


def nbody_dataset(n_features, n_particles, batch_size, seed):

    x = np.load("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/halos.npy")
    conditioning = np.array(pd.read_csv("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/cosmology.csv").values)

    # Standardize per-feature (over datasets and particles)
    x_mean = x.mean(axis=(0, 1))
    x_std = x.std(axis=(0, 1))
    x = (x - x_mean + EPS) / (x_std + EPS)

    # Finalize
    x = x[:, :n_particles, :n_features]
    mask = np.ones((x.shape[0], n_particles))  # No mask

    train_ds = make_dataloader(x, conditioning, mask, batch_size, seed)

    return train_ds


def jetnet_dataset(n_features, n_particles, batch_size, seed, jet_type=["q", "g", "t"]):

    particle_data, jet_data = JetNet.getData(jet_type=jet_type, data_dir="./data/", num_particles=n_particles)

    # Normalize everything BUT the class (first element of `jet_data`
    jet_data_mean = jet_data[:, 1:].mean(axis=(0,))
    jet_data_std = jet_data[:, 1:].std(axis=(0,))
    jet_data[:, 1:] = (jet_data[:, 1:] - jet_data_mean) / (jet_data_std + EPS)

    # Remove cardinality (last element); keep pT, eta, mass as jet features for conditioning on
    conditioning = jet_data[:, :-1]

    # Get mask (to specify varying cardinality) and particle features to be modeled (eta, phi, pT)
    mask = particle_data[:, :, -1]
    x = particle_data[:, :, :n_features]

    # Standardize per-feature (over datasets and particles)
    x_mean = x.mean(axis=(0, 1))
    x_std = x.std(axis=(0, 1))
    x = (x - x_mean + EPS) / (x_std + EPS)

    train_ds = make_dataloader(x, conditioning, mask, batch_size, seed)

    return train_ds


def load_data(dataset, n_features, n_particles, batch_size, seed, **kwargs):
    if dataset == "nbody":
        train_ds = nbody_dataset(n_features, n_particles, batch_size, seed)
    elif dataset == "jetnet":
        train_ds = jetnet_dataset(n_features, n_particles, batch_size, seed, **kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return train_ds
