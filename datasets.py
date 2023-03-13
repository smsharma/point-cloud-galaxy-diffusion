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

    if n_features == 7:
        x = x.at[:, :, -1].set(np.log10(x[:, :, -1]))  # Use log10(mass)

    x = x[:, :n_particles, :n_features]
    # Standardize per-feature (over datasets and particles)
    x_mean = x.mean(axis=(0, 1))
    x_std = x.std(axis=(0, 1))
    x = (x - x_mean + EPS) / (x_std + EPS)
    norm_dict = {"mean": x_mean, "std": x_std}

    # Finalize
    mask = np.ones((x.shape[0], n_particles))  # No mask
    conditioning = conditioning[:, [0, -1]]  # Select only omega_m and sigma_8

    train_ds = make_dataloader(x, conditioning, mask, batch_size, seed)

    return train_ds, norm_dict


def jetnet_dataset(n_features, n_particles, batch_size, seed, jet_type=["q", "g", "t"], condition_on_jet_features=True):

    particle_data, jet_data = JetNet.getData(jet_type=jet_type, data_dir="./data/", num_particles=n_particles)

    # Normalize everything BUT the class (first element of `jet_data`)
    jet_data_mean = jet_data[:, 1:].mean(axis=(0,))
    jet_data_std = jet_data[:, 1:].std(axis=(0,))
    jet_data[:, 1:] = (jet_data[:, 1:] - jet_data_mean + EPS) / (jet_data_std + EPS)
    norm_dict = {"mean": jet_data_mean, "std": jet_data_std}

    # Only keep jet class as conditioning feature
    if not condition_on_jet_features:
        conditioning = jet_data[:, :1]

    # Get mask (to specify varying cardinality) and particle features to be modeled (eta, phi, pT)
    mask = particle_data[:, :, -1]
    x = particle_data[:, :, :n_features]

    train_ds = make_dataloader(x, conditioning, mask, batch_size, seed)

    return train_ds, norm_dict


def load_data(dataset, n_features, n_particles, batch_size, seed, **kwargs):
    if dataset == "nbody":
        train_ds, norm_dict = nbody_dataset(n_features, n_particles, batch_size, seed)
    elif dataset == "jetnet":
        train_ds, norm_dict = jetnet_dataset(n_features, n_particles, batch_size, seed, **kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return train_ds, norm_dict
