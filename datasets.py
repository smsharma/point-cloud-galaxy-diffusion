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


def get_nbody_data(
    n_features,
    n_particles,
):
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
    return x, mask, conditioning, norm_dict


def nbody_dataset(n_features, n_particles, batch_size, seed):
    x, mask, conditioning, norm_dict = get_nbody_data(n_features, n_particles)
    train_ds = make_dataloader(x, conditioning, mask, batch_size, seed)
    return train_ds, norm_dict


def jetnet_dataset(n_features, n_particles, batch_size, seed, jet_type=["q", "g", "t"], condition_on_jet_features=True):
    """Return training iterator for the JetNet dataset

    Args:
        n_features (int): Take first `n_features` features from the dataset
        n_particles (int): How many particles (typically 30 or 150)
        batch_size (int): Training batch size
        seed (int): PRNGKey seed
        jet_type (list, optional): List of particle types. Defaults to ["q", "g", "t"].
        condition_on_jet_features (bool, optional): Whether to condition on jet features. Defaults to True. If false, will only condition on jet class.

    Returns:
        train_ds, norm_dict: Training iterator and normalization dictionary (mean, std keys)
    """

    particle_data, jet_data = JetNet.getData(jet_type=jet_type, data_dir="./data/", num_particles=n_particles)

    # Normalize everything BUT the jet class and number of particles (first and last elements of `jet_data`)
    # TODO: properly normalize particle features
    jet_data_mean = jet_data[:, 1:-1].mean(axis=(0,))
    jet_data_std = jet_data[:, 1:-1].std(axis=(0,))
    jet_data[:, 1:] = (jet_data[:, 1:] - jet_data_mean + EPS) / (jet_data_std + EPS)

    # Store normalization dictionary
    norm_dict = {"mean_jet": jet_data_mean, "std_jet": jet_data_std, "mean_particle": None, "std_particle": None}

    if not condition_on_jet_features:
        conditioning = jet_data[:, :1]  # Only keep jet class as conditioning feature
    else:
        conditioning = jet_data[:, :-1]  # Keep everything except number of particles

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
