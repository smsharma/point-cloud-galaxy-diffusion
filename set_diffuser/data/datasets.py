import tensorflow as tf
import jax
import jax.numpy as np
import pandas as pd
from absl import logging
from set_diffuser.data.nbody import NbodyDataset

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
    nbody_data = NbodyDataset(
        stage="train",
        include_pos_in_features=True,
        n_features=n_features,
        n_particles=n_particles,
    )
    train_ds = make_dataloader(
        x=nbody_data.features,
        conditioning=nbody_data.conditioning,
        mask=nbody_data.mask,
        batch_size=batch_size,
        seed=seed,
    )
    x_standarization_dict = nbody_data.get_standarization_dict()
    return train_ds, x_standarization_dict


def jetnet_dataset(n_features, n_particles, batch_size, seed, jet_type=["q", "g", "t"]):

    particle_data, jet_data = JetNet.getData(
        jet_type=jet_type, data_dir="./data/", num_particles=n_particles
    )

    # Normalize everything BUT the class (first element of `jet_data`
    jet_data_mean = jet_data[:, 1:].mean(axis=(0,))
    jet_data_std = jet_data[:, 1:].std(axis=(0,))
    x_standarization_dict = {"mean": jet_data_mean, "std": jet_data_std}

    # Remove cardinality (last element); keep pT, eta, mass as jet features for conditioning on
    conditioning = jet_data[:, :-1]

    # Get mask (to specify varying cardinality) and particle features to be modeled (eta, phi, pT)
    mask = particle_data[:, :, -1]
    x = particle_data[:, :, :n_features]

    train_ds = make_dataloader(x, conditioning, mask, batch_size, seed)

    return train_ds, x_standarization_dict


def load_data(dataset, n_features, n_particles, batch_size, seed, **kwargs):
    if dataset == "nbody":
        return nbody_dataset(n_features, n_particles, batch_size, seed)
    elif dataset == "jetnet":
        return jetnet_dataset(n_features, n_particles, batch_size, seed, **kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
