import tensorflow as tf
import jax
import jax.numpy as np
import pandas as pd

EPS = 1e-7


def make_dataloader(x, conditioning, mask, batch_size, seed):

    n_train = len(x)

    train_ds = tf.data.Dataset.from_tensor_slices((x, conditioning, mask))
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()

    batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

    for batch_size in reversed(batch_dims):
        train_ds = train_ds.batch(batch_size, drop_remainder=False)

    train_ds = train_ds.shuffle(n_train, seed=seed)

    return train_ds


def nbody_dataset(n_features, n_particles, batch_size, seed):

    x = np.load("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/halos.npy")
    conditioning = np.array(pd.read_csv("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/cosmology.csv").values)

    # Standardize
    x_mean = x.mean(axis=(0, 1))
    x_std = x.std(axis=(0, 1))
    x = (x - x_mean + EPS) / (x_std + EPS)

    # Finalize
    x = x[:, :n_particles, :n_features]
    mask = np.ones((x.shape[0], n_particles))

    train_ds = make_dataloader(x, conditioning, mask, batch_size, seed)

    return train_ds


def load_data(dataset, n_features, n_particles, batch_size, seed):
    if dataset == "nbody":
        train_ds = nbody_dataset(n_features, n_particles, batch_size, seed)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return train_ds
