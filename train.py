import sys
import os
import shutil

sys.path.append("./")
sys.path.append("../")

import pandas as pd
from tqdm import trange

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import checkpoints

import tensorflow as tf

from models.diffusion import VariationalDiffusionModel
from models.diffusion_utils import loss_vdm
from models.train_utils import create_input_iter, param_count, StateStore, train_step

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

# Ensure TF does not see GPU and grab all GPU memory
tf.config.set_visible_devices([], device_type="GPU")

EPS = 1e-7


def train():

    # VDM args
    n_particles = 5000
    n_features = 3  # Input features per set element
    save_every = 4000
    timesteps = 1000
    d_hidden_encoding = 512
    n_encoder_layers = 5
    d_embedding = 10
    embed_context = False

    # Training config
    learning_rate = 6e-4  # Peak learning rate
    weight_decay = 1e-4
    batch_size = 16  # Must be divisible by number of devices; this is the total batch size, not per-device
    n_train_steps = 75_000
    warmup_steps = 1000

    # Transformer args
    induced_attention = True
    n_inducing_points = 300
    d_model = 256
    d_mlp = 1024
    n_transformer_layers = 8
    n_heads = 4

    ckpt_dir = "/n/holystore01/LABS/iaifi_lab/Users/smsharma/set-diffuser/notebooks/ckpts_debug_inducing/"

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    print("{} devices visible".format(jax.device_count()))

    # Load data

    x = np.load("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/halos.npy")
    x_mean = x.mean(axis=(0, 1))
    x_std = x.std(axis=(0, 1))
    x = (x - x_mean + EPS) / (x_std + EPS)

    x = x[:, :n_particles, :n_features]
    conditioning = np.array(pd.read_csv("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/cosmology.csv").values)

    mask = np.ones((x.shape[0], n_particles))

    # Make dataloader

    n_train = len(x)

    train_ds = tf.data.Dataset.from_tensor_slices((x, conditioning, mask))
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()

    batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

    for batch_size in reversed(batch_dims):
        train_ds = train_ds.batch(batch_size, drop_remainder=False)

    train_ds = train_ds.shuffle(n_train, seed=42)

    # Model configuration

    transformer_dict = FrozenDict({"d_model": d_model, "d_mlp": d_mlp, "n_layers": n_transformer_layers, "n_heads": n_heads, "induced_attention": induced_attention, "n_inducing_points": n_inducing_points})  # Transformer args

    vdm = VariationalDiffusionModel(n_layers=n_encoder_layers, d_embedding=d_embedding, d_hidden_encoding=d_hidden_encoding, timesteps=timesteps, d_feature=n_features, transformer_dict=transformer_dict, embed_context=embed_context)
    batches = create_input_iter(train_ds)

    # Pass a test batch through to initialize model
    x_batch, conditioning_batch, mask_batch = next(batches)
    rng = jax.random.PRNGKey(42)
    _, params = vdm.init_with_output({"sample": rng, "params": rng}, x_batch[0], conditioning_batch[0], mask_batch[0])

    print(f"Params: {param_count(params):,}")

    # Training config and loop

    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=learning_rate, warmup_steps=warmup_steps, decay_steps=n_train_steps)
    opt = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    # Init state store
    store = StateStore(params, opt.init(params), rng, 0)
    pstore = replicate(store)

    vals = []
    with trange(n_train_steps) as t:
        for i in t:
            pstore, val = train_step(pstore, loss_vdm, vdm, next(batches), opt)
            v = unreplicate(val)
            t.set_postfix(val=v)
            vals.append(v)

            # Save checkpoint periodically
            if (i % save_every == 0) and (i != 0) and (jax.process_index() == 0):
                ckpt = unreplicate(pstore)
                checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, step=i, overwrite=True, keep=np.inf)

    return unreplicate(pstore)


if __name__ == "__main__":
    train()
