import sys
import os
import shutil

from absl import flags, logging
from absl import logging
from ml_collections import config_flags

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

from datasets import load_data

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)


def train(config):

    ckpt_dir = "/n/holystore01/LABS/iaifi_lab/Users/smsharma/set-diffuser/notebooks/ckpts_debug_inducing/"

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    # Load the dataset
    train_ds = load_data(config.data.dataset, config.data.n_features, config.data.n_particles, config.training.batch_size, config.seed)

    ## Model configuration

    # Transformer score model
    transformer_dict = FrozenDict({"d_model": config.transformer.d_model, "d_mlp": config.transformer.d_mlp, "n_layers": config.transformer.n_transformer_layers, "n_heads": config.transformer.n_heads, "induced_attention": config.transformer.induced_attention, "n_inducing_points": config.transformer.n_inducing_points})

    # Diffusion model
    vdm = VariationalDiffusionModel(n_layers=config.vdm.n_encoder_layers, d_embedding=config.vdm.d_embedding, d_hidden_encoding=config.vdm.d_hidden_encoding, timesteps=config.vdm.timesteps, d_feature=config.data.n_features, transformer_dict=transformer_dict, embed_context=config.vdm.embed_context)
    batches = create_input_iter(train_ds)

    rng = jax.random.PRNGKey(config.seed)

    # Pass a test batch through to initialize model
    x_batch, conditioning_batch, mask_batch = next(batches)
    _, params = vdm.init_with_output({"sample": rng, "params": rng}, x_batch[0], conditioning_batch[0], mask_batch[0])

    logging.info("Number of parameters: %d", param_count(params))

    # Training config and loop

    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=config.optim.learning_rate, warmup_steps=config.training.warmup_steps, decay_steps=config.training.n_train_steps)
    opt = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)

    # Init state store
    store = StateStore(params, opt.init(params), rng, 0)
    pstore = replicate(store)

    vals = []
    with trange(config.training.n_train_steps) as t:
        for i in t:
            pstore, val = train_step(pstore, loss_vdm, vdm, next(batches), opt)
            v = unreplicate(val)
            t.set_postfix(val=v)
            vals.append(v)

            # Save checkpoint periodically
            if (i % config.training.log_every_steps == 0) and (i != 0) and (jax.process_index() == 0):
                ckpt = unreplicate(pstore)
                checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, step=i, overwrite=True, keep=5)

    return unreplicate(pstore)


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "File path to the training or sampling hyperparameter configuration.", lock_config=True)
    FLAGS(sys.argv)  # Parse flags

    # Ensure TF does not see GPU and grab all GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("JAX total visible devices: %r", jax.device_count())

    train(FLAGS.config)
