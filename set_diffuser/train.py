import sys
import os
import yaml

from absl import flags, logging
from absl import logging
import ml_collections
from ml_collections import config_flags
from clu import metric_writers
import wandb

from tqdm import trange

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state

import tensorflow as tf

from set_diffuser.models.diffusion import VariationalDiffusionModel
from set_diffuser.models.diffusion_utils import loss_vdm
from set_diffuser.models.train_utils import (
    create_input_iter,
    param_count,
    train_step,
    to_wandb_config,
)

from set_diffuser.data import load_data

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)


def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
) -> train_state.TrainState:

    # Set up wandb run
    if config.wandb.log_train and jax.process_index() == 0:
        wandb_config = to_wandb_config(config)
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            job_type=config.wandb.job_type,
            group=config.wandb.group,
            config=wandb_config,
        )
        wandb.define_metric(
            "*", step_metric="train/step"
        )  # Set default x-axis as 'train/step'
        workdir = os.path.join(workdir, run.group, run.name)

        # Recursively create workdir
        os.makedirs(workdir, exist_ok=True)

    # Dump a yaml config file
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    # Load the dataset
    train_ds, x_standarization_dict = load_data(
        config.data.dataset,
        config.data.n_features,
        config.data.n_particles,
        config.training.batch_size,
        config.seed,
        **config.data.kwargs,
    )

    logging.info("Loaded the %s dataset", config.data.dataset)

    ## Model configuration

    # Transformer score model
    transformer_dict = FrozenDict(
        {
            "d_model": config.transformer.d_model,
            "d_mlp": config.transformer.d_mlp,
            "n_layers": config.transformer.n_transformer_layers,
            "n_heads": config.transformer.n_heads,
            "induced_attention": config.transformer.induced_attention,
            "n_inducing_points": config.transformer.n_inducing_points,
        }
    )

    # Diffusion model
    vdm = VariationalDiffusionModel(
        n_layers=config.vdm.n_encoder_layers,
        d_embedding=config.vdm.d_embedding,
        d_hidden_encoding=config.vdm.d_hidden_encoding,
        timesteps=config.vdm.timesteps,
        d_feature=config.data.n_features,
        transformer_dict=transformer_dict,
        embed_context=config.vdm.embed_context,
        n_classes=config.vdm.n_classes,
        x_standarization_dict=x_standarization_dict,
    )

    batches = create_input_iter(train_ds)

    rng = jax.random.PRNGKey(config.seed)
    rng, rng_params = jax.random.split(rng)

    # Pass a test batch through to initialize model
    x_batch, conditioning_batch, mask_batch = next(batches)
    _, params = vdm.init_with_output(
        {"sample": rng, "params": rng_params},
        x_batch[0],
        conditioning_batch[0],
        mask_batch[0],
    )

    logging.info("Instantiated the model")
    logging.info("Number of parameters: %d", param_count(params))

    ## Training config and loop

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.n_train_steps,
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)

    state = train_state.TrainState.create(apply_fn=vdm.apply, params=params, tx=tx)
    pstate = replicate(state)

    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:

            rng, *train_step_rng = jax.random.split(
                rng, num=jax.local_device_count() + 1
            )
            train_step_rng = np.asarray(train_step_rng)

            pstate, metrics = train_step(
                pstate, next(batches), train_step_rng, vdm, loss_vdm
            )
            steps.set_postfix(val=unreplicate(metrics["loss"]))
            train_metrics.append(metrics)

            # Log periodically
            if (
                (step % config.training.log_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
            ):

                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train/{k}": v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }

                writer.write_scalars(step, summary)
                train_metrics = []

                if config.wandb.log_train:
                    wandb.log({"train/step": step, **summary})

            # Save checkpoints periodically
            if (
                (step % config.training.save_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
            ):
                state_ckpt = unreplicate(pstate)
                checkpoints.save_checkpoint(
                    ckpt_dir=workdir,
                    target=state_ckpt,
                    step=step,
                    overwrite=True,
                    keep=np.inf,
                )

    logging.info("All done! Have a great day.")

    return unreplicate(pstate)


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)  # Parse flags

    # Ensure TF does not see GPU and grab all GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("JAX total visible devices: %r", jax.device_count())

    train(FLAGS.config)
