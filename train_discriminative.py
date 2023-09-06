import sys
import os
import yaml

from absl import flags, logging
from absl import logging
import ml_collections
from ml_collections import config_flags
from ml_collections.config_dict import ConfigDict
from clu import metric_writers
import wandb

sys.path.append("./")
sys.path.append("../")

from tqdm import trange

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state

import tensorflow as tf

from eval import eval_generation
from models.diffusion import VariationalDiffusionModel
from models.diffusion_utils import loss_vdm
from models.train_utils import (
    create_input_iter,
    param_count,
    to_wandb_config,
)

from models.discriminative import train_step

from datasets import load_data, augment_data

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)


def train(config: ml_collections.ConfigDict, workdir: str = "./logging/") -> train_state.TrainState:
    # Make a copy of the original workdir
    workdir_og = workdir

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
        wandb.define_metric("*", step_metric="train/step")  # Set default x-axis as 'train/step'
        workdir = os.path.join(workdir, run.group, run.name)

        # Recursively create workdir
        os.makedirs(workdir, exist_ok=True)

    # Dump a yaml config file in the output directory
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)

    # Get base model config
    config_base_file = "{}/{}/{}/config.yaml".format(workdir_og, config.wandb.base_run_group, config.wandb.base_run_name)
    with open(config_base_file, "r") as file:
        config_base = yaml.safe_load(file)

    config_base = ConfigDict(config_base)

    # Load the dataset
    train_ds, norm_dict = load_data(
        config_base.data.dataset,
        config_base.data.n_features,
        config_base.data.n_particles,
        config.training.batch_size,
        config.seed,
        shuffle=True,
        split="train",
        # **config_base.data.kwargs,
    )

    add_augmentations = True if config_base.data.add_rotations or config_base.data.add_translations else False

    batches = create_input_iter(train_ds)

    logging.info("Loaded the %s dataset", config_base.data.dataset)

    ## Model configuration
    # Instantiate base model

    # Score and (optional) encoder model configs
    score_dict = FrozenDict(config_base.score)
    encoder_dict = FrozenDict(config_base.encoder)
    decoder_dict = FrozenDict(config_base.decoder)

    # Diffusion model
    x_mean = tuple(map(float, norm_dict["mean"]))
    x_std = tuple(map(float, norm_dict["std"]))
    norm_dict_input = FrozenDict(
        {
            "x_mean": x_mean,
            "x_std": x_std,
            "box_size": config_base.data.box_size,
        }
    )
    vdm = VariationalDiffusionModel(
        d_feature=config_base.data.n_features,
        timesteps=config_base.vdm.timesteps,
        noise_schedule=config_base.vdm.noise_schedule,
        noise_scale=config_base.vdm.noise_scale,
        d_t_embedding=config_base.vdm.d_t_embedding,
        gamma_min=config_base.vdm.gamma_min,
        gamma_max=config_base.vdm.gamma_max,
        score=config_base.score.score,
        score_dict=score_dict,
        embed_context=config_base.vdm.embed_context,
        d_context_embedding=config_base.vdm.d_context_embedding,
        n_classes=config_base.vdm.n_classes,
        use_encdec=config_base.vdm.use_encdec,
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
        norm_dict=norm_dict_input,
    )

    rng = jax.random.PRNGKey(config.seed)
    rng, rng_params = jax.random.split(rng)

    # Pass a test batch through to initialize model
    # TODO: Make so we don't have to pass an entire batch (slow)
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

    # Restore from checkpoint
    ckpt_dir = "{}/{}/{}/".format(workdir_og, config.wandb.base_run_group, config.wandb.base_run_name)  # Load SLURM run
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)

    if state is restored_state:
        raise FileNotFoundError(f"Did not load checkpoint correctly")

    pstate = replicate(restored_state)

    logging.info("Restored checkpoint. Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:
            rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            train_step_rng = np.asarray(train_step_rng)
            x, conditioning, mask = next(batches)
            if add_augmentations:
                x, conditioning, mask = augment_data(
                    x=x,
                    mask=mask,
                    conditioning=conditioning,
                    rng=rng,
                    norm_dict=norm_dict,
                    n_pos_dim=config_base.data.n_pos_features,
                    box_size=config_base.data.box_size,
                    rotations=config_base.data.add_rotations,
                    translations=config_base.data.add_translations,
                )
            pstate, metrics = train_step(pstate, (x, conditioning, mask), train_step_rng, vdm, config.likelihood.n_samples, config.likelihood.n_steps)
            steps.set_postfix(val=unreplicate(metrics["loss"]))
            train_metrics.append(metrics)

            # Log periodically
            if (step % config.training.log_every_steps == 0) and (step != 0) and (jax.process_index() == 0):
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {f"train/{k}": v for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()}

                writer.write_scalars(step, summary)
                train_metrics = []

                if config.wandb.log_train:
                    wandb.log({"train/step": step, **summary})

            # Eval periodically
            if (step % config.training.eval_every_steps == 0) and (step != 0) and (jax.process_index() == 0) and (config.wandb.log_train):
                eval_generation(
                    vdm=vdm,
                    pstate=unreplicate(pstate),
                    rng=rng,
                    n_samples=config.training.batch_size,
                    n_particles=x_batch.shape[-2],  # config.data.n_particles,
                    true_samples=x_batch.reshape((-1, *x_batch.shape[2:])),
                    conditioning=conditioning_batch.reshape((-1, *conditioning_batch.shape[2:])),
                    mask=mask_batch.reshape((-1, *mask_batch.shape[2:])),
                    norm_dict=norm_dict,
                    steps=500,
                    boxsize=config_base.data.box_size,
                )

            # Save checkpoints periodically
            if (step % config.training.save_every_steps == 0) and (step != 0) and (jax.process_index() == 0):
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
