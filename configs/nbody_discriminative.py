import dataclasses
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "set-diffusion"
    wandb.group = "cosmology-augmentations-guidance-discriminative"
    wandb.job_type = "training"
    wandb.base_run_group = "cosmology-augmentations-guidance"
    wandb.base_run_name = "glowing-rain-139"
    wandb.name = None
    wandb.log_train = True

    # Likelihood
    config.likelihood = likelihood = ml_collections.ConfigDict()
    likelihood.n_steps = 10
    likelihood.n_samples = 1

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 2  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 50_001
    training.warmup_steps = 10
    training.log_every_steps = 100
    training.eval_every_steps = 2_000  # training.n_train_steps + 1  # Turn off eval for now
    training.save_every_steps = 2_000

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 3e-4
    optim.weight_decay = 1e-5

    config.seed = 42

    return config
