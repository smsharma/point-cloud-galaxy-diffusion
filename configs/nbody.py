import ml_collections


def get_config():

    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "nbody"
    wandb.job_type = "training"
    wandb.name = None

    # Vartiational diffusion model
    config.vdm = vdm = ml_collections.ConfigDict()
    vdm.timesteps = 1000
    vdm.d_hidden_encoding = 512
    vdm.n_encoder_layers = 5
    vdm.d_embedding = 10
    vdm.embed_context = False

    # Transformer score model
    config.transformer = transformer = ml_collections.ConfigDict()
    transformer.induced_attention = False
    transformer.n_inducing_points = 500
    transformer.d_model = 256
    transformer.d_mlp = 1024
    transformer.n_transformer_layers = 8
    transformer.n_heads = 4

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 16  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 500_000
    training.warmup_steps = 5000
    training.log_every_steps = 100

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "nbody"
    data.n_particles = 5000  # Select the first n_particles particles
    data.n_features = 3  # Select the first n_features features

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 6e-4
    optim.weight_decay = 1e-4

    config.seed = 42

    return config
