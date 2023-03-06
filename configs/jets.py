import ml_collections


def get_config():

    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "set-diffusion"
    wandb.group = "jets"
    wandb.job_type = "training"
    wandb.name = None
    wandb.log_train = True

    # Vartiational diffusion model
    config.vdm = vdm = ml_collections.ConfigDict()
    vdm.gamma_min = -8.0
    vdm.gamma_max = 14.0
    vdm.noise_schedule = "learned_linear"
    vdm.noise_scale = 1e-3
    vdm.timesteps = 0  # 0 for continuous-time VLB
    vdm.embed_context = True
    vdm.n_classes = 3

    # Encoder and decoder specification
    config.encdec = encdec = ml_collections.ConfigDict()
    encdec.use_encdec = False
    encdec.d_hidden = 256
    encdec.n_layers = 4
    encdec.d_embedding = 12

    # Transformer score model
    config.transformer = transformer = ml_collections.ConfigDict()
    transformer.induced_attention = False
    transformer.n_inducing_points = 500
    transformer.d_model = 256
    transformer.d_mlp = 1024
    transformer.n_transformer_layers = 6
    transformer.n_heads = 4

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 128  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 3_000_001
    training.warmup_steps = 10_000
    training.log_every_steps = 100
    training.save_every_steps = 100_000

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "jetnet"
    data.n_particles = 150  # Select the first n_particles particles
    data.n_features = 3  # Select the first n_features features
    data.kwargs = {"jet_type": ["q", "g", "t"]}

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 5e-4
    optim.weight_decay = 1e-4

    config.seed = 42

    return config
