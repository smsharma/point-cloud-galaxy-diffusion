import ml_collections


def get_config():

    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "set-diffusion"
    wandb.group = "cosmology"
    wandb.job_type = "training"
    wandb.name = None
    wandb.log_train = False

    # Vartiational diffusion model
    config.vdm = vdm = ml_collections.ConfigDict()
    vdm.timesteps = 0
    vdm.d_hidden_encoding = 256
    vdm.n_encoder_layers = 4
    vdm.d_embedding = 10
    vdm.embed_context = False
    vdm.n_classes = 0
    vdm.use_encdec = False

    # # Transformer score model
    # config.score = score = ml_collections.ConfigDict()
    # score.score = "transformer"
    # score.induced_attention = False
    # score.n_inducing_points = 200
    # score.d_model = 128
    # score.d_mlp = 512
    # score.n_layers = 5
    # score.n_heads = 2

    # Graph score model
    config.score = score = ml_collections.ConfigDict()
    score.score = "graph"
    score.k = 20
    score.num_mlp_layers = 4
    score.latent_size = 64
    score.skip_connections = True
    score.message_passing_steps = 4

    # # Equivariant score model
    # config.score = score = ml_collections.ConfigDict()
    # score.score = "equivariant"
    # score.k = 20

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 16  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 301_000
    training.warmup_steps = 5_000
    training.log_every_steps = 100
    training.eval_every_steps = 2
    training.save_every_steps = 20_000

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "nbody"
    data.n_particles = 5000  # Select the first n_particles particles
    data.n_features = 7  # Select the first n_features features
    data.kwargs = {}

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 6e-4
    optim.weight_decay = 1e-4

    config.seed = 44

    return config
