import dataclasses
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
    vdm.gamma_min = -8.0
    vdm.gamma_max = 14.0
    vdm.noise_schedule = "learned_linear"
    vdm.noise_scale = 1e-3
    vdm.timesteps = 0  # 0 for continuous-time VLB
    vdm.embed_context = False
    vdm.d_context_embedding = 32
    vdm.n_classes = 0
    vdm.use_encdec = False

    # Encoder and decoder specification
    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.d_hidden = 256
    encoder.n_layers = 4
    encoder.d_embedding = 12

    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.d_hidden = 256
    decoder.n_layers = 4

    # # Transformer score model
    # config.score = score = ml_collections.ConfigDict()
    # score.score = "transformer"
    # score.induced_attention = False
    # score.n_inducing_points = 200
    # score.d_model = 128
    # score.d_mlp = 512
    # score.n_layers = 5
    # score.n_heads = 2

    # # Graph score model
    # config.score = score = ml_collections.ConfigDict()
    # score.score = "graph"
    # score.k = 20
    # score.n_pos_features = 3
    # score.num_mlp_layers = 4
    # score.latent_size = 64
    # score.skip_connections = True
    # score.message_passing_steps = 4

    # Equivariant score model
    config.score = score = ml_collections.ConfigDict()
    score.score = "equivariant"
    score.k = 20
    score.n_pos_features = 3
    score.d_hidden = 16
    score.n_layers = 2

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 16  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 501_000
    training.warmup_steps = 5_000
    training.log_every_steps = 100
    training.eval_every_steps = 1000
    training.save_every_steps = 20_000

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "nbody"
    data.n_particles = 5000  # Select the first n_particles particles
    data.n_features = 7  # Select the first n_features features
    data.n_pos_features = 3  # Select the first n_pos_features features as coordinates (e.g., for graph-building)
    data.kwargs = {}

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 6e-4
    optim.weight_decay = 1e-4

    config.seed = 44

    return config
