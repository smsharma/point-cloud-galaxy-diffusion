import dataclasses
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "set-diffusion"
    wandb.group = "cosmology-augmentations-guidance"
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
    vdm.d_context_embedding = 16
    vdm.d_t_embedding = 16  # Timestep embedding dimension
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
    # score.d_model = 256
    # score.d_mlp = 1024
    # score.n_layers = 6
    # score.n_heads = 4
    # score.concat_conditioning = False
    # score.d_conditioning = 256

    # Graph score model
    config.score = score = ml_collections.ConfigDict()
    score.score = "graph"
    score.k = 50
    score.n_pos_features = 3
    score.num_mlp_layers = 4
    score.latent_size = 32
    score.hidden_size = 128
    score.skip_connections = True
    score.message_passing_steps = 4
    score.attention = True
    score.shared_weights = False  # GNN shares weights across message passing steps; Doesn't work
    score.use_edges = True

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 32  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 301_000
    training.warmup_steps = 5_000
    training.log_every_steps = 100
    training.eval_every_steps = 2_000  # training.n_train_steps + 1  # Turn off eval for now
    training.save_every_steps = 20_000
    training.unconditional_dropout = False  # Set to True to use unconditional dropout (randomly zero out conditioning vectors)
    training.p_uncond = 0.2  # Fraction of conditioning vectors to zero out if unconditional_dropout is True

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "nbody"
    data.n_particles = 5000  # Select the first n_particles particles
    data.n_features = 3  # Select the first n_features features
    data.n_pos_features = 3  # Select the first n_pos_features features as coordinates (e.g., for graph-building)
    data.box_size = 1000.0  # Need to know the box size for augmentations
    data.add_augmentations = True
    data.add_rotations = True
    data.add_translations = True
    data.kwargs = {}

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 3e-4
    optim.weight_decay = 1e-5

    config.seed = 42

    return config
