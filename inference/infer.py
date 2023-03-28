import sys

sys.path.append("../")
from functools import partial
import jax
import jax.numpy as np
from flax.core import FrozenDict
from ml_collections.config_dict import ConfigDict
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import numpy as vnp
import yaml
import optax
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide

from datasets import load_data, get_nbody_data
from models.diffusion import VariationalDiffusionModel
from models.train_utils import create_input_iter, param_count, StateStore, train_step


def elbo(vdm, params, rng, x, conditioning, mask, steps=10, unroll_loop=True):
    rng, spl = jax.random.split(rng)
    cond = vdm.apply(params, conditioning, method=vdm.embed)
    f = vdm.apply(params, x, conditioning, method=vdm.encode)
    loss_recon = vdm.apply(
        params, x, f, conditioning, rngs={"sample": rng}, method=vdm.recon_loss
    )
    loss_klz = vdm.apply(params, f, method=vdm.latent_loss)
    if not unroll_loop:

        def body_fun(i, val):
            loss, rng = val
            rng, spl = jax.random.split(rng)
            new_loss = vdm.apply(
                params,
                np.array([i / steps]),
                f,
                cond,
                mask,
                rngs={"sample": spl},
                method=vdm.diffusion_loss,
            )
            return (loss + (new_loss * mask[..., None]).sum((-1, -2)) / steps, rng)

        loss_diff, rng = jax.lax.fori_loop(
            0, steps, body_fun, (np.zeros(x.shape[0]), rng)
        )
    else:
        loss_diff, rng = (np.zeros(x.shape[0]), rng)
        for i in range(steps):
            rng, spl = jax.random.split(rng)
            new_loss = vdm.apply(
                params,
                np.array([i / steps]),
                f,
                cond,
                mask,
                rngs={"sample": spl},
                method=vdm.diffusion_loss,
            )
            loss_diff = loss_diff + (new_loss * mask[..., None]).sum((-1, -2)) / steps
    return (
        (loss_recon * mask[..., None]).sum((-1, -2))
        + (loss_klz * mask[..., None]).sum((-1, -2))
        + loss_diff
    )


def prior_cube(u):
    priors = np.array([[0.1, 0.5], [0.6, 1.0]])
    priors_lo = priors[:, 0]
    priors_interval = priors[:, 1] - priors[:, 0]
    for i in range(len(u) - 1):
        u[i] = u[i] * priors_interval[i] + priors_lo[i]
    return u


def log_prior(theta):
    Omega_m, sigma_8 = theta
    if 0.1 < Omega_m < 0.5 and 0.6 < sigma_8 < 1.0:
        return 0.0
    return -np.inf


#@partial(
#    jax.jit,
#    static_argnums=(
#        0,
#        1,
#        2,
#    ),
#)
def likelihood(vdm, rng, restored_state, x_test, params, n_samples=2):
    x_test = np.repeat(np.array([x_test]), n_samples, 0)
    theta_test = np.repeat(np.array([params]), n_samples, 0)
    return -elbo(
        vdm,
        restored_state.params,
        rng,
        x_test,
        theta_test,
        np.ones_like(x_test[..., 0]),
    ).mean()


def get_model(
    vdm,
    restored_state,
    rng,
):
    def model(x_test, n_samples=2):
        # Omega_m and sigma_8 prior distributions
        params = numpyro.sample(
            "params", dist.Uniform(np.array([0.1, 0.6]), np.array([0.5, 1.0]))
        )
        log_like = likelihood(
            vdm=vdm,
            rng=rng,
            restored_state=restored_state,
            x_test=x_test,
            params=params,
            n_samples=n_samples,
        )
        return numpyro.factor("log_like", log_like)

    return model


def load_data_for_inference(
    config,
):
    x, _, _, _ = get_nbody_data(
        n_features=config.data.n_features,
        n_particles=config.data.n_particles,
    )
    return x


def load_diffusion_model(config):
    train_ds, _ = load_data(
        config.data.dataset,
        config.data.n_features,
        config.data.n_particles,
        4,
        config.seed,
        **config.data.kwargs,
    )
    batches = create_input_iter(train_ds)
    # Score and (optional) encoder model configs
    score_dict = FrozenDict(config.score)
    encoder_dict = FrozenDict(config.encoder)
    decoder_dict = FrozenDict(config.decoder)

    # Diffusion model
    vdm = VariationalDiffusionModel(
        d_feature=config.data.n_features,
        timesteps=config.vdm.timesteps,
        noise_schedule=config.vdm.noise_schedule,
        noise_scale=config.vdm.noise_scale,
        gamma_min=config.vdm.gamma_min,
        gamma_max=config.vdm.gamma_max,
        score=config.score.score,
        score_dict=score_dict,
        embed_context=config.vdm.embed_context,
        d_context_embedding=config.vdm.d_context_embedding,
        n_classes=config.vdm.n_classes,
        use_encdec=config.vdm.use_encdec,
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
    )

    # Pass a test batch through to initialize model
    x_batch, conditioning_batch, mask_batch = next(batches)
    rng = jax.random.PRNGKey(42)
    _, params = vdm.init_with_output(
        {"sample": rng, "params": rng},
        x_batch[0],
        conditioning_batch[0],
        mask_batch[0],
    )

    print(f"Params: {param_count(params):,}")

    # Training config and state
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.n_train_steps,
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)
    state = train_state.TrainState.create(apply_fn=vdm.apply, params=params, tx=tx)
    ckpt_dir = "{}/{}/".format(logging_dir, run_name)  # Load SLURM run
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)

    if state is restored_state:
        raise FileNotFoundError(f"Did not load checkpoint correctly")
    return vdm, restored_state, rng


if __name__ == "__main__":
    print("{} devices visible".format(jax.device_count()))
    logging_dir = "/n/holyscratch01/iaifi_lab/ccuesta/checkpoints/"
    run_name = "warm-flower-77"  # wandb run name
    config_file = "{}/{}/config.yaml".format(logging_dir, run_name)
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        config = ConfigDict(config)
    n_steps = 1000
    lr = 1e-2

    fit_idx = 421
    x = load_data_for_inference(config)
    x_test = x[fit_idx]
    vdm, restored_state, rng = load_diffusion_model(config)
    model = get_model(
        vdm=vdm,
        restored_state=restored_state,
        rng=rng,
    )
    guide = autoguide.AutoMultivariateNormal(model)
    optimizer = optim.optax_to_numpyro(optax.sgd(lr))
    svi = SVI(
        model,
        guide,
        optimizer,
        Trace_ELBO(num_particles=1),
    )
    svi_results = svi.run(rng, n_steps, x_test)

    num_samples = 10_000

    rng, _ = jax.random.split(rng)
    posterior_dict = guide.sample_posterior(
        rng_key=rng, params=svi_results.params, sample_shape=(num_samples,)
    )
    print('Got posterior!')
    print(posterior_dict)
    post = vnp.array(posterior_dict["params"])
    print('post array')
    print(post)
    



