import sys

sys.path.append("./inference/")

import time

import yaml
import pickle
from pathlib import Path
import jax.numpy as np

import optax
import jax
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide
from ml_collections.config_dict import ConfigDict

from models.diffusion import VariationalDiffusionModel
from datasets import get_nbody_data
from inference.inference_utils import get_model

if __name__ == "__main__":
    use_test_set = True
    generated_samples = False
    conditioning_parameters = ["Omega_m", "Omega_b", "h", "sigma_8", "n_s"]
    split = "test" if use_test_set else "train"
    print("{} devices visible".format(jax.device_count()))
    # run_name = "gallant-cherry-87"
    run_name = "cool-terrain-168"  # All params
    # run_name = "leafy-pyramid-88"
    path_to_model = Path(
        f"/n/holystore01/LABS/iaifi_lab/Lab/set-diffuser-checkpoints/cosmology/{run_name}"
    )
    path_to_posteriors = Path(
        f"/n/holystore01/LABS/iaifi_lab/Lab/set-diffuser-checkpoints/cosmology/{run_name}/posteriors/"
    )
    path_to_posteriors.mkdir(exist_ok=True)
    config_file = path_to_model / "config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        config = ConfigDict(config)

    # Steps for diffusion, number ELBO evaluation to mean over, num_particles (4), n_steps for svi optimization
    # ELBO mean
    n_diffusion_steps = 10
    n_steps = 2000
    num_samples = 100_000
    num_particles = 1
    lr = 5e-3

    min_fit_idx = 0
    max_fit_idx = 30

    x, _, conditioning, norm_dict = get_nbody_data(
        n_features=config.data.n_features,
        n_particles=config.data.n_particles,
        split=split,
        conditioning_parameters=conditioning_parameters,
    )

    if generated_samples:
        x = np.load(
            path_to_posteriors.parent.parent
            / f"samples/{run_name}/generated_test_samples_500_steps.npy"
        )
        x = x.reshape((-1, config.data.n_particles, 3))
        x = (x - norm_dict["mean"]) / norm_dict["std"]

    test_idx = np.array(range(min_fit_idx, max_fit_idx))
    x = x[test_idx]

    rng = jax.random.PRNGKey(42)
    rng, spl = jax.random.split(rng)

    vdm, restored_params = VariationalDiffusionModel.from_path_to_model(
        path_to_model=path_to_model
    )

    # NumPyro model and guide
    model = get_model(
        vdm=vdm,
        restored_state_params=jax.tree_map(np.array, restored_params),
        rng=rng,
        n_samples=1,
        steps=n_diffusion_steps,
    )
    # guide = autoguide.AutoIAFNormal(
    #     model,
    #     num_flows=4,
    #     hidden_dims=[64, 64],
    #     skip_connections=True,
    #     nonlinearity=jax.example_libraries.stax.Tanh,
    # )
    guide = autoguide.AutoMultivariateNormal(model)

    optimizer = optim.optax_to_numpyro(optax.adam(lr))
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=num_particles))

    for i, x_test in enumerate(x):
        print("Starting :)")
        t0 = time.time()
        idx = test_idx[i]
        svi_results = svi.run(rng, n_steps, x_test)
        rng, _ = jax.random.split(rng)
        posterior_dict = guide.sample_posterior(
            rng_key=rng, params=svi_results.params, sample_shape=(num_samples,)
        )
        if generated_samples:
            filename = f"generated_chain_{split}_{idx}_steps10.pkl"
        else:
            filename = f"chain_mvn_{split}_{idx}_steps10.pkl"
        with open(path_to_posteriors / filename, "wb") as f:
            pickle.dump(posterior_dict, f)
        print(f"Finished chain {idx} in {time.time() - t0:.2f} seconds")
