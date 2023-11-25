import sys

sys.path.append("./inference/")

import yaml
from pathlib import Path
import argparse

import jax
import jax.numpy as np
from ml_collections.config_dict import ConfigDict

from models.diffusion import VariationalDiffusionModel
from inference.likelihood import likelihood
from tqdm import tqdm
from datasets import load_data
from models.train_utils import create_input_iter


def get_profiles(run_name, n_steps, n_elbo_samples, n_test, seed):
    path_to_model = Path(
        f"/n/holystore01/LABS/iaifi_lab/Lab/set-diffuser-checkpoints/cosmology/{run_name}"
    )
    path_to_profiles = Path(
        f"/n/holystore01/LABS/iaifi_lab/Lab/set-diffuser-checkpoints/cosmology/{run_name}/ll_profiles/"
    )
    path_to_profiles.mkdir(exist_ok=True)
    config_file = path_to_model / "config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        config = ConfigDict(config)

    train_ds, _ = load_data(
        config.data.dataset,
        config.data.n_features,
        config.data.n_particles,
        32,
        config.seed,
        shuffle=True,
        split="test",
    )

    batches = create_input_iter(train_ds)
    x, conditioning, mask = next(batches)
    x = x.reshape(-1, config.data.n_particles, config.data.n_features)
    conditioning = conditioning.reshape(-1, 2)
    mask = mask.reshape(-1, config.data.n_particles)

    rng = jax.random.PRNGKey(seed)
    rng, spl = jax.random.split(rng)

    vdm, restored_params = VariationalDiffusionModel.from_path_to_model(
        path_to_model=path_to_model
    )

    sigma_8_ary = np.linspace(0.6, 1.0, 30)
    omega_m_ary = np.linspace(0.1, 0.5, 30)

    # Get Omega_m
    log_like_cov = []
    for idx in tqdm(range(n_test)):
        log_like = []
        x_test = x[idx]
        for omega_m in omega_m_ary:
            theta_test = np.array([omega_m, conditioning[idx][1]])
            log_like.append(
                likelihood(
                    vdm,
                    rng,
                    restored_params,
                    x_test,
                    theta_test,
                    steps=n_steps,
                    n_samples=n_elbo_samples,
                )
            )
        log_like_cov.append(log_like)
    log_like_cov = np.array(log_like_cov)

    # Get sigma_8
    log_like_cov_s8 = []
    for idx in tqdm(range(n_test)):
        log_like = []
        x_test = x[idx]
        for sigma_8 in sigma_8_ary:
            theta_test = np.array([conditioning[idx][0], sigma_8])
            log_like.append(
                likelihood(
                    vdm,
                    rng,
                    restored_params,
                    x_test,
                    theta_test,
                    steps=n_steps,
                    n_samples=n_elbo_samples,
                )
            )
        log_like_cov_s8.append(log_like)
    log_like_cov_s8 = np.array(log_like_cov_s8)

    np.savez(
        path_to_profiles / f"log_like_cov_v2_{seed}.npz",
        log_like_cov=log_like_cov,
        log_like_cov_s8=log_like_cov_s8,
        omega_m_ary=omega_m_ary,
        sigma_8_ary=sigma_8_ary,
        conditioning=conditioning,
    )


if __name__ == "__main__":
    print("{} devices visible".format(jax.device_count()))

    # Read from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="gallant-cherry-87")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--n_elbo_samples", type=int, default=16)
    parser.add_argument("--n_test", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    get_profiles(
        run_name=args.run_name,
        n_steps=args.n_steps,
        n_elbo_samples=args.n_elbo_samples,
        n_test=args.n_test,
        seed=args.seed,
    )
