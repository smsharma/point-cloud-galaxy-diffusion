import sys
import jax
from pathlib import Path
import yaml

sys.path.append("./")
sys.path.append("../")

from typing import List, Dict

from jax import random
import jax.numpy as np
import numpy as onp
from models.diffusion import VariationalDiffusionModel


import wandb
import matplotlib.pyplot as plt
from ml_collections.config_dict import ConfigDict
from pycorr import TwoPointCorrelationFunction
from models.diffusion_utils import generate
from models.train_utils import create_input_iter
from models.likelihood import elbo
from datasets import nbody_dataset
from cosmo_utils.knn import get_CDFkNN
from scipy.interpolate import interp1d
from scipy.stats import chi2

import time
from tqdm import tqdm

colors = [
    "lightseagreen",
    "mediumorchid",
    "salmon",
    "royalblue",
    "rosybrown",
]


def plot_pointclouds_3D(generated_samples: np.array, true_samples: np.array, idx_to_plot: int = 0) -> plt.figure:
    """Plot pointcloud in three dimensions

    Args:
        generated_samples (np.array): samples generated by the model
        true_samples (np.array): true samples
        idx_to_plot (int, optional): idx to plot. Defaults to 0.

    Returns:
        plt.figure: figure
    """
    s = 4
    alpha = 0.5
    color = "firebrick"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), subplot_kw={"projection": "3d"})
    ax1.scatter(
        generated_samples[idx_to_plot, :, 0],
        generated_samples[idx_to_plot, :, 1],
        generated_samples[idx_to_plot, :, 2],
        alpha=alpha,
        s=s,
        color=color,
    )
    ax1.set_title("Gen")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2.scatter(
        true_samples[idx_to_plot, :, 0],
        true_samples[idx_to_plot, :, 1],
        true_samples[idx_to_plot, :, 2],
        alpha=alpha,
        s=s,
        color=color,
    )
    ax1.set_title("Gen")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    return fig


def plot_pointclouds_2D(generated_samples: np.array, true_samples: np.array, idx_to_plot: int = 0):
    """Plot pointcloud in two dimensions

    Args:
        generated_samples (np.array): samples generated by the model
        true_samples (np.array): true samples
        idx_to_plot (int, optional): idx to plot. Defaults to 0.

    Returns:
        plt.figure: figure
    """
    s = 4
    alpha = 0.5
    color = "firebrick"
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(20, 12),
    )
    ax1.scatter(
        generated_samples[idx_to_plot, :, 0],
        generated_samples[idx_to_plot, :, 1],
        alpha=alpha,
        s=s,
        color=color,
    )
    ax1.set_title("Gen")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.scatter(
        true_samples[idx_to_plot, :, 0],
        true_samples[idx_to_plot, :, 1],
        alpha=alpha,
        s=s,
        color=color,
    )
    ax1.set_title("Gen")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    return fig


def plot_knns(
    generated_samples: np.array,
    true_samples: np.array,
    conditioning: np.array,
    boxsize: float = 1000.0,
    idx_to_plot: List[int] = [0, 1, 2],
) -> plt.figure:
    """plot nearest neighbour statistics

    Args:
        generated_samples (np.array): samples generated by the model
        true_samples (np.array): true samples
        conditioning (np.array): conditioning per sample
        boxsize (float, optional): size of the simulation box. Defaults to 1000.0.
        idx_to_plot (List[int], optional): idx to plot. Defaults to [0, 1, 2].

    Returns:
        plt.figure: figure
    """
    r_bins = np.linspace(0.5, 100.0, 60)
    k_bins = [1, 5, 9]
    key = random.PRNGKey(0)
    random_points = boxsize * random.uniform(
        key,
        shape=(len(true_samples[0]) * 10, 3),
    )
    fig, _ = plt.subplots()
    for i, idx in enumerate(idx_to_plot):
        sampled_knn = get_CDFkNN(
            r_bins=r_bins,
            pos=generated_samples[idx][..., :3],
            random_pos=random_points,
            boxsize=boxsize,
            k=k_bins,
        )
        true_knn = get_CDFkNN(
            r_bins=r_bins,
            pos=true_samples[idx][..., :3],
            random_pos=random_points,
            boxsize=boxsize,
            k=k_bins,
        )

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        for k in range(len(k_bins)):
            c = plt.plot(
                r_bins,
                true_knn[k],
                label=rf"$\Omega_m={conditioning[i][0]:.2f} \,\,\sigma_8={conditioning[i][-1]:.2f}$" if (k == 1 and conditioning is not None) else None,
                ls="-",
                alpha=0.75,
                lw=1,
                color=color,
            )
            plt.plot(
                r_bins,
                sampled_knn[k],
                color=color,
                ls="--",
                alpha=0.75,
                lw=1,
            )
    plt.legend(
        fontsize=12,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
    )
    plt.text(
        25,
        0.5,
        f"k={k_bins[0]}",
        rotation=65,
    )
    plt.text(
        52,
        0.5,
        f"k={k_bins[1]}",
        rotation=65,
    )
    plt.text(
        66,
        0.5,
        f"k={k_bins[2]}",
        rotation=65,
    )
    plt.ylabel("CDF")
    plt.xlabel("r [Mpc/h]")
    return fig


def compute_2pcf(
    sample: np.array,
    boxsize: float,
    r_bins: np.array,
) -> np.array:
    """Get the monopole of the two point correlation function

    Args:
        sample (np.array): positions
        boxsize (float): size of the box
        r_bins (np.array): bins in pair separation

    Returns:
        np.array: monopole of the two point correlation function
    """
    mu_bins = np.linspace(-1, 1, 201)
    return TwoPointCorrelationFunction(
        "smu",
        edges=(onp.array(r_bins), onp.array(mu_bins)),
        data_positions1=onp.array(sample).T,
        engine="corrfunc",
        n_threads=2,
        boxsize=boxsize,
        los="z",
    )(
        ells=[0]
    )[0]


def compute_2pcf_rsd(
    positions: np.array,
    velocities: np.array,
    omega_matter: float,
    boxsize: float,
    r_bins: np.array,
    redshift: float = 0.5,
) -> np.array:
    """Get the monopole of the two point correlation function

    Args:
        sample (np.array): positions
        boxsize (float): size of the box
        r_bins (np.array): bins in pair separation

    Returns:
        np.array: monopole of the two point correlation function
    """
    omega_l = 1 - omega_matter
    H_0 = 100.0
    az = 1 / (1 + redshift)
    Hz = H_0 * np.sqrt(omega_matter * (1 + redshift) ** 3 + omega_l)
    z_rsd = positions[..., -1] + velocities[..., -1] / (Hz * az)
    z_rsd %= boxsize
    rsd_positions = positions.copy()
    rsd_positions[..., -1] = z_rsd
    mu_bins = np.linspace(-1, 1, 201)
    return TwoPointCorrelationFunction(
        "smu",
        edges=(onp.array(r_bins), onp.array(mu_bins)),
        data_positions1=onp.array(rsd_positions).T,
        engine="corrfunc",
        n_threads=2,
        boxsize=boxsize,
        los="z",
    )(ells=[0, 2])


def plot_2pcf(generated_samples: np.array, true_samples: np.array, boxsize: float) -> plt.figure:
    """Plot the two point correlation function

    Args:
        generated_samples (np.array): samples generated by the model
        true_samples (np.array): true samples
        boxsize (float): size of the box

    Returns:
        plt.figure: figure
    """
    generated_2pcfs, true_2pcfs = [], []
    r_bins = np.linspace(0.5, 120.0, 60)
    r = 0.5 * (r_bins[1:] + r_bins[:-1])
    for idx in range(len(generated_samples)):
        generated_2pcfs.append(compute_2pcf(generated_samples[idx][..., :3], boxsize, r_bins))
        true_2pcfs.append(compute_2pcf(true_samples[idx][..., :3], boxsize, r_bins))

    fig, _ = plt.subplots()
    c = plt.plot(r, r**2*onp.mean(true_2pcfs, axis=0), label="N-body")
    plt.plot(
        r,
        r**2*(onp.mean(true_2pcfs, axis=0) - onp.std(true_2pcfs, axis=0)),
        alpha=0.5,
        color=c[0].get_color(),
        linestyle="dashed",
    )
    plt.plot(
        r,
        r**2*(onp.mean(true_2pcfs, axis=0) + onp.std(true_2pcfs, axis=0)),
        alpha=0.5,
        color=c[0].get_color(),
        linestyle="dashed",
    )

    # fill_between somehow doesnt work with wandb :(
    # plt.fill_between(
    #    r,
    #    (onp.mean(true_2pcfs, axis=0) - onp.std(true_2pcfs,axis=0)),
    #    (onp.mean(true_2pcfs, axis=0) + onp.std(true_2pcfs,axis=0)),
    #    alpha=0.5,
    #    color=c[0].get_color(),
    # )
    c = plt.plot(r, r**2*onp.mean(generated_2pcfs, axis=0), label="Diffusion")
    plt.plot(
        r,
        r**2*(onp.mean(generated_2pcfs, axis=0) - onp.std(generated_2pcfs, axis=0)),
        alpha=0.5,
        color=c[0].get_color(),
        linestyle="dashed",
    )
    plt.plot(
        r,
        r**2*(onp.mean(generated_2pcfs, axis=0) + onp.std(generated_2pcfs, axis=0)),
        alpha=0.5,
        color=c[0].get_color(),
        linestyle="dashed",
    )
    # plt.fill_between(
    #    r,
    #    (onp.mean(generated_2pcfs, axis=0) - onp.std(generated_2pcfs,axis=0)),
    #    (onp.mean(generated_2pcfs, axis=0) + onp.std(generated_2pcfs,axis=0)),
    #    alpha=0.5,
    #    color=c[0].get_color(),
    # )
    plt.ylabel("2PCF")
    plt.xlabel("r [Mpc/h]")
    plt.legend(fontsize=8)
    return fig


def plot_velocity_histograms(
    generated_velocities: np.array,
    true_velocities: np.array,
    idx_to_plot: List[int],
) -> plt.figure:
    """plot histograms of velocity modulus

    Args:
        generated_velocities (np.array): generated 3D velociteis
        true_velocities (np.array): true 3D velocities
        idx_to_plot (List[int]): idx to plot

    Returns:
        plt.Figure: figure vel hist
    """
    generated_mod = onp.sqrt(onp.sum(generated_velocities**2, axis=-1))
    true_mod = onp.sqrt(onp.sum(true_velocities**2, axis=-1))
    fig, _ = plt.subplots(figsize=(15, 5))
    offset = 0
    for i, idx in enumerate(idx_to_plot):
        true_hist, bin_edges = np.histogram(
            true_mod[idx],
            bins=50,
        )
        generated_hist, bin_edges = np.histogram(
            generated_mod[idx],
            bins=bin_edges,
        )
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.plot(
            bin_centres + offset,
            true_hist,
            label="N-body" if i == 0 else None,
            color=colors[i],
        )
        plt.plot(
            bin_centres + offset,
            generated_hist,
            label="Diffusion" if i == 0 else None,
            linestyle="dashed",
            color=colors[i],
        )
        offset += onp.max(true_mod)
    plt.legend()
    plt.xlabel("|v| + offset [km/s]")
    plt.ylabel("PDF")
    return fig


def plot_hmf(
    generated_masses: np.array,
    true_masses: np.array,
    idx_to_plot: List[int],
) -> plt.figure:
    """plot halo mass functions

    Args:
        generated_masses (np.array): generated masses
        true_masses (np.array): true masses
        idx_to_plot (List[int]): idx to plot

    Returns:
        plt.Figure: hmf figure
    """
    fig, _ = plt.subplots()
    for i, idx in enumerate(idx_to_plot):
        true_hist, bin_edges = np.histogram(
            true_masses[idx],
            bins=50,
        )
        generated_hist, bin_edges = np.histogram(
            generated_masses[idx],
            bins=bin_edges,
        )
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.semilogy(
            bin_centres,
            true_hist,
            label="N-body" if i == 0 else None,
            color=colors[i],
        )
        plt.semilogy(
            bin_centres,
            generated_hist,
            label="Diffusion" if i == 0 else None,
            color=colors[i],
            linestyle="dashed",
        )

    plt.legend()
    plt.xlabel("log Halo Mass")
    plt.ylabel("PDF")
    return fig


def plot_2pcf_rsd(
    generated_positions: np.array,
    true_positions: np.array,
    generated_velocities: np.array,
    true_velocities: np.array,
    conditioning: np.array,
    boxsize: float,
) -> plt.figure:
    """plot 2pcf in redshift space

    Args:
        generated_positions (np.array): generated 3D positions
        true_positions (np.array): true 3D positions
        generated_velocities (np.array): generated 3D velociteis
        true_velocities (np.array): true 3D velocities
        conditioning (np.array): conditioning (cosmological params)
        boxsize (float): boxsize

    Returns:
        plt.figure: fig with monopole and quadrupole
    """
    generated_2pcfs, true_2pcfs = [], []
    r_bins = np.linspace(0.5, 120.0, 60)
    r = 0.5 * (r_bins[1:] + r_bins[:-1])
    for idx in range(len(generated_positions)):
        generated_2pcfs.append(
            compute_2pcf_rsd(
                positions=generated_positions[idx],
                velocities=generated_velocities[idx],
                omega_matter=conditioning[idx, 0],
                boxsize=boxsize,
                r_bins=r_bins,
            )
        )
        true_2pcfs.append(
            compute_2pcf_rsd(
                positions=true_positions[idx],
                velocities=true_velocities[idx],
                omega_matter=conditioning[idx, 0],
                boxsize=boxsize,
                r_bins=r_bins,
            )
        )
    fig, ax = plt.subplots(nrows=2, figsize=(8, 12))
    true_2pcfs = onp.array(true_2pcfs)
    true_2pcfs[:, 1, ...] = true_2pcfs[:, 1, ...] * r**2
    generated_2pcfs = onp.array(generated_2pcfs)
    generated_2pcfs[:, 1, ...] = generated_2pcfs[:, 1, ...] * r**2
    for i in range(2):
        if i == 0:
            c = ax[i].loglog(r, onp.mean(true_2pcfs, axis=0)[i], label="N-body")
        else:
            c = ax[i].semilogx(r, onp.mean(true_2pcfs, axis=0)[i], label="N-body")
        ax[i].plot(
            r,
            (onp.mean(true_2pcfs, axis=0) - onp.std(true_2pcfs, axis=0))[i],
            color=c[0].get_color(),
            linestyle="dashed",
        )
        ax[i].plot(
            r,
            (onp.mean(true_2pcfs, axis=0) + onp.std(true_2pcfs, axis=0))[i],
            color=c[0].get_color(),
            linestyle="dashed",
        )

        c = ax[i].plot(r, onp.mean(generated_2pcfs, axis=0)[i], label="Diffusion")
        ax[i].plot(
            r,
            (onp.mean(generated_2pcfs, axis=0) - onp.std(generated_2pcfs, axis=0))[i],
            color=c[0].get_color(),
            linestyle="dashed",
        )
        ax[i].plot(
            r,
            (onp.mean(generated_2pcfs, axis=0) + onp.std(generated_2pcfs, axis=0))[i],
            color=c[0].get_color(),
            linestyle="dashed",
        )
    ax[0].set_ylabel("Monopole")
    ax[1].set_ylabel("r^2 Quadrupole")
    plt.xlabel("r [Mpc/h]")
    plt.legend(fontsize=8)
    return fig


def eval_likelihood(
    vdm,
    pstate,
    rng,
    true_samples: np.array,
    conditioning: np.array,
    mask: np.array,
    log_wandb: bool = True,
):
    n_test = 16
    omega_m_ary = np.linspace(0.1, 0.5, 30)

    log_like_cov = []
    for idx in tqdm(range(n_test)):
        log_like = []
        x_test = true_samples[idx][None, ...]
        for omega_m in omega_m_ary:
            theta_test = np.array([omega_m, conditioning[idx][1]])[None, ...]
            log_like.append(elbo(vdm, pstate.params, rng, x_test, theta_test, np.ones_like(x_test[..., 0]), steps=20, unroll_loop=True))
        log_like_cov.append(log_like)
    log_like_cov = np.array(log_like_cov)

    threshold_1sigma = -chi2.isf(1 - 0.68, 1)

    intervals1 = []
    true_values = []

    for ic, idx in enumerate(range(n_test)):
        likelihood_arr = 2 * (np.array(log_like_cov[idx]) - np.max(np.array(log_like_cov[idx])))

        # Interpolate to find the 95% limits
        f_interp1 = interp1d(omega_m_ary, likelihood_arr - threshold_1sigma, kind="linear", fill_value="extrapolate")
        x_vals = np.linspace(omega_m_ary[0], omega_m_ary[-1], 1000)
        diff_signs1 = np.sign(f_interp1(x_vals))

        # Find where the sign changes
        sign_changes1 = ((diff_signs1[:-1] * diff_signs1[1:]) < 0).nonzero()[0]

        if len(sign_changes1) >= 2:  # We need at least two crossings
            intervals1.append((x_vals[sign_changes1[0]], x_vals[sign_changes1[-1]]))
            true_values.append(conditioning[idx][0])
        else:
            # Optionally handle the case where no interval is found
            pass

    # Plotting true value vs. interval
    fig = plt.figure(figsize=(10, 4))

    for value, (low, high) in zip(true_values, intervals1):
        plt.errorbar(value, (low + high) / 2.0, yerr=[[(low + high) / 2.0 - low], [high - (low + high) / 2.0]], fmt="o", capsize=5, color="k")

    plt.plot([0, 1], [0, 1], color="k", ls="--")

    plt.xlim(0.05, 0.5)
    plt.ylim(0.05, 0.5)

    plt.xlabel("True Value")
    plt.ylabel("Estimated Value and Interval")
    plt.grid(True)
    plt.tight_layout()

    if log_wandb:
        wandb.log({"eval/llprof_Om": wandb.Image(plt)})


def eval_generation(
    vdm,
    pstate,
    rng,
    n_samples: int,
    n_particles: int,
    true_samples: np.array,
    conditioning: np.array,
    mask: np.array,
    norm_dict: Dict,
    steps: int = 500,
    boxsize: float = 1000.0,
    log_wandb: bool = True,
):
    """Evaluate the model on a small subset and log figures and log figures and log figures and log figures

    Args:
        vdm (_type_): diffusion model
        pstate (_type_): model weights
        rng (_type_): random key
        n_samples (int): number of samples to generate
        n_particles (int): number of particles to sample
        true_samples (np.array): true samples
        conditioning (np.array): conditioning of the true samples
        mask (np.array): mask
        norm_dict (Dict): dictionariy with mean and std of the true samples, used to normalize the data
        steps (int, optional): number of steps to sample in diffusion. Defaults to 100.
        boxsize (float, optional): size of the simulation box. Defaults to 1000.0.
    """
    generated_samples = generate_samples(
        vdm=vdm,
        params=pstate.params,
        rng=rng,
        n_samples=n_samples,
        n_particles=n_particles,
        conditioning=conditioning,
        mask=mask,
        steps=steps,
        norm_dict=norm_dict,
        boxsize=boxsize,
    )
    true_samples = true_samples * norm_dict["std"] + norm_dict["mean"]
    true_positions = true_samples[..., :3]
    generated_positions = generated_samples[..., :3]
    if generated_samples.shape[-1] > 3:
        generated_velocities = generated_samples[..., 3:6]
        generated_masses = generated_samples[..., -1]
        true_velocities = true_samples[..., 3:6]
        if generated_samples.shape[-1] > 6:
            true_masses = true_samples[..., -1]
        else:
            generated_masses = None
            true_masses = None
    else:
        generated_velocities = None
        generated_masses = None
        true_velocities = None
        true_masses = None
    fig = plot_pointclouds_2D(generated_samples=generated_positions, true_samples=true_positions)
    """
    if log_wandb:
        # wandb.log({"eval/pointcloud": fig})
        wandb.log({"eval/pointcloud": wandb.Image(plt)})
    fig = plot_knns(
        generated_samples=generated_positions,
        true_samples=true_positions,
        conditioning=conditioning,
        boxsize=boxsize,
        idx_to_plot=range(6),
    )
    if log_wandb:
        wandb.log({"eval/knn": fig})
    """

    fig = plot_2pcf(
        generated_samples=generated_positions,
        true_samples=true_positions,
        boxsize=boxsize,
    )
    if log_wandb:
        wandb.log({"eval/2pcf": fig})

    if generated_velocities is not None:
        fig = plot_velocity_histograms(
            generated_velocities=generated_velocities,
            true_velocities=true_velocities,
            idx_to_plot=[0, 1, 2],
        )
        if log_wandb:
            wandb.log({"eval/vels": fig})
        fig = plot_2pcf_rsd(
            generated_positions=onp.array(generated_positions),
            true_positions=onp.array(true_positions),
            generated_velocities=onp.array(generated_velocities),
            true_velocities=onp.array(true_velocities),
            conditioning=onp.array(conditioning),
            boxsize=boxsize,
        )
        if log_wandb:
            wandb.log({"eval/2pcf_rsd": fig})

    if generated_masses is not None:
        fig = plot_hmf(
            generated_masses=generated_masses,
            true_masses=true_masses,
            idx_to_plot=[
                0,
                1,
                2,
                3,
            ],
        )
        if log_wandb:
            wandb.log({"eval/mass": fig})


def generate_samples(
    vdm,
    params,
    rng,
    n_samples,
    n_particles,
    conditioning,
    mask,
    steps,
    norm_dict,
    boxsize,
):
    generated_samples = generate(
        vdm,
        params,
        rng,
        (n_samples, n_particles),
        conditioning=conditioning,
        mask=mask,
        steps=steps,
    )
    generated_samples = generated_samples.mean()
    generated_samples = generated_samples * norm_dict["std"] + norm_dict["mean"]
    # make sure generated samples are inside boxsize
    generated_samples = generated_samples.at[..., :3].set(generated_samples[..., :3] % boxsize)
    return generated_samples


def generate_test_samples_from_model_folder(
    path_to_model: Path,
    steps: int = 500,
    batch_size: int = 20,
    boxsize: float = 1000.0,
    n_test: int = 200,
):
    with open(path_to_model / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
    config = ConfigDict(config)
    # get conditioning for test set
    test_ds, norm_dict = nbody_dataset(
        n_features=config.data.n_features,
        n_particles=config.data.n_particles,
        batch_size=batch_size,
        seed=config.seed,
        shuffle=False,
        split="test",
    )
    return generate_samples_for_dataset(
        ds=test_ds,
        n_particles=config.data.n_particles,
        norm_dict=norm_dict,
        n_total_samples=n_test,
        path_to_model=path_to_model,
        steps=steps,
        batch_size=batch_size,
        boxsize=boxsize,
    )


def generate_samples_for_dataset(
    ds,
    norm_dict,
    n_particles: int,
    n_total_samples: int,
    path_to_model: Path,
    steps: int = 500,
    batch_size: int = 20,
    boxsize: float = 1000.0,
):
    batches = create_input_iter(ds)
    x_batch, conditioning_batch, mask_batch = next(batches)
    vdm, params = VariationalDiffusionModel.from_path_to_model(path_to_model=path_to_model)
    rng = jax.random.PRNGKey(42)
    n_batches = n_total_samples // batch_size
    true_samples, generated_samples, conditioning_samples = [], [], []
    for i in range(n_batches):
        t0 = time.time()
        x_batch, conditioning_batch, mask_batch = next(batches)
        true_samples.append(x_batch[0] * norm_dict["std"] + norm_dict["mean"])
        generated_samples.append(
            generate_samples(
                vdm=vdm,
                params=params,
                rng=rng,
                n_samples=batch_size,
                n_particles=n_particles,
                conditioning=conditioning_batch[0],
                mask=mask_batch[0],
                steps=steps,
                norm_dict=norm_dict,
                boxsize=boxsize,
            )
        )
        conditioning_samples.append(conditioning_batch[0])
        print(f"Iteration {i} takes {time.time() - t0} seconds")
    return np.array(true_samples), np.array(generated_samples), np.array(conditioning_samples)


if __name__ == "__main__":
    t0 = time.time()
    run_name = "blooming-puddle-230"  # 'misunderstood-night-203' #'confused-gorge-138' #'chocolate-cloud-122'
    path_to_samples = Path(f"/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/set_diffuser/samples/{run_name}")
    path_to_samples.mkdir(exist_ok=True)
    path_to_model = Path(f"/n/home11/ccuestalazaro/set-diffuser/logging/cosmology/{run_name}")
    steps = 500
    true_samples, generated_samples, conditioninig_samples = generate_test_samples_from_model_folder(
        path_to_model=path_to_model,
        steps=steps,
    )
    np.save(path_to_samples / f"true_test_samples.npy", true_samples)
    np.save(path_to_samples / f"generated_test_samples_{steps}_steps.npy", generated_samples)
    np.save(path_to_samples / f"cond_test_samples_{steps}_steps.npy", conditioninig_samples)
    print(f"It takes {time.time() - t0} seconds to generate samples")
