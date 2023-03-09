import sys

sys.path.append("./")
sys.path.append("../")

from typing import List, Dict
from jax import random
import jax.numpy as np
import numpy as onp

import wandb
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction
from models.diffusion_utils import generate
from cosmo_utils.knn import get_CDFkNN


def plot_pointclouds_3D(generated_samples: np.array, true_samples: np.array, idx_to_plot: int=0)-> plt.figure:
    """ Plot pointcloud in three dimensions

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
        1, 2, figsize=(20, 12), subplot_kw={"projection": "3d"}
    )
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

def plot_pointclouds_2D(generated_samples: np.array, true_samples: np.array, idx_to_plot: int =0):
    """ Plot pointcloud in two dimensions

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
        1, 2, figsize=(20, 12), 
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
    boxsize: float =1000.0,
    idx_to_plot: List[int]=[0, 1, 2],
)->plt.figure:
    """ plot nearest neighbour statistics

    Args:
        generated_samples (np.array): samples generated by the model 
        true_samples (np.array): true samples 
        conditioning (np.array): conditioning per sample 
        boxsize (float, optional): size of the simulation box. Defaults to 1000.0.
        idx_to_plot (List[int], optional): idx to plot. Defaults to [0, 1, 2].

    Returns:
        plt.figure: figure 
    """
    r_bins = np.linspace(0.5, 120.0, 60)
    k_bins = [1, 5, 9]
    key = random.PRNGKey(0)
    random_points = boxsize*random.uniform(
        key,
        shape=(len(true_samples[0]) * 10, 3),
    )
    fig, _ = plt.subplots()
    for i, idx in enumerate(idx_to_plot):
        sampled_knn = get_CDFkNN(
            r_bins=r_bins,
            pos=generated_samples[idx][...,:3],
            random_pos=random_points,
            boxsize=boxsize,
            k=k_bins,
        )
        true_knn = get_CDFkNN(
            r_bins=r_bins,
            pos=true_samples[idx][...,:3],
            random_pos=random_points,
            boxsize=boxsize,
            k=k_bins,
        )

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        for k in range(len(k_bins)):
            c = plt.plot(
                r_bins,
                true_knn[k],
                label=rf"$\Omega_m={conditioning[i][0]:.2f} \,\,\sigma_8={conditioning[i][-1]:.2f}$"
                if k == 1
                else None,
                ls="-",
                alpha=0.75,
                lw=2,
                color=color,
            )
            plt.plot(
                r_bins,
                sampled_knn[k],
                color=color,
                ls="--",
                alpha=0.75,
                lw=2,
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

def compute_2pcf(sample: np.array, boxsize: float, r_bins: np.array,)->np.array:
    """ Get the monopole of the two point correlation function

    Args:
        sample (np.array): positions 
        boxsize (float): size of the box 
        r_bins (np.array): bins in pair separation 

    Returns:
        np.array: monopole of the two point correlation function 
    """
    mu_bins = np.linspace(-1,1,201)
    return TwoPointCorrelationFunction(
            "smu",
            edges=(onp.array(r_bins), onp.array(mu_bins)),
            data_positions1=onp.array(sample).T,
            engine="corrfunc",
            n_threads=2,
            boxsize=boxsize,
            los='z',
        )(ells=[0])[0]

def plot_2pcf(generated_samples: np.array, true_samples: np.array, boxsize: float)->plt.figure:
    """ Plot the two point correlation function

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
        generated_2pcfs.append(compute_2pcf(generated_samples[idx][...,:3], boxsize, r_bins))
        true_2pcfs.append(compute_2pcf(true_samples[idx][...,:3], boxsize, r_bins))

    fig, _ = plt.subplots()
    c = plt.loglog(r, onp.mean(true_2pcfs, axis=0), label='N-body')
    plt.plot(
        r,
        (onp.mean(true_2pcfs, axis=0) - onp.std(true_2pcfs,axis=0)), 
        alpha=0.5,
        color=c[0].get_color(),
        linestyle='dashed',
    )
    plt.plot(
        r,
        (onp.mean(true_2pcfs, axis=0) + onp.std(true_2pcfs,axis=0)), 
        alpha=0.5,
        color=c[0].get_color(),
        linestyle='dashed',
    )

    # fill_between somehow doesnt work with wandb :(
    #plt.fill_between(
    #    r, 
    #    (onp.mean(true_2pcfs, axis=0) - onp.std(true_2pcfs,axis=0)), 
    #    (onp.mean(true_2pcfs, axis=0) + onp.std(true_2pcfs,axis=0)), 
    #    alpha=0.5,
    #    color=c[0].get_color(),
    #)
    c = plt.plot(r, onp.mean(generated_2pcfs, axis=0), label='Diffusion')
    plt.plot(
        r,
        (onp.mean(generated_2pcfs, axis=0) - onp.std(generated_2pcfs,axis=0)), 
        alpha=0.5,
        color=c[0].get_color(),
        linestyle='dashed',
    )
    plt.plot(
        r,
        (onp.mean(generated_2pcfs, axis=0) + onp.std(generated_2pcfs,axis=0)), 
        alpha=0.5,
        color=c[0].get_color(),
        linestyle='dashed',
    )
    #plt.fill_between(
    #    r, 
    #    (onp.mean(generated_2pcfs, axis=0) - onp.std(generated_2pcfs,axis=0)), 
    #    (onp.mean(generated_2pcfs, axis=0) + onp.std(generated_2pcfs,axis=0)), 
    #    alpha=0.5,
    #    color=c[0].get_color(),
    #)
    plt.ylabel('2PCF')
    plt.xlabel('r [Mpc/h]')
    plt.legend(fontsize=8)
    return fig

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
    steps: int =1000,
    boxsize: float =1000.0,
):
    """ Evaluate the model on a small subset and log figures and log figures and log figures and log figures

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
    generated_samples = generate(
        vdm,
        pstate.params,
        rng,
        (n_samples, n_particles),
        conditioning=conditioning,
        mask=mask,
        steps=steps,
    )
    generated_samples = generated_samples.mean()
    generated_samples = generated_samples * norm_dict["std"] + norm_dict["mean"]
    # make sure generated samples are inside boxsize
    generated_samples = generated_samples.at[...,:3].set(generated_samples[...,:3] % boxsize) 
    true_samples = true_samples * norm_dict["std"] + norm_dict["mean"]
    fig = plot_pointclouds_2D(
        generated_samples=generated_samples, true_samples=true_samples
    )
    wandb.log({"eval/pointcloud": fig})

    fig = plot_knns(
        generated_samples=generated_samples,
        true_samples=true_samples,
        conditioning=conditioning,
        boxsize=boxsize,
    )
    wandb.log({"eval/knn": fig})
    
    fig = plot_2pcf(
        generated_samples=generated_samples,
        true_samples=true_samples,
        boxsize=boxsize,
    )
    wandb.log({"eval/2pcf": fig})
    