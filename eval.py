import sys
sys.path.append("./")
sys.path.append("../")

import wandb
import matplotlib.pyplot as plt
from models.diffusion_utils import generate

def eval_generation(
    vdm, pstate, rng, n_samples, n_particles, conditioning, mask,steps=100):
    x_samples = generate(
        vdm, pstate.params, rng, (n_samples, n_particles), conditioning=conditioning, mask=mask, steps=steps,
    )
    x_samples = x_samples.mean()
    idx = 0

    s = 4
    alpha = 0.5
    color = "firebrick"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), subplot_kw={'projection':'3d'})

    ax1.scatter(x_samples[idx, :, 0], x_samples[idx, :, 1], x_samples[idx, :, 2], alpha=alpha, s=s, color=color);
    ax1.set_title("Gen")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    wandb.log({"eval/plot": fig})


