# Set Diffuser: Transformer-guided diffusion for set modeling

Jax implementation of a transformer-guided variational diffusion model for class- and context-conditional generative modeling of and inference on set data.

![MNIST](./notebooks/mnist_dark.png#gh-dark-mode-only)
![MNIST](./notebooks/mnist_light.png#gh-light-mode-only)

> "It is in complex systems, ones in which we have little visibility of the chains of cause-consequences, that tinkering, bricolage, or similar variations of trial and error have been shown to vastly outperform the teleological — it is nature’s modus operandi" -NNT

## Description

- The diffusion backbone is based on the implementation of a [variational diffusion model](https://github.com/google-research/vdm) ([blog post](https://blog.alexalemi.com/diffusion.html)). 
- The score model is a vanilla transformer without positional encodings and with masked attention to account for sets of different cardinality.
- [Flash attention](https://github.com/lucidrains/flash-attention-jax) can be optionally installed and used to circumvent the quadratic scaling of standard attention with sequence length and scale to larger sets.
- Simple element-wise residual MLPs project the set features to and from a latent space, where diffusion is modeled.
- The model can be optionally conditioned on a class as well as a general context. If `n_classes` > 0, the first element of the conditioning vector is assumed to be the integer class of the sample.

## Examples

The [`notebooks`](notebooks/) directory contains usage example, including a simple [MNIST point cloud example](notebooks/example-mnist.ipynb) showing class-conditional generation, as well as a [particle physics example](notebooks/example-jets.ipynb). 

## Usage

``` py
import jax
import jax.numpy as np

from flax.core import FrozenDict

from models.diffusion import VariationalDiffusionModel
from models.diffusion_utils import generate, loss_vdm

# Transformer args
transformer_dict = FrozenDict({"d_model":256, "d_mlp":512, "n_layers":5, "n_heads":4, "flash_attention":True})

# Instantiate model
vdm = VariationalDiffusionModel(gamma_min=-6.0, gamma_max=6.0,  # Min and max initial log-SNR in the noise schedule
          d_feature=4,  # Number of features per set element
          transformer_dict=transformer_dict,  # Score-prediction transformer parameters
          noise_schedule="learned_linear",  # Noise schedule; "learned_linear" or "scalar"
          n_layers=3,  # Layers in encoder/decoder element-wise ResNets
          d_embedding=8,  # Dim to encode the per-element features to
          d_hidden_encoding=64,  # Hidden dim used in various contexts (for embedding context, 4 * for encoding/decoding in ResNets)
          timesteps=300,  # Number of diffusion steps
          d_t_embedding=16,  # Timestep embedding dimension
          noise_scale=1e-3,  # Data noise model
          n_classes=0)  # Number of data classes. If >0, the first element of the conditioning vector is assumed to be integer class.

rng = jax.random.PRNGKey(42)

x = jax.random.normal(rng, (32, 100, 4))  # Input set
mask = jax.random.randint(rng, (32, 100), 0, 2)  # Optional set mask; can be `None`
conditioning = jax.random.normal(rng, (32, 6))  # Optional conditioning context; can be `None`

# Call to get losses; see https://blog.alexalemi.com/diffusion.html
(loss_diff, loss_klz, loss_recon), params = vdm.init_with_output({"sample": rng, "params": rng, "uncond":rng}, x, conditioning, mask)

# Compute full loss, accounting for masking
loss_vdm(params, vdm, rng, x, conditioning, mask)  # DeviceArray(5606182.5, dtype=float32)

# Sample from model

mask_sample = jax.random.randint(rng, (24, 100), 0, 2)
conditionink_sample = jax.random.normal(rng, (24, 6))

x_samples = generate(vdm, params, rng, (24, 100), conditionink_sample, mask_sample)
x_samples.mean().shape  # Mean of decoded Normal distribution -- (24, 100, 4)
```

## TODO

- [ ] Add examples for ELBO-based likelihood inference
- [ ] Experiment with different conditioning schemes
