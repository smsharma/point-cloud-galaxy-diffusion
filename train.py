import sys
import os
import shutil

sys.path.append("./")
sys.path.append("../")

import pandas as pd

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import checkpoints

import tensorflow as tf

# Ensure TF does not see GPU and grab all GPU memory
tf.config.set_visible_devices([], device_type="GPU")

from tqdm import trange

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

from models.diffusion import VariationalDiffusionModel
from models.diffusion_utils import loss_vdm
from models.train_utils import create_input_iter, param_count, StateStore, train_step

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

n_particles = 5000
n_seq = 5000
n_features = 7  # All 7 features
batch_size = 8
train_steps = 500_000
warmup_steps = 4000
save_every = 4000
flash_attention = False
ckpt_dir = "/n/dvorkin_lab/smsharma/functional-diffusion/notebooks/ckpts_all_batch/"

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)

print("{} devices visible".format(jax.device_count()))

x = np.load("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/halos.npy")
x_mean = x.mean(axis=(0, 1))
x_std = x.std(axis=(0, 1))
x = (x - x_mean + 1e-7) / (x_std + 1e-7)

x = x[:, :n_particles, :n_features]
x = np.pad(x, [(0, 0), (0, n_seq - n_particles), (0, 0)])
conditioning = np.array(pd.read_csv("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/cosmology.csv").values)

mask = np.ones((x.shape[0], n_particles))
mask = np.pad(mask, [(0, 0), (0, n_seq - n_particles)])

batch_size = batch_size * jax.device_count()
n_train = len(x)

train_ds = tf.data.Dataset.from_tensor_slices((x, conditioning, mask))
train_ds = train_ds.cache()
train_ds = train_ds.repeat()

batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

for batch_size in reversed(batch_dims):
    train_ds = train_ds.batch(batch_size, drop_remainder=False)

train_ds = train_ds.shuffle(n_train, seed=42)
train_df = create_input_iter(train_ds)

transformer_dict = FrozenDict({"d_model": 256, "d_mlp": 1024, "n_layers": 6, "n_heads": 4, "flash_attention": flash_attention})  # Transformer args

vdm = VariationalDiffusionModel(gamma_min=-8.0, gamma_max=6.0, noise_schedule="learned_linear", n_layers=5, d_embedding=12, d_hidden_encoding=512, timesteps=300, d_t_embedding=32, d_feature=n_features, antithetic_time_sampling=True, transformer_dict=transformer_dict, n_classes=0)
batches = create_input_iter(train_ds)

# Past a test batch through to initialize model

x_batch, conditioning_batch, mask_batch = next(batches)
rng = jax.random.PRNGKey(42)
out, params = vdm.init_with_output({"sample": rng, "params": rng, "uncond": rng}, x_batch[0], conditioning_batch[0], mask_batch[0])

print(f"Params: {param_count(params):,}")

train_steps = train_steps // jax.device_count()
warmup_steps = warmup_steps // jax.device_count()

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=6e-4,
    warmup_steps=warmup_steps,
    decay_steps=train_steps,
)

opt = optax.adamw(learning_rate=schedule, weight_decay=1e-4)

# Init state store
store = StateStore(params, opt.init(params), rng, 0)
pstore = replicate(store)

vals = []
with trange(train_steps) as t:
    for i in t:
        pstore, val = train_step(pstore, loss_vdm, vdm, next(batches), opt)
        v = unreplicate(val)
        t.set_postfix(val=v)
        vals.append(v)

        if i % save_every == 0:
            ckpt = unreplicate(pstore)
            checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=ckpt, step=i, overwrite=True, keep=3)
