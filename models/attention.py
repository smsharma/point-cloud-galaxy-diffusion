from flax import linen as nn
import jax.numpy as jnp

from typing import Any, Callable, Optional

Array = Any


class MultiHeadAttentionBlock(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x: Array, y: Array, mask: Optional[Array] = None):
        mask = None if mask is None else mask[..., None, :, :]
        h = nn.LayerNorm()(x + nn.MultiHeadDotProductAttention(self.num_heads)(x, y, mask))
        return nn.LayerNorm()(h + nn.relu(nn.Dense(x.shape[-1])(h)))


class SetAttentionBlock(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array] = None):
        mask = None if mask is None else mask[..., None] * mask[..., None, :]
        return MultiHeadAttentionBlock(self.num_heads)(x, x, mask)


class PoolingByMultiHeadAttention(nn.Module):
    num_seed_vectors: int
    num_heads: int
    seed_vectors_init: Callable = nn.linear.default_embed_init

    @nn.compact
    def __call__(self, z: Array, mask: Optional[Array] = None):
        seed_vectors = self.param("seed_vectors", self.seed_vectors_init, (self.num_seed_vectors, z.shape[-1]))
        seed_vectors = jnp.broadcast_to(seed_vectors, z.shape[:-2] + seed_vectors.shape)
        mask = None if mask is None else mask[..., None, :]
        return MultiHeadAttentionBlock(self.num_heads)(seed_vectors, z, mask)


class InducedSetAttentionBlock(nn.Module):
    num_inducing_points: int
    num_heads: int
    inducing_points_init: Callable = nn.linear.default_embed_init

    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array] = None):
        h = PoolingByMultiHeadAttention(self.num_inducing_points, self.num_heads, self.inducing_points_init)(x, mask)
        mask = None if mask is None else mask[..., None]
        return MultiHeadAttentionBlock(self.num_heads)(x, h, mask)
