import dataclasses

import jax
import jax.numpy as np
import flax.linen as nn
import jraph
import e3nn_jax as e3nn

from models.transformer import Transformer
from models.gnn import GraphConvNet
from models.equivariant_transformer import EquivariantTransformer

from models.graph_utils import nearest_neighbors
from models.diffusion_utils import get_timestep_embedding


class TransformerScoreNet(nn.Module):
    d_embedding: int = 8
    d_t_embedding: int = 32
    score_dict: dict = dataclasses.field(default_factory=lambda: {"d_model": 256, "d_mlp": 512, "n_layers": 4, "n_heads": 4})

    @nn.compact
    def __call__(self, z, t, conditioning, mask):

        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(z.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)  # Timestep embeddings

        if conditioning is not None:
            cond = np.concatenate([t_embedding, conditioning], axis=1)  # Concatenate with conditioning context
        else:
            cond = t_embedding

        # Pass context through a small MLP before passing into transformer
        cond = nn.gelu(nn.Dense(features=self.d_embedding * 4)(cond))
        cond = nn.gelu(nn.Dense(features=self.d_embedding * 4)(cond))
        cond = nn.Dense(self.d_embedding)(cond)

        h = Transformer(n_input=self.d_embedding, **self.score_dict)(z, cond, mask)

        return z + h


class GraphScoreNet(nn.Module):
    d_embedding: int = 8
    d_t_embedding: int = 32
    score_dict: dict = dataclasses.field(default_factory=lambda: {"k": 20, "num_mlp_layers": 4, "latent_size": 128, "skip_connections": True, "message_passing_steps": 4})
    pos_features: int = 3  # TODO: Generalize data structure. Breaks previous transformer models.

    @nn.compact
    def __call__(self, z, t, conditioning, mask):

        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(z.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)  # Timestep embeddings

        if conditioning is not None:
            cond = np.concatenate([t_embedding, conditioning], axis=1)  # Concatenate with conditioning context
        else:
            cond = t_embedding

        # Pass context through a small MLP before passing into transformer
        cond = nn.gelu(nn.Dense(features=self.d_embedding * 4)(cond))
        cond = nn.gelu(nn.Dense(features=self.d_embedding * 4)(cond))
        cond = nn.Dense(self.d_embedding)(cond)

        k = self.score_dict["k"]

        sources, targets = jax.vmap(nearest_neighbors, in_axes=(0, None))(z[..., : self.pos_features], k, mask=mask)

        n_batch = z.shape[0]
        graph = jraph.GraphsTuple(n_node=mask.sum(-1)[:, None], n_edge=np.array(n_batch * [[k]]), nodes=z, edges=None, globals=cond, senders=sources, receivers=targets)

        # Make copy of score dict since original cannot be in-place modified; remove `k` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("k")
        score_dict.pop("score")

        h = jax.vmap(GraphConvNet(**score_dict))(graph)
        h = h.nodes

        return z + h


class EquivariantTransformereNet(nn.Module):
    d_embedding: int = 8
    d_t_embedding: int = 32
    score_dict: dict = dataclasses.field(default_factory=lambda: {"k": 20})
    pos_features: int = 3  # TODO: Generalize data structure. Breaks previous transformer models.

    @nn.compact
    def __call__(self, z, t, conditioning, mask):

        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(z.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)  # Timestep embeddings

        if conditioning is not None:
            cond = np.concatenate([t_embedding, conditioning], axis=1)  # Concatenate with conditioning context
        else:
            cond = t_embedding

        # Pass context through a small MLP before passing into transformer
        cond = nn.gelu(nn.Dense(features=self.d_embedding * 4)(cond))
        cond = nn.gelu(nn.Dense(features=self.d_embedding * 4)(cond))
        cond = nn.Dense(self.d_embedding)(cond)

        k = self.score_dict["k"]

        pos, vel, mass = z[..., : self.pos_features], z[..., self.pos_features : 2 * self.pos_features], z[..., 2 * self.pos_features :]
        sources, targets = jax.vmap(nearest_neighbors, in_axes=(0, None))(z[..., : self.pos_features], k, mask=mask)

        pos = e3nn.IrrepsArray("1o", pos)
        feat = e3nn.IrrepsArray("1o + {}x0e".format(self.d_embedding), np.concatenate([vel, mass + cond[:, None, :]], -1))

        # Make copy of score dict since original cannot be in-place modified; remove `k` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("k")
        score_dict.pop("score")

        pos_update, feat_update = jax.vmap(EquivariantTransformer(irreps_out="1o + 0e"))(pos, feat, sources, targets)
        z = np.concatenate([pos_update.array, feat_update.array], -1)

        return z


# TODO: Fix d_embedding; add comment about masking in equivariant transformer
