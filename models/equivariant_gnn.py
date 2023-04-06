from typing import Callable, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.tree_util as tree

from jraph._src import graph as gn_graph
import e3nn_jax as e3nn
import haiku as hk

from models.equivariant_batchnorm import EquivariantBatchNorm

" Implementaiton of https://arxiv.org/pdf/2101.03164.pdf "
" following https://github.com/mariogeiger/nequip-jax/tree/main"


class NEQUIP(nn.Module):
    use_node_features: bool = True
    target_irreps: e3nn.Irreps = e3nn.Irreps("2x1o") # + 0e")
    input_node_irreps: e3nn.Irreps = e3nn.Irreps(
        "1o"
    )  # + 0e") #TODO: masses currently not working due to batch norm
    avg_num_neighbors: int = 20
    n_layers: int = 3
    pos_dim: int = 3
    sh_lmax: int = 3

    @nn.compact
    def __call__(self, graphs):
        """Forward pass of a nequip graph network

        Args:
            positions (jnp.array): 3D positions
            node_features (jnp.array): node features
            senders (jnp.array): senders of the edges
            receivers (jnp.array): receivers of the edges

        Returns:
            jnp.array: new positions and node features
        """
        positions = e3nn.IrrepsArray("1o", graphs.nodes[:, : self.pos_dim])
        node_features = e3nn.IrrepsArray(
            self.input_node_irreps,
            graphs.nodes[:, self.pos_dim :],
        )
        senders = graphs.senders
        receivers = graphs.receivers
        vectors = positions[receivers] - positions[senders]
        for l in range(self.n_layers):
            layer = NEQUIPLayer(
                avg_num_neighbors=self.avg_num_neighbors,
                target_irreps=self.target_irreps,
                sh_lmax=self.sh_lmax,
            )
            node_features = layer(
                vectors=vectors,
                node_feats=node_features,
                globals=graphs.globals,
                senders=senders,
                receivers=receivers,
            )
            node_features = EquivariantBatchNorm(node_features.irreps)(node_features)

        vectors, h = node_features.filter(keep="1o"), node_features.filter(keep="0e")
        displacement, vel = vectors.slice_by_mul[:1], vectors.slice_by_mul[1:]
        positions = positions + displacement
        nodes = jnp.concatenate((positions.array, vel.array, h.array), axis=-1)
        return gn_graph.GraphsTuple(
            nodes=nodes,
            edges=graphs.edges,
            receivers=graphs.receivers,
            senders=graphs.senders,
            globals=graphs.globals_ if hasattr(graphs, "globals_") else None,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )


class NEQUIPLayer(nn.Module):
    avg_num_neighbors: float
    sh_lmax: int = 3
    target_irreps: e3nn.Irreps = e3nn.Irreps("2x1o + 0e")
    even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    n_radial_basis: int = 8

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        globals: e3nn.IrrepsArray,  # [n_globals]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ):
        n_edge = vectors.shape[0]
        n_node = node_feats.shape[0]
        assert vectors.shape == (n_edge, 3)
        assert node_feats.shape == (n_node, node_feats.irreps.dim)
        assert senders.shape == (n_edge,)
        assert receivers.shape == (n_edge,)

        # target irreps plus extra scalars for the gate activation
        target_irreps = e3nn.Irreps(self.target_irreps)
        irreps = target_irreps + target_irreps.filter(
            drop="0e + 0o"
        ).num_irreps * e3nn.Irreps("0e")

        self_connection = e3nn.flax.Linear(
            irreps,
        )(
            node_feats
        )  # [n_nodes, feature * target_irreps]

        # TODO: Concatenate globals?
        node_feats = e3nn.flax.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = MessagePassingConvolution(
            avg_num_neighbors=self.avg_num_neighbors,
            target_irreps=irreps,
            activation=self.mlp_activation,
            mlp_n_hidden=self.mlp_n_hidden,
            mlp_n_layers=self.mlp_n_layers,
            n_radial_basis=self.n_radial_basis,
            sh_lmax=self.sh_lmax,
        )(
            vectors=vectors,
            node_feats=node_feats,
            globals=globals,
            senders=senders,
            receivers=receivers,
            n_edge=n_edge,
        )

        node_feats = e3nn.flax.Linear(irreps, name="linear_down")(node_feats)

        node_feats = node_feats + self_connection  # [n_nodes, irreps]

        node_feats = e3nn.gate(
            node_feats,
            even_act=self.even_activation,
            even_gate_act=self.even_activation,
            odd_act=self.odd_activation,
            odd_gate_act=self.odd_activation,
        )

        assert node_feats.irreps == target_irreps.regroup()
        assert node_feats.shape == (n_node, target_irreps.dim)
        return node_feats


class MessagePassingConvolution(nn.Module):
    avg_num_neighbors: float
    target_irreps: e3nn.Irreps
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    n_radial_basis: int = 8
    sh_lmax: int = 3

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        globals: e3nn.IrrepsArray,  # [n_globals]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
        n_edge: Optional[int] = None,
    ) -> e3nn.IrrepsArray:
        messages = node_feats[senders]
        # Angular part
        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    e3nn.spherical_harmonics(
                        [l for l in range(1, self.sh_lmax + 1)],
                        vectors,
                        normalize=True,
                        normalization="component",
                    ),
                    filter_ir_out=self.target_irreps,
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        # Radial part
        lengths = e3nn.norm(vectors).array  # [n_edges, 1]
        lengths_in_basis = jnp.where(
            lengths == 0.0,  # discard 0 length edges that come from graph padding
            0.0,
            e3nn.bessel(lengths[:, 0], self.n_radial_basis)
            * e3nn.poly_envelope(5, 2)(lengths),
        )
        sum_n_edge = senders.shape[0]
        global_edge_attributes = tree.tree_map(
            lambda g: jnp.repeat(g, n_edge, axis=0, total_repeat_length=sum_n_edge),
            globals.reshape(1,-1),
        )
        input_mlp = jnp.concatenate([lengths_in_basis, global_edge_attributes], axis=-1)
        mix = e3nn.flax.MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
            self.activation,
            output_activation=False,
        )(
            input_mlp
        )  # [n_edges, num_irreps]

        # Product of radial and angular part
        messages = messages * mix  # [n_edges, irreps]

        # Message passing
        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
        return node_feats / jnp.sqrt(self.avg_num_neighbors)
