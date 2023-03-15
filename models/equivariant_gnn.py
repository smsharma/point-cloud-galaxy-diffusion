from typing import Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn

" Implementaiton of https://arxiv.org/pdf/2101.03164.pdf "
" following https://github.com/mariogeiger/nequip-jax/tree/main"


class NEQUIP(nn.Module):
    use_node_features: bool = True
    target_irreps: e3nn.Irreps = e3nn.Irreps("2x1o + 0e")
    n_layers: int = 3

    @nn.compact
    def __call__(self, positions: jnp.array, node_features: jnp.array, senders: jnp.array, receivers: jnp.array)->jnp.array:
        """ Forward pass of a nequip graph network

        Args:
            positions (jnp.array): 3D positions  
            node_features (jnp.array): node features 
            senders (jnp.array): senders of the edges 
            receivers (jnp.array): receivers of the edges 

        Returns:
            jnp.array: new positions and node features 
        """
        vectors = positions[receivers] - positions[senders]
        #vectors = e3nn.IrrepsArray("1o", positions[receivers] - positions[senders])
        for l in range(self.n_layers):
            layer = NEQUIPLayer(
                avg_num_neighbors=1.0,
                target_irreps=self.target_irreps,
            )
            node_features = layer(
                vectors,
                node_features,
                senders=senders,
                receivers=receivers,
            )
        print(node_features.shape)
        print(node_features)
        vectors, masses = node_features[:, "1o"], node_features[:, "0e"]
        displacement, vel = vectors.slice_by_mul[:1], vectors.slice_by_mul[1:]
        positions = positions + displacement 
        return displacement 
        #print('node features')
        #print(node_features)
        #displacements = node_features[:, "0e"] 
        #return positions + displacements
        #displacements, node_features = node_features[:, "1o"], node_features[:, "1o + 0e"]
        #return positions + displacements, node_features


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

        node_feats = e3nn.flax.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = MessagePassingConvolution(
            self.avg_num_neighbors,
            irreps,
            self.mlp_activation,
            self.mlp_n_hidden,
            self.mlp_n_layers,
            self.n_radial_basis,
            self.sh_lmax,
        )(vectors, node_feats, senders, receivers)

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
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
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
        mix = e3nn.flax.MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
            self.activation,
            output_activation=False,
        )(
            jnp.where(
                lengths == 0.0,  # discard 0 length edges that come from graph padding
                0.0,
                e3nn.bessel(lengths[:, 0], self.n_radial_basis)
                * e3nn.poly_envelope(5, 2)(lengths),
            )
        )  # [n_edges, num_irreps]

        # Product of radial and angular part
        messages = messages * mix  # [n_edges, irreps]

        # Message passing
        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
        return node_feats / jnp.sqrt(self.avg_num_neighbors)
