import jax
import jax.numpy as np
import flax.linen as nn
import e3nn_jax as e3nn

from typing import Callable, List, Tuple


def _index_max(i: np.ndarray, x: np.ndarray, out_dim: int) -> np.ndarray:
    return np.zeros((out_dim,) + x.shape[1:], x.dtype).at[i].max(x)


class EquivariantTransformerBlock(nn.Module):
    irreps_node_output: e3nn.Irreps
    list_neurons: Tuple[int, ...]
    act: Callable[[np.ndarray], np.ndarray]
    num_heads: int = 1

    @nn.compact
    def __call__(
        self,
        edge_src: np.ndarray,  # [E] dtype=int32
        edge_dst: np.ndarray,  # [E] dtype=int32
        edge_weight_cutoff: np.ndarray,  # [E] dtype=float
        edge_attr: e3nn.IrrepsArray,  # [E, D] dtype=float
        node_feat: e3nn.IrrepsArray,  # [N, D] dtype=float
    ) -> e3nn.IrrepsArray:
        r"""Equivariant Transformer.

        Args:
            edge_src (array of int32): source index of the edges
            edge_dst (array of int32): destination index of the edges
            edge_weight_cutoff (array of float): cutoff weight for the edges (typically given by ``soft_envelope``)
            edge_attr (e3nn.IrrepsArray): attributes of the edges (typically given by ``spherical_harmonics``)
            node_f (e3nn.IrrepsArray): features of the nodes

        Returns:
            e3nn.IrrepsArray: output features of the nodes
        """

        def f(x, y, filter_ir_out=None, name=None):
            out1 = e3nn.concatenate([x, e3nn.tensor_product(x, y.filter(drop="0e"))]).regroup().filter(keep=filter_ir_out)
            out2 = e3nn.flax.MultiLayerPerceptron(self.list_neurons + (out1.irreps.num_irreps,), self.act, output_activation=False, name=name)(y.filter(keep="0e"))
            return out1 * out2

        edge_key = f(node_feat[edge_src], edge_attr, node_feat.irreps, name="mlp_key")
        edge_logit = e3nn.flax.Linear(f"{self.num_heads}x0e", name="linear_logit")(e3nn.tensor_product(node_feat[edge_dst], edge_key, filter_ir_out="0e")).array  # [E, H]
        node_logit_max = _index_max(edge_dst, edge_logit, node_feat.shape[0])  # [N, H]
        exp = edge_weight_cutoff[:, None] * np.exp(edge_logit - node_logit_max[edge_dst])  # [E, H]
        z = e3nn.scatter_sum(exp, dst=edge_dst, output_size=node_feat.shape[0])  # [N, H]
        z = np.where(z == 0.0, 1.0, z)
        alpha = exp / z[edge_dst]  # [E, H]

        edge_v = f(node_feat[edge_src], edge_attr, self.irreps_node_output, "mlp_val")  # [E, D]
        edge_v = edge_v.mul_to_axis(self.num_heads)  # [E, H, D]
        edge_v = edge_v * np.sqrt(jax.nn.relu(alpha))[:, :, None]  # [E, H, D]
        edge_v = edge_v.axis_to_mul()  # [E, D]

        node_out = e3nn.scatter_sum(edge_v, dst=edge_dst, output_size=node_feat.shape[0])  # [N, D]
        return e3nn.flax.Linear(self.irreps_node_output, name="linear_out")(node_out)  # [N, D]


class EquivariantTransformer(nn.Module):
    irreps_out: e3nn.Irreps

    @nn.compact
    def __call__(
        self,
        positions: e3nn.IrrepsArray,  # [N, 3] dtype=float
        features: e3nn.IrrepsArray,  # [N, D] dtype=float
        senders: np.array,
        receivers: np.array,
        cutoff: float = 1.0,
    ):
        r"""Equivariant Transformer.

        Args:
            positions (e3nn.IrrepsArray): positions of the nodes
            features (e3nn.IrrepsArray): features of the nodes
            senders (np.array): graph senders array
            receivers (np.array): graph receivers array
            cutoff (float): cutoff radius

        Returns:
            e3nn.IrrepsArray: output features of the nodes
        """

        vectors = positions[senders] - positions[receivers]
        dist = np.linalg.norm(vectors.array, axis=1) / cutoff

        edge_attr = e3nn.concatenate([e3nn.bessel(dist, 8), e3nn.spherical_harmonics(list(range(1, 3 + 1)), vectors, True)])
        edge_weight_cutoff = e3nn.soft_envelope(dist)

        features = EquivariantTransformerBlock(
            irreps_node_output=e3nn.Irreps("1o") + self.irreps_out,
            list_neurons=(64, 64),
            act=jax.nn.gelu,
            num_heads=1,
        )(senders, receivers, edge_weight_cutoff, edge_attr, features)

        displacements, features = features.slice_by_mul[:1], features.slice_by_mul[1:]
        positions = positions + displacements
        return positions, features
