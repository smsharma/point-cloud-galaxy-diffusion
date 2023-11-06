from typing import Callable
import jax
import jax.numpy as np
import flax.linen as nn
import jraph
from models.graph_utils import get_laplacian
from models.mlp import MLP


def get_node_mlp_updates() -> Callable:
    def update_fn(
        nodes: np.ndarray,
        sent_attributes: np.ndarray,
        received_attributes: np.ndarray,
        globals: np.ndarray,
    ) -> np.ndarray:
        return received_attributes[..., None] * nodes

    return update_fn


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm; generate scale and shift parameters from conditioning context."""

    @nn.compact
    def __call__(self, x, conditioning):
        # Compute scale and shift parameters from conditioning context
        # scale_and_shift = nn.gelu(nn.Dense(2 * x.shape[-1])(conditioning))
        scale_and_shift = MLP([4 * conditioning.shape[-1], 2 * x.shape[-1]])(conditioning)
        scale, shift = np.split(scale_and_shift, 2, axis=-1)

        # Apply layer norm
        # Don't use bias or scale since these will be learnable through the conditioning context
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)

        # Apply scale and shift
        # Apple same scale, shift to all elements in sequence
        x = x * (1 + scale[None, :]) + shift[None, :]

        return x


class ChebConv(nn.Module):
    out_channels: int = 128
    K: int = 6
    bias: bool = True
    skip_connection: bool = True

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, lambda_max: float = None) -> jraph.GraphsTuple:
        (senders, receivers), norm = self.__norm__(edge_index=np.array([graph.senders, graph.receivers]), edge_weight=graph.edges, lambda_max=lambda_max, num_nodes=graph.nodes.shape[0])

        Tx_0 = graph.nodes
        Tx_1 = graph.nodes
        out = nn.Dense(self.out_channels)(Tx_0)

        if self.K > 1:
            graph_Tx_1 = graph._replace(senders=senders, receivers=receivers, edges=norm)
            graph_Tx_1 = jraph.GraphNetwork(update_node_fn=get_node_mlp_updates(), update_edge_fn=None)(graph_Tx_1)
            Tx_1 = graph_Tx_1.nodes

            out = out + nn.Dense(self.out_channels)(Tx_1)

        for _ in range(2, self.K):
            graph_Tx_2 = graph._replace(nodes=Tx_1, senders=senders, receivers=receivers, edges=norm)
            Tx_2 = 2.0 * graph_Tx_2.nodes - Tx_0
            out = out + nn.Dense(self.out_channels)(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.out_channels,))
            out = out + bias

        return graph._replace(nodes=out)

    def __norm__(self, edge_index, edge_weight, lambda_max=None, num_nodes=5000):
        # Adjusting the get_laplacian function call to the correct format
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=num_nodes)

        assert edge_weight is not None, "Edge weights cannot be None after getting the Laplacian."

        # If lambda_max is not specified, calculate it as twice the max of the edge weights
        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()

        # Normalizing edge weights
        edge_weight = (2.0 * edge_weight) / lambda_max

        return edge_index, edge_weight


class ChebConvNet(nn.Module):
    out_channels: int = 128
    K: int = 6
    bias: bool = True
    message_passing_steps: int = 4
    skip_connection: bool = True

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, lambda_max: float = None) -> jraph.GraphsTuple:
        in_channels = graph.nodes.shape[-1]

        for _ in range(self.message_passing_steps):
            graph_net = ChebConv(out_channels=self.out_channels, K=self.K, bias=self.bias)
            if self.skip_connection:
                new_graph = graph_net(graph, lambda_max)
                graph = graph._replace(nodes=new_graph.nodes + graph.nodes)
            else:
                graph = graph_net(graph, lambda_max)

            # Nonlinearity, norm, and global conditioning
            graph = graph._replace(nodes=AdaLayerNorm()(nn.gelu(graph.nodes), graph.globals))

        # Linear readout, keeping it simple
        graph = graph._replace(nodes=nn.Dense(in_channels)(graph.nodes))
        return graph
