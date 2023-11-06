from typing import Callable
from jax.experimental.sparse import BCOO
import jax.numpy as np
import flax.linen as nn
import jraph
from models.graph_utils import get_laplacian
from models.mlp import MLP


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm; generate scale and shift parameters from conditioning context.
    Same as the one for transformer, but with MLP instead of dense scale and shift output
    and dims assuming single batch.
    """

    @nn.compact
    def __call__(self, x, conditioning):
        # Compute scale and shift parameters from conditioning context
        # scale_and_shift = nn.gelu(nn.Dense(2 * x.shape[-1])(conditioning))  # Most implementations use just a linear layer
        scale_and_shift = MLP([4 * conditioning.shape[-1], 2 * x.shape[-1]])(conditioning)
        scale, shift = np.split(scale_and_shift, 2, axis=-1)

        # Don't use bias or scale since these will be learnable through the conditioning context
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)

        # Apple same element-wise scale and shift to all elements in graph
        x = x * (1 + scale[None, :]) + shift[None, :]

        return x


class ChebConv(nn.Module):
    out_channels: int = 128
    K: int = 6
    bias: bool = True
    skip_connection: bool = True

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, lambda_max: float = None) -> jraph.GraphsTuple:
        """Chebychev convolutional layer, based on https://arxiv.org/abs/1606.09375.
        Reference implementation: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html
        """
        L, (senders, receivers), norm = self.__norm__(edge_index=np.array([graph.senders, graph.receivers]), edge_weight=graph.edges, lambda_max=lambda_max, num_nodes=graph.nodes.shape[0])

        # Recursively get Chebychev polynomial coefficients
        Tx_0 = graph.nodes  # Initial feature vector
        out = nn.Dense(self.out_channels)(Tx_0)

        if self.K > 1:
            Tx_1 = L @ Tx_0
            out = out + nn.Dense(self.out_channels)(Tx_1)

        for _ in range(2, self.K):
            Tx_2 = L @ Tx_1
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + nn.Dense(self.out_channels)(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.out_channels,))
            out = out + bias

        return graph._replace(nodes=out)

    def __norm__(self, edge_index, edge_weight, lambda_max=None, num_nodes=5000):
        """Get and normalize graph Laplacian."""

        # Get graph Laplacian
        # Symmetric norm is tricky given Jax static reqs
        L, edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=num_nodes)

        assert edge_weight is not None, "Edge weights cannot be None after getting the Laplacian."

        # If lambda_max is not specified, calculate it using the edge weights
        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()

        # Normalizing edge weights
        edge_weight = (2.0 * edge_weight) / lambda_max

        return 2.0 * L / lambda_max, edge_index, edge_weight


class ChebConvNet(nn.Module):
    out_channels: int = 128
    K: int = 6
    bias: bool = True
    message_passing_steps: int = 4
    skip_connection: bool = True

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, lambda_max: float = None) -> jraph.GraphsTuple:
        in_channels = graph.nodes.shape[-1]

        # Linear embedding
        embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.out_channels))
        graph = embedder(graph)

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
