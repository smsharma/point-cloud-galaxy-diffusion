from typing import Callable
from jax.experimental.sparse import BCOO
import jax.numpy as np
import flax.linen as nn
import jraph
from models.mlp import MLP


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm; generate scale and shift parameters from conditioning context.
    Same as the one for transformer, but with MLP instead of dense scale and shift output
    and dims assuming single batch.
    """

    @nn.compact
    def __call__(self, x, conditioning):
        # Compute scale and shift parameters from conditioning context
        emb = nn.Dense(2 * x.shape[-1])(conditioning)
        scale_and_shift = nn.Dense(2 * x.shape[-1])(nn.gelu(emb))  # Most implementations use just a linear layer
        scale, shift = np.split(scale_and_shift, 2, axis=-1)

        # Don't use bias or scale since these will be learnable through the conditioning context
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)

        # Apply same element-wise scale and shift to all elements in graph
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

        # Initialize the output feature as zero
        out = np.zeros_like(graph.nodes[:, : self.out_channels])

        Tx_0 = graph.nodes  # Initial feature vector
        Tx_1 = None

        # Loop over the order of Chebyshev polynomials
        # x'_{if'} = Sum_k Sum_f Wk_{f'f} Sum_j L_{ij} Tk(x)_{jf}
        for k in range(self.K):
            # Apply the Chebyshev transformation for the k-th term
            if k == 0:
                Tx_k = Tx_0
            elif k == 1:
                Tx_1 = L @ Tx_0
                Tx_k = Tx_1
            else:
                Tx_2 = 2 * L @ Tx_1 - Tx_0
                Tx_0, Tx_1 = Tx_1, Tx_2
                Tx_k = Tx_1

            # Accumulate the results
            out += nn.Dense(self.out_channels)(Tx_k)

        if self.bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.out_channels,))
            out = out + bias

        return graph._replace(nodes=out)

    def __norm__(self, edge_index, edge_weight, lambda_max=None, num_nodes=5000):
        """Get and normalize graph Laplacian."""

        # Get graph Laplacian
        # Symmetric norm is tricky given Jax static reqs
        L, edge_index, edge_weight = self.get_laplacian(edge_index, edge_weight, num_nodes=num_nodes)

        assert edge_weight is not None, "Edge weights cannot be None after getting the Laplacian."

        # If lambda_max is not specified, calculate it using the edge weights
        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()

        # Normalizing edge weights
        edge_weight = (2.0 * edge_weight) / lambda_max

        return 2.0 * L / lambda_max, edge_index, edge_weight

    def get_laplacian(self, edge_index, edge_weight=None, num_nodes=None):
        if edge_weight is None:
            edge_weight = np.ones_like(edge_index[0])

        A = BCOO((edge_weight, edge_index.T), shape=(num_nodes, num_nodes))

        # D_ii = Sum_j A_{ij}
        deg = A.sum(axis=1)
        D = BCOO((deg.todense(), np.array([np.arange(num_nodes), np.arange(num_nodes)]).T), shape=(num_nodes, num_nodes))

        L = D - A  # L_{ij}

        return L, L.indices.T, L.data


class ChebConvNet(nn.Module):
    out_channels: int = 128
    K: int = 6
    bias: bool = True
    message_passing_steps: int = 5
    skip_connection: bool = True
    add_global: bool = True

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, lambda_max: float = 2.0) -> jraph.GraphsTuple:
        in_channels = graph.nodes.shape[-1]

        # Linear embedding
        embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.out_channels))
        graph = embedder(graph)

        for _ in range(self.message_passing_steps):
            # Optionally add an embedding of the global context to each node feature
            if self.add_global:
                emb_global = nn.Dense(self.out_channels)(graph.globals)
                graph = graph._replace(nodes=graph.nodes + nn.Dense(self.out_channels)(nn.gelu(emb_global))[None, :])

            graph_net = ChebConv(out_channels=self.out_channels, K=self.K, bias=self.bias)
            if self.skip_connection:
                new_graph = graph_net(graph, lambda_max)
                graph = graph._replace(nodes=new_graph.nodes + graph.nodes)
            else:
                graph = graph_net(graph, lambda_max)

            # Nonlinearity, norm, and global conditioning
            graph = graph._replace(nodes=AdaLayerNorm()(nn.gelu(graph.nodes), graph.globals))

        # Readout
        # graph = graph._replace(nodes=nn.Dense(in_channels)(graph.nodes))  # Linear readout
        graph = graph._replace(nodes=MLP([2 * self.out_channels, 2 * self.out_channels, in_channels])(graph.nodes))
        return graph
