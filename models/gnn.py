import jax
import jax.numpy as np
import flax.linen as nn
import jraph

from models.graph_utils import add_graphs_tuples

from typing import Sequence, Callable


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    activation: Callable[[np.ndarray], np.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
        return x


class GraphConvNet(nn.Module):
    """A Graph Convolution Network + Pooling model defined with Jraph."""

    latent_size: int
    num_mlp_layers: int
    message_passing_steps: int
    skip_connections: bool = True
    layer_norm: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:

        # We will first linearly project the original node features as 'embeddings'.

        in_features = graphs.nodes.shape[-1]

        embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.latent_size))
        processed_graphs = embedder(graphs)

        # Now, we will apply the GCN once for each message-passing round.
        for _ in range(self.message_passing_steps):
            mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
            update_node_fn = jraph.concatenated_args(MLP(mlp_feature_sizes))
            graph_conv = jraph.GraphConvolution(update_node_fn=update_node_fn, add_self_edges=False)

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(graph_conv(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_conv(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=nn.LayerNorm()(processed_graphs.nodes),
                )

        decoder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(in_features))

        processed_graphs = decoder(processed_graphs)

        return processed_graphs
