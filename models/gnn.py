from typing import Callable, Tuple
import jax
import flax.linen as nn
import jax.numpy as jnp
import jraph

from models.graph_utils import add_graphs_tuples
from models.mlp import MLP


def get_node_mlp_updates(mlp_feature_sizes: int, name: str = None) -> Callable:
    """Get a node MLP update  function

    Args:
        mlp_feature_sizes (int): number of features in the MLP

    Returns:
        Callable: update function
    """

    def update_fn(
        nodes: jnp.ndarray,
        sent_attributes: jnp.ndarray,
        received_attributes: jnp.ndarray,
        globals: jnp.ndarray,
    ) -> jnp.ndarray:
        """update node features

        Args:
            nodes (jnp.ndarray): node features
            sent_attributes (jnp.ndarray): attributes sent to neighbors
            received_attributes (jnp.ndarray): attributes received from neighbors
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated node features
        """
        if received_attributes is not None:
            inputs = jnp.concatenate([nodes, received_attributes, globals], axis=1)
        else:  # If lone node
            inputs = jnp.concatenate([nodes, globals], axis=1)
        return MLP(mlp_feature_sizes, name=name)(inputs)

    return update_fn


def get_edge_mlp_updates(mlp_feature_sizes: int, name: str = None) -> Callable:
    """Get an edge MLP update function

    Args:
        mlp_feature_sizes (int): number of features in the MLP

    Returns:
        Callable: update function
    """

    def update_fn(
        edges: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        globals: jnp.ndarray,
    ) -> jnp.ndarray:
        """update edge features

        Args:
            edges (jnp.ndarray): edge attributes
            senders (jnp.ndarray): senders node attributes
            receivers (jnp.ndarray): receivers node attributes
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated edge features
        """
        if edges is not None:
            inputs = jnp.concatenate([edges, senders, receivers, globals], axis=1)
        else:
            inputs = jnp.concatenate([senders, receivers, globals], axis=1)
        return MLP(mlp_feature_sizes, name=name)(inputs)

    return update_fn


class GraphConvNet(nn.Module):
    """A simple graph convolutional network"""

    latent_size: int
    hidden_size: int
    num_mlp_layers: int
    message_passing_steps: int
    skip_connections: bool = True
    layer_norm: bool = True
    attention: bool = False
    in_features: int = 3
    shared_weights: bool = False  # GNN shares weights across message passing steps

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Do message passing on graph

        Args:
            graphs (jraph.GraphsTuple): graph object

        Returns:
            jraph.GraphsTuple: updated graph object
        """

        # We will first linearly project the original node features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.latent_size))
        processed_graphs = embedder(graphs)
        processed_graphs = processed_graphs._replace(
            globals=processed_graphs.globals.reshape(processed_graphs.globals.shape[0], -1),
        )
        mlp_feature_sizes = [self.hidden_size] * self.num_mlp_layers + [self.latent_size]

        # Apply GCN once for each message-passing round.
        for step in range(self.message_passing_steps):
            # Initialize update functions with shared weights if specified;
            # otherwise, initialize new weights for each step
            if step == 0 or not self.shared_weights:
                suffix = "shared" if self.shared_weights else step

                update_node_fn = get_node_mlp_updates(mlp_feature_sizes, name=f"update_node_fn_{suffix}")
                update_edge_fn = get_edge_mlp_updates(mlp_feature_sizes, name=f"update_edge_fn_{suffix}")

            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
            )
            if self.skip_connections:
                processed_graphs = add_graphs_tuples(graph_net(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_net(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(nodes=nn.LayerNorm()(processed_graphs.nodes))
        decoder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.in_features))

        return decoder(processed_graphs)
