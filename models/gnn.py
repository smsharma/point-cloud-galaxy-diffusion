from typing import Callable, Tuple
import flax.linen as nn
import jax.numpy as jnp
import jraph
from models.mlp import MLP
from models.graph_utils import PairNorm, Identity


def get_node_mlp_updates(mlp_feature_sizes: int, name: str = None) -> Callable:
    """Get a node MLP update  function

    Args:
        mlp_feature_sizes (int): number of features in the MLP
        name (str, optional): name of the update function. Defaults to None.

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


def get_edge_mlp_updates(
    mlp_feature_sizes: int, name: str = None, relative_updates: bool = False
) -> Callable:
    """Get an edge MLP update function

    Args:
        mlp_feature_sizes (int): number of features in the MLP
        name (str, optional): name of the update function. Defaults to None.

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
        # If there are no edges in the initial layer
        if edges is not None:
            if relative_updates:
                inputs = jnp.concatenate([edges, senders - receivers, globals], axis=1)
            else:
                inputs = jnp.concatenate([edges, senders, receivers, globals], axis=1)

        else:
            if relative_updates:
                inputs = jnp.concatenate([senders - receivers, globals], axis=1)
            else:
                inputs = jnp.concatenate([senders, receivers, globals], axis=1)

        return MLP(mlp_feature_sizes, name=name)(inputs)

    return update_fn


def get_attention_logit_fn(name: str = None) -> Callable:
    """Get an attention logits function for each edge

    Args:
        name (str, optional): name of the function. Defaults to None.

    Returns:
        Callable: update function
    """

    def attention_logit_fn(edges, senders, receivers, globals):
        """Returns the attention logits for each edge."""
        inputs = jnp.concatenate([edges, senders, receivers, globals], axis=-1)
        pre_activations = MLP([1], name=name)(inputs)
        return nn.gelu(pre_activations)  # Apply activation to get attention logits

    return attention_logit_fn


def attention_reduce_fn(edges, weights):
    """Applies attention weights to the edge features."""
    edges = edges * weights
    return edges


class GraphConvNet(nn.Module):
    """A simple graph convolutional network"""

    latent_size: int
    hidden_size: int
    num_mlp_layers: int
    message_passing_steps: int
    skip_connections: bool = True
    edge_skip_connections: bool = True
    norm: str = "layer"
    attention: bool = False
    in_features: int = 3
    shared_weights: bool = False  # GNN shares weights across message passing steps
    relative_updates: bool = False

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Do message passing on graph

        Args:
            graphs (jraph.GraphsTuple): graph object

        Returns:
            jraph.GraphsTuple: updated graph object
        """

        mlp_feature_sizes = [self.hidden_size] * self.num_mlp_layers + [
            self.latent_size
        ]

        # First linearly project the original features as 'embeddings'.
        if graph.edges is None:
            embedder = jraph.GraphMapFeatures(embed_node_fn=MLP(mlp_feature_sizes))
        else:
            embedder = jraph.GraphMapFeatures(
                embed_node_fn=MLP(mlp_feature_sizes),
                embed_edge_fn=MLP(mlp_feature_sizes),
            )
        graph = embedder(graph)
        graph = graph._replace(
            globals=graph.globals.reshape(graph.globals.shape[0], -1)
        )

        # Apply GCN once for each message-passing round.
        for step in range(self.message_passing_steps):
            # Initialize update functions with shared weights if specified;
            # otherwise, initialize new weights for each step
            if step == 0 or not self.shared_weights:
                suffix = "shared" if self.shared_weights else step

                update_node_fn = get_node_mlp_updates(
                    mlp_feature_sizes,
                    name=f"update_node_fn_{suffix}",
                )
                update_edge_fn = get_edge_mlp_updates(
                    mlp_feature_sizes,
                    name=f"update_edge_fn_{suffix}",
                    relative_updates=self.relative_updates,
                )
                attention_logit_fn = (
                    get_attention_logit_fn(name=f"attention_logit_fn_{suffix}")
                    if self.attention
                    else None
                )

                # Update nodes and edges; no need to update globals as they only condition
                graph_net = jraph.GraphNetwork(
                    update_node_fn=update_node_fn,
                    update_edge_fn=update_edge_fn,
                    attention_logit_fn=attention_logit_fn,
                    attention_reduce_fn=attention_reduce_fn if self.attention else None,
                )

            # Update graph, optionally with residual connection
            if self.skip_connections:
                new_graph = graph_net(graph)
                graph = graph._replace(
                    nodes=graph.nodes + new_graph.nodes,
                    edges=new_graph.edges
                    if (graph.edges is None or not self.edge_skip_connections)
                    else graph.edges + new_graph.edges,
                )
            else:
                graph = graph_net(graph)

            # Optional normalization
            if self.norm == "layer":
                norm = nn.LayerNorm()
            elif self.norm == "pair":
                norm = PairNorm()
            else:
                norm = Identity()  # No normalization

            graph = graph._replace(nodes=norm(graph.nodes), edges=norm(graph.edges))

        # Decode the final node features back to the original feature dimension
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=MLP(
                [self.hidden_size] * self.num_mlp_layers + [self.in_features]
            )
        )

        return decoder(graph)
