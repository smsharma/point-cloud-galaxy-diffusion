from typing import Callable, Tuple
import jax
import flax.linen as nn
import jax.numpy as jnp
import jraph
from einops import rearrange
from models.mlp import MLP


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


def get_edge_mlp_updates(mlp_feature_sizes: int, name: str = None) -> Callable:
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
            inputs = jnp.concatenate([edges, senders, receivers, globals], axis=1)
        else:
            inputs = jnp.concatenate([senders, receivers, globals], axis=1)
        return MLP(mlp_feature_sizes, name=name)(inputs)

    return update_fn


def get_attention_logit_fn(num_heads: int = 1, name: str = None) -> Callable:
    """Get an attention logits function for each edge

    Args:
        name (str, optional): name of the function. Defaults to None.

    Returns:
        Callable: update function
    """

    def attention_logit_fn(edges, senders, receivers, globals):
        """Returns the attention logits for each edge."""
        inputs = jnp.concatenate([edges, senders, receivers, globals], axis=-1)
        pre_activations = MLP([num_heads], name=name)(inputs)
        return nn.gelu(pre_activations)  # Apply activation to get attention logits

    return attention_logit_fn


def attention_reduce_fn(edges, weights):
    """Applies attention weights to the edge features."""
    # `edges`` has shape (n_edges, n_edges_features).
    # `weights`` has shape (n_edges, n_attention_heads).
    # Do an outer product over the last dim
    edges = jnp.einsum("ef,eh->efh", edges, weights)
    # Concatenate attention heads together
    edges = rearrange(edges, "e f h -> e (f h)")
    return edges


class GraphConvNet(nn.Module):
    """A simple graph convolutional network"""

    latent_size: int
    hidden_size: int
    num_mlp_layers: int
    message_passing_steps: int
    skip_connections: bool = True
    layer_norm: bool = True
    attention: bool = False
    num_heads: int = 1
    in_features: int = 3
    shared_weights: bool = False  # GNN shares weights across message passing steps

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Do message passing on graph

        Args:
            graphs (jraph.GraphsTuple): graph object

        Returns:
            jraph.GraphsTuple: updated graph object
        """

        # First linearly project the original node features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.latent_size))
        graph = embedder(graph)
        graph = graph._replace(
            globals=graph.globals.reshape(graph.globals.shape[0], -1),
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
                attention_logit_fn = get_attention_logit_fn(self.num_heads, name=f"attention_logit_fn_{suffix}") if self.attention else None

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
                    edges=new_graph.edges if graph.edges is None else graph.edges + new_graph.edges,
                )
            else:
                graph = graph_net(graph)

            # Optional layer norm
            if self.layer_norm:
                graph = graph._replace(nodes=nn.LayerNorm()(graph.nodes))

        decoder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.in_features))

        return decoder(graph)
