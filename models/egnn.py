import jax
from typing import Callable
import flax.linen as nn
import jax.numpy as jnp
import jraph

from models.graph_utils import add_graphs_tuples
from models.mlp import MLP


class CoordNorm(nn.Module):
    """Coordinate normalization, from
    https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py#LL67C28-L67C28
    """

    eps: float = 1e-8
    scale_init: float = 1.0

    def setup(self):
        self.scale = self.param("scale", nn.initializers.ones, (1,))

    def __call__(self, coors):
        norm = jnp.linalg.norm(coors, axis=-1, keepdims=True)
        normed_coors = coors / jax.lax.clamp(self.eps, norm, jnp.inf)
        return normed_coors * self.scale


def get_edge_mlp_updates(
    d_hidden, n_layers, activation, position_only=False
) -> Callable:
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

        # Split senders and receivers into coordinates, velocities, and scalar attrs
        x_i, v_i, h_i = senders[:, :3], senders[:, 3:6], senders[:, 6:]
        x_j, v_j, h_j = receivers[:, :3], receivers[:, 3:6], receivers[:, 6:]

        # Messages from Eqs. (3) and (4)/(7)
        phi_e = MLP([d_hidden] * (n_layers), activation=activation)
        phi_x = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)

        m_ij = phi_e(
            jnp.concatenate(
                [
                    h_i,
                    h_j,
                    jnp.linalg.norm(x_i - x_j, axis=1, keepdims=True) ** 2,
                    globals,
                ],
                axis=-1,
            )
        )
        return (x_i - x_j) * phi_x(m_ij), m_ij

    def update_fn_position_only(
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

        # Split senders and receivers into coordinates, velocities, and scalar attrs
        x_i = senders
        x_j = receivers

        # Messages from Eqs. (3) and (4)/(7)
        phi_e = MLP([d_hidden] * (n_layers), activation=activation)
        phi_x = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)

        # Get invariants
        message_scalars = jnp.concatenate(
            [jnp.linalg.norm(x_i - x_j, axis=1, keepdims=True) ** 2, globals], axis=-1
        )
        jax.debug.print(f'nans in message scalars = {jnp.sum(jnp.isnan(message_scalars))}')
        ''''
        if edges is not None:
            # edges[1] = m_ij? -> edges are updated, so after one iteration it won't be None but the message
            message_scalars = jnp.concatenate(
                [message_scalars, edges[1]], axis=-1
            )  # Add edge features if available
        '''
        m_ij = phi_e(message_scalars)
        jax.debug.print(f'nans in m_ij = {jnp.sum(jnp.isnan(m_ij))}')
        return (x_i - x_j) * phi_x(m_ij), m_ij

    return update_fn if not position_only else update_fn_position_only


def get_node_mlp_updates(
    d_hidden, n_layers, activation, n_edge, position_only=False
) -> Callable:
    """Get an node MLP update function

    Args:
        mlp_feature_sizes (int): number of features in the MLP

    Returns:
        Callable: update function
    """

    def update_fn(
        nodes: jnp.ndarray,
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

        sum_x_ij, m_i = receivers  # Get aggregated messages
        x_i, v_i, h_i = nodes[:, :3], nodes[:, 3:6], nodes[:, 6:]  # Split node attrs

        # From Eqs. (6) and (7)
        phi_v = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)
        phi_h = MLP(
            [d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation
        )

        # Apply updates
        v_i_p = sum_x_ij / (n_edge - 1) + phi_v(h_i) * v_i
        x_i_p = x_i + v_i_p
        h_i_p = phi_h(jnp.concatenate([h_i, m_i], -1)) + h_i  # Skip connection

        return jnp.concatenate([x_i_p, v_i_p, h_i_p], -1)

    def update_fn_position_only(
        nodes: jnp.ndarray,
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

        sum_x_ij, _ = receivers  # Get aggregated messages
        x_i = nodes

        # Apply updates
        x_i_p = x_i + sum_x_ij

        return x_i_p

    return update_fn if not position_only else update_fn_position_only


class EGNN(nn.Module):
    """A simple graph convolutional network"""

    message_passing_steps: int = 4
    skip_connections: bool = False
    norm_layer: bool = True
    d_hidden: int = 64
    n_layers: int = 3
    activation: str = "gelu"

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Do message passing on graph

        Args:
            graphs (jraph.GraphsTuple): graph object

        Returns:
            jraph.GraphsTuple: updated graph object
        """
        in_features = graphs.nodes.shape[-1]
        processed_graphs = graphs
        processed_graphs = processed_graphs._replace(
            globals=processed_graphs.globals.reshape(
                processed_graphs.globals.shape[0], -1
            )
        )
        activation = getattr(nn, self.activation)

        # Switch for whether to use positions-only version of edge/node updates
        if (graphs.nodes.shape[-1] > 3) & (graphs.nodes.shape[-1] < 6):
            raise NotImplementedError(
                "Number of features should be either 3 (just positions) or >= 6 (positions, velocities, and scalars)"
            )

        positions_only = True if graphs.nodes.shape[-1] == 3 else False

        update_node_fn = get_node_mlp_updates(
            self.d_hidden,
            self.n_layers,
            activation,
            n_edge=processed_graphs.n_edge,
            position_only=positions_only,
        )
        update_edge_fn = get_edge_mlp_updates(
            self.d_hidden, self.n_layers, activation, position_only=positions_only
        )
        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):
            jax.debug.print('**********')
            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn, update_edge_fn=update_edge_fn
            )
            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                    graph_net(processed_graphs), processed_graphs
                )
            else:
                processed_graphs = graph_net(processed_graphs)
            jax.debug.breakpoint()
            jax.debug.print(f'nans before norm layers = {jnp.sum(jnp.isnan(processed_graphs.nodes))}')
            if self.norm_layer:
                processed_graphs = self.norm(
                    processed_graphs, positions_only=positions_only
                )
        return processed_graphs

    def norm(self, graph, positions_only=False):
        if not positions_only:
            x, v, h = graph.nodes[..., :3], graph.nodes[..., 3:6], graph.nodes[..., 6:]

            # Only apply LN if scalars have more than one feature
            x, v, h = (
                CoordNorm()(x),
                CoordNorm()(v),
                h if h.shape[-1] == 1 else nn.LayerNorm()(h),
            )
            graph = graph._replace(nodes=jnp.concatenate([x, v, h], -1))
        else:
            x = CoordNorm()(graph.nodes)
            graph = graph._replace(nodes=x)
        return graph
