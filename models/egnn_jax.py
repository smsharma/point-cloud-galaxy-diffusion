from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
from jax.tree_util import Partial
from models.mlp import MLP
from models.graph_utils import nearest_neighbors


class EGNNLayer(nn.Module):
    """EGNN layer."""

    layer_num: int
    hidden_size: int
    output_size: int
    blocks: int = 1
    act_fn: Callable = jax.nn.silu
    pos_aggregate_fn: Optional[Callable] = jraph.segment_sum
    msg_aggregate_fn: Optional[Callable] = jraph.segment_sum
    residual: bool = True
    attention: bool = False
    normalize: bool = False
    tanh: bool = False
    eps: float = 1e-8
    coord_mean: jnp.ndarray = None
    coord_std: jnp.ndarray = None
    box_size: float = None

    def setup(self):
        # message network
        self.edge_mlp = MLP([self.hidden_size] * self.blocks + [self.hidden_size], activation=self.act_fn, activate_final=True)

        # update network
        self.node_mlp = MLP([self.hidden_size] * self.blocks + [self.output_size], activation=self.act_fn, activate_final=False)

        # position update network
        # self.pos_mlp = MLP([self.hidden_size] * self.blocks + [1], activation=self.act_fn)
        # Use a separate output layer for position update, with small kernel init
        self.pos_mlp = MLP([self.hidden_size] * self.blocks, activation=self.act_fn, activate_final=True)
        self.pos_mlp_last_layer = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.variance_scaling(scale=1e-3, mode="fan_in", distribution="truncated_normal"))

        # attention
        self.attention_mlp = None
        if self.attention:
            # self.attention_mlp = hk.Sequential([hk.Linear(hidden_size), jax.nn.sigmoid])
            # Rewrite using Flax and MLP
            self.attention_mlp = MLP([self.hidden_size], activation=self.act_fn)

    def pos_update(
        self,
        pos: jnp.ndarray,
        graph: jraph.GraphsTuple,
        coord_diff: jnp.ndarray,
    ) -> jnp.ndarray:
        trans = self.pos_mlp_last_layer(self.pos_mlp(graph.edges))

        if self.tanh:
            trans = jax.nn.tanh(trans)

        trans = coord_diff * trans

        # NOTE: was in the original code
        trans = jnp.clip(trans, -100, 100)
        return self.pos_aggregate_fn(trans, graph.senders, num_segments=pos.shape[0])

    def message(
        self,
        radial: jnp.ndarray,
        edge_attribute: jnp.ndarray,
        edge_features: Any,
        incoming: jnp.ndarray,
        outgoing: jnp.ndarray,
        globals: Any,
    ) -> jnp.ndarray:
        msg = jnp.concatenate([incoming, outgoing, radial, globals], axis=-1)
        if edge_attribute is not None:
            msg = jnp.concatenate([msg, edge_attribute], axis=-1)
        msg = self.edge_mlp(msg)
        if self.attention_mlp:
            att = self.attention_mlp(msg)
            att = nn.sigmoid(att)
            msg = msg * att
        return msg

    def update(
        self,
        node_attribute: jnp.ndarray,
        nodes: jnp.ndarray,
        senders: Any,
        msg: jnp.ndarray,
        globals: Any,
    ) -> jnp.ndarray:
        _ = senders

        x = jnp.concatenate([nodes, msg], axis=-1)
        if node_attribute is not None:
            x = jnp.concatenate([x, node_attribute], axis=-1)
        x = self.node_mlp(x)
        if self.residual:
            x = nodes + x
        return x

    def coord2radial(self, graph: jraph.GraphsTuple, coord: jnp.array) -> Tuple[jnp.array, jnp.array]:
        if self.box_size is not None:
            coord_diff_unnormed = (coord[graph.senders] - coord[graph.receivers]) * self.coord_std + self.coord_mean  # Compute distance and un-normalize
            coord_diff_unnormed = coord_diff_unnormed - self.box_size * jnp.round(coord_diff_unnormed / self.box_size)  # Get distances in periodic box
            coord_diff = (coord_diff_unnormed - self.coord_mean) / self.coord_std  # Normalize again
        else:
            coord_diff = coord[graph.senders] - coord[graph.receivers]
        radial = jnp.sum(coord_diff**2, 1)[:, jnp.newaxis]
        if self.normalize:
            norm = jnp.sqrt(radial)
            coord_diff = coord_diff / (norm + self.eps)
        return radial, coord_diff

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        """
        Apply EGNN layer.

        Args:
            graph: Graph from previous step
            pos: Node position, updated separately
            edge_attribute: Edge attribute (optional)
            node_attribute: Node attribute (optional)
        """
        radial, coord_diff = self.coord2radial(graph, pos)

        graph = jraph.GraphNetwork(
            update_edge_fn=Partial(self.message, radial, edge_attribute),
            update_node_fn=Partial(self.update, node_attribute),
            aggregate_edges_for_nodes_fn=self.msg_aggregate_fn,
        )(graph)

        pos = pos + self.pos_update(pos, graph, coord_diff)

        return graph, pos


class EGNN(nn.Module):
    r"""
    E(n) Graph Neural Network (https://arxiv.org/abs/2102.09844).

    Original implementation: https://github.com/vgsatorras/egnn
    """

    hidden_size: int = 128
    act_fn: Callable = jax.nn.gelu
    num_layers: int = 4
    residual: bool = True
    attention: bool = True
    normalize: bool = False
    tanh: bool = True

    @nn.compact
    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
        coord_mean: Optional[jnp.ndarray] = None,
        coord_std: Optional[jnp.ndarray] = None,
        box_size: float = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply EGNN.

        Args:
            graph: Input graph
            pos: Node position
            edge_attribute: Edge attribute (optional)
            node_attribute: Node attribute (optional)

        Returns:
            Tuple of updated node features and positions
        """

        output_shape = graph.nodes.shape[-1]
        graph = graph._replace(globals=graph.globals.reshape(1, -1))

        # message passing
        for n in range(self.num_layers):
            graph, pos = EGNNLayer(
                layer_num=n,
                hidden_size=self.hidden_size,
                blocks=2,
                output_size=output_shape,
                act_fn=self.act_fn,
                residual=self.residual,
                attention=self.attention,
                normalize=self.normalize,
                tanh=self.tanh,
                coord_mean=coord_mean,
                coord_std=coord_std,
                box_size=box_size,
            )(graph, pos, edge_attribute=edge_attribute, node_attribute=node_attribute)
            graph = self.recompute_edges(graph, pos, coord_mean, coord_std, box_size)
        return pos

    def recompute_edges(self, graph, pos, coord_mean, coord_std, box_size):
        # TODO: Generalize to arbitrary k
        pos_unnormed = pos * coord_std + coord_mean
        sources, targets = nearest_neighbors(pos_unnormed, 20, box_size)
        graph._replace(senders=sources, receivers=targets)
        return graph
