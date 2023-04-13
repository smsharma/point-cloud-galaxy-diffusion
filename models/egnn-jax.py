from typing import Any, Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
from jax.tree_util import Partial


class EGNNLayer(hk.Module):
    """EGNN layer."""

    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        output_size: int,
        blocks: int = 1,
        act_fn: Callable = jax.nn.silu,
        pos_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        msg_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(f"layer_{layer_num}")

        # message network
        self._edge_mlp = hk.nets.MLP(
            [hidden_size] * blocks + [hidden_size],
            activation=act_fn,
            activate_final=True,
        )

        # update network
        self._node_mlp = hk.nets.MLP(
            [hidden_size] * blocks + [output_size],
            activation=act_fn,
            activate_final=False,
        )

        # position update network
        net = [hk.Linear(hidden_size)] * blocks
        # NOTE: from https://github.com/vgsatorras/egnn/blob/main/models/gcl.py#L254
        a = 0.001 * jnp.sqrt(6 / hidden_size)
        net += [
            act_fn,
            hk.Linear(1, with_bias=False, w_init=hk.initializers.TruncatedNormal(a)),
        ]
        if tanh:
            net.append(jax.nn.tanh)
        self._pos_mlp = hk.Sequential(net)

        # attention
        self._attention_mlp = None
        if attention:
            self._attention_mlp = hk.Sequential([hk.Linear(hidden_size), jax.nn.sigmoid])

        self.pos_aggregate_fn = pos_aggregate_fn
        self.msg_aggregate_fn = msg_aggregate_fn
        self._residual = residual
        self._normalize = normalize
        self._eps = eps

    def _pos_update(
        self,
        pos: jnp.ndarray,
        graph: jraph.GraphsTuple,
        coord_diff: jnp.ndarray,
    ) -> jnp.ndarray:
        trans = coord_diff * self._pos_mlp(graph.edges)
        # NOTE: was in the original code
        trans = jnp.clip(trans, -100, 100)
        return self.pos_aggregate_fn(trans, graph.senders, num_segments=pos.shape[0])

    def _message(
        self,
        radial: jnp.ndarray,
        edge_attribute: jnp.ndarray,
        edge_features: Any,
        incoming: jnp.ndarray,
        outgoing: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = edge_features
        _ = globals_
        msg = jnp.concatenate([incoming, outgoing, radial], axis=-1)
        if edge_attribute is not None:
            msg = jnp.concatenate([msg, edge_attribute], axis=-1)
        msg = self._edge_mlp(msg)
        if self._attention_mlp:
            att = self._attention_mlp(msg)
            msg = msg * att
        return msg

    def _update(
        self,
        node_attribute: jnp.ndarray,
        nodes: jnp.ndarray,
        senders: Any,
        msg: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = senders
        _ = globals_
        x = jnp.concatenate([nodes, msg], axis=-1)
        if node_attribute is not None:
            x = jnp.concatenate([x, node_attribute], axis=-1)
        x = self._node_mlp(x)
        if self._residual:
            x = nodes + x
        return x

    def _coord2radial(self, graph: jraph.GraphsTuple, coord: jnp.array) -> Tuple[jnp.array, jnp.array]:
        coord_diff = coord[graph.senders] - coord[graph.receivers]
        radial = jnp.sum(coord_diff**2, 1)[:, jnp.newaxis]
        if self._normalize:
            norm = jnp.sqrt(radial)
            coord_diff = coord_diff / (norm + self._eps)
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
        radial, coord_diff = self._coord2radial(graph, pos)

        graph = jraph.GraphNetwork(
            update_edge_fn=Partial(self._message, radial, edge_attribute),
            update_node_fn=Partial(self._update, node_attribute),
            aggregate_edges_for_nodes_fn=self.msg_aggregate_fn,
        )(graph)

        pos = pos + self._pos_update(pos, graph, coord_diff)

        return graph, pos


class EGNN(hk.Module):
    r"""
    E(n) Graph Neural Network (https://arxiv.org/abs/2102.09844).

    Original implementation: https://github.com/vgsatorras/egnn
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        act_fn: Callable = jax.nn.silu,
        num_layers: int = 4,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
    ):
        r"""
        Initialize the network.

        Args:
            hidden_size: Number of hidden features
            output_size: Number of features for 'h' at the output
            act_fn: Non-linearity
            num_layers: Number of layer for the EGNN
            residual: Use residual connections, we recommend not changing this one
            attention: Whether using attention or not
            normalize: Normalizes the coordinates messages such that:
                instead of: x^{l+1}_i = x^{l}_i + \sum(x_i - x_j)\phi_x(m_{ij})
                use:        x^{l+1}_i = x^{l}_i + \sum(x_i - x_j)\phi_x(m_{ij})\|x_i - x_j\|
                It may help in the stability or generalization. Not used in the paper.
            tanh: Sets a tanh activation function at the output of \phi_x(m_{ij}). It
                bounds the output of \phi_x(m_{ij}) which definitely improves in
                stability but it may decrease in accuracy. Not used in the paper.
        """
        super().__init__()

        self._hidden_size = hidden_size
        self._output_size = output_size
        self._act_fn = act_fn
        self._num_layers = num_layers
        self._residual = residual
        self._attention = attention
        self._normalize = normalize
        self._tanh = tanh

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
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
        # input node embedding
        h = hk.Linear(self._hidden_size, name="embedding")(graph.nodes)
        graph = graph._replace(nodes=h)
        # message passing
        for n in range(self._num_layers):
            graph, pos = EGNNLayer(
                layer_num=n,
                hidden_size=self._hidden_size,
                output_size=self._hidden_size,
                act_fn=self._act_fn,
                residual=self._residual,
                attention=self._attention,
                normalize=self._normalize,
                tanh=self._tanh,
            )(graph, pos, edge_attribute=edge_attribute, node_attribute=node_attribute)
        # node readout
        h = hk.Linear(self._output_size, name="readout")(graph.nodes)
        return h, pos
