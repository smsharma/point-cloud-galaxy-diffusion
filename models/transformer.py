import jax
import jax.numpy as jnp
from flax import linen as nn


class Transformer(nn.Module):
    """Simple decoder-only transformer for autoregressive sequence modeling"""

    n_input: int
    d_model: int = 256
    d_mlp: int = 1024
    max_len_seq: int = 30
    n_layers: int = 6
    n_heads: int = 4
    d_heads: int = 256 // 4
    p_dropout: float = 0.0
    use_mlp: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, conditioning: jnp.ndarray):

        # Input embedding
        x = nn.Dense(int(self.d_model))(x)  # (seq_len, d_model)
        conditioning = nn.Dense(int(self.d_model))(conditioning)  # (d_model,)

        # Transformer layers
        for _ in range(self.n_layers):

            x += conditioning[None, :]  # (seq_len, d_model)

            # LayerNorm each time residual stream is written onto
            x1 = nn.LayerNorm()(x)

            x_heads = []

            # Multi-head self-attention
            # Across-head computation can be vectorized
            for _ in range(self.n_heads):

                query = nn.Dense(self.d_heads)(x1)
                key = nn.Dense(self.d_heads)(x1)

                score = query @ key.T
                attn = jax.nn.softmax(self.d_heads**-0.5 * score, axis=1)

                value = nn.Dense(self.d_heads)(x1)

                self_attn = attn @ value

                x_heads.append(self_attn)

            # Concatenate attention from different heads
            x_heads = jnp.concatenate(x_heads, -1)

            # Output
            x_heads = nn.Dense(self.d_model)(x_heads)

            x += x_heads  # Write residual stream

            if self.use_mlp:

                # LayerNorm
                x2 = nn.LayerNorm()(x)

                # MLP
                x2 = nn.Dense(self.d_mlp)(x2)
                x2 = jax.nn.gelu(x2)
                x2 = nn.Dense(self.d_model)(x2)

            x += x2  # Write residual stream

        # Final LayerNorm
        x = nn.LayerNorm()(x)

        # Unembed into logits
        x = nn.Dense(self.n_input, kernel_init=jax.nn.initializers.zeros)(x)

        return x
