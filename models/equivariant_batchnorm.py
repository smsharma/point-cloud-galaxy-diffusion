from functools import partial
from math import prod

import jax
import jax.numpy as jnp
import flax.linen as nn

from e3nn_jax import Irreps, IrrepsArray, config


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11))
def _batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    normalization,
    reduce,
    is_training,
    is_instance,
    has_affine,
    momentum,
    epsilon,
):
    def _roll_avg(curr, update):
        return (1 - momentum) * curr + momentum * jax.lax.stop_gradient(update)

    batch, *size = input.shape[:-1]
    # TODO add test case for when prod(size) == 0

    input = input.reshape((batch, prod(size), -1))

    new_means = []
    new_vars = []

    fields = []

    i_wei = 0  # index for running_var and weight
    i_rmu = 0  # index for running_mean
    i_bia = 0  # index for bias

    for (mul, ir), field in zip(input.irreps, input.list):
        if field is None:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_training or is_instance:
                    if not is_instance:
                        new_means.append(jnp.zeros((mul,), dtype=input.dtype))
                i_rmu += mul

            if is_training or is_instance:
                if not is_instance:
                    new_vars.append(jnp.ones((mul,), input.dtype))

            if has_affine and ir.is_scalar():  # scalars
                i_bia += mul

            fields.append(field)  # [batch, sample, mul, repr]
        else:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_training or is_instance:
                    if is_instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(_roll_avg(running_mean[i_rmu : i_rmu + mul], field_mean))
                else:
                    field_mean = running_mean[i_rmu : i_rmu + mul]
                i_rmu += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if is_training or is_instance:
                if normalization == "norm":
                    field_norm = jnp.square(field).sum(3)  # [batch, sample, mul]
                elif normalization == "component":
                    field_norm = jnp.square(field).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(normalization))

                if reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif reduce == "max":
                    field_norm = field_norm.max(1)  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(reduce))

                if not is_instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(_roll_avg(running_var[i_wei : i_wei + mul], field_norm))
            else:
                field_norm = running_var[i_wei : i_wei + mul]

            field_norm = jax.lax.rsqrt((1 - epsilon) * field_norm + epsilon)  # [(batch,) mul]

            if has_affine:
                sub_weight = weight[i_wei : i_wei + mul]  # [mul]
                field_norm = field_norm * sub_weight  # [(batch,) mul]

            # TODO add test case for when mul == 0
            field_norm = field_norm[..., None, :, None]  # [(batch,) 1, mul, 1]
            field = field * field_norm  # [batch, sample, mul, repr]

            if has_affine and ir.is_scalar():  # scalars
                sub_bias = bias[i_bia : i_bia + mul]  # [mul]
                field += sub_bias.reshape(mul, 1)  # [batch, sample, mul, repr]
                i_bia += mul

            fields.append(field)  # [batch, sample, mul, repr]
        i_wei += mul

    output = IrrepsArray.from_list(input.irreps, fields, (batch, prod(size)), input.dtype)
    output = output.reshape((batch,) + tuple(size) + (-1,))
    return output, new_means, new_vars


class EquivariantBatchNorm(nn.Module):
    """Equivariant Batch Normalization.
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations are orthonormal.
    Args:
        irreps: Irreducible representations of the input and output (unchanged)
        eps (float): epsilon for numerical stability, has to be between 0 and 1.
            the field norm is transformed to ``(1 - eps) * norm + eps``
            leading to a slower convergence toward norm 1.
        momentum: momentum for moving average
        affine: whether to include learnable biases
        reduce: reduce mode, either 'mean' or 'max'
        instance: whether to use instance normalization
        normalization: normalization mode, either 'norm' or 'component'
    """

    irreps: Irreps = None
    eps: float = 1e-4
    momentum: float = 0.1
    affine: bool = False
    reduce: str = "mean"
    instance: bool = True
    normalization: str = "norm"

    @nn.compact
    def __call__(self, input: IrrepsArray, is_training: bool = True) -> IrrepsArray:
        r"""Evaluate the batch normalization.
        Args:
            input: input tensor of shape ``(batch, [spatial], irreps.dim)``
            is_training: whether to train or evaluate
        Returns:
            output: normalized tensor of shape ``(batch, [spatial], irreps.dim)``
        """
        if self.irreps is not None:
            input = input._convert(self.irreps)

        output, _, _ = _batch_norm(
            input,
            running_mean=None,
            running_var=None,
            weight=None,
            bias=None,
            normalization=self.normalization,
            reduce=self.reduce,
            is_training=is_training,
            is_instance=self.instance,
            has_affine=self.affine,
            momentum=self.momentum,
            epsilon=self.eps,
        )

        return output
