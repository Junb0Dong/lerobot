"""Masked reductions shared by ActionCodec objectives."""

import torch


def masked_mean(elementwise_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average an elementwise loss over the entries selected by a mask.

    The mask is moved onto the loss device and dtype and broadcast to its shape, so a per-sample
    or per-timestep mask can weight a full ``[B, T, D]`` loss tensor. An all-zero mask yields
    ``0`` instead of a division by zero.

    Args:
        elementwise_loss: Unreduced loss tensor of any shape.
        mask: Mask broadcastable to the shape of ``elementwise_loss``; non-boolean masks act as
            per-element weights.

    Returns:
        Scalar tensor with the masked mean.
    """
    expanded = mask.to(device=elementwise_loss.device, dtype=elementwise_loss.dtype).expand_as(
        elementwise_loss
    )
    return (elementwise_loss * expanded).sum() / expanded.sum().clamp_min(1)
