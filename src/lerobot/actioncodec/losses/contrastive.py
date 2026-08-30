"""Symmetric InfoNCE losses used by optional tokenizer auxiliary training."""

import torch
import torch.nn.functional as F  # noqa: N812


def symmetric_info_nce(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Align two views one-to-one with a symmetric InfoNCE objective.

    Row ``i`` of ``x`` is treated as the single positive of row ``i`` of ``y``, and the loss is
    averaged over both retrieval directions so neither modality is privileged.

    Args:
        x: First view of shape ``[B, E]``; normalized internally.
        y: Second view of shape ``[B, E]``; normalized internally.
        temperature: Softmax temperature applied to the cosine similarity logits.

    Returns:
        Scalar tensor with the mean of the two cross-entropy directions.
    """
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    logits = (x @ y.t()) / temperature
    labels = torch.arange(x.shape[0], device=x.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def symmetric_info_nce_multi_positive(
    x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Align two views with a symmetric InfoNCE objective allowing several positives per row.

    Every pair of rows sharing a label counts as a positive, and each row's loss is the mean
    negative log-probability over its own positives, so rows with many positives do not dominate.

    Args:
        x: First view of shape ``[B, E]``; normalized internally.
        y: Second view of shape ``[B, E]``; normalized internally.
        labels: Integer tensor with ``B`` elements grouping rows into positive sets.
        temperature: Softmax temperature applied to the cosine similarity logits.

    Returns:
        Scalar tensor with the mean of the two retrieval directions.

    Raises:
        ValueError: If ``labels`` does not have one entry per row of ``x``.
    """
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    labels = labels.view(-1)
    if labels.shape[0] != x.shape[0]:
        raise ValueError("labels length does not match batch size")
    logits = (x @ y.t()) / temperature
    positive = (labels[:, None] == labels[None, :]).to(logits.dtype)
    log_xy = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_xy = -((log_xy * positive).sum(1) / positive.sum(1).clamp_min(1)).mean()
    log_yx = logits.t() - torch.logsumexp(logits.t(), dim=1, keepdim=True)
    loss_yx = -((log_yx * positive.t()).sum(1) / positive.sum(0).clamp_min(1)).mean()
    return 0.5 * (loss_xy + loss_yx)
