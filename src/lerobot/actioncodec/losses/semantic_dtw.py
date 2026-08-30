"""DTW pair mining and contrastive loss used for semantic alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SemanticPairMiningResult:
    """Positive and negative pairs mined from hard-DTW distances between action chunks.

    Attributes:
        distances: Symmetric ``[B, B]`` matrix of banded hard-DTW distances. The diagonal is ``0``
            and pairs that were never scored stay ``inf``.
        similarity: ``[B, B]`` matrix of ``exp(-distance / similarity_scale / temperature)`` on the
            scored pairs and ``0`` elsewhere. Negative mining thresholds on it; the losses
            themselves consume the masks instead.
        positive_mask: Boolean ``[B, B]`` mask keeping, for every row, the ``positive_topk``
            closest candidates of that row.
        negative_mask: Boolean ``[B, B]`` mask of up to ``negative_topk`` far-away candidates per
            row, disjoint from ``positive_mask``.
        candidate_mask: Boolean ``[B, B]`` mask of the pairs eligible for mining, i.e. both chunks
            move more than ``min_delta_norm`` and are not the same sample.
        similarity_scale: Scalar tensor with the median finite candidate distance, used to make
            the similarity scale invariant to the units of the state deltas.
        num_positive: Number of ``True`` entries in ``positive_mask``.
        num_negative: Number of ``True`` entries in ``negative_mask``.
        candidate_pairs: Number of ``True`` entries in ``candidate_mask``, counting both
            directions of each pair.
    """

    distances: torch.Tensor
    similarity: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    candidate_mask: torch.Tensor
    similarity_scale: torch.Tensor
    num_positive: int
    num_negative: int
    candidate_pairs: int


@dataclass(frozen=True)
class SemanticContrastiveLossResult:
    """Breakdown of the semantic contrastive loss.

    Attributes:
        total: Scalar tensor with the weighted sum of the two terms; this is what callers add to
            the training objective.
        positive: Scalar tensor with the unweighted attraction term averaged over positive pairs.
        negative: Scalar tensor with the unweighted margin repulsion term averaged over negative
            pairs.
    """

    total: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor


def banded_dtw_distance(first: torch.Tensor, second: torch.Tensor, *, band: int) -> torch.Tensor:
    """Return detached hard-DTW distance using a Sakoe-Chiba band."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[-1] != second.shape[-1]:
        raise ValueError("DTW inputs must have shape [T, D] with matching D")
    if first.shape[0] == 0 or second.shape[0] == 0 or band < 0:
        raise ValueError("DTW inputs must be non-empty and band must be non-negative")
    return _dtw(first.float(), second.float(), int(band)).detach()


def distance_to_similarity(
    distances: torch.Tensor,
    *,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a DTW distance matrix into a scale-free similarity matrix.

    Distances are divided by the median candidate distance before the exponential, so the
    similarity range does not depend on the units or the horizon of the underlying trajectories.

    Args:
        distances: Square ``[B, B]`` distance matrix; non-finite entries are treated as unscored.
        candidate_mask: Boolean ``[B, B]`` mask of the pairs that were actually scored.
        temperature: Positive sharpening factor applied on top of the median normalization.

    Returns:
        Tuple of the ``[B, B]`` similarity matrix, which is ``0`` off the candidate set and on the
        diagonal, and the detached scalar median used for normalization.

    Raises:
        ValueError: If ``temperature`` is not positive or the mask shape does not match
            ``distances``.
    """
    if temperature <= 0 or distances.shape != candidate_mask.shape:
        raise ValueError("distance matrix, candidate mask, and temperature are invalid")
    finite = candidate_mask & torch.isfinite(distances)
    values = distances[finite]
    scale = values.float().median().clamp_min(1e-6) if values.numel() else distances.new_tensor(1.0)
    similarity = distances.new_zeros(distances.shape, dtype=torch.float32)
    similarity[finite] = torch.exp(-distances[finite].float() / scale / temperature)
    similarity.fill_diagonal_(0)
    return similarity, scale.detach()


def _dtw(first: torch.Tensor, second: torch.Tensor, band: int) -> torch.Tensor:
    """Run the classic DTW recursion restricted to a Sakoe-Chiba band.

    Only cells within ``band`` steps of the diagonal are relaxed, which bounds how far the warping
    path may drift and keeps the cost quadratic in the band width rather than in the sequence
    length product.

    Args:
        first: Sequence of shape ``[T1, D]``.
        second: Sequence of shape ``[T2, D]``.
        band: Half-width of the Sakoe-Chiba band, in steps.

    Returns:
        Scalar tensor with the accumulated cost of the best warping path, or ``inf`` when the band
        is too narrow for any path to reach the final cell.
    """
    cost = (first[:, None] - second[None]).pow(2).sum(-1)
    rows, cols = cost.shape
    inf = cost.new_tensor(float("inf"))
    values = [[inf for _ in range(cols + 1)] for _ in range(rows + 1)]
    values[0][0] = cost.new_zeros(())
    for row in range(1, rows + 1):
        for col in range(max(1, row - band), min(cols, row + band) + 1):
            values[row][col] = cost[row - 1, col - 1] + min(
                values[row - 1][col], values[row][col - 1], values[row - 1][col - 1]
            )
    return values[rows][cols]


def chunk_hard_dtw_targets(
    delta_state: torch.Tensor,
    *,
    positive_topk: int = 4,
    negative_topk: int = 0,
    negative_quantile: float = 0.2,
    temperature: float = 1.0,
    band_frac: float = 0.2,
    max_candidate_pairs: int | None = None,
    pair_batch_size: int = 8192,
    min_delta_norm: float = 1e-6,
    normalize_delta: bool = True,
    **_: object,
) -> SemanticPairMiningResult:
    """Mine positive and negative chunk pairs inside a batch from banded hard-DTW distances.

    Chunks that barely move are dropped, the remaining pairs are scored with banded DTW on the
    (optionally standardized) state deltas, and every row keeps its closest partners as positives
    and, optionally, a tail of far-away partners as negatives.

    Args:
        delta_state: Per-step state deltas of shape ``[B, T, D]``.
        positive_topk: Number of closest partners kept as positives for each row.
        negative_topk: Number of far-away partners kept as negatives for each row; ``0`` disables
            negative mining.
        negative_quantile: Similarity quantile below which a candidate qualifies as a negative.
        temperature: Sharpening factor of the similarity used for the negative quantile test.
        band_frac: Sakoe-Chiba band half-width as a fraction of the chunk length ``T``.
        max_candidate_pairs: Cap on the number of unordered pairs to score. When the candidate set
            exceeds twice this value it is randomly subsampled down to it.
        pair_batch_size: Number of pairs processed per inner loop slice.
        min_delta_norm: Root-mean-square motion below which a chunk counts as static and is
            excluded from mining altogether.
        normalize_delta: Standardize the deltas using statistics of the moving chunks only, so
            that dimensions with a large native range do not dominate the distance.
        **_: Extra keyword arguments accepted for API compatibility and ignored.

    Returns:
        A :class:`SemanticPairMiningResult` with the ``[B, B]`` distance and similarity matrices
        plus the pair masks, all detached from the autograd graph.

    Raises:
        ValueError: If ``delta_state`` is not 3-D, or if the mining configuration is out of range.
    """
    if delta_state.ndim != 3:
        raise ValueError(f"delta_state must have shape [B, T, D], got {tuple(delta_state.shape)}")
    if positive_topk <= 0 or negative_topk < 0 or not 0 <= negative_quantile <= 1 or temperature <= 0:
        raise ValueError("invalid semantic pair mining configuration")
    batch = int(delta_state.shape[0])
    motion = delta_state.float().pow(2).mean((-1, -2)).sqrt()
    moving = motion > float(min_delta_norm)
    candidate = (
        moving[:, None] & moving[None, :] & ~torch.eye(batch, dtype=torch.bool, device=delta_state.device)
    )
    if max_candidate_pairs and int(candidate.sum()) > 2 * int(max_candidate_pairs):
        pairs = torch.triu(candidate, diagonal=1).nonzero(as_tuple=False)
        selected = pairs[torch.randperm(pairs.shape[0], device=pairs.device)[: int(max_candidate_pairs)]]
        candidate = torch.zeros_like(candidate)
        candidate[selected[:, 0], selected[:, 1]] = True
        candidate[selected[:, 1], selected[:, 0]] = True
    distances = delta_state.new_full((batch, batch), float("inf"), dtype=torch.float32)
    distances.fill_diagonal_(0)
    raw_delta_state = delta_state.float()
    moving_values = raw_delta_state[moving] if moving.any() else raw_delta_state
    mean = moving_values.mean((0, 1))
    std = moving_values.std((0, 1), unbiased=False).clamp_min(1e-6)
    normalized = (raw_delta_state - mean) / std if normalize_delta else raw_delta_state
    pairs = torch.triu(candidate, diagonal=1).nonzero(as_tuple=False).tolist()
    for start in range(0, len(pairs), max(1, int(pair_batch_size))):
        for first, second in pairs[start : start + max(1, int(pair_batch_size))]:
            value = _dtw(
                normalized[first],
                normalized[second],
                max(0, round(band_frac * delta_state.shape[1])),
            )
            distances[first, second] = value
            distances[second, first] = value
    finite = candidate & torch.isfinite(distances)
    distance_values = distances[finite]
    scale = (
        distance_values.float().median().clamp_min(1e-6)
        if distance_values.numel()
        else delta_state.new_tensor(1.0)
    )
    similarity_matrix = torch.zeros_like(distances)
    similarity_matrix[finite] = torch.exp(-distances[finite].float() / scale / temperature)
    positive = torch.zeros_like(candidate)
    negative = torch.zeros_like(candidate)
    for row in range(batch):
        choices = candidate[row].nonzero().flatten()
        if choices.numel() == 0:
            continue
        order = distances[row, choices].argsort()
        positive[row, choices[order[: min(positive_topk, choices.numel())]]] = True
        if negative_topk:
            similarity = similarity_matrix[row, choices]
            tail = choices[similarity <= torch.quantile(similarity, negative_quantile)]
            tail = tail[~positive[row, tail]]
            negative[row, tail[: min(negative_topk, tail.numel())]] = True
    return SemanticPairMiningResult(
        distances=distances.detach(),
        similarity=similarity_matrix.detach(),
        positive_mask=positive,
        negative_mask=negative,
        candidate_mask=candidate,
        similarity_scale=scale.detach(),
        num_positive=int(positive.sum()),
        num_negative=int(negative.sum()),
        candidate_pairs=int(candidate.sum()),
    )


def _masked_pair_squared_distances(embeddings: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    """Compute squared distances only for the selected directed pairs.

    This is the sparse counterpart of ``(embeddings[:, None] - embeddings[None]).pow(2).sum(-1)``.
    Restricting the computation to the mined pairs avoids materializing a ``[B, B, D]`` tensor, and
    it keeps the unselected entries — the zero-distance diagonal in particular — out of the autograd
    graph, where a later ``sqrt`` would otherwise backpropagate ``0 / 0``.

    Args:
        embeddings: Chunk embeddings of shape ``[B, D]``.
        pair_mask: Boolean ``[B, B]`` mask selecting the directed pairs to score.

    Returns:
        Squared distances of shape ``[num_selected_pairs]``, empty when the mask selects nothing.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or ``pair_mask`` is not a boolean ``[B, B]`` matrix.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [B, D], got {tuple(embeddings.shape)}")
    expected_shape = (int(embeddings.shape[0]), int(embeddings.shape[0]))
    if pair_mask.shape != expected_shape or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must be a boolean [B, B] matrix")
    pair_indices = pair_mask.nonzero(as_tuple=False)
    if pair_indices.numel() == 0:
        return embeddings.new_empty((0,))
    first = embeddings.index_select(0, pair_indices[:, 0])
    second = embeddings.index_select(0, pair_indices[:, 1])
    return (first - second).pow(2).sum(dim=-1)


def semantic_contrastive_loss(
    embeddings: torch.Tensor,
    *,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    positive_weight: float,
    negative_weight: float,
    negative_margin: float,
) -> SemanticContrastiveLossResult:
    """Pull mined positive pairs together and push mined negative pairs apart.

    Positives are penalized by their squared Euclidean distance, negatives by a hinge that is only
    active while they sit closer than ``negative_margin``. When a mask is empty its term collapses
    to a zero that still depends on ``embeddings``, so the loss stays differentiable.

    Args:
        embeddings: Chunk embeddings of shape ``[B, E]``, typically L2-normalized by the caller.
        positive_mask: Boolean ``[B, B]`` mask of pairs that should be close.
        negative_mask: Boolean ``[B, B]`` mask of pairs that should be at least
            ``negative_margin`` apart.
        positive_weight: Weight of the attraction term in the total.
        negative_weight: Weight of the repulsion term in the total.
        negative_margin: Distance beyond which a negative pair stops being penalized.

    Returns:
        A :class:`SemanticContrastiveLossResult` whose ``total`` is a scalar suitable for
        backpropagation, alongside the two unweighted terms.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or the masks are not ``[B, B]``.
    """
    if embeddings.ndim != 2 or positive_mask.shape != (embeddings.shape[0], embeddings.shape[0]):
        raise ValueError("embeddings and pair masks have incompatible shapes")
    zero = embeddings.sum() * 0.0
    if positive_mask.any():
        positive = _masked_pair_squared_distances(embeddings, positive_mask).mean()
    else:
        positive = zero
    if negative_mask.any():
        distance = _masked_pair_squared_distances(embeddings, negative_mask).clamp_min(0.0).sqrt()
        negative = torch.relu(float(negative_margin) - distance).pow(2).mean()
    else:
        negative = zero

    return SemanticContrastiveLossResult(
        float(positive_weight) * positive + float(negative_weight) * negative, positive, negative
    )
