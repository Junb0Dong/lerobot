"""Torch Soft-DTW distances and detached alignments used by tokenizer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DTWBackend = Literal["auto", "torch", "cuda"]


@dataclass
class ChunkSoftDTWResult:
    """Soft-DTW mining results for a batch of independent action chunks.

    Attributes:
        distances: Symmetric ``[B, B]`` matrix of soft-DTW distances. The diagonal is ``0`` and
            pairs that were never scored stay ``inf``.
        positive_mask: Boolean ``[B, B]`` mask keeping, for every row, the ``positive_topk``
            closest candidates of that row.
        candidate_mask: Boolean ``[B, B]`` mask of the pairs eligible for mining, i.e. both chunks
            move more than ``min_delta_norm`` and are not the same sample.
        distance_threshold: Scalar tensor holding the ``positive_topk``-th smallest distance over
            all finite candidate pairs. It is reported for logging only; ``positive_mask`` is
            built per row rather than by thresholding.
        num_positive: Number of ``True`` entries in ``positive_mask``.
        candidate_pairs: Number of ``True`` entries in ``candidate_mask``, counting both
            directions of each pair.
    """

    distances: torch.Tensor
    positive_mask: torch.Tensor
    candidate_mask: torch.Tensor
    distance_threshold: torch.Tensor
    num_positive: int
    candidate_pairs: int


@dataclass
class TrajectorySoftDTWResult:
    """Soft-DTW mining results for whole trajectories made of several chunks.

    Attributes:
        distances: Symmetric ``[B, B]`` matrix of soft-DTW distances between trajectories. The
            diagonal is ``0`` and pairs that were never scored stay ``inf``.
        alignments: ``[B, B, M, M]`` soft alignment matrices over the chunk grid, zero-padded past
            the number of valid chunks. ``alignments[i, j]`` sums to one and ``alignments[j, i]``
            is its transpose.
        positive_mask: Boolean ``[B, B]`` mask of candidate pairs whose distance is at or below
            ``distance_threshold``. It is upper-triangular because ``candidate_mask`` is.
        candidate_mask: Boolean upper-triangular ``[B, B]`` mask of eligible trajectory pairs.
        distance_threshold: Scalar tensor with the ``positive_topk``-th smallest finite candidate
            distance, or ``inf`` when no pair could be scored.
        num_positive: Number of ``True`` entries in ``positive_mask``.
        candidate_pairs: Number of ``True`` entries in ``candidate_mask``.
    """

    distances: torch.Tensor
    alignments: torch.Tensor
    positive_mask: torch.Tensor
    candidate_mask: torch.Tensor
    distance_threshold: torch.Tensor
    num_positive: int
    candidate_pairs: int


def _softmin3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute the ``gamma``-smoothed minimum of three cumulative costs.

    Args:
        a: First candidate cumulative cost.
        b: Second candidate cumulative cost.
        c: Third candidate cumulative cost.
        gamma: Smoothing temperature; as it goes to zero the result approaches a hard ``min``.

    Returns:
        Scalar tensor equal to ``-gamma * logsumexp([a, b, c] / -gamma)``.
    """
    return -gamma * torch.logsumexp(torch.stack((a, b, c)) / -gamma, dim=0)


def _soft_dtw_distance(cost: torch.Tensor, gamma: float) -> torch.Tensor:
    """Run the soft-DTW forward recursion over a precomputed step-cost matrix.

    Implements ``r[i, j] = cost[i, j] + softmin(r[i-1, j], r[i, j-1], r[i-1, j-1])`` on a table
    padded with ``inf`` except at the origin, so every warping path must start at ``(0, 0)`` and
    end at ``(N-1, M-1)``.

    Args:
        cost: Pairwise step cost of shape ``[N, M]``.
        gamma: Smoothing temperature of the soft-minimum; must be positive.

    Returns:
        Scalar tensor with the soft-DTW distance, differentiable with respect to ``cost``.

    Raises:
        ValueError: If ``cost`` is not 2-D or ``gamma`` is not positive.
    """
    if cost.ndim != 2 or gamma <= 0:
        raise ValueError("cost must be [N, M] and gamma must be positive")
    rows, columns = cost.shape
    infinity = cost.new_tensor(float("inf"))
    table = [[infinity for _ in range(columns + 1)] for _ in range(rows + 1)]
    table[0][0] = cost.new_zeros(())
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            table[row][column] = cost[row - 1, column - 1] + _softmin3(
                table[row - 1][column],
                table[row][column - 1],
                table[row - 1][column - 1],
                gamma,
            )
    return table[rows][columns]


def _step_cost(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Build the pairwise step-cost matrix between two time series.

    Args:
        first: Sequence of shape ``[T1, D]``.
        second: Sequence of shape ``[T2, D]``.

    Returns:
        Cost matrix of shape ``[T1, T2]`` holding the squared difference averaged over the feature
        axis, so the cost scale is independent of ``D``.

    Raises:
        ValueError: If either input is not 2-D or the two feature dimensions differ.
    """
    if first.ndim != 2 or second.ndim != 2 or first.shape[-1] != second.shape[-1]:
        raise ValueError("DTW trajectories must have shape [T, D] with matching D")
    return (first[:, None] - second[None]).pow(2).mean(-1)


def _chunk_cost(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Build the pairwise cost matrix between two sequences of action chunks.

    This is the trajectory-level counterpart of :func:`_step_cost`: a warping step moves between
    whole chunks rather than single timesteps, so the squared difference is averaged over both the
    intra-chunk time axis and the feature axis.

    Args:
        first: Chunk sequence of shape ``[M1, T, D]``.
        second: Chunk sequence of shape ``[M2, T, D]``.

    Returns:
        Cost matrix of shape ``[M1, M2]``.

    Raises:
        ValueError: If either input is not 3-D or the two ``[T, D]`` inner shapes differ.
    """
    if first.ndim != 3 or second.ndim != 3 or first.shape[1:] != second.shape[1:]:
        raise ValueError("DTW chunk sequences must have shape [M, T, D] with matching [T, D]")
    return (first[:, None] - second[None]).pow(2).mean(dim=(-1, -2))


def _pairwise_cost(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Dispatch to the step-level or chunk-level cost depending on the input rank.

    Args:
        first: Either a ``[T, D]`` timestep sequence or an ``[M, T, D]`` chunk sequence.
        second: Sequence with the same rank as ``first``.

    Returns:
        Cost matrix whose two axes index the warping units of ``first`` and ``second``.

    Raises:
        ValueError: If the two inputs have different ranks, or a rank other than 2 or 3.
    """
    if first.ndim != second.ndim:
        raise ValueError(f"DTW inputs must have the same rank, got {first.ndim} and {second.ndim}")
    if first.ndim == 3:
        return _chunk_cost(first, second)
    return _step_cost(first, second)


def soft_dtw_distance_and_alignment(
    first: torch.Tensor, second: torch.Tensor, gamma: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score two sequences with soft-DTW and recover the soft alignment between their steps.

    The alignment is the gradient of the soft-DTW distance with respect to the step-cost matrix,
    i.e. the expected occupancy of each ``(i, j)`` cell under the Gibbs distribution over warping
    paths. Both inputs are detached and the returned tensors carry no autograd history: this
    produces alignment targets for other objectives, not a trainable loss term.

    Args:
        first: Either a ``[T1, D]`` timestep sequence or an ``[M1, T, D]`` chunk sequence.
        second: Sequence with the same rank as ``first``.
        gamma: Soft-minimum temperature. Smaller values concentrate the alignment on the single
            best warping path; larger values spread it over more paths.

    Returns:
        Tuple of a detached scalar distance and a detached alignment matrix over the warping units
        of the two inputs, clamped to be non-negative and renormalized to sum to one.
    """
    with torch.enable_grad():
        cost = _pairwise_cost(first.detach().float(), second.detach().float()).requires_grad_(True)
        distance = _soft_dtw_distance(cost, gamma)
        alignment = torch.autograd.grad(distance, cost)[0]
    alignment = alignment.clamp_min(0)
    return distance.detach(), (alignment / alignment.sum().clamp_min(1e-12)).detach()


def soft_dtw_distance_only(first: torch.Tensor, second: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Score two sequences with soft-DTW without recovering an alignment.

    Args:
        first: Either a ``[T1, D]`` timestep sequence or an ``[M1, T, D]`` chunk sequence.
        second: Sequence with the same rank as ``first``.
        gamma: Soft-minimum temperature; must be positive.

    Returns:
        Detached scalar tensor with the soft-DTW distance.
    """
    with torch.no_grad():
        return _soft_dtw_distance(_pairwise_cost(first.float(), second.float()), gamma).detach()


def chunk_soft_dtw_targets(
    delta_state: torch.Tensor,
    positive_topk: int = 4,
    gamma: float = 0.1,
    max_candidate_pairs: int | None = None,
    pair_batch_size: int = 8192,
    dtw_backend: DTWBackend = "auto",
    min_delta_norm: float = 1e-6,
    normalize_delta: bool = True,
) -> ChunkSoftDTWResult:
    """Mine positive chunk pairs inside a batch from soft-DTW distances on state deltas.

    Every unordered pair of chunks that moves more than ``min_delta_norm`` is scored with soft-DTW
    on the (optionally standardized) state deltas, and each row keeps its ``positive_topk`` closest
    partners. The resulting masks are meant to be fed to a contrastive loss over chunk embeddings,
    so that chunks describing similar motion are pulled together in latent space.

    Args:
        delta_state: Per-step state deltas of shape ``[B, T, D]``.
        positive_topk: Number of closest partners kept as positives for each row.
        gamma: Soft-minimum temperature forwarded to soft-DTW.
        max_candidate_pairs: Cap on the number of unordered pairs to score. When the candidate set
            exceeds twice this value it is randomly subsampled down to it.
        pair_batch_size: Accepted for API compatibility with the batched backends and ignored.
        dtw_backend: Accepted for API compatibility and ignored; scoring always runs in torch.
        min_delta_norm: Root-mean-square motion below which a chunk counts as static and is
            excluded from mining altogether.
        normalize_delta: Standardize the deltas before scoring, so that dimensions with a large
            native range do not dominate the distance. The statistics come from the moving chunks
            only, so that a batch full of near-static chunks does not deflate the scale.

    Returns:
        A :class:`ChunkSoftDTWResult` holding the ``[B, B]`` distance matrix and the pair masks,
        all detached from the autograd graph.

    Raises:
        ValueError: If ``delta_state`` is not 3-D, or if ``positive_topk`` or ``gamma`` are not
            positive.
    """
    del pair_batch_size, dtw_backend
    if delta_state.ndim != 3:
        raise ValueError("delta_state must have shape [B, T, D]")
    if positive_topk <= 0 or gamma <= 0:
        raise ValueError("positive_topk and gamma must be positive")
    device = delta_state.device
    batch = int(delta_state.shape[0])
    raw = delta_state.detach().float()
    moving = raw.pow(2).mean((-1, -2)).sqrt() > min_delta_norm
    candidate = moving[:, None] & moving[None, :] & ~torch.eye(batch, dtype=torch.bool, device=device)
    if max_candidate_pairs and int(candidate.sum()) > 2 * max_candidate_pairs:
        pairs = torch.triu(candidate, diagonal=1).nonzero(as_tuple=False)
        pairs = pairs[torch.randperm(len(pairs), device=device)[:max_candidate_pairs]]
        candidate = torch.zeros_like(candidate)
        candidate[pairs[:, 0], pairs[:, 1]] = True
        candidate[pairs[:, 1], pairs[:, 0]] = True
    moving_values = raw[moving] if moving.any() else raw
    standardized = (raw - moving_values.mean((0, 1))) / moving_values.std((0, 1), unbiased=False).clamp_min(
        1e-6
    )
    values = standardized if normalize_delta else raw
    distances = raw.new_full((batch, batch), float("inf"), dtype=torch.float32)
    distances.fill_diagonal_(0)
    for first, second in torch.triu(candidate, diagonal=1).nonzero(as_tuple=False).tolist():
        distance = soft_dtw_distance_only(values[first], values[second], gamma)
        distances[first, second] = distance
        distances[second, first] = distance
    positive = torch.zeros_like(candidate)
    selected = distances[candidate & torch.isfinite(distances)]
    threshold = (
        selected.new_tensor(float("inf"))
        if selected.numel() == 0
        else selected.topk(min(positive_topk, selected.numel()), largest=False).values.max()
    )
    for row in range(batch):
        choices = candidate[row].nonzero(as_tuple=False).flatten()
        if choices.numel():
            keep = min(positive_topk, choices.numel())
            positive[row, choices[distances[row, choices].topk(keep, largest=False).indices]] = True
    return ChunkSoftDTWResult(
        distances=distances.detach(),
        positive_mask=positive,
        candidate_mask=candidate,
        distance_threshold=threshold.detach(),
        num_positive=int(positive.sum()),
        candidate_pairs=int(candidate.sum()),
    )


def trajectory_soft_dtw_alignments(
    delta_state: torch.Tensor,
    chunk_valid_mask: torch.Tensor,
    gamma: float = 0.1,
    positive_topk: int = 1,
    task_uid: torch.Tensor | None = None,
    same_task_only: bool = False,
    max_candidate_pairs: int | None = None,
    max_chunks_per_traj: int | None = None,
    dtw_backend: DTWBackend = "auto",
) -> TrajectorySoftDTWResult:
    """Align whole trajectories against each other at the chunk level with soft-DTW.

    Each trajectory is reduced to its valid chunks and warped against every other candidate
    trajectory, so the returned alignment matrices say which chunk of trajectory ``i`` corresponds
    to which chunk of trajectory ``j`` even when the two were executed at different speeds.
    Positives are selected globally: the ``positive_topk`` smallest distances set a threshold and
    every candidate at or below it is kept.

    Args:
        delta_state: Per-chunk state deltas of shape ``[B, M, T, D]``, where ``M`` is the number of
            chunk slots per trajectory and ``T`` the number of steps per chunk.
        chunk_valid_mask: Boolean mask of shape ``[B, M]`` marking which chunk slots are filled.
        gamma: Soft-minimum temperature forwarded to soft-DTW.
        positive_topk: Number of globally closest candidate pairs used to set the distance
            threshold.
        task_uid: Per-trajectory task identifier of shape ``[B]``; required when
            ``same_task_only`` is set.
        same_task_only: Restrict candidates to trajectory pairs that share a task identifier.
        max_candidate_pairs: Cap on the number of pairs actually scored; the candidate list is
            randomly subsampled beyond it.
        max_chunks_per_traj: Accepted for API compatibility and ignored.
        dtw_backend: Accepted for API compatibility and ignored; scoring always runs in torch.

    Returns:
        A :class:`TrajectorySoftDTWResult` with the ``[B, B]`` distance matrix, the
        ``[B, B, M, M]`` alignment tensor and the pair masks, all detached.

    Raises:
        ValueError: If ``delta_state`` is not 4-D, if ``chunk_valid_mask`` does not match its
            leading two dimensions, or if ``same_task_only`` is set without ``task_uid``.
    """
    del max_chunks_per_traj, dtw_backend
    if delta_state.ndim != 4 or chunk_valid_mask.shape != delta_state.shape[:2]:
        raise ValueError("delta_state must be [B, M, T, D] with a matching validity mask")
    batch, chunks = delta_state.shape[:2]
    valid_counts = chunk_valid_mask.bool().sum(1)
    candidate = valid_counts[:, None].gt(0) & valid_counts[None, :].gt(0)
    candidate &= torch.triu(torch.ones_like(candidate), diagonal=1)
    if same_task_only:
        if task_uid is None:
            raise ValueError("task_uid is required when same_task_only=True")
        candidate &= task_uid[:, None] == task_uid[None, :]
    distances = delta_state.new_full((batch, batch), float("inf"), dtype=torch.float32)
    alignments = delta_state.new_zeros((batch, batch, chunks, chunks), dtype=torch.float32)
    distances.fill_diagonal_(0)
    pairs = candidate.nonzero(as_tuple=False)
    if max_candidate_pairs and len(pairs) > max_candidate_pairs:
        pairs = pairs[torch.randperm(len(pairs), device=pairs.device)[:max_candidate_pairs]]
    for first, second in pairs.tolist():
        a = delta_state[first, chunk_valid_mask[first].bool()]
        b = delta_state[second, chunk_valid_mask[second].bool()]
        distances[first, second], alignment = soft_dtw_distance_and_alignment(a, b, gamma)
        distances[second, first] = distances[first, second]
        alignments[first, second, : alignment.shape[0], : alignment.shape[1]] = alignment
        alignments[second, first, : alignment.shape[1], : alignment.shape[0]] = alignment.t()
    positive = torch.zeros_like(candidate)
    finite = candidate & torch.isfinite(distances)
    values = distances[finite]
    if values.numel():
        threshold = values.topk(min(positive_topk, values.numel()), largest=False).values.max()
        positive = finite & (distances <= threshold)
    else:
        threshold = delta_state.new_tensor(float("inf"))
    return TrajectorySoftDTWResult(
        distances=distances.detach(),
        alignments=alignments.detach(),
        positive_mask=positive,
        candidate_mask=candidate,
        distance_threshold=threshold.detach(),
        num_positive=int(positive.sum()),
        candidate_pairs=int(candidate.sum()),
    )
