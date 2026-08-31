"""Rolling codebook occupancy meters for tokenizer training logs."""

from __future__ import annotations

from collections import deque

import torch


def _first_codebook_indices(indices: torch.Tensor) -> torch.Tensor:
    """Flatten the coarsest codebook assignments of one training step to CPU longs."""
    codes = indices[..., 0] if indices.ndim >= 3 else indices
    return codes.detach().reshape(-1).long().cpu()


def _perplexity(counts: torch.Tensor) -> float:
    """Exponentiated entropy of a non-negative assignment histogram."""
    total = counts.sum().clamp_min(1)
    probabilities = counts.float() / total
    occupied = probabilities[probabilities > 0]
    if occupied.numel() == 0:
        return 0.0
    entropy = -(occupied * occupied.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy))


class CodebookOccupancyMeter:
    """Track per-batch, rolling-window, and lifetime codebook usage.

    ``unique_codes_batch`` is the number of distinct codes in the current step. Window and
    total stats answer the question "how much of the vocabulary has actually been used?"
    across many steps, which a single batch of unique ids cannot.
    """

    def __init__(self, codebook_size: int, window: int = 2000) -> None:
        """Initialize rolling histograms for a codebook of size ``codebook_size``.

        Args:
            codebook_size: Number of entries in the coarsest codebook (``K``).
            window: Number of recent steps kept in the occupancy window.
        """
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self.codebook_size = int(codebook_size)
        self.window = int(window)
        self._window_histograms: deque[torch.Tensor] = deque()
        self._window_counts = torch.zeros(self.codebook_size, dtype=torch.long)
        self._lifetime_counts = torch.zeros(self.codebook_size, dtype=torch.long)

    def update(self, indices: torch.Tensor) -> dict[str, float]:
        """Ingest one step of code indices and return the occupancy scalars to log.

        Args:
            indices: Code ids of shape ``[B, N, num_codebooks]`` or any tensor whose first
                codebook (or the whole tensor if it is already 1-D/2-D ids) should be counted.

        Returns:
            Mapping with batch unique count, window/total occupied counts, usage fractions,
            and window perplexity.
        """
        flat = _first_codebook_indices(indices)
        unique_batch = int(torch.unique(flat).numel()) if flat.numel() else 0
        histogram = torch.bincount(flat, minlength=self.codebook_size)[: self.codebook_size]
        self._window_histograms.append(histogram)
        self._window_counts += histogram
        self._lifetime_counts += histogram
        if len(self._window_histograms) > self.window:
            self._window_counts -= self._window_histograms.popleft()
        occupied_window = int((self._window_counts > 0).sum())
        occupied_total = int((self._lifetime_counts > 0).sum())
        size = float(self.codebook_size)
        return {
            "unique_codes_batch": float(unique_batch),
            "unique_codes": float(unique_batch),
            "codebook_occupied_window": float(occupied_window),
            "codebook_usage_window": occupied_window / size,
            "codebook_occupied_total": float(occupied_total),
            "codebook_usage_total": occupied_total / size,
            "codebook_perplexity_window": _perplexity(self._window_counts),
        }
