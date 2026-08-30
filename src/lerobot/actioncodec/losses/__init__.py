"""Semantic alignment losses for ActionCodec."""

from .contrastive import symmetric_info_nce, symmetric_info_nce_multi_positive
from .reduction import masked_mean
from .semantic_dtw import chunk_hard_dtw_targets, semantic_contrastive_loss
from .soft_dtw import (
    chunk_soft_dtw_targets,
    soft_dtw_distance_and_alignment,
    soft_dtw_distance_only,
    trajectory_soft_dtw_alignments,
)

__all__ = [
    "chunk_hard_dtw_targets",
    "semantic_contrastive_loss",
    "chunk_soft_dtw_targets",
    "soft_dtw_distance_and_alignment",
    "soft_dtw_distance_only",
    "trajectory_soft_dtw_alignments",
    "symmetric_info_nce",
    "symmetric_info_nce_multi_positive",
    "masked_mean",
]
