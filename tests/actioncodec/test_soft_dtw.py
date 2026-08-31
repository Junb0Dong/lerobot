from __future__ import annotations

import pytest
import torch

from lerobot.actioncodec.losses.soft_dtw import (
    _soft_dtw_distance,
    _soft_dtw_distance_batch,
    _step_cost,
    chunk_soft_dtw_targets,
)


def test_soft_dtw_distance_batch_matches_scalar_pairs():
    torch.manual_seed(0)
    first = torch.randn(5, 8, 4)
    second = torch.randn(5, 8, 4)
    cost = (first[:, :, None] - second[:, None, :]).pow(2).mean(-1)
    batched = _soft_dtw_distance_batch(cost, gamma=0.1)
    expected = torch.stack([_soft_dtw_distance(cost[index], gamma=0.1) for index in range(5)])
    torch.testing.assert_close(batched, expected, atol=1e-5, rtol=1e-5)


def test_chunk_soft_dtw_targets_match_scalar_reference():
    torch.manual_seed(1)
    delta_state = torch.randn(6, 20, 14)
    result = chunk_soft_dtw_targets(
        delta_state,
        positive_topk=1,
        gamma=0.1,
        pair_batch_size=2,
        dtw_backend="torch",
        normalize_delta=False,
    )
    expected = delta_state.new_full((6, 6), float("inf"))
    expected.fill_diagonal_(0)
    pairs = torch.triu(result.candidate_mask, diagonal=1).nonzero(as_tuple=False)
    for first, second in pairs.tolist():
        distance = _soft_dtw_distance(_step_cost(delta_state[first], delta_state[second]), gamma=0.1)
        expected[first, second] = distance
        expected[second, first] = distance
    torch.testing.assert_close(result.distances, expected, atol=1e-5, rtol=1e-5)
    assert result.distances.device == delta_state.device


def test_chunk_soft_dtw_cuda_backend_requires_extension():
    delta_state = torch.randn(3, 4, 2)
    with pytest.raises(RuntimeError, match="softdtw-cuda"):
        chunk_soft_dtw_targets(delta_state, dtw_backend="cuda", normalize_delta=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the GPU parity check")
def test_chunk_soft_dtw_targets_match_scalar_reference_on_cuda():
    torch.manual_seed(2)
    delta_state = torch.randn(4, 20, 14, device="cuda")
    result = chunk_soft_dtw_targets(
        delta_state,
        positive_topk=1,
        gamma=0.1,
        pair_batch_size=3,
        dtw_backend="auto",
        normalize_delta=False,
    )
    expected = delta_state.new_full((4, 4), float("inf"))
    expected.fill_diagonal_(0)
    pairs = torch.triu(result.candidate_mask, diagonal=1).nonzero(as_tuple=False)
    for first, second in pairs.tolist():
        distance = _soft_dtw_distance(_step_cost(delta_state[first], delta_state[second]), gamma=0.1)
        expected[first, second] = distance
        expected[second, first] = distance
    torch.testing.assert_close(result.distances, expected, atol=1e-5, rtol=1e-5)
    assert result.distances.device.type == "cuda"
