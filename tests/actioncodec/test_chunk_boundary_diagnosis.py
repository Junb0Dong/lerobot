"""Protect the metric from norm, temporal alignment and episode-mixing mistakes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnose_xlerobot_chunk_boundary.py"
spec = importlib.util.spec_from_file_location("boundary_diagnosis", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_execution_seam_uses_last_executed_action_and_l2():
    chunks = np.zeros((2, 20, 2))
    chunks[0, 15] = [1, 2]
    chunks[0, 19] = [100, 100]  # Unexecuted tail must not be used at stride 16.
    chunks[1, 0] = [4, 6]
    delta, valid = module.boundary_vectors(chunks, [0, 0], [0, 16], 16)
    np.testing.assert_equal(valid, [True])
    np.testing.assert_allclose(np.linalg.norm(delta, axis=-1), [5])
    full, _ = module.boundary_vectors(chunks, [0, 0], [0, 20], 20)
    np.testing.assert_allclose(full, [[-96, -94]])


def test_contiguous_gt_tiles_equal_actual_trajectory_step():
    trajectory = np.arange(100)[:, None] * np.array([[3, 4]])
    for stride in (16, 20):
        starts = np.arange(0, 81, stride)
        chunks = np.stack([trajectory[s : s + 20] for s in starts])
        delta, _ = module.boundary_vectors(chunks, np.zeros(len(starts)), starts, stride)
        np.testing.assert_allclose(np.linalg.norm(delta, axis=-1), 5)


def test_skip_episode_resets_and_nonadjacent_chunks():
    chunks = np.zeros((4, 20, 2))
    chunks[2:] = 1000
    delta, valid = module.boundary_vectors(chunks, [0, 0, 1, 1], [0, 16, 0, 32], 16)
    np.testing.assert_equal(valid, [True, False, False])
    np.testing.assert_equal(delta, [[0, 0]])


def test_normalized_l2_uses_per_dimension_scale():
    chunks = np.zeros((2, 20, 2))
    chunks[1, :, :] = [3, 4]
    result = module.metrics(chunks, [0, 0], [0, 20], 20, np.array([3.0, 4.0]), {"all": [0, 1]})
    assert result["raw"]["all"]["boundary_l2"]["p50"] == 5
    assert result["normalized"]["all"]["boundary_l2"]["p50"] == pytest.approx(2**0.5)


def test_nonfinite_actions_are_not_silently_dropped():
    with pytest.raises(ValueError, match="Nonfinite"):
        module.summary([1, np.nan])
