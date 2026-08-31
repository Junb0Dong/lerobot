# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Worker-side RGB resize used to cut DataLoader IPC for policies that train at image_size."""

import pytest
import torch

from lerobot.datasets.video_utils import decode_video_frames, decoder_cache_capacity, resize_uint8_hw
from lerobot.utils.import_utils import get_safe_default_video_backend


def test_resize_uint8_chw():
    frames = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)
    out = resize_uint8_hw(frames, 16)
    assert out.shape == (3, 16, 16)
    assert out.dtype == torch.uint8


def test_resize_uint8_tchw():
    frames = torch.randint(0, 256, (2, 3, 48, 64), dtype=torch.uint8)
    out = resize_uint8_hw(frames, 16)
    assert out.shape == (2, 3, 16, 16)
    assert out.dtype == torch.uint8


def test_resize_uint8_skips_when_already_target_size():
    frames = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
    out = resize_uint8_hw(frames, 16)
    assert out is frames


def test_resize_uint8_rejects_non_positive_size():
    frames = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="positive"):
        resize_uint8_hw(frames, 0)


def test_decoder_cache_capacity_covers_all_rgb_videos():
    assert decoder_cache_capacity(n_episodes=47, n_rgb_video_keys=3, default=100) == 141
    assert decoder_cache_capacity(n_episodes=2, n_rgb_video_keys=2, default=100) == 100


def test_decode_video_frames_forwards_decoder_cache(monkeypatch):
    captured = {}

    def fake_torchcodec(
        video_path, timestamps, tolerance_s, decoder_cache=None, return_uint8=False, **kwargs
    ):
        del video_path, timestamps, tolerance_s, return_uint8, kwargs
        captured["decoder_cache"] = decoder_cache
        return torch.zeros(1, 3, 4, 4)

    monkeypatch.setattr("lerobot.datasets.video_utils.decode_video_frames_torchcodec", fake_torchcodec)
    cache = object()
    decode_video_frames("x.mp4", [0.0], 0.1, backend="torchcodec", decoder_cache=cache)
    assert captured["decoder_cache"] is cache


def test_dataset_reader_reuses_torchcodec_cache(tmp_path, lerobot_dataset_factory):
    if get_safe_default_video_backend() != "torchcodec":
        pytest.skip("torchcodec is the cached decoder path")
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=4,
        use_videos=True,
        return_uint8=True,
    )
    if not dataset.meta.video_keys:
        pytest.skip("factory produced no video keys")
    dataset[0]
    cache = dataset.reader._decoder_cache
    assert cache is not None
    size_after_first = cache.size()
    assert size_after_first >= 1
    dataset[1]
    assert dataset.reader._decoder_cache is cache
    assert cache.size() == size_after_first


def test_getitem_resizes_rgb_videos(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=4,
        use_videos=True,
        decode_image_size=16,
        return_uint8=True,
    )
    if not dataset.meta.video_keys:
        pytest.skip("factory produced no video keys")
    item = dataset[0]
    rgb_keys = [key for key in dataset.meta.video_keys if key not in dataset.meta.depth_keys]
    assert rgb_keys
    for key in rgb_keys:
        assert tuple(item[key].shape[-2:]) == (16, 16)
        assert item[key].dtype == torch.uint8


def test_getitem_keeps_native_size_when_decode_image_size_is_none(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=4,
        use_videos=True,
        return_uint8=True,
    )
    if not dataset.meta.video_keys:
        pytest.skip("factory produced no video keys")
    item = dataset[0]
    rgb_keys = [key for key in dataset.meta.video_keys if key not in dataset.meta.depth_keys]
    native = tuple(dataset.meta.features[rgb_keys[0]]["shape"][:2])  # H, W in metadata
    for key in rgb_keys:
        assert tuple(item[key].shape[-2:]) == native
