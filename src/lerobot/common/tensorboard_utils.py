#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import atexit
import logging
from pathlib import Path

from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import require_package


class TensorBoardLogger:
    """Write training scalars (and best-effort eval videos) to local TensorBoard event files."""

    def __init__(self, cfg: TrainPipelineConfig):
        if cfg.tensorboard.log_dir:
            log_dir = Path(cfg.tensorboard.log_dir)
        elif cfg.output_dir is not None:
            log_dir = Path(cfg.output_dir) / "tb"
        else:
            raise ValueError("output_dir is required for TensorBoard logging")
        env_fps = cfg.env.fps if cfg.env else None
        self._init_writer(log_dir, env_fps=env_fps)
        self.cfg = cfg.tensorboard

    @classmethod
    def from_log_dir(cls, log_dir: str | Path, env_fps: float | None = None) -> "TensorBoardLogger":
        """Open a logger at ``log_dir`` without a full training pipeline config."""
        self = cls.__new__(cls)
        self._init_writer(Path(log_dir), env_fps=env_fps)
        self.cfg = None
        return self

    def _init_writer(self, log_dir: Path, env_fps: float | None = None) -> None:
        require_package("tensorboard", extra="training")
        self.env_fps = env_fps
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self._closed = False
        atexit.register(self.close)
        logging.info(colored("Logs will be written to TensorBoard.", "blue", attrs=["bold"]))
        logging.info(
            "View with: %s",
            colored(f"tensorboard --logdir {self.log_dir} --port 6006", "yellow", attrs=["bold"]),
        )

    def log_policy(self, checkpoint_dir: Path) -> None:
        """No-op: TensorBoard does not store model artifacts."""

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        if custom_step_key is not None:
            if custom_step_key not in d:
                raise KeyError(f'custom_step_key "{custom_step_key}" is missing from the logged dict.')
            step = int(d[custom_step_key])

        for k, v in d.items():
            if custom_step_key is not None and k == custom_step_key:
                continue
            if not isinstance(v, (int, float)):
                logging.warning(
                    'TensorBoard logging of key "%s" was ignored as its type "%s" is not a scalar.',
                    k,
                    type(v),
                )
                continue
            self._writer.add_scalar(f"{mode}/{k}", v, global_step=step)

    def flush(self) -> None:
        """Flush pending events without closing the logger."""
        self._writer.flush()

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        try:
            video = _load_mp4_as_tb_video(video_path)
            fps = self.env_fps if self.env_fps is not None else 30
            self._writer.add_video(f"{mode}/video", video, global_step=step, fps=fps)
        except Exception:
            logging.warning(
                "Failed to decode eval video for TensorBoard (%s); skipping video log.",
                video_path,
                exc_info=True,
            )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._writer.flush()
            self._writer.close()
        finally:
            self._closed = True


def _load_mp4_as_tb_video(video_path: str):
    """Decode an mp4 into a TensorBoard ``add_video`` tensor of shape ``(1, T, C, H, W)``."""
    import torchvision

    # torchvision returns (T, H, W, C) in [0, 255].
    video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Decoded zero frames from {video_path}")
    video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    return video
