#!/usr/bin/env python

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
from pathlib import Path

import pytest

from lerobot.common.tensorboard_utils import TensorBoardLogger
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig

tensorboard = pytest.importorskip("tensorboard")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def _make_cfg(tmp_path: Path, log_dir: str | None = None) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    cfg.output_dir = tmp_path
    cfg.tensorboard.log_dir = log_dir
    return cfg


def _scalar_events(log_dir: Path, tag: str) -> list:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator.Scalars(tag)


def test_log_dict_writes_readable_scalars(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    logger = TensorBoardLogger(cfg)
    logger.log_dict({"loss": 1.5, "grad_norm": 0.25}, step=10, mode="train")
    logger.log_dict({"eval_loss": 0.75}, step=20, mode="eval")
    logger.close()

    log_dir = tmp_path / "tb"
    assert any(log_dir.glob("events.out.tfevents.*"))

    train_loss = _scalar_events(log_dir, "train/loss")
    assert len(train_loss) == 1
    assert train_loss[0].step == 10
    assert train_loss[0].value == pytest.approx(1.5)

    train_grad = _scalar_events(log_dir, "train/grad_norm")
    assert len(train_grad) == 1
    assert train_grad[0].step == 10
    assert train_grad[0].value == pytest.approx(0.25)

    eval_loss = _scalar_events(log_dir, "eval/eval_loss")
    assert len(eval_loss) == 1
    assert eval_loss[0].step == 20
    assert eval_loss[0].value == pytest.approx(0.75)


def test_log_dict_custom_step_key_is_x_axis(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    logger = TensorBoardLogger(cfg)
    logger.log_dict(
        {"loss": 0.5, "Optimization step": 42},
        mode="train",
        custom_step_key="Optimization step",
    )
    logger.close()

    events = _scalar_events(tmp_path / "tb", "train/loss")
    assert len(events) == 1
    assert events[0].step == 42
    assert events[0].value == pytest.approx(0.5)
    accumulator = EventAccumulator(str(tmp_path / "tb"))
    accumulator.Reload()
    assert "train/Optimization step" not in accumulator.Tags()["scalars"]


def test_log_policy_is_noop(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    logger = TensorBoardLogger(cfg)
    logger.log_policy(tmp_path / "checkpoint")
    logger.close()
    assert any((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_close_is_idempotent_and_flushes(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    logger = TensorBoardLogger(cfg)
    logger.log_dict({"loss": 0.1}, step=1)
    logger.close()
    logger.close()
    events = _scalar_events(tmp_path / "tb", "train/loss")
    assert len(events) == 1
    assert events[0].step == 1
    assert events[0].value == pytest.approx(0.1)


def test_from_log_dir_writes_readable_scalars(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    logger = TensorBoardLogger.from_log_dir(log_dir)
    logger.log_dict({"loss": 2.25}, step=3)
    logger.close()

    events = _scalar_events(log_dir, "train/loss")
    assert len(events) == 1
    assert events[0].step == 3
    assert events[0].value == pytest.approx(2.25)


def test_legacy_train_config_gets_tensorboard_defaults(tmp_path: Path) -> None:
    import json

    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    payload = cfg.to_dict()
    assert payload["tensorboard"] == {"enable": True, "log_dir": None}
    assert payload["wandb"]["enable"] is False
    payload.pop("tensorboard")
    config_file = tmp_path / "train_config.json"
    config_file.write_text(json.dumps(payload))
    loaded = TrainPipelineConfig.from_pretrained(config_file)
    assert loaded.tensorboard.enable is True
    assert loaded.tensorboard.log_dir is None
    assert loaded.wandb.enable is False
