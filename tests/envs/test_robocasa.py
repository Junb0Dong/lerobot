from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from lerobot.envs import robocasa
from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig
from lerobot.processor import RoboCasaActionClipProcessorStep, RoboCasaTaskIndexProcessorStep


def _instantiate_envs(
    factories: Sequence[Callable[[], robocasa.RoboCasaEnv]],
) -> list[robocasa.RoboCasaEnv]:
    return [factory() for factory in factories]


def test_robocasa_config_uses_registered_horizon_by_default() -> None:
    assert RoboCasaEnvConfig().episode_length is None


def test_multi_task_envs_use_registered_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    horizons = {"CloseFridge": 900, "SearingMeat": 4350}
    get_task_horizon = Mock(side_effect=horizons.__getitem__)
    monkeypatch.setattr(robocasa, "_get_task_horizon", get_task_horizon)

    envs = robocasa.create_robocasa_envs(
        task="CloseFridge,SearingMeat",
        n_envs=1,
        env_cls=_instantiate_envs,
    )

    assert envs["CloseFridge"][0][0]._max_episode_steps == 900
    assert envs["SearingMeat"][0][0]._max_episode_steps == 4350
    assert get_task_horizon.call_args_list == [call("CloseFridge"), call("SearingMeat")]


def test_explicit_episode_length_overrides_registered_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    get_task_horizon = Mock()
    monkeypatch.setattr(robocasa, "_get_task_horizon", get_task_horizon)

    envs = robocasa.create_robocasa_envs(
        task="CloseFridge,SearingMeat",
        n_envs=1,
        env_cls=_instantiate_envs,
        episode_length=1234,
    )

    assert envs["CloseFridge"][0][0]._max_episode_steps == 1234
    assert envs["SearingMeat"][0][0]._max_episode_steps == 1234
    get_task_horizon.assert_not_called()


def test_task_index_processor_maps_batched_language_and_rejects_unknown() -> None:
    processor = RoboCasaTaskIndexProcessorStep(
        task_index_map={"Close the right drawer.": 0, "Close the left drawer.": 1}
    )
    mapped = processor.complementary_data({"task": ["Close the right drawer.", "Close the left drawer."]})
    torch.testing.assert_close(mapped["task_index"], torch.tensor([0, 1]))

    with pytest.raises(KeyError, match="Unmapped RoboCasa task descriptions"):
        processor.complementary_data({"task": ["Close some other drawer."]})


def test_robocasa_config_loads_task_map_and_clips_actions(tmp_path) -> None:
    mapping_path = tmp_path / "task_indices.json"
    mapping_path.write_text(
        json.dumps({"task_index_map": {"Turn off the sink faucet.": 7}}), encoding="utf-8"
    )
    config = RoboCasaEnvConfig(task_index_map_path=mapping_path)
    preprocessor, postprocessor = config.get_env_processors()

    mapped = preprocessor({"task": ["Turn off the sink faucet."]})
    torch.testing.assert_close(mapped["task_index"], torch.tensor([7]))
    clipped = postprocessor({"action": torch.tensor([[2.0, -2.0, 0.25]])})["action"]
    torch.testing.assert_close(clipped, torch.tensor([[1.0, -1.0, 0.25]]))
    clip_step = postprocessor.steps[0]
    assert isinstance(clip_step, RoboCasaActionClipProcessorStep)
    assert clip_step.metrics() == {
        "clipped_elements": 2,
        "total_elements": 3,
        "clipping_fraction": 2 / 3,
    }


def test_terminal_step_does_not_reset_underlying_robocasa_env() -> None:
    wrapper = object.__new__(robocasa.RoboCasaEnv)
    wrapper.task = "TurnOffSinkFaucet"
    wrapper._env = Mock()
    wrapper._env.step.return_value = ({"raw": np.zeros(1)}, 1.0, False, False, {"success": True})
    wrapper._ensure_env = Mock()
    wrapper._format_raw_obs = Mock(return_value={"agent_pos": np.zeros(16, dtype=np.float32)})

    _, reward, terminated, truncated, info = wrapper.step(np.zeros(12, dtype=np.float32))

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    wrapper._env.reset.assert_not_called()


def test_reset_preserves_explicit_vector_env_seed() -> None:
    wrapper = object.__new__(robocasa.RoboCasaEnv)
    wrapper.task = "TurnOffSinkFaucet"
    wrapper.episode_index = 3
    wrapper._env = Mock()
    wrapper._env.reset.return_value = ({"raw": np.zeros(1)}, {})
    wrapper._env.env.get_ep_meta.return_value = {"lang": "Turn off the sink faucet."}
    wrapper._ensure_env = Mock()
    wrapper._format_raw_obs = Mock(return_value={"agent_pos": np.zeros(16, dtype=np.float32)})

    robocasa.RoboCasaEnv.reset(wrapper, seed=43)

    wrapper._env.reset.assert_called_once_with(seed=43)


def test_robocasa_numpy_version_compat_is_narrow_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robocasa.np, "__version__", "1.26.4")
    with robocasa._robocasa_numpy_version_compat():
        assert robocasa.np.__version__ == "2.2.5"
    assert robocasa.np.__version__ == "1.26.4"
