"""Processor glue for task-token conditioning."""

from __future__ import annotations

from typing import Any

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("ActionCodecTaskToken")
class ActionCodecTaskTokenProcessorStep(ProcessorStep):
    """Expose LeRobotDataset's task_index under the policy's task_uid name."""

    def __init__(self, task_key: str = "task_index") -> None:
        self.task_key = task_key

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if self.task_key in complementary:
            complementary = dict(complementary)
            complementary["task_uid"] = complementary[self.task_key]
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"task_key": self.task_key}

    def transform_features(self, features):
        return features


def make_actioncodec_pre_post_processors(config, dataset_stats=None, dataset_meta=None):
    """Build the native LeRobot normalization pipeline plus task-token glue."""
    steps = make_default_policy_processor_steps(config, dataset_stats)
    preprocessor, postprocessor = make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            ActionCodecTaskTokenProcessorStep(),
            steps.to_device,
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
    return preprocessor, postprocessor
