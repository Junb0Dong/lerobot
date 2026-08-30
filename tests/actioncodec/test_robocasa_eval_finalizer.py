from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TASKS = ["CloseDrawer", "StartCoffeeMachine", "TurnOffMicrowave", "TurnOffSinkFaucet"]
CHECKPOINT_FILES = [
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_4_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "tokenizer/action_stats.json",
    "tokenizer/dataset_contract.json",
    "tokenizer/model_config.json",
    "tokenizer/model.safetensors",
]


def test_finalizer_validates_hashes_and_writes_per_task_summary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config = {
        "type": "actioncodec",
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "horizon": 20,
        "latent_horizon": 16,
        "action_dim": 12,
        "num_tasks": 9,
        "temperature": 0.0,
        "top_k": 10,
        "input_features": {
            f"observation.images.{camera}": {"type": "VISUAL", "shape": [3, 256, 256]}
            for camera in ("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right")
        }
        | {"observation.state": {"type": "STATE", "shape": [16]}},
        "output_features": {"action": {"type": "ACTION", "shape": [12]}},
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for relative_path in CHECKPOINT_FILES:
        path = checkpoint / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode())

    task_map = tmp_path / "task_map.json"
    task_map.write_text(
        json.dumps(
            {"task_index_map": {f"task-{index}": value for index, value in enumerate([0, 1, 3, 5, 7])}}
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "eval"
    script = Path(__file__).parents[2] / "scripts/dlc/finalize_robocasa_actioncodec_eval.py"
    common_args = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--task-index-map",
        str(task_map),
        "--split",
        "target",
        "--episodes-per-task",
        "2",
        "--batch-size",
        "2",
        "--seed",
        "42",
    ]
    subprocess.run([*common_args, "--phase", "before"], check=True)

    per_task = []
    for task in TASKS:
        per_task.append(
            {
                "task_group": task,
                "task_id": 0,
                "metrics": {
                    "sum_rewards": [0.0, 1.0],
                    "max_rewards": [0.0, 1.0],
                    "successes": [False, True],
                    "video_paths": [f"{task}.mp4"],
                    "predicted_video_paths": [],
                },
            }
        )
    eval_info = {
        "per_task": per_task,
        "per_group": {},
        "overall": {"n_episodes": 8, "pc_success": 50.0},
        "processor_metrics": {"RoboCasaActionClipProcessorStep": {"clipping_fraction": 0.01}},
    }
    (output_dir / "eval_info.json").write_text(json.dumps(eval_info), encoding="utf-8")
    subprocess.run([*common_args, "--phase", "after"], check=True)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["per_task"]["CloseDrawer"]["success_rate_percent"] == 50.0
    assert summary["per_task"]["StartCoffeeMachine"]["report_name"] == "CoffeePressButton"
    assert summary["processor_metrics"]["RoboCasaActionClipProcessorStep"]["clipping_fraction"] == 0.01
