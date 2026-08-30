#!/usr/bin/env python
"""Validate and summarize the fixed atomic4 ActionCodec RoboCasa rollout protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Any

TASKS = ["CloseDrawer", "StartCoffeeMachine", "TurnOffMicrowave", "TurnOffSinkFaucet"]
TASK_HORIZONS = {
    "CloseDrawer": 450,
    "StartCoffeeMachine": 300,
    "TurnOffMicrowave": 300,
    "TurnOffSinkFaucet": 300,
}
REPORT_ALIASES = {"StartCoffeeMachine": "CoffeePressButton"}
CHECKPOINT_FILES = [
    "config.json",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_checkpoint(checkpoint: Path) -> dict[str, str]:
    missing = [name for name in CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")
    return {name: _sha256(checkpoint / name) for name in CHECKPOINT_FILES}


def _package_versions() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for package in ("torch", "robocasa", "robosuite", "mujoco", "gymnasium", "lerobot"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def _validate_contract(checkpoint: Path, task_index_map: Path) -> dict[str, Any]:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    expected_scalars = {
        "type": "actioncodec",
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "horizon": 20,
        "latent_horizon": 16,
        "action_dim": 12,
        "num_tasks": 9,
        "temperature": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_scalars.items()
        if config.get(key) != expected
    }
    expected_images = {
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_right",
    }
    input_features = config.get("input_features", {})
    actual_images = {key for key, feature in input_features.items() if feature.get("type") == "VISUAL"}
    state_shape = input_features.get("observation.state", {}).get("shape")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if actual_images != expected_images:
        mismatches["image_keys"] = {
            "expected": sorted(expected_images),
            "actual": sorted(actual_images),
        }
    if state_shape != [16]:
        mismatches["state_shape"] = {"expected": [16], "actual": state_shape}
    if action_shape != [12]:
        mismatches["action_shape"] = {"expected": [12], "actual": action_shape}
    if mismatches:
        raise ValueError(f"Checkpoint does not match atomic4 rollout contract: {mismatches}")

    mapping_payload = json.loads(task_index_map.read_text(encoding="utf-8"))
    mapping = mapping_payload.get("task_index_map")
    if not isinstance(mapping, dict) or sorted(mapping.values()) != [0, 1, 3, 5, 7]:
        raise ValueError("Atomic4 task-index map must contain the trained IDs 0,1,3,5,7")
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index-map", type=Path, required=True)
    parser.add_argument("--split", choices=("pretrain", "target"), required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    config = _validate_contract(checkpoint, args.task_index_map)
    hashes = _hash_checkpoint(checkpoint)
    protocol = {
        "protocol": "lerobot_robocasa_target_lightwheel_v1",
        "phase": args.phase,
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": hashes,
        "versions": _package_versions(),
        "renderer": "egl",
        "robocasa_numpy1_import_compat": True,
        "object_registries": ["lightwheel"],
        "split": args.split,
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "batch_size": args.batch_size,
        "use_async_envs": True,
        "tasks": TASKS,
        "task_horizons": TASK_HORIZONS,
        "report_aliases": REPORT_ALIASES,
        "policy": {
            "horizon": config["horizon"],
            "latent_horizon": config["latent_horizon"],
            "n_obs_steps": config["n_obs_steps"],
            "n_action_steps": config["n_action_steps"],
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "weights": "last",
            "ema": False,
        },
    }

    hash_path = output_dir / "checkpoint_hashes_before.json"
    if args.phase == "before":
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(hash_path, hashes)
        _write_json(output_dir / "protocol.json", protocol)
        print(json.dumps({"status": "contract_ok", "output_dir": str(output_dir)}))
        return

    before_hashes = json.loads(hash_path.read_text(encoding="utf-8"))
    if hashes != before_hashes:
        raise RuntimeError("Checkpoint files changed during RoboCasa evaluation")
    eval_info_path = output_dir / "eval_info.json"
    eval_info = json.loads(eval_info_path.read_text(encoding="utf-8"))
    expected_episodes = len(TASKS) * args.episodes_per_task
    actual_episodes = eval_info.get("overall", {}).get("n_episodes")
    if actual_episodes != expected_episodes:
        raise RuntimeError(f"Expected {expected_episodes} rollout episodes, got {actual_episodes}")

    per_task = {}
    for entry in eval_info.get("per_task", []):
        task = entry["task_group"]
        metrics = entry["metrics"]
        successes = metrics.get("successes", [])
        if not successes:
            raise RuntimeError(f"Evaluation returned no episodes for task {task}")
        per_task[task] = {
            "report_name": REPORT_ALIASES.get(task, task),
            "n_episodes": len(successes),
            "success_rate_percent": 100.0 * fmean(bool(value) for value in successes),
            "video_paths": metrics.get("video_paths", []),
        }
    if set(per_task) != set(TASKS):
        raise RuntimeError(f"Evaluation task set mismatch: {sorted(per_task)}")
    wrong_counts = {
        task: values["n_episodes"]
        for task, values in per_task.items()
        if values["n_episodes"] != args.episodes_per_task
    }
    if wrong_counts:
        raise RuntimeError(f"Expected {args.episodes_per_task} episodes for every task, got {wrong_counts}")

    summary = {
        **protocol,
        "phase": "complete",
        "per_task": per_task,
        "overall": eval_info["overall"],
        "processor_metrics": eval_info.get("processor_metrics", {}),
        "eval_info": str(eval_info_path),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"status": "eval_ok", "summary": str(output_dir / "summary.json")}))


if __name__ == "__main__":
    main()
