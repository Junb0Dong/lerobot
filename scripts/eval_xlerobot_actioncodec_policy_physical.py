#!/usr/bin/env python3
"""Read-only physical metrics for ActionCodec policy checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.actioncodec.modeling_actioncodec import ActionCodecPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    result = torch.quantile(values.float(), torch.tensor([0.5, 0.95, 0.99]))
    return {name: float(value) for name, value in zip(("p50", "p95", "p99"), result, strict=True)}


def _episode_split(metadata: LeRobotDatasetMetadata, eval_split: float) -> list[int]:
    if eval_split == 0:
        return list(range(len(metadata.episodes["dataset_from_index"])))
    task_to_episodes: dict[str, list[int]] = {}
    for episode_index, task_ids in enumerate(metadata.episodes["tasks"]):
        task_to_episodes.setdefault(str(task_ids[0]), []).append(episode_index)
    selected = []
    for episodes in task_to_episodes.values():
        count = math.ceil(len(episodes) * eval_split)
        selected.extend(episodes[-count:])
    return sorted(selected)


def _pair_starts(
    metadata: LeRobotDatasetMetadata,
    episode_indices: list[int],
    horizon: int,
    shift: int,
    stride: int,
) -> list[int]:
    starts: list[int] = []
    for episode_index in episode_indices:
        start = int(metadata.episodes["dataset_from_index"][episode_index])
        end = int(metadata.episodes["dataset_to_index"][episode_index])
        starts.extend(range(start, end - horizon - shift + 1, stride))
    return starts


def _select_starts(starts: list[int], sample_count: int, seed: int, pair_file: Path | None) -> list[int]:
    if pair_file is not None:
        payload = json.loads(pair_file.read_text(encoding="utf-8"))
        selected = payload["pair_starts"] if isinstance(payload, dict) else payload
        return [int(start) for start in selected]
    if sample_count > len(starts):
        raise ValueError(f"sample_count={sample_count} exceeds {len(starts)} available pairs")
    return sorted(random.Random(seed).sample(starts, sample_count))


def _max_per_sample(values: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    return values[..., list(indices)].abs().amax(dim=(-2, -1))


def _evaluate_checkpoint(
    checkpoint: Path,
    dataset: LeRobotDataset,
    pair_starts: list[int],
    device: torch.device,
    continuous: tuple[int, ...] | None,
    batch_size: int,
    shift: int,
) -> dict[str, object]:
    policy = ActionCodecPolicy.from_pretrained(
        checkpoint,
        cli_overrides=[f"--device={device}"],
        local_files_only=True,
    )
    policy.config.temperature = 0.0
    policy.config.top_k = 0
    policy.eval()
    if continuous is None:
        continuous = tuple(policy.config.continuous_action_indices or range(policy.config.action_dim))
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    absolute_to_relative = dataset.absolute_to_relative_idx
    normalized_maximum = []
    first_target_delta = []
    reconstruction_error = []
    velocity_error = []
    overlap_error = []
    seam_error = []
    gt_seam_error = []
    maximum_action_delta = []
    gt_maximum_action_delta = []

    def relative(index: int) -> int:
        return index if absolute_to_relative is None else absolute_to_relative[index]

    for offset in range(0, len(pair_starts), batch_size):
        starts = pair_starts[offset : offset + batch_size]
        items_a = [dataset[relative(start)] for start in starts]
        items_b = [dataset[relative(start + shift)] for start in starts]
        state_a = torch.stack([item["observation.state"][-1] for item in items_a]).float()
        gt_a = torch.stack([item["action"] for item in items_a]).float()
        gt_b = torch.stack([item["action"] for item in items_b]).float()
        batch_a = preprocessor(lerobot_collate_fn(items_a))
        batch_b = preprocessor(lerobot_collate_fn(items_b))
        with torch.inference_mode():
            pred_a_norm = policy.predict_action_chunk(batch_a).cpu()
            pred_b_norm = policy.predict_action_chunk(batch_b).cpu()
            pred_a = postprocessor(pred_a_norm.to(device)).cpu()
            pred_b = postprocessor(pred_b_norm.to(device)).cpu()
        normalized_maximum.append(
            torch.cat((_max_per_sample(pred_a_norm, continuous), _max_per_sample(pred_b_norm, continuous)))
        )
        first_target_delta.append(_max_per_sample(pred_a[:, :1] - state_a[:, None], continuous))
        reconstruction_error.append(torch.cat(((pred_a - gt_a).abs(), (pred_b - gt_b).abs()), dim=0))
        velocity_error.append(
            torch.cat(
                (
                    (torch.diff(pred_a, dim=1) - torch.diff(gt_a, dim=1)).abs(),
                    (torch.diff(pred_b, dim=1) - torch.diff(gt_b, dim=1)).abs(),
                ),
                dim=0,
            )
        )
        overlap_error.append((pred_a[:, 16:20] - pred_b[:, :4]).abs())
        seam_error.append((pred_b[:, 0] - pred_a[:, 15]).abs())
        gt_seam_error.append((gt_b[:, 0] - gt_a[:, 15]).abs())
        maximum_action_delta.append(
            torch.stack((torch.diff(pred_a, dim=1).abs(), torch.diff(pred_b, dim=1).abs()), dim=1)
        )
        gt_maximum_action_delta.append(
            torch.stack((torch.diff(gt_a, dim=1).abs(), torch.diff(gt_b, dim=1).abs()), dim=1)
        )

    reconstruction = torch.cat(reconstruction_error)
    velocity = torch.cat(velocity_error)
    overlap = torch.cat(overlap_error)
    seam = torch.cat(seam_error)
    gt_seam = torch.cat(gt_seam_error)
    maximum_delta = torch.cat(maximum_action_delta)
    gt_maximum_delta = torch.cat(gt_maximum_action_delta)
    return {
        "checkpoint": str(checkpoint.absolute()),
        "continuous_action_indices": list(continuous),
        "normalized_action_maximum": _quantiles(torch.cat(normalized_maximum)),
        "first_target_delta": _quantiles(torch.cat(first_target_delta)),
        "decoded_mae": _quantiles(reconstruction[..., list(continuous)].mean(dim=-1).flatten()),
        "velocity_error": _quantiles(velocity[..., list(continuous)].mean(dim=-1).flatten()),
        "same_time_overlap": _quantiles(overlap[..., list(continuous)].mean(dim=(-2, -1))),
        "rollout_seam": _quantiles(seam[..., list(continuous)].amax(dim=-1)),
        "gt_rollout_seam": _quantiles(gt_seam[..., list(continuous)].amax(dim=-1)),
        "rollout_seam_excess": _quantiles(
            (seam[..., list(continuous)].amax(dim=-1) - gt_seam[..., list(continuous)].amax(dim=-1)).flatten()
        ),
        "maximum_action_delta": _quantiles(maximum_delta[..., list(continuous)].amax(dim=(-2, -1)).flatten()),
        "gt_maximum_action_delta": _quantiles(
            gt_maximum_delta[..., list(continuous)].amax(dim=(-2, -1)).flatten()
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.dataset_root)
    repo_id = args.repo_id or f"local/{root.name}"
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    episodes = _episode_split(metadata, args.eval_split)
    starts = _pair_starts(metadata, episodes, args.horizon, args.overlap_shift, args.window_stride)
    pair_starts = _select_starts(
        starts, args.sample_count, args.seed, Path(args.pair_file) if args.pair_file else None
    )
    pair_set = set(starts)
    if any(start not in pair_set for start in pair_starts):
        raise ValueError("pair file contains a start outside the selected episode split")
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(
            args.policy_config,
            metadata,
            {},
        ),
        return_uint8=True,
        decode_videos=True,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    continuous = tuple(json.loads(args.continuous_action_indices)) if args.continuous_action_indices else None
    result = {
        "dataset_root": str(root.absolute()),
        "repo_id": repo_id,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "overlap_shift": args.overlap_shift,
        "pair_starts": pair_starts,
        "continuous_action_indices": None if continuous is None else list(continuous),
        "checkpoints": {},
    }
    checkpoint_root = Path(args.checkpoint_root)
    for step in args.steps:
        checkpoint = checkpoint_root / "checkpoints" / f"{step:06d}" / "pretrained_model_ema"
        if not checkpoint.is_dir():
            checkpoint = checkpoint.with_name("pretrained_model")
        checkpoint_result = _evaluate_checkpoint(
            checkpoint,
            dataset,
            pair_starts,
            device,
            continuous,
            args.batch_size,
            args.overlap_shift,
        )
        result["checkpoints"][str(step)] = checkpoint_result
        if result["continuous_action_indices"] is None:
            result["continuous_action_indices"] = checkpoint_result["continuous_action_indices"]
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
    )
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--pair-file", default=None)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--overlap-shift", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--continuous-action-indices", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.policy_config is None:
        from lerobot.policies.actioncodec.configuration_actioncodec import ActionCodecConfig

        args.policy_config = ActionCodecConfig(device=args.device)
    result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
