#!/usr/bin/env python3
"""Matched A/B/C L2 chunk-boundary diagnosis; optional D from recorded action chunks.

Each episode is tiled from its first frame with stride S. Only full decoded H-step
windows are used. J compares current[0] to previous[S-1], so S=H implements the
full-chunk definition and S=n_action_steps measures the deployed execution seam.
Rollout NPZ input must contain raw, ordered ``chunks[N,H,D]``, ``episode_ids[N]``
and ``start_frames[N]``; the caller declares the recorded execution stride.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.actioncodec.modeling_actioncodec import ActionCodecPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        raise ValueError("No valid boundary samples")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite action metric")
    return dict(
        count=len(values),
        mean=float(values.mean()),
        **{
            key: float(value)
            for key, value in zip(
                ("p50", "p95", "p99", "max"), np.quantile(values, [0.5, 0.95, 0.99, 1]), strict=True
            )
        },
    )


def boundary_vectors(chunks, episode_ids, start_frames, stride):
    chunks = np.asarray(chunks)
    if chunks.ndim != 3 or not 1 <= stride <= chunks.shape[1]:
        raise ValueError("Expected [N,H,D] chunks with 1 <= stride <= H")
    episodes, starts = np.asarray(episode_ids), np.asarray(start_frames)
    if episodes.shape != (len(chunks),) or starts.shape != episodes.shape:
        raise ValueError("Episode/frame metadata must match the chunk count")
    valid = (episodes[1:] == episodes[:-1]) & (starts[1:] - starts[:-1] == stride)
    return chunks[1:, 0][valid] - chunks[:-1, stride - 1][valid], valid


def metrics(chunks, episodes, starts, stride, std, groups, gt=None):
    delta, valid = boundary_vectors(chunks, episodes, starts, stride)
    internal = np.diff(chunks[:, :stride], axis=1)
    result = {}
    for space, scale in (("raw", np.ones_like(std)), ("normalized", std + 1e-8)):
        result[space] = {}
        for group, indices in groups.items():
            jump = np.linalg.norm((delta / scale)[..., indices], axis=-1)
            inside = np.linalg.norm((internal / scale)[..., indices], axis=-1)
            entry = {"boundary_l2": summary(jump), "internal_step_l2": summary(inside)}
            if gt is not None:
                gt_delta, _ = boundary_vectors(gt, episodes, starts, stride)
                gt_jump = np.linalg.norm((gt_delta / scale)[..., indices], axis=-1)
                entry["paired_boundary_excess"] = summary(jump - gt_jump)
                entry["boundary_vector_error_l2"] = summary(
                    np.linalg.norm(((delta - gt_delta) / scale)[..., indices], axis=-1)
                )
                entry["reconstruction_mae"] = float(np.abs((chunks - gt) / scale)[..., indices].mean())
            entry["per_episode_boundary_l2"] = {
                str(e): summary(jump[np.asarray(episodes)[1:][valid] == e])
                for e in np.unique(np.asarray(episodes)[1:][valid])
            }
            result[space][group] = entry
    return result


def evaluate(args):
    torch.manual_seed(20260905)
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    policy = ActionCodecPolicy.from_pretrained(
        args.checkpoint, cli_overrides=[f"--device={device}"], local_files_only=True
    ).eval()
    policy.config.temperature = 0.0
    policy.config.top_k = 0
    checkpoint = Path(args.checkpoint)
    training = json.loads((checkpoint / "train_config.json").read_text())
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    root = Path(args.dataset_root)
    metadata = LeRobotDatasetMetadata(training["dataset"]["repo_id"], root=root)
    # Reuse the training split rule: last ceil(n * eval_split) episodes per task.
    from eval_xlerobot_actioncodec_policy_physical import _episode_split

    episodes = _episode_split(metadata, training["dataset"]["eval_split"])
    if training["dataset"].get("episodes") or training["dataset"].get("exclude_episodes"):
        raise ValueError("Explicit training episode subsets need their matching split")
    dataset = LeRobotDataset(
        training["dataset"]["repo_id"],
        root=root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata, {}),
        return_uint8=True,
        decode_videos=True,
        video_backend="torchcodec",
    )
    stats = json.loads((Path(policy.config.tokenizer_path) / "action_stats.json").read_text())
    mean = torch.tensor(stats["mean"], device=device)
    std = torch.tensor(stats["std"], device=device)
    names = metadata.features["action"]["names"]
    groups = {
        "all": list(range(len(names))),
        "arm": [i for i, name in enumerate(names) if "gripper" not in name],
        "gripper": [i for i, name in enumerate(names) if "gripper" in name],
    }
    horizon = policy.config.horizon
    strides = sorted({horizon, policy.config.n_action_steps})
    records = {}
    grids = {}
    for stride in strides:
        grid = []
        for episode in episodes:
            begin = int(metadata.episodes["dataset_from_index"][episode])
            end = int(metadata.episodes["dataset_to_index"][episode])
            starts = list(range(begin, end - horizon + 1, stride))
            if args.max_chunks_per_episode:
                starts = starts[: args.max_chunks_per_episode]
            for start in starts:
                records[start] = episode
                grid.append(start)
        grids[stride] = grid
    all_starts = sorted(records)
    values = {key: [] for key in ("A", "B", "C", "gt_tokens", "policy_tokens")}
    checked = False
    for offset in range(0, len(all_starts), args.batch_size):
        starts = all_starts[offset : offset + args.batch_size]
        mapping = dataset.absolute_to_relative_idx
        items = [dataset[start if mapping is None else mapping[start]] for start in starts]
        gt = torch.stack([item["action"] for item in items]).to(device).float()
        batch = pre(lerobot_collate_fn(items))
        with torch.inference_mode():
            norm = (gt - mean) / (std + 1e-8)
            if not checked:
                torch.testing.assert_close(batch["action"], norm, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(post(norm).cpu(), gt.cpu(), rtol=1e-5, atol=1e-5)
            gt_tokens = policy.tokenizer.tokenize(norm)
            recon = post(policy.tokenizer.detokenize(gt_tokens))
            features = policy.obs_encoder(batch)
            bos = torch.full((len(starts), 1), policy.config.codebook_size, device=device, dtype=torch.long)
            tokens = policy.model.generate(
                bos, features, policy.config.latent_horizon, policy._task_ids(batch), 0.0, 0
            )[:, 1:]
            predicted_norm = policy.tokenizer.detokenize(tokens)
            predicted = post(predicted_norm)
            if not checked:
                torch.testing.assert_close(predicted_norm, policy.predict_action_chunk(batch), rtol=0, atol=0)
                torch.testing.assert_close(
                    recon, post(policy.tokenizer.detokenize(gt_tokens)), rtol=0, atol=0
                )
                checked = True
        for key, tensor in zip(values, (gt, recon, predicted, gt_tokens, tokens), strict=True):
            values[key].append(tensor.cpu().numpy())
        print(
            f"{checkpoint.parent.parent.parent.name}: {min(offset + len(starts), len(all_starts))}/{len(all_starts)} chunks",
            flush=True,
        )
    values = {key: np.concatenate(parts) for key, parts in values.items()}
    np.savez_compressed(
        output / "chunks.npz", **values, start_frames=all_starts, episode_ids=[records[s] for s in all_starts]
    )
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "dataset_root": str(root.resolve()),
        "heldout_episodes": episodes,
        "action_names": names,
        "groups": groups,
        "horizon": horizon,
        "n_action_steps": policy.config.n_action_steps,
        "units": "dataset raw action units; degrees not independently verified",
        "inference": "CUDA FP32, greedy AR, deterministic full diffusion decoder",
        "normalization_verified_against_checkpoint_processors": checked,
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "D": {"status": "missing real rollout chunk/action logs"},
        "protocols": {},
    }
    index = {start: i for i, start in enumerate(all_starts)}
    for stride, starts in grids.items():
        selected = [index[s] for s in starts]
        ep = np.array([records[s] for s in starts])
        gt = values["A"][selected]
        result["protocols"][str(stride)] = {
            "stride": stride,
            "chunk_count": len(starts),
            "definition": f"norm(chunk[t,0] - chunk[t-1,{stride - 1}], 2)",
            "coverage": {str(e): int((ep == e).sum()) for e in episodes},
            "groups": {
                key: metrics(
                    values[key][selected],
                    ep,
                    starts,
                    stride,
                    std.cpu().numpy(),
                    groups,
                    gt=gt if key != "A" else None,
                )
                for key in ("A", "B", "C")
            },
            "free_running_token_accuracy": float(
                (values["gt_tokens"][selected] == values["policy_tokens"][selected]).mean()
            ),
        }
    if args.rollout_npz:
        with np.load(args.rollout_npz, allow_pickle=False) as rollout:
            result["D"] = {
                "source": str(Path(args.rollout_npz).resolve()),
                "stride": args.rollout_stride,
                "metrics": metrics(
                    rollout["chunks"],
                    rollout["episode_ids"],
                    rollout["start_frames"],
                    args.rollout_stride,
                    std.cpu().numpy(),
                    groups,
                ),
            }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-chunks-per-episode", type=int, default=0, help="Smoke only; 0 uses all full windows"
    )
    parser.add_argument("--rollout-npz")
    parser.add_argument("--rollout-stride", type=int, default=16)
    evaluate(parser.parse_args())
