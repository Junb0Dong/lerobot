#!/usr/bin/env python3
"""Deterministic physical continuity evaluation for an ActionCodec tokenizer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from lerobot.actioncodec.config import load_tokenizer_config
from lerobot.actioncodec.models.tokenizer import ActionCodecTokenizer
from lerobot.actioncodec.trainer import _delta_timestamps, _paired_window_start_indices
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata


def _parse_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(index) for index in json.loads(value))


def _build_tokenizer(checkpoint: Path, device: torch.device) -> ActionCodecTokenizer:
    config = load_tokenizer_config(checkpoint)
    model = ActionCodecTokenizer(
        action_dim=config.action_dim,
        window_size=config.horizon,
        model_dim=config.model_dim,
        num_tokens=config.latent_horizon,
        quantizer_type=config.quantizer_type,
        fsq_levels=config.fsq_levels,
        codebook_size=config.codebook_size,
        num_codebooks=config.num_codebooks,
        num_heads=config.num_heads,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        encoder_cross_layers=config.encoder_cross_layers,
        decoder_cross_layers=config.decoder_cross_layers,
        use_encoder_latent_self_attn=config.use_encoder_latent_self_attn,
        use_decoder_latent_self_attn=config.use_decoder_latent_self_attn,
        share_encoder_latent_transformer=config.share_encoder_latent_transformer,
        share_decoder_latent_transformer=config.share_decoder_latent_transformer,
        share_encoder_cross_attn=config.share_encoder_cross_attn,
        share_decoder_cross_attn=config.share_decoder_cross_attn,
        dropout=config.dropout,
        vq_beta=config.vq_beta,
        soft_assignment_temperature=config.soft_assignment_temperature,
        dead_code_threshold=config.dead_code_threshold,
        reset_noise_scale=config.reset_noise_scale,
        decoder_type=config.decoder_type,
        diffusion_config=config.diffusion_config,
        use_vl_embedder=config.use_vl_embedder,
    ).to(device)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    return model.eval()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(values.float(), torch.tensor([0.5, 0.95, 0.99]))
    return {name: float(value) for name, value in zip(("p50", "p95", "p99"), quantiles, strict=True)}


def _group_scores(values: torch.Tensor, indices: tuple[int, ...]) -> dict[str, float] | None:
    if not indices:
        return None
    return _quantiles(values[..., list(indices)].reshape(-1))


def _max_group_scores(values: torch.Tensor, indices: tuple[int, ...]) -> dict[str, float] | None:
    if not indices:
        return None
    scores = values[..., list(indices)].amax(dim=(-2, -1))
    return _quantiles(scores.reshape(-1))


def _pair_records(
    metadata: LeRobotDatasetMetadata,
    horizon: int,
    shift: int,
    stride: int,
    sample_count: int,
    seed: int,
    pair_file: Path | None,
) -> list[tuple[int, int]]:
    episodes = metadata.episodes
    available_starts = _paired_window_start_indices(
        episodes["dataset_from_index"],
        episodes["dataset_to_index"],
        horizon=horizon,
        shift=shift,
        stride=stride,
    )
    starts = available_starts
    available_start_set = set(available_starts)
    if pair_file is not None and pair_file.is_file():
        payload = json.loads(pair_file.read_text(encoding="utf-8"))
        starts = payload["pair_starts"] if isinstance(payload, dict) else payload
        starts = [int(start) for start in starts]
        if any(start not in available_start_set for start in starts):
            raise ValueError("pair file contains a start that is not a valid episode-local pair")
    elif pair_file is not None:
        raise FileNotFoundError(pair_file)
    else:
        if sample_count > len(starts):
            raise ValueError(f"sample_count={sample_count} exceeds {len(starts)} available pairs")
        starts = sorted(random.Random(seed).sample(starts, sample_count))
    episode_records = []
    for start in starts:
        for episode_index, (episode_start, episode_end) in enumerate(
            zip(
                metadata.episodes["dataset_from_index"],
                metadata.episodes["dataset_to_index"],
                strict=True,
            )
        ):
            if int(episode_start) <= start < int(episode_end):
                episode_records.append((start, episode_index))
                break
    if len(episode_records) != len(starts):
        raise ValueError("pair file contains a start outside the dataset episode bounds")
    return [(start, episode_index) for start, episode_index in sorted(episode_records)]


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.tokenizer_path)
    root = Path(args.dataset_root)
    repo_id = args.repo_id or f"local/{root.name}"
    config = load_tokenizer_config(checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    pair_records = _pair_records(
        metadata,
        horizon=config.horizon,
        shift=args.overlap_shift,
        stride=args.window_stride,
        sample_count=args.sample_count,
        seed=args.seed,
        pair_file=Path(args.pair_file) if args.pair_file else None,
    )
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=_delta_timestamps(metadata.fps, config.horizon),
        return_uint8=True,
        decode_videos=False,
    )
    model = _build_tokenizer(checkpoint, device)
    stats = json.loads((checkpoint / "action_stats.json").read_text(encoding="utf-8"))
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    continuous = _parse_indices(args.continuous_action_indices)
    if continuous is None:
        continuous = tuple(range(config.action_dim))
    gripper = tuple(index for index in range(config.action_dim) if index not in continuous)

    gt_a_parts, gt_b_parts, recon_a_parts, recon_b_parts = [], [], [], []
    for offset in range(0, len(pair_records), args.batch_size):
        batch_records = pair_records[offset : offset + args.batch_size]
        gt_a = torch.stack([dataset[start]["action"] for start, _ in batch_records]).to(device).float()
        gt_b = (
            torch.stack([dataset[start + args.overlap_shift]["action"] for start, _ in batch_records])
            .to(device)
            .float()
        )
        with torch.inference_mode():
            recon_a = model.detokenize(model.tokenize((gt_a - mean) / (std + 1e-8))) * std + mean
            recon_b = model.detokenize(model.tokenize((gt_b - mean) / (std + 1e-8))) * std + mean
        gt_a_parts.append(gt_a.cpu())
        gt_b_parts.append(gt_b.cpu())
        recon_a_parts.append(recon_a.cpu())
        recon_b_parts.append(recon_b.cpu())

    gt_a = torch.cat(gt_a_parts)
    gt_b = torch.cat(gt_b_parts)
    recon_a = torch.cat(recon_a_parts)
    recon_b = torch.cat(recon_b_parts)
    recon = torch.stack((recon_a, recon_b), dim=1)
    target = torch.stack((gt_a, gt_b), dim=1)
    recon_error = (recon - target).abs()
    recon_velocity_error = (torch.diff(recon, dim=2) - torch.diff(target, dim=2)).abs()
    overlap_error = (recon_a[:, args.overlap_shift :] - recon_b[:, : -args.overlap_shift]).abs()
    seam_index = 15
    seam_error = (recon_a[:, seam_index] - recon_b[:, 0]).abs()
    gt_seam_error = (gt_a[:, seam_index] - gt_b[:, 0]).abs()
    seam_excess = seam_error - gt_seam_error
    recon_max_delta = torch.diff(recon, dim=2).abs()

    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(root),
        "sample_count": len(pair_records),
        "overlap_shift": args.overlap_shift,
        "seed": args.seed,
        "pair_starts": [start for start, _ in pair_records],
        "pair_episode_indices": [episode for _, episode in pair_records],
        "continuous_action_indices": list(continuous),
        "gripper_action_indices": list(gripper),
        "physical_reconstruction_mae_per_joint": recon_error.mean(dim=(0, 1, 2)).tolist(),
        "arm_reconstruction_mae": float(recon_error[..., list(continuous)].mean().item()),
        "gripper_reconstruction_mae": (
            float(recon_error[..., list(gripper)].mean().item()) if gripper else None
        ),
        "velocity_error": {
            "arm": _group_scores(recon_velocity_error, continuous),
            "gripper": _group_scores(recon_velocity_error, gripper),
        },
        "same_time_overlap_inconsistency": {
            "arm": _max_group_scores(overlap_error, continuous),
            "gripper": _max_group_scores(overlap_error, gripper),
        },
        "rollout_seam_delta": {
            "arm": _max_group_scores(seam_error.unsqueeze(-2), continuous),
            "gripper": _max_group_scores(seam_error.unsqueeze(-2), gripper),
        },
        "gt_rollout_seam_delta": {
            "arm": _max_group_scores(gt_seam_error.unsqueeze(-2), continuous),
            "gripper": _max_group_scores(gt_seam_error.unsqueeze(-2), gripper),
        },
        "seam_excess_over_gt": {
            "arm": _quantiles(seam_excess[..., list(continuous)].amax(dim=-1)),
            "gripper": (_quantiles(seam_excess[..., list(gripper)].amax(dim=-1)) if gripper else None),
        },
        "maximum_action_delta": {
            "arm": _max_group_scores(recon_max_delta, continuous),
            "gripper": _max_group_scores(recon_max_delta, gripper),
        },
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", "--checkpoint", dest="tokenizer_path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--overlap-shift", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continuous-action-indices", default=None)
    parser.add_argument("--pair-file", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.repo_id is None:
        args.repo_id = f"local/{Path(args.dataset_root).name}"
    result = evaluate(args)
    print(f"checkpoint: {result['checkpoint']}")
    print(f"pairs: {result['sample_count']} seed={result['seed']} shift={result['overlap_shift']}")
    print(f"arm reconstruction MAE: {result['arm_reconstruction_mae']:.4f} raw units")
    print(f"gripper reconstruction MAE: {result['gripper_reconstruction_mae']}")
    for name in (
        "velocity_error",
        "same_time_overlap_inconsistency",
        "rollout_seam_delta",
        "maximum_action_delta",
    ):
        print(f"{name}: {json.dumps(result[name], sort_keys=True)}")
    print(f"gt_rollout_seam_delta: {json.dumps(result['gt_rollout_seam_delta'], sort_keys=True)}")
    print(f"seam_excess_over_gt: {json.dumps(result['seam_excess_over_gt'], sort_keys=True)}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
