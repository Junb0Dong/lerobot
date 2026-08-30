"""Small, independent semantic tokenizer trainer backed by LeRobotDataset v3."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

from .config import ActionCodecTokenizerConfig, save_tokenizer_artifact
from .models.tokenizer import ActionCodecTokenizer


@dataclass
class SemanticTokenizerTrainConfig:
    """Command-line surface of the standalone semantic tokenizer training run.

    This is the subset of :class:`~lerobot.actioncodec.config.ActionCodecTokenizerConfig` that a
    user is expected to set, plus the dataset and output location. Everything else keeps its
    default when :meth:`to_tokenizer_config` builds the full tokenizer configuration.

    Attributes:
        repo_id: LeRobotDataset repository identifier to train on.
        root: Local dataset root; ``None`` resolves through the default cache.
        output_dir: Directory the tokenizer checkpoint is written to.
        steps: Number of optimizer steps to run.
        batch_size: Number of action windows per batch.
        num_workers: Dataloader worker processes.
        learning_rate: AdamW learning rate.
        device: Torch device string to train on.
        seed: Seed applied to ``random``, ``numpy`` and ``torch``.
        model_dim: Width of the encoder and decoder transformers.
        action_dim: Number of action dimensions, checked against the dataset feature shape.
        action_horizon: Number of action timesteps per window; the contract requires ``20``.
        latent_horizon: Number of latent tokens emitted per window; the contract requires ``16``.
        codebook_size: Token vocabulary size; the contract requires ``1024``.
        num_codebooks: Number of residual quantization stages; the contract requires ``1``.
        alignment_weight: Weight of the **soft-DTW** chunk alignment loss
            (``weight_chunk_align``). The default ``0.1`` matches the source tokenizer recipe.
            ``0`` turns alignment off.
        hard_alignment_weight: Weight of the older hard-DTW alignment (``weight_align``). Default
            ``0``; keep it off unless you explicitly want both.
        decoder_type: ``"diffusion"`` (default) or ``"perceiver"``.
        diffusion_config: Extra keyword arguments for the diffusion decoder.
        log_freq: Number of steps between training log lines; ``0`` disables logging.
    """

    repo_id: str
    root: str | None = None
    output_dir: str = "outputs/actioncodec_tokenizer"
    steps: int = 1000
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 2e-4
    device: str = "cpu"
    seed: int = 42
    model_dim: int = 256
    action_dim: int = 7
    action_horizon: int = 20
    latent_horizon: int = 16
    codebook_size: int = 1024
    num_codebooks: int = 1
    alignment_weight: float = 0.1
    hard_alignment_weight: float = 0.0
    decoder_type: str = "diffusion"
    diffusion_config: dict[str, object] | None = None
    log_freq: int = 10

    def alignment_loss_config(self) -> dict[str, object]:
        """Map CLI weights onto the tokenizer ``loss_config`` keys.

        ``alignment_weight`` is the soft-DTW term used by the source recipe. Hard-DTW stays
        available behind ``hard_alignment_weight`` so the two mining paths are not mixed by
        accident.

        Returns:
            A dict suitable for :meth:`ActionCodecTokenizer.forward`.
        """
        return {
            "weight_align": self.hard_alignment_weight,
            "weight_chunk_align": self.alignment_weight,
            "chunk_align_positive_topk": 1,
            "chunk_align_gamma": 0.1,
        }

    def to_tokenizer_config(self) -> ActionCodecTokenizerConfig:
        """Expand these options into the full tokenizer configuration.

        Returns:
            A validated :class:`~lerobot.actioncodec.config.ActionCodecTokenizerConfig` whose
            unexposed fields keep their defaults.

        Raises:
            ValueError: If the requested horizons or codebook layout violate the semantic
                contract.
        """
        config = ActionCodecTokenizerConfig(
            action_dim=self.action_dim,
            horizon=self.action_horizon,
            latent_horizon=self.latent_horizon,
            model_dim=self.model_dim,
            codebook_size=self.codebook_size,
            num_codebooks=self.num_codebooks,
            decoder_type=self.decoder_type,
            diffusion_config=self.diffusion_config,
            learning_rate=self.learning_rate,
            device=self.device,
            steps=self.steps,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
        )
        config.validate()
        return config


def _delta_timestamps(fps: float, horizon: int) -> dict[str, list[float]]:
    """Build the ``delta_timestamps`` that make the dataset return whole windows.

    Args:
        fps: Frame rate of the dataset, used to convert step indices into seconds.
        horizon: Number of consecutive steps to gather per sample.

    Returns:
        Mapping from feature key to the relative timestamps, covering both ``action`` and
        ``observation.state`` so alignment losses see the same window as reconstruction.
    """
    timestamps = [index / float(fps) for index in range(horizon)]
    return {"action": timestamps, "observation.state": timestamps}


def _normalize_action(action: torch.Tensor, stats: dict[str, object]) -> torch.Tensor:
    """Match the policy processor's MEAN_STD action normalization."""
    if "mean" not in stats or "std" not in stats:
        raise ValueError("ActionCodec tokenizer training requires action mean/std statistics")
    mean = torch.as_tensor(stats["mean"], device=action.device, dtype=action.dtype)
    std = torch.as_tensor(stats["std"], device=action.device, dtype=action.dtype)
    if tuple(mean.shape) != (action.shape[-1],) or tuple(std.shape) != (action.shape[-1],):
        raise ValueError(
            f"Action statistics must have shape ({action.shape[-1]},), "
            f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )
    return (action - mean) / (std + 1e-8)


def train_semantic_tokenizer(cfg: SemanticTokenizerTrainConfig) -> Path:
    """Train a semantic tokenizer on a LeRobotDataset and save the resulting checkpoint.

    Actions are normalized with the dataset mean and std exactly the way the policy processor does
    at inference, then reconstructed through the quantizer for ``cfg.steps`` optimizer steps,
    cycling the dataloader as many times as needed. When either alignment weight is positive the
    per-step state deltas are also fed to the tokenizer so the semantic alignment loss is active.
    On completion the checkpoint directory holds ``model.safetensors``, ``model_config.json``,
    ``action_stats.json`` with the statistics used for normalization, and
    ``dataset_contract.json`` recording the dataset the tokens were defined against.

    Args:
        cfg: Training options, including the dataset to read and the output directory.

    Returns:
        Path of the directory the checkpoint was written to.

    Raises:
        ValueError: If the semantic contract is violated, if the dataset has no ``action``
            feature, if its action shape disagrees with ``cfg.action_dim``, or if the dataset
            carries no action mean/std statistics.
    """
    config = cfg.to_tokenizer_config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    metadata = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root)
    if "action" not in metadata.features:
        raise ValueError("LeRobotDataset must contain an 'action' feature")
    action_shape = tuple(metadata.features["action"]["shape"])
    if action_shape != (config.action_dim,):
        raise ValueError(f"action feature shape {action_shape} does not match action_dim={config.action_dim}")
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=cfg.root,
        delta_timestamps=_delta_timestamps(metadata.fps, config.horizon),
        return_uint8=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    model = ActionCodecTokenizer(
        action_dim=config.action_dim,
        window_size=config.horizon,
        model_dim=config.model_dim,
        num_tokens=config.latent_horizon,
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
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    model.train()
    iterator = iter(loader)
    for step in range(config.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        action = batch["action"].to(config.device).float()
        action = _normalize_action(action, metadata.stats.get("action", {}) if metadata.stats else {})
        delta_state = None
        if "observation.state" in batch and (cfg.alignment_weight > 0 or cfg.hard_alignment_weight > 0):
            state = batch["observation.state"].to(config.device).float()
            delta_state = torch.cat((state[:, 1:] - state[:, :-1], state[:, -1:]), dim=1)
        output = model(
            action,
            delta_state=delta_state,
            loss_config=cfg.alignment_loss_config(),
        )
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        optimizer.step()
        model.quantizer.apply_pending_codebook_updates()
        if cfg.log_freq > 0 and (step % cfg.log_freq == 0 or step + 1 == config.steps):
            logging.info(
                "step: %d loss: %.4f recon: %.4f vq: %.4f align: %.4f unique_codes: %d",
                step,
                output["loss"].item(),
                output["loss_recon"].item(),
                output["loss_vq"].item(),
                output["loss_align"].item(),
                int(output["indices"][..., 0].unique().numel()),
            )
    stats = metadata.stats.get("action", {}) if metadata.stats else {}
    contract = {
        "dataset_codebase_version": getattr(metadata, "_version", "v3.0").__str__(),
        "repo_id": cfg.repo_id,
        "fps": metadata.fps,
        "action_key": "action",
        "horizon": config.horizon,
        "latent_horizon": config.latent_horizon,
        "action_dim": config.action_dim,
    }
    output_dir = Path(cfg.output_dir)
    save_tokenizer_artifact(output_dir, model, config, action_stats=stats, dataset_contract=contract)
    return output_dir
