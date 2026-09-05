"""Small, independent semantic tokenizer trainer backed by LeRobotDataset v3."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from lerobot.common.tensorboard_utils import TensorBoardLogger
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

from .config import ActionCodecTokenizerConfig, save_tokenizer_artifact
from .metrics import CodebookOccupancyMeter
from .models.tokenizer import ActionCodecTokenizer


@dataclass
class SemanticTokenizerTrainConfig:
    """Command-line surface of the standalone semantic tokenizer training run.

    Defaults follow ``../actioncodec`` matched_h20 (diffusion decoder, 8 shared cross-attn
    rounds, ``model_dim=512``, ``vq_beta=1.0``, batch 512, 20k steps, stride 4). CLIP is off.
    ``device`` stays ``cpu`` so unit tests do not require a GPU.

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
        encoder_cross_layers: Cross-attention rounds in the encoder.
        decoder_cross_layers: Cross-attention rounds in the decoder.
        share_encoder_latent_transformer: Reuse one encoder latent transformer across rounds.
        share_decoder_latent_transformer: Reuse one decoder latent transformer across rounds.
        share_encoder_cross_attn: Reuse one encoder cross-attention block across rounds.
        share_decoder_cross_attn: Reuse one decoder cross-attention block across rounds.
        vq_beta: Commitment weight of the VQ loss.
        use_vl_embedder: Build the unused CLIP embedder. Keep ``False``.
        window_stride: Episode-local stride between full horizon window starts.
        adam_beta1: AdamW ``betas[0]``.
        adam_beta2: AdamW ``betas[1]``.
        grad_clip: Global gradient-norm clip; ``0`` disables clipping.
        lr_warmup_steps: Linear warmup length before cosine decay.
        lr_min_ratio: Cosine floor as a fraction of ``learning_rate``.
        amp: Enable CUDA autocast + GradScaler. Ignored on CPU.
        occupancy_window: Number of recent steps aggregated for rolling codebook usage.
        alignment_weight: Weight of the **soft-DTW** chunk alignment loss
            (``weight_chunk_align``). The default ``0.1`` matches the source tokenizer recipe.
            ``0`` turns alignment off.
        hard_alignment_weight: Weight of the older hard-DTW alignment (``weight_align``). Default
            ``0``; keep it off unless you explicitly want both.
        overlap_consistency_weight: Weight of the paired-window physical overlap loss.
        overlap_shift: Shift between paired windows in frames.
        physical_recon_weight: Weight of physical action reconstruction loss.
        physical_velocity_weight: Weight of physical action velocity loss.
        physical_unit_scale: Raw action units represented by one loss unit.
        continuous_action_indices: Action dimensions included in physical smoothness losses.
        decoder_type: ``"diffusion"`` (default) or ``"perceiver"``.
        diffusion_config: Extra keyword arguments for the diffusion decoder.
        log_freq: Number of steps between training log lines; ``0`` disables logging.
    """

    repo_id: str
    root: str | None = None
    output_dir: str = "outputs/actioncodec_tokenizer"
    steps: int = 20000
    batch_size: int = 512
    num_workers: int = 8
    learning_rate: float = 2e-4
    device: str = "cpu"
    seed: int = 42
    model_dim: int = 512
    action_dim: int = 7
    action_horizon: int = 20
    latent_horizon: int = 16
    codebook_size: int = 1024
    quantizer_type: str = "vq"
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    num_codebooks: int = 1
    encoder_cross_layers: int = 8
    decoder_cross_layers: int = 8
    share_encoder_latent_transformer: bool = True
    share_decoder_latent_transformer: bool = True
    share_encoder_cross_attn: bool = True
    share_decoder_cross_attn: bool = True
    vq_beta: float = 1.0
    use_vl_embedder: bool = False
    window_stride: int = 4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    lr_warmup_steps: int = 1000
    lr_min_ratio: float = 0.1
    amp: bool = True
    amp_dtype: str = "float16"
    occupancy_window: int = 2000
    alignment_weight: float = 0.1
    hard_alignment_weight: float = 0.0
    overlap_consistency_weight: float = 0.0
    overlap_shift: int = 16
    physical_recon_weight: float = 0.0
    physical_velocity_weight: float = 0.0
    physical_unit_scale: float = 1.0
    continuous_action_indices: tuple[int, ...] | None = None
    decoder_type: str = "diffusion"
    diffusion_config: dict[str, object] | None = None
    log_freq: int = 10

    def __post_init__(self) -> None:
        """Normalize and validate the optional physical continuity settings."""
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16")
        if self.continuous_action_indices is not None:
            self.continuous_action_indices = tuple(int(index) for index in self.continuous_action_indices)
        for name in ("overlap_consistency_weight", "physical_recon_weight", "physical_velocity_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.overlap_consistency_weight > 0 and not 0 < self.overlap_shift < self.action_horizon:
            raise ValueError(
                "overlap_shift must satisfy 0 < overlap_shift < action_horizon when overlap is enabled"
            )
        if self.overlap_consistency_weight > 0 and self.batch_size % 2:
            raise ValueError("batch_size must be even when overlap consistency is enabled")
        if self.physical_unit_scale <= 0:
            raise ValueError("physical_unit_scale must be positive")
        if self.continuous_action_indices is not None and any(
            index < 0 or index >= self.action_dim for index in self.continuous_action_indices
        ):
            raise ValueError("continuous_action_indices must be valid action dimensions")

    def alignment_loss_config(self) -> dict[str, object]:
        """Map CLI weights onto the tokenizer ``loss_config`` keys.

        ``alignment_weight`` is the soft-DTW term used by the source recipe. Hard-DTW stays
        available behind ``hard_alignment_weight`` so the two mining paths are not mixed by
        accident. ``chunk_align_dtw_backend`` stays ``auto`` because this tree's CUDA DTW
        extension is not wired; ``max_candidate_pairs=1024`` caps pair scoring at batch 512.

        Returns:
            A dict suitable for :meth:`ActionCodecTokenizer.forward`.
        """
        return {
            "weight_align": self.hard_alignment_weight,
            "weight_chunk_align": self.alignment_weight,
            "chunk_align_positive_topk": 1,
            "chunk_align_gamma": 0.1,
            "chunk_align_pair_batch_size": 8192,
            "chunk_align_dtw_backend": "auto",
            "chunk_align_max_candidate_pairs": 1024,
            "chunk_align_min_delta_norm": 1e-6,
            "chunk_align_normalize_delta": True,
            "overlap_consistency_weight": self.overlap_consistency_weight,
            "overlap_shift": self.overlap_shift,
            "physical_recon_weight": self.physical_recon_weight,
            "physical_velocity_weight": self.physical_velocity_weight,
            "physical_unit_scale": self.physical_unit_scale,
            "continuous_action_indices": self.continuous_action_indices,
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
            quantizer_type=self.quantizer_type,
            fsq_levels=self.fsq_levels,
            codebook_size=self.codebook_size,
            num_codebooks=self.num_codebooks,
            encoder_cross_layers=self.encoder_cross_layers,
            decoder_cross_layers=self.decoder_cross_layers,
            share_encoder_latent_transformer=self.share_encoder_latent_transformer,
            share_decoder_latent_transformer=self.share_decoder_latent_transformer,
            share_encoder_cross_attn=self.share_encoder_cross_attn,
            share_decoder_cross_attn=self.share_decoder_cross_attn,
            vq_beta=self.vq_beta,
            use_vl_embedder=self.use_vl_embedder,
            window_stride=self.window_stride,
            overlap_consistency_weight=self.overlap_consistency_weight,
            overlap_shift=self.overlap_shift,
            physical_recon_weight=self.physical_recon_weight,
            physical_velocity_weight=self.physical_velocity_weight,
            physical_unit_scale=self.physical_unit_scale,
            continuous_action_indices=self.continuous_action_indices,
            decoder_type=self.decoder_type,
            diffusion_config=self.diffusion_config,
            learning_rate=self.learning_rate,
            amp_dtype=self.amp_dtype,
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


def _as_int_list(values: object) -> list[int]:
    """Coerce an episode-index column into a flat list of ints."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"episode index column must be a sequence, got {type(values)!r}")
    out: list[int] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            value = value[0]
        out.append(int(value))
    return out


def _window_start_indices(
    from_index: Sequence[int],
    to_index: Sequence[int],
    horizon: int,
    stride: int,
) -> list[int]:
    """Return in-episode starts of full ``horizon`` windows, strided like matched-h20.

    ``to_index`` is exclusive. A window starting at ``t`` needs ``[t, t+horizon)``, so the last
    legal start in an episode is ``end - horizon``. This drops LeRobot's padded tail windows.

    Args:
        from_index: Inclusive start frame of each episode in the map-style dataset.
        to_index: Exclusive end frame of each episode.
        horizon: Window length in frames.
        stride: Step between consecutive window starts.

    Returns:
        Flat list of dataset indices to sample.

    Raises:
        ValueError: If ``horizon`` or ``stride`` is not positive, if the two columns differ in
            length, or if no full window exists.
    """
    if horizon <= 0 or stride <= 0:
        raise ValueError("horizon and stride must be positive")
    starts: list[int] = []
    for start, end in zip(_as_int_list(from_index), _as_int_list(to_index), strict=True):
        last = int(end) - int(horizon)
        starts.extend(range(int(start), last + 1, int(stride)))
    if not starts:
        raise ValueError(
            f"No full horizon={horizon} windows remain with stride={stride}; check episode lengths"
        )
    return starts


def _paired_window_start_indices(
    from_index: Sequence[int],
    to_index: Sequence[int],
    horizon: int,
    shift: int,
    stride: int,
) -> list[int]:
    """Return starts whose ``t:t+horizon`` and ``t+shift:t+shift+horizon`` fit one episode."""
    if horizon <= 0 or stride <= 0 or not 0 < shift < horizon:
        raise ValueError("horizon and stride must be positive and shift must satisfy 0 < shift < horizon")
    starts: list[int] = []
    for start, end in zip(_as_int_list(from_index), _as_int_list(to_index), strict=True):
        last = int(end) - int(horizon) - int(shift)
        starts.extend(range(int(start), last + 1, int(stride)))
    if not starts:
        raise ValueError(
            f"No full paired horizon={horizon}, shift={shift} windows remain; check episode lengths"
        )
    return starts


class PairedWindowDataset(torch.utils.data.Dataset):
    """Expose two episode-local action windows as ``[A, B]`` for a paired batch."""

    def __init__(self, dataset: torch.utils.data.Dataset, starts: Sequence[int], shift: int) -> None:
        """Build a pair view over a horizon-window dataset and absolute frame starts."""
        self.dataset = dataset
        self.starts = list(starts)
        self.shift = int(shift)

    def __len__(self) -> int:
        """Return the number of paired window starts."""
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return ``t`` and ``t+shift`` stacked along a pair dimension."""
        start = self.starts[index]
        first = self.dataset[start]
        second = self.dataset[start + self.shift]
        item = {"action": torch.stack((first["action"], second["action"]))}
        if "observation.state" in first and "observation.state" in second:
            item["observation.state"] = torch.stack((first["observation.state"], second["observation.state"]))
        return item


def _resolve_window_indices(dataset: object, horizon: int, stride: int) -> list[int] | None:
    """Read episode bounds off a LeRobotDataset when they exist; otherwise return ``None``."""
    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is None:
        return None
    try:
        return _window_start_indices(
            episodes["dataset_from_index"],
            episodes["dataset_to_index"],
            horizon=horizon,
            stride=stride,
        )
    except (KeyError, TypeError):
        return None


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float,
) -> SequentialLR | CosineAnnealingLR:
    """Linear warmup then cosine decay, matching the ActionCodec tokenizer trainer."""
    total_steps = max(1, int(steps))
    warmup = max(0, int(warmup_steps))
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup),
        eta_min=float(base_lr) * float(min_lr_ratio),
    )
    if warmup <= 0:
        return cosine
    linear = LinearLR(
        optimizer,
        start_factor=1.0 / float(warmup),
        end_factor=1.0,
        total_iters=warmup,
    )
    return SequentialLR(optimizer, [linear, cosine], milestones=[warmup])


def train_semantic_tokenizer(cfg: SemanticTokenizerTrainConfig) -> Path:
    """Train a semantic tokenizer on a LeRobotDataset and save the resulting checkpoint.

    Actions are normalized with the dataset mean and std exactly the way the policy processor does
    at inference, then reconstructed through the quantizer for ``cfg.steps`` optimizer steps,
    cycling the dataloader as many times as needed. When overlap consistency is enabled, the
    dataloader returns episode-local ``t``/``t+overlap_shift`` pairs and concatenates them as
    ``[A_batch, B_batch]`` before the model forward. When ``alignment_weight`` is positive the
    normalized action window is fed to soft-DTW; hard-DTW still uses state deltas when
    ``hard_alignment_weight`` is positive. Scalars are written to ``{output_dir}/tb``.
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
        decode_videos=False,
    )
    logging.info("tokenizer dataloader skips video decode (action/state windows only)")
    overlap_enabled = config.overlap_consistency_weight > 0
    if overlap_enabled:
        episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
        if episodes is None:
            raise ValueError("Overlap consistency requires episode metadata for paired windows")
        pair_indices = _paired_window_start_indices(
            episodes["dataset_from_index"],
            episodes["dataset_to_index"],
            horizon=config.horizon,
            shift=config.overlap_shift,
            stride=config.window_stride,
        )
        train_data = PairedWindowDataset(dataset, pair_indices, config.overlap_shift)
        loader_batch_size = config.batch_size // 2
    else:
        window_indices = _resolve_window_indices(dataset, config.horizon, config.window_stride)
        train_data = Subset(dataset, window_indices) if window_indices is not None else dataset
        loader_batch_size = config.batch_size
    drop_last = len(train_data) >= loader_batch_size
    pin_memory = torch.device(config.device).type == "cuda"
    loader = DataLoader(
        train_data,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    if len(loader) == 0:
        raise ValueError(
            f"Tokenizer dataloader is empty: {len(train_data)} samples, batch_size={loader_batch_size}"
        )
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
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        steps=config.steps,
        warmup_steps=cfg.lr_warmup_steps,
        base_lr=config.learning_rate,
        min_lr_ratio=cfg.lr_min_ratio,
    )
    device = torch.device(config.device)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = getattr(torch, cfg.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    occupancy = CodebookOccupancyMeter(config.codebook_size, window=cfg.occupancy_window)
    output_dir = Path(cfg.output_dir)
    tb_logger = TensorBoardLogger.from_log_dir(output_dir / "tb")
    stats = metadata.stats.get("action", {}) if metadata.stats else {}
    action_std = torch.as_tensor(stats["std"], device=device, dtype=torch.float32)
    model.train()
    iterator = iter(loader)
    try:
        for step in range(config.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            action_batch = batch["action"].to(config.device, non_blocking=pin_memory).float()
            if overlap_enabled:
                action = torch.cat((action_batch[:, 0], action_batch[:, 1]), dim=0)
            else:
                action = action_batch
            action = _normalize_action(action, stats)
            delta_state = None
            if cfg.hard_alignment_weight > 0:
                state_batch = batch["observation.state"].to(config.device, non_blocking=pin_memory).float()
                state = (
                    torch.cat((state_batch[:, 0], state_batch[:, 1]), dim=0)
                    if overlap_enabled
                    else state_batch
                )
                delta_state = torch.cat((state[:, 1:] - state[:, :-1], state[:, -1:]), dim=1)
            elif cfg.alignment_weight > 0:
                delta_state = action
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                model_kwargs = {
                    "delta_state": delta_state,
                    "loss_config": cfg.alignment_loss_config(),
                }
                model_kwargs["action_std"] = action_std
                output = model(action, **model_kwargs)
                loss = output["loss"]
            if config.quantizer_type == "semantic_fsq" and not torch.isfinite(loss):
                raise FloatingPointError(f"FSQ non-finite loss at step {step}")
            if not torch.isfinite(loss):
                model.quantizer.discard_pending_codebook_updates()
                logging.warning("step: %d skipped non-finite loss", step)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            else:
                gradient_norm = torch.sqrt(
                    sum(
                        parameter.grad.detach().float().pow(2).sum()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                )
            if config.quantizer_type == "semantic_fsq" and not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"FSQ non-finite gradient at step {step}")
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            step_applied = (not use_amp) or float(scaler.get_scale()) >= scale_before
            if step_applied:
                scheduler.step()
                model.quantizer.apply_pending_codebook_updates()
            else:
                model.quantizer.discard_pending_codebook_updates()
            occupancy_stats = occupancy.update(output["indices"])
            if cfg.log_freq > 0 and (step % cfg.log_freq == 0 or step + 1 == config.steps):
                logging.info(
                    "step: %d loss: %.4f recon: %.4f vq: %.4f align: %.4f "
                    "phys_recon: %.4f phys_velocity: %.4f overlap: %.4f grad_norm: %.4f "
                    "unique_codes_batch: %d occupied_window: %d occupied_total: %d usage_total: %.4f",
                    step,
                    float(output["loss"].item()),
                    float(output["loss_recon"].item()),
                    float(output["loss_vq"].item()),
                    float(output["loss_align"].item()),
                    float(output["loss_phys_recon"].item()),
                    float(output["loss_phys_velocity"].item()),
                    float(output["loss_overlap"].item()),
                    float(gradient_norm.item()),
                    int(occupancy_stats["unique_codes_batch"]),
                    int(occupancy_stats["codebook_occupied_window"]),
                    int(occupancy_stats["codebook_occupied_total"]),
                    occupancy_stats["codebook_usage_total"],
                )
                tb_logger.log_dict(
                    {
                        "loss": float(output["loss"].item()),
                        "loss_recon": float(output["loss_recon"].item()),
                        "loss_vq": float(output["loss_vq"].item()),
                        "loss_align": float(output["loss_align"].item()),
                        "loss_phys_recon": float(output["loss_phys_recon"].item()),
                        "loss_phys_velocity": float(output["loss_phys_velocity"].item()),
                        "loss_overlap": float(output["loss_overlap"].item()),
                        "weighted_phys_recon": float(output["weighted_phys_recon"].item()),
                        "weighted_phys_velocity": float(output["weighted_phys_velocity"].item()),
                        "weighted_overlap": float(output["weighted_overlap"].item()),
                        "physical_recon_mae": float(output["physical_recon_mae"].item()),
                        "physical_velocity_mae": float(output["physical_velocity_mae"].item()),
                        "overlap_mae": float(output["overlap_mae"].item()),
                        "grad_norm": float(gradient_norm.item()),
                        **occupancy_stats,
                    },
                    step=step,
                )
    finally:
        tb_logger.close()
    contract = {
        "dataset_codebase_version": getattr(metadata, "_version", "v3.0").__str__(),
        "repo_id": cfg.repo_id,
        "fps": metadata.fps,
        "action_key": "action",
        "horizon": config.horizon,
        "latent_horizon": config.latent_horizon,
        "action_dim": config.action_dim,
        "window_stride": cfg.window_stride,
        "overlap_shift": config.overlap_shift,
        "continuous_action_indices": config.continuous_action_indices,
    }
    save_tokenizer_artifact(output_dir, model, config, action_stats=stats, dataset_contract=contract)
    return output_dir
