"""Conditional diffusion decoder used by the source ActionCodec tokenizer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .perceiver import (
    CrossAttentionBlock,
    EmbodimentEmbedding,
    LatentTransformer,
    PositionalEmbedding,
    _prepare_embodiment_metadata,
    _validate_embodiment_ids,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype).expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = int(num_steps)
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float32) / float(steps)
    alpha_bar = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0].clamp_min(1e-12)
    return (1.0 - alpha_bar[1:] / alpha_bar[:-1].clamp_min(1e-12)).clamp(1e-5, 0.999)


def _linear_beta_schedule(num_steps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 2e-2, int(num_steps), dtype=torch.float32)


def _timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = max(dim // 2, 1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(embedding, (0, max(0, dim - embedding.shape[-1])))[:, :dim]


class ActionTokenConditioner(nn.Module):
    """Perceiver conditioner that maps latent tokens to a diffusion sequence."""

    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        window_size: int,
        num_heads: int,
        latent_depth: int,
        num_cross_layers: int = 1,
        dropout: float = 0.0,
        embodiment_config: dict[str, Any] | None = None,
        use_soft_prompt: bool = True,
        use_latent_self_attn: bool = True,
        share_latent_transformer: bool = False,
        share_cross_attn: bool = False,
    ) -> None:
        """Build the conditioner.

        Args:
            action_dim: Action dimension of the default embodiment.
            model_dim: Width of the transformer, shared with the latent tokens.
            window_size: Maximum number of conditioning steps, i.e. the longest action horizon the
                diffusion decoder can produce.
            num_heads: Number of attention heads.
            latent_depth: Depth of each latent self-attention stack.
            num_cross_layers: Number of cross-attention rounds over the latent tokens.
            dropout: Dropout probability used throughout.
            embodiment_config: Mapping from embodiment name to ``action_dim``, ``freq`` and
                ``duration``. Defaults to a single ``"default"`` embodiment at 20 Hz whose duration
                covers exactly ``window_size`` steps.
            use_soft_prompt: Use per-embodiment learned output queries instead of a shared bank.
            use_latent_self_attn: Apply latent self-attention after each cross-attention round.
            share_latent_transformer: Reuse one latent transformer across all rounds.
            share_cross_attn: Reuse one cross-attention block across all rounds.
        """
        super().__init__()
        embodiment_config = embodiment_config or {
            "default": {
                "action_dim": action_dim,
                "freq": 20.0,
                "duration": float(window_size) / 20.0,
            }
        }
        (
            self.embodiment_names,
            _,
            action_dims,
            freqs,
            durations,
        ) = _prepare_embodiment_metadata(embodiment_config, action_dim)
        self.register_buffer("embodiment_action_dims", action_dims.long(), persistent=False)
        self.register_buffer("embodiment_freqs", freqs.float(), persistent=False)
        self.register_buffer("embodiment_durations", durations.float(), persistent=False)
        self.window_size = int(window_size)
        self.use_soft_prompt = bool(use_soft_prompt)
        if self.use_soft_prompt:
            self.cls_tokens = EmbodimentEmbedding(len(self.embodiment_names), window_size, model_dim)
        else:
            self.output_queries = nn.Parameter(torch.randn(window_size, model_dim) * 0.02)
        self.pos_emb_q = PositionalEmbedding(model_dim, "fourier")
        self.pos_emb_kv = PositionalEmbedding(model_dim, "sincos")
        self.num_cross_layers = int(num_cross_layers)
        self.use_latent_self_attn = bool(use_latent_self_attn)
        self.share_latent_transformer = bool(share_latent_transformer)
        self.share_cross_attn = bool(share_cross_attn)
        if share_cross_attn:
            self.cross_shared = CrossAttentionBlock(model_dim, num_heads, dropout)
            self.cross_blocks = None
        else:
            self.cross_blocks = nn.ModuleList(
                [CrossAttentionBlock(model_dim, num_heads, dropout) for _ in range(num_cross_layers)]
            )
            self.cross_shared = None
        if use_latent_self_attn and share_latent_transformer:
            self.latent_shared = LatentTransformer(model_dim, num_heads, latent_depth, dropout)
            self.latent_blocks = None
        elif use_latent_self_attn:
            self.latent_blocks = nn.ModuleList(
                [
                    LatentTransformer(model_dim, num_heads, latent_depth, dropout)
                    for _ in range(num_cross_layers)
                ]
            )
            self.latent_shared = None
        else:
            self.latent_shared = None
            self.latent_blocks = None

    def forward(self, latents: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Expand latent tokens into one conditioning vector per action step.

        Args:
            latents: Quantized latent tokens of shape ``[B, num_tokens, model_dim]``.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Per-step conditioning of shape ``[B, horizon, model_dim]``, where ``horizon`` is
            ``round(freq * duration)`` for the batch's embodiment.

        Raises:
            ValueError: If the batch mixes embodiments with different horizons, or if the requested
                horizon exceeds ``window_size``.
        """
        if embodiment_ids is None:
            embodiment_ids = latents.new_zeros((latents.shape[0],), dtype=torch.long)
        ids = _validate_embodiment_ids(embodiment_ids, len(self.embodiment_names), "embodiment_ids")
        freq = self.embodiment_freqs.index_select(0, ids)
        duration = self.embodiment_durations.index_select(0, ids)
        horizon = torch.round(freq * duration).long()
        if not torch.equal(horizon, horizon[:1].expand_as(horizon)):
            raise ValueError("Mixed embodiment horizons are not supported")
        length = int(horizon[0])
        if length > self.window_size:
            raise ValueError("Requested horizon exceeds conditioner window_size")
        memory = latents + self.pos_emb_kv(latents)
        query = (
            self.cls_tokens(ids)[:, :length]
            if self.use_soft_prompt
            else self.output_queries[:length].unsqueeze(0).expand(latents.shape[0], -1, -1)
        )
        query = query + self.pos_emb_q(query, freq=freq, duration=duration)
        for index in range(self.num_cross_layers):
            cross = self.cross_shared if self.share_cross_attn else self.cross_blocks[index]
            query = cross(query, memory)
            if self.use_latent_self_attn:
                latent = self.latent_shared if self.share_latent_transformer else self.latent_blocks[index]
                query = latent(query)
        return query


class _FiLMTemporalBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=int(kernel_size), padding=padding)
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=int(kernel_size), padding=padding)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x
        scale, shift, gate = self.modulation(condition).chunk(3, dim=-1)
        y = self.norm1(x) * (1.0 + scale) + shift
        y = self.dropout(F.silu(self.conv1(y.transpose(1, 2)).transpose(1, 2)))
        y = self.conv2(self.norm2(y).transpose(1, 2)).transpose(1, 2)
        return residual + torch.tanh(gate) * y


class ActionDiffusionDenoiser(nn.Module):
    """Temporal convolutional network that predicts the clean action window from a noisy one.

    The per-step conditioning from the token conditioner and the diffusion timestep embedding are
    merged into a single FiLM modulation signal that is applied inside every temporal block.
    """

    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        denoiser_layers: int,
        kernel_size: int,
        dropout: float,
        embodiment_action_dims: torch.Tensor,
    ) -> None:
        """Build the denoiser.

        Args:
            action_dim: Action dimension of the default embodiment; widened if some embodiment in
                ``embodiment_action_dims`` needs more channels.
            model_dim: Hidden width, matching the conditioning produced by the token conditioner.
            denoiser_layers: Number of stacked FiLM-modulated temporal blocks.
            kernel_size: Temporal convolution kernel size; the receptive field over action steps
                grows with it.
            dropout: Dropout probability inside each block.
            embodiment_action_dims: Integer tensor of shape ``[num_embodiments]`` giving each
                embodiment's action dimension; one output head is created per entry.
        """
        super().__init__()
        self.max_action_dim = max(int(action_dim), int(embodiment_action_dims.max().item()))
        self.input_proj = nn.Linear(self.max_action_dim, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4), nn.SiLU(), nn.Linear(model_dim * 4, model_dim)
        )
        self.cond_proj = nn.Linear(model_dim * 2, model_dim)
        self.blocks = nn.ModuleList(
            [_FiLMTemporalBlock(model_dim, kernel_size, dropout) for _ in range(denoiser_layers)]
        )
        self.out_norm = nn.LayerNorm(model_dim)
        self.head = nn.ModuleList([nn.Linear(model_dim, int(dim.item())) for dim in embodiment_action_dims])

    def forward(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        embodiment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the clean action window behind a noisy one.

        Args:
            noisy_action: Noised actions of shape ``[B, T, max_action_dim]``.
            timesteps: Diffusion step index per sample, shape ``[B]``.
            condition: Per-step conditioning of shape ``[B, T, model_dim]``.
            embodiment_ids: Integer tensor of shape ``[B]`` selecting the output head per sample.

        Returns:
            Predicted clean actions of shape ``[B, T, max_action_dim]``, zero past each
            embodiment's own action dimension.
        """
        time = _timestep_embedding(timesteps, condition.shape[-1]).to(condition)
        time = self.time_mlp(time).unsqueeze(1).expand(-1, condition.shape[1], -1)
        modulation = self.cond_proj(torch.cat((condition, time), dim=-1))
        x = self.input_proj(noisy_action)
        for block in self.blocks:
            x = block(x, modulation)
        x = self.out_norm(x)
        # Allocate from the Linear head output, not from LayerNorm ``x``. Under AMP, LayerNorm
        # stays fp32 while Linear is autocast to fp16; indexed copy then requires matching dtypes.
        output = None
        for index, head in enumerate(self.head):
            mask = embodiment_ids == index
            if not mask.any():
                continue
            decoded = head(x[mask])
            if output is None:
                output = decoded.new_zeros((x.shape[0], x.shape[1], self.max_action_dim))
            output[mask, :, : decoded.shape[-1]] = decoded
        if output is None:
            raise ValueError("No valid embodiment denoiser outputs were produced for the batch")
        return output


class ActionDiffusionDecoder(nn.Module):
    """DDPM x0-prediction decoder with source-compatible train/sample methods."""

    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        window_size: int,
        num_heads: int,
        latent_depth: int,
        num_cross_layers: int = 1,
        dropout: float = 0.0,
        embodiment_config: dict[str, Any] | None = None,
        use_soft_prompt: bool = True,
        use_latent_self_attn: bool = True,
        share_latent_transformer: bool = False,
        share_cross_attn: bool = False,
        num_train_steps: int = 1000,
        num_sample_steps: int = 27,
        beta_schedule: str = "cosine",
        predict_target: str = "x0",
        denoiser_layers: int = 6,
        kernel_size: int = 5,
    ) -> None:
        """Build the conditioner, the denoiser and the diffusion noise schedule.

        Args:
            action_dim: Action dimension of the default embodiment.
            model_dim: Width shared by the latent tokens, the conditioner and the denoiser.
            window_size: Longest action horizon the decoder can emit.
            num_heads: Number of attention heads in the conditioner.
            latent_depth: Depth of each latent self-attention stack in the conditioner.
            num_cross_layers: Number of cross-attention rounds in the conditioner.
            dropout: Dropout probability used throughout.
            embodiment_config: Mapping from embodiment name to ``action_dim``, ``freq`` and
                ``duration``, forwarded to the conditioner.
            use_soft_prompt: Use per-embodiment learned output queries in the conditioner.
            use_latent_self_attn: Apply latent self-attention in the conditioner.
            share_latent_transformer: Reuse one latent transformer across conditioner rounds.
            share_cross_attn: Reuse one cross-attention block across conditioner rounds.
            num_train_steps: Length of the noise schedule; training samples a timestep uniformly
                from ``[0, num_train_steps)``.
            num_sample_steps: Number of denoising steps at inference. They are spread evenly over
                the training schedule, trading sample quality for latency.
            beta_schedule: Shape of the noise schedule, either ``"cosine"`` or ``"linear"``.
            predict_target: Denoiser parameterization; only ``"x0"`` (predict the clean action) is
                implemented.
            denoiser_layers: Number of temporal blocks in the denoiser.
            kernel_size: Temporal convolution kernel size in the denoiser.

        Raises:
            ValueError: If ``predict_target`` is not ``"x0"``, or ``beta_schedule`` is unsupported.
        """
        super().__init__()
        if predict_target != "x0":
            raise ValueError(f"Unsupported diffusion predict_target: {predict_target}")
        if beta_schedule not in {"cosine", "linear"}:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        self.conditioner = ActionTokenConditioner(
            model_dim=model_dim,
            window_size=window_size,
            action_dim=action_dim,
            num_heads=num_heads,
            latent_depth=latent_depth,
            num_cross_layers=num_cross_layers,
            dropout=dropout,
            embodiment_config=embodiment_config,
            use_soft_prompt=use_soft_prompt,
            use_latent_self_attn=use_latent_self_attn,
            share_latent_transformer=share_latent_transformer,
            share_cross_attn=share_cross_attn,
        )
        self.embodiment_names = self.conditioner.embodiment_names
        self.embodiment_action_dims = self.conditioner.embodiment_action_dims
        self.max_action_dim = int(self.embodiment_action_dims.max().item())
        self.denoiser = ActionDiffusionDenoiser(
            action_dim, model_dim, denoiser_layers, kernel_size, dropout, self.embodiment_action_dims
        )
        betas = (
            _cosine_beta_schedule(num_train_steps)
            if beta_schedule == "cosine"
            else _linear_beta_schedule(num_train_steps)
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))
        self.num_train_steps = int(num_train_steps)
        self.num_sample_steps = int(num_sample_steps)

    def _prepare_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] < self.max_action_dim:
            return F.pad(action, (0, self.max_action_dim - action.shape[-1]))
        return action[..., : self.max_action_dim]

    def _valid_action_mask(self, reference: torch.Tensor, embodiment_ids: torch.Tensor) -> torch.Tensor:
        dims = self.embodiment_action_dims.to(reference.device).index_select(0, embodiment_ids.long())
        return torch.arange(reference.shape[-1], device=reference.device).view(1, 1, -1) < dims.view(-1, 1, 1)

    def forward_train(
        self,
        latents: torch.Tensor,
        action: torch.Tensor,
        embodiment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one training step of the diffusion objective.

        A single timestep is drawn per sample, the ground-truth window is noised accordingly, and
        the denoiser is scored on how well it recovers the clean window. The loss is masked so that
        the padding channels of narrower embodiments do not contribute.

        Args:
            latents: Quantized latent tokens of shape ``[B, num_tokens, model_dim]``.
            action: Ground-truth action window of shape ``[B, T, A]``; it is zero-padded or
                truncated to ``max_action_dim``.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Tuple of the predicted clean actions of shape ``[B, T, max_action_dim]`` and the scalar
            masked smooth-L1 reconstruction loss.
        """
        if embodiment_ids is None:
            embodiment_ids = latents.new_zeros((latents.shape[0],), dtype=torch.long)
        condition = self.conditioner(latents, embodiment_ids)
        x0 = self._prepare_action(action.to(condition))
        timesteps = torch.randint(0, self.num_train_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        alpha = self.alphas_cumprod.index_select(0, timesteps).to(x0)
        noisy = alpha.sqrt().view(-1, 1, 1) * x0 + (1.0 - alpha).sqrt().view(-1, 1, 1) * noise
        pred = self.denoiser(noisy, timesteps, condition, embodiment_ids)
        loss = _masked_mean(
            F.smooth_l1_loss(pred, x0, reduction="none"),
            self._valid_action_mask(pred, embodiment_ids),
        )
        return pred, loss

    @torch.no_grad()
    def forward(self, latents: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Sample an action window from latent tokens with DDIM-style deterministic denoising.

        Args:
            latents: Quantized latent tokens of shape ``[B, num_tokens, model_dim]``.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Sampled actions of shape ``[B, horizon, max_action_dim]``, where ``horizon`` comes from
            the conditioner and the channels past each embodiment's action dimension are zero.
        """
        if embodiment_ids is None:
            embodiment_ids = latents.new_zeros((latents.shape[0],), dtype=torch.long)
        condition = self.conditioner(latents, embodiment_ids)
        x = torch.zeros(
            condition.shape[0],
            condition.shape[1],
            self.max_action_dim,
            device=condition.device,
            dtype=condition.dtype,
        )
        steps = (
            torch.linspace(self.num_train_steps - 1, 0, steps=max(1, self.num_sample_steps), device=x.device)
            .round()
            .long()
            .unique(sorted=True)
            .flip(0)
        )
        for index, step in enumerate(steps):
            timestep = torch.full((x.shape[0],), int(step.item()), device=x.device, dtype=torch.long)
            x0 = self.denoiser(x, timestep, condition, embodiment_ids)
            if index == len(steps) - 1:
                return x0
            next_step = steps[index + 1]
            alpha = self.alphas_cumprod[step].to(x)
            next_alpha = self.alphas_cumprod[next_step].to(x)
            noise = (x - alpha.sqrt() * x0) / (1.0 - alpha).sqrt().clamp_min(1e-6)
            x = next_alpha.sqrt() * x0 + (1.0 - next_alpha).sqrt() * noise
        return x
