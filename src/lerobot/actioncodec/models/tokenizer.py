"""ActionCodec tokenizer model and semantic training losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ..losses.contrastive import symmetric_info_nce
from ..losses.semantic_dtw import chunk_hard_dtw_targets, semantic_contrastive_loss
from ..losses.soft_dtw import chunk_soft_dtw_targets
from .diffusion_decoder import ActionDiffusionDecoder
from .fsq import SemanticFSQQuantizer
from .perceiver import ActionPerceiverDecoder, ActionPerceiverEncoder
from .quantizer import ResidualVectorQuantizer
from .vl_embed import VisualLanguageEmbedder


def _continuous_indices(
    action_dim: int, indices: tuple[int, ...] | list[int] | None, device: torch.device
) -> torch.Tensor:
    if indices is None:
        return torch.arange(action_dim, device=device)
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def _physical_loss_and_mae(
    error: torch.Tensor,
    action_indices: torch.Tensor,
    unit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = error.index_select(-1, action_indices)
    loss = F.smooth_l1_loss(selected / float(unit_scale), torch.zeros_like(selected))
    return loss, selected.abs().mean()


def _overlap_loss_and_mae(
    reconstruction: torch.Tensor,
    action_std: torch.Tensor,
    shift: int,
    action_indices: torch.Tensor,
    unit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_batch = reconstruction.shape[0] // 2
    error = (reconstruction[:pair_batch, shift:] - reconstruction[pair_batch:, :-shift]).float() * action_std
    return _physical_loss_and_mae(error, action_indices, unit_scale)


class ActionCodecTokenizer(nn.Module):
    """Encode a continuous action horizon into semantic discrete tokens."""

    def __init__(
        self,
        action_dim: int = 7,
        window_size: int = 20,
        model_dim: int = 512,
        num_tokens: int = 16,
        codebook_size: int = 1024,
        num_codebooks: int = 1,
        num_heads: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        encoder_cross_layers: int = 8,
        decoder_cross_layers: int = 8,
        use_encoder_latent_self_attn: bool = True,
        use_decoder_latent_self_attn: bool = True,
        share_encoder_latent_transformer: bool = True,
        share_decoder_latent_transformer: bool = True,
        share_encoder_cross_attn: bool = True,
        share_decoder_cross_attn: bool = True,
        dropout: float = 0.1,
        vq_beta: float = 1.0,
        embodiment_config: dict[str, Any] | None = None,
        soft_assignment_temperature: float = 1.0,
        dead_code_threshold: int = 100,
        reset_noise_scale: float = 1e-3,
        decoder_type: str = "diffusion",
        diffusion_config: dict[str, Any] | None = None,
        use_vl_embedder: bool = False,
        quantizer_type: str = "vq",
        fsq_levels: tuple[int, ...] = (8, 5, 5, 5),
    ) -> None:
        """Build the encoder, quantizer, decoder and optional auxiliary heads.

        Args:
            quantizer_type: VQ (default) or semantic_fsq.
            fsq_levels: Fixed semantic FSQ scalar levels [8,5,5,5].
            action_dim: Action dimension of the default embodiment.
            window_size: Number of action steps in one window, i.e. the tokenized horizon.
            model_dim: Width shared by the encoder, the codebook and the decoder.
            num_tokens: Number of discrete tokens emitted per window.
            codebook_size: Token vocabulary size.
            num_codebooks: Number of residual quantization stages.
            num_heads: Number of attention heads in the encoder and decoder.
            encoder_layers: Depth of each latent self-attention stack in the encoder.
            decoder_layers: Depth of each latent self-attention stack in the decoder.
            encoder_cross_layers: Number of cross-attention rounds in the encoder.
            decoder_cross_layers: Number of cross-attention rounds in the decoder.
            use_encoder_latent_self_attn: Apply latent self-attention in the encoder.
            use_decoder_latent_self_attn: Apply latent self-attention in the decoder.
            share_encoder_latent_transformer: Reuse one latent transformer across encoder rounds.
            share_decoder_latent_transformer: Reuse one latent transformer across decoder rounds.
            share_encoder_cross_attn: Reuse one cross-attention block across encoder rounds.
            share_decoder_cross_attn: Reuse one cross-attention block across decoder rounds.
            dropout: Dropout probability used throughout.
            vq_beta: Commitment weight of the quantizer's VQ loss.
            embodiment_config: Mapping from embodiment name to ``action_dim``, ``freq`` and
                ``duration``, letting one tokenizer serve robots with different action layouts.
            soft_assignment_temperature: Temperature of the quantizer's soft assignment used for
                the codebook entropy loss.
            dead_code_threshold: Consecutive unused training steps after which a code is resampled.
            reset_noise_scale: Noise added to a resampled dead code.
            decoder_type: ``"diffusion"`` (default) for iterative reconstruction, or ``"perceiver"``
                for a single-pass decoder.
            diffusion_config: Overrides for the diffusion decoder, read for the keys
                ``num_train_steps``, ``num_sample_steps``, ``beta_schedule``, ``predict_target``,
                ``denoiser_layers`` and ``kernel_size``. Ignored unless ``decoder_type`` is
                ``"diffusion"``.
            use_vl_embedder: Build the visual-language embedder needed by the CLIP loss. Default
                ``False`` because CLIP is not part of the matched-h20 tokenizer recipe.

        Raises:
            ValueError: If ``decoder_type`` is not ``"perceiver"`` or ``"diffusion"``.
        """
        super().__init__()
        if decoder_type not in {"perceiver", "diffusion"}:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.decoder_type = str(decoder_type)
        self.action_dim = int(action_dim)
        self.window_size = int(window_size)
        self.num_tokens = int(num_tokens)
        self.encoder = ActionPerceiverEncoder(
            action_dim=action_dim,
            model_dim=model_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            latent_depth=encoder_layers,
            num_cross_layers=encoder_cross_layers,
            dropout=dropout,
            embodiment_config=embodiment_config,
            use_latent_self_attn=use_encoder_latent_self_attn,
            share_latent_transformer=share_encoder_latent_transformer,
            share_cross_attn=share_encoder_cross_attn,
        )
        self.quantizer_type = quantizer_type
        if quantizer_type == "semantic_fsq":
            if codebook_size != 1000 or num_codebooks != 1:
                raise ValueError("semantic_fsq requires a single vocabulary of 1000")
            self.quantizer = SemanticFSQQuantizer(model_dim, fsq_levels)
        elif quantizer_type == "vq":
            self.quantizer = ResidualVectorQuantizer(
                codebook_size,
                model_dim,
                num_codebooks,
                vq_beta,
                soft_assignment_temperature=soft_assignment_temperature,
                dead_code_threshold=dead_code_threshold,
                reset_noise_scale=reset_noise_scale,
            )
        else:
            raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")
        if self.decoder_type == "diffusion":
            diffusion_config = diffusion_config or {}
            self.decoder = ActionDiffusionDecoder(
                action_dim=action_dim,
                model_dim=model_dim,
                window_size=window_size,
                num_heads=num_heads,
                latent_depth=decoder_layers,
                num_cross_layers=decoder_cross_layers,
                dropout=dropout,
                embodiment_config=embodiment_config,
                use_latent_self_attn=use_decoder_latent_self_attn,
                share_latent_transformer=share_decoder_latent_transformer,
                share_cross_attn=share_decoder_cross_attn,
                num_train_steps=int(diffusion_config.get("num_train_steps", 1000)),
                num_sample_steps=int(diffusion_config.get("num_sample_steps", 27)),
                beta_schedule=str(diffusion_config.get("beta_schedule", "cosine")),
                predict_target=str(diffusion_config.get("predict_target", "x0")),
                denoiser_layers=int(diffusion_config.get("denoiser_layers", 6)),
                kernel_size=int(diffusion_config.get("kernel_size", 5)),
            )
        else:
            self.decoder = ActionPerceiverDecoder(
                action_dim=action_dim,
                model_dim=model_dim,
                window_size=window_size,
                num_heads=num_heads,
                latent_depth=decoder_layers,
                num_cross_layers=decoder_cross_layers,
                dropout=dropout,
                embodiment_config=embodiment_config,
                use_latent_self_attn=use_decoder_latent_self_attn,
                share_latent_transformer=share_decoder_latent_transformer,
                share_cross_attn=share_decoder_cross_attn,
            )
        self.use_vl_embedder = bool(use_vl_embedder)
        self.vl_embedder = VisualLanguageEmbedder(model_dim) if self.use_vl_embedder else None
        self.token_proj = nn.Linear(model_dim, model_dim)
        self.last_refreshed_codes = 0

    def encode(self, action: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Compress an action window into continuous latent tokens.

        Args:
            action: Action window of shape ``[B, window_size, A]``.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Latent tokens of shape ``[B, num_tokens, model_dim]``, before quantization.
        """
        return self.encoder(action, embodiment_ids)

    def decode(self, latents: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct an action window from latent tokens.

        Args:
            latents: Latent tokens of shape ``[B, num_tokens, model_dim]``, quantized or not.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Actions of shape ``[B, horizon, max_action_dim]``, padded past each embodiment's own
            action dimension. With the diffusion decoder this runs the full sampling loop.
        """
        return self.decoder(latents, embodiment_ids)

    def decode_train(self, latents: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Decode train-time continuous latents without cutting their upstream gradient."""
        if self.decoder_type == "diffusion":
            return self.decoder.sample_differentiable(latents, embodiment_ids)
        return self.decoder(latents, embodiment_ids)

    def freeze_encoder(self) -> None:
        """Stop training the encoder, e.g. when only the decoder is being fine-tuned."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def freeze_first_codebook(self) -> None:
        """Pin the coarsest codebook so previously emitted token ids keep their meaning."""
        self.quantizer.freeze_codebook(0)

    @torch.no_grad()
    def tokenize(self, action: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Turn an action window into the discrete token ids a policy predicts over.

        Only the first codebook is kept, matching the single-codebook ActionCodec contract.

        Args:
            action: Action window of shape ``[B, window_size, A]``, already normalized the same way
                as during tokenizer training.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Token ids of shape ``[B, num_tokens]`` in ``[0, codebook_size)``.
        """
        latents = self.encode(action.float(), embodiment_ids)
        return self.quantizer.encode_indices(latents)[..., 0].long()

    @torch.no_grad()
    def detokenize(self, tokens: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Turn predicted token ids back into an executable action window.

        Args:
            tokens: Token ids of shape ``[B, num_tokens]`` in ``[0, codebook_size)``.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Actions of shape ``[B, horizon, action_dim]``, trimmed to the configured action
            dimension.

        Raises:
            ValueError: If ``tokens`` is not 2D or holds an id outside the codebook range.
        """
        tokens = tokens.long()
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B, {self.num_tokens}], got {tuple(tokens.shape)}")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.quantizer.codebook_size):
            raise ValueError(f"tokens must be in [0, {self.quantizer.codebook_size - 1}]")
        latents = self.quantizer.indices_to_embedding(tokens.unsqueeze(-1))
        return self.decode(latents, embodiment_ids)[..., : self.action_dim]

    def forward(
        self,
        action: torch.Tensor,
        embodiment_ids: torch.Tensor | None = None,
        delta_state: torch.Tensor | None = None,
        loss_config: dict[str, Any] | None = None,
        image: torch.Tensor | None = None,
        prompts: list[str] | None = None,
        step: int = 0,
        stage: str = "pretrain",
        action_std: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one tokenizer training step and assemble the weighted training objective.

        Beyond reconstruction and the VQ terms, three optional signals shape the token space: a DTW
        alignment loss that pulls together windows with similar state trajectories, a CLIP loss that
        anchors tokens to the visual-language embedding, and codebook regularizers. Physical action
        losses and paired-window overlap consistency are inert unless their weights in
        ``loss_config`` are positive and ``action_std`` is supplied.

        Args:
            action: Action window of shape ``[B, window_size, action_dim]``.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.
            delta_state: Per-step state differences of shape ``[B, T, D]``, used to mine positive
                and negative window pairs for the alignment losses.
            loss_config: Loss weights and hyperparameters. Recognized keys are ``weight_recon``,
                ``weight_vq``, ``weight_l1``, ``weight_codebook_entropy``, ``weight_clip``,
                ``clip_temperature``, ``clip_start_step``, ``weight_align`` with its hard-DTW
                mining options (``positive_topk``, ``negative_topk``, ``negative_quantile``,
                ``temperature``, ``band_frac``, ``weight_negative``, ``negative_margin``), and
                ``weight_chunk_align`` with its soft-DTW options (``chunk_align_positive_topk``,
                ``chunk_align_gamma``, ``chunk_align_max_candidate_pairs``,
                ``chunk_align_pair_batch_size``, ``chunk_align_dtw_backend``).
            image: Channel-last frames of shape ``[B, H, W, 3]`` for the CLIP loss.
            prompts: ``B`` task strings paired with ``image``.
            step: Global training step, compared against ``clip_start_step`` so the CLIP loss can be
                delayed until reconstruction has stabilized.
            stage: Training stage; the CLIP loss is only applied during ``"pretrain"``.
            action_std: Dataset action standard deviation used to convert normalized errors to raw
                action units.
            timesteps: Optional diffusion timestep(s), shared across paired windows when a pair
                batch supplies one timestep per pair.
            noise: Optional union noise of shape ``[pair_batch, horizon + overlap_shift, action_dim]``
                for paired diffusion training, or per-window noise passed through unchanged.

        Returns:
            Mapping whose ``"loss"`` entry is the scalar objective to backpropagate. The remaining
            entries are diagnostics: the detached ``loss_*`` terms, the detached reconstruction
            ``"recon"`` of shape ``[B, window_size, action_dim]``, the integer code ids
                ``"indices"`` of shape ``[B, num_tokens, num_codebooks]``, and ``"refreshed_codes"``
                counting the dead codes queued for replacement this step.

        Raises:
            ValueError: If ``action`` does not have shape ``[B, window_size, action_dim]``.
        """
        if action.ndim != 3 or action.shape[1] != self.window_size or action.shape[-1] != self.action_dim:
            raise ValueError(f"action must have shape [B, {self.window_size}, {self.action_dim}]")
        loss_config = loss_config or {}
        z = self.encode(action.float(), embodiment_ids)
        zq, indices, vq_loss = self.quantizer(z)
        alignment_z = (
            self.quantizer.last_quantized_coordinates if self.quantizer_type == "semantic_fsq" else z
        )
        overlap_weight = float(loss_config.get("overlap_consistency_weight", 0.0))
        overlap_shift = int(loss_config.get("overlap_shift", 16))
        decoder_timesteps = timesteps
        decoder_noise = noise
        if self.decoder_type == "diffusion" and overlap_weight > 0:
            if action.shape[0] % 2:
                raise ValueError("Overlap consistency requires an even batch of [A_batch, B_batch]")
            pair_batch = action.shape[0] // 2
            if timesteps is None:
                pair_timesteps = torch.randint(
                    0, self.decoder.num_train_steps, (pair_batch,), device=action.device
                )
            else:
                pair_timesteps = timesteps
            if pair_timesteps.shape[0] == pair_batch:
                decoder_timesteps = torch.cat((pair_timesteps, pair_timesteps), dim=0)
            if noise is None:
                union_noise = torch.randn(
                    pair_batch,
                    self.window_size + overlap_shift,
                    action.shape[-1],
                    device=action.device,
                    dtype=action.dtype,
                )
            else:
                union_noise = noise
            if (
                union_noise.shape[0] == pair_batch
                and union_noise.shape[1] == self.window_size + overlap_shift
            ):
                noise_a = union_noise[:, : self.window_size]
                noise_b = union_noise[:, overlap_shift : overlap_shift + self.window_size]
                decoder_noise = torch.cat((noise_a, noise_b), dim=0)
        if self.decoder_type == "diffusion":
            if decoder_timesteps is None and decoder_noise is None:
                reconstruction, reconstruction_loss = self.decoder.forward_train(zq, action, embodiment_ids)
            else:
                reconstruction, reconstruction_loss = self.decoder.forward_train(
                    zq,
                    action,
                    embodiment_ids,
                    timesteps=decoder_timesteps,
                    noise=decoder_noise,
                )
            reconstruction = reconstruction[..., : self.action_dim]
        else:
            reconstruction = self.decode(zq, embodiment_ids)[..., : self.action_dim]
            reconstruction_loss = F.mse_loss(reconstruction, action.float())
        self.last_refreshed_codes = self.quantizer.last_refreshed_codes
        alignment_loss = action.new_zeros(())
        if delta_state is not None and float(loss_config.get("weight_align", 0.0)) > 0:
            pairs = chunk_hard_dtw_targets(
                delta_state.float(),
                positive_topk=int(loss_config.get("positive_topk", 4)),
                negative_topk=int(loss_config.get("negative_topk", 0)),
                negative_quantile=float(loss_config.get("negative_quantile", 0.2)),
                temperature=float(loss_config.get("temperature", 1.0)),
                band_frac=float(loss_config.get("band_frac", 0.2)),
            )
            embeddings = F.normalize(alignment_z.float().mean(1), dim=-1)
            alignment_loss = semantic_contrastive_loss(
                embeddings,
                positive_mask=pairs.positive_mask,
                negative_mask=pairs.negative_mask,
                positive_weight=float(loss_config["weight_align"]),
                negative_weight=float(loss_config.get("weight_negative", 0.0)),
                negative_margin=float(loss_config.get("negative_margin", 1.0)),
            ).total
        if delta_state is not None and float(loss_config.get("weight_chunk_align", 0.0)) > 0:
            soft_pairs = chunk_soft_dtw_targets(
                delta_state.float(),
                positive_topk=int(loss_config.get("chunk_align_positive_topk", 4)),
                gamma=float(loss_config.get("chunk_align_gamma", 0.1)),
                max_candidate_pairs=loss_config.get("chunk_align_max_candidate_pairs"),
                pair_batch_size=int(loss_config.get("chunk_align_pair_batch_size", 8192)),
                dtw_backend=str(loss_config.get("chunk_align_dtw_backend", "auto")),
            )
            chunk_embeddings = F.normalize(alignment_z.float().mean(1), dim=-1)
            alignment_loss = (
                alignment_loss
                + semantic_contrastive_loss(
                    chunk_embeddings,
                    positive_mask=soft_pairs.positive_mask,
                    negative_mask=torch.zeros_like(soft_pairs.positive_mask),
                    positive_weight=float(loss_config["weight_chunk_align"]),
                    negative_weight=0.0,
                    negative_margin=1.0,
                ).total
            )
        l1_loss = zq.abs().mean()
        entropy_loss = self.quantizer.last_soft_entropy_loss
        clip_loss = action.new_zeros(())
        if (
            self.vl_embedder is not None
            and image is not None
            and prompts is not None
            and float(loss_config.get("weight_clip", 0.0)) > 0
            and stage == "pretrain"
            and int(step) >= int(loss_config.get("clip_start_step", 0))
        ):
            token_global = self.token_proj(z.mean(1))
            vl_global = self.vl_embedder(image, prompts)
            clip_loss = float(loss_config["weight_clip"]) * symmetric_info_nce(
                token_global,
                vl_global,
                temperature=float(loss_config.get("clip_temperature", 0.07)),
            )
        physical_recon_weight = float(loss_config.get("physical_recon_weight", 0.0))
        physical_velocity_weight = float(loss_config.get("physical_velocity_weight", 0.0))
        if action_std is None and (
            physical_recon_weight > 0 or physical_velocity_weight > 0 or overlap_weight > 0
        ):
            raise ValueError("Physical ActionCodec losses require action_std")
        physical_enabled = action_std is not None
        zero = action.new_zeros(())
        loss_phys_recon = zero
        loss_phys_velocity = zero
        loss_overlap = zero
        physical_recon_mae = zero
        physical_velocity_mae = zero
        overlap_mae = zero
        if physical_enabled:
            action_std = action_std.to(device=action.device, dtype=torch.float32)
            if tuple(action_std.shape) != (self.action_dim,):
                raise ValueError(f"action_std must have shape ({self.action_dim},)")
            action_indices = _continuous_indices(
                self.action_dim, loss_config.get("continuous_action_indices"), action.device
            )
            unit_scale = float(loss_config.get("physical_unit_scale", 1.0))
            physical_error = (reconstruction.float() - action.float()) * action_std
            loss_phys_recon, physical_recon_mae = _physical_loss_and_mae(
                physical_error, action_indices, unit_scale
            )
            pred_velocity = (reconstruction[:, 1:] - reconstruction[:, :-1]).float() * action_std
            gt_velocity = (action[:, 1:] - action[:, :-1]).float() * action_std
            loss_phys_velocity, physical_velocity_mae = _physical_loss_and_mae(
                pred_velocity - gt_velocity, action_indices, unit_scale
            )
            if overlap_weight > 0:
                loss_overlap, overlap_mae = _overlap_loss_and_mae(
                    reconstruction, action_std, overlap_shift, action_indices, unit_scale
                )
        weighted_phys_recon = physical_recon_weight * loss_phys_recon if physical_recon_weight > 0 else zero
        weighted_phys_velocity = (
            physical_velocity_weight * loss_phys_velocity if physical_velocity_weight > 0 else zero
        )
        weighted_overlap = overlap_weight * loss_overlap if overlap_weight > 0 else zero
        total = (
            float(loss_config.get("weight_recon", 1.0)) * reconstruction_loss
            + float(loss_config.get("weight_vq", 1.0)) * vq_loss
            + float(loss_config.get("weight_l1", 0.0)) * l1_loss
            + float(loss_config.get("weight_codebook_entropy", 0.0)) * entropy_loss
            + clip_loss
            + alignment_loss
            + weighted_phys_recon
            + weighted_phys_velocity
            + weighted_overlap
        )
        return {
            "loss": total,
            "loss_recon": reconstruction_loss.detach(),
            "loss_vq": vq_loss.detach(),
            "loss_align": alignment_loss.detach(),
            "loss_l1": l1_loss.detach(),
            "loss_codebook_entropy": entropy_loss.detach(),
            "loss_clip": clip_loss.detach(),
            "loss_phys_recon": loss_phys_recon.detach(),
            "loss_phys_velocity": loss_phys_velocity.detach(),
            "loss_overlap": loss_overlap.detach(),
            "weighted_phys_recon": weighted_phys_recon.detach(),
            "weighted_phys_velocity": weighted_phys_velocity.detach(),
            "weighted_overlap": weighted_overlap.detach(),
            "physical_recon_mae": physical_recon_mae.detach(),
            "physical_velocity_mae": physical_velocity_mae.detach(),
            "overlap_mae": overlap_mae.detach(),
            "recon": reconstruction.detach(),
            "indices": indices,
            "refreshed_codes": action.new_tensor(float(self.last_refreshed_codes)),
        }
