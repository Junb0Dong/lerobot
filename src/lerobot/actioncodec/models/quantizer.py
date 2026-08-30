"""Residual vector quantization with straight-through gradients."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ matching the ActionCodec assignment and loss semantics."""

    def __init__(
        self,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        num_codebooks: int = 1,
        beta: float = 0.25,
        soft_assignment_temperature: float = 1.0,
        dead_code_threshold: int = 100,
        reset_noise_scale: float = 1e-3,
    ) -> None:
        """Build the residual codebook stack.

        Args:
            codebook_size: Number of code vectors per codebook, i.e. the token vocabulary size.
            embed_dim: Dimension of each code vector; must match the latent token width.
            num_codebooks: Number of residual stages. Stage ``i + 1`` quantizes what stage ``i``
                left over, so more stages buy precision at the cost of more tokens per latent.
            beta: Commitment weight on the encoder-side term of the VQ loss. The codebook term is
                always weighted 1.
            soft_assignment_temperature: Temperature of the softmax over negative distances used to
                build the batch-averaged assignment distribution behind
                ``last_soft_entropy_loss``. Lower values make that distribution peakier.
            dead_code_threshold: Number of consecutive training forward passes a code may go unused
                before it is queued for resampling. Values ``<= 0`` disable dead-code refresh.
            reset_noise_scale: Standard deviation of the Gaussian noise added to the encoder vector
                that a dead code is resampled from, so two dead codes never collapse onto one point.

        Raises:
            ValueError: If ``codebook_size``, ``embed_dim`` or ``num_codebooks`` is not positive.
        """
        super().__init__()
        if codebook_size <= 0 or embed_dim <= 0 or num_codebooks <= 0:
            raise ValueError("codebook_size, embed_dim and num_codebooks must be positive")
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.num_codebooks = int(num_codebooks)
        self.beta = float(beta)
        self.soft_assignment_temperature = float(soft_assignment_temperature)
        self.dead_code_threshold = int(dead_code_threshold)
        self.reset_noise_scale = float(reset_noise_scale)
        self._entropy_norm = max(math.log(float(max(self.codebook_size, 2))), 1e-6)
        self.codebooks = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.embed_dim) for _ in range(num_codebooks)]
        )
        bound = self.embed_dim**-0.5
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -bound, bound)
        self.register_buffer(
            "inactive_steps",
            torch.zeros(self.num_codebooks, self.codebook_size, dtype=torch.long),
            persistent=False,
        )
        self.last_soft_entropy_loss = torch.tensor(0.0)
        self.last_refreshed_codes = 0
        self._pending_refreshes: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self._inactive_steps_snapshots: dict[int, torch.Tensor] = {}

    def freeze_codebook(self, idx: int) -> None:
        """Stop training one codebook, e.g. to keep an already-published token vocabulary stable.

        A frozen codebook is also skipped by dead-code refresh, so its codes stay byte-identical.

        Args:
            idx: Index of the codebook to freeze.
        """
        for parameter in self.codebooks[idx].parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def _queue_inactive_code_refresh(
        self, codebook_idx: int, assignments: torch.Tensor, flat_inputs: torch.Tensor
    ) -> int:
        if (
            self.dead_code_threshold <= 0
            or flat_inputs.numel() == 0
            or not self.codebooks[codebook_idx].weight.requires_grad
        ):
            return 0
        if codebook_idx not in self._inactive_steps_snapshots:
            self._inactive_steps_snapshots[codebook_idx] = self.inactive_steps[codebook_idx].clone()
        counts = torch.bincount(assignments, minlength=self.codebook_size)
        inactive = counts == 0
        self.inactive_steps[codebook_idx][inactive] += 1
        self.inactive_steps[codebook_idx][~inactive] = 0
        dead = self.inactive_steps[codebook_idx] >= self.dead_code_threshold
        num_dead = int(dead.sum())
        if num_dead == 0:
            return 0
        sample_idx = torch.randint(0, flat_inputs.shape[0], (num_dead,), device=flat_inputs.device)
        new_codes = flat_inputs[sample_idx]
        if self.reset_noise_scale > 0:
            new_codes = new_codes + self.reset_noise_scale * torch.randn_like(new_codes)
        self._pending_refreshes.append((codebook_idx, dead.clone(), new_codes.detach()))
        self.inactive_steps[codebook_idx][dead] = 0
        return num_dead

    @torch.no_grad()
    def apply_pending_codebook_updates(self) -> int:
        """Commit the dead-code replacements that ``forward`` queued during training.

        Refreshes are queued rather than applied in place so that the caller can run them after
        ``optimizer.step()``, where the optimizer can no longer overwrite the resampled codes with
        an update computed from the stale weights.

        Returns:
            Number of code vectors that were overwritten.
        """
        applied = 0
        for codebook_idx, dead_mask, new_codes in self._pending_refreshes:
            weight = self.codebooks[codebook_idx].weight
            weight[dead_mask] = new_codes.to(weight)
            applied += int(dead_mask.sum())
        self._pending_refreshes.clear()
        self._inactive_steps_snapshots.clear()
        self.last_refreshed_codes = applied
        return applied

    @torch.no_grad()
    def discard_pending_codebook_updates(self) -> int:
        """Drop the queued dead-code replacements and roll the inactivity counters back.

        Used when a training step is thrown away, for example on a non-finite loss, so that the
        codebook bookkeeping stays consistent with the parameters that were actually kept.

        Returns:
            Number of code vectors whose queued refresh was discarded.
        """
        discarded = sum(int(mask.sum()) for _, mask, _ in self._pending_refreshes)
        for codebook_idx, snapshot in self._inactive_steps_snapshots.items():
            self.inactive_steps[codebook_idx].copy_(snapshot)
        self._pending_refreshes.clear()
        self._inactive_steps_snapshots.clear()
        self.last_refreshed_codes = 0
        return discarded

    @torch.no_grad()
    def copy_codebook_from(self, src: ResidualVectorQuantizer, src_idx: int, dst_idx: int) -> None:
        """Import one codebook from another quantizer, e.g. to warm-start from a trained tokenizer.

        Args:
            src: Quantizer to read the code vectors from.
            src_idx: Codebook index within ``src``.
            dst_idx: Codebook index in this quantizer to overwrite.
        """
        self.codebooks[dst_idx].weight.copy_(src.codebooks[src_idx].weight)

    def _nearest(self, residual: torch.Tensor, codebook: nn.Embedding) -> tuple[torch.Tensor, torch.Tensor]:
        flat = residual.reshape(-1, self.embed_dim)
        weight = codebook.weight
        distances = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ weight.t() + weight.pow(2).sum(1)
        indices = distances.argmin(dim=1).view(residual.shape[0], residual.shape[1])
        return indices, codebook(indices).view_as(residual)

    @torch.no_grad()
    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Assign latent tokens to code indices without building an autograd graph.

        Args:
            z: Latent tokens of shape ``[B, N, embed_dim]``.

        Returns:
            Code indices of shape ``[B, N, num_codebooks]``, ordered from the coarsest residual
            stage to the finest.
        """
        residual = z
        indices = []
        for codebook in self.codebooks:
            ids, quantized = self._nearest(residual, codebook)
            indices.append(ids)
            residual = residual - quantized
        return torch.stack(indices, dim=-1)

    def indices_to_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Rebuild latent tokens by summing the code vector of every residual stage.

        This is the inverse of ``encode_indices`` and the entry point used when a downstream policy
        has predicted discrete tokens that must be turned back into continuous latents.

        Args:
            indices: Code indices of shape ``[B, N, num_codebooks]``.

        Returns:
            Reconstructed latents of shape ``[B, N, embed_dim]``.

        Raises:
            ValueError: If ``indices`` is not 3D or its last dimension is not ``num_codebooks``.
        """
        if indices.ndim != 3 or indices.shape[-1] != self.num_codebooks:
            raise ValueError(f"indices must have shape [B, N, {self.num_codebooks}]")
        result = torch.zeros(
            indices.shape[:2] + (self.embed_dim,), device=indices.device, dtype=self.codebooks[0].weight.dtype
        )
        for idx, codebook in enumerate(self.codebooks):
            result = result + codebook(indices[..., idx].long())
        return result

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latent tokens stage by stage with straight-through gradients.

        Side effects: ``last_soft_entropy_loss`` is set to the negated, log-vocabulary-normalized
        entropy of the batch-averaged soft assignment (so minimizing it spreads usage over the
        codebook), and in training mode dead codes are queued for refresh and counted in
        ``last_refreshed_codes``. The queued refreshes only take effect once
        ``apply_pending_codebook_updates`` is called.

        Args:
            z: Latent tokens of shape ``[B, N, embed_dim]``.

        Returns:
            Tuple of the quantized latents of shape ``[B, N, embed_dim]``, the code indices of
            shape ``[B, N, num_codebooks]``, and the scalar VQ loss summed over all stages.

        Raises:
            ValueError: If ``z`` is not 3D or its last dimension is not ``embed_dim``.
        """
        if z.ndim != 3 or z.shape[-1] != self.embed_dim:
            raise ValueError(f"z must have shape [B, N, {self.embed_dim}], got {tuple(z.shape)}")
        residual = z
        quantized = torch.zeros_like(z)
        losses = []
        entropies = []
        ids = []
        refreshed_codes = 0
        temperature = max(self.soft_assignment_temperature, 1e-6)
        for codebook_idx, codebook in enumerate(self.codebooks):
            flat = residual.reshape(-1, self.embed_dim)
            weight = codebook.weight
            distances = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ weight.t() + weight.pow(2).sum(1)
            probabilities = torch.softmax((-distances.float()) / temperature, dim=-1).to(z.dtype)
            mean_probabilities = probabilities.mean(0)
            entropies.append(-(mean_probabilities * mean_probabilities.clamp_min(1e-12).log()).sum())
            indices = distances.argmin(dim=1).view(z.shape[0], z.shape[1])
            q = codebook(indices).view_as(residual)
            losses.append(
                (q - residual.detach()).pow(2).mean() + self.beta * (q.detach() - residual).pow(2).mean()
            )
            quantized = quantized + residual + (q - residual).detach()
            residual = residual - q
            ids.append(indices)
            if self.training:
                refreshed_codes += self._queue_inactive_code_refresh(
                    codebook_idx, indices.reshape(-1), flat.detach()
                )
        self.last_soft_entropy_loss = -torch.stack(entropies).mean() / self._entropy_norm
        self.last_refreshed_codes = refreshed_codes
        return quantized, torch.stack(ids, dim=-1), torch.stack(losses).sum()


def codebook_stats(indices: torch.Tensor, codebook_size: int) -> dict[str, torch.Tensor]:
    """Summarize how evenly a batch of assignments spreads over the codebook.

    These are the diagnostics used to detect codebook collapse during tokenizer training.

    Args:
        indices: Code indices of any shape; they are flattened before being counted.
        codebook_size: Total number of codes, used as the denominator of the usage ratio.

    Returns:
        Mapping with three scalar tensors: ``"usage"`` (fraction of codes hit at least once),
        ``"dead_ratio"`` (``1 - usage``) and ``"perplexity"`` (exponentiated entropy of the
        assignment histogram, so ``codebook_size`` for a perfectly uniform batch). Empty
        ``indices`` yield zero usage, a dead ratio of one and zero perplexity.
    """
    flat = indices.reshape(-1)
    if flat.numel() == 0:
        return {
            "usage": indices.new_zeros((), dtype=torch.float32),
            "dead_ratio": indices.new_ones((), dtype=torch.float32),
            "perplexity": indices.new_zeros((), dtype=torch.float32),
        }
    counts = torch.bincount(flat, minlength=codebook_size).float()
    probabilities = counts / counts.sum().clamp_min(1.0)
    usage = (counts > 0).float().sum() / float(codebook_size)
    perplexity = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {"usage": usage, "dead_ratio": 1.0 - usage, "perplexity": perplexity}
