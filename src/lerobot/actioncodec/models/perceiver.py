"""Perceiver encoder/decoder used by the semantic action tokenizer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class MLP(nn.Module):
    """Two-layer feed-forward block with GELU activation and dropout."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """Build the feed-forward block.

        Args:
            dim: Input and output feature dimension.
            mlp_ratio: Expansion factor for the hidden dimension.
            dropout: Dropout probability applied after each linear layer.
        """
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block.

        Args:
            x: Input tensor with a trailing feature dimension of size ``dim``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with a residual feed-forward tail."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        """Build the cross-attention block.

        Args:
            dim: Feature dimension shared by the query and key/value streams.
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention and the feed-forward block.
        """
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Attend from the query stream to the key/value stream.

        Args:
            q: Query tensor of shape ``[B, Lq, dim]``.
            kv: Key/value tensor of shape ``[B, Lkv, dim]``.

        Returns:
            Updated query tensor of shape ``[B, Lq, dim]``.
        """
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(kv)
        output = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)[0]
        output = q + output
        return output + self.mlp(self.ln_mlp(output))


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with a residual feed-forward tail."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        """Build the self-attention block.

        Args:
            dim: Feature dimension of the token stream.
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention and the feed-forward block.
        """
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention followed by the feed-forward block.

        Args:
            x: Token tensor of shape ``[B, L, dim]``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        normalized = self.ln_attn(x)
        x = x + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.mlp(self.ln_mlp(x))


class LatentTransformer(nn.Module):
    """Stack of self-attention blocks applied to the latent token stream."""

    def __init__(self, dim: int, num_heads: int, depth: int, dropout: float = 0.0) -> None:
        """Build the latent transformer.

        Args:
            dim: Feature dimension of the latent tokens.
            num_heads: Number of attention heads per block.
            depth: Number of stacked self-attention blocks.
            dropout: Dropout probability inside each block.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(dim, num_heads, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the latent tokens through every self-attention block.

        Args:
            x: Latent tensor of shape ``[B, L, dim]``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        for block in self.blocks:
            x = block(x)
        return x


class EmbodimentEmbedding(nn.Module):
    """Per-embodiment learned token bank used as soft prompts."""

    def __init__(self, num_embeddings: int, out_len: int, out_dim: int) -> None:
        """Build the embedding table.

        Args:
            num_embeddings: Number of embodiments to allocate prompts for.
            out_len: Number of tokens produced per embodiment.
            out_dim: Feature dimension of each token.
        """
        super().__init__()
        self.out_len = int(out_len)
        self.out_dim = int(out_dim)
        self.embedding = nn.Embedding(num_embeddings, self.out_len * self.out_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, embodiment_ids: torch.Tensor) -> torch.Tensor:
        """Look up the prompt tokens for a batch of embodiments.

        Args:
            embodiment_ids: Integer tensor of shape ``[B]``.

        Returns:
            Prompt tokens of shape ``[B, out_len, out_dim]``.
        """
        return self.embedding(embodiment_ids.long()).view(embodiment_ids.shape[0], self.out_len, self.out_dim)


class PositionalEmbedding(nn.Module):
    """Sinusoidal or Fourier positional embeddings.

    The ``sincos`` variant indexes positions by integer step, while the ``fourier`` variant
    divides the step index by the embodiment control frequency so that positions carry
    wall-clock meaning across embodiments with different control rates.
    """

    def __init__(self, dim: int, encoding_type: str) -> None:
        """Build the positional embedding.

        Args:
            dim: Feature dimension of the tensor the embedding is added to.
            encoding_type: Either ``"sincos"`` or ``"fourier"``.

        Raises:
            ValueError: If ``encoding_type`` is not a supported variant.
        """
        super().__init__()
        if encoding_type not in {"sincos", "fourier"}:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")
        self.dim = int(dim)
        self.encoding_type = encoding_type

    def forward(
        self,
        x: torch.Tensor,
        freq: torch.Tensor | None = None,
        duration: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute positional embeddings matching the layout of ``x``.

        Args:
            x: Reference tensor of shape ``[B, L, dim]``; only its shape, device and dtype are used.
            freq: Per-sample control frequency of shape ``[B]``, required for ``"fourier"``.
            duration: Optional per-sample time cap of shape ``[B]`` clamping the position axis.

        Returns:
            Positional embedding tensor of shape ``[B, L, dim]``.

        Raises:
            ValueError: If ``x`` has an unexpected feature dimension, or if ``freq`` is missing
                for the ``"fourier"`` variant.
        """
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got dim={dim}")
        if self.encoding_type == "sincos":
            base = (
                torch.arange(seq_len, device=x.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            if freq is None:
                raise ValueError("fourier positional embeddings require freq")
            freq = freq.to(device=x.device, dtype=torch.float32).reshape(batch_size, 1).clamp_min(1e-6)
            base = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(0) / freq
            if duration is not None:
                base = torch.minimum(base, duration.to(x).reshape(batch_size, 1))
        half_dim = max(self.dim // 2, 1)
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        inverse_freq = torch.exp(-scale * torch.arange(half_dim, device=x.device, dtype=torch.float32))
        angles = base.unsqueeze(-1) * inverse_freq.view(1, 1, -1)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding[:, :, : self.dim].to(dtype=x.dtype)


def _prepare_embodiment_metadata(
    embodiment_config: dict[str, Any], expected_action_dim: int
) -> tuple[list[str], int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not embodiment_config:
        raise ValueError("embodiment_config must be non-empty")
    names = list(embodiment_config)
    action_dims = torch.tensor([int(embodiment_config[name]["action_dim"]) for name in names])
    freqs = torch.tensor([float(embodiment_config[name]["freq"]) for name in names])
    durations = torch.tensor([float(embodiment_config[name]["duration"]) for name in names])
    return names, max(int(action_dims.max()), int(expected_action_dim)), action_dims, freqs, durations


def _validate_embodiment_ids(embodiment_ids: torch.Tensor, num_embeddings: int, name: str) -> torch.Tensor:
    ids = embodiment_ids.long()
    if ids.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if int(ids.min()) < 0 or int(ids.max()) >= num_embeddings:
        raise ValueError(f"{name} out of range [0, {num_embeddings - 1}]")
    return ids


class ActionPerceiverEncoder(nn.Module):
    """Compress a variable-length action window into a fixed set of latent tokens.

    Actions are projected per embodiment so that arms with different action dimensions share one
    encoder, then cross-attended into a fixed-size latent query bank.
    """

    def __init__(
        self,
        action_dim: int,
        model_dim: int,
        num_tokens: int,
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
        """Build the encoder.

        Args:
            action_dim: Action dimension of the default embodiment.
            model_dim: Width of the transformer.
            num_tokens: Number of latent tokens produced per window.
            num_heads: Number of attention heads.
            latent_depth: Depth of each latent self-attention stack.
            num_cross_layers: Number of cross-attention rounds over the action stream.
            dropout: Dropout probability used throughout.
            embodiment_config: Mapping from embodiment name to ``action_dim``, ``freq`` and
                ``duration``. Defaults to a single ``"default"`` embodiment at 20 Hz for 1 s.
            use_soft_prompt: Use per-embodiment learned prompts instead of a shared latent bank.
            use_latent_self_attn: Apply latent self-attention after each cross-attention round.
            share_latent_transformer: Reuse one latent transformer across all rounds.
            share_cross_attn: Reuse one cross-attention block across all rounds.
        """
        super().__init__()
        embodiment_config = embodiment_config or {
            "default": {"action_dim": action_dim, "freq": 20.0, "duration": 1.0}
        }
        (
            self.embodiment_names,
            self.max_action_dim,
            action_dims,
            freqs,
            durations,
        ) = _prepare_embodiment_metadata(embodiment_config, action_dim)
        self.register_buffer("embodiment_action_dims", action_dims.long(), persistent=False)
        self.register_buffer("embodiment_freqs", freqs.float(), persistent=False)
        self.register_buffer("embodiment_durations", durations.float(), persistent=False)
        self.action_proj = nn.ModuleList([nn.Linear(int(dim), model_dim) for dim in action_dims])
        self.use_soft_prompt = bool(use_soft_prompt)
        if self.use_soft_prompt:
            self.cls_tokens = EmbodimentEmbedding(len(self.embodiment_names), num_tokens, model_dim)
        else:
            self.latent_tokens = nn.Parameter(torch.randn(num_tokens, model_dim) * 0.02)
        self.pos_emb_q = PositionalEmbedding(model_dim, "sincos")
        self.pos_emb_kv = PositionalEmbedding(model_dim, "fourier")
        self.num_cross_layers = int(num_cross_layers)
        self.use_latent_self_attn = bool(use_latent_self_attn)
        self.share_latent_transformer = bool(share_latent_transformer)
        self.share_cross_attn = bool(share_cross_attn)
        if self.share_cross_attn:
            self.cross_shared = CrossAttentionBlock(model_dim, num_heads, dropout)
            self.cross_blocks = None
        else:
            self.cross_blocks = nn.ModuleList(
                [CrossAttentionBlock(model_dim, num_heads, dropout) for _ in range(self.num_cross_layers)]
            )
            self.cross_shared = None
        if self.use_latent_self_attn and self.share_latent_transformer:
            self.latent_shared = LatentTransformer(model_dim, num_heads, latent_depth, dropout)
            self.latent_blocks = None
        elif self.use_latent_self_attn:
            self.latent_blocks = nn.ModuleList(
                [
                    LatentTransformer(model_dim, num_heads, latent_depth, dropout)
                    for _ in range(self.num_cross_layers)
                ]
            )
            self.latent_shared = None
        else:
            self.latent_shared = None
            self.latent_blocks = None
        self.ln_out = nn.LayerNorm(model_dim)

    def forward(self, actions: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Encode an action window into latent tokens.

        Args:
            actions: Action window of shape ``[B, T, A]``, zero-padded to the widest embodiment.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Latent tokens of shape ``[B, num_tokens, model_dim]``.

        Raises:
            ValueError: If no embodiment matched any sample in the batch.
        """
        if embodiment_ids is None:
            embodiment_ids = actions.new_zeros((actions.shape[0],), dtype=torch.long)
        ids = _validate_embodiment_ids(embodiment_ids, len(self.embodiment_names), "embodiment_ids")
        x = None
        for index, projection in enumerate(self.action_proj):
            mask = ids == index
            if mask.any():
                projected = projection(actions[mask, :, : int(self.embodiment_action_dims[index])])
                if x is None:
                    x = projected.new_zeros((actions.shape[0], actions.shape[1], projected.shape[-1]))
                x[mask] = projected
        if x is None:
            raise ValueError("No valid embodiment projections were produced")
        freq = self.embodiment_freqs.index_select(0, ids)
        duration = self.embodiment_durations.index_select(0, ids)
        x = x + self.pos_emb_kv(x, freq=freq, duration=duration)
        query = (
            self.cls_tokens(ids)
            if self.use_soft_prompt
            else self.latent_tokens.unsqueeze(0).expand_as(x[:, : self.latent_tokens.shape[0]])
        )
        query = query + self.pos_emb_q(query)
        for index in range(self.num_cross_layers):
            cross = self.cross_shared if self.share_cross_attn else self.cross_blocks[index]
            query = cross(query, x)
            if self.use_latent_self_attn:
                latent = self.latent_shared if self.share_latent_transformer else self.latent_blocks[index]
                query = latent(query)
        return self.ln_out(query)


class ActionPerceiverDecoder(ActionPerceiverEncoder):
    """Reconstruct an action window from latent tokens.

    Shares the encoder's block layout but swaps the roles of the two positional embeddings and
    replaces the input projections with per-embodiment output heads.
    """

    def __init__(self, action_dim: int, model_dim: int, window_size: int, **kwargs: Any) -> None:
        """Build the decoder.

        Args:
            action_dim: Action dimension of the default embodiment.
            model_dim: Width of the transformer.
            window_size: Maximum number of action steps the decoder can emit.
            **kwargs: Remaining encoder arguments, forwarded verbatim.
        """
        super().__init__(action_dim, model_dim, window_size, **kwargs)
        del self.action_proj
        self.window_size = int(window_size)
        self.pos_emb_q = PositionalEmbedding(model_dim, "fourier")
        self.pos_emb_kv = PositionalEmbedding(model_dim, "sincos")
        self.head = nn.ModuleList([nn.Linear(model_dim, int(dim)) for dim in self.embodiment_action_dims])

    def forward(self, latents: torch.Tensor, embodiment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent tokens back into an action window.

        Args:
            latents: Latent tokens of shape ``[B, num_tokens, model_dim]``.
            embodiment_ids: Integer tensor of shape ``[B]``. Defaults to embodiment 0.

        Returns:
            Actions of shape ``[B, horizon, max_action_dim]``, zero-padded past each embodiment's
            own action dimension.

        Raises:
            ValueError: If the batch mixes embodiments with different horizons, or if the
                requested horizon exceeds ``window_size``.
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
            raise ValueError(f"Requested horizon {length} exceeds decoder window_size {self.window_size}")
        x = latents + self.pos_emb_kv(latents)
        query = (
            self.cls_tokens(ids)[:, :length]
            if self.use_soft_prompt
            else self.latent_tokens[:length].unsqueeze(0).expand(latents.shape[0], -1, -1)
        )
        query = query + self.pos_emb_q(query, freq=freq, duration=duration)
        for index in range(self.num_cross_layers):
            cross = self.cross_shared if self.share_cross_attn else self.cross_blocks[index]
            query = cross(query, x)
            if self.use_latent_self_attn:
                latent = self.latent_shared if self.share_latent_transformer else self.latent_blocks[index]
                query = latent(query)
        output = None
        for index, head in enumerate(self.head):
            mask = ids == index
            if not mask.any():
                continue
            decoded = head(query[mask])
            if output is None:
                output = decoded.new_zeros((query.shape[0], length, self.max_action_dim))
            output[mask, :, : decoded.shape[-1]] = decoded
        if output is None:
            raise ValueError("No valid embodiment decoder outputs were produced for the batch")
        return output
