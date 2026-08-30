"""Dependency-free visual-language embedding used by the optional tokenizer auxiliary loss."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
import torch.nn as nn


def _stable_hash_token(token: str, vocab_size: int) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % vocab_size


class PromptHashEncoder(nn.Module):
    """Bag-of-words text encoder that needs no tokenizer or pretrained weights.

    Each whitespace-separated word is folded into a fixed number of buckets by a stable MD5 hash, so
    the same instruction always maps to the same embedding across runs and processes.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """Build the hashed word embedding table.

        Args:
            vocab_size: Number of hash buckets; every word is mapped into this range.
            embed_dim: Feature dimension of each word embedding.
        """
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        """Embed each prompt as the mean of its hashed word embeddings.

        Args:
            prompts: ``B`` instruction strings. A prompt with no words falls back to ``"<empty>"``.
            device: Device the hashed word indices are allocated on.

        Returns:
            Layer-normalized prompt embeddings of shape ``[B, embed_dim]``.
        """
        outputs = []
        for prompt in prompts:
            words = [word for word in prompt.lower().strip().split() if word] or ["<empty>"]
            ids = torch.tensor(
                [_stable_hash_token(word, self.vocab_size) for word in words],
                device=device,
                dtype=torch.long,
            )
            outputs.append(self.embed(ids).mean(0))
        return self.ln(torch.stack(outputs))


class TinyVisionEncoder(nn.Module):
    """Small strided CNN that reduces one raw frame to a single visual embedding.

    Kept deliberately tiny because it only has to provide a training signal for the auxiliary
    contrastive loss, not a general-purpose visual backbone.
    """

    def __init__(self, output_dim: int) -> None:
        """Build the convolutional trunk and its projection head.

        Args:
            output_dim: Feature dimension of the projected image embedding.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Embed a batch of frames.

        Args:
            image: Channel-last frames of shape ``[B, H, W, 3]`` with values in ``[0, 255]``; they
                are rescaled to ``[0, 1]`` internally.

        Returns:
            Image embeddings of shape ``[B, output_dim]``.

        Raises:
            ValueError: If ``image`` is not a 4D tensor.
        """
        if image.ndim != 4:
            raise ValueError("VL image must have shape [B, H, W, 3]")
        x = self.conv(image.permute(0, 3, 1, 2).float() / 255.0).flatten(1)
        return self.proj(x)


class VisualLanguageEmbedder(nn.Module):
    """Fuse a frame and its instruction into the target embedding of the tokenizer's CLIP loss.

    The tokenizer contrasts its pooled action tokens against this embedding, which pushes windows
    performing the same instruction in the same visual context towards the same discrete codes. The
    256-dimensional vision and text branches are concatenated before fusion.
    """

    def __init__(self, model_dim: int, text_vocab_size: int = 8192) -> None:
        """Build the vision, text and fusion branches.

        Args:
            model_dim: Output width; must match the tokenizer's model dimension so that the two
                embeddings can be compared directly.
            text_vocab_size: Number of hash buckets used by the prompt encoder.
        """
        super().__init__()
        self.text = PromptHashEncoder(text_vocab_size, 256)
        self.vision = TinyVisionEncoder(256)
        self.fuse = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, image: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Fuse the visual and language branches into one embedding per sample.

        Args:
            image: Channel-last frames of shape ``[B, H, W, 3]`` with values in ``[0, 255]``.
            prompts: ``B`` instruction strings, aligned with the frames in ``image``.

        Returns:
            Fused visual-language embeddings of shape ``[B, model_dim]``.
        """
        return self.fuse(torch.cat((self.vision(image), self.text(prompts, image.device)), dim=-1))
