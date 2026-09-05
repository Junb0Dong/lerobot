"""Semantic FSQ matching actioncodec's tanh bounds and mixed-radix ordering."""

import torch
from torch import nn


class FSQGrid(nn.Module):
    """Fixed four-dimensional finite scalar lattice."""

    def __init__(self, levels=(8, 5, 5, 5)):
        """Build the fixed grid and, for the quantizer, learned projections."""
        super().__init__()
        if tuple(levels) != (8, 5, 5, 5):
            raise ValueError("semantic_fsq requires levels [8,5,5,5]")
        self.register_buffer("levels", torch.tensor(levels), persistent=False)
        self.register_buffer("basis", torch.tensor([1, 8, 40, 200]), persistent=False)

    def indices_to_scalar_classes(self, indices):
        """Expand IDs [... ] into four mixed-radix scalar classes."""
        return (indices.long().unsqueeze(-1) // self.basis) % self.levels

    def scalar_classes_to_indices(self, classes):
        """Combine four scalar classes into IDs [...]."""
        return (classes.long() * self.basis).sum(-1)

    def scalar_classes_to_coordinates(self, classes):
        """Map scalar classes to the normalized FSQ lattice."""
        half = self.levels // 2
        return (classes.float() - half) / half

    def coordinates_to_scalar_classes(self, coordinates):
        """Round normalized coordinates to scalar classes."""
        half = self.levels // 2
        return (coordinates.float() * half + half).round().long()

    def indices_to_coordinates(self, indices):
        """Expand token IDs into four normalized coordinates."""
        return self.scalar_classes_to_coordinates(self.indices_to_scalar_classes(indices))

    def bound(self, z):
        """Apply reference tanh bounds in FP32, including the even-level offset."""
        z = z.float()
        half_l = (self.levels.float() - 1) * 1.001 / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        return (z + (offset / half_l).atanh()).tanh() * half_l - offset

    def forward(self, z):
        """Quantize with straight-through gradients and return coordinates or decoder latents."""
        bounded = self.bound(z)
        rounded = bounded + (bounded.round() - bounded).detach()
        coordinates = rounded / (self.levels // 2)
        indices = self.scalar_classes_to_indices(self.coordinates_to_scalar_classes(coordinates))
        return coordinates, indices


class SemanticFSQQuantizer(FSQGrid):
    """Project encoder latents through the FSQ lattice into decoder latents."""

    def __init__(self, embed_dim, levels=(8, 5, 5, 5)):
        """Build the fixed grid and, for the quantizer, learned projections."""
        super().__init__(levels)
        self.input_projection = nn.Linear(embed_dim, 4)
        self.output_projection = nn.Linear(4, embed_dim)
        self.codebook_size = 1000
        self.num_codebooks = 1
        self.last_refreshed_codes = 0
        self.last_soft_entropy_loss = torch.tensor(0.0)
        self.last_quantized_coordinates = None

    def quantize_coordinates(self, z):
        """Project encoder latents and retain the differentiable alignment coordinates."""
        coordinates, indices = super().forward(self.input_projection(z))
        self.last_quantized_coordinates = coordinates
        return coordinates, indices

    def forward(self, z):
        """Quantize with straight-through gradients and return coordinates or decoder latents."""
        coordinates, indices = self.quantize_coordinates(z)
        zero = z.new_zeros(())
        self.last_soft_entropy_loss = zero
        return (
            self.output_projection(coordinates.to(self.output_projection.weight.dtype)),
            indices.unsqueeze(-1),
            zero,
        )

    @torch.no_grad()
    def encode_indices(self, z):
        """Encode latents as IDs with a trailing single-codebook axis."""
        return self.quantize_coordinates(z)[1].unsqueeze(-1)

    def indices_to_embedding(self, indices):
        """Decode IDs [..., 1] through the learned output projection."""
        coordinates = self.indices_to_coordinates(indices.squeeze(-1))
        return self.output_projection(coordinates.to(self.output_projection.weight.dtype))

    def freeze_codebook(self, idx):
        """Validate the single-codebook index; the FSQ lattice is always fixed."""
        if idx != 0:
            raise IndexError(idx)

    def apply_pending_codebook_updates(self):
        """Return zero because FSQ has no learned codebook to refresh."""
        return 0

    def discard_pending_codebook_updates(self):
        """Return zero because FSQ never queues codebook updates."""
        return 0
