"""Observation encoders for the ActionCodec semantic policy.

Ports the oat-exact-policy ``SimpleObservationEncoder`` family onto LeRobot batch keys:
one encoder per camera from ``config.image_features``, concatenated robot state, and a
normalized task scalar. ``resnet_spatial`` is the default; ``oat_exact_robomimic`` is the
optional OAT-exact vision path and requires ``robomimic``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _robomimic_available, require_package

from .configuration_actioncodec import ActionCodecConfig

if TYPE_CHECKING or _robomimic_available:
    import robomimic.models.base_nets as rmbn
else:
    rmbn = None


def random_crop_2d(images: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    """Per-image torch-RNG crop used identically in train and eval."""
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got {tuple(images.shape)}")
    _, _, height, width = images.shape
    if crop_height > height or crop_width > width:
        raise ValueError("Crop cannot exceed image dimensions")
    rows = torch.randint(0, height - crop_height + 1, (images.shape[0],), device=images.device)
    cols = torch.randint(0, width - crop_width + 1, (images.shape[0],), device=images.device)
    return torch.stack(
        [
            image[:, row : row + crop_height, col : col + crop_width]
            for image, row, col in zip(images, rows.tolist(), cols.tolist(), strict=True)
        ]
    )


def _as_btchw(images: torch.Tensor) -> torch.Tensor:
    """Accept ``[B, T, C, H, W]`` or channel-last ``[B, T, H, W, C]`` RGB tensors."""
    if images.ndim != 5:
        raise ValueError(f"images must have shape [B, T, C, H, W], got {tuple(images.shape)}")
    if images.shape[-1] == 3 and images.shape[2] != 3:
        images = images.permute(0, 1, 4, 2, 3)
    if images.shape[2] != 3:
        raise ValueError(f"RGB image must have three channels, got {images.shape[2]}")
    return images


def _normalize_rgb(images: torch.Tensor, rgb_mode: str) -> torch.Tensor:
    """Map uint8 or unit-interval RGB to the requested encoder range."""
    x = images.float()
    byte_valued = bool(x.numel() and float(x.max()) > 1.5)
    if rgb_mode == "minus_one_to_one":
        return x / 127.5 - 1.0 if byte_valued else x * 2.0 - 1.0
    if rgb_mode in {"zero_to_one", "none"}:
        return x / 255.0 if byte_valued else x
    raise ValueError(f"Unsupported rgb_mode={rgb_mode!r}")


def _flatten_bt(images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    bsz, steps, channels, height, width = images.shape
    return images.reshape(bsz * steps, channels, height, width), bsz, steps


class SmallCnnEncoder(nn.Module):
    """Legacy three-layer CNN kept for old checkpoints and cheap tests."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
        )

    def forward(self, images: torch.Tensor, image_size: int) -> torch.Tensor:
        images = _as_btchw(images)
        flat, bsz, steps = _flatten_bt(images)
        if flat.numel() and float(flat.max()) > 1.5:
            flat = flat / 255.0
        flat = F.interpolate(flat, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return self.net(flat).view(bsz, steps, -1)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(16, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.proj: nn.Module = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class SpatialSoftmax(nn.Module):
    def __init__(self, channels: int, num_kp: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_kp = int(num_kp)
        self.temperature = float(temperature)
        self.keypoint_proj = nn.Conv2d(self.channels, self.num_kp, kernel_size=1)
        self.register_buffer("_coords", torch.empty(0), persistent=False)

    def _coords_for(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        expected = height * width * 2
        if self._coords.numel() == expected and self._coords.device == device and self._coords.dtype == dtype:
            return self._coords
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self._coords = coords
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        logits = self.keypoint_proj(x).reshape(bsz, self.num_kp, height * width)
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        coords = self._coords_for(height, width, x.device, x.dtype)
        keypoints = torch.einsum("bkh,hd->bkd", probs, coords)
        return keypoints.reshape(bsz, self.num_kp * 2)


class ResNetSpatialEncoder(nn.Module):
    """Self-contained OAT-style ResNet + SpatialSoftmax encoder (no robomimic)."""

    def __init__(self, feature_dim: int = 64, num_kp: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            _group_norm(32),
            nn.SiLU(),
        )
        self.backbone = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.pool = SpatialSoftmax(256, num_kp=num_kp)
        self.proj = nn.Sequential(nn.Linear(num_kp * 2, feature_dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.backbone(self.stem(x))))


def _replace_batch_norm_with_oat_group_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = child.num_features // 16
            if groups <= 0 or child.num_features % groups:
                raise ValueError(f"OAT GroupNorm rule is invalid for {child.num_features} channels")
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _replace_batch_norm_with_oat_group_norm(child)


class OATExactRobomimicEncoder(nn.Module):
    """Per-camera robomimic VisualCore with BN→GroupNorm(C/16), matching oat-exact."""

    def __init__(self, image_keys: list[str], crop_shape: tuple[int, int] = (76, 76)) -> None:
        super().__init__()
        require_package("robomimic", extra="robomimic")
        if rmbn is None:
            raise ImportError("vision_encoder='oat_exact_robomimic' requires robomimic")
        self.image_keys = list(image_keys)
        self.crop_shape = (int(crop_shape[0]), int(crop_shape[1]))
        cropped_shape = (3, *self.crop_shape)
        self.networks = nn.ModuleDict()
        for key in self.image_keys:
            net = rmbn.VisualCore(
                input_shape=cropped_shape,
                feature_dimension=64,
                backbone_class="ResNet18Conv",
                backbone_kwargs={"input_channels": 3, "input_coord_conv": False},
                pool_class="SpatialSoftmax",
                pool_kwargs={"num_kp": 32, "temperature": 1.0, "noise": 0.0},
                flatten=True,
            )
            _replace_batch_norm_with_oat_group_norm(net)
            self.networks[key] = net
        self.feature_dim = 64

    def encode_dict(self, images_by_key: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for key in self.image_keys:
            value = images_by_key[key]
            bsz, steps = value.shape[:2]
            cropped = random_crop_2d(value.reshape(bsz * steps, *value.shape[2:]), *self.crop_shape)
            outputs.append(self.networks[key](cropped).reshape(bsz, steps, self.feature_dim))
        return outputs


class ObservationEncoder(nn.Module):
    """Encode N cameras + state + task scalar into the OAT condition sequence."""

    def __init__(self, config: ActionCodecConfig) -> None:
        super().__init__()
        self.image_keys = list(config.image_features)
        self.state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature is not None else 0
        self.image_size = config.image_size
        self.num_tasks = max(1, config.num_tasks)
        self.vision_encoder = str(config.vision_encoder)
        self.rgb_mode = str(config.rgb_mode)
        self.crop_shape = (
            None if config.crop_shape is None else (int(config.crop_shape[0]), int(config.crop_shape[1]))
        )
        self.vision_feature_dim = int(config.vision_feature_dim)
        if self.vision_encoder == "small_cnn":
            self.image_encoders = nn.ModuleList(
                [SmallCnnEncoder(self.vision_feature_dim) for _ in self.image_keys]
            )
            self.robomimic_encoder = None
        elif self.vision_encoder == "resnet_spatial":
            self.image_encoders = nn.ModuleList(
                [
                    ResNetSpatialEncoder(
                        feature_dim=self.vision_feature_dim, num_kp=int(config.spatial_num_kp)
                    )
                    for _ in self.image_keys
                ]
            )
            self.robomimic_encoder = None
        elif self.vision_encoder == "oat_exact_robomimic":
            if self.crop_shape is None:
                raise ValueError("oat_exact_robomimic requires crop_shape")
            self.image_encoders = nn.ModuleList()
            self.robomimic_encoder = OATExactRobomimicEncoder(self.image_keys, crop_shape=self.crop_shape)
            self.vision_feature_dim = int(self.robomimic_encoder.feature_dim)
        else:
            raise ValueError(f"Unsupported vision_encoder={self.vision_encoder!r}")
        self.output_dim = len(self.image_keys) * self.vision_feature_dim + self.state_dim + 1

    def _prepare_camera(self, images: torch.Tensor, *, crop: bool) -> torch.Tensor:
        images = _as_btchw(images)
        flat, bsz, steps = _flatten_bt(_normalize_rgb(images, self.rgb_mode))
        flat = F.interpolate(
            flat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        if crop and self.crop_shape is not None:
            flat = random_crop_2d(flat, *self.crop_shape)
        _, _, height, width = flat.shape
        return flat.view(bsz, steps, 3, height, width)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.robomimic_encoder is not None:
            prepared = {key: self._prepare_camera(batch[key], crop=False) for key in self.image_keys}
            features = self.robomimic_encoder.encode_dict(prepared)
        elif self.vision_encoder == "small_cnn":
            features = [
                encoder(batch[key], self.image_size)
                for key, encoder in zip(self.image_keys, self.image_encoders, strict=True)
            ]
        else:
            features = []
            for key, encoder in zip(self.image_keys, self.image_encoders, strict=True):
                prepared = self._prepare_camera(batch[key], crop=True)
                flat, bsz, steps = _flatten_bt(prepared)
                features.append(encoder(flat).view(bsz, steps, -1))
        state = batch[OBS_STATE].float()
        if state.ndim != 3 or state.shape[-1] != self.state_dim:
            raise ValueError(f"observation.state must have shape [B, T, {self.state_dim}]")
        task_uid = batch.get("task_uid", batch.get("task_index"))
        if task_uid is None:
            raise KeyError("task_uid is required by ActionCodec task-token conditioning")
        if task_uid.ndim == 2 and task_uid.shape[-1] == 1:
            task_uid = task_uid[:, 0]
        if task_uid.ndim == 3:
            task_uid = task_uid[:, 0, 0]
        task_scalar = 2 * task_uid.float().view(-1, 1, 1) / max(1, self.num_tasks - 1) - 1
        features.extend([state, task_scalar.expand(-1, state.shape[1], -1)])
        return torch.cat(features, dim=-1)
