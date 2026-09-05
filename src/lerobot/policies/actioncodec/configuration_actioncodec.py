"""LeRobot configuration for the ActionCodec semantic policy."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig, LRSchedulerConfig

_VISION_ENCODERS = ("small_cnn", "resnet_spatial", "oat_exact_robomimic")
_RGB_MODES = ("minus_one_to_one", "zero_to_one", "none")


@PreTrainedConfig.register_subclass("actioncodec")
@dataclass
class ActionCodecConfig(PreTrainedConfig):
    horizon: int = 20
    latent_horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 16
    action_dim: int = 7
    codebook_size: int = 1024
    quantizer_type: str = "vq"
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    num_codebooks: int = 1
    tokenizer_path: Path | None = None
    task_conditioning: str = "normalized_scalar_plus_token"
    num_tasks: int = 0
    image_size: int | tuple[int, int] = 128
    vision_feature_dim: int = 64
    vision_encoder: str = "oat_exact_robomimic"
    spatial_num_kp: int = 32
    crop_shape: tuple[int, int] | None = (76, 76)
    rgb_mode: str = "minus_one_to_one"
    embed_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    top_k: int = 10
    temperature: float | None = None
    codebook_distance_loss_weight: float = 0.0
    decoded_action_loss_weight: float = 0.0
    decoded_velocity_loss_weight: float = 0.0
    decoded_first_target_loss_weight: float = 0.0
    decoded_overlap_loss_weight: float = 0.0
    decoded_seam_loss_weight: float = 0.0
    continuous_action_indices: tuple[int, ...] | None = None
    physical_unit_scale: float = 1.0
    token_relaxation: str = "gumbel_st"
    token_relaxation_temperature: float = 1.0
    auxiliary_batch_fraction: float = 1.0
    # Extra decoded diagnostics every N training forwards; eval always computes them.
    decoded_metrics_interval: int = 1
    prefix_corruption_prob: float = 0.0
    auxiliary_seed: int = 42
    overlap_shift: int = 16
    task_token_init_seed: int = 42
    optimizer_lr: float = 5e-5
    optimizer_lr_obs_encoder: float = 1e-5
    optimizer_weight_decay: float = 0.0
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 100
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.temperature is None:
            self.temperature = 0.0 if self.quantizer_type == "semantic_fsq" else 1.0
        if isinstance(self.image_size, int):
            if self.image_size <= 0:
                raise ValueError(f"image_size must be positive, got {self.image_size}")
        else:
            if len(self.image_size) != 2:
                raise ValueError(f"image_size must be an int or (height, width), got {self.image_size}")
            image_height, image_width = (int(value) for value in self.image_size)
            if image_height <= 0 or image_width <= 0:
                raise ValueError(f"image_size must have positive dimensions, got {self.image_size}")
            self.image_size = (image_height, image_width)
        if (self.horizon, self.latent_horizon, self.n_obs_steps, self.n_action_steps) != (20, 16, 2, 16):
            raise ValueError(
                "ActionCodec contract requires horizon=20, latent_horizon=16, "
                "n_obs_steps=2, n_action_steps=16"
            )
        self.fsq_levels = tuple(self.fsq_levels)
        if self.quantizer_type not in {"vq", "semantic_fsq"}:
            raise ValueError(f"Unsupported quantizer_type: {self.quantizer_type}")
        if self.quantizer_type == "semantic_fsq" and self.fsq_levels != (8, 5, 5, 5):
            raise ValueError("semantic_fsq requires fsq_levels=[8,5,5,5]")
        expected_vocab = 1000 if self.quantizer_type == "semantic_fsq" else 1024
        if self.codebook_size != expected_vocab or self.num_codebooks != 1:
            raise ValueError(f"ActionCodec requires one codebook with vocab_size={expected_vocab}")
        if self.task_conditioning != "normalized_scalar_plus_token":
            raise ValueError("ActionCodec uses task-token conditioning")
        if self.vision_encoder not in _VISION_ENCODERS:
            raise ValueError(
                f"Unsupported vision_encoder={self.vision_encoder!r}; expected one of {_VISION_ENCODERS}"
            )
        if self.rgb_mode not in _RGB_MODES:
            raise ValueError(f"Unsupported rgb_mode={self.rgb_mode!r}; expected one of {_RGB_MODES}")
        loss_weights = (
            self.codebook_distance_loss_weight,
            self.decoded_action_loss_weight,
            self.decoded_velocity_loss_weight,
            self.decoded_first_target_loss_weight,
            self.decoded_overlap_loss_weight,
            self.decoded_seam_loss_weight,
        )
        if self.quantizer_type == "semantic_fsq" and any(weight != 0 for weight in loss_weights):
            raise ValueError(
                "semantic_fsq supports scalar CE only; distance/decoded auxiliary losses must be zero"
            )
        if any(weight < 0 for weight in loss_weights):
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.token_relaxation != "gumbel_st":  # nosec B105: quantization method, not a credential
            raise ValueError("ActionCodec supports token_relaxation='gumbel_st'")
        if self.token_relaxation_temperature <= 0:
            raise ValueError("token_relaxation_temperature must be positive")
        if not 0 < self.auxiliary_batch_fraction <= 1:
            raise ValueError("auxiliary_batch_fraction must be in (0, 1]")
        if self.decoded_metrics_interval < 1:
            raise ValueError("decoded_metrics_interval must be positive")
        if not 0 <= self.prefix_corruption_prob <= 1:
            raise ValueError("prefix_corruption_prob must be in [0, 1]")
        if self.physical_unit_scale <= 0:
            raise ValueError("physical_unit_scale must be positive")
        if any((self.decoded_overlap_loss_weight, self.decoded_seam_loss_weight)) and (
            self.decoded_action_loss_weight <= 0 or self.overlap_shift != 16
        ):
            raise ValueError("decoded overlap/seam losses require decoded action loss and overlap_shift=16")
        if self.continuous_action_indices is not None:
            self.continuous_action_indices = tuple(int(index) for index in self.continuous_action_indices)
            if any(index < 0 or index >= self.action_dim for index in self.continuous_action_indices):
                raise ValueError("continuous_action_indices must be valid action dimensions")
        if self.crop_shape is not None:
            image_height, image_width = self.image_shape
            crop_h, crop_w = int(self.crop_shape[0]), int(self.crop_shape[1])
            if crop_h <= 0 or crop_w <= 0:
                raise ValueError(f"crop_shape must have positive dimensions, got {self.crop_shape}")
            if self.vision_encoder != "small_cnn" and (crop_h > image_height or crop_w > image_width):
                raise ValueError(f"crop_shape {self.crop_shape} must fit inside image_size={self.image_size}")
            if self.vision_encoder == "oat_exact_robomimic" and (
                crop_h >= image_height or crop_w >= image_width
            ):
                raise ValueError(
                    "oat_exact_robomimic CropRandomizer requires crop_shape dimensions to be "
                    f"strictly smaller than image_size={self.image_size}; use crop_shape=None to disable cropping"
                )

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return the configured encoder input geometry as ``(height, width)``."""
        if isinstance(self.image_size, int):
            return (self.image_size, self.image_size)
        return self.image_size

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(-(self.n_obs_steps - 1), 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=self.optimizer_betas,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("ActionCodec requires at least one observation image")
        if self.action_feature is None or tuple(self.action_feature.shape) != (self.action_dim,):
            raise ValueError(f"ActionCodec requires action shape ({self.action_dim},)")
        if self.robot_state_feature is None:
            raise ValueError("ActionCodec requires observation.state")
        if self.num_tasks < 1:
            raise ValueError("ActionCodec task-token conditioning requires num_tasks >= 1")
