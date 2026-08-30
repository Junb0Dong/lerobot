"""LeRobot configuration for the ActionCodec semantic policy."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("actioncodec")
@dataclass
class ActionCodecConfig(PreTrainedConfig):
    horizon: int = 20
    latent_horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 16
    action_dim: int = 7
    codebook_size: int = 1024
    num_codebooks: int = 1
    tokenizer_path: Path | None = None
    task_conditioning: str = "normalized_scalar_plus_token"
    num_tasks: int = 0
    image_size: int = 128
    vision_feature_dim: int = 64
    embed_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    top_k: int = 10
    temperature: float = 0.0
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.0
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.horizon, self.latent_horizon, self.n_obs_steps, self.n_action_steps) != (20, 16, 2, 16):
            raise ValueError(
                "ActionCodec contract requires horizon=20, latent_horizon=16, "
                "n_obs_steps=2, n_action_steps=16"
            )
        if self.codebook_size != 1024 or self.num_codebooks != 1:
            raise ValueError("ActionCodec contract requires one vocab-1024 codebook")
        if self.task_conditioning != "normalized_scalar_plus_token":
            raise ValueError("ActionCodec uses task-token conditioning")

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
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("ActionCodec requires at least one observation image")
        if self.action_feature is None or tuple(self.action_feature.shape) != (self.action_dim,):
            raise ValueError(f"ActionCodec requires action shape ({self.action_dim},)")
        if self.robot_state_feature is None:
            raise ValueError("ActionCodec requires observation.state")
        if self.num_tasks < 2:
            raise ValueError("ActionCodec task-token conditioning requires num_tasks >= 2")
