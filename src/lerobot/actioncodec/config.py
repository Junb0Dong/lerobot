"""Configuration and artifact contract for the semantic tokenizer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch


@dataclass
class ActionCodecTokenizerConfig:
    """Architecture, optimization and dataset contract of the ActionCodec semantic tokenizer.

    This is the configuration that is serialized next to the tokenizer weights, so it fully
    determines how an action window is encoded into discrete tokens and decoded back. Downstream
    policies reload it to rebuild a compatible tokenizer, which is why :meth:`validate` pins the
    contract-critical fields to the values the token vocabulary was defined with.

    Attributes:
        action_dim: Number of action dimensions per timestep.
        horizon: Number of action timesteps in one window; pinned to ``20`` by the contract.
        latent_horizon: Number of latent tokens emitted per window, i.e. the compression ratio
            against ``horizon``; pinned to ``16`` by the contract.
        model_dim: Width of the encoder and decoder transformers. Matched-h20 default is ``512``.
        codebook_size: Number of entries per codebook, i.e. the token vocabulary size; pinned to
            ``1024`` by the contract.
        num_codebooks: Number of residual quantization stages; pinned to ``1`` by the contract.
        num_heads: Number of attention heads in every transformer block.
        encoder_layers: Depth of each encoder latent self-attention stack.
        decoder_layers: Depth of each decoder latent self-attention stack.
        encoder_cross_layers: Number of cross-attention rounds in the encoder. Matched-h20
            default is ``8``, reused across rounds when ``share_encoder_cross_attn`` is true.
        decoder_cross_layers: Number of cross-attention rounds in the decoder. Matched-h20
            default is ``8``.
        use_encoder_latent_self_attn: Apply latent self-attention between encoder cross-attention
            rounds.
        use_decoder_latent_self_attn: Apply latent self-attention between decoder cross-attention
            rounds.
        share_encoder_latent_transformer: Reuse one latent transformer across encoder rounds.
        share_decoder_latent_transformer: Reuse one latent transformer across decoder rounds.
        share_encoder_cross_attn: Reuse one cross-attention block across encoder rounds.
        share_decoder_cross_attn: Reuse one cross-attention block across decoder rounds.
        dropout: Dropout probability used throughout the tokenizer.
        vq_beta: Weight of the commitment term that keeps encoder outputs near their codeword.
            Matched-h20 default is ``1.0``.
        soft_assignment_temperature: Temperature of the soft codebook assignment used by the
            codebook entropy regularizer; must be positive.
        dead_code_threshold: Number of steps a codeword may stay unused before it is reset onto a
            live encoder output.
        reset_noise_scale: Magnitude of the noise added when reviving a dead codeword.
        decoder_type: ``"diffusion"`` (default) for the iterative denoising decoder, or
            ``"perceiver"`` for single-pass regression.
        diffusion_config: Extra keyword arguments forwarded to the diffusion decoder, ignored when
            ``decoder_type`` is ``"perceiver"``.
        use_vl_embedder: Instantiate the vision-language embedder that powers the optional CLIP
            style auxiliary loss. Default ``False``; CLIP is not part of the matched-h20 recipe.
        window_stride: Episode-local stride between tokenizer window starts. ``4`` matches the
            source HDF5 chunking; ``1`` is every frame.
        learning_rate: AdamW learning rate used by the standalone trainer.
        weight_decay: AdamW weight decay used by the standalone trainer.
        device: Torch device string the trainer moves the model to.
        steps: Number of optimizer steps the trainer runs. Matched-h20 default is ``20000``.
        batch_size: Number of action windows per training batch. Matched-h20 default is ``512``.
        num_workers: Dataloader worker processes. Matched-h20 default is ``8``.
        seed: Seed applied to ``random``, ``numpy`` and ``torch``.
        action_key: Dataset feature key holding the action window.
        delta_state_key: Dataset feature key holding the per-step state deltas consumed by the
            semantic alignment losses.
    """

    action_dim: int = 7
    horizon: int = 20
    latent_horizon: int = 16
    model_dim: int = 512
    codebook_size: int = 1024
    num_codebooks: int = 1
    num_heads: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 3
    encoder_cross_layers: int = 8
    decoder_cross_layers: int = 8
    use_encoder_latent_self_attn: bool = True
    use_decoder_latent_self_attn: bool = True
    share_encoder_latent_transformer: bool = True
    share_decoder_latent_transformer: bool = True
    share_encoder_cross_attn: bool = True
    share_decoder_cross_attn: bool = True
    dropout: float = 0.1
    vq_beta: float = 1.0
    soft_assignment_temperature: float = 1.0
    dead_code_threshold: int = 100
    reset_noise_scale: float = 1e-3
    decoder_type: str = "diffusion"
    diffusion_config: dict[str, Any] | None = None
    use_vl_embedder: bool = False
    window_stride: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    device: str = "cpu"
    steps: int = 20000
    batch_size: int = 512
    num_workers: int = 8
    seed: int = 42
    action_key: str = "action"
    delta_state_key: str = "delta_state"

    def validate(self) -> None:
        """Check the fields that the token vocabulary and the policy interface depend on.

        Raises:
            ValueError: If the horizon, latent horizon or codebook layout deviates from the
                semantic contract, or if a dimension, temperature or decoder type is invalid.
        """
        if self.horizon != 20 or self.latent_horizon != 16:
            raise ValueError("ActionCodec semantic contract requires horizon=20 and latent_horizon=16")
        if self.codebook_size != 1024 or self.num_codebooks != 1:
            raise ValueError("ActionCodec semantic contract requires one codebook with vocab_size=1024")
        if self.action_dim <= 0 or self.model_dim <= 0:
            raise ValueError("action_dim and model_dim must be positive")
        if self.decoder_type not in {"perceiver", "diffusion"}:
            raise ValueError(f"Unsupported decoder_type: {self.decoder_type}")
        if self.soft_assignment_temperature <= 0:
            raise ValueError("soft_assignment_temperature must be positive")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration after re-checking the semantic contract.

        Returns:
            Plain dictionary of every field, ready to be written as JSON.

        Raises:
            ValueError: If :meth:`validate` rejects the current field values.
        """
        self.validate()
        return asdict(self)


def save_tokenizer_artifact(
    directory: str | Path,
    model: Any,
    config: ActionCodecTokenizerConfig,
    *,
    action_stats: dict[str, Any] | None = None,
    dataset_contract: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Write a self-contained tokenizer checkpoint to ``directory``.

    The directory is the unit a policy reloads, so it always carries the weights, the
    configuration and the normalization statistics the weights were trained under. It receives
    ``model.safetensors``, ``model_config.json``, ``action_stats.json`` and
    ``dataset_contract.json``, plus ``training_state.pt`` when resumable state is supplied.

    Args:
        directory: Destination directory; created together with its parents if missing.
        model: Module whose ``state_dict`` is saved as ``model.safetensors``.
        config: Tokenizer configuration, revalidated before anything is written.
        action_stats: Action normalization statistics, typically the dataset mean and std.
        dataset_contract: Provenance of the training data, such as the repo id, fps and horizon.
        training_state: Optimizer and step state for resuming; skipped when ``None``.

    Raises:
        ValueError: If ``config`` violates the semantic contract.
    """
    config.validate()
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state_dict, output / "model.safetensors")
    (output / "model_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output / "action_stats.json").write_text(
        json.dumps(action_stats or {}, indent=2, default=_json_default), encoding="utf-8"
    )
    (output / "dataset_contract.json").write_text(
        json.dumps(dataset_contract or {}, indent=2, default=_json_default), encoding="utf-8"
    )
    if training_state is not None:
        torch.save(training_state, output / "training_state.pt")


def _json_default(value: Any) -> Any:
    """Coerce tensors and numpy scalars into JSON-serializable values.

    Args:
        value: Object that the default JSON encoder could not handle.

    Returns:
        A list for array-like values, or a Python scalar for zero-dimensional ones.

    Raises:
        TypeError: If the object exposes neither ``tolist`` nor ``item``.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_tokenizer_config(directory: str | Path) -> ActionCodecTokenizerConfig:
    """Read back the configuration written by :func:`save_tokenizer_artifact`.

    Args:
        directory: Checkpoint directory containing ``model_config.json``.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If ``model_config.json`` is missing.
        ValueError: If the stored fields violate the semantic contract.
    """
    payload = json.loads((Path(directory) / "model_config.json").read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(ActionCodecTokenizerConfig)}
    config = ActionCodecTokenizerConfig(**{key: value for key, value in payload.items() if key in allowed})
    config.validate()
    return config
