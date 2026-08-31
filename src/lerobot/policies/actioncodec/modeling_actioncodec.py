"""OAT-exact cached semantic policy with a frozen ActionCodec tokenizer."""

from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file

from lerobot.actioncodec.config import load_tokenizer_config
from lerobot.actioncodec.models.tokenizer import ActionCodecTokenizer
from lerobot.lerobot_types import PolicyAction
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_actioncodec import ActionCodecConfig
from .obs_encoder import ObservationEncoder


class FrozenTokenizerAdapter(nn.Module):
    """Loads a semantic tokenizer artifact and prevents optimizer leakage."""

    def __init__(self, config: ActionCodecConfig) -> None:
        super().__init__()
        if config.tokenizer_path is not None:
            tokenizer_config = load_tokenizer_config(config.tokenizer_path)
            contract_path = Path(config.tokenizer_path) / "dataset_contract.json"
            stats_path = Path(config.tokenizer_path) / "action_stats.json"
            if not contract_path.is_file() or not stats_path.is_file():
                raise FileNotFoundError(
                    "ActionCodec tokenizer artifact must contain dataset_contract.json and action_stats.json"
                )
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            for key, expected in {
                "horizon": config.horizon,
                "latent_horizon": config.latent_horizon,
                "action_dim": config.action_dim,
            }.items():
                if key not in contract or int(contract[key]) != expected:
                    raise ValueError(f"Tokenizer dataset contract field {key!r} does not match policy config")
            if not isinstance(stats, dict) or not all(key in stats for key in ("mean", "std")):
                raise ValueError("ActionCodec tokenizer artifact must contain action mean/std statistics")
            mean_shape = tuple(torch.as_tensor(stats["mean"]).shape)
            std_shape = tuple(torch.as_tensor(stats["std"]).shape)
            if mean_shape != (config.action_dim,):
                raise ValueError("Tokenizer action mean statistics do not match action_dim")
            if std_shape != (config.action_dim,):
                raise ValueError("Tokenizer action std statistics do not match action_dim")
            if tokenizer_config.action_dim != config.action_dim or tokenizer_config.horizon != config.horizon:
                raise ValueError("Tokenizer action_dim/horizon does not match ActionCodec policy config")
            if (
                tokenizer_config.latent_horizon != config.latent_horizon
                or tokenizer_config.codebook_size != config.codebook_size
            ):
                raise ValueError(
                    "Tokenizer latent_horizon/codebook_size does not match ActionCodec policy config"
                )
        else:
            tokenizer_config = None
        self.model = ActionCodecTokenizer(
            action_dim=config.action_dim,
            window_size=config.horizon,
            model_dim=256 if tokenizer_config is None else tokenizer_config.model_dim,
            num_tokens=config.latent_horizon,
            codebook_size=config.codebook_size,
            num_codebooks=config.num_codebooks,
            num_heads=8 if tokenizer_config is None else tokenizer_config.num_heads,
            encoder_layers=3 if tokenizer_config is None else tokenizer_config.encoder_layers,
            decoder_layers=3 if tokenizer_config is None else tokenizer_config.decoder_layers,
            encoder_cross_layers=1 if tokenizer_config is None else tokenizer_config.encoder_cross_layers,
            decoder_cross_layers=1 if tokenizer_config is None else tokenizer_config.decoder_cross_layers,
            use_encoder_latent_self_attn=True
            if tokenizer_config is None
            else tokenizer_config.use_encoder_latent_self_attn,
            use_decoder_latent_self_attn=True
            if tokenizer_config is None
            else tokenizer_config.use_decoder_latent_self_attn,
            share_encoder_latent_transformer=False
            if tokenizer_config is None
            else tokenizer_config.share_encoder_latent_transformer,
            share_decoder_latent_transformer=False
            if tokenizer_config is None
            else tokenizer_config.share_decoder_latent_transformer,
            share_encoder_cross_attn=False
            if tokenizer_config is None
            else tokenizer_config.share_encoder_cross_attn,
            share_decoder_cross_attn=False
            if tokenizer_config is None
            else tokenizer_config.share_decoder_cross_attn,
            dropout=0.1 if tokenizer_config is None else tokenizer_config.dropout,
            vq_beta=0.25 if tokenizer_config is None else tokenizer_config.vq_beta,
            soft_assignment_temperature=1.0
            if tokenizer_config is None
            else tokenizer_config.soft_assignment_temperature,
            dead_code_threshold=100 if tokenizer_config is None else tokenizer_config.dead_code_threshold,
            reset_noise_scale=1e-3 if tokenizer_config is None else tokenizer_config.reset_noise_scale,
            decoder_type="perceiver" if tokenizer_config is None else tokenizer_config.decoder_type,
            diffusion_config=None if tokenizer_config is None else tokenizer_config.diffusion_config,
            use_vl_embedder=False if tokenizer_config is None else tokenizer_config.use_vl_embedder,
        )
        if config.tokenizer_path is not None:
            state_path = Path(config.tokenizer_path) / "model.safetensors"
            self.model.load_state_dict(load_file(str(state_path)), strict=True)
        self.horizon = config.horizon
        self.latent_horizon = config.latent_horizon
        self.vocab_size = config.codebook_size
        self.action_dim = config.action_dim
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenTokenizerAdapter:
        """Keep the frozen tokenizer deterministic when its parent policy trains."""
        super().train(False)
        return self

    @torch.inference_mode()
    def tokenize(self, action: torch.Tensor) -> torch.Tensor:
        return self.model.tokenize(action)

    @torch.inference_mode()
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.detokenize(tokens)


def _token_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = pred.eq(target).float().mean()
        k = min(5, int(logits.shape[-1]))
        top5 = logits.topk(k, dim=-1).indices.eq(target.unsqueeze(-1)).any(dim=-1).float().mean()
    return {"token_accuracy": float(acc.item()), "token_top5_acc": float(top5.item())}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight


class _OATBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.heads, self.head_dim = heads, dim // heads
        self.self_norm, self.cross_norm, self.mlp_norm = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.self_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def _reshape(self, value: torch.Tensor) -> torch.Tensor:
        bsz, length, dim = value.shape
        return value.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)

    def memory_kv(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.kv_proj(memory).chunk(2, dim=-1)
        return self._reshape(key), self._reshape(value)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, past=None, memory_kv=None):
        bsz, length, dim = x.shape
        query, key, value = map(self._reshape, self.qkv(self.self_norm(x)).chunk(3, dim=-1))
        if past is not None:
            key, value = torch.cat((past[0], key), -2), torch.cat((past[1], value), -2)
        self_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=past is None and length > 1,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        self_out = self.dropout(self.self_proj(self_out.transpose(1, 2).reshape(bsz, length, dim)))
        x = x + self_out
        q = self._reshape(self.q_proj(self.cross_norm(x)))
        k, v = self.memory_kv(memory) if memory_kv is None else memory_kv
        cross = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.dropout(self.cross_proj(cross.transpose(1, 2).reshape(bsz, length, dim)))
        return x + self.mlp(self.mlp_norm(x)), (key, value)


class OATExactCached(nn.Module):
    """OAT exact causal decoder; generation reuses self/cross attention KV caches."""

    def __init__(self, config: ActionCodecConfig, cond_dim: int) -> None:
        super().__init__()
        if config.embed_dim % config.n_heads:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.vocab_size = config.codebook_size + 1
        self.generation_vocab_size = config.codebook_size
        self.max_seq_len = config.latent_horizon + 1
        self.max_cond_len = config.n_obs_steps
        self.embed_dim = config.embed_dim
        self.token_emb = nn.Embedding(self.vocab_size, self.embed_dim)
        self.token_pos_emb = nn.Parameter(torch.zeros(1, self.max_seq_len, self.embed_dim))
        self.cond_proj = nn.Linear(cond_dim, self.embed_dim)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, self.max_cond_len, self.embed_dim))
        self.drop = nn.Dropout(config.dropout)
        self.cond_encoder = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.Mish(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
        )
        self.blocks = nn.ModuleList(
            [_OATBlock(self.embed_dim, config.n_heads, config.dropout) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self._init_weights()
        self.use_task_token = True
        self.num_tasks = int(config.num_tasks)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.task_token_init_seed))
            self.task_embedding = nn.Embedding(self.num_tasks, self.embed_dim)
            nn.init.xavier_uniform_(self.task_embedding.weight)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def _memory(self, cond: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        if cond.shape[1] > self.max_cond_len:
            raise ValueError("condition exceeds max_cond_len")
        memory = self.cond_encoder(self.drop(self.cond_proj(cond) + self.cond_pos_emb[:, : cond.shape[1]]))
        if task_ids.ndim == 2 and task_ids.shape[-1] == 1:
            task_ids = task_ids[:, 0]
        if task_ids.ndim != 1 or task_ids.min() < 0 or task_ids.max() >= self.task_embedding.num_embeddings:
            raise ValueError("task_ids are out of range")
        return torch.cat((memory, self.task_embedding(task_ids)[:, None]), dim=1)

    def forward(self, input_ids: torch.Tensor, cond: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("token sequence exceeds max_seq_len")
        x = self.drop(self.token_emb(input_ids) + self.token_pos_emb[:, : input_ids.shape[1]])
        memory = self._memory(cond, task_ids)
        for block in self.blocks:
            x, _ = block(x, memory)
        return self.head(self.norm(x))

    @torch.inference_mode()
    def sequence_logprobs(
        self,
        generated_ids: torch.Tensor,
        cond: torch.Tensor,
        task_ids: torch.Tensor,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if generated_ids.ndim != 2 or generated_ids.shape[1] < 2:
            raise ValueError("generated_ids must have shape [B, T>=2]")
        logits = self.forward(generated_ids[:, :-1], cond, task_ids)
        target = generated_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1).gather(-1, target[..., None]).squeeze(-1)
        valid = torch.ones_like(target, dtype=torch.bool)
        if eos_id is not None:
            valid = torch.cumsum(target.eq(eos_id).long(), dim=1) <= 1
        return (log_probs * valid).sum(1) / valid.sum(1).clamp_min(1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        cond: torch.Tensor,
        max_new_tokens: int,
        task_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        memory = self._memory(cond, task_ids)
        memory_cache = [block.memory_kv(memory) for block in self.blocks]
        x = self.drop(self.token_emb(input_ids) + self.token_pos_emb[:, : input_ids.shape[1]])
        past = []
        for block, cache in zip(self.blocks, memory_cache, strict=True):
            x, present = block(x, memory, memory_kv=cache)
            past.append(present)
        logits = self.head(self.norm(x[:, -1:]))
        result = input_ids
        for index in range(max_new_tokens):
            scores = logits[:, -1]
            scores = scores[:, : self.generation_vocab_size]
            if temperature > 0:
                scores = scores / temperature
                if 0 < top_k < scores.shape[-1]:
                    threshold = scores.topk(top_k, dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -float("inf"))
                next_id = torch.multinomial(scores.softmax(-1), 1)
            else:
                next_id = scores.argmax(-1, keepdim=True)
            result = torch.cat((result, next_id), dim=1)
            if index + 1 == max_new_tokens:
                break
            position = input_ids.shape[1] + index
            x = self.drop(self.token_emb(next_id) + self.token_pos_emb[:, position : position + 1])
            for layer, block in enumerate(self.blocks):
                x, past[layer] = block(x, memory, past[layer], memory_cache[layer])
            logits = self.head(self.norm(x))
        return result


class ActionCodecPolicy(PreTrainedPolicy):
    config_class: ClassVar[type[ActionCodecConfig]] = ActionCodecConfig
    name: ClassVar[str] = "actioncodec"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """Pass the checkpoint root through for relative tokenizer artifacts.

        The base implementation consumes ``pretrained_name_or_path`` itself, so the root is
        forwarded to ``__init__`` under a separate name to avoid a duplicate argument.
        """
        kwargs["checkpoint_root"] = pretrained_name_or_path
        return super().from_pretrained(pretrained_name_or_path, *args, **kwargs)

    def __init__(
        self,
        config: ActionCodecConfig,
        dataset_stats: dict[str, Any] | None = None,
        dataset_meta: Any | None = None,
        checkpoint_root: str | Path | None = None,
        **_: Any,
    ) -> None:
        if config.tokenizer_path is not None:
            config.tokenizer_path = Path(config.tokenizer_path)
        if dataset_meta is not None and config.num_tasks == 0:
            config.num_tasks = int(dataset_meta.total_tasks)
        if (
            config.tokenizer_path is not None
            and not config.tokenizer_path.is_absolute()
            and checkpoint_root is not None
        ):
            config.tokenizer_path = Path(checkpoint_root) / config.tokenizer_path
        super().__init__(config)
        config.validate_features()
        self.tokenizer = FrozenTokenizerAdapter(config)
        self.obs_encoder = ObservationEncoder(config)
        self.model = OATExactCached(config, self.obs_encoder.output_dim)
        self._action_queue: deque[PolicyAction] = deque(maxlen=config.n_action_steps)
        self._observation_queues: dict[str, deque[torch.Tensor]] = {}
        self.reset()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the policy and colocate its frozen tokenizer artifact."""
        tokenizer_path = self.config.tokenizer_path
        if tokenizer_path is not None and tokenizer_path.is_dir():
            bundled_path = save_directory / "tokenizer"
            if tokenizer_path.resolve() != bundled_path.resolve():
                shutil.copytree(tokenizer_path, bundled_path, dirs_exist_ok=True)
            self.config.tokenizer_path = Path("tokenizer")
        try:
            super()._save_pretrained(save_directory)
        finally:
            self.config.tokenizer_path = tokenizer_path

    def get_optim_params(self) -> list[dict[str, Any]]:  # type: ignore[override]
        policy_decay, policy_nodecay, obs_decay, obs_nodecay = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_obs = name.startswith("obs_encoder.")
            decay = param.ndim >= 2
            if is_obs and decay:
                obs_decay.append(param)
            elif is_obs:
                obs_nodecay.append(param)
            elif decay:
                policy_decay.append(param)
            else:
                policy_nodecay.append(param)
        groups = [
            {
                "params": policy_decay,
                "lr": self.config.optimizer_lr,
                "weight_decay": self.config.optimizer_weight_decay,
                "name": "policy_decay",
            },
            {
                "params": policy_nodecay,
                "lr": self.config.optimizer_lr,
                "weight_decay": 0.0,
                "name": "policy_nodecay",
            },
            {
                "params": obs_decay,
                "lr": self.config.optimizer_lr_obs_encoder,
                "weight_decay": self.config.optimizer_weight_decay,
                "name": "obs_decay",
            },
            {
                "params": obs_nodecay,
                "lr": self.config.optimizer_lr_obs_encoder,
                "weight_decay": 0.0,
                "name": "obs_nodecay",
            },
        ]
        return [group for group in groups if len(group["params"]) > 0]

    def reset(self) -> None:
        """Clear action and online observation history at an episode boundary."""
        self._action_queue.clear()
        self._observation_queues = {
            key: deque(maxlen=self.config.n_obs_steps) for key in [*self.config.image_features, OBS_STATE]
        }

    def _update_observation_history(self, batch: dict[str, torch.Tensor]) -> None:
        """Append one live observation, or seed history from an explicit temporal batch."""
        for key, queue in self._observation_queues.items():
            if key not in batch:
                raise KeyError(f"{key} is required for ActionCodec online inference")
            value = batch[key]
            live_ndim = 4 if key in self.config.image_features else 2
            if value.ndim == live_ndim + 1:
                if value.shape[1] != self.config.n_obs_steps:
                    raise ValueError(
                        f"{key} temporal input must have {self.config.n_obs_steps} steps, "
                        f"got shape {tuple(value.shape)}"
                    )
                queue.clear()
                queue.extend(value[:, index] for index in range(self.config.n_obs_steps))
                continue
            if value.ndim != live_ndim:
                raise ValueError(
                    f"{key} live input must have {live_ndim} dimensions, got shape {tuple(value.shape)}"
                )
            if not queue:
                queue.extend(value for _ in range(self.config.n_obs_steps))
            else:
                queue.append(value)

    def _online_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Build a two-step policy batch from the online history queues."""
        online = dict(batch)
        for key, queue in self._observation_queues.items():
            if len(queue) != self.config.n_obs_steps:
                raise RuntimeError(f"{key} observation history is not initialized")
            online[key] = torch.stack(tuple(queue), dim=1)
        online.pop(ACTION, None)
        return online

    def _task_ids(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        value = batch.get("task_uid", batch.get("task_index"))
        if value is None:
            raise KeyError("task_uid is required for ActionCodec")
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value[:, 0]
        return value.long()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        action = batch[ACTION].float()
        with torch.inference_mode():
            tokens = self.tokenizer.tokenize(action)
        bos = torch.full(
            (action.shape[0], 1),
            self.config.codebook_size,
            dtype=torch.long,
            device=action.device,
        )
        input_ids = torch.cat((bos, tokens), dim=1)
        features = self.obs_encoder(batch)
        task_ids = self._task_ids(batch)
        logits = self.model(input_ids[:, :-1], features, task_ids)
        target = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        metrics = {"token_ce": float(loss.detach().item()), **_token_metrics(logits, target)}
        if (not self.training) and self.config.num_tasks >= 2:
            swapped_ids = (task_ids + 1) % self.config.num_tasks
            swapped_logits = self.model(input_ids[:, :-1], features, swapped_ids)
            swapped_loss = F.cross_entropy(
                swapped_logits.reshape(-1, swapped_logits.shape[-1]), target.reshape(-1)
            )
            metrics["task_token_swap_ce_gap"] = float((swapped_loss - loss).item())
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        features = self.obs_encoder(batch)
        task_ids = self._task_ids(batch)
        bos = torch.full(
            (features.shape[0], 1),
            self.config.codebook_size,
            dtype=torch.long,
            device=features.device,
        )
        generated = self.model.generate(
            bos,
            features,
            self.config.latent_horizon,
            task_ids,
            self.config.temperature,
            self.config.top_k,
        )
        actions = self.tokenizer.detokenize(generated[:, 1:])
        return actions[:, : self.config.horizon]

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> PolicyAction:
        """Update observation history and return the next action from the current chunk."""
        self.eval()
        self._update_observation_history(batch)
        if not self._action_queue:
            chunk = self.predict_action_chunk(self._online_batch(batch))[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
