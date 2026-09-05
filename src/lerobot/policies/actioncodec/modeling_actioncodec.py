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
from lerobot.actioncodec.models.fsq import FSQGrid
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
            if tokenizer_config.quantizer_type != config.quantizer_type or (
                config.quantizer_type == "semantic_fsq"
                and tokenizer_config.fsq_levels != tuple(config.fsq_levels)
            ):
                raise ValueError("Tokenizer quantizer_type/fsq_levels does not match policy config")
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
            quantizer_type=config.quantizer_type,
            fsq_levels=config.fsq_levels,
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

    def decode_train(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode continuous latents without disabling gradients to the policy inputs."""
        return self.model.decode_train(latents)[..., : self.action_dim]

    @property
    def codebook_weight(self) -> torch.Tensor:
        return self.model.quantizer.codebooks[0].weight


def _codebook_distance_loss(
    logits: torch.Tensor, target: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    """Expected squared code distance, normalized by the mean over all code pairs.

    Exclude BOS, freeze the tokenizer geometry, and use FP32 even under AMP.
    This is E[distance(code, target)], not distance(E[code], target).
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        weight = codebook.detach().float()
        weight = weight - weight.mean(dim=0)
        target_weight = F.embedding(target, weight)
        distances = (
            target_weight.square().sum(-1, keepdim=True)
            + weight.square().sum(-1)
            - 2 * target_weight @ weight.t()
        ).clamp_min(0)
        distances.scatter_(-1, target.unsqueeze(-1), 0)
        mean_pair_distance = 2 * weight.square().sum(-1).mean()
        probabilities = logits[..., : weight.shape[0]].float().softmax(-1)
        return (probabilities * (distances / mean_pair_distance)).sum(-1).mean()


def _relaxed_one_hot(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator | None = None,
    stochastic: bool = True,
) -> torch.Tensor:
    values = logits.float()
    if stochastic:
        uniform = torch.rand(values.shape, device=values.device, generator=generator)
        gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-6)))
        soft = F.softmax((values + gumbel) / float(temperature), dim=-1)
    else:
        soft = F.softmax(values / float(temperature), dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), values.shape[-1]).to(soft)
    return hard + soft - soft.detach()


def _continuous_indices(
    action_dim: int, indices: tuple[int, ...] | None, device: torch.device
) -> torch.Tensor:
    if indices is None:
        return torch.arange(action_dim, device=device)
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def _physical_loss(
    error: torch.Tensor, indices: torch.Tensor, unit_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = error.index_select(-1, indices).float()
    loss = F.smooth_l1_loss(selected / float(unit_scale), torch.zeros_like(selected))
    return loss, selected.abs().mean()


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
        return self.forward_embeddings(self.token_emb(input_ids), cond, task_ids)

    def forward_embeddings(
        self, embeddings: torch.Tensor, cond: torch.Tensor, task_ids: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.shape[1] > self.max_seq_len:
            raise ValueError("token sequence exceeds max_seq_len")
        x = self.drop(embeddings + self.token_pos_emb[:, : embeddings.shape[1]])
        memory = self._memory(cond, task_ids)
        for block in self.blocks:
            x, _ = block(x, memory)
        return self.head(self.norm(x))

    def generate_differentiable(
        self,
        bos_embedding: torch.Tensor,
        cond: torch.Tensor,
        max_new_tokens: int,
        task_ids: torch.Tensor,
        codebook_weight: torch.Tensor,
        temperature: float,
        prefix_tokens: torch.Tensor,
        prefix_corruption_prob: float,
        seed: int,
        stochastic: bool,
    ) -> torch.Tensor:
        """Generate ST token embeddings and codebook latents through the causal KV cache."""
        memory = self._memory(cond, task_ids)
        memory_cache = [block.memory_kv(memory) for block in self.blocks]
        x = self.drop(bos_embedding + self.token_pos_emb[:, :1])
        past = []
        for block, cache in zip(self.blocks, memory_cache, strict=True):
            x, present = block(x, memory, memory_kv=cache)
            past.append(present)
        logits = self.head(self.norm(x[:, -1:]))[:, 0, : self.generation_vocab_size]
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(int(seed))
        latents = []
        for index in range(max_new_tokens):
            probabilities = _relaxed_one_hot(
                logits,
                temperature,
                generator=generator,
                stochastic=stochastic,
            )
            latents.append(probabilities @ codebook_weight)
            if index + 1 == max_new_tokens:
                break
            generated_embedding = (
                probabilities @ self.token_emb.weight[: self.generation_vocab_size]
            ).unsqueeze(1)
            target_embedding = self.token_emb(prefix_tokens[:, index : index + 1])
            if prefix_corruption_prob <= 0:
                next_embedding = target_embedding
            elif prefix_corruption_prob >= 1:
                next_embedding = generated_embedding
            else:
                use_generated = torch.rand(
                    (logits.shape[0], 1, 1), device=logits.device, generator=generator
                ) < float(prefix_corruption_prob)
                next_embedding = torch.where(use_generated, generated_embedding, target_embedding)
            position = index + 1
            x = self.drop(next_embedding + self.token_pos_emb[:, position : position + 1])
            for layer, block in enumerate(self.blocks):
                x, past[layer] = block(x, memory, past[layer], memory_cache[layer])
            logits = self.head(self.norm(x))[:, 0, : self.generation_vocab_size]
        return torch.stack(latents, dim=1)

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


class _FSQInput(nn.Module):
    def __init__(self, dim, levels):
        super().__init__()
        self.grid = FSQGrid(levels)
        self.projection = nn.Linear(4, dim)
        self.bos = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.bos, std=0.02)

    def forward(self, ids):
        coordinates = self.grid.indices_to_coordinates(ids)
        return torch.where(ids[..., None] == 1000, self.bos, self.projection(coordinates))


class _FSQHeads(nn.Module):
    def __init__(self, dim, levels):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, level, bias=False) for level in levels])

    def forward(self, x):
        return tuple(head(x) for head in self.heads)


def _scalar_ce(logits, classes):
    return torch.stack(
        [
            F.cross_entropy(head.reshape(-1, head.shape[-1]), classes[..., i].reshape(-1))
            for i, head in enumerate(logits)
        ]
    )


class FSQOATExactCached(OATExactCached):
    """Four independent scalar heads per causal position, sharing OAT attention/cache."""

    def __init__(self, config, cond_dim):
        super().__init__(config, cond_dim)
        self.token_emb = _FSQInput(config.embed_dim, config.fsq_levels)
        self.head = _FSQHeads(config.embed_dim, config.fsq_levels)
        for module in (self.token_emb.projection, *self.head.heads):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.inference_mode()
    def generate(self, input_ids, cond, max_new_tokens, task_ids, temperature=0.0, top_k=0):
        if input_ids.shape[1] + max_new_tokens > self.max_seq_len:
            raise ValueError("token sequence exceeds max_seq_len")
        memory = self._memory(cond, task_ids)
        memory_cache = [block.memory_kv(memory) for block in self.blocks]
        x = self.drop(self.token_emb(input_ids) + self.token_pos_emb[:, : input_ids.shape[1]])
        past = []
        for block, cache in zip(self.blocks, memory_cache, strict=True):
            x, present = block(x, memory, memory_kv=cache)
            past.append(present)
        result = input_ids
        for index in range(max_new_tokens):
            logits = self.head(self.norm(x[:, -1:]))
            classes = []
            for head in logits:
                scores = head[:, -1].float()
                if temperature > 0:
                    scores = scores / temperature
                    if 0 < top_k < scores.shape[-1]:
                        threshold = scores.topk(top_k, dim=-1).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, -float("inf"))
                    classes.append(torch.multinomial(scores.softmax(-1), 1).squeeze(-1))
                else:
                    classes.append(scores.argmax(-1))
            next_id = self.token_emb.grid.scalar_classes_to_indices(torch.stack(classes, -1))[:, None]
            result = torch.cat((result, next_id), dim=1)
            if index + 1 == max_new_tokens:
                break
            position = input_ids.shape[1] + index
            x = self.drop(self.token_emb(next_id) + self.token_pos_emb[:, position : position + 1])
            for layer, block in enumerate(self.blocks):
                x, past[layer] = block(x, memory, past[layer], memory_cache[layer])
        return result

    @torch.inference_mode()
    def sequence_logprobs(self, generated_ids, cond, task_ids, eos_id=None):
        if eos_id is not None:
            raise ValueError("FSQ uses a fixed horizon without EOS")
        logits = self(generated_ids[:, :-1], cond, task_ids)
        classes = self.token_emb.grid.indices_to_scalar_classes(generated_ids[:, 1:])
        return (
            torch.stack(
                [
                    head.float().log_softmax(-1).gather(-1, classes[..., i, None]).squeeze(-1)
                    for i, head in enumerate(logits)
                ]
            )
            .sum(0)
            .mean(1)
        )


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
        model_class = FSQOATExactCached if config.quantizer_type == "semantic_fsq" else OATExactCached
        self.model = model_class(config, self.obs_encoder.output_dim)
        action_stats = None if dataset_stats is None else dataset_stats.get(ACTION)
        self.register_buffer(
            "action_mean",
            torch.zeros(config.action_dim)
            if action_stats is None
            else torch.as_tensor(action_stats["mean"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_std",
            torch.ones(config.action_dim)
            if action_stats is None
            else torch.as_tensor(action_stats["std"], dtype=torch.float32),
            persistent=False,
        )
        self._has_action_stats = action_stats is not None
        self._auxiliary_call_index = 0
        self._auxiliary_train_call_index = 0
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

    def _tokenizer_latents(self, logits: torch.Tensor, seed: int, stochastic: bool) -> torch.Tensor:
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(int(seed))
        probabilities = _relaxed_one_hot(
            logits[..., : self.config.codebook_size],
            self.config.token_relaxation_temperature,
            generator=generator,
            stochastic=stochastic,
        )
        return probabilities @ self.tokenizer.codebook_weight

    def _physical_terms(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = self.action_mean.to(device=prediction.device, dtype=torch.float32)
        std = self.action_std.to(device=prediction.device, dtype=torch.float32)
        prediction_raw = prediction.float() * std + mean
        target_raw = target.float() * std + mean
        all_indices = torch.arange(self.config.action_dim, device=prediction.device)
        continuous = _continuous_indices(
            self.config.action_dim, self.config.continuous_action_indices, prediction.device
        )
        recon_loss, recon_mae = _physical_loss(
            prediction_raw - target_raw, all_indices, self.config.physical_unit_scale
        )
        velocity_loss, velocity_mae = _physical_loss(
            (prediction_raw[:, 1:] - prediction_raw[:, :-1]) - (target_raw[:, 1:] - target_raw[:, :-1]),
            continuous,
            self.config.physical_unit_scale,
        )
        first_loss, first_mae = _physical_loss(
            prediction_raw[:, 0] - target_raw[:, 0], all_indices, self.config.physical_unit_scale
        )
        return {
            "recon_loss": recon_loss,
            "recon_mae": recon_mae,
            "velocity_loss": velocity_loss,
            "velocity_mae": velocity_mae,
            "first_loss": first_loss,
            "first_mae": first_mae,
            "raw": prediction_raw,
            "target_raw": target_raw,
        }

    def _decode_training_branch(
        self,
        action: torch.Tensor,
        features: torch.Tensor,
        task_ids: torch.Tensor,
        seed: int,
        compute_metrics: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, str]:
        with torch.inference_mode():
            token_ids = self.tokenizer.tokenize(action)
        tokens = token_ids.clone()
        bos = torch.full(
            (action.shape[0], 1),
            self.config.codebook_size,
            dtype=torch.long,
            device=action.device,
        )
        input_ids = torch.cat((bos, tokens), dim=1)
        teacher_logits = self.model(input_ids[:, :-1], features, task_ids)
        # Keep this forward's dropout draws independent of diagnostic frequency.
        # Only the expensive frozen decode below is conditional.
        probability_seed = seed + 1
        teacher_decoded = None
        free_decoded = None
        if self.config.prefix_corruption_prob == 0 and self.training:
            teacher_latents = self._tokenizer_latents(teacher_logits, probability_seed, stochastic=True)
            teacher_decoded = self.tokenizer.decode_train(teacher_latents)
            auxiliary_decoded = teacher_decoded
            mode = "teacher_forced"
        else:
            if compute_metrics:
                with torch.no_grad():
                    teacher_latents = self._tokenizer_latents(
                        teacher_logits, probability_seed, stochastic=False
                    )
                    teacher_decoded = self.tokenizer.decode_train(teacher_latents)
            bos_embedding = self.model.token_emb(bos)
            auxiliary_latents = self.model.generate_differentiable(
                bos_embedding,
                features,
                self.config.latent_horizon,
                task_ids,
                self.tokenizer.codebook_weight,
                self.config.token_relaxation_temperature,
                tokens,
                self.config.prefix_corruption_prob,
                seed + 2,
                stochastic=self.training,
            )
            auxiliary_decoded = self.tokenizer.decode_train(auxiliary_latents)
            mode = "free_running" if self.config.prefix_corruption_prob == 1 else "scheduled_sampling"
        if self.config.prefix_corruption_prob == 1:
            free_decoded = auxiliary_decoded
        elif compute_metrics:
            # Diagnostic AR dropout must not change the RNG stream used by the paired loss branch.
            devices = [features.device] if features.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                free_latents = self.model.generate_differentiable(
                    self.model.token_emb(bos),
                    features,
                    self.config.latent_horizon,
                    task_ids,
                    self.tokenizer.codebook_weight,
                    self.config.token_relaxation_temperature,
                    tokens,
                    1.0,
                    seed + 3,
                    stochastic=False,
                )
                free_decoded = self.tokenizer.decode_train(free_latents)
        return teacher_decoded, auxiliary_decoded, free_decoded, mode

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
        if self.config.quantizer_type == "semantic_fsq":
            classes = self.model.token_emb.grid.indices_to_scalar_classes(target)
            head_ce = _scalar_ce(logits, classes)
            loss = head_ce.mean()
            correct = torch.stack([head.argmax(-1) for head in logits], -1).eq(classes)
            metrics = {
                "token_ce": float(loss.detach()),
                "token_nll": float(head_ce.sum().detach()),
                "scalar_accuracy": float(correct.float().mean()),
                "token_accuracy": float(correct.all(-1).float().mean()),
            }
            for index in range(4):
                metrics[f"scalar_{index}_ce"] = float(head_ce[index].detach())
                metrics[f"scalar_{index}_accuracy"] = float(correct[..., index].float().mean())
            if not self.training and self.config.num_tasks >= 2:
                swapped = self.model(input_ids[:, :-1], features, (task_ids + 1) % self.config.num_tasks)
                metrics["task_token_swap_ce_gap"] = float(
                    (_scalar_ce(swapped, classes).mean() - loss).detach()
                )
            return loss, metrics
        token_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss = token_ce
        metrics = {"token_ce": float(token_ce.detach().item()), **_token_metrics(logits, target)}
        if self.config.codebook_distance_loss_weight > 0:
            distance_loss = _codebook_distance_loss(logits, target, self.tokenizer.codebook_weight)
            weighted_distance = self.config.codebook_distance_loss_weight * distance_loss
            loss = loss + weighted_distance
            metrics["codebook_distance_loss"] = float(distance_loss.detach())
            metrics["weighted_codebook_distance_loss"] = float(weighted_distance.detach())
            metrics["total_loss"] = float(loss.detach())
        auxiliary_enabled = any(
            (
                self.config.decoded_action_loss_weight,
                self.config.decoded_velocity_loss_weight,
                self.config.decoded_first_target_loss_weight,
                self.config.decoded_overlap_loss_weight,
                self.config.decoded_seam_loss_weight,
            )
        )
        if not auxiliary_enabled:
            if (not self.training) and self.config.num_tasks >= 2:
                swapped_ids = (task_ids + 1) % self.config.num_tasks
                swapped_logits = self.model(input_ids[:, :-1], features, swapped_ids)
                swapped_loss = F.cross_entropy(
                    swapped_logits.reshape(-1, swapped_logits.shape[-1]), target.reshape(-1)
                )
                metrics["task_token_swap_ce_gap"] = float((swapped_loss - token_ce).item())
            return loss, metrics
        if not self._has_action_stats:
            raise ValueError("ActionCodec physical auxiliary training requires dataset action stats")
        pair = batch.get("_actioncodec_pair")
        if (
            self.config.decoded_overlap_loss_weight > 0 or self.config.decoded_seam_loss_weight > 0
        ) and pair is None:
            raise ValueError("ActionCodec overlap/seam losses require an episode-safe paired batch")
        seed = self.config.auxiliary_seed + self._auxiliary_call_index * 10_000
        self._auxiliary_call_index += 1
        if self.training:
            self._auxiliary_train_call_index += 1
        compute_metrics = (
            not self.training or self._auxiliary_train_call_index % self.config.decoded_metrics_interval == 0
        )
        auxiliary_count = max(1, round(action.shape[0] * self.config.auxiliary_batch_fraction))
        teacher_a, auxiliary_a, free_a, mode = self._decode_training_branch(
            action[:auxiliary_count],
            features[:auxiliary_count],
            task_ids[:auxiliary_count],
            seed,
            compute_metrics=compute_metrics,
        )
        target_a = action[:auxiliary_count]
        predictions = [(auxiliary_a, target_a)]
        teacher_predictions = [] if teacher_a is None else [(teacher_a, target_a)]
        free_predictions = [] if free_a is None else [(free_a, target_a)]
        if pair is not None:
            pair_action = pair[ACTION].float()[:auxiliary_count]
            pair_features = self.obs_encoder(pair)[:auxiliary_count]
            pair_task_ids = self._task_ids(pair)[:auxiliary_count]
            teacher_b, auxiliary_b, free_b, _ = self._decode_training_branch(
                pair_action, pair_features, pair_task_ids, seed + 5_000, compute_metrics=compute_metrics
            )
            predictions.append((auxiliary_b, pair_action))
            if teacher_b is not None:
                teacher_predictions.append((teacher_b, pair_action))
            if free_b is not None:
                free_predictions.append((free_b, pair_action))

        auxiliary_terms = [self._physical_terms(prediction, target) for prediction, target in predictions]
        teacher_terms = [
            self._physical_terms(prediction, target) for prediction, target in teacher_predictions
        ]
        free_terms = [self._physical_terms(prediction, target) for prediction, target in free_predictions]

        def average(name: str, terms: list[dict[str, torch.Tensor]]) -> torch.Tensor:
            return torch.stack([item[name] for item in terms]).mean()

        weighted = (
            self.config.decoded_action_loss_weight * average("recon_loss", auxiliary_terms)
            + self.config.decoded_velocity_loss_weight * average("velocity_loss", auxiliary_terms)
            + self.config.decoded_first_target_loss_weight * average("first_loss", auxiliary_terms)
        )
        overlap_loss = action.new_zeros(())
        seam_loss = action.new_zeros(())
        overlap_mae = action.new_zeros(())
        seam_mae = action.new_zeros(())
        teacher_overlap_loss = action.new_zeros(())
        teacher_overlap_mae = action.new_zeros(())
        teacher_seam_loss = action.new_zeros(())
        teacher_seam_mae = action.new_zeros(())
        free_overlap_loss = action.new_zeros(())
        free_overlap_mae = action.new_zeros(())
        free_seam_loss = action.new_zeros(())
        free_seam_mae = action.new_zeros(())
        if pair is not None:
            raw_a, raw_b = auxiliary_terms[0]["raw"], auxiliary_terms[1]["raw"]
            raw_target_a, raw_target_b = auxiliary_terms[0]["target_raw"], auxiliary_terms[1]["target_raw"]
            continuous = _continuous_indices(
                self.config.action_dim, self.config.continuous_action_indices, action.device
            )
            overlap_error = raw_a[:, 16:20] - raw_b[:, :4]
            overlap_loss, overlap_mae = _physical_loss(
                overlap_error, continuous, self.config.physical_unit_scale
            )
            seam_error = (raw_b[:, 0] - raw_a[:, 15]) - (raw_target_b[:, 0] - raw_target_a[:, 15])
            seam_loss, seam_mae = _physical_loss(seam_error, continuous, self.config.physical_unit_scale)

            def boundary(terms: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
                first, second = terms[0]["raw"], terms[1]["raw"]
                overlap, overlap_metric = _physical_loss(
                    first[:, 16:20] - second[:, :4], continuous, self.config.physical_unit_scale
                )
                seam, seam_metric = _physical_loss(
                    (second[:, 0] - first[:, 15]) - (raw_target_b[:, 0] - raw_target_a[:, 15]),
                    continuous,
                    self.config.physical_unit_scale,
                )
                return overlap, overlap_metric, seam, seam_metric

            if teacher_terms:
                teacher_overlap_loss, teacher_overlap_mae, teacher_seam_loss, teacher_seam_mae = boundary(
                    teacher_terms
                )
            if free_terms:
                free_overlap_loss, free_overlap_mae, free_seam_loss, free_seam_mae = boundary(free_terms)
            weighted = weighted + (
                self.config.decoded_overlap_loss_weight * overlap_loss
                + self.config.decoded_seam_loss_weight * seam_loss
            )
        loss = loss + weighted

        def log_terms(prefix: str, terms: list[dict[str, torch.Tensor]]) -> None:
            metrics[f"{prefix}_decoded_reconstruction_loss"] = float(average("recon_loss", terms).detach())
            metrics[f"{prefix}_decoded_reconstruction_mae"] = float(average("recon_mae", terms).detach())
            metrics[f"{prefix}_decoded_velocity_loss"] = float(average("velocity_loss", terms).detach())
            metrics[f"{prefix}_decoded_velocity_mae"] = float(average("velocity_mae", terms).detach())
            metrics[f"{prefix}_decoded_first_target_loss"] = float(average("first_loss", terms).detach())
            metrics[f"{prefix}_decoded_first_target_mae"] = float(average("first_mae", terms).detach())

        if teacher_terms:
            log_terms("teacher_forced", teacher_terms)
            metrics["teacher_forced_decoded_overlap_loss"] = float(teacher_overlap_loss.detach())
            metrics["teacher_forced_decoded_overlap_mae"] = float(teacher_overlap_mae.detach())
            metrics["teacher_forced_decoded_seam_loss"] = float(teacher_seam_loss.detach())
            metrics["teacher_forced_decoded_seam_mae"] = float(teacher_seam_mae.detach())
        if free_terms:
            log_terms("free_running", free_terms)
            metrics["free_running_decoded_overlap_loss"] = float(free_overlap_loss.detach())
            metrics["free_running_decoded_overlap_mae"] = float(free_overlap_mae.detach())
            metrics["free_running_decoded_seam_loss"] = float(free_seam_loss.detach())
            metrics["free_running_decoded_seam_mae"] = float(free_seam_mae.detach())
        if mode == "scheduled_sampling":
            log_terms("scheduled_sampling", auxiliary_terms)
            metrics["scheduled_sampling_decoded_overlap_loss"] = float(overlap_loss.detach())
            metrics["scheduled_sampling_decoded_overlap_mae"] = float(overlap_mae.detach())
            metrics["scheduled_sampling_decoded_seam_loss"] = float(seam_loss.detach())
            metrics["scheduled_sampling_decoded_seam_mae"] = float(seam_mae.detach())
        metrics["total_loss"] = float(loss.detach())
        if (not self.training) and self.config.num_tasks >= 2:
            swapped_ids = (task_ids + 1) % self.config.num_tasks
            swapped_logits = self.model(input_ids[:, :-1], features, swapped_ids)
            swapped_loss = F.cross_entropy(
                swapped_logits.reshape(-1, swapped_logits.shape[-1]), target.reshape(-1)
            )
            metrics["task_token_swap_ce_gap"] = float((swapped_loss - token_ce).item())
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
