from __future__ import annotations

import math

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.actioncodec.configuration_actioncodec import ActionCodecConfig
from lerobot.policies.actioncodec.modeling_actioncodec import ActionCodecPolicy, OATExactCached
from lerobot.policies.actioncodec.obs_encoder import ObservationEncoder
from lerobot.utils.import_utils import _robomimic_available


def _tiny_config(**overrides) -> ActionCodecConfig:
    kwargs = {
        "num_tasks": 2,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "vision_encoder": "small_cnn",
        "input_features": {
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (7,))},
        "embed_dim": 32,
        "n_heads": 4,
        "n_layers": 1,
    }
    kwargs.update(overrides)
    return ActionCodecConfig(**kwargs)


def _image_batch(*keys: str, batch: int = 2, steps: int = 2, size: int = 8) -> dict[str, torch.Tensor]:
    data = {
        "action": torch.randn(batch, 20, 7),
        "observation.state": torch.randn(batch, steps, 7),
        "task_uid": torch.zeros(batch, dtype=torch.long),
    }
    for key in keys:
        data[key] = torch.rand(batch, steps, 3, size, size)
    return data


def test_oat_xavier_tied_head_and_task_token_seed():
    torch.manual_seed(0)
    first = OATExactCached(_tiny_config(task_token_init_seed=42), cond_dim=24)
    torch.manual_seed(0)
    second = OATExactCached(_tiny_config(task_token_init_seed=42), cond_dim=24)
    torch.manual_seed(0)
    third = OATExactCached(_tiny_config(task_token_init_seed=7), cond_dim=24)
    assert first.head.weight.data_ptr() == first.token_emb.weight.data_ptr()
    torch.testing.assert_close(first.task_embedding.weight, second.task_embedding.weight)
    torch.testing.assert_close(first.token_emb.weight, second.token_emb.weight)
    assert not torch.equal(first.task_embedding.weight, third.task_embedding.weight)
    torch.testing.assert_close(first.token_emb.weight, third.token_emb.weight)


def test_resnet_spatial_condition_dim_crop_seed_and_independent_cameras():
    config = _tiny_config(
        vision_encoder="resnet_spatial",
        image_size=32,
        crop_shape=(16, 16),
        input_features={
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
    )
    encoder = ObservationEncoder(config)
    batch = _image_batch("observation.images.front", "observation.images.wrist", size=32)
    torch.manual_seed(0)
    first = encoder(batch)
    torch.manual_seed(0)
    second = encoder(batch)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 2, 2 * 64 + 7 + 1)
    weights_match = all(
        torch.equal(left, right)
        for left, right in zip(
            encoder.image_encoders[0].parameters(), encoder.image_encoders[1].parameters(), strict=True
        )
    )
    assert not weights_match


def test_optimizer_has_four_oat_param_groups():
    policy = ActionCodecPolicy(_tiny_config())
    groups = {group["name"]: group for group in policy.get_optim_params()}
    assert set(groups) == {"policy_decay", "policy_nodecay", "obs_decay", "obs_nodecay"}
    assert groups["policy_decay"]["lr"] == pytest.approx(policy.config.optimizer_lr)
    assert groups["obs_decay"]["lr"] == pytest.approx(policy.config.optimizer_lr_obs_encoder)
    assert groups["policy_decay"]["weight_decay"] == pytest.approx(policy.config.optimizer_weight_decay)
    assert groups["policy_nodecay"]["weight_decay"] == 0.0
    assert groups["obs_nodecay"]["weight_decay"] == 0.0
    assert all(param.ndim >= 2 for param in groups["policy_decay"]["params"])
    assert all(param.ndim < 2 for param in groups["policy_nodecay"]["params"])
    assert all(param.ndim >= 2 for param in groups["obs_decay"]["params"])
    assert all(param.ndim < 2 for param in groups["obs_nodecay"]["params"])
    tokenizer_ids = {id(param) for param in policy.tokenizer.parameters()}
    grouped_ids = {id(param) for group in groups.values() for param in group["params"]}
    assert tokenizer_ids.isdisjoint(grouped_ids)
    trainable_ids = {id(param) for param in policy.parameters() if param.requires_grad}
    assert grouped_ids == trainable_ids


def test_generation_never_emits_bos_at_zero_or_positive_temperature():
    oat = OATExactCached(_tiny_config(), cond_dim=24).eval()
    condition = torch.randn(4, 2, 24)
    task_ids = torch.tensor([0, 1, 0, 1])
    bos = torch.full((4, 1), 1024, dtype=torch.long)
    greedy = oat.generate(bos, condition, 16, task_ids, temperature=0.0)
    torch.manual_seed(0)
    sampled = oat.generate(bos, condition, 16, task_ids, temperature=1.0, top_k=10)
    for generated in (greedy, sampled):
        assert generated.shape == (4, 17)
        assert int(generated[:, 1:].min()) >= 0
        assert int(generated[:, 1:].max()) < 1024


def test_eval_reports_finite_task_token_swap_ce_gap():
    policy = ActionCodecPolicy(_tiny_config(num_tasks=3))
    batch = _image_batch("observation.images.front")
    batch["task_uid"] = torch.tensor([0, 2])
    policy.train()
    _, train_logs = policy(batch)
    assert "task_token_swap_ce_gap" not in train_logs
    policy.eval()
    loss, eval_logs = policy(batch)
    assert loss.isfinite()
    assert math.isfinite(eval_logs["task_token_swap_ce_gap"])
    assert "token_top5_acc" in eval_logs


@pytest.mark.skipif(not _robomimic_available, reason="robomimic is optional")
def test_oat_exact_robomimic_encoder_condition_dim():
    config = _tiny_config(
        vision_encoder="oat_exact_robomimic",
        image_size=32,
        crop_shape=(16, 16),
        input_features={
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
    )
    encoder = ObservationEncoder(config)
    batch = _image_batch("observation.images.front", "observation.images.wrist", size=32)
    features = encoder(batch)
    assert features.shape == (2, 2, 2 * 64 + 7 + 1)
    assert features.isfinite().all()


@pytest.mark.skipif(_robomimic_available, reason="only checks the missing-package path")
def test_oat_exact_robomimic_requires_package():
    config = _tiny_config(
        vision_encoder="oat_exact_robomimic",
        image_size=32,
        crop_shape=(16, 16),
    )
    with pytest.raises(ImportError, match="robomimic"):
        ObservationEncoder(config)
