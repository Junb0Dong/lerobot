from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.actioncodec.configuration_actioncodec import ActionCodecConfig
from lerobot.policies.actioncodec.modeling_actioncodec import ActionCodecPolicy, OATExactCached
from lerobot.policies.actioncodec.obs_encoder import ObservationEncoder
from lerobot.utils import import_utils
from lerobot.utils.import_utils import _robomimic_available

if _robomimic_available:
    import robomimic.models.base_nets as rmbn
    import robomimic.models.obs_nets as rmon
    import robomimic.utils.obs_utils as obs_utils


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


def test_actioncodec_defaults_to_oat_exact_robomimic():
    kwargs = {
        "num_tasks": 2,
        "input_features": {
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (7,))},
    }
    config = ActionCodecConfig(**kwargs)
    assert config.vision_encoder == "oat_exact_robomimic"
    assert config.image_size == 128
    assert config.crop_shape == (76, 76)


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
def test_oat_exact_robomimic_encoder_matches_oat_dataflow_and_parameter_count():
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
    exact = encoder.robomimic_encoder
    assert exact is not None
    assert isinstance(exact.encoder.activation, nn.ReLU)

    camera_nets = list(exact.encoder.obs_nets.values())
    assert len(camera_nets) == 2
    assert [sum(parameter.numel() for parameter in net.parameters()) for net in camera_nets] == [
        11_197_088,
        11_197_088,
    ]
    assert next(camera_nets[0].parameters()).data_ptr() != next(camera_nets[1].parameters()).data_ptr()
    assert not any(isinstance(module, nn.BatchNorm2d) for module in exact.modules())
    assert any(isinstance(module, nn.GroupNorm) for module in exact.modules())

    # Build the OAT reference directly from robomimic instead of going through the LeRobot wrapper.
    reference_keys = ["front", "wrist"]
    obs_utils.initialize_obs_modality_mapping_from_dict({"rgb": reference_keys})
    reference = rmon.ObservationEncoder()
    for key, source_net in zip(reference_keys, camera_nets, strict=True):
        net = rmbn.VisualCore(
            input_shape=(3, 16, 16),
            feature_dimension=64,
            backbone_class="ResNet18Conv",
            backbone_kwargs={"input_channels": 3, "input_coord_conv": False},
            pool_class="SpatialSoftmax",
            pool_kwargs={"num_kp": 32, "temperature": 1.0, "noise": 0.0},
            flatten=True,
        )
        for name, module in list(net.named_modules()):
            if not isinstance(module, nn.BatchNorm2d):
                continue
            parent_name, _, child_name = name.rpartition(".")
            parent = net.get_submodule(parent_name) if parent_name else net
            setattr(parent, child_name, nn.GroupNorm(module.num_features // 16, module.num_features))
        net.load_state_dict(source_net.state_dict(), strict=True)
        reference.register_obs_key(
            name=key,
            shape=(3, 32, 32),
            net=net,
            randomizer=rmbn.CropRandomizer(
                input_shape=(3, 32, 32),
                crop_height=16,
                crop_width=16,
                num_crops=1,
                pos_enc=False,
            ),
        )
    reference.make()
    assert isinstance(reference.activation, nn.ReLU)

    for training in (True, False):
        encoder.train(training)
        torch.manual_seed(123)
        features = encoder(batch)
        torch.manual_seed(123)
        repeated = encoder(batch)
        torch.testing.assert_close(features, repeated)
        assert features.shape == (2, 2, 2 * 64 + 7 + 1)
        assert features.isfinite().all()
        assert (features[..., : 2 * 64] >= 0).all()

    byte_images = torch.zeros(1, 2, 3, 32, 32, dtype=torch.uint8)
    byte_images[:, :, 1] = 255
    normalized = encoder._prepare_camera(byte_images, crop=False)
    assert normalized[:, :, 0].eq(-1).all()
    assert normalized[:, :, 1].eq(1).all()

    prepared = {key: encoder._prepare_camera(batch[key], crop=False) for key in exact.image_keys}
    aliased = {
        internal_key: prepared[public_key].reshape(4, 3, 32, 32)
        for public_key, internal_key in zip(exact.image_keys, exact.internal_keys, strict=True)
    }
    reference_inputs = {
        reference_key: prepared[public_key].reshape(4, 3, 32, 32)
        for public_key, reference_key in zip(exact.image_keys, reference_keys, strict=True)
    }
    for internal_key, reference_key in zip(exact.internal_keys, reference_keys, strict=True):
        torch.manual_seed(222)
        wrapped_crop = exact.encoder.obs_randomizers[internal_key].forward_in(aliased[internal_key])
        torch.manual_seed(222)
        reference_crop = reference.obs_randomizers[reference_key].forward_in(reference_inputs[reference_key])
        torch.testing.assert_close(wrapped_crop, reference_crop)

    torch.manual_seed(321)
    reference_features = reference(reference_inputs).reshape(2, 2, 2 * 64)
    torch.manual_seed(321)
    wrapped = torch.cat(exact.encode_dict(prepared), dim=-1)
    torch.testing.assert_close(wrapped, reference_features)

    round_tripped = ObservationEncoder(config)
    round_tripped.load_state_dict(encoder.state_dict(), strict=True)
    round_tripped.eval()
    encoder.eval()
    torch.manual_seed(456)
    expected = encoder(batch)
    torch.manual_seed(456)
    actual = round_tripped(batch)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not _robomimic_available, reason="robomimic is optional")
def test_oat_exact_robomimic_policy_forward():
    config = _tiny_config(
        vision_encoder="oat_exact_robomimic",
        image_size=32,
        crop_shape=(16, 16),
        input_features={
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
    )
    policy = ActionCodecPolicy(config)
    policy.tokenizer.tokenize = lambda action: torch.zeros(
        action.shape[0], config.latent_horizon, dtype=torch.long, device=action.device
    )
    loss, metrics = policy(_image_batch("observation.images.front", batch=1, size=32))
    assert loss.isfinite()
    assert math.isfinite(metrics["token_ce"])


def test_oat_exact_robomimic_requires_actioncodec_extra(monkeypatch):
    config = _tiny_config(
        vision_encoder="oat_exact_robomimic",
        image_size=32,
        crop_shape=(16, 16),
    )
    monkeypatch.setitem(import_utils._require_package_cache, "robomimic", False)
    with pytest.raises(ImportError, match=r"lerobot\[actioncodec\]"):
        ObservationEncoder(config)
