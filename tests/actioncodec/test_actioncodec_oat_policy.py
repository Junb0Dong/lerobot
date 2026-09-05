from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lerobot.actioncodec.models.diffusion_decoder import ActionDiffusionDecoder
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.actioncodec.configuration_actioncodec import ActionCodecConfig
from lerobot.policies.actioncodec.modeling_actioncodec import (
    ActionCodecPolicy,
    OATExactCached,
    _codebook_distance_loss,
    _physical_loss,
    _relaxed_one_hot,
)
from lerobot.policies.actioncodec.obs_encoder import ObservationEncoder
from lerobot.scripts.lerobot_train import _ActionCodecPairedWindowDataset
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


def _image_batch(
    *keys: str,
    batch: int = 2,
    steps: int = 2,
    size: int | tuple[int, int] = 8,
) -> dict[str, torch.Tensor]:
    image_height, image_width = (size, size) if isinstance(size, int) else size
    data = {
        "action": torch.randn(batch, 20, 7),
        "observation.state": torch.randn(batch, steps, 7),
        "task_uid": torch.zeros(batch, dtype=torch.long),
    }
    for key in keys:
        data[key] = torch.rand(batch, steps, 3, image_height, image_width)
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


def test_oat_exact_robomimic_rejects_noop_crop_shape():
    with pytest.raises(ValueError, match="crop_shape=None to disable cropping"):
        _tiny_config(
            vision_encoder="oat_exact_robomimic",
            image_size=32,
            crop_shape=(32, 32),
        )


def test_actioncodec_accepts_rectangular_image_size_and_validates_crop_per_axis():
    config = _tiny_config(image_size=(24, 32), crop_shape=None)
    assert config.image_size == (24, 32)
    assert config.image_shape == (24, 32)

    with pytest.raises(ValueError, match="must fit inside"):
        _tiny_config(
            vision_encoder="resnet_spatial",
            image_size=(24, 32),
            crop_shape=(25, 16),
        )


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


def test_policy_auxiliary_config_defaults_are_compatible():
    config = _tiny_config()
    assert config.codebook_distance_loss_weight == 0.0
    assert config.decoded_action_loss_weight == 0.0
    assert config.decoded_velocity_loss_weight == 0.0
    assert config.decoded_first_target_loss_weight == 0.0
    assert config.decoded_overlap_loss_weight == 0.0
    assert config.decoded_seam_loss_weight == 0.0
    assert config.overlap_shift == 16
    assert config.decoded_metrics_interval == 1
    with pytest.raises(ValueError, match="decoded_metrics_interval"):
        _tiny_config(decoded_metrics_interval=0)


def test_policy_auxiliary_config_round_trips_and_migrates_missing_fields(tmp_path):
    config = _tiny_config(
        codebook_distance_loss_weight=0.5,
        decoded_action_loss_weight=0.5,
        decoded_velocity_loss_weight=0.1,
        decoded_first_target_loss_weight=0.25,
        decoded_overlap_loss_weight=0.2,
        decoded_seam_loss_weight=0.3,
        continuous_action_indices=(0, 1, 2),
        token_relaxation_temperature=0.7,
        prefix_corruption_prob=0.5,
        decoded_metrics_interval=100,
    )
    config._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.codebook_distance_loss_weight == pytest.approx(0.5)
    assert loaded.decoded_action_loss_weight == pytest.approx(0.5)
    assert loaded.continuous_action_indices == (0, 1, 2)
    assert loaded.prefix_corruption_prob == pytest.approx(0.5)
    assert loaded.decoded_metrics_interval == 100

    config_json = tmp_path / "config.json"
    payload = json.loads(config_json.read_text())
    for key in (
        "codebook_distance_loss_weight",
        "decoded_action_loss_weight",
        "decoded_velocity_loss_weight",
        "decoded_first_target_loss_weight",
        "decoded_overlap_loss_weight",
        "decoded_seam_loss_weight",
        "continuous_action_indices",
        "physical_unit_scale",
        "token_relaxation",
        "token_relaxation_temperature",
        "auxiliary_batch_fraction",
        "decoded_metrics_interval",
        "prefix_corruption_prob",
        "auxiliary_seed",
        "overlap_shift",
    ):
        payload.pop(key, None)
    config_json.write_text(json.dumps(payload))
    migrated = PreTrainedConfig.from_pretrained(tmp_path)
    assert migrated.codebook_distance_loss_weight == 0.0
    assert migrated.decoded_action_loss_weight == 0.0
    assert migrated.overlap_shift == 16
    assert migrated.decoded_metrics_interval == 1


@pytest.mark.parametrize("amp", [False, True])
def test_codebook_distance_matches_expected_risk_and_freezes_geometry(amp):
    torch.manual_seed(42)
    codebook = torch.randn(5, 3, requires_grad=True)
    logits = torch.randn(2, 4, 6, requires_grad=True)  # Last category is BOS.
    target = torch.randint(5, (2, 4))
    distances = (codebook.detach()[target, None] - codebook.detach()).square().sum(-1)
    scale = (codebook.detach()[:, None] - codebook.detach()[None]).square().sum(-1).mean()
    expected = (logits[..., :5].softmax(-1) * distances / scale).sum(-1).mean()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = _codebook_distance_loss(logits, target, codebook)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad[..., :5].abs().sum() > 0
    assert logits.grad[..., 5].count_nonzero() == 0
    assert codebook.grad is None
    torch.testing.assert_close(actual, _codebook_distance_loss(logits, target, codebook * 7 + 3))


def test_codebook_distance_penalizes_far_errors_without_cancellation():
    codebook = torch.tensor([[0.0], [1.0], [4.0], [-4.0]])
    target = torch.tensor([[0]])
    near = torch.tensor([[[0.0, 3.0, -20.0, -20.0]]])
    far = torch.tensor([[[0.0, -20.0, 3.0, -20.0]]])
    assert _codebook_distance_loss(near, target, codebook) < _codebook_distance_loss(far, target, codebook)
    cancelling = torch.tensor([[[-torch.inf, -torch.inf, 0.0, 0.0]]])
    assert _codebook_distance_loss(cancelling, target, codebook) > 0
    correct = torch.tensor([[[0.0, -torch.inf, -torch.inf, -torch.inf]]])
    assert _codebook_distance_loss(correct, target, codebook) == 0


def test_policy_codebook_distance_preserves_ce_and_backpropagates_without_decode(monkeypatch):
    policy = ActionCodecPolicy(_tiny_config(dropout=0.0))
    batch = _image_batch("observation.images.front")
    policy.eval()
    baseline, baseline_metrics = policy(batch)
    state_keys = set(policy.state_dict())

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Codebook distance must not invoke the action decoder")

    monkeypatch.setattr(policy.tokenizer, "decode_train", unexpected_decode)
    policy.config.codebook_distance_loss_weight = 0.5
    loss, metrics = policy(batch)
    assert metrics["token_ce"] == baseline_metrics["token_ce"]
    assert metrics["task_token_swap_ce_gap"] == baseline_metrics["task_token_swap_ce_gap"]
    assert loss.item() == pytest.approx(baseline.item() + 0.5 * metrics["codebook_distance_loss"])
    assert metrics["total_loss"] == loss.item()
    assert "codebook_distance_loss" not in baseline_metrics
    assert set(policy.state_dict()) == state_keys
    (loss - baseline).backward()  # Isolate distance gradients, excluding CE.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.obs_encoder.parameters())
    assert all(p.grad is None for p in policy.tokenizer.parameters())
    policy.config.codebook_distance_loss_weight = 0.0
    restored, restored_metrics = policy(batch)
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
    assert restored_metrics == baseline_metrics


def test_st_token_assignment_is_hard_and_has_logit_gradient():
    logits = torch.randn(3, 5, requires_grad=True)
    generator = torch.Generator().manual_seed(42)
    probabilities = _relaxed_one_hot(logits, 0.7, generator=generator)
    assert torch.allclose(probabilities.detach().sum(-1), torch.ones(3))
    assert torch.all(probabilities.detach().count_nonzero(-1) == 1)
    probabilities[..., 0].sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_policy_physical_auxiliary_backward_reaches_policy_only():
    policy = ActionCodecPolicy(
        _tiny_config(
            decoded_action_loss_weight=0.5,
            decoded_velocity_loss_weight=0.1,
            decoded_first_target_loss_weight=0.5,
            prefix_corruption_prob=1.0,
        ),
        dataset_stats={"action": {"mean": torch.zeros(7), "std": torch.arange(1, 8, dtype=torch.float32)}},
    ).train()
    policy.tokenizer.model.decoder = ActionDiffusionDecoder(
        action_dim=7,
        model_dim=256,
        window_size=20,
        num_heads=8,
        latent_depth=1,
        num_train_steps=10,
        num_sample_steps=2,
        denoiser_layers=1,
        kernel_size=3,
    )
    policy.tokenizer.model.decoder_type = "diffusion"
    policy.tokenizer.model.decoder.eval()
    for parameter in policy.tokenizer.model.decoder.parameters():
        parameter.requires_grad_(False)
    decoded_latents = []

    def capture_latents(module, args):
        if args[0].requires_grad:
            args[0].retain_grad()
            decoded_latents.append(args[0])

    handle = policy.tokenizer.model.decoder.conditioner.register_forward_pre_hook(capture_latents)
    loss, metrics = policy(_image_batch("observation.images.front"))
    loss.backward()
    handle.remove()
    assert decoded_latents
    assert all(
        latents.grad is not None and latents.grad.isfinite().all() and latents.grad.abs().sum() > 0
        for latents in decoded_latents
    )
    assert metrics["free_running_decoded_reconstruction_mae"] >= 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in policy.model.parameters()
        if parameter.requires_grad
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in policy.obs_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in policy.tokenizer.parameters())


def test_differentiable_generation_probability_boundaries_are_seeded():
    model = OATExactCached(_tiny_config(dropout=0), cond_dim=24).eval()
    bos = model.token_emb(torch.full((2, 1), 1024, dtype=torch.long))
    condition = torch.randn(2, 2, 24)
    task_ids = torch.tensor([0, 1])
    prefix = torch.randint(0, 1024, (2, 16))
    codebook = torch.randn(1024, 32)
    args = (bos, condition, 16, task_ids, codebook, 0.7, prefix)
    teacher_prefix = model.generate_differentiable(*args, 0.0, 17, False)
    teacher_prefix_again = model.generate_differentiable(*args, 0.0, 17, False)
    free_prefix = model.generate_differentiable(*args, 1.0, 17, False)
    sampled = model.generate_differentiable(*args, 0.5, 17, True)
    sampled_again = model.generate_differentiable(*args, 0.5, 17, True)
    torch.testing.assert_close(teacher_prefix, teacher_prefix_again)
    torch.testing.assert_close(sampled, sampled_again)
    assert teacher_prefix.shape == free_prefix.shape == sampled.shape == (2, 16, 32)


def test_actioncodec_paired_windows_stay_inside_selected_episodes():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(episodes={"dataset_from_index": [0, 100], "dataset_to_index": [100, 200]}),
        episodes=[1],
        absolute_to_relative_idx=None,
    )
    paired = _ActionCodecPairedWindowDataset(dataset)
    assert paired.pairs
    assert all(first >= 100 and second - first == 16 for first, second in paired.pairs)
    assert all(second + 20 <= 200 for _, second in paired.pairs)


def test_paired_overlap_and_seam_match_gt_boundary():
    config = _tiny_config(
        decoded_action_loss_weight=1.0,
        decoded_overlap_loss_weight=1.0,
        decoded_seam_loss_weight=1.0,
        prefix_corruption_prob=1.0,
    )
    policy = ActionCodecPolicy(
        config,
        dataset_stats={"action": {"mean": torch.zeros(7), "std": torch.ones(7)}},
    ).train()
    anchor = _image_batch("observation.images.front", batch=1)
    pair = _image_batch("observation.images.front", batch=1)
    anchor["action"].zero_()
    pair["action"].zero_()
    batch = {**anchor, "_actioncodec_pair": pair}

    policy._decode_training_branch = lambda action, features, task_ids, seed, **kwargs: (
        action,
        action,
        action,
        "free_running",
    )
    _, aligned = policy(batch)
    assert aligned["free_running_decoded_overlap_loss"] == 0
    assert aligned["free_running_decoded_seam_loss"] == 0

    calls = 0

    def wrong_boundary(action, features, task_ids, seed, **kwargs):
        nonlocal calls
        calls += 1
        prediction = action.clone()
        if calls == 2:
            prediction[:, :4] += 1
        return prediction, prediction, prediction, "free_running"

    policy._decode_training_branch = wrong_boundary
    _, wrong = policy(batch)
    assert wrong["free_running_decoded_overlap_loss"] > 0
    assert wrong["free_running_decoded_seam_loss"] > 0


@pytest.mark.parametrize("prefix_probability", [0.0, 0.5, 1.0])
def test_decoded_metrics_frequency_preserves_paired_loss_gradients_and_rng(prefix_probability, monkeypatch):
    policy = ActionCodecPolicy(
        _tiny_config(
            image_size=8,
            crop_shape=None,
            decoded_action_loss_weight=0.5,
            decoded_velocity_loss_weight=0.1,
            decoded_first_target_loss_weight=0.5,
            decoded_overlap_loss_weight=0.2,
            decoded_seam_loss_weight=0.2,
            prefix_corruption_prob=prefix_probability,
        ),
        dataset_stats={"action": {"mean": torch.zeros(7), "std": torch.ones(7)}},
    ).train()
    batch = _image_batch("observation.images.front", batch=1)
    batch["_actioncodec_pair"] = _image_batch("observation.images.front", batch=1)
    decode = policy.tokenizer.decode_train
    calls = []

    def counted_decode(latents):
        calls.append(torch.is_grad_enabled())
        return decode(latents)

    monkeypatch.setattr(policy.tokenizer, "decode_train", counted_decode)
    results = []
    for interval in (1, 100):
        policy.config.decoded_metrics_interval = interval
        policy._auxiliary_call_index = 0
        policy._auxiliary_train_call_index = 0
        policy.zero_grad(set_to_none=True)
        calls.clear()
        torch.manual_seed(71)
        loss, metrics = policy(batch)
        loss.backward()
        results.append(
            (
                loss.detach(),
                {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None},
                torch.get_rng_state(),
            )
        )
        # Loss-bearing A/B decodes always run, while extra diagnostics can be skipped.
        assert sum(calls) == 2
        assert len(calls) == (2 if interval == 100 else (6 if prefix_probability == 0.5 else 4))
        assert ("teacher_forced_decoded_reconstruction_mae" in metrics) == (
            interval == 1 or prefix_probability == 0
        )
        assert ("free_running_decoded_reconstruction_mae" in metrics) == (
            interval == 1 or prefix_probability == 1
        )
        assert all(p.grad is None for p in policy.tokenizer.parameters())
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    # Eval must emit all diagnostics without advancing the training cadence.
    policy.eval()
    with torch.no_grad():
        _, metrics = policy(batch)
    assert "teacher_forced_decoded_reconstruction_mae" in metrics
    assert "free_running_decoded_reconstruction_mae" in metrics
    assert policy._auxiliary_train_call_index == 1

    policy.train()
    policy._auxiliary_train_call_index = 99
    _, metrics = policy(batch)
    assert "teacher_forced_decoded_reconstruction_mae" in metrics
    assert "free_running_decoded_reconstruction_mae" in metrics


def test_physical_terms_use_action_std_and_continuous_indices():
    policy = ActionCodecPolicy(
        _tiny_config(continuous_action_indices=(0, 1)),
        dataset_stats={"action": {"mean": torch.zeros(7), "std": torch.arange(1, 8, dtype=torch.float32)}},
    )
    target = torch.zeros(1, 20, 7)
    offset = torch.ones_like(target)
    terms = policy._physical_terms(offset, target)
    assert terms["recon_mae"] == pytest.approx(torch.arange(1, 8, dtype=torch.float32).mean().item())
    constant_velocity = policy._physical_terms(offset + 5.0, target)["velocity_loss"]
    assert constant_velocity == 0
    slope = target.clone()
    slope[:, :, 0] = torch.arange(20, dtype=torch.float32)
    slope[:, :, 2] = torch.arange(20, dtype=torch.float32) * 10
    assert policy._physical_terms(slope, target)["velocity_loss"] > 0
    excluded = _physical_loss(
        torch.cat((torch.ones(20, 1), torch.zeros(20, 6)), dim=-1),
        torch.tensor([1, 2]),
        1.0,
    )[1]
    assert excluded == 0


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
def test_oat_exact_robomimic_encoder_can_use_full_resized_image_without_crop():
    config = _tiny_config(
        vision_encoder="oat_exact_robomimic",
        image_size=(24, 32),
        crop_shape=None,
        input_features={
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 24, 32)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
    )
    encoder = ObservationEncoder(config)
    batch = _image_batch("observation.images.front", size=(24, 32))

    exact = encoder.robomimic_encoder
    assert exact is not None
    assert exact.image_shape == (24, 32)
    assert exact.crop_shape is None
    assert all(randomizer is None for randomizer in exact.encoder.obs_randomizers.values())
    assert not any(isinstance(module, rmbn.CropRandomizer) for module in exact.modules())

    first = encoder(batch)
    second = encoder(batch)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 2, 64 + 7 + 1)
    assert first.isfinite().all()


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
