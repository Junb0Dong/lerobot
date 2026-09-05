import inspect
from dataclasses import asdict

import pytest
import torch

from lerobot.actioncodec.config import ActionCodecTokenizerConfig, save_tokenizer_artifact
from lerobot.actioncodec.models.fsq import FSQGrid, SemanticFSQQuantizer
from lerobot.actioncodec.models.tokenizer import ActionCodecTokenizer
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.actioncodec.configuration_actioncodec import ActionCodecConfig
from lerobot.policies.actioncodec.modeling_actioncodec import ActionCodecPolicy, FSQOATExactCached


def config(**overrides):
    values = {
        "device": "cpu",
        "quantizer_type": "semantic_fsq",
        "codebook_size": 1000,
        "num_tasks": 2,
        "vision_encoder": "small_cnn",
        "embed_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.0,
        "input_features": {
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (7,))},
        "temperature": 0.0,
    }
    values.update(overrides)
    return ActionCodecConfig(**values)


def test_grid_exhaustive_roundtrip_and_even_boundary():
    grid = FSQGrid()
    ids = torch.arange(1000)
    classes = grid.indices_to_scalar_classes(ids)
    coords = grid.indices_to_coordinates(ids)
    assert grid.basis.tolist() == [1, 8, 40, 200]
    assert torch.equal(grid.scalar_classes_to_indices(classes), ids)
    assert torch.equal(grid.coordinates_to_scalar_classes(coords), classes)
    bounds, bound_ids = grid(torch.tensor([[-100.0] * 4, [100.0] * 4]))
    torch.testing.assert_close(bounds, torch.tensor([[-1.0] * 4, [0.75, 1.0, 1.0, 1.0]]))
    assert bound_ids.tolist() == [0, 999]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_amp_indices_and_projection_gradients(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    quantizer = SemanticFSQQuantizer(32).cuda()
    z = torch.randn(4, 16, 32, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        output, ids, loss = quantizer(z)
        objective = output.square().mean() + quantizer.last_quantized_coordinates.square().mean()
    objective.backward()
    assert ids.min() >= 0 and ids.max() < 1000
    assert loss == 0 and quantizer.last_refreshed_codes == 0
    for grad in (z.grad, quantizer.input_projection.weight.grad, quantizer.output_projection.weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_causal_heads_cached_generation_and_sequence_nll():
    torch.manual_seed(42)
    model = FSQOATExactCached(config(), 24).eval()
    cond, tasks = torch.randn(2, 2, 24), torch.tensor([0, 1])
    prefix = torch.cat((torch.full((2, 1), 1000), torch.randint(1000, (2, 15))), 1)
    altered = prefix.clone()
    altered[:, 8:] = (altered[:, 8:] + 301) % 1000
    for a, b in zip(model(prefix, cond, tasks), model(altered, cond, tasks), strict=True):
        torch.testing.assert_close(a[:, :8], b[:, :8])
    full = prefix[:, :1]
    log_probs = []
    for _ in range(16):
        heads = model(full, cond, tasks)
        classes = torch.stack([head[:, -1].argmax(-1) for head in heads], -1)
        log_probs.append(
            torch.stack(
                [
                    head[:, -1].log_softmax(-1).gather(-1, classes[:, i, None]).squeeze(-1)
                    for i, head in enumerate(heads)
                ]
            ).sum(0)
        )
        next_id = model.token_emb.grid.scalar_classes_to_indices(classes)[:, None]
        full = torch.cat((full, next_id), 1)
    assert torch.equal(model.generate(prefix[:, :1], cond, 16, tasks), full)
    torch.testing.assert_close(model.sequence_logprobs(full, cond, tasks), torch.stack(log_probs).mean(0))


def test_artifact_freezing_alignment_gradient_and_policy_reload(tmp_path):
    cfg = ActionCodecTokenizerConfig(
        quantizer_type="semantic_fsq",
        codebook_size=1000,
        model_dim=32,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        encoder_cross_layers=1,
        decoder_cross_layers=1,
        decoder_type="perceiver",
        dropout=0.0,
    )
    kwargs = {k: v for k, v in asdict(cfg).items() if k in inspect.signature(ActionCodecTokenizer).parameters}
    model = ActionCodecTokenizer(**kwargs)
    action = torch.randn(2, 20, 7)
    result = model(
        action,
        delta_state=action,
        loss_config={
            "weight_chunk_align": 0.1,
            "chunk_align_max_candidate_pairs": 2,
            "chunk_align_positive_topk": 1,
        },
    )
    result["loss"].backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert model.quantizer.last_quantized_coordinates.shape == (2, 16, 4)
    assert model.tokenize(action).shape == (2, 16)
    assert model.detokenize(model.tokenize(action)).shape == (2, 20, 7)
    artifact = tmp_path / "tokenizer"
    save_tokenizer_artifact(
        artifact,
        model,
        cfg,
        action_stats={"mean": [0.0] * 7, "std": [1.0] * 7},
        dataset_contract={"horizon": 20, "latent_horizon": 16, "action_dim": 7},
    )
    policy = ActionCodecPolicy(config(tokenizer_path=artifact))
    batch = {
        "action": action,
        "observation.state": torch.randn(2, 2, 7),
        "observation.images.front": torch.rand(2, 2, 3, 8, 8),
        "task_uid": torch.tensor([0, 1]),
    }
    policy.train()
    loss, metrics = policy(batch)
    loss.backward()
    assert metrics["token_nll"] == pytest.approx(metrics["token_ce"] * 4)
    assert not policy.tokenizer.training
    assert all(not p.requires_grad and p.grad is None for p in policy.tokenizer.parameters())
    assert all(head.weight.grad.abs().sum() > 0 for head in policy.model.head.heads)
    prediction = policy.predict_action_chunk(batch)
    policy.save_pretrained(tmp_path / "policy")
    restored = ActionCodecPolicy.from_pretrained(tmp_path / "policy")
    torch.testing.assert_close(prediction, restored.predict_action_chunk(batch), rtol=0, atol=0)
    with pytest.raises(ValueError, match="quantizer_type"):
        ActionCodecPolicy(config(quantizer_type="vq", codebook_size=1024, tokenizer_path=artifact))


@pytest.mark.parametrize(
    "weight",
    [
        "codebook_distance_loss_weight",
        "decoded_action_loss_weight",
        "decoded_velocity_loss_weight",
        "decoded_first_target_loss_weight",
        "decoded_overlap_loss_weight",
        "decoded_seam_loss_weight",
    ],
)
def test_fsq_rejects_auxiliary(weight):
    with pytest.raises(ValueError, match="scalar CE only"):
        config(**{weight: 0.1})


def test_fsq_alignment_uses_quantized_coordinates(monkeypatch):
    from types import SimpleNamespace

    import torch.nn.functional as functional

    from lerobot.actioncodec.models import tokenizer as tokenizer_module

    captured = []

    def alignment(embeddings, **kwargs):
        captured.append(embeddings)
        return SimpleNamespace(total=(embeddings[0] - embeddings[1]).square().sum())

    monkeypatch.setattr(tokenizer_module, "semantic_contrastive_loss", alignment)
    model = ActionCodecTokenizer(
        quantizer_type="semantic_fsq",
        codebook_size=1000,
        model_dim=32,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        encoder_cross_layers=1,
        decoder_cross_layers=1,
        decoder_type="perceiver",
        dropout=0.0,
    )
    actions = torch.randn(2, 20, 7)
    result = model(
        actions,
        delta_state=actions,
        loss_config={"weight_recon": 0.0, "weight_chunk_align": 0.1, "chunk_align_positive_topk": 1},
    )
    expected = functional.normalize(model.quantizer.last_quantized_coordinates.mean(1), dim=-1)
    torch.testing.assert_close(captured[0], expected)
    result["loss"].backward()
    assert model.quantizer.input_projection.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert model.quantizer.output_projection.weight.grad.abs().sum() == 0


def test_default_sampling_and_artifact_contract():
    assert config(temperature=None).temperature == 0
    assert config(quantizer_type="vq", codebook_size=1024, temperature=None).temperature == 1
    assert config(temperature=0.7).temperature == 0.7
    for overrides in ({"codebook_size": 1024}, {"num_codebooks": 2}, {"fsq_levels": (5, 5, 5, 8)}):
        with pytest.raises(ValueError):
            config(**overrides)
