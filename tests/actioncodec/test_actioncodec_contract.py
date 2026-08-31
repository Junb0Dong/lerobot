from unittest.mock import Mock

import pytest
import torch

from lerobot.actioncodec import ActionCodecTokenizer
from lerobot.actioncodec.config import ActionCodecTokenizerConfig
from lerobot.actioncodec.losses.semantic_dtw import chunk_hard_dtw_targets, semantic_contrastive_loss
from lerobot.actioncodec.losses.soft_dtw import trajectory_soft_dtw_alignments
from lerobot.actioncodec.metrics import CodebookOccupancyMeter
from lerobot.actioncodec.models.perceiver import PositionalEmbedding
from lerobot.actioncodec.models.quantizer import ResidualVectorQuantizer
from lerobot.actioncodec.trainer import SemanticTokenizerTrainConfig, _window_start_indices
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.actioncodec.configuration_actioncodec import ActionCodecConfig
from lerobot.policies.actioncodec.modeling_actioncodec import ActionCodecPolicy, OATExactCached
from lerobot.policies.factory import get_policy_class
from lerobot.processor.pipeline import ProcessorStepRegistry


def _tiny_policy_kwargs(**overrides):
    kwargs = {
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
    return kwargs


def test_tokenizer_contract_shapes_and_single_codebook():
    config = ActionCodecTokenizerConfig(
        model_dim=32, num_heads=4, encoder_layers=1, decoder_layers=1, decoder_type="perceiver"
    )
    config.validate()
    tokenizer = ActionCodecTokenizer(
        action_dim=config.action_dim,
        window_size=config.horizon,
        model_dim=config.model_dim,
        num_tokens=config.latent_horizon,
        codebook_size=config.codebook_size,
        num_heads=config.num_heads,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        encoder_cross_layers=1,
        decoder_cross_layers=1,
        decoder_type="perceiver",
    ).eval()
    actions = torch.randn(2, 20, 7)
    tokens = tokenizer.tokenize(actions)
    assert tokens.shape == (2, 16)
    assert tokens.max() < 1024
    assert tokenizer.detokenize(tokens).shape == (2, 20, 7)
    output = tokenizer(actions)
    assert output["loss"].ndim == 0 and output["loss"].requires_grad
    assert output["indices"].shape == (2, 16, 1)

    pairs = chunk_hard_dtw_targets(
        torch.randn(4, 20, 7),
        positive_topk=1,
        negative_topk=1,
    )
    assert not (pairs.positive_mask & pairs.negative_mask).any()
    alignment = semantic_contrastive_loss(
        torch.randn(4, 32),
        positive_mask=pairs.positive_mask,
        negative_mask=pairs.negative_mask,
        positive_weight=1.0,
        negative_weight=1.0,
        negative_margin=1.0,
    )
    assert alignment.total.isfinite()


def test_tokenizer_train_defaults_are_soft_dtw_and_diffusion():
    config = ActionCodecTokenizerConfig()
    assert config.decoder_type == "diffusion"
    assert config.model_dim == 512
    assert config.vq_beta == 1.0
    assert config.encoder_cross_layers == 8
    assert config.decoder_cross_layers == 8
    assert config.share_encoder_latent_transformer is True
    assert config.share_decoder_latent_transformer is True
    assert config.share_encoder_cross_attn is True
    assert config.share_decoder_cross_attn is True
    assert config.use_vl_embedder is False
    assert config.batch_size == 512
    assert config.steps == 20000
    assert config.window_stride == 4
    train = SemanticTokenizerTrainConfig(repo_id="dummy")
    assert train.decoder_type == "diffusion"
    assert train.model_dim == 512
    assert train.vq_beta == 1.0
    assert train.encoder_cross_layers == 8
    assert train.batch_size == 512
    assert train.steps == 20000
    assert train.use_vl_embedder is False
    assert train.window_stride == 4
    assert train.adam_beta2 == 0.95
    assert train.grad_clip == 1.0
    assert train.alignment_weight == 0.1
    assert train.hard_alignment_weight == 0.0
    loss_config = train.alignment_loss_config()
    assert loss_config["weight_chunk_align"] == 0.1
    assert loss_config["weight_align"] == 0.0
    assert loss_config["chunk_align_dtw_backend"] == "auto"
    assert loss_config["chunk_align_pair_batch_size"] == 8192
    assert loss_config["chunk_align_max_candidate_pairs"] == 1024
    assert loss_config["chunk_align_min_delta_norm"] == 1e-6
    assert loss_config["chunk_align_normalize_delta"] is True
    tokenizer_config = train.to_tokenizer_config()
    assert tokenizer_config.model_dim == 512
    assert tokenizer_config.vq_beta == 1.0
    assert tokenizer_config.encoder_cross_layers == 8
    assert tokenizer_config.share_encoder_cross_attn is True
    assert tokenizer_config.use_vl_embedder is False


def test_codebook_occupancy_meter_tracks_batch_window_and_total():
    meter = CodebookOccupancyMeter(codebook_size=8, window=1)
    first = meter.update(torch.tensor([[[0], [1], [0]]]))
    assert first["unique_codes_batch"] == 2
    assert first["codebook_occupied_window"] == 2
    assert first["codebook_occupied_total"] == 2
    assert first["codebook_usage_total"] == pytest.approx(2 / 8)
    second = meter.update(torch.tensor([[[2], [3], [2], [3]]]))
    assert second["unique_codes_batch"] == 2
    assert second["codebook_occupied_window"] == 2
    assert second["codebook_occupied_total"] == 4
    assert second["codebook_usage_window"] == pytest.approx(2 / 8)
    assert second["codebook_usage_total"] == pytest.approx(4 / 8)
    assert second["unique_codes"] == second["unique_codes_batch"]
    assert second["codebook_perplexity_window"] > 1.0


def test_window_start_indices_keeps_full_horizon_windows():
    assert _window_start_indices([0], [20], horizon=20, stride=4) == [0]
    assert _window_start_indices([0], [28], horizon=20, stride=4) == [0, 4, 8]
    assert _window_start_indices([10], [38], horizon=20, stride=4) == [10, 14, 18]
    with pytest.raises(ValueError, match="No full horizon"):
        _window_start_indices([0], [19], horizon=20, stride=4)


def test_diffusion_decoder_train_and_sample_contract():
    tokenizer = ActionCodecTokenizer(
        action_dim=7,
        window_size=20,
        model_dim=32,
        num_tokens=16,
        codebook_size=1024,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        encoder_cross_layers=1,
        decoder_cross_layers=1,
        decoder_type="diffusion",
        diffusion_config={"num_train_steps": 8, "num_sample_steps": 2, "denoiser_layers": 1},
    )
    output = tokenizer(torch.randn(2, 20, 7))
    assert output["recon"].shape == (2, 20, 7)
    assert output["indices"].shape == (2, 16, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP")
def test_diffusion_decoder_amp_forward_is_finite():
    tokenizer = ActionCodecTokenizer(
        action_dim=7,
        window_size=20,
        model_dim=32,
        num_tokens=16,
        codebook_size=1024,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        encoder_cross_layers=1,
        decoder_cross_layers=1,
        decoder_type="diffusion",
        diffusion_config={"num_train_steps": 8, "num_sample_steps": 2, "denoiser_layers": 1},
    ).cuda()
    actions = torch.randn(2, 20, 7, device="cuda")
    with torch.amp.autocast(device_type="cuda"):
        output = tokenizer(actions)
    assert output["loss"].isfinite()
    assert output["recon"].shape == (2, 20, 7)


def test_semantic_contrastive_loss_backward_is_finite():
    # The zero-distance diagonal must stay out of the graph, otherwise sqrt backprops 0/0 as NaN.
    embeddings = torch.randn(4, 8, requires_grad=True)
    pairs = chunk_hard_dtw_targets(torch.randn(4, 20, 7), positive_topk=1, negative_topk=1)
    loss = semantic_contrastive_loss(
        embeddings,
        positive_mask=pairs.positive_mask,
        negative_mask=pairs.negative_mask,
        positive_weight=1.0,
        negative_weight=1.0,
        negative_margin=1.0,
    )
    loss.total.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_trajectory_soft_dtw_warps_over_the_chunk_axis():
    delta_state = torch.randn(3, 4, 20, 7)
    valid = torch.ones(3, 4, dtype=torch.bool)
    result = trajectory_soft_dtw_alignments(delta_state, valid, positive_topk=1)
    assert result.distances.shape == (3, 3)
    assert result.alignments.shape == (3, 3, 4, 4)
    assert torch.isfinite(result.distances[result.candidate_mask]).all()


def test_positional_embedding_pads_odd_dimensions():
    # An odd dim makes the sin/cos concatenation one column short of `dim`, exercising the pad branch.
    x = torch.zeros(2, 4, 7)
    assert PositionalEmbedding(7, "sincos")(x).shape == (2, 4, 7)
    freq = torch.full((2,), 20.0)
    assert PositionalEmbedding(7, "fourier")(x, freq=freq).shape == (2, 4, 7)


def test_quantizer_refresh_is_applied_after_forward():
    quantizer = ResidualVectorQuantizer(codebook_size=8, embed_dim=4, dead_code_threshold=1)
    quantizer.train()
    quantizer(torch.zeros(2, 3, 4))
    assert quantizer.last_refreshed_codes == 7
    assert quantizer.apply_pending_codebook_updates() == 7


def test_trainer_passes_normalized_action_to_soft_dtw(monkeypatch):
    import lerobot.actioncodec.trainer as trainer_module

    captured = []

    class Metadata:
        fps = 20
        features = {"action": {"shape": (14,)}}
        stats = {"action": {"mean": [1.0] * 14, "std": [2.0] * 14}}
        _version = "v3.0"

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            del index
            return {
                "action": torch.full((20, 14), 3.0),
                "observation.state": torch.full((20, 14), 99.0),
            }

    class Quantizer:
        def apply_pending_codebook_updates(self):
            return 0

        def discard_pending_codebook_updates(self):
            return 0

    class Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            del kwargs
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.quantizer = Quantizer()

        def forward(self, action, delta_state=None, loss_config=None):
            del loss_config
            captured.append(delta_state.detach().clone())
            loss = self.scale * action.mean()
            return {
                "loss": loss,
                "loss_recon": loss.detach(),
                "loss_vq": loss.detach(),
                "loss_align": loss.detach(),
                "indices": torch.zeros(action.shape[0], 16, 1, dtype=torch.long),
            }

    dataset_kwargs = {}

    def _fake_dataset(*args, **kwargs):
        del args
        dataset_kwargs.update(kwargs)
        return Dataset()

    monkeypatch.setattr(trainer_module, "LeRobotDatasetMetadata", lambda *args, **kwargs: Metadata())
    monkeypatch.setattr(trainer_module, "LeRobotDataset", _fake_dataset)
    monkeypatch.setattr(trainer_module, "ActionCodecTokenizer", Model)
    monkeypatch.setattr(trainer_module, "save_tokenizer_artifact", lambda *args, **kwargs: None)

    class DummyTB:
        def log_dict(self, *args, **kwargs):
            del args, kwargs

        def close(self):
            return None

    monkeypatch.setattr(trainer_module.TensorBoardLogger, "from_log_dir", lambda *args, **kwargs: DummyTB())

    trainer_module.train_semantic_tokenizer(
        SemanticTokenizerTrainConfig(
            repo_id="dummy",
            output_dir="unused",
            steps=1,
            batch_size=2,
            num_workers=0,
            action_dim=14,
            model_dim=32,
            decoder_type="perceiver",
            alignment_weight=1.0,
            hard_alignment_weight=0.0,
            log_freq=0,
        )
    )

    assert dataset_kwargs.get("decode_videos") is False
    assert len(captured) == 1
    assert captured[0].shape == (2, 20, 14)
    torch.testing.assert_close(captured[0], torch.ones(2, 20, 14))


def test_policy_contract_and_task_token_oat_generation():
    config = ActionCodecConfig(
        **_tiny_policy_kwargs(
            num_tasks=3,
            input_features={
                "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
                "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
                "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            },
        )
    )
    config.validate_features()
    oat = OATExactCached(config, cond_dim=2 * 8 + 7 + 1).eval()
    condition = torch.randn(2, 2, 24)
    task_ids = torch.tensor([0, 2])
    bos = torch.full((2, 1), 1024, dtype=torch.long)
    generated = oat.generate(bos, condition, 16, task_ids, temperature=0.0)
    assert generated.shape == (2, 17)
    assert generated[:, 1:].max() < 1024


def test_policy_keeps_tokenizer_frozen_and_processor_registered():
    config = ActionCodecConfig(**_tiny_policy_kwargs(num_tasks=2))
    policy = ActionCodecPolicy(config)
    policy.train()
    assert not policy.tokenizer.training
    assert all(not parameter.requires_grad for parameter in policy.tokenizer.parameters())
    assert ProcessorStepRegistry.get("ActionCodecTaskToken").__name__ == "ActionCodecTaskTokenProcessorStep"
    assert get_policy_class("actioncodec") is ActionCodecPolicy


def test_policy_online_history_updates_while_action_chunk_is_queued():
    config = ActionCodecConfig(**_tiny_policy_kwargs(num_tasks=2))
    policy = ActionCodecPolicy(config)

    generated_chunks = [torch.full((1, 20, 7), float(chunk_index)) for chunk_index in range(2)]
    policy.predict_action_chunk = Mock(side_effect=generated_chunks)

    for step in range(17):
        value = float(step)
        action = policy.select_action(
            {
                "observation.images.front": torch.full((1, 3, 8, 8), value),
                "observation.state": torch.full((1, 7), value),
                "task_uid": torch.tensor([1]),
            }
        )
        assert action.shape == (1, 7)

    assert policy.predict_action_chunk.call_count == 2
    first_batch = policy.predict_action_chunk.call_args_list[0].args[0]
    second_batch = policy.predict_action_chunk.call_args_list[1].args[0]
    assert first_batch["observation.state"].shape == (1, 2, 7)
    assert first_batch["observation.state"][0, :, 0].tolist() == [0.0, 0.0]
    assert second_batch["observation.state"][0, :, 0].tolist() == [15.0, 16.0]
    assert second_batch["observation.images.front"][0, :, 0, 0, 0].tolist() == [15.0, 16.0]

    policy.reset()
    assert not policy._action_queue
    assert all(not queue for queue in policy._observation_queues.values())


def test_policy_accepts_single_task():
    config = ActionCodecConfig(**_tiny_policy_kwargs(num_tasks=1))
    config.validate_features()
    policy = ActionCodecPolicy(config)
    batch = {
        "action": torch.randn(2, 20, 7),
        "observation.images.front": torch.rand(2, 2, 3, 8, 8),
        "observation.state": torch.randn(2, 2, 7),
        "task_uid": torch.zeros(2, dtype=torch.long),
    }
    loss, logs = policy(batch)
    assert loss.ndim == 0 and loss.isfinite()
    features = policy.obs_encoder(batch)
    assert features.isfinite().all()
    assert float(logs["token_ce"]) == pytest.approx(float(loss.item()))
    assert "token_top5_acc" in logs
    assert "task_token_swap_ce_gap" not in logs
