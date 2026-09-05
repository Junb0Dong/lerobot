#!/usr/bin/env python3
"""Verify a trained FSQ artifact before launching its dependent policy job."""

import argparse
import gc
import json
from pathlib import Path

import torch

from lerobot.actioncodec.models.fsq import FSQGrid
from lerobot.policies.actioncodec.configuration_actioncodec import ActionCodecConfig
from lerobot.policies.actioncodec.modeling_actioncodec import ActionCodecPolicy, FrozenTokenizerAdapter


def verify(tokenizer_path: Path, policy_path: Path | None = None) -> None:
    grid = FSQGrid().cuda()
    ids = torch.arange(1000, device="cuda")
    assert torch.equal(grid.scalar_classes_to_indices(grid.indices_to_scalar_classes(ids)), ids)
    cfg = ActionCodecConfig(
        quantizer_type="semantic_fsq",
        codebook_size=1000,
        action_dim=12,
        num_tasks=1,
        tokenizer_path=tokenizer_path,
        device="cuda",
    )
    tokenizer = FrozenTokenizerAdapter(cfg).cuda()
    assert all(torch.isfinite(p).all() for p in tokenizer.parameters())
    torch.manual_seed(42)
    action = torch.randn(2, 20, 12, device="cuda")
    tokens = tokenizer.tokenize(action)
    prediction = tokenizer.detokenize(tokens)
    assert tokens.shape == (2, 16) and prediction.shape == (2, 20, 12)
    assert tokens.min() >= 0 and tokens.max() < 1000 and torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction, tokenizer.detokenize(tokens), rtol=0, atol=0)
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if policy_path:
        policy = ActionCodecPolicy.from_pretrained(policy_path, local_files_only=True).cuda().eval()
        assert policy.config.quantizer_type == "semantic_fsq"
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        batch = {
            "observation.state": torch.zeros(1, 2, 12, device="cuda"),
            "task_uid": torch.zeros(1, dtype=torch.long, device="cuda"),
        }
        for key, feature in policy.config.image_features.items():
            batch[key] = torch.zeros(1, 2, *feature.shape, device="cuda")
        prediction = policy.predict_action_chunk(batch)
        assert torch.isfinite(prediction).all() and prediction.shape == (1, 20, 12)
        torch.testing.assert_close(prediction, policy.predict_action_chunk(batch), rtol=0, atol=0)
        assert all(not p.requires_grad for p in policy.tokenizer.parameters())
        print(
            json.dumps(
                {
                    "total_parameters": sum(p.numel() for p in policy.parameters()),
                    "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                    "ar_parameters": sum(p.numel() for p in policy.model.parameters()),
                }
            )
        )
    print("FSQ artifact verification passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path)
    args = parser.parse_args()
    verify(args.tokenizer_path, args.policy_path)
