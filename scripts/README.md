# XLeRobot / ActionCodec scripts

`real-robot` keeps the current 12D, three-camera XLeRobot training and offline evaluation workflows.
Launchers default to `../data/my_dataset_merged_0902_no_head_96x128`; override `DATASET_ROOT`,
`TOKENIZER_PATH`, and `OUTPUT_DIR` for another checkout. Datasets and checkpoints are local artifacts.
The current model contract and experiment status are in
[agent_docs/04](../agent_docs/04_xlerobot_actioncodec_status_0901.md).

## Training

| Workflow                                             | Entry point                                                                             |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| VQ tokenizer, h20 / 16 tokens                        | `train_xlerobot_actioncodec_tokenizer_no_head.sh`                                       |
| VQ tokenizer with overlap/physical losses            | `train_xlerobot_actioncodec_tokenizer_overlap_phys_no_head.sh`                          |
| VQ policy, native 96×128 / no crop                   | `train_xlerobot_actioncodec_oat_exact_native_nocrop_torchcodec.sh`                      |
| Historical 128×128 / 96×96 crop baseline             | `train_xlerobot_actioncodec_oat_exact_torchcodec.sh`                                    |
| VQ tokenizer followed by policy                      | `train_xlerobot_actioncodec_overlap_phys_then_policy.sh`                                |
| 295M VQ policy with codebook-distance loss           | `train_xlerobot_actioncodec_codebook_distance_295m.sh`                                  |
| VQ policy with decoded physical losses               | `train_xlerobot_actioncodec_oat_exact_physical_aux_nocrop_torchcodec.sh`                |
| FSQ tokenizer followed by 295M policy and evaluation | `train_xlerobot_actioncodec_fsq_then_policy.sh`                                         |
| Separate FSQ stages                                  | `train_xlerobot_actioncodec_tokenizer_fsq.sh`, `train_xlerobot_actioncodec_fsq_295m.sh` |

The old square no-crop launcher has been retired in favor of the native-resolution entry point.
Checkpoints trained with crops retain their saved visual configuration.

The standalone physical-aux smoke script has been removed. Use the full launcher with a fresh output
and CLI overrides for a short startup check (this command starts training):

```bash
SKIP_UV_SYNC=1 POLICY_STEPS=100 BATCH_SIZE=2 DECODED_METRICS_INTERVAL=10 \
OUTPUT_DIR=outputs/physical_aux_smoke_new \
bash scripts/train_xlerobot_actioncodec_oat_exact_physical_aux_nocrop_torchcodec.sh \
  --num_workers=0 --persistent_workers=false \
  --log_freq=10 --eval_steps=50 --max_eval_samples=2 --save_freq=100 --ema.enable=false
```

`SKIP_UV_SYNC=1` reuses an already prepared environment. Install missing training dependencies with
`uv sync --locked --extra training --extra diffusion --extra actioncodec`.

## Offline evaluation

- `verify_xlerobot_actioncodec_fsq.py`: FSQ artifact loading, finite outputs and deterministic decode.
- `eval_xlerobot_actioncodec_tokenizer_continuity.py`: reconstruction, overlap and seam metrics.
- `eval_xlerobot_actioncodec_policy_physical.py`: held-out policy metrics using checkpoint processors.
- `diagnose_xlerobot_chunk_boundary.py`: matched GT/tokenizer/policy boundary L2; optional recorded
  rollout chunks for the deployment comparison.

Each Python script exposes `--help`. For policy runs whose early checkpoints were deleted, pass
`--steps 50000` (or another retained step); the default sweep expects all 5k checkpoints.
Use the same pair manifest, normalization and EMA/sampling convention for comparisons.
These evaluations do not connect hardware. Simulation workflows remain under [dlc/](dlc/README.md).
