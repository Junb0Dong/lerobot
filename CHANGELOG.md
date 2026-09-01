# Changelog

## [0.6.2] - 2026-09-01

### Features

- Changed the ActionCodec policy default vision encoder to OAT's robomimic
  `ResNet18Conv + SpatialSoftmax(32)` path, with one independent network per camera and a
  64-dimensional feature per camera.
- Matched OAT's full vision dataflow: robomimic `CropRandomizer` with a 76×76 crop,
  BatchNorm-to-GroupNorm replacement, RGB normalization to `[-1, 1]`, the
  `ObservationEncoder` output ReLU, and crop `forward_out` handling.
- Added the `lerobot[actioncodec]` extra with `robomimic==0.2.0`, included it in
  `lerobot[all]`, and added dependency checks to local and DLC training entry points.
- Added matched-h20 tokenizer and policy launchers for the current 12D XLeRobot no-head dataset,
  with a shared dataset/tokenizer/output naming contract.

### Design Rationale

- The previous self-contained `resnet_spatial` encoder preserved the 64-D observation contract
  but was substantially narrower than OAT. Using OAT's native robomimic implementation removes
  topology, crop-boundary, and output-activation differences without changing the policy forward
  or prediction APIs.
- `resnet_spatial` and `small_cnn` remain explicit configuration options so existing checkpoints
  can retain their original parameter shapes and behavior.

### Notes & Caveats

- This is an architecture-changing default. Checkpoints that saved
  `vision_encoder="resnet_spatial"` or `vision_encoder="small_cnn"` continue to load unchanged.
  Historical checkpoints without a `vision_encoder` field must explicitly select the encoder they
  were trained with; there is no silent fallback or weight migration.
- Install the aligned encoder environment with
  `uv sync --locked --extra training --extra diffusion --extra actioncodec`.
- Train the 12D tokenizer before launching the policy; 12D no-head artifacts are incompatible
  with the historical 14D dataset and checkpoints.
- The OAT-exact encoder has 11,197,088 parameters per camera. For the three-camera RoboCasa
  contract, the trainable policy grows from approximately 13.43M to 38.64M parameters, increasing
  memory use and step time.
