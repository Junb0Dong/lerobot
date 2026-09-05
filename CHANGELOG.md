# Changelog

## [Unreleased] - 2026-09-05

### Features

- Organized the current XLeRobot training, FSQ, physical-loss and offline evaluation workflows
  on the `real-robot` branch; documented maintained entry points in `scripts/README.md`.
- Removed the superseded square no-crop launcher, standalone physical-aux smoke script and empty
  project lessons copy. Excluded local network tooling and cleaned generated test caches.
- Restored the baseline tokenizer launcher to the supported 16-token contract and matching output
  path; removed automatic deletion of existing outputs from the cropped baseline launcher.

### Design Rationale

- Keep regression tests and checkpoint compatibility paths. Reuse the physical-aux training
  launcher with short-run overrides instead of maintaining a duplicate smoke command.
- Consolidate current project status while retaining the separate physical evaluation evidence.

### Notes & Caveats

- Datasets, checkpoints, evaluation outputs and local network configuration are not published.
  Existing runs with pruned checkpoints require explicit retained steps for offline evaluation.
- No hardware was connected during branch preparation.

## [0.6.8] - 2026-09-05

### Features

- Added semantic FSQ tokenizer (`512 -> 4 -> [8,5,5,5] -> 512`, vocabulary 1000) and
  four-head cached ActionCodec AR policy. Artifact/config fields select VQ or FSQ explicitly.
- FSQ DTW alignment uses quantized 4D coordinates with the existing mining, pooling and loss;
  overlap/physical tokenizer objectives and Perceiver/diffusion architecture remain available.
- Four scalar CEs are averaged for training; logs include each head's CE/accuracy, scalar and
  complete-token accuracy, and summed `token_nll`. FSQ defaults to greedy scalar generation.
- Added isolated 20k tokenizer -> 50k policy launchers, artifact verification, continuity evaluation
  support and FSQ contract/gradient/causality/cache/checkpoint tests.

### Design Rationale

- This experiment compares the complete FSQ approach: quantization, alignment representation and
  policy heads change together. Use decoded physical metrics to judge it; scalar CE is not directly
  comparable to VQ token CE. The matched 1024-width/15-layer/16-head recipe now has 294,028,256
  trainable parameters (342,655,984 including its frozen tokenizer).
- Tokenizer AMP dtype is explicit and saved in the artifact config. The first FP16 smoke aborted
  on a non-finite scaled gradient at step 0; FSQ uses BF16 after a successful 100-step retry.
  Existing VQ training retains its FP16 default and checkpoint parameter names.

### Notes & Caveats

- Old artifacts default to VQ and strict-load unchanged. Quantizer/artifact mismatches are rejected;
  FSQ currently rejects nonzero policy distance or decoded auxiliary losses.
- FSQ has no learned codebook, commitment loss or dead-code refresh. Full experiment metrics remain
  pending; smoke checkpoints only verify execution, saving and reloading.
- Launchers use `uv run --no-sync`, independent output/log directories, GPU 0 and policy port 29517;
  full-run evaluation uses fixed pair manifests, checkpoint normalization, greedy policy and EMA.

## [0.6.7] - 2026-09-05

### Features

- Added optional ActionCodec `codebook_distance_loss_weight`: CE plus expected squared distance
  to the target's frozen tokenizer code, normalized by the mean distance over all code pairs.
- Added a 295M experiment launcher reusing the existing native-resolution training script,
  with distance weight 0.5 and all decoded physical losses disabled.

### Design Rationale

- Penalize far-away code errors more than nearby ones without retraining the tokenizer, changing
  the AR head, or backpropagating through the diffusion decoder. Compute geometry in FP32 under AMP.

### Notes & Caveats

- The default weight is zero; model parameters, state-dict keys and inference are unchanged.
  Latent distance is only a proxy for physical action distance, not a guarantee of chunk continuity.
- The experiment starts a new 50k run with the baseline architecture/seed/data split; it does not
  fine-tune or overwrite the previous 50k checkpoint. It uses the existing environment (`uv run
--no-sync`); `SKIP_UV_SYNC=0` explicitly enables the base launcher's dependency sync.
- Compare `token_ce` separately from total loss; evaluate free-running held-out physical metrics
  before concluding that the new loss improves action continuity.

## [0.6.6] - 2026-09-05

### Features

- Added `decoded_metrics_interval` to throttle extra ActionCodec decoded diagnostics during
  training; the physical-aux launcher computes them every 100 forwards and the smoke launcher
  every 10 forwards. Evaluation always computes the full diagnostics.
- Cached diffusion sampling indices on the CPU and added a single-embodiment denoiser path to
  avoid per-timestep CUDA scalar extraction and boolean-mask synchronization.

### Design Rationale

- Retain every loss-bearing decode and all 27 sampling steps, while avoiding repeated diagnostic
  decodes between logging points. Isolate diagnostic AR dropout from the training RNG stream.

### Notes & Caveats

- The config default remains 1 for older checkpoints. Physical loss weights, batch size and
  sampling schedule are unchanged; existing running processes must restart to use the code.
- Use a diagnostic interval matching the logging interval. Sampled diagnostic averages cover
  only the measured forwards, while loss-bearing branch metrics remain available every step.

## [0.6.5] - 2026-09-04

### Features

- Added optional episode-local overlapping-window consistency and denormalized physical action
  reconstruction/velocity losses for ActionCodec tokenizer training.
- Added deterministic XLeRobot tokenizer continuity evaluation and an independent overlap/physical
  training launcher.
- Added optional decoded physical auxiliary losses to the ActionCodec semantic policy, using
  straight-through Gumbel token embeddings, a frozen differentiable decoder, prefix corruption,
  and episode-local paired windows.
- Added a read-only semantic-policy physical evaluator and physical-aux training launchers for
  100-step smoke and 50k training.

### Design Rationale

- Paired windows share diffusion timestep and aligned union noise so continuity supervision measures
  window context differences instead of independent training noise.
- Auxiliary weights default to zero, preserving the existing tokenizer training contract unless the
  experiment explicitly enables them.
- The semantic policy keeps token CE and sends decoded reconstruction, velocity, first-target,
  overlap, and seam gradients through the frozen codebook/decoder into the policy and observation
  encoder.

### Notes & Caveats

- The launcher uses `overlap_shift=16`, 256 pairs per effective 512-window batch, and excludes the
  two gripper dimensions through explicit continuous action indices.
- The differentiable auxiliary path runs the complete deterministic 27-step decoder; this increases
  memory and step time. `temperature=0` in the launcher is an inference setting, not a training
  amplitude constraint.
- The 100-step smoke and evaluator comparison are startup checks; they do not replace full-training
  quality acceptance.

## [0.6.4] - 2026-09-03

### Features

- Added rectangular ActionCodec vision inputs and a native-resolution XLeRobot launcher that keeps
  the dataset's 96×128 RGB frames when using TorchCodec and disables random cropping.

### Design Rationale

- A scalar `image_size` forced all ActionCodec camera inputs to a square. Accepting `(height, width)`
  lets the OAT-exact visual encoder preserve the source aspect ratio without a redundant resize.

### Notes & Caveats

- Integer `image_size` values retain their existing square-input behavior. The 96×128 no-crop
  configuration changes the visual feature geometry and requires training a new policy checkpoint.
- The native launcher starts with 8 persistent workers and `prefetch_factor=4`. Increase workers only
  when measured DataLoader wait is significant; TorchCodec decoder caches are per worker, so RAM and
  open-file usage grow with worker count.

## [0.6.3] - 2026-09-02

### Features

- Added an ActionCodec OAT-exact no-crop vision mode: `crop_shape=None` now sends the full resized
  image directly through each camera's robomimic `ResNet18Conv + SpatialSoftmax` visual core.
- Added an XLeRobot no-crop policy launcher using 128×128 worker-side resize, TorchCodec decoder
  caching, persistent DataLoader workers, `temperature=0.1`, and `top_k=3`.

### Design Rationale

- The previous robomimic `CropRandomizer` also sampled crops during evaluation and rollout, so the
  same camera frame could produce different visual features and action chunks. Building the visual
  core for the full resized image removes this source of spatial randomness.

### Notes & Caveats

- No-crop changes the visual encoder's input geometry and requires retraining the semantic policy;
  checkpoints trained with 76×76 or 96×96 crops should retain their saved crop configuration.
- `temperature=0.1` with `top_k=3` remains stochastic. Set `temperature=0` at deployment time for
  deterministic token generation.

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
