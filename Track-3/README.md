# WorldArena Track 3 Post-Training

This directory provides the training, checkpoint, offline-evaluation, and
deployment entry points used for WorldArena Track 3.1 (UniVTAC) and Track 3.2
(Franka). Raw datasets, model weights, organizer credentials, and hidden test
sets are not included.

Detailed references:

- [Track 3.1 quickstart](docs/TRACK31_QUICKSTART.md)
- [Track 3.1 data and evaluation contract](docs/TRACK31_UNIVTAC.md)
- [Track 3.2 Franka contract](docs/TRACK32_FRANKA.md)
- [Post-training configuration](docs/POST_TRAINING.md)
- [Deployment](docs/DEPLOY.md)

## Install

### Requirements

- Linux
- Python 3.10, 3.11, or 3.12
- Git and `uv` (recommended)
- A BF16-capable accelerator exposed to PyTorch through `torch.cuda`
- Sufficient local/shared storage for the base model, converted datasets,
  latents, and checkpoints

NVIDIA CUDA is the portable path. HCU users must start from the vendor PyTorch
runtime that exposes the compatible `cuda:N` API. Do not replace a working HCU
runtime with a generic CUDA wheel.

### Clone and create the environment

```bash
git clone https://github.com/destinyls/WorldArena-2.0-Challenge.git
cd WorldArena-2.0-Challenge/Track-3

uv venv --python 3.12
source .venv/bin/activate
uv pip install --editable '.[track31,track32,dev]'

# LeRobot 0.3.3 declares a historical torch<2.8 dependency. Keep the active
# project/vendor Torch runtime and install only the package itself.
uv pip install --no-deps 'lerobot==0.3.3'
uv pip install 'huggingface-hub[cli]'
```

The equivalent `venv`/`pip` installation is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable '.[track31,track32,dev]'
python -m pip install --no-deps 'lerobot==0.3.3'
python -m pip install 'huggingface-hub[cli]'
```

### Download the released base model

Keep mutable work products outside the Git checkout:

```bash
export TRACK3_ROOT="$PWD"
export N0_WORK=/absolute/path/to/worldarena-track3-work
export N0_BASE="$N0_WORK/models/n0-twam-base"

mkdir -p "$N0_WORK/models" "$N0_WORK/cache" \
  "$N0_WORK/requests" "$N0_WORK/runs"
hf download NeoteAI/n0-twam-base --local-dir "$N0_BASE"

test -f "$N0_BASE/transformer/config.json"
test -f "$N0_BASE/transformer/diffusion_pytorch_model.safetensors"
test -f "$N0_BASE/empty_emb.pt"
test ! -L "$N0_BASE/empty_emb.pt"
```

### Verify the installation

```bash
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
n0-twam --version
n0-twam track31 --help
n0-twam track32 --help
```

`flash-attn` is optional and is needed only when explicitly selecting the
FlashAttention backend. See [INSTALL.md](docs/INSTALL.md). The source is
provided under the terms in [LICENSE](LICENSE).

## Prepare Dataset

All raw inputs should be immutable. Use new output directories for conversion,
latent encoding, requests, runs, and evaluation artifacts. Do not place any
output directory inside a raw dataset or model directory.

### Tactile profiles

Training uses one explicit profile:

| Profile | Dataset contract | Tactile parameters |
|---|---|---|
| `vision_tactile` | Every selected repository contains declared tactile streams | Trainable |
| `mixed` | `per_repo_tactile_keys` explicitly lists tactile and RGB-only repositories | Trainable; RGB-only batches use `tactile_cond_drop` |
| `vision_only` | No repository supplies tactile streams | Frozen and excluded from AdamW |

`tactile_optional=True` is not a formal profile. Missing tactile payloads fail
closed instead of silently converting a tactile repository to RGB-only. The
editable mixed template is `n0_twam/configs/twam_mixed_cfg.py`.

### Track 3.1: prepare UniVTAC

Provide the raw eight-task UniVTAC HDF5 tree, an immutable 40-episode
development/evaluation declaration, and the one-episode quarantine declaration.
The raw episodes must contain head/wrist RGB, two tactile RGB streams, joint
observations, and timestamps. The exact split schema is documented in
[TRACK31_UNIVTAC.md](docs/TRACK31_UNIVTAC.md).

```bash
export N0_RAW=/absolute/path/to/UniVTAC
export N0_SPLITS=/absolute/path/to/univtac-splits
export N0_ARTIFACTS="$N0_WORK/track31/artifacts"
export N0_LEROBOT="$N0_WORK/track31/lerobot"

python script/track3_1/prepare_univtac.py \
  --data-root "$N0_RAW" \
  --validation-manifest "$N0_SPLITS/validation.json" \
  --quarantine-manifest "$N0_SPLITS/quarantine.json" \
  --artifact-dir "$N0_ARTIFACTS" \
  --materialize-root "$N0_LEROBOT" \
  --repo-id univtac_track31 \
  --emit-standard-views
```

Encode only `train759` while developing or training. Keep `frozen40` sealed
until a checkpoint has been selected for evaluation.

```bash
export N0_ENCODER_ID="$N0_WORK/cache/track31_encoder_source_identity.json"
export N0_PROMPT_CACHE="$N0_WORK/cache/train759_prompt_embeddings.pt"

python script/track3_1/cache_encoder_source_identity.py \
  --model-path "$N0_BASE" \
  --output-path "$N0_ENCODER_ID"

CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
python script/track3_1/precompute_hcu_prompt_cache.py \
  --dataset-root "$N0_LEROBOT/train759" \
  --model-path "$N0_BASE" \
  --artifact-root "$N0_ARTIFACTS" \
  --split train759 \
  --encoder-source-identity-path "$N0_ENCODER_ID" \
  --output-path "$N0_PROMPT_CACHE" \
  --device cuda:0

python script/encode_lerobot_n0_latents.py \
  --dataset-root "$N0_LEROBOT/train759" \
  --model-path "$N0_BASE" \
  --artifact-root "$N0_ARTIFACTS" \
  --split train759 \
  --prompt-embedding-cache-path "$N0_PROMPT_CACHE" \
  --encoder-source-identity-path "$N0_ENCODER_ID" \
  --device cuda:0 \
  --text-encoder-device cuda:0 \
  --dtype bf16 \
  --target-fps 10 \
  --height 256 \
  --width 256 \
  --max-sequence-length 512

python script/encode_tactile_latent.py \
  --dataset-root "$N0_LEROBOT/train759" \
  --model-path "$N0_BASE" \
  --artifact-root "$N0_ARTIFACTS" \
  --split train759 \
  --tactile-keys observation.images.tactile_a observation.images.tactile_b \
  --mode both \
  --local-mode current \
  --device cuda:0 \
  --encoder-source-identity-path "$N0_ENCODER_ID" \
  --dtype bf16 \
  --target-fps 10 \
  --height 128 \
  --width 128

python script/track3_1/finalize_track31_latents.py \
  --dataset-root "$N0_LEROBOT/train759" \
  --model-path "$N0_BASE" \
  --artifact-root "$N0_ARTIFACTS" \
  --split train759 \
  --kind both \
  --validate-pair
```

Expected outputs include `universe_manifest_v4.json`, `conversion_report.json`,
standard views and normalizers under `$N0_ARTIFACTS`, plus content-addressed
video/tactile inventories under `$N0_LEROBOT/train759`.

### Track 3.2: prepare official Franka data

The single preparation command freezes the official revision, verifies and
downloads 600 episodes, converts both RGB streams and 8D `end_pose_base`
labels, creates development/final views, and encodes the video latents:

```bash
export N0_FRANKA_WORK="$N0_WORK/track32-franka"

./prepare_track32_franka_data.sh \
  "$N0_FRANKA_WORK" \
  "$N0_BASE" \
  8
```

A complete preparation contains:

```text
$N0_FRANKA_WORK/
├── provenance/official_franka_inventory.json
├── provenance/download_receipt.json
├── data/official/{clear_up,pour,wipe}/
├── data/lerobot/all600/
└── artifacts/
    ├── prepare_receipt.json
    ├── conversion_report.json
    ├── franka_video_latent_inventory.json
    ├── normalizers/
    └── views/
```

The development view uses 540 episodes for training and 60 for internal
validation. `final_refit` uses all 600 public episodes and no internal
validation. This split is not the organizer's hidden real-robot test set.

## Track 3.1

### Training contract

| Item | Value |
|---|---|
| Dataset | UniVTAC `train759` |
| Default profile | `vision_tactile` |
| Action target | 8D `qpos8_next_step` |
| Default recipe | 1500 optimizer steps, save every 300, validate every 100 |
| Entry point | `n0-twam track31` or `./run_track31.sh` |

### Create and validate the request

```bash
n0-twam track31 template train \
  --output "$N0_WORK/requests/track31-development.json"
```

Edit the generated JSON and bind it to the prepared artifact root, LeRobot
root, base model, `empty_emb.pt`, released transformer SHA-256, a new output
root, and the desired local device IDs. For a fresh Stage A development run,
use `profile=multitask_pretrain_v1`, `run_role=development`, `batch_size=1`,
`gradient_accumulation_steps=1`, and no resume checkpoint.

```bash
python -m json.tool "$N0_WORK/requests/track31-development.json" >/dev/null
n0-twam track31 train \
  --config "$N0_WORK/requests/track31-development.json" \
  --dry-run
```

### Start training

```bash
n0-twam track31 train \
  --config "$N0_WORK/requests/track31-development.json" \
  > "$N0_WORK/runs/track31-development.result.json"
```

The command performs preflight, starts one local `torchrun`, waits for all
workers, and verifies the strict checkpoint sidecars before returning
`status=complete`. Monitor the rank logs and accelerator separately:

```bash
tail -F "$N0_WORK/runs/track31-development/logs"/train.*.log
# NVIDIA: nvidia-smi dmon
# HCU:     watch -n 1 hy-smi
```

For strict resume, create a new request and a new output root. Keep the same
world size, profile, run role, batch settings, scheduler horizon, data identity,
and code/runtime identity; set `resume_from` to a complete checkpoint. A direct
1500-step run is valid. The `20 -> 25 -> 1500` ladder is optional hardware
bring-up, not an algorithmic requirement.

After recipe selection, start `final_refit` again from the released base model
with a new request/output root and all 759 training episodes. Do not resume the
719-episode development optimizer state into `final_refit`.

### Target-10 offline evaluation

After selecting a checkpoint, encode the sealed `frozen40` repository with a
separate prompt cache and inventories. Then create and run the strict request:

```bash
n0-twam track31 template eval \
  --output "$N0_WORK/requests/target10.eval.json"

# Bind the checkpoint, frozen view, raw root, normalizer, approved metric, and
# golden calibration identities in the generated JSON before running.
n0-twam track31 eval \
  --request "$N0_WORK/requests/target10.eval.json"
```

The default repository evaluation is five `Insert HDMI` plus five `Lift Bottle`
episodes and reports tactile PSNR/SSIM. The metric/golden assets are external
and must be explicitly supplied. This offline protocol is not automatically an
organizer hidden-test or simulator-success result.

## Track 3.2

### Training contract

| Item | Value |
|---|---|
| Dataset | Official 600-episode Franka release |
| Default profile | `vision_only` |
| Official action | `[x, y, z, qw, qx, qy, qz, gripper]` |
| Model action mapping | 8D pose -> EE10 -> channels `0..9` of the 20D head; channels `10..19` masked |
| Default recipe | 1500 optimizer steps, save every 300, validate every 100 |
| Entry point | `n0-twam track32` or `./run_track32_franka.sh` |

The Franka route has no tactile input. Tactile-only parameters remain in the
checkpoint-compatible model structure but are frozen and excluded from AdamW.
`joint_qpos` is an observation/fallback signal, not the official action label.

### Build and validate the request

```bash
mkdir -p "$N0_FRANKA_WORK/requests" "$N0_FRANKA_WORK/runs"

n0-twam track32 build-request \
  --output "$N0_FRANKA_WORK/requests/franka-development.json" \
  --run-id franka-development-v1 \
  --devices 0,1,2,3,4,5,6,7 \
  --accelerator-profile portable \
  --artifact-root "$N0_FRANKA_WORK/artifacts" \
  --lerobot-root "$N0_FRANKA_WORK/data/lerobot" \
  --base-model "$N0_BASE" \
  --empty-embedding "$N0_BASE/empty_emb.pt" \
  --init-from "$N0_BASE" \
  --output-root "$N0_FRANKA_WORK/runs/franka-development-v1" \
  --run-role development \
  --num-steps 1500 \
  --stop-after-step 1500 \
  --save-interval 300 \
  --val-interval 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 1

n0-twam track32 train \
  --config "$N0_FRANKA_WORK/requests/franka-development.json" \
  --dry-run
```

Use `--accelerator-profile hcu_performance` only in the validated HCU image and
also provide `--collective-network-interface <interface>`.

### Start training

```bash
./run_track32_franka.sh \
  "$N0_FRANKA_WORK/requests/franka-development.json" \
  | tee "$N0_FRANKA_WORK/franka-development.result.json"
```

Create `final_refit` with a new run ID/output root and initialize again from
the released base model. Do not resume the 540-episode development optimizer
state into the 600-episode final-refit run.

### Generate offline metrics

First seal a complete checkpoint and create the Policy config:

```bash
export N0_FRANKA_CHECKPOINT=/absolute/path/to/checkpoint_step_1500
export N0_FRANKA_CHECKPOINT_ID=CHECKPOINT_IDENTITY_FROM_TRAINING_RECEIPT
export N0_FRANKA_NORMALIZER="$N0_FRANKA_WORK/artifacts/normalizers/franka_dev_train540_v1.json"
export N0_FRANKA_VIEW="$N0_FRANKA_WORK/artifacts/views/franka_dev_validation60_v1.json"

n0-twam track32 serve-bundle \
  --checkpoint "$N0_FRANKA_CHECKPOINT" \
  --checkpoint-identity-sha256 "$N0_FRANKA_CHECKPOINT_ID" \
  --base-model "$N0_BASE" \
  --normalizer "$N0_FRANKA_NORMALIZER" \
  --normalizer-sha256 NORMALIZER_SEMANTIC_SHA_FROM_REQUEST \
  --output "$N0_FRANKA_WORK/serve-bundle"

n0-twam track32 policy-template \
  --output "$N0_FRANKA_WORK/policy.json"
```

Use the serve-bundle receipt and frozen validation view to generate and score
future predictions:

```bash
mkdir -p "$N0_FRANKA_WORK/eval"

n0-twam track32 generate-predictions \
  --checkpoint "$N0_FRANKA_CHECKPOINT" \
  --checkpoint-identity-sha256 "$N0_FRANKA_CHECKPOINT_ID" \
  --serve-bundle "$N0_FRANKA_WORK/serve-bundle" \
  --serve-bundle-receipt-sha256 SERVE_BUNDLE_RECEIPT_FILE_SHA256 \
  --serve-output "$N0_FRANKA_WORK/eval/direct-backend-output" \
  --artifact-root "$N0_FRANKA_WORK/artifacts" \
  --lerobot-root "$N0_FRANKA_WORK/data/lerobot" \
  --base-model "$N0_BASE" \
  --normalizer "$N0_FRANKA_NORMALIZER" \
  --dataset-view "$N0_FRANKA_VIEW" \
  --output "$N0_FRANKA_WORK/eval/franka-future-predictions.npz" \
  --device 0 \
  > "$N0_FRANKA_WORK/eval/generation-result.json"

n0-twam track32 score-predictions \
  --predictions "$N0_FRANKA_WORK/eval/franka-future-predictions.npz" \
  --checkpoint-identity-sha256 "$N0_FRANKA_CHECKPOINT_ID" \
  --dataset-view-id franka_dev_validation60_v1 \
  --dataset-view-sha256 VALIDATION_VIEW_SHA256_FROM_GENERATION_RESULT \
  --decoder-sha256 VAE_DECODER_SHA256_FROM_GENERATION_RESULT \
  --output "$N0_FRANKA_WORK/eval/franka-offline-metrics.json"
```

The offline report contains future-RGB PSNR/SSIM and end-position MAE/RMSE.
These are model-development proxies, not a Track 3.2 real-robot success rate.

### Policy replay and official worker

Replace every `CALIBRATE_*` value in `policy.json` with limits approved for the
target robot cell. Before connecting to a robot, run the real Policy backend on
a frozen non-pickled observation NPZ:

```bash
n0-twam track32 policy-replay \
  --config "$N0_FRANKA_WORK/policy.json" \
  --observation "$N0_FRANKA_WORK/offline-observation.npz" \
  --prompt "clear the table" \
  --steps 7 \
  --output "$N0_FRANKA_WORK/offline-policy-replay.json"
```

Only after organizer approval, credentials, and bridge verification should the
official outbound worker be started:

```bash
export WORLD_ARENA_ROOT=/absolute/path/to/WorldArena-2.0
export N0_TRACK32_POLICY_CONFIG="$N0_FRANKA_WORK/policy.json"
export HUB_POLICY_URL=https://ORGANIZER_GATEWAY/policy
export POLICY_ID=ORGANIZER_ASSIGNED_WORKER_KEY
export HUB_TOKEN=ORGANIZER_BEARER_TOKEN  # optional

bash run_track32_franka_worker.sh
```

Data conversion, finite training loss, checkpoint completion, offline metrics,
and Policy replay are engineering evidence only. The official Track 3.2 result
requires the organizer worker, approved Franka cell, and real task-success
evaluation.
