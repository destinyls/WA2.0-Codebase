# WorldArena Track 3 Post-Training

This directory provides the training, checkpoint, offline-evaluation, and
deployment entry points used for WorldArena Track 3.1 (UniVTAC) and Track 3.2
(Franka and AgileX). Raw datasets, model weights, organizer credentials, and
hidden test sets are not included.

Detailed references:

- [Track 3.1 quickstart](docs/TRACK31_QUICKSTART.md)
- [Track 3.1 data and evaluation contract](docs/TRACK31_UNIVTAC.md)
- [Track 3.2 Franka contract](docs/TRACK32_FRANKA.md)
- [Track 3.2 AgileX qpos14 contract](docs/TRACK32_AGILEX.md)
- [Post-training configuration](docs/POST_TRAINING.md)
- [Deployment](docs/DEPLOY.md)

### Support and evidence status

| Route | Implemented in this repository | Evidence boundary |
|---|---|---|
| UniVTAC Track 3.1 | Pinned download, frozen splits, conversion, latent encoding, strict training, and Target-10 proxy evaluation | Public-data/offline engineering evidence; not a hidden-test or robot result |
| Franka Track 3.2 | Automated 600-episode preparation, EE20 post-training, strict checkpoints, offline metrics, synchronous Policy replay, and organizer-worker gate | The worker still requires organizer credentials, an approved robot cell, and official task-success evaluation |
| AgileX Track 3.2 | Pinned official reader and materializer, qpos14 training, `vision_tactile`/`mixed`/`vision_only`, strict checkpoints, Stage-B weights-only initialization, offline metrics, synchronous re-grounding, and an in-process Policy adapter | Offline evaluation uses training-distribution samples; physical-camera, real-robot, and leaderboard evaluation remain incomplete |

The test count is intentionally not pinned in this document because it changes
with every revision. Run the commands in [Development and verification](#development-and-verification)
against the exact source revision being evaluated. A passing software suite is
not a claim of real-robot success or official leaderboard acceptance.

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

| Dataset | Raw-data source | Preprocessing entry point |
|---|---|---|
| UniVTAC | Official [`byml/UniVTAC`](https://huggingface.co/datasets/byml/UniVTAC) snapshot | `script/track3_1/prepare_univtac.py` plus the video/tactile latent encoders |
| Franka | Official [`WorldArena/WorldArena2.0_Franka_FR3`](https://huggingface.co/datasets/WorldArena/WorldArena2.0_Franka_FR3) snapshot | `prepare_track32_franka_data.sh` downloads, converts, normalizes, and encodes |
| AgileX | Official [`WorldArena/WorldArena2.0`](https://huggingface.co/datasets/WorldArena/WorldArena2.0) Track 3 task directories | Official-layout raw reader -> public AgileX conversion APIs -> latent encoders -> `agilex-build-artifacts` |

WorldArena publishes both Franka and AgileX real-robot demonstrations for
post-training. The Franka route is fully automated. UniVTAC additionally needs
the frozen 40-episode validation declaration. The AgileX route includes a
pinned downloader, official-layout reader, audited converter, qpos14 normalizer,
and sharded RGB/tactile latent encoders. Dataset availability, preprocessing,
training, offline evaluation, and real-robot evaluation remain separate
completion gates.

The current Track 3 description links the unified `WorldArena/WorldArena2.0`
repository. The automated Franka pipeline consumes the dedicated snapshot in
the same official Hugging Face organization. These are related official data
surfaces, not interchangeable layouts.

### Tactile profiles

Training uses one explicit profile:

| Profile | Dataset contract | Tactile parameters |
|---|---|---|
| `vision_tactile` | Every selected repository contains its declared tactile streams | Tactile modules are used and trainable |
| `mixed` | Repository routes explicitly distinguish tactile and RGB-only data | Tactile modules are trainable; AgileX uses content-addressed `contact_cond_drop` while retaining valid tactile targets |
| `vision_only` | No repository supplies tactile streams | Tactile topology remains instantiated, but is frozen, bypassed, and excluded from AdamW |

`tactile_optional=True` is not a formal profile. Missing tactile payloads fail
closed instead of silently converting a tactile repository to RGB-only. The
editable mixed template is `n0_twam/configs/twam_mixed_cfg.py`.

### UniVTAC: download and preprocess

#### 1. Download the official eight-task snapshot

The pinned public snapshot contains 800 episodes, 100 for each task. Download
only the dataset payload; the upstream `checkpoints/` directory is not needed
for N0-TWAM post-training.

```bash
export N0_UNIVTAC_REVISION=172331dbbce95bc04c3e59b22f32dc72ba5561ae
export N0_RAW="$N0_WORK/raw/UniVTAC-$N0_UNIVTAC_REVISION"

mkdir -p "$N0_WORK/raw"
hf download byml/UniVTAC \
  --repo-type dataset \
  --revision "$N0_UNIVTAC_REVISION" \
  --exclude 'checkpoints/**' \
  --local-dir "$N0_RAW"

for task in grasp_classify insert_HDMI insert_hole insert_tube \
  lift_bottle lift_can pull_out_key put_bottle_in_shelf; do
  test -d "$N0_RAW/$task/clean"
done
```

The upstream project also provides `data/download.sh`; the command above pins
the exact Hugging Face revision consumed by this README.

#### 2. Freeze splits and convert to LeRobot

Provide an immutable 40-episode validation declaration and the one-episode
quarantine declaration. Both files are JSON lists whose entries contain
`hdf5_path` and `task`. The validation file must contain exactly five episodes
from each task. The quarantine file must contain exactly:

```json
[
  {
    "hdf5_path": "grasp_classify/clean/90.hdf5",
    "task": "grasp_classify"
  }
]
```

Do not create a new validation split after inspecting model results. Use the
frozen declaration supplied with the experiment/challenge bundle. The raw
episodes must contain head/wrist RGB, two tactile RGB streams, joint
observations, and timestamps. The complete schema is documented in
[TRACK31_UNIVTAC.md](docs/TRACK31_UNIVTAC.md).

```bash
# Keep N0_RAW from the download step above.
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

This audits all source HDF5 bytes, fits qpos8 q01/q99 statistics from training
data only, and materializes separate `train759` and `frozen40` repositories.

#### 3. Encode video and tactile latents

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

### Franka: download and preprocess

The single preparation command pins
`WorldArena/WorldArena2.0_Franka_FR3@aed59b39c5a903be5e435c13c0ed1efdd54d5ad9`,
verifies and downloads 600 episodes, converts both RGB streams and 8D
scalar-last XYZW `end_pose_base` labels, creates development/final views, fits
the normalizers, and encodes the video latents. The download is resumable.

```bash
export N0_FRANKA_WORK="$N0_WORK/track32-franka"
export N0_TRACK32_HF_ENDPOINT=https://huggingface.co

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
├── data/official/WorldArena2.0_Franka_FR3_aed59b39/
│   ├── clear_up/
│   ├── pour/
│   └── wipe/
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

### AgileX: download and preprocess

AgileX uses the native 14D absolute command
`[left_6j, left_gripper, right_6j, right_gripper]`; it does not reuse the
Franka pose mapping. The three supported profiles are `vision_tactile`,
`mixed`, and `vision_only`.

#### Preferred automated path

The HCU-oriented wrapper performs the pinned download, official-layout audit,
conversion, normalizer fitting, sharded RGB/tactile latent encoding, artifact
finalization, request construction, dry-run preflight, and a fresh 1500-step
training run:

```bash
export N0_TRACK32_HF_ENDPOINT=https://huggingface.co

./prepare_and_train_track32_agilex.sh \
  /absolute/path/to/agilex-work \
  /absolute/path/to/n0-twam-base \
  /absolute/path/to/init-checkpoint \
  8
```

All paths must be absolute and every lineage must use new request, run, log,
status, and output paths. The wrapper expects the validated HCU runtime and
`hy-smi`; it is not the portable NVIDIA launcher and it is not the two-node
Stage-B controller. Use the decomposed commands below when adapting the
pipeline to another accelerator or scheduler.

#### 1. Download the official Track 3 snapshot

WorldArena publishes AgileX real-robot demonstrations in
`WorldArena/WorldArena2.0`. The downloader pins revision
`af1ac34d3881f84096345542c631fbb1b9540d50`, inventories only the ten Track 3
task directories plus their prompt map, and verifies every downloaded byte.

```bash
export N0_AGILEX_WORK="$N0_WORK/track32-agilex"
export N0_AGILEX_RAW="$N0_AGILEX_WORK/data/official/WorldArena2.0_af1ac34"

python -m script.track3_2.download_agilex \
  --inventory "$N0_AGILEX_WORK/provenance/official_agilex_inventory.json" \
  --raw-root "$N0_AGILEX_RAW" \
  --receipt "$N0_AGILEX_WORK/provenance/agilex_download_receipt.json" \
  --endpoint https://huggingface.co \
  --workers 16
```

The seven AgileX vision-only tasks provide head and left/right wrist RGB. The
three vision-tactile tasks (`insert`, `peel_cucumber`, and
`pick_potato_chip`) additionally provide left/right tactile videos and tactile
mechanics data. Each episode publishes `episode.hdf5`; the formal action and
`observations/qpos` streams are 14D in canonical left-arm-then-right-arm
order. Most tasks contain about 100 demonstrations; `pour_over_coffee`
currently contains 83 valid demonstrations.

The official task table marks `clean_table`, `pour_water`, and `wipe_table` as
containing AgileX data plus Franka variants. A formal reader must identify the
embodiment from the episode metadata/schema and seal that choice in the source
manifest; directory names alone are not an embodiment label.

Do not substitute the Track 1 `dataset_track1.tar.gz` or a community RoboTwin
archive. They do not establish the Track 3 camera, tactile, wrench, and
commanded-qpos14 contract used by the real-robot policy.

#### 2. Audit, convert, and fit the normalizer

The official reader validates embodiment metadata, three canonical RGB streams,
optional tactile/wrench streams, same-row qpos14 labels, temporal alignment, and
stable sensor identities. It refuses a partial audit and requires all 983
episodes before publishing route-homogeneous LeRobot repositories, manifests,
receipts, and the q01/q99 action normalizer:

```bash
python -m script.track3_2.prepare_agilex \
  --raw-root "$N0_AGILEX_RAW" \
  --artifact-root "$N0_AGILEX_WORK/artifacts/agilex_mixed_v1" \
  --dataset-root "$N0_AGILEX_WORK/lerobot/agilex_mixed_v1"
```

See [TRACK32_AGILEX.md](docs/TRACK32_AGILEX.md) for the exact public reader,
route, action-label, and normalizer contracts.

#### 3. Encode latents for every converted repository

Use the frozen encoder identity and the AgileX sharded encoder. Each shard
validates the canonical RGB roster and encodes tactile latents only for signed
contact routes. Existing latent payloads are accepted only after their schema,
shape, episode/frame identity, and encoder identity pass validation.

```bash
export N0_AGILEX_LEROBOT="$N0_AGILEX_WORK/lerobot"
export N0_AGILEX_ARTIFACTS="$N0_AGILEX_WORK/artifacts/agilex_mixed_v1"
export N0_AGILEX_ENCODER_ID="$N0_AGILEX_ARTIFACTS/encoder_source_identity.json"

python -m script.track3_2.cache_encoder_identity \
  --model-path "$N0_BASE" \
  --output "$N0_AGILEX_ENCODER_ID"

# Submit this command once for every SHARD_INDEX in [0, NUM_SHARDS).
export NUM_SHARDS=8
export SHARD_INDEX=0
python -m script.track3_2.encode_agilex_latents \
  --dataset-root "$N0_AGILEX_LEROBOT/agilex_mixed_v1" \
  --model-path "$N0_BASE" \
  --encoder-source-identity "$N0_AGILEX_ENCODER_ID" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --device cuda:0

# Run only after all NUM_SHARDS workers have exited successfully.
python -m script.track3_2.finalize_agilex \
  --artifact-root "$N0_AGILEX_ARTIFACTS" \
  --dataset-root "$N0_AGILEX_LEROBOT/agilex_mixed_v1"
```

Run every shard successfully before finalization. Submit shards through the
cluster scheduler when device ownership is managed externally. Never launch
latent workers onto devices already used by training or another user.

Finalization and `agilex-build-artifacts` re-hash the current LeRobot tables and
every latent payload; a historical receipt cannot hide changed dataset bytes.

## Track 3.1

| Item | Value |
|---|---|
| Dataset | UniVTAC `train759` |
| Default profile | `vision_tactile` |
| Action target | 8D `qpos8_next_step` |
| Default recipe | 1500 optimizer steps, save every 300, validate every 100 |
| Entry point | `n0-twam track31` or `./run_track31.sh` |

Create a request, replace its placeholders with the prepared artifact/model
identities and new output paths, then run preflight and training:

```bash
n0-twam track31 template train \
  --output "$N0_WORK/requests/track31-development.json"
python -m json.tool "$N0_WORK/requests/track31-development.json" >/dev/null
n0-twam track31 train \
  --config "$N0_WORK/requests/track31-development.json" \
  --dry-run
n0-twam track31 train \
  --config "$N0_WORK/requests/track31-development.json" \
  > "$N0_WORK/runs/track31-development.result.json"
```

Strict resume must retain world size, profile, role, recipe, data, code, and
runtime identities. `final_refit` starts again from the released base with all
759 training episodes; it does not resume the 719-episode development optimizer.
After checkpoint selection, evaluate only the sealed `frozen40` view:

```bash
n0-twam track31 template eval \
  --output "$N0_WORK/requests/target10.eval.json"
n0-twam track31 eval \
  --request "$N0_WORK/requests/target10.eval.json"
```

Bind the checkpoint, view, normalizer, metric, and golden-calibration identities
before evaluation. The default Target-10 reports tactile PSNR/SSIM and is not an
organizer hidden-test or simulator-success result. See
[TRACK31_QUICKSTART.md](docs/TRACK31_QUICKSTART.md) for the full request schema.

## Track 3.2

### Training contract

| Item | Value |
|---|---|
| Dataset | Official Franka release or manifest-bound AgileX dual-arm data |
| Profiles | Franka: `vision_only`; AgileX: `vision_tactile`, `mixed`, `vision_only` |
| Franka action | `[x, y, z, qx, qy, qz, qw, gripper]` -> EE10 -> EE20 mask |
| AgileX action | Native 14D absolute dual-arm qpos (`6+1` per arm) |
| Default recipe | 1500 optimizer steps, save every 300, validate every 100 |
| Entry point | `n0-twam track32` (`./run_track32_franka.sh` for Franka) |

### Franka XYZW contract revision

The Franka wire command is scalar-last `xyzw`; it is converted geometrically
from pose8 to EE10 and embedded into channels `0..9` of the released EE20 head.
The frozen identities are `franka_end_pose_base_xyzw8_v2`,
`franka_ee10_rot6d_columns_from_xyzw_v2`, and
`franka_track32_vision_only_xyzw_v2`.

This is incompatible with the former WXYZ/v1 lineage. Converted repositories,
normalizers, checkpoints, prediction artifacts, serve bundles, and replay
receipts must be regenerated, not resumed or relabeled. The old participant
bridge patch is removed; the worker audits a clean pinned WorldArena checkout
with a non-identity XYZW probe in both directions before loading the Policy.

Franka remains `vision_only`: tactile parameters stay frozen; `joint_qpos` is observation-only.

### AgileX: build artifacts and train

The public AgileX entry points form one reusable sequence. Start by creating
new artifact, request, run, and serving directories outside the source tree:

```bash
export N0_AGILEX_WORK="$N0_WORK/track32-agilex"
mkdir -p "$N0_AGILEX_WORK/artifacts" \
  "$N0_AGILEX_WORK/requests" "$N0_AGILEX_WORK/runs"

# Recompute immutable conversion and latent identities from the live converted
# LeRobot tree. See docs/TRACK32_AGILEX.md for every required hash argument.
n0-twam track32 agilex-build-artifacts --help

n0-twam track32 agilex-template \
  --output "$N0_AGILEX_WORK/requests/agilex-development.json"
```

Replace every placeholder in the generated request. Select exactly one of
`vision_tactile`, `mixed`, or `vision_only`; bind the frozen source, route,
temporal, normalizer, conversion, latent, released-base, and `empty_emb.pt`
identities; and use a new output root. Then execute the same entry point first
as a dry run and then as the real run:

```bash
python -m json.tool \
  "$N0_AGILEX_WORK/requests/agilex-development.json" >/dev/null

n0-twam track32 agilex-train \
  --config "$N0_AGILEX_WORK/requests/agilex-development.json" \
  --dry-run

n0-twam track32 agilex-train \
  --config "$N0_AGILEX_WORK/requests/agilex-development.json" \
  | tee "$N0_AGILEX_WORK/agilex-development.result.json"
```

A fresh run migrates the released 20D checkpoint by reusing compatible
non-action weights and deterministically reinitializing the complete 14D action
projection. Subsequent qpos14 resumes are strict: profile, world size, data
identities, `run_role`, seed, save interval, and validation interval must remain
unchanged. Formal `mixed` training currently requires `batch_size=1` so every
batch has an unambiguous contact-conditioning contract.

AgileX supports three deliberately different initialization lineages:

| Lineage | Restored state | Required interpretation |
|---|---|---|
| Released 20D -> qpos14 migration | Compatible non-action transformer weights; new qpos14 action projections | Fresh optimizer, scheduler, RNG, and data cursor |
| Strict qpos14 resume | Transformer, optimizer, scheduler, RNG, data cursor, and the complete signed recipe | Same experiment; recipe, world size, environment, and data identities cannot drift |
| qpos14 Stage-B weights-only initialization | Audited qpos14 transformer weights and inherited migration lineage | New experiment and new optimizer state; cumulative step labels are bookkeeping, not a strict continuous resume |

The official mixed preparation contains all 983 public AgileX episodes. The
current `final_refit` request builder trains on that complete corpus and points
its periodic validation loader at the same converted root. Those validation
losses are useful for numerical-regression monitoring but have 100% episode
overlap with training and cannot select a generalizing checkpoint. Establish a
frozen development/holdout split before training, or use genuinely new
episodes/robot trials, when checkpoint selection or generalization is the goal.

After a complete checkpoint, use the remaining public entry points in order:

```bash
n0-twam track32 agilex-serve-bundle --help

n0-twam track32 agilex-policy-template \
  --output "$N0_AGILEX_WORK/agilex.policy.json"

# Fill every bundle, route, normalizer, task-route, and calibrated safety hash.
n0-twam track32 agilex-policy-check \
  --config "$N0_AGILEX_WORK/agilex.policy.json"
```

The complete `agilex-build-artifacts` and `agilex-serve-bundle` commands are in
[TRACK32_AGILEX.md](docs/TRACK32_AGILEX.md). `agilex-policy-check` is
allocation-free; it verifies identities but does not load the model or contact
a robot.

Run the complete AgileX offline engineering evaluation after the Policy config
is verified:

```bash
n0-twam track32 agilex-offline-eval \
  --train-request "$N0_AGILEX_WORK/requests/agilex-development.json" \
  --config "$N0_AGILEX_WORK/agilex.policy.json" \
  --dataset-root "$N0_AGILEX_WORK/data/lerobot" \
  --output-root "$N0_AGILEX_WORK/evaluations/agilex-offline-v1" \
  --device 0 --samples-per-task 1 --replay-steps 7
```

This single command freezes a ten-task proxy roster, generates future RGB and
qpos14 predictions, reports PSNR/SSIM and qpos14 MAE/RMSE, and runs Policy
safety/latency replay. It uses training-distribution samples and therefore is
not an independent holdout, real-robot success rate, or organizer leaderboard
result. See [TRACK32_AGILEX.md](docs/TRACK32_AGILEX.md#offline-evaluation) for
the artifact contract and interpretation.

### Franka: build and validate the request

Create a new development request from the prepared artifact and LeRobot roots.
The full argument contract is documented in
[TRACK32_FRANKA.md](docs/TRACK32_FRANKA.md):

```bash
mkdir -p "$N0_FRANKA_WORK/requests" "$N0_FRANKA_WORK/runs"
n0-twam track32 build-request --help
n0-twam track32 train \
  --config "$N0_FRANKA_WORK/requests/franka-development.json" \
  --dry-run
./run_track32_franka.sh \
  "$N0_FRANKA_WORK/requests/franka-development.json" \
  | tee "$N0_FRANKA_WORK/franka-development.result.json"
```

For six-node Franka HCU training, use the fail-closed workflow in
[TRACK32_FRANKA_HSDP.md](docs/TRACK32_FRANKA_HSDP.md); it binds the 6x8 HSDP
mesh, SHCA/RDMA transport and checkpoint receipt. `final_refit` starts from the released base.

Seal a complete checkpoint, create and calibrate the Policy config, generate
predictions for the frozen validation view, and score them in that order:

```bash
n0-twam track32 serve-bundle --help
n0-twam track32 policy-template --output "$N0_FRANKA_WORK/policy.json"
n0-twam track32 generate-predictions --help
n0-twam track32 score-predictions --help
```

The offline report contains future-RGB PSNR/SSIM and end-position MAE/RMSE.
Before connecting to a robot, replay the real Policy backend on a frozen,
non-pickled observation and require enough refills to evaluate latency:

```bash
n0-twam track32 policy-replay \
  --config "$N0_FRANKA_WORK/policy.json" \
  --observation "$N0_FRANKA_WORK/offline-observation.npz" \
  --prompt "clear the table" \
  --steps 31 \
  --control-hz 15 \
  --minimum-refill-samples 2 \
  --require-realtime \
  --output "$N0_FRANKA_WORK/offline-policy-replay.json"
```

Only after organizer approval, robot-cell calibration, credentials, and the
launcher's clean-checkout XYZW bridge audit should the worker be started:

```bash
export WORLD_ARENA_ROOT=/absolute/path/to/WorldArena-2.0
export N0_TRACK32_POLICY_CONFIG="$N0_FRANKA_WORK/policy.json"
export HUB_POLICY_URL=https://ORGANIZER_GATEWAY/policy
export POLICY_ID=ORGANIZER_ASSIGNED_WORKER_KEY
bash run_track32_franka_worker.sh
```

Never put the optional bearer token in a committed file. Offline metrics and
Policy replay remain engineering proxies, not a real-robot success rate.

## Runtime and inference semantics

### Synchronous closed loop and re-grounding

The AgileX and Franka Policy paths both support synchronous action-chunk
execution with rolling cache re-grounding. A generated chunk is safety-projected
against the latest measured state, executed actions and post-action observations
are recorded, and the completed history is committed with
`compute_kv_cache=True` before the next chunk is generated. Predicted cache
entries are cleared while previously committed real observations are preserved.

AgileX internally generates 12 qpos14 actions and returns one `[1,14]` action per
Policy call; contact routes also commit their tactile and wrench history. Franka
uses its EE20 internal representation and commits only its vision/action history.
Neither Policy overlaps model inference with robot execution. A synchronous
Policy replay therefore verifies protocol, cache, and safety behavior, not an
asynchronous realtime controller.

### Action-denoise KV reuse

Both server configs expose the strict runtime switch:

```bash
export N0_ACTION_DENOISE_KV_REUSE=1  # enabled; use 0 for the full reference path
```

The value must be exactly `0` or `1` and is included in the content-addressed
server runtime contract. The cached path preserves a causal fixed-context mask,
runs intermediate action-denoise steps from the prepared KV state, and uses a
full terminal pass for predicted-cache publication. Use
`script/track3_2/benchmark_agilex_action_kv.py` and
`script/track3_2/compare_action_kv_benchmarks.py` for paired output-parity and
latency receipts. Do not claim a speedup from the feature flag alone; publish
accelerator-specific p50/p95 results and action parity from isolated OFF/ON
processes.

### Latest-only realtime sensor core

`n0_twam.integrations.worldarena.realtime_sensors` provides a transport-neutral
sensor core shared by AgileX and Franka:

- one acquisition thread per RGB, tactile, qpos/pose, or wrench source;
- bounded drop-old rings and a single latest-overwrite coherent snapshot slot;
- immutable payloads with capture timestamps, sequence IDs, and generations;
- skew, age, reconnect-generation, planner-deadline, and stale-result gates;
- strict bridges to the existing AgileX and Franka Policy observation schemas.

The core prevents a slow planner from consuming an unbounded FIFO of historical
frames. Hardware SDK/ROS sources, calibrated clock correlation, generation-bound
Policy/cache reset, robot action arbitration, and supervised physical-camera
tests remain deployment integrations. Unit tests with simulated 10-Hz input and
multi-second fake inference are software evidence only.

## Development and verification

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --editable '.[dev,track31,track32]'
uv pip install --no-deps 'lerobot==0.3.3'
pytest -q tests/unit/test_agilex_policy.py
pytest -q tests/unit tests/integration tests/regression
black --check n0_twam script tests
flake8 n0_twam script tests
mypy --strict --follow-imports=skip path/to/changed_module.py
git diff --check
```

Also run `bash -n` for changed launchers. Synthetic tests do not replace an
accelerator smoke: verify preflight, a finite update, save/reload, validation,
checkpoint sidecars, shutdown, and the applicable offline receipt.

## Troubleshooting

| Symptom | Check first |
|---|---|
| Preflight fails before accelerator allocation | Read the generated preflight log and verify every path, file SHA-256, action schema, profile, run role, and source/runtime identity |
| Strict resume rejects the request | Restore the exact world size, recipe, environment, and data identities, or start a new weights-only lineage; never relabel it as strict resume |
| HCU collective initialization fails | Use the validated vendor image, confirm the requested network interface and SHCA/UCX/GDR transport, and reject Socket fallback |
| Offline metrics look good but Policy misses its deadline | Treat prediction quality and systems latency as separate gates; inspect cold generation, cache refill, and action-denoise timing receipts |
| Sensor input becomes historical | Use bounded rings plus the latest snapshot slot; do not feed the Policy from an unbounded FIFO or synthesize capture timestamps at inference time |

## Security and deployment safety

- Never commit organizer tokens, robot credentials, passwords, private keys,
  internal endpoints, or calibrated cell secrets. Pass credentials through the
  runtime environment or an external secret manager.
- Treat manifests, requests, normalizers, Policy configs, and bundles as
  untrusted until schema/hash preflight passes. Offline safety envelopes are not
  robot calibrations; deployment requires cell limits, independent safety,
  operator supervision, and organizer approval.

## Contributing

Open an issue describing the route, data contract, and intended evidence tier;
use a focused branch; keep data, weights, checkpoints, credentials, and local
artifacts out of Git; add regression tests for contract changes; and run focused
plus adjacent checks. Use a Conventional Commit pull-request title such as
`feat(track32): ...`, `fix(runtime): ...`, or `docs(readme): ...`.

By contributing, you agree that your changes are distributed under the project
license. Third-party code or assets must retain their original license and
notice and must be compatible with this repository's distribution terms.

## Citation

If this repository is useful in academic work, cite the N0-TWAM paper:

```bibtex
@misc{n0twam2026,
  title={$N_0$-TWAM: Scaling Tactile-Native World-Action Model for Contact-Rich Manipulation},
  author={NeoteAI Team and Fudan TEAI Team},
  year={2026},
  eprint={2607.23783},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.23783}
}
```

## Acknowledgments

Built on [N0-TWAM](https://github.com/neoteai/N0-TWAM), [WorldArena 2.0](https://github.com/WorldArena2/WorldArena-2.0), [Wan2.2](https://github.com/Wan-Video/Wan2.2), and [LeRobot](https://github.com/huggingface/lerobot).

## License

Released under [CC BY-NC-SA 4.0](LICENSE), a non-commercial share-alike license
that is not OSI-approved. Review it before redistribution or commercial use.
Third-party data, models, libraries, and code retain their own terms.
