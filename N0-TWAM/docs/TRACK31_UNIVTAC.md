# N0-TWAM × UniVTAC Track 3.1

> New users: start with the portable one-command interface in
> [TRACK31_QUICKSTART.md](TRACK31_QUICKSTART.md). This document also preserves
> the detailed formal HCU/operator protocol and therefore contains site-shaped
> example paths.

This path trains N0-TWAM end to end with native 8D Panda joint-position actions.
ACT is a comparison/reference implementation only; it is not imported by the
training, checkpoint, dataset, or evaluation path described here.

## 1. Prepare immutable data artifacts

```bash
python script/track3_1/prepare_univtac.py \
  --data-root /absolute/path/to/workspace/UniVTAC \
  --validation-manifest /absolute/path/to/workspace/UniVTAC/validation.json \
  --quarantine-manifest /absolute/path/to/workspace/UniVTAC/quarantine.json \
  --artifact-dir /absolute/path/to/workspace/n0_track31/artifacts \
  --materialize-root /absolute/path/to/workspace/n0_track31/lerobot \
  --repo-id univtac_track31 \
  --emit-standard-views
```

The formal command audits all eight tasks, freezes source SHA-256 values, keeps
the 759 training episodes disjoint from the 40 frozen validation episodes,
computes q01/q99 only from training actions, and materializes separate LeRobot
v2.1 repositories named `train759` and `frozen40`.

If the schema-v4 artifact bundle is already frozen but LeRobot conversion has
not run, use the materialize-only transaction instead of repeating the source
audit and normalizer pass:

```bash
python script/track3_1/materialize_univtac.py \
  --artifact-dir /absolute/path/to/workspace/n0_track31/artifacts \
  --target-root /absolute/path/to/workspace/n0_track31/lerobot \
  --repo-id univtac_track31
```

The target must not exist. The command loads `universe_manifest_v4.json`
without a duplicate preflight source-hash pass, then rechecks each HDF5 SHA-256
immediately before converting that episode. It builds both physical repos in a
sibling incomplete directory, verifies their full conversion inventories,
atomically renames the verified directory, verifies the final-root identities,
and only then publishes `conversion_report.json`. On failure, the previous
report remains unchanged and the incomplete directory retains
`.materialization_state.json` for diagnosis.

The transaction records the raw baseline-report hash, raw candidate hash, and
logical conversion-report hash before committing the target. If a process is
killed after the directory rename, the next invocation only recovers a target
whose marker matches the exact target, staging, and manifest identities. It
finishes the atomic report CAS when the recorded candidate and baseline still
match, recognizes an already committed report and restores the complete marker,
or rolls an uncommitted target back to its original staging path without
clobbering another report. A committed report is never followed by target
rollback. A persistent complete-marker write failure is reported as degraded;
the target and report remain in place for a later recovery invocation.

Materialization is supported on Darwin with `renamex_np` and glibc Linux with
`renameat2`; the command probes no-replace and exchange support on the actual
target filesystem before conversion. JSON files and affected parent-directory
entries are fsynced. The large LeRobot payload tree is not recursively fsynced,
so this is not a full power-loss durability guarantee. The artifact lock
serializes cooperating prepare/materialize writers; it does not claim protection
against an arbitrary process that ignores the lock and mutates the dataset tree.

`--emit-standard-views` is mandatory on the **prepare CLI** for this formal
protocol. The materialize-only CLI always creates the standard
`train759`/`frozen40` repositories and therefore has no such flag. Without the
flag, the prepare CLI intentionally retains the legacy two-task
`train`/`validation` behavior for backward compatibility. The quarantine manifest must identify the known
43-frame `grasp_classify/clean/90.hdf5`; quarantine data is never materialized.

UniVTAC's HDF5 `step` is a raw simulator counter sampled by the collector, not
a dense frame index. The bridge requires it to be non-negative and strictly
increasing, retains it as provenance, and defines `qpos8_next_step` as recorded
row `t -> t+1`. Video timestamps use recorded row index / configured FPS.

There is one content-addressed training-only exception:
`lift_can/clean/62.hdf5` (SHA-256
`335f6c32947b1d37acb638e67a756b7c37204df4c849121822234c7931fd63f2`)
contains a reset at raw row `227 -> 228` (`step 594 -> 586`). Its pinned
`pinned_maximal_monotonic_prefix_v1` policy retains raw rows `[0, 228)` and
therefore exports exactly 227 next-step rows. The reset pair and the remaining
28-row suffix do not enter normalization, conversion, latent encoding, or
sampling. Path, full-file hash, raw length, boundary, step values, usable range,
and policy are canonical manifest fields; any mismatch fails closed. Frozen
evaluation episodes must use full strictly monotonic timelines. This preserves
759 source episodes and one LeRobot/sample identity per source episode.
Every converted Parquet row stores both `source.row_index` (the exact raw HDF5
row) and `source.frame_index` (the simulator `step`). Preflight requires the
former to equal the declared usable row range exactly and the latter to remain
strictly increasing, so a shifted or whole-episode table cannot pass by merely
rewriting report hashes.

Released UniVTAC files use the explicit legacy image contract
`opencv_imencode_rgb_input_v1`: simulator RGB arrays were passed directly to
OpenCV's JPEG encoder, so decode passthrough restores their numeric RGB order.
The manifest records this contract and `output_color_space=RGB`; unknown or
mixed color contracts fail closed.

## 2. Encode N0 video and tactile latents

Before training, precompute the complete prompt set once with umT5 on HCU, then
run both encoders for `train759` only. Formal video workers do not load their own
umT5 replicas:

```bash
python script/track3_1/precompute_hcu_prompt_cache.py \
  --dataset-root /absolute/path/to/workspace/n0_track31/lerobot/train759 \
  --model-path /absolute/path/to/workspace/models/n0-base \
  --artifact-root /absolute/path/to/workspace/n0_track31/artifacts \
  --split train759 \
  --encoder-source-identity-path /path/to/encoder_source_identity.json \
  --output-path /path/to/prompt_embedding_cache.pt \
  --device cuda:0

python script/encode_lerobot_n0_latents.py \
  --dataset-root /absolute/path/to/workspace/n0_track31/lerobot/train759 \
  --model-path /absolute/path/to/workspace/models/n0-base \
  --artifact-root /absolute/path/to/workspace/n0_track31/artifacts \
  --split train759 \
  --prompt-embedding-cache-path /path/to/prompt_embedding_cache.pt

python script/encode_tactile_latent.py \
  --dataset-root /absolute/path/to/workspace/n0_track31/lerobot/train759 \
  --model-path /absolute/path/to/workspace/models/n0-base \
  --mode both \
  --local-mode current \
  --tactile-keys observation.images.tactile_a observation.images.tactile_b \
  --artifact-root /absolute/path/to/workspace/n0_track31/artifacts \
  --split train759
```

Do **not** mount or encode `frozen40` during recipe development, Stage A
training, checkpoint selection, or optional Stage B training. Keep that physical
repository sealed until the route and evaluation checkpoint are frozen. At the
evaluation phase, encode `frozen40` in a dedicated job with only the selected
checkpoint-independent encoder assets and the sealed repository mounted
read-only; then finalize its inventories before prediction generation. This
prevents validation exposure while retaining the same versioned encoder
contract for evaluation.

In formal mode, the encoder reads `universe_manifest_v4.json` and
`conversion_report.json`,
requires the dataset directory name to equal the physical split, and verifies
only that selected physical repository. It does not read the legacy
`dataset_manifest.json`/`qpos8_normalizer.json` pair or require the sibling repo
to be mounted. The explicit `train`/`validation` split values remain supported
only for old artifacts. Each encoder invalidates its old ready marker before
writing and only publishes a schema-v3 content-addressed inventory whose
`split` is the physical repo name after every expected episode/segment and
stream has the required shape, temporal metadata, size, SHA-256, and exact
canonical encoding contract. Every formal video `.pth` carries provenance-v3
(tactile uses its independently versioned provenance-v2)
binding the schema-v4 manifest digest, conversion-report digest, exact source
MP4 path/size/SHA-256, complete encoder-model identity, stream/mode, frozen task
prompt, encoding contract, code schema, and actual execution device/type.
Finalization recomputes those external identities and requires the exact full
sampled frame range implied by each physical episode; two modalities with the
same self-consistent but shortened range still fail. Old schema-v2 inventories
and old payloads without provenance are not migrated in place. Re-run the
affected shard with `--overwrite`, then run the single finalizer. Preflight
verifies both inventories again; empty or partial latent directories are
rejected. Do not recompute normalization during training.

`code_schema_version=3` is currently the formal encoder implementation identity;
it is a reviewed versioned contract, not a cryptographic hash of the Python
source tree. Any code change affecting output or verification semantics---such
as decode/sampling, resizing, residual construction, prompt encoding, VAE
normalization, flattening, payload publication, or finalizer interpretation---
must bump this schema and the relevant encoding contract, then fully re-encode
formal payloads. This deliberate trust boundary assumes reviewed code follows
the bump-and-re-encode rule; without a bump, byte-identical external inputs
could otherwise make an old payload appear reusable.

For a fresh multi-HCU encode, do not split the formal repositories with manual
`--episodes` lists. Launch deterministic modulo workers instead. Each worker
must receive a unique global shard index and must defer the ready inventory:

```bash
# Official vendor HCU order is fixed; each process exposes one physical HCU and
# therefore always addresses it as cuda:0 inside the process.
HCU_ORDER=(0 1 5 4 2 3 7 6)
# Use eight workers per node across eight nodes. GLOBAL_SHARD is therefore
# NODE_RANK * 8 + LOCAL_WORKER and LOCAL_WORKER is in [0, 8).
HCU_ID="${HCU_ORDER[$LOCAL_WORKER]}"
export HIP_VISIBLE_DEVICES="$HCU_ID"
export ROCR_VISIBLE_DEVICES="$HCU_ID"
export CUDA_VISIBLE_DEVICES="$HCU_ID"
GLOBAL_SHARD=$((NODE_RANK * 8 + LOCAL_WORKER))

/usr/bin/python3 script/encode_lerobot_n0_latents.py \
  --dataset-root /absolute/path/to/workspace/n0_track31/lerobot/train759 \
  --model-path /absolute/path/to/workspace/models/n0-base \
  --artifact-root /absolute/path/to/workspace/n0_track31/artifacts \
  --split train759 \
  --num-shards 64 \
  --shard-index "$GLOBAL_SHARD" \
  --defer-inventory \
  --device cuda:0 \
  --text-encoder-device cuda:0 \
  --prompt-embedding-cache-path /path/to/prompt_embedding_cache.pt \
  --dtype bf16 \
  --target-fps 10 \
  --height 256 \
  --width 256 \
  --max-sequence-length 512

/usr/bin/python3 script/encode_tactile_latent.py \
  --dataset-root /absolute/path/to/workspace/n0_track31/lerobot/train759 \
  --model-path /absolute/path/to/workspace/models/n0-base \
  --artifact-root /absolute/path/to/workspace/n0_track31/artifacts \
  --split train759 \
  --tactile-keys observation.images.tactile_a observation.images.tactile_b \
  --mode both \
  --local-mode current \
  --num-shards 64 \
  --shard-index "$GLOBAL_SHARD" \
  --defer-inventory \
  --device cuda:0 \
  --dtype bf16 \
  --target-fps 10 \
  --height 128 \
  --width 128
```

Do not pass the physical `HCU_ID` to `--device`: after
`HIP_VISIBLE_DEVICES="$HCU_ID"`, the only visible accelerator is `cuda:0`.
Requesting `cuda`/`cuda:0` without an available accelerator is a hard error;
there is no silent CPU fallback. Formal prompt precompute requires umT5 on its
single visible HCU (`cuda:0`); formal video and tactile workers require the Wan
VAE on their respective single visible HCU. The versioned prompt cache is
read-only and is checked against the full frozen prompt set, manifest,
conversion report, encoder identity, and encoding contract before any video
payload is written. CPU model execution is rejected; tokenizer and filesystem
I/O remain host-side. Legacy non-formal encoding may still request CPU
explicitly.

Formal umT5 loading uses `low_cpu_mem_usage=True` with an exact root
`device_map` to `cuda:0`. This materializes model parameters directly on the
dedicated precompute HCU and avoids both the unsupported full CPU-model then
`.to(cuda:0)` peak and 64 redundant umT5 replicas. After one inference-only
batch over at most eight unique frozen prompts, umT5 is released and the 64 HCU
workers consume the shared CPU-resident embedding payload only as VAE input.

The 759 episodes divide into 55 shards of 12 and 9 shards of 11.
Workers publish only UUID-namespaced atomic `.pth` payloads; no worker may
publish `status=ready`. Run video and tactile as separate phases on each HCU,
and do not start the finalizer until every worker in both phases has exited
zero. Independent encoding workers do not use NCCL or `torchrun`.

The formal launcher computes one encoder identity receipt per node, compares all
eight receipts, and uses an invocation-scoped prompt cache created by node 0
before its canary workers start. After the canary produces one payload per HCU,
the launcher expands to all 64 modulo shards. Keep both `--num-shards` and
`GLOBAL_SHARD` consistent; do not bypass the cache/identity gates to reduce
startup time.

After the cluster-wide barrier, one process on the shared filesystem verifies
the schema-v4 physical bundle and the complete VAE/tokenizer/text-encoder byte
identity, rebuilds both inventories from every payload, and compares their
temporal grids:

```bash
/usr/bin/python3 script/track3_1/finalize_track31_latents.py \
  --dataset-root /absolute/path/to/workspace/n0_track31/lerobot/train759 \
  --model-path /absolute/path/to/workspace/models/n0-base \
  --artifact-root /absolute/path/to/workspace/n0_track31/artifacts \
  --split train759 \
  --kind both \
  --validate-pair
```

At evaluation time, repeat encoding and finalization for the still-sealed
`frozen40` repository only after the route and evaluation checkpoint are locked.
`--kind video` or `--kind tactile` is available for diagnostics;
`--validate-pair` deliberately requires `--kind both`. A missing, unexpected,
stale, mismatched, or temporally
misaligned payload leaves all safely unlinkable requested ready markers absent,
including marker symlinks (the symlink itself is removed without following its
target). An unsafe non-regular marker such as a directory causes explicit
failure while cleanup continues for the other requested kind. Launch and
training orchestration must therefore accept readiness only after full
inventory content/digest validation; `Path.exists()` alone is never a readiness
signal.

## 3. Preflight and train

```bash
export N0_TRACK31_ARTIFACT_ROOT=/absolute/path/to/workspace/n0_track31/artifacts
export N0_TRACK31_LEROBOT_ROOT=/absolute/path/to/workspace/n0_track31/lerobot
export N0_BASE_MODEL=/absolute/path/to/workspace/models/n0-base
export N0_RELEASED_CHECKPOINT=/absolute/path/to/workspace/models/n0-twam-release
# Lowercase SHA-256 recorded while verifying the immutable released bundle.
export N0_RELEASED_TRANSFORMER_SHA256="REPLACE_WITH_VERIFIED_SHA256"
export N0_EMPTY_EMBEDDING=/absolute/path/to/workspace/models/n0-base/empty_emb.pt
export N0_TRACK31_SAVE_ROOT=/absolute/path/to/workspace/n0_track31/runs
export N0_TRACK31_BATCH_SIZE=1
export N0_TRACK31_GRADIENT_ACCUMULATION_STEPS=1
# Optional on vendor Torch images: an isolated --no-deps dependency overlay.
export N0_TRACK31_PYTHON_OVERLAY=/absolute/path/to/workspace/n0_track31/python_overlay

NGPU=1 bash run_track31_univtac.sh
```

The launcher exposes two immutable profiles and two run roles:

| profile / role | train view | validation view | normalizer |
|---|---|---|---|
| `multitask_pretrain_v1 / development` | `stage_a_dev719_v1` | `internal_dev40_v1` | `qpos8_dev719_v1` |
| `multitask_pretrain_v1 / final_refit` | `stage_a_final759_v1` | none | `qpos8_final759_v1` |
| `target_finetune_v1 / development` | `stage_b_dev180_v1` | `internal_target_dev10_v1` | inherits Stage A dev |
| `target_finetune_v1 / final_refit` | `stage_b_final190_v1` | none | inherits Stage A final |

Stage A is mandatory. Stage B is optional and starts from a same-role Stage A
checkpoint with model weights only; optimizer, scheduler, RNG, and data cursor
are fresh. For example:

```bash
export N0_TRACK31_TRAIN_PROFILE=target_finetune_v1
export N0_TRACK31_RUN_ROLE=final_refit
export N0_TRACK31_INIT_FROM=/path/to/stage_a_final/checkpoint_step_N
NGPU=1 bash run_track31_univtac.sh
```

`pad_global` never drops a training episode. It deterministically repeats only
the minimum tail needed to fill the last distributed global batch and writes an
exposure report. With 16 ranks and local batch size 1, Stage A final exposes all
759 unique episodes and pads the epoch to 768 samples; the repeated identities
are auditable and are not counted as new data.

Development validation uses the declared `internal_dev40_v1` or
`internal_target_dev10_v1` view, never the active training view. When the world
size exceeds the validation set, every rank still executes the same number of
FSDP forwards, but deterministic padding is masked out of a global sum/count
aggregation. Per-sample validation RNG derives from immutable sample identity,
so the same development sample receives the same noise on 32 or 48 ranks.

Never run `pip install .`, `pip install -r requirements.txt`, or `uv sync` in a
vendor HCU container: the repository pins upstream Torch and torchvision. Use a
content-addressed `--target` overlay with `--no-deps`, then assert the vendor
Torch versions are unchanged. The HCU image used during development required
`diffusers==0.36.0` for the released Wan VAE config and `lerobot==0.3.3` plus
`datasets==3.6.0` for conversion.

The first checkpoint load uses explicit `migrate_action`: every non-action tensor
must match exactly; the full 20D action input/output projection is reset under the
recorded seed for the 8D qpos schema. No action-head slicing is allowed. Preflight
parses the safetensors header, validates the MoT/action sentinel keys and action
shapes, and requires its streamed SHA-256 to equal
`N0_RELEASED_TRANSFORMER_SHA256`.

Track 3.1 deliberately preserves the released model capacity
`max_tactile_streams=4`. UniVTAC activates only sensor IDs `0` and `1`; reducing
the embedding to two rows would introduce a separate non-action weight migration.
Preflight therefore checks the four-row model contract, while every generated
`train_meta.json` records both `max_tactile_streams=4` and
`active_tactile_sensor_count=2` (plus the exact ID map). The saved transformer
config and training metadata also bind `patch_size`, tactile input/token fields,
and `snr_shift`, so evaluation cannot silently use a different tactile scheduler
or model shape.

To resume a Track 3.1 run, set the generated checkpoint root instead of the
released prior:

```bash
export N0_TRACK31_RESUME_FROM=/path/to/checkpoint_step_500
NGPU=1 bash run_track31_univtac.sh
```

Strict resume requires the transformer, optimizer, scheduler, training progress,
and per-rank RNG files. It also requires the same world size and gradient
accumulation setting, and never resets the trained qpos8 action head. The saved
transformer byte identity is bound independently into `training_state.json` and
`checkpoint_complete.json`; preflight recomputes it before accepting a resume.

### Historical 48-rank development recipe

Keep `N0_TRACK31_NUM_STEPS` fixed at the final scheduler horizon across every
stage. Bound one launcher invocation with an absolute global optimizer step:

The following six-node recipe records the retired 48-rank development run. It
is retained only as a historical record, is not accepted by the current formal
launcher, and must not be copied into a new run. For that historical run, keep
the scheduler horizon at 5000
optimizer steps, use six nodes with eight ranks per node, local batch size 2,
gradient accumulation 4 (effective global batch 384), save every 500 steps, and
validate every 100 steps. First prove an exact strict-resume boundary at
20 -> 25 before continuing the same trajectory:

```bash
export N0_TRACK31_NUM_STEPS=5000
export N0_TRACK31_BATCH_SIZE=2
export N0_TRACK31_GRADIENT_ACCUMULATION_STEPS=4
export N0_TRACK31_SAVE_INTERVAL=500
export N0_TRACK31_VAL_INTERVAL=100
export N0_TRACK31_STOP_AFTER_STEP=20
NNODES=6 NGPU=8 bash run_track31_univtac.sh

export N0_TRACK31_RESUME_FROM=/path/to/checkpoint_step_20
export N0_TRACK31_STOP_AFTER_STEP=25
NNODES=6 NGPU=8 bash run_track31_univtac.sh

export N0_TRACK31_RESUME_FROM=/path/to/checkpoint_step_25
unset N0_TRACK31_STOP_AFTER_STEP
NNODES=6 NGPU=8 bash run_track31_univtac.sh
```

Each node uses the same environment and a distinct `NODE_RANK=0..5`. The first
invocation starts from the audited released checkpoint; the second and third
invocations are strict full-state resumes. The development route uses
`stage_a_dev719_v1` plus `internal_dev40_v1`. After recipe lock, run an
independent `final_refit` from the released prior on `stage_a_final759_v1`
without a validation loader; do not continue it from the development checkpoint.

`N0_TRACK31_STOP_AFTER_STEP` is an invocation-only contract and is deliberately
absent from schema6 checkpoints. It may change between stages, while the saved
and current execution contracts—including `N0_TRACK31_NUM_STEPS`—must still
match exactly. Preflight reports `start_step`, `stop_after_step`, `num_steps`,
and the number of optimizer steps in the invocation, and rejects anything that
does not satisfy `start_step < stop_after_step <= num_steps`.
After distributed initialization, every rank collectively verifies the same
three step boundaries and fails fast on launcher-environment drift.

Stopping occurs only after a complete optimizer step. The stop point always
publishes a strict-resume checkpoint; if it also lands on `save_interval`, the
checkpoint is written once. An unset stop defaults to `num_steps`. Logs say
`Training invocation completed` for intermediate stages and
`Final training completed` only when the scheduler horizon is reached.

### Multi-node formal orchestrator

For the current formal run, use `script/track3_1/launch_hcu_stage_a.sh` with the
explicit two-node roster and 16 ranks. The launcher retains six-node/48-rank
topology compatibility, but every new formal launch uses the current
`batch_size=1, gradient_accumulation_steps=1` contract; the historical batch
recipe above is no longer accepted.
Set `N0_TRACK31_RUN_ROLE=development` for the
719/40 internal-development route or `final_refit` for the validation-free
759-episode route. The orchestrator accepts only the exact
`20 -> 25 -> 5000` recipe above and
requires immutable source, overlay, image, empty-embedding, known-hosts, dataset,
and launch-manifest identities. Configure these paths and digests outside the
repository; credentials must never be stored in the source tree or launch
manifest.

```bash
export N0_TRACK31_NODES=node0,node1
export N0_TRACK31_RUN_ROLE=final_refit
export N0_TRACK31_SSH_PORT=36000
export N0_TRACK31_KNOWN_HOSTS=/absolute/path/to/known_hosts
export N0_TRACK31_CONTAINER_NAME=n0-track31-stage-a
export N0_TRACK31_IMAGE=registry.example/project/image:version
export N0_TRACK31_IMAGE_ID=sha256:REPLACE_WITH_EXACT_IMAGE_ID
export N0_TRACK31_SOURCE_ROOT_HOST=/absolute/path/to/frozen/source
export N0_TRACK31_SOURCE_MANIFEST_SHA256=REPLACE_WITH_SOURCE_MANIFEST_SHA256
export N0_TRACK31_ARTIFACT_ROOT_HOST=/absolute/path/to/artifacts
export N0_TRACK31_TRAIN759_ROOT_HOST=/absolute/path/to/dataset/train759
export N0_TRACK31_RAW_ROOT_HOST=/absolute/path/to/raw/source
export N0_TRACK31_MODEL_ROOT_HOST=/absolute/path/to/model
export N0_TRACK31_OVERLAY_ROOT_HOST=/absolute/path/to/frozen/python_overlay
export N0_TRACK31_OVERLAY_MANIFEST_SHA256=REPLACE_WITH_OVERLAY_MANIFEST_SHA256
export N0_EMPTY_EMBEDDING_SHA256=REPLACE_WITH_EMPTY_EMBEDDING_SHA256
export N0_RELEASED_TRANSFORMER_SHA256=REPLACE_WITH_TRANSFORMER_SHA256
export N0_TRACK31_RUN_ROOT_HOST=/absolute/path/to/final759/run

bash script/track3_1/launch_hcu_stage_a.sh containers

export N0_TRACK31_STOP_AFTER_STEP=20
export N0_TRACK31_INVOCATION_ID=stage-a-step20-v1

# This foreground action performs the real 16-rank NCCL all-reduce and
# broadcast. It writes a new read-only, invocation-scoped report and prints
# the two N0_TRACK31_COLLECTIVE_SMOKE_* values required by `launch`.
bash script/track3_1/launch_hcu_stage_a.sh collective-smoke

export N0_TRACK31_COLLECTIVE_SMOKE_REPORT=/formal/run/preflight/collective_smoke.stage-a-step20-v1.json
export N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256=REPLACE_WITH_PRINTED_SMOKE_SHA256
bash script/track3_1/launch_hcu_stage_a.sh launch
```

The current two-node formal recipe uses local batch size 1 and gradient
accumulation 1, so each microbatch immediately performs an optimizer update and
the effective global batch is 16. This is an intentional new optimization
trajectory, not a batch-equivalent continuation of the retired recipe. The
final759 sampler still
pads 759 to 768 deterministically, so all source episodes remain represented
and the nine repeated assignments remain auditable.

For steps 25 and 5000, use a new invocation ID and the exact prior checkpoint
required by the launcher. Run `collective-smoke` again after each prior
invocation has stopped, then bind the newly printed report path and SHA before
`launch`. A report is accepted only for its matching invocation, exact node and
rank roster, current container IDs, HCU order, NCCL settings, image/source/
overlay identities, run role, and batch contract; it also has a bounded age.
`status` and `stop` operate only on containers and process groups carrying the
formal ownership labels and job receipts.

The training containers mount source, artifacts, `train759`, the materialization
marker, model, and Python overlay read-only; only the run root is writable.
Neither the raw HDF5 source nor `frozen40` is mounted. Health is accepted only
when every node has exactly one `torchrun` parent and eight direct
`n0_twam.train` children in the same container PID namespace and process group.
The launch receipt is trusted-operator provenance, not cryptographic attestation
against a caller controlling the same UID or run root; Trainer rank zero still
revalidates the stable training identity independently.

## 4. Tactile Prediction Quality

The fair-evaluation path is a two-stage, prediction-only protocol. It is
separate from the older `render_mot.py` internal/oracle diagnostic.

The single default unified evaluation set is `frozen_target10_v1`: the official
`metadata_val.json` projection for `insert_HDMI` and `lift_bottle`, with source
episode IDs `0,1,2,3,5` per task (10 episodes total). All leaderboard-oriented
and cross-checkpoint reports must use this cohort. `frozen_other30_v1` is an
optional generalization diagnostic and must never be merged into, substituted
for, or used to select the default Target-10 score.

### Stage A: causal prediction artifact

Run Stage A with only the converted `frozen40` repository and model artifacts
mounted. The default invocation requires `frozen_target10_v1`. Running the
optional `frozen_other30_v1` diagnostic additionally requires the explicit
`--allow-diagnostic-view` flag and produces a separate non-leaderboard report.

```bash
PYTHONHASHSEED=20260801 python \
  script/track3_1/generate_tactile_prediction_artifact.py \
  --config-name track31_univtac \
  --ckpt /absolute/path/to/workspace/n0_track31/runs/checkpoint_step_500 \
  --vae /absolute/path/to/workspace/models/n0-base/vae \
  --output /absolute/path/to/workspace/n0_track31/evaluation/raw/prediction_artifact \
  --evaluation-view-manifest \
    /absolute/path/to/workspace/n0_track31/artifacts/views/frozen_target10_v1.json \
  --protocol causal_future_only_v1 \
  --n-steps 8 \
  --seed 20260801 \
  --device cuda:0
```

There is no caller-controlled sample count: Stage A must cover the complete
10-episode or 30-episode view exactly once. The model receives task text, video
frame 0, tactile frame 0, and zero action/mask tensors. Future video, tactile,
and ground-truth actions are removed by the causal input sanitizer; tests mutate
every future tensor and require the model payload hash to remain unchanged.

The sealed schema-v2 artifact contains only predictions and row/step provenance;
it embeds no ground truth or LeRobot H264 copy. It binds checkpoint, decoder,
manifest, conversion, normalizer, training/evaluation views, deterministic
seeds, and payload hashes. The checkpoint's training manifest/normalizer/
conversion identities must exactly match the evaluation bundle.

### Stage B: raw-HDF5 diagnostic metric

Only Stage B receives a read-only raw UniVTAC mount. It verifies the artifact
seal and complete frozen-view roster before opening HDF5, reconstructs all 17
absolute RGB frames as `clip(round(raw_frame0 + 255 * residual[t]))`,
concatenates the two tactile sensors by width, and invokes one pre-approved
metric script whose SHA-256 is supplied explicitly:

```bash
python script/track3_1/evaluate_raw_tactile_quality.py \
  --prediction-artifact \
    /absolute/path/to/workspace/n0_track31/evaluation/raw/prediction_artifact \
  --evaluation-view-manifest \
    /absolute/path/to/workspace/n0_track31/artifacts/views/frozen_target10_v1.json \
  --raw-root /absolute/path/to/workspace/UniVTAC \
  --manifest /absolute/path/to/workspace/n0_track31/artifacts/universe_manifest_v4.json \
  --conversion-report \
    /absolute/path/to/workspace/n0_track31/artifacts/conversion_report.json \
  --official-metric-script \
    /path/to/visual-tactile_world_model_pipeline/metric/val_psnr_ssim.py \
  --official-metric-sha256 REPLACE_WITH_APPROVED_SHA256 \
  --output /absolute/path/to/workspace/n0_track31/evaluation/raw/raw_tactile_quality \
  --fps 10
```

### One-command Target-10 runner

For repeated runs, use the resumable coordinator instead of invoking the two
stages manually. It records a request fingerprint, Stage-A log, sealed
prediction artifact, raw-metric report, full Stage-B file-inventory receipt,
and a final receipt under one output root. A repeated invocation reuses only
artifacts whose checkpoint SHA-256, VAE decoder identity, causal sampling
configuration, sealed Target-10 view, and official metric-script SHA-256 all
match the same request; it never overwrites a disagreeing result.

Create the immutable request once (the template lists every required identity):

```bash
python script/track3_1/run_target10_tactile_evaluation.py \
  --print-request-template > target10_step1500_request.json
```

Then use one command for generation, raw-HDF5 reconstruction, official metric
invocation, and resumable reporting:

```bash
python script/track3_1/run_target10_tactile_evaluation.py \
  --request /absolute/path/to/target10_step1500_request.json
```

This prints the final PSNR/SSIM summary only when both stages are complete.
During execution, follow `OUTPUT_ROOT/evaluation_state.json` and
`OUTPUT_ROOT/logs/stage_a.log`. Reuse additionally verifies
`OUTPUT_ROOT/raw_tactile_quality_receipt.json` against every generated video
and report file; a stopped process releases the persistent OS lock
automatically. The resulting raw diagnostic remains distinct from an
organizer-confirmed leaderboard score until the golden frame/domain/backend
contract is independently reproduced.

The current report is deliberately named
`causal_raw17_legacy_metric_diagnostic_v1` and records
`leaderboard_oriented=false`, `leaderboard_compatible=false`, and
`published_score_comparable=false`. It scores 17 frames at 128x256, including
the conditioning frame. Therefore it must not yet be compared directly with the
published Wan2.2 `21.26 / 0.746` result.

The `wan22_target10_reference9_continuous41_v3` path unlocks direct comparison only after
a content-addressed Wan2.2 golden artifact reproduces `21.26 / 0.746`. It fixes
the target roster to `insert_HDMI` and `lift_bottle`, episodes `0,1,2,3,5`, raw
rows `[0,5,...,40]`, decoded frames `1..8`, `192x512` concatenated tactile RGB,
the pinned SciPy Gaussian SSIM implementation, finite PSNR cap 100, and
frame-to-episode-to-task-to-two-task macro aggregation. Stage A receives only a
sealed frame-0 condition bundle. The model produces one continuous 41-frame
trajectory and the adapter selects output indices `[0,5,...,40]`; it never
relabels nine consecutive outputs as stride-5 time. Stage B alone can open raw
HDF5. Because training used at most 17 decoded frames, the 41-frame inference
horizon is recorded as `evaluation_horizon_ood_vs_training17` in every artifact
and final receipt.

A standalone frame-pair evaluator is also available:

```bash
python script/track3_1/evaluate_tactile_quality.py \
  --prediction-root /path/to/prediction \
  --ground-truth-root /path/to/ground_truth \
  --output /path/to/tactile_quality.json
```

The standalone output is schema-v1 and not bound to a checkpoint. It exposes no
caller-selected closed-loop label; closed-loop evaluation needs external
execution evidence. PSNR/SSIM do not establish task success. NVIDIA simulator
success rates remain a separate final phase. Seeded replay is best-effort on one
runtime, not claimed bitwise-identical across HCU/CUDA hardware or grouped-SDPA
kernels.

### v18 1500-step calibrated one-command runner

The v18 entry now delegates to the complete reference9 pipeline. The metric
script SHA-256 is compiled into the protocol and cannot be auto-approved or
overridden. The checkpoint, normalizer, empty text embedding, and golden
manifest identities must be explicit. The request also binds the base model,
converted LeRobot root, and normalizer source view so a fresh container does
not depend on hidden
environment variables. Sampling is fixed to 50 Euler steps with seed 2026;
the published Wan2.2 reference used UniPC with shift 5, so sampler identity is
recorded and never silently described as identical. First write and review the
immutable request:

```bash
cd /path/to/WorldArena-2.0/N0-TWAM
python script/track3_1/run_target10_tactile_evaluation_v18_step1500.py \
  --print-template --save-template /absolute/path/to/target10_reference9.json
```

After reviewing every path and replacing the four SHA placeholders with
verified values, one command runs golden calibration, causal input preparation,
HCU generation, raw-HDF5 materialization, the strict metric, and the final
receipt:

```bash
export N0_TWAM_V18_TARGET10_REQUEST_JSON=/absolute/path/to/workspace/n0_twam_track31_.../target10_v18_step1500_request.json
./script/track3_1/run_target10_tactile_evaluation_v18_step1500.sh
```

The final result is
`OUTPUT_ROOT/evaluation_receipt.json`. `published_score_comparable=true` is
emitted only after the golden gate passes. `leaderboard_compatible` remains
false until the organizer confirms that this local contract is the submission
contract; NVIDIA simulator evaluation is a separate later phase.

Request schema v6 makes the calibration policy explicit. The generated
template uses `calibration_policy=require_published_golden`, which remains the
strict default and stops before HCU inference when the golden score is not
`21.26 / 0.746`. If the author-level golden prediction set is unavailable, an
operator may explicitly select `allow_protocol_aligned_report`. That mode
continues only after the content-addressed golden inventory and metric run are
valid but the displayed score mismatches; missing files, SHA drift, malformed
inventories, and metric failures still fail closed. Its receipt always records
`published_score_comparable=false`, stores the failed calibration evidence, and
labels the `21.26 / 0.746` delta as numeric-only rather than a fair published
comparison.

统一 wrapper 不再接受未审核的默认占位值。未设置
`N0_TWAM_V18_TARGET10_REQUEST_JSON` 时会立即失败；`--request` 也不能与任何
path/hash override 混用，避免请求文件被静默覆盖。只有完整 request、canonical
Target-10 view、checkpoint、decoder、base model、normalizer、latent inventories、
dataset、metric script 与密封 receipts 均一致时，
才会复用已有 artifact 与报告。任一身份不一致或 receipt 缺失都会停止，并要求使用
新的 output root 重新评测，不会静默切换请求或为旧指标重新签发 receipt。
checkpoint 还会通过 `train_meta.json`、`training_state.json` 与
`checkpoint_complete.json` 的 strict sidecar inventory 做三方 runtime/source 与
invocation identity 校验；该 identity 在 HCU 生成前、生成后和 artifact 发布后都会
重新捕获，并写入 schema-v5 prediction artifact。任何中途字节漂移都会 fail closed，
旧 schema artifact 也不会被本入口静默复用。

## Verification boundary

Local synthetic tests validate contracts, alignment, conversion, and metrics.
Real UniVTAC training is not complete until remote preflight, one-batch forward /
backward, checkpoint save/reload, validation loss, and final simulator evaluation
all pass on the target NVIDIA environment.
