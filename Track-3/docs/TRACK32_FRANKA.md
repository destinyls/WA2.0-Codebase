# N0-TWAM Franka Track 3.2 protocol

This document defines the reproducible vision-only Franka post-training and
deployment path. It is intentionally separate from the UniVTAC Track 3.1
pipeline: no UniVTAC split, tactile tensor, ACT target, or qpos8 action codec is
imported into this workflow.

## Official interface frozen by this branch

The current WorldArena Franka command is `end_pose_base`:

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

- position and orientation are expressed in the robot base frame;
- quaternion order is scalar-first `wxyz`;
- `joint_qpos=[joint_0..joint_6, gripper]` is an observation, not the action;
- the Policy returns one `float32[1,8]` command per long-poll call and declares
  `action_format=end_pose_base`;
- the current Franka tasks expose two unique RGB views and no tactile stream.

Always re-check the organizer documents before a submission:

- [WorldArena Policy guide](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/policy_guide.md)
- [WorldArena Track 3 description](https://github.com/WorldArena2/WorldArena-2.0/blob/main/assets/track3_description.md)

## Action representation

The released N0-TWAM Action Expert remains 20D. Replacing it with an 8D head
would discard compatible pretrained action weights, so this branch uses:

```text
official pose8 (wxyz)
    -> EE10 = [xyz, first two rotation-matrix columns, gripper]
    -> EE20[0:10] = EE10
    -> EE20[10:20] = 0, validity mask false
```

The rotation conversion is geometric and invertible up to quaternion sign. It
is not forward kinematics. `joint_qpos` may be used in a future fallback only
with a frozen URDF, TCP, base frame, calibration identity, and consistency
audit; it is not used as a training action in this implementation.

The Policy re-orthogonalizes predicted rot6d with Gram-Schmidt, converts it to
`wxyz`, and chooses the quaternion sign closest to the current/previous command.

## No-tactile contract

The Franka profile explicitly sets:

```text
tactile_profile = vision_only
tactile_mode = disabled
tactile_keys = []
synthetic_tactile_data = false
use_local_tactile = false
use_contact_gate = false
tactile_diffusion_loss_weight = 0
freeze_tactile_parameters = true
```

The training model rejects any accidental tactile tensor. Tactile-only
parameters remain in the checkpoint for compatibility but are frozen before
FSDP and before AdamW is built, so neither gradients nor weight decay can change
them. A no-tactile step reports `tactile_loss=0`.

## Pinned official dataset

The source is:

```text
repo: WorldArena/WorldArena2.0_Franka_FR3
revision: aed59b39c5a903be5e435c13c0ed1efdd54d5ad9
tasks: clear_up, pour, wipe
episodes: 200 per task, 600 total
files: 3,602
bytes: 59,922,514,413
canonical record SHA256:
67118a93230e13a5ecf8072df9cad4b30882367471017b4f1b49e43b6c8d4635
```

Each episode must contain one HDF5 file, two MP4 files, and three JSON
sidecars. Conversion reads `observations/end_pose[:,0:8]`, maps
`third_person.mp4` to `observation.images.top`, maps `wrist.mp4` to
`observation.images.wrist_l`, and stores the next recorded end pose as the
action target. The final source row is omitted because it has no next target.

Run the complete data pipeline from the repository root:

```bash
./prepare_track32_franka_data.sh \
  /absolute/franka-work \
  /absolute/n0-twam-base \
  8
```

Set `N0_TRACK32_HF_ENDPOINT=https://huggingface.co` to use the primary endpoint;
the script defaults to the HTTPS mirror used by the validated server setup. The
manifest generator rejects any API roster whose pinned record hash, count, or
byte total differs. Downloaded files are then verified by LFS SHA256 or Git blob
SHA1 before publication.

## Temporal alignment

The official data remain at 15 Hz. RGB preprocessing samples at 10 Hz, producing
a non-uniform native-row pattern such as `0,2,3,4,6,...`. Wan latent anchors are
every fourth sampled video frame and therefore land exactly at native rows
`0,6,12,...`.

The action alignment retains the released N0 cold-start convention. Action
frame zero repeats the first target as the learned cold slot; action frame
`f>0` contains the six targets following video anchor `f-1`. With
`frame_chunk_size=2`, N0 predicts an internal `[20,2,6]` chunk. The Policy skips
the cold slot, executes its first 6 commands one at a time, and executes all 12
commands for each later chunk. Every command is followed by a real robot
observation. Per six commands, post-action RGB rows `2,3,4,6` are retained,
exactly reproducing the training-time 15→10 Hz grid; this gives 4 cold or 8
regular grounding images and exactly 2 Wan latent frames. The generic
constant-stride action aligner is not used.

## Development and final-refit views

The public source contains demonstrations, not the organizer's hidden real
evaluation episodes. This repository creates only a development split:

| View | Membership | Purpose |
|---|---:|---|
| `franka_dev_train540_v1` | IDs 0..179 per task | recipe development |
| `franka_dev_validation60_v1` | IDs 180..199 per task | internal validation |
| `franka_final_refit600_v1` | all 600 | final weights after recipe lock |

Views are episode-level, task-stratified, disjoint, and hash-bound. The
development normalizer uses only its 540 training episodes. The final normalizer
uses all 600 public demonstrations. Do not call the 60-episode view an official
test set or report it as a leaderboard score.

## Training requests and accelerator profiles

Create requests with `n0-twam track32 build-request`; it fills every SHA field
after auditing the inputs. The public runner is request-only and single-node.
It removes ambient `N0_*`, Python-path, loader-injection, and device-selection
variables before constructing the child environment.

Two profiles are supported without code edits:

| Profile | Intended runtime | Attention / FSDP policy |
|---|---|---|
| `portable` | NVIDIA or standard PyTorch | grouped SDPA, FP32 reduction, activation checkpointing |
| `hcu_performance` | validated vendor HCU image | vendor grouped Flash Attention, BF16 reduction, expert pre/post reuse, no activation checkpointing |

The HCU profile fails closed when the vendor Flash Attention package is absent.
It also requires one explicit collective network interface; the runner binds
both NCCL and Gloo to that interface and rejects names absent from the runtime
container. Do not select it merely to make a generic GPU command faster.

Example:

```bash
n0-twam track32 build-request \
  --output /work/requests/dev.json \
  --run-id franka-dev-v1 \
  --devices 0,1,2,3,4,5,6,7 \
  --accelerator-profile hcu_performance \
  --collective-network-interface bond1 \
  --artifact-root /work/artifacts \
  --lerobot-root /work/data/lerobot \
  --base-model /models/n0-twam-base \
  --empty-embedding /models/n0-twam-base/empty_emb.pt \
  --init-from /models/n0-twam-base \
  --output-root /work/runs/dev-v1 \
  --run-role development \
  --num-steps 1500 \
  --stop-after-step 1500 \
  --save-interval 300 \
  --val-interval 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 1

./run_track32_franka.sh /work/requests/dev.json
```

One optimizer step is one local microbatch per rank when batch size and
gradient accumulation are both one. The effective global batch is therefore
the world size. Logical epoch length is `ceil(view_size / global_batch)` after
the sampler's deterministic global padding; use the observed log/receipt rather
than a hard-coded epoch conversion.

## Checkpoint and resume rules

A checkpoint is complete only if its model, optimizer, scheduler, RNG files,
training state, train metadata, code/environment identities, data identities,
and `checkpoint_complete.json` all pass the strict snapshot verifier.

Strict resume requires the same world size, accelerator execution contract,
data/view/normalizer identities, scientific recipe, and source package. Use a
new request, invocation ID, and output root. To change hardware profile or
switch from development to final refit, start a weights-only branch from the
released base (or an explicitly reviewed compatible transformer) instead of
claiming a strict resume.

## Offline reference metrics

Before Policy replay or real-robot evaluation, score the decoded future RGB and
Franka end-pose predictions with one strict command:

```bash
n0-twam track32 pack-predictions \
  --metadata /work/eval/prediction-metadata.json \
  --arrays /work/eval/prediction-arrays.npz \
  --output /work/eval/franka-future-predictions.npz

n0-twam track32 score-predictions \
  --predictions /work/eval/franka-future-predictions.npz \
  --checkpoint-identity-sha256 CHECKPOINT_IDENTITY_FROM_RECEIPT \
  --dataset-view-id franka_dev_validation60_v1 \
  --dataset-view-sha256 VALIDATION_VIEW_SHA256 \
  --decoder-sha256 VAE_DECODER_SHA256 \
  --output /work/eval/franka-offline-metrics.json
```

The prediction NPZ is non-pickled and has an exact schema. It binds the
checkpoint, dataset view, VAE decoder, seed, run role, and prediction mode, plus
the following arrays:

- `predicted_rgb` and `target_rgb`: `uint8[N,2,T,H,W,3]`, ordered as
  `cam_high`, `cam_left_wrist`;
- `video_valid`: `bool[N,2,T]`;
- `predicted_end_pose` and `target_end_pose`: finite float `[N,A,8]` in the
  official `end_pose_base=[xyz,wxyz,gripper]` layout;
- `action_valid`: `bool[N,A]`;
- `lerobot_episode_ids`: unique integer `[N]` used to audit the canonical view
  roster;
- positive `frame_offsets` and `action_offsets`; offset zero is rejected so the
  conditioning observation cannot inflate image quality;
- unique `sample_ids` and official `task_ids`.

`pack-predictions` is the supported O_EXCL, atomic public materializer. Its
metadata JSON contains exactly `checkpoint_identity_sha256`, `dataset_view_id`,
`dataset_view_sha256`, `decoder_sha256`, `seed`, `run_role`, and
`prediction_mode`; its array NPZ contains exactly the arrays listed above plus
the view names, offsets, sample IDs, and task IDs.

The report contains episode-macro PSNR/SSIM, Position MAE/RMSE in centimetres,
per-task, per-view, and per-sample results. PSNR is computed on RGB range
`[0,255]` and capped at 100 dB for byte-identical frames. Position error is the
L2 distance between predicted and target `xyz`: MAE is the mean distance and
RMSE is the root mean squared distance. SSIM fixes its complete algorithm
parameter set and the Python dependency pins the backend by Python version. The
command independently requires the
expected checkpoint/view/decoder identities instead of trusting self-declared
NPZ metadata.

The scorer checks whether episode IDs, order, tasks, and view SHA match a frozen
public view. It deliberately never self-certifies `generalization_claim_valid`:
an NPZ cannot independently prove that its target bytes came from the canonical
dataset. Until a checkpoint-to-target generation receipt closes that boundary,
development scores are non-organizer proxy evidence; `final_refit600` results
are additionally in-sample because all public episodes were used for fitting.

## Official Policy deployment

Create an immutable bundle:

```bash
n0-twam track32 serve-bundle \
  --checkpoint /work/runs/final/checkpoints/checkpoint_step_N \
  --checkpoint-identity-sha256 SHA_FROM_TRAINING_RECEIPT \
  --base-model /models/n0-twam-base \
  --normalizer /work/artifacts/normalizers/franka_final_refit600_v1.json \
  --normalizer-sha256 SHA_FROM_TRAINING_REQUEST \
  --output /work/serve-bundle

n0-twam track32 policy-template --output /work/policy.json
```

Run the bundled Policy across the cold-cache grounding boundary with a frozen,
non-pickled observation fixture:

```bash
n0-twam track32 policy-replay \
  --config /work/policy.json \
  --observation /work/offline-observation.npz \
  --prompt "clear the table" \
  --steps 7 \
  --output /work/offline-policy-replay.json
```

The NPZ must contain exactly `cam_high`, `cam_left_wrist`, `left_end_pose`, and
`joint_qpos`. This replay proves Policy loading, pose8 protocol, safety gating,
post-action observation accounting, and one cache commit. Its receipt always
states that organizer and real-robot evaluation remain incomplete.

The serve bundle links the large immutable model components and copies the
small normalizer, then re-audits all identities whenever the Policy starts.
`policy.json` deliberately contains invalid `CALIBRATE_*` placeholders. A robot
operator must replace them with signed limits for the exact cell; the template
cannot accidentally run with guessed workspace limits.

At runtime:

```bash
export N0_TRACK32_POLICY_CONFIG=/work/policy.json
```

The top-level `policy_franka.py` exports `Policy`. `reset()` clears episode
state. `infer()` maps `cam_high` and `cam_left_wrist`/`cam_wrist`, uses
`left_end_pose` plus the gripper component of `joint_qpos`, generates EE20,
decodes the first EE10, applies signed safety limits, feeds the actually executed
chunk plus time-aligned post-action RGB back to the N0 cache, and returns one
official pose8 action. An incomplete chunk is discarded on reset and is never
committed. Policy metadata reports whether safety intervened.

### Organizer bridge quaternion audit and worker launch

The official wire order is `wxyz`, while the canonical WorldArena packet stores
quaternion fields as `xyzw`. Both directions must be explicit:

- canonical `Quaternion(x,y,z,w)` must become Franka `left_end_pose`
  `[x,y,z,qw,qx,qy,qz]`;
- Policy action `[x,y,z,qw,qx,qy,qz,gripper]` must become canonical
  `Quaternion(x=qx,y=qy,z=qz,w=qw)`.

The reviewed organizer checkout
`6f5a981b34232fe77812b818a6ad7a4e6b8728ac` used positional unpacking in both
places and therefore fails the identity-quaternion probe. Keep this Policy and
dataset in official `wxyz` order. After the organizers approve the
participant-side compatibility patch, prepare the exact checkout and prove the
bridge before loading the model:

```bash
git -C "$WORLD_ARENA_ROOT" checkout \
  6f5a981b34232fe77812b818a6ad7a4e6b8728ac

n0-twam track32 bridge-patch \
  --worldarena-root "$WORLD_ARENA_ROOT"

n0-twam track32 bridge-audit \
  --worldarena-root "$WORLD_ARENA_ROOT"
```

`bridge-patch` accepts only the pinned revision and an otherwise clean
`real_world_benchmark/` tree. It applies one packaged patch and then runs the
same two-direction semantic probe as `bridge-audit`. The audit also binds the
bridge and official Hub worker file hashes. It rejects the unpatched revision,
shadow imports, unrelated worker-code changes, or either wrong quaternion
direction.

For the official outbound HTTPS long-poll process, set the organizer-provided
values and use the single public launcher:

```bash
export WORLD_ARENA_ROOT=/absolute/path/to/WorldArena-2.0
export N0_TRACK32_POLICY_CONFIG=/absolute/path/to/policy.json
export HUB_POLICY_URL=https://ORGANIZER_GATEWAY/policy
export POLICY_ID=ORGANIZER_ASSIGNED_WORKER_KEY
export HUB_TOKEN=ORGANIZER_BEARER_TOKEN  # optional

bash run_track32_franka_worker.sh
```

The script idempotently applies/audits the pinned patch and then invokes the
official WorldArena `run_policy_hub_worker`. The token is copied to a dedicated
environment variable and is never placed in the command line. `POLICY_ID` must
equal `policy_id` in the signed policy config. The worker accepts HTTPS only;
`--allow-local-http` permits only a loopback dummy Hub, and `--dry-run` performs
all identity/config checks without loading N0 or opening a network connection.
If the organizers publish a newer corrected revision, do not apply this pinned
patch: pass its exact revision and bridge SHA to `n0-twam track32 worker`.

## Evidence boundary

The following are separate milestones:

1. code and unit tests pass;
2. official data and latents are fully verified;
3. distributed smoke produces finite losses and a complete checkpoint;
4. final refit completes;
5. offline Policy replay passes protocol/safety checks;
6. organizer infrastructure accepts the submission;
7. the model achieves a measured real-robot success rate.

Only milestone 7 is a Track 3.2 result. This repository cannot produce that
score without the organizer worker/Hub credentials and access to the target
Franka setup.
