# Track 3.2 AgileX qpos14 Post-Training

This workflow adds an AgileX dual-arm route without changing the existing
UniVTAC qpos8 or Franka EE20 routes. It is designed for manifest-bound real
data and a 14D absolute joint command:

```text
[left_joint_1..6, left_gripper, right_joint_1..6, right_gripper]
```

The canonical action schema is `qpos14_joint_absolute_v1`. Formal labels must
come from an explicit same-row `commanded` or `executed` action column. Measured
`joint_qpos[t+1]` is available only under the separately named engineering
contract `measured_next_qpos_v1`; it is never silently promoted to a formal
label.

## Data contracts

A formal run requires immutable, SHA-256-bound inputs:

1. A frozen source inventory and its raw root.
2. A converted LeRobot root containing `action`, `action.valid`,
   `observation.joint_qpos`, RGB streams, and the declared contact streams.
3. A content-addressed repository-route manifest.
4. A self-hashed per-repository temporal-alignment manifest.
5. A q01/q99 normalizer fitted only on valid training actions.
6. Complete conversion and latent-inventory receipts.

The route manifest fixes, per repository:

- action schema and embodiment;
- RGB, tactile, and wrench keys;
- stable tactile/wrench sensor IDs;
- channel order, units, gripper encoding, and temporal identity.

The temporal manifest fixes each repository's explicit action offsets per
latent anchor. The adapter does not infer FPS ratios and does not add a second
`t -> t+1` shift.

The public integration APIs are:

```python
from n0_twam.integrations.worldarena.agilex_manifest import load_agilex_manifest
from n0_twam.integrations.worldarena.agilex_source import audit_episode
from n0_twam.integrations.worldarena.agilex_convert import (
    write_agilex_lerobot_dataset,
)
from n0_twam.integrations.worldarena.agilex_normalizer import (
    fit_agilex_normalizer,
)
```

The official Track 3 real-robot demonstrations are published in
[`WorldArena/WorldArena2.0`](https://huggingface.co/datasets/WorldArena/WorldArena2.0).
Use the pinned revision documented in the project README. A storage-specific
reader must decode the published `episode.hdf5`, RGB MP4, and optional tactile
files and pass their arrays into `audit_episode`. The repository verifies the
official channel, unit, gripper, and temporal contracts instead of inferring
semantics from tensor shape alone.

The shared `clean_table`, `pour_water`, and `wipe_table` task directories may
also contain Franka variants. The reader must route by audited metadata and
schema, never by task-directory name alone.

## Tactile profiles

All profiles use the same qpos14 Action Expert:

| Profile | Repository requirement | Contact modules |
|---|---|---|
| `vision_tactile` | Every repo has declared tactile and wrench streams | Used and trainable |
| `mixed` | Explicit tactile and vision-only repo routes | Used and trainable; content-addressed contact drop |
| `vision_only` | No tactile or wrench streams | Instantiated for topology compatibility, frozen, bypassed |

`contact_cond_drop` removes conditioning, not supervision: a tactile repo can
retain its tactile diffusion target while clean Local/Global tactile and wrench
conditions are masked. Vision-only samples have no tactile target.

## Training

After conversion and latent encoding, publish the two live-byte inventories in
one command. Every input hash is checked, both payloads are fully built before
publication, and existing outputs are never overwritten:

```bash
n0-twam track32 agilex-build-artifacts \
  --dataset-root /absolute/path/to/lerobot \
  --source-manifest /absolute/path/to/source.json \
  --source-manifest-sha256 <sha256> \
  --repo-route-manifest /absolute/path/to/routes.json \
  --repo-route-manifest-file-sha256 <sha256> \
  --temporal-alignment /absolute/path/to/temporal.json \
  --temporal-alignment-file-sha256 <sha256> \
  --conversion-output /absolute/path/to/conversion.receipt.json \
  --latent-output /absolute/path/to/latents.inventory.json
```

Then generate a strict training request, replace every placeholder path and
digest, and run the dry-run before allocating accelerators:

```bash
n0-twam track32 agilex-template \
  --output /absolute/path/to/agilex.train.json

n0-twam track32 agilex-train \
  --config /absolute/path/to/agilex.train.json \
  --dry-run

n0-twam track32 agilex-train \
  --config /absolute/path/to/agilex.train.json
```

The request selects exactly one of `vision_tactile`, `mixed`, or
`vision_only`. A fresh run uses `init_from` with the released 20D checkpoint and
`migrate_action`; all non-action tensors are copied while the complete action
input/output projection is deterministically reset to 14D. After the first
qpos14 checkpoint, every resume is strict and must retain the same profile,
route, temporal, normalizer, runtime, world-size, and training-lineage identity.

Use a new request, run ID, output root, and invocation for every run. A direct
full run is supported. A `20 -> 25 -> final` ladder is only a hardware bring-up
procedure and does not change the model recipe.

## Policy boundary

Seal the completed strict checkpoint together with the exact training route,
normalizer, task routes, and calibrated safety contract:

```bash
n0-twam track32 agilex-serve-bundle \
  --checkpoint /absolute/path/to/checkpoint_step_1500 \
  --checkpoint-identity-sha256 <sha256> \
  --base-model /absolute/path/to/released-base \
  --normalizer /absolute/path/to/normalizer.json \
  --normalizer-file-sha256 <sha256> \
  --normalizer-contract-sha256 <sha256> \
  --source-manifest-sha256 <sha256> \
  --repo-route-manifest /absolute/path/to/routes.json \
  --repo-route-manifest-file-sha256 <sha256> \
  --repo-route-manifest-sha256 <sha256> \
  --tactile-profile vision_tactile \
  --contact-profile-contract-sha256 <sha256> \
  --task-routes /absolute/path/to/task-routes.json \
  --task-routes-sha256 <sha256> \
  --safety-contract /absolute/path/to/safety.json \
  --safety-contract-sha256 <sha256> \
  --output /absolute/path/to/agilex-serve-bundle
```

All three JSON contracts are self-hashed and externally file-hash-bound. The
command refuses an existing output and re-audits the physical transformer and
strict checkpoint lineage before publication.

Generate the deliberately non-runnable local Policy template after sealing a
serve bundle, fill its hashes, task routes, and calibrated safety limits, then
run the allocation-free identity check:

```bash
n0-twam track32 agilex-policy-template \
  --output /absolute/path/to/agilex.policy.json

n0-twam track32 agilex-policy-check \
  --config /absolute/path/to/agilex.policy.json
```

The check revalidates the config self-hash, strict checkpoint, serve-bundle
receipt and component links, encoder source, route manifest, normalizer, task
routes, and safety contract. It does not load model weights, initialize a
process group, create the serve output, contact a robot, or claim an official
evaluation.

`n0_twam.integrations.worldarena.agilex_policy.AgileXPolicy` validates:

- exact top/left-wrist/right-wrist RGB input;
- finite float32 `joint_qpos[14]`;
- profile-bound tactile/wrench presence;
- sealed joint bounds and per-second step limits;
- observation freshness, monotonic timestamps, and inference deadline;
- one finite float32 absolute-qpos target with shape `[1, 14]` per call.

The Policy core accepts an injected backend. The local Direct N0 backend uses
the profile-bound AgileX server config to generate an exact 12-action internal
chunk. The Policy safety-projects one action at a time against the latest real
`joint_qpos`, records the action actually returned to the robot, and collects
the post-action observations. After action 12, it commits the executed
`[14,1,12]` history plus four time-aligned 10-Hz RGB keyframes to the rolling KV
cache; contact routes commit the corresponding tactile and wrench history.
Mixed RGB-only routes commit RGB/action history without fabricating contact.
The next call then refills the queue from the re-grounded cache. Reset discards
an incomplete chunk. This is a synchronous receding-horizon contract; it does
not overlap model inference with robot execution and does not invent an
organizer transport protocol.

The organizer-facing in-process adapter is
`n0_twam.integrations.worldarena.agilex_official_policy.Policy`. It maps
`cam_high/cam_wrist_left/cam_wrist_right` to the internal RGB route, verifies
`state`, `joint_qpos`, and the two 7D split-arm states agree, maps the documented
nested tactile/wrench payload, and returns one float32 `[1,14]` absolute joint
target per call. The organizer-defined `tactile_profile` is treated only as a
required non-empty label; modality presence is governed by the content-addressed
task route sealed inside the verified Policy config. This is an integrity
contract, not a claim of an organizer cryptographic signature.

For an approved in-process runner, bind the already verified policy config:

```bash
export N0_TRACK3_AGILEX_POLICY_CONFIG=/absolute/path/to/agilex.policy.json
python -c 'from n0_twam.integrations.worldarena.agilex_official_policy import Policy; Policy().close()'
```

This validates and allocates the local model, but does not contact a robot. An
official Hub worker may be connected only after the organizer publishes or
approves the exact transport, credentials, endpoint, and robot-cell safety
contract.

Conversion receipts, finite loss, strict checkpoints, offline replay, and
latency measurements are engineering evidence. They are not an official
real-robot success rate or leaderboard result.

## Offline evaluation

After the strict checkpoint has been sealed into a verified serve bundle and
the Policy config passes `agilex-policy-check`, run the complete offline
engineering closeout with one command:

```bash
n0-twam track32 agilex-offline-eval \
  --train-request /absolute/path/to/agilex.train.json \
  --config /absolute/path/to/agilex.policy.json \
  --dataset-root /absolute/path/to/converted/lerobot \
  --output-root /absolute/new/path/to/agilex-offline-eval-v1 \
  --device 0 \
  --samples-per-task 1 \
  --replay-steps 7
```

The output root must not already exist. The pipeline freezes one deterministic
sample for each of the ten official AgileX tasks, runs the real Direct N0
backend, and publishes:

- `evaluation_view.json`: ordered, self-hashed all-task proxy roster;
- `predictions.npz`: pickle-free future RGB and qpos14 predictions;
- `metrics.json`: RGB PSNR/SSIM plus qpos14 MAE/RMSE overall, per task,
  per camera, and for left arm/gripper/right arm/gripper groups;
- `replay_observation.npz` and `policy_replay.json`: seven-step Policy protocol,
  safety-projection, and backend-latency replay;
- `closeout.json`: immutable identities and the compact result summary.

The conditioning frame and action offset zero are excluded from metrics. The
view is sampled from converted training-distribution data, so the report sets
`independent_holdout=false`, `organizer_evaluation_completed=false`, and
`real_robot_evaluation_completed=false`. PSNR/SSIM and qpos14 errors are useful
model-development proxies; Policy replay is a systems check. Neither is an
official robot success rate or leaderboard score. Latency is reported, but the
command does not fail solely because a large generative backend misses the
requested control period unless a separately calibrated deployment gate is
used.

For this offline-only closeout, the Policy safety JSON may use an explicitly
labelled broad dataset envelope so generated actions can be inspected without
robot commands. Such a config must never be reused by the official worker or a
physical robot. Real deployment requires a separately approved calibration
contract with robot/cell joint limits and rate limits.
