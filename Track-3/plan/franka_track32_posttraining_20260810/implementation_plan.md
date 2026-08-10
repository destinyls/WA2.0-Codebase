# Franka Track 3.2 Detailed Implementation Plan

## 1. Data and representation

1. Audit every official episode before conversion:
   - required files: `episode.hdf5`, `third_person.mp4`, `wrist.mp4`, `metadata.json`, `robot_state.json`, `camera_timestamps.json`;
   - required label: `observations/end_pose[:, 0:8]`;
   - reject non-finite position/gripper, near-zero quaternion norm, frame-count mismatch, missing cameras, or duplicate episode identity.
2. Convert each official action from quaternion `wxyz` to the N0 convention `rot[:, :2].T.reshape(6)`.
3. Store LeRobot action as EE10, record source 8D and transform revision in metadata, and let the existing loader embed EE10 into 20D using `used_action_channel_ids=0..9`.
4. Convert camera names deterministically:
   - `third_person.mp4 -> observation.images.top`;
   - `wrist.mp4 -> observation.images.wrist_l`.
5. Produce immutable source and converted manifests with per-file SHA-256 and exact episode/task roster.

## 2. Configurable no-tactile training

The Franka profile will set:

- `tactile_keys=[]`, `per_repo_tactile_keys={}`;
- `tactile_optional=False`, `synthetic_tactile_data=False`;
- `use_local_tactile=False`, `use_contact_gate=False`, `server_tactile_denoise=False`;
- `tactile_cfg_prob=0`, `noisy_cond_prob_tactile=0`;
- `action_dim=20`, `used_action_channel_ids=list(range(10))`;
- `batch_size`, `gradient_accumulation_steps`, steps, validation cadence, and save cadence only from the signed request.

The model receives the existing no-tactile zero-token/zero-anchor path. Tactile targets are absent, tactile loss is exactly zero, and a runtime assertion rejects any accidental tactile payload or synthetic fallback.

## 3. Training package

- Add `n0_twam/track32/` for immutable request parsing, preflight, provenance, and runner orchestration.
- Add `n0_twam/configs/twam_track32_franka_cfg.py`, driven only by validated environment values emitted by the runner.
- Add one public command: `./run_track32_franka.sh /abs/path/request.json`.
- Preflight binds dataset/source manifest, converted manifest, normalizer, base model, initialization checkpoint, empty embedding, code manifest, action transform, tactile profile, and output-root disjointness.
- Data preparation is idempotent: download, verify, convert, build pool/statistics, encode video/text latents, then launch smoke/formal training.

## 4. Track 3.2 Policy adapter

- Implement the official `Policy.__init__`, `reset`, and `infer` interface.
- Map live `cam_high` and `cam_left_wrist` to the N0 camera keys.
- Convert `left_end_pose` plus current gripper to EE10/20D state.
- Decode the first 10 model channels to `end_pose_base` 8D; convert rot6d via Gram-Schmidt and emit normalized `wxyz` quaternion.
- Apply quaternion sign continuity against the current/previous command.
- Reject NaN/Inf, degenerate rotations, wrong chunk/dimension, stale episode state, or out-of-contract metadata before returning an action.
- Expose one action per official call. Accumulate all post-action observations,
  subsample native rows `2,3,4,6` per six actions to the training 10-Hz grid,
  and commit only complete cold/regular brackets to the N0 cache.
- Do not silently clip workspace/gripper ranges. Optional limits must come from the signed task request and any intervention is reported in metadata.

## 5. Verification gates

1. Geometry tests: identity/random quaternion round trip, sign invariance, degenerate inputs, batched shapes.
2. Mask tests: EE10 embeds into channels `0..9`, channels `10..19` are zero with false masks, gradients only use active channels.
3. No-tactile tests: dataset construction without tactile inventory, forward/backward finite losses, tactile loss exactly zero, no synthetic stream.
4. Converter tests: tiny real-layout HDF5 + MP4 fixture, frame counts, metadata, deterministic manifest, rerun idempotency.
5. Request/preflight tests: missing/mismatched hashes, path overlap, wrong action schema, accidental tactile configuration, checkpoint mismatch.
6. Policy tests: official observation aliases, exact `(chunk,8)` output, quaternion continuity, reset isolation, failure boundaries.
7. Regression tests: released EE20 codec and existing Track 3.1 paths remain unchanged.
8. Remote gates: official manifest verification, one-device smoke, distributed collective smoke, finite loss/grad norm, complete checkpoint receipt, offline replay.

## 6. Formal training and leaderboard boundary

- Start from the released N0-TWAM checkpoint as a weights initialization, not a fake strict resume.
- Select recipe from a small measured smoke (1 HCU then multi-HCU); do not promise a utilization number before profiling.
- Save complete checkpoints at configured intervals and retain optimizer/scheduler/RNG/provenance sidecars.
- Use a frozen held-out split grouped by episode and task; no frame-level leakage.
- Offline replay validates the integration but is not a Track 3.2 score. Final ranking requires the organizer-provided Hub URL/worker key and real Franka trials.
