# Task Plan: Franka Track 3.2 Post-Training

## Goal

基于官方 N0-TWAM 与 WorldArena 官方 `WorldArena2.0_Franka_FR3` 数据，交付可配置无触觉、官方 Franka `end_pose_base` 8D 到 N0-TWAM 单臂 EE10/20D mask 的后训练和 Track 3.2 Policy 入口，并完成本地验证、远端数据准备、训练 smoke 与正式训练启动。

## Frozen Contracts

- Official action: `end_pose_base=[x,y,z,qw,qx,qy,qz,gripper]`, base frame, quaternion `wxyz`.
- Observation `joint_qpos=[joint_0..joint_6,gripper]` is not the action label.
- Model action: `[xyz, rot6d(first two rotation-matrix columns), gripper]` in channels `0..9`; channels `10..19` are padded and masked.
- Tactile profile: no tactile keys, no synthetic tactile, LocalTactile/contact gate off, tactile loss exactly zero.
- Official Policy output: one finite `float32[1,8]` command per long-poll call
  with `action_format=end_pose_base`; internal N0 chunks remain `[20,2,6]`.

## Phases

- [x] Phase 1: Freeze the official Franka protocol and audit local/remote assets.
- [x] Phase 2: Implement the Franka codec, official HDF5 audit/conversion, no-tactile config, preflight, and one-command training entry.
- [x] Phase 3: Implement the Track 3.2 Policy adapter, output safety checks, and public documentation.
- [ ] Phase 4: Run unit/integration/build/lint/security verification and fix failures (targeted regression passed; full unit suite running).
- [ ] Phase 5: Sync the isolated source to remote storage, download/verify official data, convert it, and encode video/text latents.
- [ ] Phase 6: Run single-device and distributed training smoke; validate finite losses, masks, gradients, checkpoints, and process contracts.
- [ ] Phase 7: Start formal post-training and record immutable run receipts.
- [ ] Phase 8: Run offline replay/Policy protocol checks and prepare the real-robot leaderboard handoff.

## Tactile Profile Expansion

This is the next local release gate. It does not modify the immutable source or
controllers used by the currently running remote Franka job.

- [x] Define the fail-closed `vision_tactile`, `mixed`, and `vision_only`
  contracts and wire them into configs, dataset selection, model inputs,
  trainability, checkpoint sidecars, serving metadata, and documentation.
- [x] Close implementation details in one pass: validate the exact selected
  repository roster, preserve the native three-expert MoT state dict, reject
  inconsistent mode/profile combinations, and keep the public launcher
  config-driven.
- [x] Run one unified targeted matrix covering all three profiles: config
  validation, repo routing, tactile/no-tactile batches, forward/backward,
  bitwise frozen tactile weights after AdamW, strict resume mismatch rejection,
  and server/Policy metadata.
- [ ] Run the complete public unit suite plus Python compilation, shell syntax,
  diff hygiene, and an independent read-only code review. Fix all regressions
  before any release action.
- [ ] Freeze a new local commit and source manifest, update the public README
  with the three one-command examples, then publish with a normal non-force push
  only after the local commit is fully verified.
- [ ] Keep the current Track 3.2 remote training on its existing frozen source.
  Introduce the new profile-aware source remotely only as a new immutable
  version and only after the current run or an explicitly approved new smoke.

## Decisions

- Development is isolated on branch/worktree `Franka-PostTraining`; the existing `UniVTAC-PostTraining` worktree remains untouched.
- The released 20D Action Expert is retained for checkpoint compatibility; no 8D replacement head is introduced.
- 8D to 10D is quaternion-to-rot6d geometry, not FK. FK is optional validation only if a future source lacks `end_pose`.
- Official source data and immutable manifests are never modified in place; converted LeRobot data and latents use separate output roots.
- Training and Policy each expose one public request-file-driven command and fail closed before allocating accelerators.
- Formal training is reported as started only after the official dataset, base model, checkpoint, finite optimizer loss, and a complete checkpoint have all been observed.

## Verified Remote Assets

- Shared storage has 4.1 TiB free.
- Official manifest identifies `WorldArena/WorldArena2.0_Franka_FR3`: 3 tasks (`clear_up`, `pour`, `wipe`), 200 episodes each, 3,602 files, 59,922,514,413 bytes.
- The exact manifest is frozen; one complete official episode has been downloaded
  and byte-verified, while the remaining snapshot is not yet materialized.
- N0 base-model assets are referenced read-only. Frozen SHA256 values are
  `77af33c1...cc39` for `empty_emb.pt` and `41156d91...f0f3` for the released
  20D transformer.

## Status

**Phase 4 final verification in progress** - the three-profile implementation,
exact mixed shape bucketing, signed serving routes, and fail-closed legacy
checkpoint migration are complete in the local working tree. Focused tests,
format/lint/type checks, compilation, shell syntax, diff hygiene, and package
content verification pass; the fresh full unit suite and final independent
review are the remaining local gates. The active remote Franka run is
intentionally unchanged.

## Errors Encountered

- Focused profile regression initially failed during test collection because
  the generic post-training config did not declare
  `active_tactile_sensor_count`. The stricter profile contract exposed this
  missing field; the base config now derives it from the sensor map and also
  declares `tactile_global_zero=False` explicitly before the same matrix is
  rerun.
