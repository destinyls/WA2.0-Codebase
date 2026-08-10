# Notes: Franka Track 3.2 Post-Training

## Official protocol

- WorldArena `assets/policy_guide.md` defines Franka `end_pose_base` as 8D `[x,y,z,qw,qx,qy,qz,gripper]` in the robot base frame.
- `joint_qpos` is a separate 8D observation: seven arm joints plus gripper.
- Franka has two unique RGB sources: `cam_high` and `cam_left_wrist`; `cam_wrist` is an alias.
- Current official Franka vision-only tasks provide no tactile observation.
- The public protocol permits `(chunk,8)`, but this adapter deliberately returns
  `(1,8)` so every executed command receives a real post-action observation.

## Official dataset

- Hugging Face dataset: `WorldArena/WorldArena2.0_Franka_FR3`.
- Frozen expected API manifest SHA-256: `67118a93230e13a5ecf8072df9cad4b30882367471017b4f1b49e43b6c8d4635`.
- 600 episodes total: `clear_up=200`, `pour=200`, `wipe=200`.
- 59,922,514,413 bytes across 3,602 records.
- Each episode contains HDF5, two MP4s, and three JSON sidecars.

## Local baseline

- New worktree: `N0-TWAM-Franka-PostTraining`, branch `Franka-PostTraining`, initial commit `9036c13`.
- This baseline already contains the generic 20D action-mask and no-tactile model paths validated during prior work; Franka-specific code will live in new modules and will not import UniVTAC integration code.
- The N0 rot6d convention is the first two rotation-matrix columns flattened as `[col0(3), col1(3)]`.

## Remote audit

- Login alias `ssh hpu` reaches host `j03r3n08`.
- One complete official episode is materialized and verified; 599 episodes remain
  to be downloaded before conversion.
- Released asset SHA256 values are frozen: `empty_emb.pt =
  77af33c105e3d3204dd55217f1f336df48effd1ef0c05be3524ffd23a175cc39`
  and transformer `=
  41156d915e5dc23fcf2123960507483c0f8cd33ca0385043f088daceb86ef0f3`.
