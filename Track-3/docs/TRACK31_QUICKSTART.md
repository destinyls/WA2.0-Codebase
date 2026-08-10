# UniVTAC Track 3.1 public quickstart

The public interface is one executable with three subcommands. It replaces the
large collection of training environment variables with strict JSON requests:

```text
n0-twam track31 template train|eval
n0-twam track31 train --config TRAIN.json
n0-twam track31 eval --request EVAL.json
```

From a source checkout, `./run_track31.sh` is the exact same interface. The old
`run_track31_univtac.sh` and `script/track3_1/launch_hcu_stage_a.sh` remain for
advanced formal HCU deployments; new users should not start there.

## 1. Install

```bash
pip install '.[track31]'
# LeRobot 0.3.3 pins an incompatible historical Torch range. The validated
# runtime uses this project's Torch and installs LeRobot without its pins.
pip install lerobot==0.3.3 --no-deps
```

Prepare and latent-encode UniVTAC first. The detailed immutable data contract is
documented in [TRACK31_UNIVTAC.md](TRACK31_UNIVTAC.md).

## 2. Train

Generate a development-safe template:

```bash
n0-twam track31 template train --output track31.train.json
```

Edit only that JSON. Relative paths are resolved relative to the request file,
not the current shell directory. Replace both SHA-256 placeholders with the
audited file hashes. `output_root` must be disjoint from every immutable input;
when resuming, use a new output directory and point `resume_from` at the prior
complete checkpoint. Then validate the route and resolved command without
starting devices:

```bash
n0-twam track31 train --config track31.train.json --dry-run
```

Start the synchronous run:

```bash
n0-twam track31 train --config track31.train.json \
  2> >(tee track31.console.log >&2)
```

The command performs filesystem preflight, launches one local `torchrun`, waits
for completion, verifies the strict checkpoint sidecars, and prints one JSON
completion receipt. `runtime.devices` selects physical CUDA/HCU IDs; every
worker still uses its logical remapped device. Multi-node HCU deployment is kept
out of this public request because it contains site-specific SSH, Docker, NCCL,
and trusted-operator identities.

Public runs use `execution_tier=local_package`. Their receipts explicitly set
`formal_track31=false` and `leaderboard_eligible=false`; the launcher never
fabricates an internal Docker image ID. A strict resume requires the same local
code/environment identity and automatically rejects a cross-tier resume.

## 3. Evaluate the unified Target-10 view

The default evaluation is the frozen ten-episode projection: five `Insert HDMI`
and five `Lift Bottle` episodes. Create the portable request:

```bash
n0-twam track31 template eval --output target10.eval.json
```

Bind every placeholder to the immutable checkpoint, `frozen_target10_v1` view,
normalizer, raw UniVTAC root, official metric implementation, and published
golden calibration assets. Then run:

```bash
n0-twam track31 eval --request target10.eval.json
```

This command builds the causal 41-frame input bundle, generates the exact
0,5,...,40 tactile grid, materializes predictions, recomputes PSNR/SSIM from the
MP4 inventory, and signs the final receipt. It intentionally fails closed when
any checkpoint sidecar, dataset identity, golden manifest, or request binding
differs. It does not claim that the public Target-10 projection is the hidden
competition server set; it is the repository's unified, leakage-free comparison
protocol.

## 4. Help and source-tree usage

```bash
n0-twam track31 --help
./run_track31.sh template train --output track31.train.json
./run_track31.sh train --config track31.train.json --dry-run
./run_track31.sh eval --request target10.eval.json
```
