# Franka 48-card HSDP training contract

This document describes the package-level multi-node path for N0-TWAM Franka
Track 3.2 post-training. The reference topology is six HCU nodes with eight
visible devices per node:

```text
HSDP mesh [replicate=6, shard=8]
world size = 6 nodes * 8 local ranks = 48 ranks
```

The eight-rank shard group stays within one physical node. The six-rank
replicate group crosses nodes over the validated RCCL/RDMA transport. This path
still uses the Franka `vision_only` profile and scalar-last XYZW action contract.
It is not an AgileX launcher.

## Scope and prerequisites

The module creates immutable launch provenance, validates every node, starts
one `torchrun` parent per node, and verifies the completed checkpoint. An
external scheduler or operator must start the node-local commands
simultaneously; the module does not manage SSH credentials or remote process
supervision.

All nodes must provide:

- the same source checkout, Python environment, model, dataset and artifact
  paths, normally through shared storage;
- exactly eight visible HCUs in the request's device order;
- the `hcu_performance` accelerator profile and one common collective network
  interface;
- four ordered SHCA HCA ports mapped to `uverbs0..3`;
- active 400-Gb/s links, a non-empty GID at index 3, and readable/writable
  `/dev/infiniband/rdma_cm` plus `/dev/infiniband/uverbs0..3`;
- the pinned SHCA RCCL network plugin at the audited host path and SHA256.

The contract fixes `NCCL_NET_PLUGIN=shca`, `NCCL_NET_GDR_LEVEL=PHB`, and leaves
`NCCL_DMABUF_ENABLE` unset. Preflight rejects transport, plugin, device, HCA,
topology, or hash drift.

## 1. Build the Franka request

Create a normal Franka request with local devices `0..7`, profile
`hcu_performance`, and the intended collective interface. The request's output
root must not exist. For formal completion, use `run_role=final_refit` and make
`stop_after_step` equal `num_steps`.

```bash
n0-twam track32 build-request \
  --output /shared/franka-hsdp/control/final-refit.json \
  --run-id franka-final-hsdp-48-v1 \
  --devices 0,1,2,3,4,5,6,7 \
  --master-port 29500 \
  --accelerator-profile hcu_performance \
  --collective-network-interface COLLECTIVE_IFACE \
  --artifact-root /shared/franka/artifacts \
  --lerobot-root /shared/franka/data/lerobot \
  --base-model /shared/models/n0-twam-base \
  --empty-embedding /shared/models/n0-twam-base/empty_emb.pt \
  --init-from /shared/models/n0-twam-base \
  --output-root /shared/franka-hsdp/runs/final-refit-v1 \
  --run-role final_refit \
  --num-steps 3000 \
  --stop-after-step 3000 \
  --save-interval 300 \
  --val-interval 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 1
```

Use the actual locked recipe rather than copying the example step counts.

## 2. Prepare the shared launch identity

Run `prepare` once on the coordinator. The request's master port and the
argument below must match. Node names are immutable provenance labels; the
master address must be reachable through the selected collective interface.

```bash
python -m n0_twam.track32.multinode prepare \
  --request /shared/franka-hsdp/control/final-refit.json \
  --nodes hcu-node-0,hcu-node-1,hcu-node-2,hcu-node-3,hcu-node-4,hcu-node-5 \
  --local-world-size 8 \
  --master-addr MASTER_ADDR \
  --master-port 29500 \
  --nccl-ib-hca shca_0:1,shca_1:1,shca_2:1,shca_3:1 \
  --environment /shared/franka-hsdp/control/environment.json \
  --descriptor /shared/franka-hsdp/control/descriptor.json
```

The environment and descriptor use create-once semantics. Ambient token,
password, credential, cookie, private-key, authentication, and user-session
variables are removed before the environment is persisted.

## 3. Run node-local preflight

Run the following once on every node with a unique rank from 0 to 5:

```bash
python -m n0_twam.track32.multinode run-child \
  --environment /shared/franka-hsdp/control/environment.json \
  --node-rank "$NODE_RANK" \
  --mode preflight
```

Do not proceed unless all six commands return zero.

## 4. Run the collective smoke

Start one smoke parent on every node, using an unused port shared by all six
commands. Global rank zero writes the immutable report.

```bash
python -m n0_twam.track32.multinode run-smoke \
  --environment /shared/franka-hsdp/control/environment.json \
  --node-rank "$NODE_RANK" \
  --master-port 29501 \
  --report /shared/franka-hsdp/control/hsdp-collective-smoke.json
```

The smoke verifies the two-dimensional mesh, node-local shard groups,
cross-node replica groups, a 48-rank all-reduce, and measured collective
bandwidth. Inspect the report and RCCL logs before training. The current
implementation does not automatically schedule nodes or replace operator
review of this report.

## 5. Start training

After all preflights and the smoke pass, start one training parent on every
node at approximately the same time:

```bash
python -m n0_twam.track32.multinode run-child \
  --environment /shared/franka-hsdp/control/environment.json \
  --node-rank "$NODE_RANK" \
  --mode train
```

Each parent replaces itself with `torchrun --nnodes=6 --nproc-per-node=8` and
loads `track32_franka`. Checkpoint metadata records the HSDP mesh, RCCL HCA
roster, GDR mode, plugin identity, and Franka XYZW/profile identities.

## 6. Verify completion

After every node exits successfully, verify the final checkpoint from a node
with access to the shared output:

```bash
python -m n0_twam.track32.multinode verify \
  --request /shared/franka-hsdp/control/final-refit.json \
  --descriptor /shared/franka-hsdp/control/descriptor.json
```

Verification requires a complete strict checkpoint at the requested stop step,
world size 48, matching invocation/source identity, Franka XYZW profile, HSDP
shape, and pinned transport metadata. Only a complete `final_refit` request is
marked `formal_track32_training_completed=true`; development and partial runs
remain false. The receipt always records
`leaderboard_evaluation_completed=false`.

## Evidence boundary

Local unit tests validate contract construction and fail-closed drift checks.
They do not prove that a 48-HCU collective, full training run, official remote
Franka evaluation, or leaderboard submission has completed. Preserve the
collective report, per-rank logs, immutable descriptor, checkpoint snapshot,
and final training receipt as separate evidence artifacts.
