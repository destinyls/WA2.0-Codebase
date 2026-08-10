#!/bin/bash
# N0-native UniVTAC Track 3.1 training launcher. No ACT code is imported.
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

if (($# != 0)); then
  echo "run_track31_univtac.sh does not accept CLI arguments; use N0_TRACK31_* environment variables" >&2
  exit 1
fi

: "${N0_TRACK31_ARTIFACT_ROOT:?set N0_TRACK31_ARTIFACT_ROOT}"
: "${N0_TRACK31_LEROBOT_ROOT:?set N0_TRACK31_LEROBOT_ROOT}"
: "${N0_BASE_MODEL:?set N0_BASE_MODEL}"
: "${N0_EMPTY_EMBEDDING:?set N0_EMPTY_EMBEDDING}"
: "${N0_EMPTY_EMBEDDING_SHA256:?set N0_EMPTY_EMBEDDING_SHA256}"
[[ "$N0_EMPTY_EMBEDDING_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "N0_EMPTY_EMBEDDING_SHA256 must be a lowercase SHA256" >&2
  exit 1
}

export N0_TRACK31_TRAIN_PROFILE=${N0_TRACK31_TRAIN_PROFILE:-multitask_pretrain_v1}
export N0_TRACK31_RUN_ROLE=${N0_TRACK31_RUN_ROLE:-final_refit}

case "$N0_TRACK31_TRAIN_PROFILE" in
  multitask_pretrain_v1|target_finetune_v1) ;;
  *)
    echo "N0_TRACK31_TRAIN_PROFILE must be multitask_pretrain_v1 or target_finetune_v1" >&2
    exit 1
    ;;
esac
case "$N0_TRACK31_RUN_ROLE" in
  development|final_refit) ;;
  *)
    echo "N0_TRACK31_RUN_ROLE must be development or final_refit" >&2
    exit 1
    ;;
esac
if [[ "$N0_TRACK31_RUN_ROLE" == "final_refit" ]] \
    && [[ -n "${N0_TRACK31_VALIDATION_VIEW_PATH:-}" ]]; then
  echo "final_refit must not use a validation view" >&2
  exit 1
fi
if [[ -n "${N0_TRACK31_RESUME_FROM:-}" ]]; then
  if [[ -n "${N0_TRACK31_INIT_FROM:-}" ]]; then
    echo "N0_TRACK31_INIT_FROM and N0_TRACK31_RESUME_FROM are mutually exclusive" >&2
    exit 1
  fi
elif [[ "$N0_TRACK31_TRAIN_PROFILE" == "multitask_pretrain_v1" ]]; then
  if [[ -n "${N0_TRACK31_INIT_FROM:-}" ]]; then
    echo "N0_TRACK31_INIT_FROM is reserved for target_finetune_v1" >&2
    exit 1
  fi
  : "${N0_RELEASED_CHECKPOINT:?set N0_RELEASED_CHECKPOINT for initial migration}"
  : "${N0_RELEASED_TRANSFORMER_SHA256:?set the audited released transformer SHA256}"
else
  : "${N0_TRACK31_INIT_FROM:?set N0_TRACK31_INIT_FROM to a Stage A checkpoint}"
fi

# Keep profile outputs disjoint by default while preserving an explicit legacy
# N0_TRACK31_SAVE_ROOT override.
if [[ -z "${N0_TRACK31_SAVE_ROOT:-}" ]]; then
  export N0_TRACK31_SAVE_ROOT="${N0_TRACK31_ARTIFACT_ROOT}/runs/${N0_TRACK31_TRAIN_PROFILE}/${N0_TRACK31_RUN_ROLE}"
fi

# NUM_STEPS is the immutable scheduler horizon. STOP_AFTER_STEP is intentionally
# optional: preflight and Trainer both resolve an unset value to NUM_STEPS without
# adding it to the schema6 execution contract.

NGPU=${NGPU:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-${PORT:-29631}}

for value_name in NGPU NNODES; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 1
  fi
done
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || ((NODE_RANK >= NNODES)); then
  echo "NODE_RANK must be an integer in [0, NNODES), got: $NODE_RANK" >&2
  exit 1
fi
if [[ ! "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || ((MASTER_PORT > 65535)); then
  echo "MASTER_PORT must be an integer in [1, 65535], got: $MASTER_PORT" >&2
  exit 1
fi
if ((NNODES > 1)) && [[ "$MASTER_ADDR" == "127.0.0.1" ]]; then
  echo "MASTER_ADDR must be reachable by every node when NNODES > 1" >&2
  exit 1
fi

WORLD_SIZE=$((NNODES * NGPU))
export N0_TRACK31_EXPECTED_WORLD_SIZE=$WORLD_SIZE
if [[ -n "${N0_TRACK31_PYTHON_OVERLAY:-}" ]]; then
  if [[ ! -d "$N0_TRACK31_PYTHON_OVERLAY" ]]; then
    echo "N0_TRACK31_PYTHON_OVERLAY is not a directory: $N0_TRACK31_PYTHON_OVERLAY" >&2
    exit 1
  fi
  export PYTHONPATH="$N0_TRACK31_PYTHON_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD:$PWD/n0_twam"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=${PYTHONHASHSEED:-20260801}

PYTHON_BIN=${N0_TRACK31_PYTHON_BIN:-/usr/bin/python3}
TORCHRUN_BIN=${N0_TRACK31_TORCHRUN_BIN:-/usr/local/bin/torchrun}

orchestrated_preflight_status() {
  local protocol=${N0_TRACK31_ORCHESTRATED_PREFLIGHT:-}
  local receipt=${N0_TRACK31_LAUNCH_RECEIPT:-}
  local receipt_sha=${N0_TRACK31_LAUNCH_RECEIPT_SHA256:-}
  local launch_sha=${N0_TRACK31_LAUNCH_MANIFEST_SHA256:-}
  local invocation=${N0_TRACK31_INVOCATION_ID:-}
  local source_sha=${N0_TRACK31_SOURCE_MANIFEST_SHA256:-}
  local overlay_sha=${N0_TRACK31_OVERLAY_MANIFEST_SHA256:-}
  local image_id=${N0_TRACK31_IMAGE_ID:-}
  local known_hosts_sha=${N0_TRACK31_KNOWN_HOSTS_SHA256:-}
  local environment_manifest=${N0_TRACK31_ENVIRONMENT_MANIFEST:-}
  local environment_sha=${N0_TRACK31_ENVIRONMENT_MANIFEST_SHA256:-}
  local collective_report=${N0_TRACK31_COLLECTIVE_SMOKE_REPORT:-}
  local collective_sha=${N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256:-}
  local collective_max_age=${N0_TRACK31_COLLECTIVE_SMOKE_MAX_AGE_SECONDS:-900}
  local node_address=${N0_TRACK31_NODE_ADDRESS:-}
  local container_id=${N0_TRACK31_CONTAINER_ID:-}
  local nodes=${N0_TRACK31_NODES:-}
  local hcu_order=${HIP_VISIBLE_DEVICES:-}
  local batch_size=${N0_TRACK31_BATCH_SIZE:-}
  local gradient_accumulation_steps=${N0_TRACK31_GRADIENT_ACCUMULATION_STEPS:-}
  local num_steps=${N0_TRACK31_NUM_STEPS:-}
  local save_interval=${N0_TRACK31_SAVE_INTERVAL:-}
  local val_interval=${N0_TRACK31_VAL_INTERVAL:-}
  if [[ -z "$protocol" ]]; then
    if [[ -n "$receipt" || -n "$receipt_sha" || -n "$launch_sha" ]]; then
      echo "partial orchestrated preflight configuration is forbidden" >&2
      return 2
    fi
    return 1
  fi
  if [[ "$protocol" != "track31-stage-a-v2" ]]; then
    echo "unknown N0_TRACK31_ORCHESTRATED_PREFLIGHT protocol" >&2
    return 2
  fi
  if [[ ! "$invocation" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] \
      || [[ ! "$receipt_sha" =~ ^[0-9a-f]{64}$ ]] \
      || [[ "$launch_sha" != "$receipt_sha" ]] \
      || [[ ! "$source_sha" =~ ^[0-9a-f]{64}$ ]] \
      || [[ ! "$overlay_sha" =~ ^[0-9a-f]{64}$ ]] \
      || [[ ! "$known_hosts_sha" =~ ^[0-9a-f]{64}$ ]] \
      || [[ ! "$environment_sha" =~ ^[0-9a-f]{64}$ ]] \
      || [[ ! "$collective_sha" =~ ^[0-9a-f]{64}$ ]] \
      || [[ ! "$collective_max_age" =~ ^[1-9][0-9]*$ ]] \
      || ((collective_max_age > 3600)) \
      || [[ -z "$node_address" ]] \
      || [[ ! "$container_id" =~ ^[0-9a-f]{64}$ ]] \
      || [[ -z "$nodes" ]] \
      || [[ ! "$num_steps" =~ ^[1-9][0-9]*$ ]] \
      || [[ ! "$save_interval" =~ ^[1-9][0-9]*$ ]] \
      || [[ ! "$val_interval" =~ ^[1-9][0-9]*$ ]] \
      || [[ "${N0_TRACK31_SSH_PORT:-}" != "36000" ]] \
      || [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "invalid orchestrated preflight identity" >&2
    return 2
  fi
  local expected_gradient_accumulation_steps=1
  if [[ "$N0_TRACK31_TRAIN_PROFILE" != "multitask_pretrain_v1" ]] \
      || [[ "$N0_TRACK31_RUN_ROLE" != "development" \
            && "$N0_TRACK31_RUN_ROLE" != "final_refit" ]] \
      || [[ "$NNODES" != "2" && "$NNODES" != "4" && "$NNODES" != "6" ]] \
      || [[ "$NGPU" != "8" ]] \
      || [[ "$hcu_order" != "0,1,5,4,2,3,7,6" ]] \
      || [[ "$batch_size" != "1" ]] \
      || [[ "$gradient_accumulation_steps" != \
            "$expected_gradient_accumulation_steps" ]] \
      || [[ "${NCCL_IB_DISABLE:-}" != "0" ]] \
      || [[ "${NCCL_IB_GID_INDEX:-}" != "3" ]] \
      || [[ "${NCCL_IB_QPS_PER_CONNECTION:-}" != "4" ]] \
      || [[ "${NCCL_IB_TC:-}" != "160" ]] \
      || [[ "${NCCL_IB_TIMEOUT:-}" != "22" ]] \
      || [[ "${NCCL_NET_GDR_LEVEL:-}" != "2" ]] \
      || [[ "${NCCL_ROCE_SRC_PORT_LIST:-}" != "60000,60051,57663,57804" ]] \
      || [[ "${NCCL_SOCKET_IFNAME:-}" != "bond1" ]] \
      || [[ "${NCCL_DEBUG:-}" != "INFO" ]] \
      || [[ "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}" != "1" ]] \
      || [[ "${TORCH_NCCL_BLOCKING_WAIT:-}" != "1" ]]; then
    echo "orchestrated preflight is restricted to the exact formal Stage A runtime" >&2
    return 2
  fi
  local expected_receipt="${N0_TRACK31_SAVE_ROOT}/launch_manifests/launch_manifest.${invocation}.json"
  if [[ "$receipt" != "$expected_receipt" ]] || [[ ! -f "$receipt" ]] \
      || [[ -L "$receipt" ]] || [[ "$(realpath "$receipt")" != "$receipt" ]]; then
    echo "orchestrated preflight receipt path is not canonical" >&2
    return 2
  fi
  local expected_collective_report="$N0_TRACK31_SAVE_ROOT/preflight/collective_smoke.${invocation}.json"
  if [[ "$collective_report" != "$expected_collective_report" ]] \
      || [[ ! -f "$collective_report" ]] || [[ -L "$collective_report" ]] \
      || [[ "$(realpath "$collective_report")" != "$collective_report" ]]; then
    echo "collective smoke report path is not canonical" >&2
    return 2
  fi
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -B \
    "$SCRIPT_ROOT/script/track3_1/validate_track31_collective_smoke.py" \
    --report-path "$collective_report" \
    --expected-report-sha256 "$collective_sha" \
    --expected-smoke-id "$invocation" \
    --expected-nodes "$nodes" \
    --expected-hcu-order "$hcu_order" \
    --expected-nccl-env NCCL_IB_DISABLE=0 \
    --expected-nccl-env NCCL_IB_GID_INDEX=3 \
    --expected-nccl-env NCCL_IB_QPS_PER_CONNECTION=4 \
    --expected-nccl-env NCCL_IB_TC=160 \
    --expected-nccl-env NCCL_IB_TIMEOUT=22 \
    --expected-nccl-env NCCL_NET_GDR_LEVEL=2 \
    --expected-nccl-env NCCL_ROCE_SRC_PORT_LIST=60000,60051,57663,57804 \
    --expected-nccl-env NCCL_SOCKET_IFNAME=bond1 \
    --expected-nccl-env NCCL_DEBUG=INFO \
    --expected-nccl-env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --expected-nccl-env TORCH_NCCL_BLOCKING_WAIT=1 \
    --expected-image-id "$image_id" \
    --expected-source-manifest-sha256 "$source_sha" \
    --expected-overlay-manifest-sha256 "$overlay_sha" \
    --expected-run-role "$N0_TRACK31_RUN_ROLE" \
    --expected-batch-size "$batch_size" \
    --expected-gradient-accumulation-steps "$gradient_accumulation_steps" \
    --expected-world-size "$((NNODES * NGPU))" \
    --expected-node-count "$NNODES" \
    --expected-hcu-per-node "$NGPU" \
    --expected-container-id "$node_address=$container_id" \
    --max-age-seconds "$collective_max_age" || {
      echo "collective smoke report validation failed" >&2
      return 2
    }
  /usr/bin/python3 -c 'import hashlib,json,os,stat,sys
p,sha,inv,source,overlay,image_id,empty_sha,source_manifest,overlay_manifest,empty_path,known_hosts_sha,environment_manifest,environment_sha,collective_report,collective_sha,nnodes,run_role,master,stop,resume,num_steps,save_interval,val_interval=sys.argv[1:]
def require(value):
    if not value: raise SystemExit(1)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def file_digest(path):
    st=os.lstat(path)
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode))
    return digest(open(path,"rb").read())
def resume_identity(path, expected):
    raw={}
    for name in ("training_state.json","checkpoint_complete.json","train_meta.json"):
        sidecar=os.path.join(path,name); st=os.lstat(sidecar)
        require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode) and st.st_size <= 1048576)
        raw[name]=open(sidecar,"rb").read()
    state=json.loads(raw["training_state.json"]); complete=json.loads(raw["checkpoint_complete.json"])
    require(state.get("step") == expected and not isinstance(state.get("step"),bool))
    require(complete.get("status") == "complete" and complete.get("step") == expected)
    transformer=state.get("transformer_identity"); require(isinstance(transformer,dict))
    transformer_sha=transformer.get("sha256"); require(isinstance(transformer_sha,str) and len(transformer_sha) == 64)
    require(complete.get("transformer_identity") == transformer)
    return {"path":path,"step":expected,"training_state_sha256":digest(raw["training_state.json"]),"checkpoint_complete_sha256":digest(raw["checkpoint_complete.json"]),"train_meta_sha256":digest(raw["train_meta.json"]),"transformer_sha256":transformer_sha}
st=os.lstat(p)
require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode))
require(st.st_mode & 0o222 == 0)
raw=open(p,"rb").read()
require(hashlib.sha256(raw).hexdigest() == sha)
d=json.loads(raw)
node_count=int(nnodes); require(node_count in (2,4,6))
require(d.get("schema_version") == 2 and d.get("invocation_id") == inv)
require(d.get("source_manifest_sha256") == source)
require(d.get("overlay_manifest_sha256") == overlay)
require(d.get("image_id") == image_id)
require(d.get("empty_embedding_sha256") == empty_sha)
require(d.get("ssh_port") == 36000 and d.get("known_hosts_sha256") == known_hosts_sha)
require(d.get("environment_manifest") == {"path":environment_manifest,"sha256":environment_sha})
require(file_digest(environment_manifest) == environment_sha)
require(d.get("collective_smoke") == {"path":collective_report,"sha256":collective_sha})
require(file_digest(collective_report) == collective_sha)
require(file_digest(source_manifest) == source and file_digest(overlay_manifest) == overlay)
require(file_digest(empty_path) == empty_sha)
require(d.get("nodes") and len(d["nodes"]) == node_count and d["nodes"][0] == master)
require(d.get("run_role") == run_role)
contracts={"development":{"physical_split":"train759","train_view_id":"stage_a_dev719_v1","train_episode_count":719,"validation_view_id":"internal_dev40_v1","normalizer_id":"qpos8_dev719_v1"},"final_refit":{"physical_split":"train759","train_view_id":"stage_a_final759_v1","train_episode_count":759,"validation_view_id":None,"normalizer_id":"qpos8_final759_v1"}}
require(run_role in contracts and d.get("dataset_contract") == contracts[run_role])
require(d.get("topology") == {"node_count":node_count,"hcu_per_node":8,"world_size":node_count*8})
require(d.get("hcu") == {"per_node":8,"order":"0,1,5,4,2,3,7,6"})
require(d.get("nccl") == {"NCCL_IB_DISABLE":"0","NCCL_IB_GID_INDEX":"3","NCCL_IB_QPS_PER_CONNECTION":"4","NCCL_IB_TC":"160","NCCL_IB_TIMEOUT":"22","NCCL_NET_GDR_LEVEL":"2","NCCL_ROCE_SRC_PORT_LIST":"60000,60051,57663,57804","NCCL_SOCKET_IFNAME":"bond1","NCCL_DEBUG":"INFO","TORCH_NCCL_ASYNC_ERROR_HANDLING":"1","TORCH_NCCL_BLOCKING_WAIT":"1"})
require(d.get("multinode_hcu_preflight") == "passed")
expected_grad_acc=1
require(d.get("recipe") == {"num_steps":int(num_steps),"batch_size":1,"gradient_accumulation_steps":expected_grad_acc,"save_interval":int(save_interval),"val_interval":int(val_interval),"max_latent_frames":5,"load_worker":0,"train_seed":20260801,"action_init_seed":0,"pythonhashseed":20260801})
require(d.get("rank0_foreground_preflight") == {"node":master,"node_rank":0,"status":"passed"})
stop_int=int(stop); require(d.get("stop_after_step") == stop_int)
require(d.get("resume_from") == (resume or None))
expected_parent=None if stop_int == 20 else resume_identity(resume,20 if stop_int == 25 else 25)
require(d.get("resume_parent_identity") == expected_parent)
require(d.get("provenance_trust_boundary","").startswith("trusted-operator provenance only"))' \
    "$receipt" "$receipt_sha" "$invocation" "$source_sha" \
    "$overlay_sha" "$image_id" "$N0_EMPTY_EMBEDDING_SHA256" \
    "$PWD/.source_manifest.sha256" \
    "$N0_TRACK31_PYTHON_OVERLAY/.overlay_manifest.sha256" \
    "$N0_EMPTY_EMBEDDING" "$known_hosts_sha" "$environment_manifest" \
    "$environment_sha" "$collective_report" "$collective_sha" \
    "$NNODES" "$N0_TRACK31_RUN_ROLE" "$MASTER_ADDR" \
    "${N0_TRACK31_STOP_AFTER_STEP:-$N0_TRACK31_NUM_STEPS}" \
    "${N0_TRACK31_RESUME_FROM:-}" "$num_steps" "$save_interval" \
    "$val_interval" || {
      echo "orchestrated preflight receipt validation failed" >&2
      return 2
    }
  return 0
}

if orchestrated_preflight_status; then
  echo "validated trusted-operator launch provenance; full shell preflight remains rank-zero-only" >&2
else
  preflight_status=$?
  if ((preflight_status == 1)); then
    "$PYTHON_BIN" script/track3_1/preflight_train.py
  else
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
exec "$TORCHRUN_BIN" \
  --nnodes="$NNODES" \
  --nproc-per-node="$NGPU" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  --tee 3 \
  -m n0_twam.train --config-name track31_univtac
