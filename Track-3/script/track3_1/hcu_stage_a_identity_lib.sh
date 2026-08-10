#!/usr/bin/env bash
# Formal recipe, frozen-input, and launch-provenance gates for Stage A.

require_formal_configuration() {
    local names=(
        N0_TRACK31_IMAGE N0_TRACK31_IMAGE_ID N0_TRACK31_SOURCE_ROOT_HOST
        N0_TRACK31_SOURCE_MANIFEST_SHA256 N0_TRACK31_ARTIFACT_ROOT_HOST
        N0_TRACK31_TRAIN759_ROOT_HOST N0_TRACK31_RAW_ROOT_HOST
        N0_TRACK31_MODEL_ROOT_HOST N0_TRACK31_OVERLAY_ROOT_HOST
        N0_TRACK31_OVERLAY_MANIFEST_SHA256 N0_EMPTY_EMBEDDING_SHA256
        N0_TRACK31_RUN_ROOT_HOST N0_RELEASED_TRANSFORMER_SHA256
    ) name
    for name in "${names[@]}"; do require_env "$name"; done
    for name in N0_TRACK31_SOURCE_ROOT_HOST N0_TRACK31_ARTIFACT_ROOT_HOST \
        N0_TRACK31_TRAIN759_ROOT_HOST N0_TRACK31_RAW_ROOT_HOST \
        N0_TRACK31_MODEL_ROOT_HOST N0_TRACK31_OVERLAY_ROOT_HOST \
        N0_TRACK31_RUN_ROOT_HOST; do
        require_safe_absolute_path "$name"
    done
    validate_image_reference
    for name in N0_TRACK31_SOURCE_MANIFEST_SHA256 \
        N0_TRACK31_OVERLAY_MANIFEST_SHA256 N0_EMPTY_EMBEDDING_SHA256 \
        N0_RELEASED_TRANSFORMER_SHA256; do
        [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || \
            die "${name} must be a lowercase SHA256"
    done
    [[ "$(basename -- "$N0_TRACK31_TRAIN759_ROOT_HOST")" == "train759" ]] || \
        die "N0_TRACK31_TRAIN759_ROOT_HOST must name the sealed train759 repo"
}

require_formal_recipe_identity() {
    require_env N0_TRACK31_INVOCATION_ID
    require_env N0_TRACK31_RUN_ROLE
    validate_invocation_id
    case "$N0_TRACK31_RUN_ROLE" in
        development|final_refit) ;;
        *) die "N0_TRACK31_RUN_ROLE must be development or final_refit" ;;
    esac
    N0_TRACK31_NUM_STEPS=${N0_TRACK31_NUM_STEPS:-1500}
    require_fixed_value N0_TRACK31_NUM_STEPS 1500
    require_fixed_value N0_TRACK31_BATCH_SIZE 1
    FORMAL_GRADIENT_ACCUMULATION_STEPS=1
    require_fixed_value N0_TRACK31_GRADIENT_ACCUMULATION_STEPS \
        "$FORMAL_GRADIENT_ACCUMULATION_STEPS"
    N0_TRACK31_SAVE_INTERVAL=${N0_TRACK31_SAVE_INTERVAL:-300}
    N0_TRACK31_VAL_INTERVAL=${N0_TRACK31_VAL_INTERVAL:-100}
    require_fixed_value N0_TRACK31_SAVE_INTERVAL 300
    require_fixed_value N0_TRACK31_VAL_INTERVAL 100
    require_fixed_value N0_TRACK31_MAX_LATENT_FRAMES 5
    require_fixed_value N0_TRACK31_LOAD_WORKER 0
    require_fixed_value N0_TRAIN_SEED 20260801
    require_fixed_value N0_ACTION_INIT_SEED 0
    require_fixed_value PYTHONHASHSEED 20260801
    N0_FSDP_TOPOLOGY=${N0_FSDP_TOPOLOGY:-global_shard}
    case "$N0_FSDP_TOPOLOGY" in
        global_shard|hsdp) ;;
        *) die "N0_FSDP_TOPOLOGY must be global_shard or hsdp" ;;
    esac
    N0_FSDP_SHARD_SIZE=${N0_FSDP_SHARD_SIZE:-8}
    require_fixed_value N0_FSDP_SHARD_SIZE 8
    N0_FSDP_EXPERT_RESHARD_POLICY=${N0_FSDP_EXPERT_RESHARD_POLICY:-after_layer}
    case "$N0_FSDP_EXPERT_RESHARD_POLICY" in
        after_layer|after_backward) ;;
        *) die "formal expert reshard policy must be after_layer or after_backward" ;;
    esac
    N0_TRACK31_SAMPLER_RANK_ALIGNMENT=${N0_TRACK31_SAMPLER_RANK_ALIGNMENT:-contiguous}
    case "$N0_TRACK31_SAMPLER_RANK_ALIGNMENT" in
        contiguous|shape_balanced) ;;
        *) die "invalid N0_TRACK31_SAMPLER_RANK_ALIGNMENT" ;;
    esac
    N0_MOT_CROSS_ATTENTION_BACKEND=${N0_MOT_CROSS_ATTENTION_BACKEND:-sdpa}
    case "$N0_MOT_CROSS_ATTENTION_BACKEND" in
        sdpa|flash_attn) ;;
        *) die "invalid N0_MOT_CROSS_ATTENTION_BACKEND" ;;
    esac
    N0_TRACK31_MASTER_PORT=${N0_TRACK31_MASTER_PORT:-29660}
    require_positive_integer N0_TRACK31_MASTER_PORT
    ((N0_TRACK31_MASTER_PORT <= 65535)) || die "N0_TRACK31_MASTER_PORT exceeds 65535"
    HEALTH_ATTEMPTS=${N0_TRACK31_HEALTH_ATTEMPTS:-$DEFAULT_HEALTH_ATTEMPTS}
    HEALTH_POLL_SECONDS=${N0_TRACK31_HEALTH_POLL_SECONDS:-$DEFAULT_HEALTH_POLL_SECONDS}
    [[ "$HEALTH_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || die "invalid health attempts"
    [[ "$HEALTH_POLL_SECONDS" =~ ^[0-9]+$ ]] || die "invalid health poll seconds"
    ((HEALTH_ATTEMPTS <= 120 && HEALTH_POLL_SECONDS <= 30)) || \
        die "health polling bounds are unsafe"
}

require_formal_stop_contract() {
    local expected_parent_step
    require_env N0_TRACK31_STOP_AFTER_STEP
    case "$N0_TRACK31_STOP_AFTER_STEP" in
        20) [[ -z "${N0_TRACK31_RESUME_FROM:-}" ]] || \
                die "step 20 migration probe must not resume" ;;
        25|1500)
            require_safe_absolute_path N0_TRACK31_RESUME_FROM
            [[ "$N0_TRACK31_RESUME_FROM" == "$CONTAINER_RUN_ROOT/checkpoints/"* ]] || \
                die "resume checkpoint must be inside the formal run namespace"
            expected_parent_step=20
            [[ "$N0_TRACK31_STOP_AFTER_STEP" == "25" ]] || expected_parent_step=25
            [[ "$(basename -- "$N0_TRACK31_RESUME_FROM")" == \
                "checkpoint_step_${expected_parent_step}" ]] || \
                die "STOP ${N0_TRACK31_STOP_AFTER_STEP} requires checkpoint_step_${expected_parent_step}" ;;
        *) die "N0_TRACK31_STOP_AFTER_STEP must be exactly 20, 25, or 1500" ;;
    esac
}

require_collective_smoke_evidence_configuration() {
    local expected_report
    require_safe_absolute_path N0_TRACK31_COLLECTIVE_SMOKE_REPORT
    require_env N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256
    expected_report="$CONTAINER_RUN_ROOT/preflight/collective_smoke.${N0_TRACK31_INVOCATION_ID}.json"
    [[ "$N0_TRACK31_COLLECTIVE_SMOKE_REPORT" == "$expected_report" ]] || \
        die "collective smoke report must be invocation-specific: ${expected_report}"
    [[ "$N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
        die "collective smoke report SHA256 must be lowercase"
    COLLECTIVE_SMOKE_MAX_AGE_SECONDS=${N0_TRACK31_COLLECTIVE_SMOKE_MAX_AGE_SECONDS:-900}
    [[ "$COLLECTIVE_SMOKE_MAX_AGE_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
        die "invalid collective smoke max age"
    ((COLLECTIVE_SMOKE_MAX_AGE_SECONDS <= 3600)) || \
        die "collective smoke max age exceeds the formal bound"
}

require_formal_recipe() {
    require_formal_recipe_identity
    require_formal_stop_contract
    require_collective_smoke_evidence_configuration
}

verify_collective_smoke_report() {
    local host_path
    host_path="$N0_TRACK31_RUN_ROOT_HOST/${N0_TRACK31_COLLECTIVE_SMOKE_REPORT#"$CONTAINER_RUN_ROOT/"}"
    validate_formal_collective_smoke_report "$host_path" \
        "$N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256"
}

expected_mounts() {
    local dataset_parent
    dataset_parent=$(dirname -- "$N0_TRACK31_TRAIN759_ROOT_HOST")
    printf 'bind\t%s\t%s\t%s\n' \
        /opt/hyhal /opt/hyhal false /etc/hfm /etc/hfm false \
        "$N0_TRACK31_SOURCE_ROOT_HOST" "$CONTAINER_SOURCE_ROOT" false \
        "$N0_TRACK31_ARTIFACT_ROOT_HOST" "$CONTAINER_ARTIFACT_ROOT" false \
        "$N0_TRACK31_TRAIN759_ROOT_HOST" "$N0_TRACK31_TRAIN759_ROOT_HOST" false \
        "$dataset_parent/$MATERIALIZATION_MARKER_NAME" \
            "$dataset_parent/$MATERIALIZATION_MARKER_NAME" false \
        "$N0_TRACK31_MODEL_ROOT_HOST" "$CONTAINER_MODEL_ROOT" false \
        "$N0_TRACK31_OVERLAY_ROOT_HOST" "$CONTAINER_OVERLAY_ROOT" false \
        "$N0_TRACK31_RUN_ROOT_HOST" "$CONTAINER_RUN_ROOT" true
}

verify_host_roster_identity() {
    local host=$1 root=$2 manifest_name=$3 expected_sha=$4 actual manifest_path
    manifest_path="$root/$manifest_name"
    remote_exec "$host" test -f "$manifest_path" || return 1
    actual=$(remote_exec "$host" sha256sum "$manifest_path") || return 1
    [[ "${actual%% *}" == "$expected_sha" ]] || return 1
    verify_roster_host "$host" "$root" "$manifest_name"
}

verify_host_inputs() {
    local host=$1 actual dataset_parent path raw_real
    verify_exact_image "$host" "$N0_TRACK31_IMAGE" "$N0_TRACK31_IMAGE_ID" || return 1
    for path in "$N0_TRACK31_SOURCE_ROOT_HOST" "$N0_TRACK31_ARTIFACT_ROOT_HOST" \
        "$N0_TRACK31_TRAIN759_ROOT_HOST" "$N0_TRACK31_RAW_ROOT_HOST" \
        "$N0_TRACK31_MODEL_ROOT_HOST" "$N0_TRACK31_OVERLAY_ROOT_HOST" \
        "$N0_TRACK31_RUN_ROOT_HOST"; do
        remote_exec "$host" test -d "$path" || return 1
        [[ "$(remote_exec "$host" realpath -e "$path")" == "${path%/}" ]] || return 1
    done
    verify_host_roster_identity "$host" "$N0_TRACK31_SOURCE_ROOT_HOST" \
        "$SOURCE_MANIFEST_NAME" "$N0_TRACK31_SOURCE_MANIFEST_SHA256" || return 1
    verify_host_roster_identity "$host" "$N0_TRACK31_OVERLAY_ROOT_HOST" \
        "$OVERLAY_MANIFEST_NAME" "$N0_TRACK31_OVERLAY_MANIFEST_SHA256" || return 1
    path="$N0_TRACK31_MODEL_ROOT_HOST/empty_emb.pt"
    remote_exec "$host" test -f "$path" || return 1
    remote_exec "$host" test ! -L "$path" || return 1
    actual=$(remote_exec "$host" sha256sum "$path") || return 1
    [[ "${actual%% *}" == "$N0_EMPTY_EMBEDDING_SHA256" ]] || return 1
    dataset_parent=$(dirname -- "$N0_TRACK31_TRAIN759_ROOT_HOST")
    path="$dataset_parent/$MATERIALIZATION_MARKER_NAME"
    remote_exec "$host" test -f "$path" || return 1
    [[ "$(remote_exec "$host" realpath -e "$path")" == "$path" ]] || return 1
    remote_exec "$host" test ! -e "$N0_TRACK31_TRAIN759_ROOT_HOST/frozen40" || return 1
    [[ "$(remote_exec "$host" realpath -e "$dataset_parent/train759")" == \
        "$N0_TRACK31_TRAIN759_ROOT_HOST" ]] || return 1
    [[ "$(remote_exec "$host" realpath -e "$dataset_parent/frozen40")" == \
        "$dataset_parent/frozen40" ]] || return 1
    raw_real=$(remote_exec "$host" realpath -e "$N0_TRACK31_RAW_ROOT_HOST") || return 1
    paths_overlap "$dataset_parent" "$raw_real" && return 1
    for path in "$N0_TRACK31_SOURCE_ROOT_HOST" "$N0_TRACK31_ARTIFACT_ROOT_HOST" \
        "$N0_TRACK31_MODEL_ROOT_HOST" "$N0_TRACK31_OVERLAY_ROOT_HOST" \
        "$N0_TRACK31_RUN_ROOT_HOST"; do
        ! paths_overlap "$path" "$dataset_parent" || return 1
        ! paths_overlap "$path" "$raw_real" || return 1
    done
}

verify_container_frozen_inputs() {
    local host=$1 actual
    verify_roster_container "$host" "$N0_TRACK31_CONTAINER_NAME" \
        "$CONTAINER_SOURCE_ROOT" "$SOURCE_MANIFEST_NAME" || return 1
    verify_roster_container "$host" "$N0_TRACK31_CONTAINER_NAME" \
        "$CONTAINER_OVERLAY_ROOT" "$OVERLAY_MANIFEST_NAME" || return 1
    remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" \
        test ! -L "$CONTAINER_MODEL_ROOT/empty_emb.pt" || return 1
    actual=$(remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" \
        sha256sum "$CONTAINER_MODEL_ROOT/empty_emb.pt") || return 1
    [[ "${actual%% *}" == "$N0_EMPTY_EMBEDDING_SHA256" ]]
}

lightweight_hcu_check() {
    local host=$1 program='import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
for index in range(8): assert torch.ones(1,device=f"cuda:{index}").item() == 1.0
torch.cuda.synchronize()'
    remote_exec "$host" docker exec -e "HIP_VISIBLE_DEVICES=$HCU_ORDER" \
        "$N0_TRACK31_CONTAINER_NAME" bash -lc \
        'command -v setsid >/dev/null; source /opt/hyhal/env.sh; exec /usr/bin/python3 -c "$1"' \
        _ "$program"
}

resolve_resume_parent_identity() {
    local expected_step host_path program
    RESUME_PARENT_IDENTITY_JSON=null
    [[ "$N0_TRACK31_STOP_AFTER_STEP" != "20" ]] || return 0
    [[ "$N0_TRACK31_STOP_AFTER_STEP" == "25" ]] && expected_step=20 || expected_step=25
    host_path="$N0_TRACK31_RUN_ROOT_HOST/${N0_TRACK31_RESUME_FROM#"$CONTAINER_RUN_ROOT/"}"
    program='import hashlib,json,os,stat,sys
root,container_path,expected=sys.argv[1],sys.argv[2],int(sys.argv[3])
if os.path.realpath(root) != root: raise SystemExit(1)
names=("training_state.json","checkpoint_complete.json","train_meta.json"); raw={}
for name in names:
 path=os.path.join(root,name); metadata=os.lstat(path); mode=metadata.st_mode
 if not stat.S_ISREG(mode) or stat.S_ISLNK(mode) or metadata.st_size > 1048576: raise SystemExit(1)
 raw[name]=open(path,"rb").read()
state=json.loads(raw["training_state.json"]); complete=json.loads(raw["checkpoint_complete.json"]); step=state.get("step")
if isinstance(step,bool) or step != expected or complete.get("status") != "complete" or complete.get("step") != expected: raise SystemExit(1)
transformer=state.get("transformer_identity"); transformer_sha=transformer.get("sha256") if isinstance(transformer,dict) else None
if not isinstance(transformer_sha,str) or len(transformer_sha) != 64 or complete.get("transformer_identity") != transformer: raise SystemExit(1)
digest=lambda name: hashlib.sha256(raw[name]).hexdigest()
print(json.dumps({"path":container_path,"step":expected,"training_state_sha256":digest("training_state.json"),"checkpoint_complete_sha256":digest("checkpoint_complete.json"),"train_meta_sha256":digest("train_meta.json"),"transformer_sha256":transformer_sha},sort_keys=True,separators=(",",":")))'
    RESUME_PARENT_IDENTITY_JSON=$(remote_exec "${NODES[0]}" python3 -c \
        "$program" "$host_path" "$N0_TRACK31_RESUME_FROM" "$expected_step") || return 1
}

prepare_launch_manifest() {
    local resume=${N0_TRACK31_RESUME_FROM:-}
    LAUNCH_MANIFEST_PAYLOAD=$(python3 -c 'import json,sys
nodes=sys.argv[2].split(","); node_count=int(sys.argv[14]); hcu_per_node=8; grad_acc=int(sys.argv[15])
if len(nodes) != node_count or node_count not in (2,4,6): raise SystemExit(1)
role=sys.argv[13]
contracts={"development":{"physical_split":"train759","train_view_id":"stage_a_dev719_v1","train_episode_count":719,"validation_view_id":"internal_dev40_v1","normalizer_id":"qpos8_dev719_v1"},"final_refit":{"physical_split":"train759","train_view_id":"stage_a_final759_v1","train_episode_count":759,"validation_view_id":None,"normalizer_id":"qpos8_final759_v1"}}
if role not in contracts: raise SystemExit(1)
print(json.dumps({"schema_version":2,"invocation_id":sys.argv[1],"nodes":nodes,"run_role":role,"dataset_contract":contracts[role],"topology":{"node_count":node_count,"hcu_per_node":hcu_per_node,"world_size":node_count*hcu_per_node},"collective_smoke":{"path":sys.argv[19],"sha256":sys.argv[20]},"image_id":sys.argv[3],"source_manifest_sha256":sys.argv[4],"overlay_manifest_sha256":sys.argv[5],"empty_embedding_sha256":sys.argv[6],"ssh_port":36000,"known_hosts_sha256":sys.argv[10],"environment_manifest":{"path":sys.argv[11],"sha256":sys.argv[12]},"rank0_foreground_preflight":{"node":nodes[0],"node_rank":0,"status":"passed"},"multinode_hcu_preflight":"passed","hcu":{"per_node":hcu_per_node,"order":"0,1,5,4,2,3,7,6"},"nccl":{"NCCL_IB_DISABLE":"0","NCCL_IB_GID_INDEX":"3","NCCL_IB_QPS_PER_CONNECTION":"4","NCCL_IB_TC":"160","NCCL_IB_TIMEOUT":"22","NCCL_NET_GDR_LEVEL":"2","NCCL_ROCE_SRC_PORT_LIST":"60000,60051,57663,57804","NCCL_SOCKET_IFNAME":"bond1","NCCL_DEBUG":"INFO","TORCH_NCCL_ASYNC_ERROR_HANDLING":"1","TORCH_NCCL_BLOCKING_WAIT":"1"},"recipe":{"num_steps":int(sys.argv[16]),"batch_size":1,"gradient_accumulation_steps":grad_acc,"save_interval":int(sys.argv[17]),"val_interval":int(sys.argv[18]),"max_latent_frames":5,"load_worker":0,"train_seed":20260801,"action_init_seed":0,"pythonhashseed":20260801},"stop_after_step":int(sys.argv[7]),"resume_from":sys.argv[8] or None,"resume_parent_identity":json.loads(sys.argv[9]),"provenance_trust_boundary":"trusted-operator provenance only; not an attestation against a caller controlling the same UID, environment, or save root; Trainer rank zero independently revalidates training identity"},sort_keys=True,separators=(",",":"))+"\n")' \
        "$N0_TRACK31_INVOCATION_ID" "$N0_TRACK31_NODES" "$N0_TRACK31_IMAGE_ID" \
        "$N0_TRACK31_SOURCE_MANIFEST_SHA256" "$N0_TRACK31_OVERLAY_MANIFEST_SHA256" \
        "$N0_EMPTY_EMBEDDING_SHA256" "$N0_TRACK31_STOP_AFTER_STEP" "$resume" \
        "$RESUME_PARENT_IDENTITY_JSON" "$KNOWN_HOSTS_SHA256" \
        "$ENVIRONMENT_MANIFEST_CONTAINER_PATH" "$ENVIRONMENT_MANIFEST_SHA256" \
        "$N0_TRACK31_RUN_ROLE" "$FORMAL_NODE_COUNT" \
        "$FORMAL_GRADIENT_ACCUMULATION_STEPS" \
        "$N0_TRACK31_NUM_STEPS" "$N0_TRACK31_SAVE_INTERVAL" \
        "$N0_TRACK31_VAL_INTERVAL" \
        "$N0_TRACK31_COLLECTIVE_SMOKE_REPORT" \
        "$N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256")
    LAUNCH_MANIFEST_PAYLOAD=$(python3 -c 'import json,sys
payload=json.loads(sys.argv[1])
node_count=payload["topology"]["node_count"]; world_size=payload["topology"]["world_size"]; topology=sys.argv[2]; shard_size=int(sys.argv[3]); replicate_size=node_count if topology=="hsdp" else 1; mesh_shape=[replicate_size,shard_size] if topology=="hsdp" else [world_size]
payload["performance"]={"activation_checkpointing":False,"attention_backend":"grouped_flash_attn","mot_cross_attention_backend":sys.argv[6],"fsdp":{"topology":topology,"mesh_shape":mesh_shape,"replicate_size":replicate_size,"shard_size":shard_size if topology=="hsdp" else world_size,"expert_reshard_policy":sys.argv[4],"keep_expert_params_between_pre_post":True,"reduce_dtype":"bfloat16"},"sampler_rank_alignment":sys.argv[5],"sync_attention_window":True,"torchinductor_compile_threads":1}
print(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n")' \
        "$LAUNCH_MANIFEST_PAYLOAD" "$N0_FSDP_TOPOLOGY" \
        "$N0_FSDP_SHARD_SIZE" "$N0_FSDP_EXPERT_RESHARD_POLICY" \
        "$N0_TRACK31_SAMPLER_RANK_ALIGNMENT" \
        "$N0_MOT_CROSS_ATTENTION_BACKEND")
    LAUNCH_MANIFEST_HOST_PATH="$N0_TRACK31_RUN_ROOT_HOST/launch_manifests/launch_manifest.${N0_TRACK31_INVOCATION_ID}.json"
    LAUNCH_RECEIPT_CONTAINER_PATH="$CONTAINER_RUN_ROOT/launch_manifests/launch_manifest.${N0_TRACK31_INVOCATION_ID}.json"
    LAUNCH_RECEIPT_SHA256=$(printf '%s' "$LAUNCH_MANIFEST_PAYLOAD" | sha256sum)
    LAUNCH_RECEIPT_SHA256=${LAUNCH_RECEIPT_SHA256%% *}
    [[ "$LAUNCH_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]
}

create_launch_manifest() {
    local sha
    [[ -n "${LAUNCH_MANIFEST_PAYLOAD:-}" \
        && -n "${LAUNCH_MANIFEST_HOST_PATH:-}" \
        && -n "${LAUNCH_RECEIPT_CONTAINER_PATH:-}" \
        && "$LAUNCH_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || return 1
    publish_atomic_file "${NODES[0]}" "$LAUNCH_MANIFEST_HOST_PATH" \
        "$LAUNCH_MANIFEST_PAYLOAD" || return 1
    sha=$(remote_exec "${NODES[0]}" sha256sum "$LAUNCH_MANIFEST_HOST_PATH") || \
        return 1
    [[ "${sha%% *}" == "$LAUNCH_RECEIPT_SHA256" ]] || return 1
    echo "launch_manifest=$LAUNCH_MANIFEST_HOST_PATH"
    echo "launch_manifest_sha256=$LAUNCH_RECEIPT_SHA256"
}
