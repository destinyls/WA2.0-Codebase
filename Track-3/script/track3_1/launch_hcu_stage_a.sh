#!/usr/bin/env bash
# Transactional multi-node Track 3.1 Stage A launcher.
set -euo pipefail

readonly HCU_PER_NODE=8
readonly HCU_ORDER="0,1,5,4,2,3,7,6"
readonly CONTAINER_SOURCE_ROOT="/workspace/N0-TWAM"
readonly CONTAINER_ARTIFACT_ROOT="/formal/artifacts"
readonly CONTAINER_MODEL_ROOT="/formal/model"
readonly CONTAINER_OVERLAY_ROOT="/formal/overlay"
readonly CONTAINER_RUN_ROOT="/formal/run"
readonly MATERIALIZATION_MARKER_NAME=".materialization_state.json"
readonly SOURCE_MANIFEST_NAME=".source_manifest.sha256"
readonly OVERLAY_MANIFEST_NAME=".overlay_manifest.sha256"
readonly DEFAULT_HEALTH_ATTEMPTS=30
readonly DEFAULT_HEALTH_POLL_SECONDS=2
readonly REGISTRATION_ACK_POLLS=10
readonly REGISTRATION_ACK_POLL_SECONDS=1
readonly CLEANUP_REGISTRATION_GRACE_POLLS="$REGISTRATION_ACK_POLLS"
readonly CLEANUP_STABLE_ZERO_POLLS=3
readonly STOP_POLLS=15

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=script/track3_1/hcu_stage_a_lib.sh
source "${SCRIPT_DIR}/hcu_stage_a_lib.sh"
# shellcheck source=script/track3_1/hcu_stage_a_identity_lib.sh
source "${SCRIPT_DIR}/hcu_stage_a_identity_lib.sh"
# shellcheck source=script/track3_1/hcu_stage_a_environment_lib.sh
source "${SCRIPT_DIR}/hcu_stage_a_environment_lib.sh"
# shellcheck source=script/track3_1/hcu_stage_a_collective_smoke_lib.sh
source "${SCRIPT_DIR}/hcu_stage_a_collective_smoke_lib.sh"

require_operational_configuration() {
    require_env N0_TRACK31_KNOWN_HOSTS
    require_env N0_TRACK31_NODES
    require_env N0_TRACK31_CONTAINER_NAME
    require_safe_absolute_path N0_TRACK31_KNOWN_HOSTS
    [[ -f "$N0_TRACK31_KNOWN_HOSTS" && -s "$N0_TRACK31_KNOWN_HOSTS" \
        && ! -L "$N0_TRACK31_KNOWN_HOSTS" ]] || \
        die "N0_TRACK31_KNOWN_HOSTS must be a regular non-symlink file"
    KNOWN_HOSTS_SHA256=$(python3 -c 'import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
        "$N0_TRACK31_KNOWN_HOSTS")
    validate_container_name
    require_fixed_value N0_TRACK31_SSH_PORT 36000
    load_formal_nodes
}

create_container() {
    local host=$1 dataset_parent existing
    dataset_parent=$(dirname -- "$N0_TRACK31_TRAIN759_ROOT_HOST")
    existing=$(remote_exec "$host" docker ps -a \
        --filter "name=^/$N0_TRACK31_CONTAINER_NAME$" --format '{{.Names}}') || return 1
    [[ -z "$existing" ]] || {
        [[ "$existing" == "$N0_TRACK31_CONTAINER_NAME" ]] || return 1
        return 2
    }
    if remote_exec "$host" docker run -d --name "$N0_TRACK31_CONTAINER_NAME" \
        --label "n0.track31.creation_token=$CREATE_TOKEN" \
        --label "n0.track31.owner=formal-stage-a-v1" \
        --network host --ipc host --device /dev/kfd --device /dev/dri \
        --group-add video --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined --security-opt label=disable \
        --ulimit nofile=1048576:1048576 --ulimit stack=-1:-1 --ulimit memlock=-1:-1 \
        --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
        --mount type=bind,src=/etc/hfm,dst=/etc/hfm,readonly \
        --mount "type=bind,src=$N0_TRACK31_SOURCE_ROOT_HOST,dst=$CONTAINER_SOURCE_ROOT,readonly" \
        --mount "type=bind,src=$N0_TRACK31_ARTIFACT_ROOT_HOST,dst=$CONTAINER_ARTIFACT_ROOT,readonly" \
        --mount "type=bind,src=$N0_TRACK31_TRAIN759_ROOT_HOST,dst=$N0_TRACK31_TRAIN759_ROOT_HOST,readonly" \
        --mount "type=bind,src=$dataset_parent/$MATERIALIZATION_MARKER_NAME,dst=$dataset_parent/$MATERIALIZATION_MARKER_NAME,readonly" \
        --mount "type=bind,src=$N0_TRACK31_MODEL_ROOT_HOST,dst=$CONTAINER_MODEL_ROOT,readonly" \
        --mount "type=bind,src=$N0_TRACK31_OVERLAY_ROOT_HOST,dst=$CONTAINER_OVERLAY_ROOT,readonly" \
        --mount "type=bind,src=$N0_TRACK31_RUN_ROOT_HOST,dst=$CONTAINER_RUN_ROOT" \
        --workdir "$CONTAINER_SOURCE_ROOT" --entrypoint bash "$N0_TRACK31_IMAGE_ID" \
        -lc 'while :; do sleep 3600; done' >/dev/null; then
        return 0
    fi
    [[ "$(container_creation_token "$host" "$N0_TRACK31_CONTAINER_NAME" || true)" == \
        "$CREATE_TOKEN" ]] && return 3
    return 1
}

rollback_created_containers() {
    local host owner container_id failed=0
    for host in "$@"; do
        container_id=$(container_identity "$host" "$N0_TRACK31_CONTAINER_NAME" || true)
        [[ -n "$container_id" ]] || { failed=1; continue; }
        owner=$(container_creation_token "$host" "$container_id" || true)
        if [[ "$owner" != "$CREATE_TOKEN" ]]; then failed=1; continue; fi
        remote_exec "$host" docker rm -f "$container_id" >/dev/null 2>&1 || failed=1
    done
    return "$failed"
}

create_all_containers() {
    local host rc mounts created=()
    mounts=$(expected_mounts)
    CREATE_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
    for host in "${NODES[@]}"; do
        ensure_run_evidence_dirs "$host" || return 1
        verify_host_inputs "$host" || return 1
    done
    for host in "${NODES[@]}"; do
        if create_container "$host"; then
            created+=("$host")
            if ! verify_container_contract "$host" "$N0_TRACK31_CONTAINER_NAME" \
                "$N0_TRACK31_IMAGE_ID" "$mounts"; then rc=1; else continue; fi
        else
            rc=$?
            [[ "$rc" -eq 3 ]] && created+=("$host")
            [[ "$rc" -eq 2 ]] && echo "${host}: container already exists" >&2
        fi
        rollback_created_containers "${created[@]}" || \
            echo "one or more newly-created containers could not be rolled back" >&2
        return 1
    done
}

build_environment() {
    local host=$1 rank=$2 mode=${3:-launch} dataset_parent container_id
    dataset_parent=$(dirname -- "$N0_TRACK31_TRAIN759_ROOT_HOST")
    container_id=$(container_identity "$host" "$N0_TRACK31_CONTAINER_NAME") || return 1
    DOCKER_ENV=(
        -e "NGPU=$HCU_PER_NODE" -e "NNODES=$FORMAL_NODE_COUNT" -e "NODE_RANK=$rank"
        -e "N0_TRACK31_NODES=$N0_TRACK31_NODES"
        -e "MASTER_ADDR=${NODES[0]}" -e "MASTER_PORT=$N0_TRACK31_MASTER_PORT"
        -e "HIP_VISIBLE_DEVICES=$HCU_ORDER" -e HSA_FORCE_FINE_GRAIN_PCIE=1
        -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_QPS_PER_CONNECTION=4
        -e NCCL_IB_TC=160 -e NCCL_IB_TIMEOUT=22 -e NCCL_NET_GDR_LEVEL=2
        -e NCCL_ROCE_SRC_PORT_LIST=60000,60051,57663,57804 -e NCCL_SOCKET_IFNAME=bond1
        -e NCCL_DEBUG=INFO -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        -e TORCH_NCCL_BLOCKING_WAIT=1
        -e N0_FLEX_ATTENTION_BACKEND=grouped_flash_attn
        -e N0_GROUPED_SDPA_MAX_QUERY_TOKENS=16384
        -e "N0_MOT_CROSS_ATTENTION_BACKEND=$N0_MOT_CROSS_ATTENTION_BACKEND"
        -e N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST=1
        -e N0_FSDP_REDUCE_DTYPE=bfloat16
        -e "N0_FSDP_TOPOLOGY=$N0_FSDP_TOPOLOGY"
        -e "N0_FSDP_SHARD_SIZE=$N0_FSDP_SHARD_SIZE"
        -e "N0_FSDP_EXPERT_RESHARD_POLICY=$N0_FSDP_EXPERT_RESHARD_POLICY"
        -e N0_MOT_ACTIVATION_CHECKPOINTING=0
        -e N0_SYNC_ATTENTION_WINDOW=1
        -e "N0_TRACK31_SAMPLER_RANK_ALIGNMENT=$N0_TRACK31_SAMPLER_RANK_ALIGNMENT"
        -e TORCHINDUCTOR_COMPILE_THREADS=1
        -e "N0_TRACK31_ARTIFACT_ROOT=$CONTAINER_ARTIFACT_ROOT"
        -e "N0_TRACK31_LEROBOT_ROOT=$dataset_parent" -e "N0_BASE_MODEL=$CONTAINER_MODEL_ROOT"
        -e "N0_EMPTY_EMBEDDING=$CONTAINER_MODEL_ROOT/empty_emb.pt"
        -e "N0_EMPTY_EMBEDDING_SHA256=$N0_EMPTY_EMBEDDING_SHA256"
        -e "N0_RELEASED_CHECKPOINT=$CONTAINER_MODEL_ROOT"
        -e "N0_RELEASED_TRANSFORMER_SHA256=$N0_RELEASED_TRANSFORMER_SHA256"
        -e "N0_TRACK31_PYTHON_OVERLAY=$CONTAINER_OVERLAY_ROOT"
        -e "N0_TRACK31_SAVE_ROOT=$CONTAINER_RUN_ROOT"
        -e N0_TRACK31_TRAIN_PROFILE=multitask_pretrain_v1
        -e "N0_TRACK31_RUN_ROLE=$N0_TRACK31_RUN_ROLE"
        -e "N0_TRACK31_NUM_STEPS=$N0_TRACK31_NUM_STEPS"
        -e "N0_TRACK31_STOP_AFTER_STEP=$N0_TRACK31_STOP_AFTER_STEP"
        -e "N0_TRACK31_SAVE_INTERVAL=$N0_TRACK31_SAVE_INTERVAL"
        -e "N0_TRACK31_VAL_INTERVAL=$N0_TRACK31_VAL_INTERVAL"
        -e N0_TRACK31_BATCH_SIZE=1
        -e "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS=$FORMAL_GRADIENT_ACCUMULATION_STEPS"
        -e N0_TRACK31_MAX_LATENT_FRAMES=5 -e N0_TRACK31_LOAD_WORKER=0
        -e N0_TRAIN_SEED=20260801 -e N0_ACTION_INIT_SEED=0 -e PYTHONHASHSEED=20260801
        -e PYTHONDONTWRITEBYTECODE=1
        -e "N0_TRACK31_SOURCE_MANIFEST_SHA256=$N0_TRACK31_SOURCE_MANIFEST_SHA256"
        -e "N0_TRACK31_IMAGE_ID=$N0_TRACK31_IMAGE_ID"
        -e "N0_TRACK31_OVERLAY_MANIFEST_SHA256=$N0_TRACK31_OVERLAY_MANIFEST_SHA256"
        -e "N0_TRACK31_INVOCATION_ID=$N0_TRACK31_INVOCATION_ID"
        -e "N0_TRACK31_CONTAINER_ID=$container_id"
        -e "N0_TRACK31_NODE_ADDRESS=$host"
        -e "N0_TRACK31_SSH_PORT=36000"
        -e "N0_TRACK31_KNOWN_HOSTS_SHA256=$KNOWN_HOSTS_SHA256"
        -e "N0_TRACK31_ENVIRONMENT_MANIFEST=$ENVIRONMENT_MANIFEST_CONTAINER_PATH"
        -e "N0_TRACK31_ENVIRONMENT_MANIFEST_SHA256=$ENVIRONMENT_MANIFEST_SHA256"
        -e "N0_TRACK31_COLLECTIVE_SMOKE_REPORT=$N0_TRACK31_COLLECTIVE_SMOKE_REPORT"
        -e "N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256=$N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256"
        -e "N0_TRACK31_COLLECTIVE_SMOKE_MAX_AGE_SECONDS=$COLLECTIVE_SMOKE_MAX_AGE_SECONDS"
    )
    [[ -z "${N0_TRACK31_RESUME_FROM:-}" ]] || \
        DOCKER_ENV+=(-e "N0_TRACK31_RESUME_FROM=$N0_TRACK31_RESUME_FROM")
    if [[ "$mode" == "launch" ]]; then
        [[ -n "${LAUNCH_RECEIPT_CONTAINER_PATH:-}" && \
            -n "${LAUNCH_RECEIPT_SHA256:-}" ]] || return 1
        DOCKER_ENV+=(
            -e N0_TRACK31_ORCHESTRATED_PREFLIGHT=track31-stage-a-v2
            -e "N0_TRACK31_LAUNCH_RECEIPT=$LAUNCH_RECEIPT_CONTAINER_PATH"
            -e "N0_TRACK31_LAUNCH_RECEIPT_SHA256=$LAUNCH_RECEIPT_SHA256"
            -e "N0_TRACK31_LAUNCH_MANIFEST_SHA256=$LAUNCH_RECEIPT_SHA256"
        )
    elif [[ "$mode" == "preflight" ]]; then
        [[ "${LAUNCH_RECEIPT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || return 1
        DOCKER_ENV+=(
            -e "PYTHONPATH=$CONTAINER_OVERLAY_ROOT:$CONTAINER_SOURCE_ROOT:$CONTAINER_SOURCE_ROOT/n0_twam"
            -e "N0_TRACK31_LAUNCH_MANIFEST_SHA256=$LAUNCH_RECEIPT_SHA256"
        )
    else
        return 1
    fi
}

foreground_preflight() {
    local host=$1 rank=$2 log
    log="$CONTAINER_RUN_ROOT/logs/preflight_rank${rank}.${N0_TRACK31_INVOCATION_ID}.log"
    build_environment "$host" "$rank" preflight
    remote_exec "$host" docker exec "${DOCKER_ENV[@]}" "$N0_TRACK31_CONTAINER_NAME" \
        bash -lc "set -euo pipefail; set -o noclobber; : > '$log'; source /opt/hyhal/env.sh; cd '$CONTAINER_SOURCE_ROOT'; /usr/bin/python3 script/track3_1/preflight_train.py >> '$log' 2>&1"
}

wait_for_rank_registration() {
    local host=$1 rank=$2 attempt job invocation pgid leader receipt_sha
    for ((attempt = 0; attempt < REGISTRATION_ACK_POLLS; attempt++)); do
        job=$(active_owned_job "$host" "$N0_TRACK31_CONTAINER_NAME" \
            "$rank" "$N0_TRACK31_INVOCATION_ID") || {
            echo "${host}: rank ${rank} registration receipt validation failed" >&2
            return 1
        }
        if [[ "$job" != "NONE" ]]; then
            read -r invocation pgid leader receipt_sha <<<"$job"
            [[ "$invocation" == "$N0_TRACK31_INVOCATION_ID" \
                && "$pgid" =~ ^[1-9][0-9]*$ && "$leader" == "$pgid" \
                && "$receipt_sha" == "$LAUNCH_RECEIPT_SHA256" ]] || {
                echo "${host}: rank ${rank} registration receipt identity mismatch" >&2
                return 1
            }
            return 0
        fi
        sleep "$REGISTRATION_ACK_POLL_SECONDS"
    done
    echo "${host}: rank ${rank} registration receipt timed out" >&2
    return 1
}

launch_rank() {
    local host=$1 rank=$2 log job program container_id command
    log="$CONTAINER_RUN_ROOT/logs/node_rank${rank}.${N0_TRACK31_INVOCATION_ID}.log"
    job="$CONTAINER_RUN_ROOT/launch_manifests/job.${N0_TRACK31_INVOCATION_ID}.rank${rank}.json"
    build_environment "$host" "$rank" launch
    container_id=$(container_identity "$host" "$N0_TRACK31_CONTAINER_NAME") || return 1
    remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" test ! -e "$log" || return 1
    remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" test ! -e "$job" || return 1
    program='import json,os,sys
p,container_id,invocation,rank,manifest_sha=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5]
leader=os.getpgrp(); raw=open(f"/proc/{leader}/stat",encoding="utf-8").read(); rest=raw.rsplit(")",1)[1].split()
if int(rest[2]) != leader: raise SystemExit(1)
payload=json.dumps({"schema_version":1,"container_id":container_id,"invocation_id":invocation,"node_rank":rank,"leader_pid":leader,"process_group_id":leader,"leader_start_ticks":int(rest[19]),"launch_manifest_sha256":manifest_sha},sort_keys=True,separators=(",",":"))+"\n"
tmp=p+".tmp"; fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
try:
 os.write(fd,payload.encode()); os.fsync(fd)
finally: os.close(fd)
try: os.link(tmp,p)
finally: os.unlink(tmp)
fd=os.open(os.path.dirname(p),os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)'
    command="set -euo pipefail; set -o noclobber; : > $log; "
    command+="/usr/bin/python3 -c \"\$1\" \"\$2\" \"\$3\" \"\$4\" \"\$5\" \"\$6\"; "
    command+="source /opt/hyhal/env.sh; cd $CONTAINER_SOURCE_ROOT; "
    command+="exec bash run_track31_univtac.sh >> $log 2>&1"
    remote_exec "$host" docker exec -d "${DOCKER_ENV[@]}" "$N0_TRACK31_CONTAINER_NAME" \
        setsid bash -lc "$command" _ "$program" "$job" "$container_id" \
        "$N0_TRACK31_INVOCATION_ID" "$rank" "$LAUNCH_RECEIPT_SHA256" || {
        echo "${host}: rank ${rank} detached launch command failed" >&2
        return 1
    }
    wait_for_rank_registration "$host" "$rank"
}

signal_invocation_all() {
    local signal=$1 invocation=$2 manifest_sha=$3 index host job pgid
    local registration_group failed=0 job_failed registration_failed
    for index in "${!NODES[@]}"; do
        host=${NODES[$index]}
        verify_formal_container_owner "$host" "$N0_TRACK31_CONTAINER_NAME" || {
            failed=1; continue;
        }
        job_failed=0
        registration_failed=0
        pgid=
        job=$(active_owned_job "$host" "$N0_TRACK31_CONTAINER_NAME" \
            "$index" "$invocation") || { job_failed=1; failed=1; job=NONE; }
        registration_group=NONE
        if [[ "$job" != "NONE" ]]; then
            read -r _ pgid _ _ <<<"$job"
        else
            registration_group=$(active_owned_invocation_group "$host" \
                "$N0_TRACK31_CONTAINER_NAME" "$index" "$invocation" \
                "$manifest_sha" "$FORMAL_NODE_COUNT") || {
                registration_failed=1
                failed=1
                registration_group=NONE
            }
            if [[ "$registration_group" != "NONE" ]]; then
                pgid=$registration_group
            else
                ((job_failed == 0 && registration_failed == 0)) || continue
                continue
            fi
        fi
        if [[ -z "${pgid:-}" ]]; then
            failed=1
            continue
        fi
        signal_owned_process_group "$host" "$N0_TRACK31_CONTAINER_NAME" \
            "$signal" "$pgid" || failed=1
    done
    return "$failed"
}

invocation_is_stopped() {
    local invocation=$1 manifest_sha=$2 index host job counts registration_group
    local failed=0 running=0
    for index in "${!NODES[@]}"; do
        host=${NODES[$index]}
        job=$(active_owned_job "$host" "$N0_TRACK31_CONTAINER_NAME" \
            "$index" "$invocation") || { failed=1; continue; }
        registration_group=NONE
        if [[ "$job" == "NONE" ]]; then
            registration_group=$(active_owned_invocation_group "$host" \
                "$N0_TRACK31_CONTAINER_NAME" "$index" "$invocation" \
                "$manifest_sha" "$FORMAL_NODE_COUNT") || { failed=1; continue; }
        fi
        counts=$(training_process_counts "$host" "$N0_TRACK31_CONTAINER_NAME") || {
            failed=1
            continue
        }
        [[ "$job" == "NONE" && "$registration_group" == "NONE" \
            && "$counts" == "0 0" ]] || running=1
    done
    ((failed == 0)) || return 2
    ((running == 0))
}

terminate_invocation_bounded() {
    local invocation=$1 manifest_sha=$2 attempt observation_rc stable_zero=0 failed=0
    local term_polls=$((CLEANUP_REGISTRATION_GRACE_POLLS + STOP_POLLS + \
        CLEANUP_STABLE_ZERO_POLLS))
    local kill_polls=$((STOP_POLLS + CLEANUP_STABLE_ZERO_POLLS))

    echo "cleanup invocation=${invocation}: TERM with registration grace" >&2
    for ((attempt = 0; attempt < term_polls; attempt++)); do
        signal_invocation_all TERM "$invocation" "$manifest_sha" || failed=1
        if invocation_is_stopped "$invocation" "$manifest_sha"; then
            if ((attempt >= CLEANUP_REGISTRATION_GRACE_POLLS)); then
                ((stable_zero += 1))
            else
                stable_zero=0
            fi
        else
            observation_rc=$?
            ((observation_rc == 2)) && failed=1
            stable_zero=0
        fi
        ((stable_zero >= CLEANUP_STABLE_ZERO_POLLS)) && break
        sleep 1
    done

    if ((stable_zero < CLEANUP_STABLE_ZERO_POLLS)); then
        echo "cleanup invocation=${invocation}: escalating to KILL" >&2
        stable_zero=0
        for ((attempt = 0; attempt < kill_polls; attempt++)); do
            signal_invocation_all KILL "$invocation" "$manifest_sha" || failed=1
            if invocation_is_stopped "$invocation" "$manifest_sha"; then
                ((stable_zero += 1))
            else
                observation_rc=$?
                ((observation_rc == 2)) && failed=1
                stable_zero=0
            fi
            ((stable_zero >= CLEANUP_STABLE_ZERO_POLLS)) && break
            sleep 1
        done
    fi

    if ((stable_zero < CLEANUP_STABLE_ZERO_POLLS)); then
        echo "cleanup invocation=${invocation}: processes did not reach stable zero" >&2
        return 1
    fi
    invocation_is_stopped "$invocation" "$manifest_sha" || {
        echo "cleanup invocation=${invocation}: final process count is not zero" >&2
        return 1
    }
    ((failed == 0)) || {
        echo "cleanup invocation=${invocation}: completed with control errors" >&2
        return 1
    }
    echo "cleanup invocation=${invocation}: stable process count is 0 0 on all nodes" >&2
}

cleanup_failed_launch() {
    local reason=$1
    echo "${reason}; terminating invocation ${N0_TRACK31_INVOCATION_ID}" >&2
    terminate_invocation_bounded "$N0_TRACK31_INVOCATION_ID" \
        "$LAUNCH_RECEIPT_SHA256"
}

launch_all_ranks() {
    local index host host_index counts state job discovered failed=0 mounts
    local torch_count train_count parent_pid parent_pgid direct_count
    local job_pgid job_leader job_sha
    mounts=$(expected_mounts)
    for host_index in "${!NODES[@]}"; do
        host=${NODES[$host_index]}
        ensure_run_evidence_dirs "$host" || return 1
        verify_host_inputs "$host" || return 1
        verify_container_contract "$host" "$N0_TRACK31_CONTAINER_NAME" \
            "$N0_TRACK31_IMAGE_ID" "$mounts" || return 1
        verify_container_frozen_inputs "$host" || return 1
        discovered=$(discover_owned_invocation_group "$host" \
            "$N0_TRACK31_CONTAINER_NAME" "$host_index") || {
            echo "${host}: formal invocation discovery failed" >&2
            return 1
        }
        [[ "$discovered" == "NONE" ]] || {
            echo "${host}: formal invocation already active or registering: ${discovered}" >&2
            return 1
        }
        counts=$(training_process_counts "$host" "$N0_TRACK31_CONTAINER_NAME") || return 1
        [[ "$counts" == "0 0" ]] || { echo "${host}: training already exists" >&2; return 1; }
        lightweight_hcu_check "$host" || {
            echo "${host}: lightweight HCU check failed" >&2; return 1;
        }
    done
    resolve_resume_parent_identity || return 1
    verify_collective_smoke_report || {
        echo "multi-node collective smoke evidence failed validation" >&2
        return 1
    }
    create_environment_manifest || return 1
    prepare_launch_manifest || return 1
    foreground_preflight "${NODES[0]}" 0 || {
        echo "rank-zero foreground preflight failed" >&2
        return 1
    }
    create_launch_manifest || return 1
    for ((index = 1; index < FORMAL_NODE_COUNT; index++)); do
        launch_rank "${NODES[$index]}" "$index" || {
            cleanup_failed_launch "rank ${index} launch or registration failed" || \
                echo "failed-launch cleanup did not prove stable zero" >&2
            return 1
        }
    done
    launch_rank "${NODES[0]}" 0 || {
        cleanup_failed_launch "rank 0 launch or registration failed" || \
            echo "failed-launch cleanup did not prove stable zero" >&2
        return 1
    }
    for ((index = 0; index < HEALTH_ATTEMPTS; index++)); do
        failed=0
        for host_index in "${!NODES[@]}"; do
            host=${NODES[$host_index]}
            job=$(active_owned_job "$host" "$N0_TRACK31_CONTAINER_NAME" \
                "$host_index" "$N0_TRACK31_INVOCATION_ID") || { failed=1; continue; }
            state=$(training_process_state "$host" "$N0_TRACK31_CONTAINER_NAME") || {
                failed=1; continue;
            }
            read -r torch_count train_count parent_pid parent_pgid direct_count <<<"$state"
            if [[ "$job" == "NONE" ]]; then
                failed=1
            else
                read -r _ job_pgid job_leader job_sha <<<"$job"
                [[ "$torch_count $train_count $direct_count" == "1 8 8" \
                    && "$parent_pid" == "$job_leader" \
                    && "$parent_pgid" == "$job_pgid" \
                    && "$job_sha" == "$LAUNCH_RECEIPT_SHA256" ]] || failed=1
            fi
            remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" \
                test -s "$CONTAINER_RUN_ROOT/logs/node_rank${host_index}.${N0_TRACK31_INVOCATION_ID}.log" || failed=1
        done
        ((failed == 0)) && {
            echo "healthy_process_contract=${FORMAL_NODE_COUNT}x(1_torchrun+8_train)"
            return 0
        }
        sleep "$HEALTH_POLL_SECONDS"
    done
    cleanup_failed_launch "bounded launch health check failed" || \
        echo "failed-launch cleanup did not prove stable zero" >&2
    return 1
}

show_status() {
    local host state job discovered failed=0 index
    local discovered_invocation discovered_pgid discovered_sha job_sha
    local torch_count train_count parent_pid parent_pgid direct_count
    for index in "${!NODES[@]}"; do
        host=${NODES[$index]}
        echo "== ${host} rank=${index} =="
        remote_exec "$host" docker ps -a --filter "name=^/$N0_TRACK31_CONTAINER_NAME$" \
            --format '{{.Names}}\t{{.Status}}' || failed=1
        verify_formal_container_owner "$host" "$N0_TRACK31_CONTAINER_NAME" || {
            echo "unowned_container=true"; failed=1; continue;
        }
        state=$(training_process_state "$host" "$N0_TRACK31_CONTAINER_NAME") || {
            failed=1; continue;
        }
        job=$(active_owned_job "$host" "$N0_TRACK31_CONTAINER_NAME" "$index") || {
            failed=1; continue;
        }
        discovered=NONE
        if [[ "$job" == "NONE" ]]; then
            discovered=$(discover_owned_invocation_group "$host" \
                "$N0_TRACK31_CONTAINER_NAME" "$index") || { failed=1; continue; }
        fi
        read -r torch_count train_count parent_pid parent_pgid direct_count <<<"$state"
        if [[ "$job" == "NONE" ]]; then
            if [[ "$discovered" != "NONE" ]]; then
                read -r discovered_invocation discovered_pgid discovered_sha \
                    <<<"$discovered"
                echo "registration_pending invocation=$discovered_invocation pgid=$discovered_pgid manifest_sha=$discovered_sha"
                failed=1
                continue
            fi
            [[ "$torch_count $train_count $direct_count" == "0 0 0" ]] || failed=1
            echo "invocation=none torchrun=$torch_count train=$train_count direct=$direct_count"
        else
            read -r invocation pgid leader job_sha <<<"$job"
            [[ "$parent_pid" == "0" || \
                ( "$parent_pid" == "$leader" && "$parent_pgid" == "$pgid" ) ]] || \
                failed=1
            [[ "$torch_count $train_count $direct_count" == "1 8 8" ]] || failed=1
            echo "invocation=$invocation pgid=$pgid torchrun=$torch_count train=$train_count direct=$direct_count"
        fi
    done
    return "$failed"
}

stop_training() {
    local host job discovered invocation='' manifest_sha='' index current current_sha
    local current_pgid job_pgid job_sha discovered_invocation discovered_sha
    for index in "${!NODES[@]}"; do
        host=${NODES[$index]}
        verify_formal_container_owner "$host" "$N0_TRACK31_CONTAINER_NAME" || return 1
        job=$(active_owned_job "$host" "$N0_TRACK31_CONTAINER_NAME" "$index") || return 1
        discovered=NONE
        if [[ "$job" == "NONE" ]]; then
            discovered=$(discover_owned_invocation_group "$host" \
                "$N0_TRACK31_CONTAINER_NAME" "$index") || return 1
        fi
        if [[ "$job" == "NONE" && "$discovered" == "NONE" ]]; then
            [[ "$(training_process_counts "$host" "$N0_TRACK31_CONTAINER_NAME")" == \
                "0 0" ]] || return 1
            continue
        fi
        if [[ "$job" != "NONE" ]]; then
            read -r current job_pgid _ job_sha <<<"$job"
            current_sha=$job_sha
        else
            read -r current current_pgid current_sha <<<"$discovered"
        fi
        [[ -z "$invocation" || "$invocation" == "$current" ]] || \
            die "refusing to stop mixed invocations"
        [[ -z "$manifest_sha" || "$manifest_sha" == "$current_sha" ]] || \
            die "refusing to stop mixed launch manifests"
        invocation=$current
        manifest_sha=$current_sha
    done
    [[ -n "$invocation" ]] || return 0
    terminate_invocation_bounded "$invocation" "$manifest_sha"
}

main() {
    local action=${1:-status}
    (($# <= 1)) || die "usage: $0 {containers|collective-smoke|launch|status|stop}"
    require_operational_configuration
    case "$action" in
        status) show_status ;;
        stop) stop_training ;;
        containers) require_formal_configuration; create_all_containers ;;
        collective-smoke)
            require_formal_configuration
            require_collective_smoke_recipe
            run_collective_smoke
            ;;
        launch) require_formal_configuration; require_formal_recipe; launch_all_ranks ;;
        *) die "usage: $0 {containers|collective-smoke|launch|status|stop}" ;;
    esac
}

main "$@"
