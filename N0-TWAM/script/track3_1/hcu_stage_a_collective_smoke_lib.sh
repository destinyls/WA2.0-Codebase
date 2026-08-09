#!/usr/bin/env bash
# Invocation-scoped multi-node NCCL smoke for formal Track 3.1 Stage A.

readonly COLLECTIVE_SMOKE_PRODUCER_REL="script/track3_1/smoke_track31_multinode_collectives.py"
readonly COLLECTIVE_SMOKE_VALIDATOR_REL="script/track3_1/validate_track31_collective_smoke.py"
readonly DEFAULT_COLLECTIVE_SMOKE_TIMEOUT_SECONDS=180
readonly DEFAULT_COLLECTIVE_SMOKE_MAX_AGE_SECONDS=900

require_collective_smoke_recipe() {
    require_formal_recipe_identity
    N0_TRACK31_SMOKE_MASTER_PORT=${N0_TRACK31_SMOKE_MASTER_PORT:-29661}
    require_positive_integer N0_TRACK31_SMOKE_MASTER_PORT
    ((N0_TRACK31_SMOKE_MASTER_PORT <= 65535)) || \
        die "N0_TRACK31_SMOKE_MASTER_PORT exceeds 65535"
    [[ "$N0_TRACK31_SMOKE_MASTER_PORT" != "$N0_TRACK31_MASTER_PORT" ]] || \
        die "collective smoke and training must use distinct master ports"
    COLLECTIVE_SMOKE_TIMEOUT_SECONDS=${N0_TRACK31_SMOKE_TIMEOUT_SECONDS:-$DEFAULT_COLLECTIVE_SMOKE_TIMEOUT_SECONDS}
    COLLECTIVE_SMOKE_MAX_AGE_SECONDS=${N0_TRACK31_COLLECTIVE_SMOKE_MAX_AGE_SECONDS:-$DEFAULT_COLLECTIVE_SMOKE_MAX_AGE_SECONDS}
    [[ "$COLLECTIVE_SMOKE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
        die "invalid collective smoke timeout"
    [[ "$COLLECTIVE_SMOKE_MAX_AGE_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
        die "invalid collective smoke max age"
    ((COLLECTIVE_SMOKE_TIMEOUT_SECONDS <= 600)) || \
        die "collective smoke timeout exceeds the formal bound"
    ((COLLECTIVE_SMOKE_MAX_AGE_SECONDS <= 3600)) || \
        die "collective smoke max age exceeds the formal bound"
    COLLECTIVE_SMOKE_CONTAINER_PATH="$CONTAINER_RUN_ROOT/preflight/collective_smoke.${N0_TRACK31_INVOCATION_ID}.json"
    COLLECTIVE_SMOKE_HOST_PATH="$N0_TRACK31_RUN_ROOT_HOST/preflight/collective_smoke.${N0_TRACK31_INVOCATION_ID}.json"
}

collective_smoke_process_count() {
    local host=$1
    remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" \
        ps -eo args | awk \
        -v producer="$COLLECTIVE_SMOKE_PRODUCER_REL" \
        'index($0, producer) { count += 1 } END { print count + 0 }'
}

build_collective_smoke_environment() {
    local host=$1 rank=$2 container_id=$3 log=$4
    COLLECTIVE_SMOKE_DOCKER_ENV=(
        -e "NGPU=$HCU_PER_NODE" -e "NNODES=$FORMAL_NODE_COUNT" -e "NODE_RANK=$rank"
        -e "MASTER_ADDR=${NODES[0]}" -e "MASTER_PORT=$N0_TRACK31_SMOKE_MASTER_PORT"
        -e "HIP_VISIBLE_DEVICES=$HCU_ORDER" -e HSA_FORCE_FINE_GRAIN_PCIE=1
        -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_QPS_PER_CONNECTION=4
        -e NCCL_IB_TC=160 -e NCCL_IB_TIMEOUT=22 -e NCCL_NET_GDR_LEVEL=2
        -e NCCL_ROCE_SRC_PORT_LIST=60000,60051,57663,57804 -e NCCL_SOCKET_IFNAME=bond1
        -e NCCL_DEBUG=INFO -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        -e TORCH_NCCL_BLOCKING_WAIT=1
        -e "N0_TRACK31_NODE_ADDRESS=$host" -e "N0_TRACK31_CONTAINER_ID=$container_id"
        -e "N0_TRACK31_NODES=$N0_TRACK31_NODES"
        -e "N0_TRACK31_HCU_ORDER=$HCU_ORDER" -e "N0_TRACK31_RUN_ROLE=$N0_TRACK31_RUN_ROLE"
        -e "N0_TRACK31_INVOCATION_ID=$N0_TRACK31_INVOCATION_ID"
        -e N0_TRACK31_BATCH_SIZE=1
        -e "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS=$FORMAL_GRADIENT_ACCUMULATION_STEPS"
        -e "N0_TRACK31_SOURCE_MANIFEST_SHA256=$N0_TRACK31_SOURCE_MANIFEST_SHA256"
        -e "N0_TRACK31_OVERLAY_MANIFEST_SHA256=$N0_TRACK31_OVERLAY_MANIFEST_SHA256"
        -e "N0_TRACK31_IMAGE_ID=$N0_TRACK31_IMAGE_ID"
        -e "N0_TRACK31_SMOKE_ID=$N0_TRACK31_INVOCATION_ID"
        -e "N0_TRACK31_SMOKE_REPORT=$COLLECTIVE_SMOKE_CONTAINER_PATH"
        -e "N0_TRACK31_SMOKE_LOG=$log"
        -e "PYTHONPATH=$CONTAINER_OVERLAY_ROOT:$CONTAINER_SOURCE_ROOT:$CONTAINER_SOURCE_ROOT/n0_twam"
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONHASHSEED=20260801
    )
}

run_collective_smoke_rank() {
    local host=$1 rank=$2 container_id=$3 log
    log="$CONTAINER_RUN_ROOT/logs/collective_smoke.rank${rank}.${N0_TRACK31_INVOCATION_ID}.log"
    build_collective_smoke_environment "$host" "$rank" "$container_id" "$log"
    # This function is always backgrounded by run_collective_smoke. Replace its
    # subshell with ssh so the controller PID is the actual client process and
    # bounded cleanup cannot strand a local child ssh.
    remote_exec_replace_shell "$host" docker exec "${COLLECTIVE_SMOKE_DOCKER_ENV[@]}" \
        "$N0_TRACK31_CONTAINER_NAME" /usr/bin/timeout --signal=TERM \
        --kill-after=10 "$COLLECTIVE_SMOKE_TIMEOUT_SECONDS" bash -lc \
        'set -euo pipefail
set -o noclobber
: > "$N0_TRACK31_SMOKE_LOG"
source /opt/hyhal/env.sh
cd /workspace/N0-TWAM
exec /usr/local/bin/torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$NGPU" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  --tee 3 \
  script/track3_1/smoke_track31_multinode_collectives.py \
  --report-path "$N0_TRACK31_SMOKE_REPORT" \
  --smoke-id "$N0_TRACK31_SMOKE_ID" \
  --expected-nodes "$N0_TRACK31_NODES" \
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
  --timeout-seconds 90 >> "$N0_TRACK31_SMOKE_LOG" 2>&1'
}

collect_current_container_identities() {
    local index identity hostname
    COLLECTIVE_SMOKE_CONTAINER_IDS=()
    COLLECTIVE_SMOKE_CONTAINER_HOSTNAMES=()
    for index in "${!NODES[@]}"; do
        identity=$(container_identity "${NODES[$index]}" \
            "$N0_TRACK31_CONTAINER_NAME") || return 1
        hostname=$(remote_exec "${NODES[$index]}" docker exec \
            "$N0_TRACK31_CONTAINER_NAME" hostname) || return 1
        [[ "$identity" =~ ^[0-9a-f]{64}$ ]] || return 1
        [[ -n "$hostname" && "$hostname" != *[[:space:]]* ]] || return 1
        COLLECTIVE_SMOKE_CONTAINER_IDS+=("$identity")
        COLLECTIVE_SMOKE_CONTAINER_HOSTNAMES+=("$hostname")
    done
}

validate_formal_collective_smoke_report() {
    local host_path=$1 expected_sha=$2 index host actual
    local -a validator_args
    [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || return 1
    collect_current_container_identities || return 1
    for host in "${NODES[@]}"; do
        remote_exec "$host" test -f "$host_path" || return 1
        remote_exec "$host" test ! -L "$host_path" || return 1
        [[ "$(remote_exec "$host" realpath -e "$host_path")" == "$host_path" ]] || \
            return 1
        actual=$(remote_exec "$host" sha256sum "$host_path") || return 1
        [[ "${actual%% *}" == "$expected_sha" ]] || return 1
    done
    validator_args=(
        --report-path "$host_path"
        --expected-report-sha256 "$expected_sha"
        --smoke-id "$N0_TRACK31_INVOCATION_ID"
        --expected-nodes "$N0_TRACK31_NODES"
        --expected-hcu-order "$HCU_ORDER"
        --expected-nccl-env NCCL_IB_DISABLE=0
        --expected-nccl-env NCCL_IB_GID_INDEX=3
        --expected-nccl-env NCCL_IB_QPS_PER_CONNECTION=4
        --expected-nccl-env NCCL_IB_TC=160
        --expected-nccl-env NCCL_IB_TIMEOUT=22
        --expected-nccl-env NCCL_NET_GDR_LEVEL=2
        --expected-nccl-env NCCL_ROCE_SRC_PORT_LIST=60000,60051,57663,57804
        --expected-nccl-env NCCL_SOCKET_IFNAME=bond1
        --expected-nccl-env NCCL_DEBUG=INFO
        --expected-nccl-env TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        --expected-nccl-env TORCH_NCCL_BLOCKING_WAIT=1
        --expected-image-id "$N0_TRACK31_IMAGE_ID"
        --expected-source-manifest-sha256 "$N0_TRACK31_SOURCE_MANIFEST_SHA256"
        --expected-overlay-manifest-sha256 "$N0_TRACK31_OVERLAY_MANIFEST_SHA256"
        --expected-run-role "$N0_TRACK31_RUN_ROLE"
        --expected-batch-size 1
        --expected-gradient-accumulation-steps "$FORMAL_GRADIENT_ACCUMULATION_STEPS"
        --expected-world-size "$((FORMAL_NODE_COUNT * HCU_PER_NODE))"
        --expected-node-count "$FORMAL_NODE_COUNT"
        --expected-hcu-per-node "$HCU_PER_NODE"
        --max-age-seconds "$COLLECTIVE_SMOKE_MAX_AGE_SECONDS"
    )
    for index in "${!NODES[@]}"; do
        validator_args+=(--expected-container-id \
            "${NODES[$index]}=${COLLECTIVE_SMOKE_CONTAINER_IDS[$index]}")
        validator_args+=(--expected-hostname \
            "${NODES[$index]}=${COLLECTIVE_SMOKE_CONTAINER_HOSTNAMES[$index]}")
    done
    remote_exec "${NODES[0]}" /usr/bin/python3 -B \
        "$N0_TRACK31_SOURCE_ROOT_HOST/$COLLECTIVE_SMOKE_VALIDATOR_REL" \
        "${validator_args[@]}"
}

wait_collective_smoke_jobs() {
    local timeout_seconds=$1 index pid remaining failed=0 timed_out=0
    local deadline=$((SECONDS + timeout_seconds + 30))
    shift
    local -a pids=("$@") completed=()
    while :; do
        remaining=0
        for index in "${!pids[@]}"; do
            [[ "${completed[$index]:-0}" == "0" ]] || continue
            pid=${pids[$index]}
            if kill -0 "$pid" 2>/dev/null; then
                remaining=1
                continue
            fi
            wait "$pid" || failed=1
            completed[$index]=1
        done
        ((remaining != 0)) || break
        if ((SECONDS >= deadline)); then
            timed_out=1
            break
        fi
        sleep 1
    done
    if ((timed_out != 0)); then
        for index in "${!pids[@]}"; do
            [[ "${completed[$index]:-0}" == "0" ]] || continue
            kill -TERM "${pids[$index]}" 2>/dev/null || true
        done
        sleep 1
        for index in "${!pids[@]}"; do
            [[ "${completed[$index]:-0}" == "0" ]] || continue
            kill -KILL "${pids[$index]}" 2>/dev/null || true
            wait "${pids[$index]}" 2>/dev/null || true
        done
        echo "collective smoke controller deadline exceeded" >&2
        return 1
    fi
    return "$failed"
}

run_collective_smoke() {
    local host index failed=0 process_count report_sha
    local -a pids=()
    collect_current_container_identities || return 1
    for index in "${!NODES[@]}"; do
        host=${NODES[$index]}
        ensure_run_evidence_dirs "$host" || return 1
        verify_host_inputs "$host" || return 1
        verify_container_contract "$host" "$N0_TRACK31_CONTAINER_NAME" \
            "$N0_TRACK31_IMAGE_ID" "$(expected_mounts)" || return 1
        verify_container_frozen_inputs "$host" || return 1
        [[ "$(training_process_counts "$host" "$N0_TRACK31_CONTAINER_NAME")" == \
            "0 0" ]] || { echo "${host}: training already exists" >&2; return 1; }
        process_count=$(collective_smoke_process_count "$host") || return 1
        [[ "$process_count" == "0" ]] || {
            echo "${host}: collective smoke already exists" >&2
            return 1
        }
        lightweight_hcu_check "$host" || return 1
        remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" \
            test ! -e "$COLLECTIVE_SMOKE_CONTAINER_PATH" || return 1
        remote_exec "$host" docker exec "$N0_TRACK31_CONTAINER_NAME" \
            test ! -e "$CONTAINER_RUN_ROOT/logs/collective_smoke.rank${index}.${N0_TRACK31_INVOCATION_ID}.log" || return 1
    done
    for index in "${!NODES[@]}"; do
        run_collective_smoke_rank "${NODES[$index]}" "$index" \
            "${COLLECTIVE_SMOKE_CONTAINER_IDS[$index]}" &
        pids+=("$!")
    done
    wait_collective_smoke_jobs "$COLLECTIVE_SMOKE_TIMEOUT_SECONDS" \
        "${pids[@]}" || failed=1
    for host in "${NODES[@]}"; do
        process_count=$(collective_smoke_process_count "$host") || failed=1
        [[ "$process_count" == "0" ]] || failed=1
    done
    ((failed == 0)) || {
        echo "bounded collective smoke failed; inspect invocation-scoped logs" >&2
        return 1
    }
    report_sha=$(remote_exec "${NODES[0]}" sha256sum \
        "$COLLECTIVE_SMOKE_HOST_PATH") || return 1
    report_sha=${report_sha%% *}
    validate_formal_collective_smoke_report "$COLLECTIVE_SMOKE_HOST_PATH" \
        "$report_sha" || return 1
    printf 'N0_TRACK31_COLLECTIVE_SMOKE_REPORT=%s\n' \
        "$COLLECTIVE_SMOKE_CONTAINER_PATH"
    printf 'N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256=%s\n' "$report_sha"
}
