#!/usr/bin/env bash
# Secret-free, immutable environment evidence for formal Stage A launches.

ensure_run_evidence_dirs() {
    local host=$1 name path
    remote_exec "$host" mkdir -p "$N0_TRACK31_RUN_ROOT_HOST/logs" \
        "$N0_TRACK31_RUN_ROOT_HOST/launch_manifests" \
        "$N0_TRACK31_RUN_ROOT_HOST/environment" \
        "$N0_TRACK31_RUN_ROOT_HOST/preflight" || return 1
    for name in logs launch_manifests environment preflight; do
        path="$N0_TRACK31_RUN_ROOT_HOST/$name"
        [[ "$(remote_exec "$host" realpath -e "$path")" == "$path" ]] || return 1
    done
}

runtime_probe() {
    local host=$1 program
    program='import json,platform,sys,torch
devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"python":sys.version.split()[0],"platform":platform.platform(),"torch":torch.__version__,"hip":torch.version.hip,"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),"device_names":devices,"python_overlay":"/formal/overlay"},sort_keys=True,separators=(",",":")))'
    remote_exec "$host" docker exec -e "HIP_VISIBLE_DEVICES=$HCU_ORDER" \
        -e "PYTHONPATH=$CONTAINER_OVERLAY_ROOT" "$N0_TRACK31_CONTAINER_NAME" \
        bash -lc \
        'set -euo pipefail; source /opt/hyhal/env.sh; exec /usr/bin/python3 -B -c "$1"' \
        _ "$program"
}

driver_probe() {
    remote_exec "$1" docker exec "$N0_TRACK31_CONTAINER_NAME" bash -lc \
        'if [[ -x /opt/hyhal/bin/hy-smi ]]; then /opt/hyhal/bin/hy-smi --version 2>&1 | head -n 1; else printf unavailable; fi'
}

create_node_environment_record() {
    local host=$1 rank=$2 container_id image_id hostname runtime driver payload path sha
    container_id=$(container_identity "$host" "$N0_TRACK31_CONTAINER_NAME") || return 1
    image_id=$(remote_exec "$host" docker container inspect --format '{{.Image}}' \
        "$N0_TRACK31_CONTAINER_NAME") || return 1
    hostname=$(remote_exec "$host" hostname) || return 1
    runtime=$(runtime_probe "$host") || return 1
    driver=$(driver_probe "$host") || return 1
    payload=$(python3 -c 'import json,sys
runtime=json.loads(sys.argv[6])
if runtime.get("device_count") != 8 or not runtime.get("cuda_available"): raise SystemExit(1)
print(json.dumps({"schema_version":1,"node_address":sys.argv[1],"node_rank":int(sys.argv[2]),"container_id":sys.argv[3],"image_id":sys.argv[4],"hostname":sys.argv[5],"runtime":runtime,"hyhal_driver_summary":sys.argv[7]},sort_keys=True,separators=(",",":"))+"\n")' \
        "$host" "$rank" "$container_id" "$image_id" "$hostname" "$runtime" "$driver") || \
        return 1
    path="$N0_TRACK31_RUN_ROOT_HOST/environment/node_rank${rank}.${N0_TRACK31_INVOCATION_ID}.json"
    publish_atomic_file "${NODES[0]}" "$path" "$payload" || return 1
    sha=$(remote_exec "${NODES[0]}" sha256sum "$path") || return 1
    ENVIRONMENT_NODE_RECORDS+=("$rank|$CONTAINER_RUN_ROOT/environment/$(basename -- "$path")|${sha%% *}")
}

create_pip_environment_record() {
    local freeze payload path sha
    freeze=$(remote_exec "${NODES[0]}" docker exec \
        -e "PYTHONPATH=$CONTAINER_OVERLAY_ROOT" "$N0_TRACK31_CONTAINER_NAME" \
        /usr/bin/python3 -m pip freeze --all) || return 1
    payload=$(python3 -c 'import json,re,sys
lines=sorted(line for line in sys.argv[2].splitlines() if line)
unsafe=re.compile(r"://[^/\s]*@|(?:token|password|secret|api[_-]?key)=(?!=)",re.IGNORECASE)
if any(unsafe.search(line) for line in lines): raise SystemExit(1)
print(json.dumps({"schema_version":1,"python_overlay":sys.argv[1],"pip_freeze":lines},sort_keys=True,separators=(",",":"))+"\n")' \
        "$CONTAINER_OVERLAY_ROOT" "$freeze") || return 1
    path="$N0_TRACK31_RUN_ROOT_HOST/environment/pip_freeze.${N0_TRACK31_INVOCATION_ID}.json"
    publish_atomic_file "${NODES[0]}" "$path" "$payload" || return 1
    sha=$(remote_exec "${NODES[0]}" sha256sum "$path") || return 1
    PIP_ENVIRONMENT_CONTAINER_PATH="$CONTAINER_RUN_ROOT/environment/$(basename -- "$path")"
    PIP_ENVIRONMENT_SHA256=${sha%% *}
}

create_environment_manifest() {
    local rank record records_json payload path sha
    ENVIRONMENT_NODE_RECORDS=()
    for rank in "${!NODES[@]}"; do
        create_node_environment_record "${NODES[$rank]}" "$rank" || return 1
    done
    create_pip_environment_record || return 1
    records_json=$(printf '%s\n' "${ENVIRONMENT_NODE_RECORDS[@]}" | \
        python3 -c 'import json,sys
records=[]
for line in sys.stdin:
 rank,path,sha=line.rstrip("\n").split("|",2)
 records.append({"node_rank":int(rank),"path":path,"sha256":sha})
print(json.dumps(records,sort_keys=True,separators=(",",":")))') || return 1
    payload=$(python3 -c 'import json,sys
print(json.dumps({"schema_version":1,"invocation_id":sys.argv[1],"ssh_port":36000,"known_hosts_sha256":sys.argv[2],"node_records":json.loads(sys.argv[3]),"pip_freeze":{"path":sys.argv[4],"sha256":sys.argv[5]}},sort_keys=True,separators=(",",":"))+"\n")' \
        "$N0_TRACK31_INVOCATION_ID" "$KNOWN_HOSTS_SHA256" "$records_json" \
        "$PIP_ENVIRONMENT_CONTAINER_PATH" "$PIP_ENVIRONMENT_SHA256") || return 1
    path="$N0_TRACK31_RUN_ROOT_HOST/environment/environment_manifest.${N0_TRACK31_INVOCATION_ID}.json"
    publish_atomic_file "${NODES[0]}" "$path" "$payload" || return 1
    sha=$(remote_exec "${NODES[0]}" sha256sum "$path") || return 1
    ENVIRONMENT_MANIFEST_CONTAINER_PATH="$CONTAINER_RUN_ROOT/environment/$(basename -- "$path")"
    ENVIRONMENT_MANIFEST_SHA256=${sha%% *}
    echo "environment_manifest=$path"
    echo "environment_manifest_sha256=$ENVIRONMENT_MANIFEST_SHA256"
}
