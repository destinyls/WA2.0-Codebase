#!/usr/bin/env bash
# Shared fail-closed primitives for formal Track 3.1 HCU launchers.

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_env() {
    local name=$1
    [[ -n "${!name:-}" ]] || die "required environment variable is unset: ${name}"
}

require_positive_integer() {
    local name=$1 value=${!1:-}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
}

require_fixed_value() {
    local name=$1 expected=$2 value=${!1:-$2}
    [[ "$value" == "$expected" ]] || \
        die "${name} must equal the formal value ${expected}, got ${value}"
}

require_safe_absolute_path() {
    local name=$1 value=${!1:-}
    require_env "$name"
    [[ "$value" == /* && "$value" != "/" ]] || \
        die "${name} must be a non-root absolute path"
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *$'\t'* ]] || \
        die "${name} contains a control character"
}

validate_container_name() {
    [[ "$N0_TRACK31_CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
        die "invalid N0_TRACK31_CONTAINER_NAME"
}

validate_image_reference() {
    local component='[a-z0-9]+([._-][a-z0-9]+)*'
    local tag='[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}'
    [[ "$N0_TRACK31_IMAGE" =~ ^${component}(:[0-9]+)?(/${component})+(:${tag})?$ ]] || \
        die "invalid N0_TRACK31_IMAGE"
    [[ "$N0_TRACK31_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || \
        die "N0_TRACK31_IMAGE_ID must be a full lowercase docker image ID"
}

validate_invocation_id() {
    [[ "$N0_TRACK31_INVOCATION_ID" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || \
        die "N0_TRACK31_INVOCATION_ID is unsafe"
}

render_command() {
    local rendered
    printf -v rendered '%q ' "$@"
    printf '%s' "$rendered"
}

remote_exec() {
    local host=$1 command port=${N0_TRACK31_SSH_PORT:-36000}
    shift
    [[ "$port" == "36000" ]] || die "N0_TRACK31_SSH_PORT must equal 36000"
    command=$(render_command "$@")
    ssh -p "$port" -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking=yes \
        -o "UserKnownHostsFile=${N0_TRACK31_KNOWN_HOSTS}" \
        "$host" "$command"
}

remote_exec_replace_shell() {
    local host=$1 command port=${N0_TRACK31_SSH_PORT:-36000}
    shift
    [[ "$port" == "36000" ]] || die "N0_TRACK31_SSH_PORT must equal 36000"
    command=$(render_command "$@")
    exec ssh -p "$port" -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking=yes \
        -o "UserKnownHostsFile=${N0_TRACK31_KNOWN_HOSTS}" \
        "$host" "$command"
}

load_formal_nodes() {
    IFS=',' read -r -a NODES <<<"$N0_TRACK31_NODES"
    FORMAL_NODE_COUNT=${#NODES[@]}
    case "$FORMAL_NODE_COUNT" in
        2|4|6) ;;
        *) die "formal Stage A supports exactly 2, 4, or 6 nodes" ;;
    esac
    local host seen='|'
    for host in "${NODES[@]}"; do
        [[ "$host" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]] || \
            die "invalid node address: ${host}"
        [[ "$host" != *".."* && "$seen" != *"|${host}|"* ]] || \
            die "duplicate or invalid node: ${host}"
        seen+="${host}|"
    done
}

paths_overlap() {
    local first=${1%/} second=${2%/}
    [[ "$first" == "$second" || "$first" == "$second/"* || "$second" == "$first/"* ]]
}

canonical_roster_audit_script() {
    cat <<'BASH'
set -euo pipefail
cd "$1"
manifest=${2:-.source_manifest.sha256}
test -f "$manifest"
test ! -L "$manifest"
declare -A seen=()
count=0
manifest_paths=$(mktemp)
actual_paths=$(mktemp)
trap "rm -f \"$manifest_paths\" \"$actual_paths\"" EXIT
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[0-9a-f]{64}[[:space:]][[:space:]](.+)$ ]]
    rel=${BASH_REMATCH[1]}
    [[ "$rel" =~ ^[A-Za-z0-9._/-]+$ && "$rel" != /* && "$rel" != ./* \
        && "$rel" != *//* ]]
    IFS=/ read -r -a parts <<<"$rel"
    for part in "${parts[@]}"; do
        [[ -n "$part" && "$part" != "." && "$part" != ".." ]]
    done
    [[ -z "${seen[$rel]:-}" && -f "$rel" && ! -L "$rel" ]]
    seen[$rel]=1
    printf "%s\n" "$rel" >>"$manifest_paths"
    ((count += 1))
done <"$manifest"
((count > 0))
find . -type f ! -path "./$manifest" -print | sed "s#^\\./##" | \
    LC_ALL=C sort >"$actual_paths"
LC_ALL=C sort -o "$manifest_paths" "$manifest_paths"
cmp -s "$manifest_paths" "$actual_paths"
[[ -z "$(find . ! -type f ! -type d -print -quit)" ]]
sha256sum --check --strict "$manifest" >/dev/null
BASH
}

source_roster_audit_script() {
    canonical_roster_audit_script
}

verify_roster_host() {
    local host=$1 root=$2 manifest=$3 script
    script=$(canonical_roster_audit_script)
    remote_exec "$host" bash -lc "$script" _ "$root" "$manifest"
}

verify_roster_container() {
    local host=$1 container=$2 root=$3 manifest=$4 script
    script=$(canonical_roster_audit_script)
    remote_exec "$host" docker exec "$container" bash -lc "$script" \
        _ "$root" "$manifest"
}

verify_exact_image() {
    local host=$1 image_ref=$2 image_id=$3 actual
    actual=$(remote_exec "$host" docker image inspect --format '{{.Id}}' "$image_ref")
    [[ "$actual" == "$image_id" ]] || {
        echo "${host}: image identity mismatch: ${actual}" >&2
        return 1
    }
}

container_creation_token() {
    local host=$1 container=$2
    remote_exec "$host" docker container inspect --format \
        '{{index .Config.Labels "n0.track31.creation_token"}}' "$container"
}

container_identity() {
    remote_exec "$1" docker container inspect --format '{{.Id}}' "$2"
}

verify_formal_container_owner() {
    local value
    value=$(remote_exec "$1" docker container inspect --format \
        '{{index .Config.Labels "n0.track31.owner"}}' "$2") || return 1
    [[ "$value" == "formal-stage-a-v1" ]]
}

json_array_equals() {
    python3 -c 'import json,sys
actual=json.loads(sys.argv[1])
expected=json.loads(sys.argv[2])
raise SystemExit(0 if sorted(actual) == sorted(expected) else 1)' "$1" "$2"
}

verify_hcu_devices() {
    python3 -c 'import json,sys
actual=json.loads(sys.argv[1])
normalized=sorted((x.get("PathOnHost"),x.get("PathInContainer"),x.get("CgroupPermissions")) for x in actual)
expected=sorted((("/dev/kfd","/dev/kfd","rwm"),("/dev/dri","/dev/dri","rwm")))
raise SystemExit(0 if normalized == expected else 1)' "$1"
}

verify_container_contract() {
    local host=$1 container=$2 image_id=$3 expected_mounts=$4 actual value
    verify_formal_container_owner "$host" "$container" || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{.Image}}' "$container")
    [[ "$value" == "$image_id" ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{.State.Running}}' "$container")
    [[ "$value" == "true" ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{.HostConfig.Privileged}}' "$container")
    [[ "$value" == "false" ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{json .Config.Entrypoint}}' "$container")
    [[ "$value" == '["bash"]' ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{json .Config.Cmd}}' "$container")
    [[ "$value" == '["-lc","while :; do sleep 3600; done"]' ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{.Config.WorkingDir}}' "$container")
    [[ "$value" == "$CONTAINER_SOURCE_ROOT" ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{.HostConfig.NetworkMode}}' "$container")
    [[ "$value" == "host" ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{.HostConfig.IpcMode}}' "$container")
    [[ "$value" == "host" ]] || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{json .HostConfig.GroupAdd}}' "$container")
    json_array_equals "$value" '["video"]' || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{json .HostConfig.CapAdd}}' "$container")
    json_array_equals "$value" '["SYS_PTRACE"]' || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{json .HostConfig.SecurityOpt}}' "$container")
    json_array_equals "$value" '["seccomp=unconfined","label=disable"]' || return 1
    value=$(remote_exec "$host" docker container inspect --format '{{json .HostConfig.Devices}}' "$container")
    verify_hcu_devices "$value" || return 1
    actual=$(remote_exec "$host" docker container inspect --format \
        '{{range .Mounts}}{{printf "%s\t%s\t%s\t%t\n" .Type .Source .Destination .RW}}{{end}}' "$container")
    [[ "$(LC_ALL=C sort <<<"$actual")" == "$(LC_ALL=C sort <<<"$expected_mounts")" ]] || {
        echo "${host}: container mount contract mismatch" >&2
        return 1
    }
}

training_process_state() {
    local host=$1 container=$2 processes
    # Job receipts are written from inside the container PID namespace. Keep the
    # process inventory in that same namespace so leader PID/PGID comparisons are
    # meaningful when Docker uses its default private PID namespace.
    processes=$(remote_exec "$host" docker exec "$container" \
        ps -eo pid,ppid,pgid,args) || \
        return 1
    awk '
        NR == 1 { next }
        {
            pid = $1
            ppid = $2
            pgid = $3
            $1 = $2 = $3 = ""
            args = $0
            is_parent = 0
            if (args ~ /(^|\/)torchrun([[:space:]]|$)/) {
                parent_count += 1
                parent_pid = pid
                parent_pgid = pgid
                is_parent = 1
            }
            if (!is_parent && args ~ /-m[[:space:]]+n0_twam[.]train([[:space:]]|$)/) {
                worker_ppid[pid] = ppid
                worker_count += 1
            }
        }
        END {
            direct_workers = 0
            if (parent_count == 1) {
                for (pid in worker_ppid) {
                    if (worker_ppid[pid] == parent_pid) direct_workers += 1
                }
            } else {
                parent_pid = 0
                parent_pgid = 0
            }
            print parent_count + 0, worker_count + 0, parent_pid + 0,
                parent_pgid + 0, direct_workers + 0
        }
    ' <<<"$processes"
}

training_process_counts() {
    local state
    state=$(training_process_state "$1" "$2") || return 1
    read -r torch_count train_count _ <<<"$state"
    printf '%s %s\n' "$torch_count" "$train_count"
}

active_owned_job() {
    local host=$1 container=$2 rank=$3 expected_invocation=${4:-}
    local container_id expected_workers=${HCU_PER_NODE:-} program
    [[ "$expected_workers" == "8" ]] || return 1
    container_id=$(container_identity "$host" "$container") || return 1
    program='import glob,hashlib,json,os,re,stat,sys
root,rank,container_id,expected,host,expected_workers=sys.argv[1:]
rank=int(rank); expected_workers=int(expected_workers)
if not os.path.isabs(root) or root == "/" or rank < 0 or expected_workers != 8: raise SystemExit(1)
if not re.fullmatch(r"[0-9a-f]{64}",container_id): raise SystemExit(1)
active=[]

def immutable_bytes(path):
 st=os.lstat(path)
 if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_mode & 0o222 or st.st_size > 1048576: raise SystemExit(1)
 with open(path,"rb") as stream: return stream.read()

def process_stat(pid):
 with open(f"/proc/{pid}/stat","rb") as stream: raw=stream.read()
 parts=raw.rsplit(b")",1)
 if len(parts) != 2: raise SystemExit(1)
 fields=parts[1].split()
 if len(fields) <= 19: raise SystemExit(1)
 try: return fields[0].decode("ascii"),int(fields[1]),int(fields[2]),int(fields[19])
 except (UnicodeDecodeError,ValueError): raise SystemExit(1)

def process_argv(pid):
 with open(f"/proc/{pid}/cmdline","rb") as stream: raw=stream.read()
 return [os.fsdecode(value) for value in raw.split(b"\0") if value]

def process_environ(pid):
 with open(f"/proc/{pid}/environ","rb") as stream: raw=stream.read()
 result={}
 for item in raw.split(b"\0"):
  if not item: continue
  key,separator,value=item.partition(b"=")
  if not separator: raise SystemExit(1)
  try: key=key.decode("ascii"); value=value.decode("utf-8")
  except UnicodeDecodeError: raise SystemExit(1)
  if key in result: raise SystemExit(1)
  result[key]=value
 return result

def is_train_worker(argv):
 if not argv or not re.fullmatch(r"python(?:[0-9]+(?:[.]?[0-9]+)*)?",os.path.basename(argv[0])): return False
 return any(argv[index:index+2] == ["-m","n0_twam.train"] for index in range(len(argv)-1))

def recoverable_orphan(data,manifest_path,node_count):
 invocation=data["invocation_id"]; pgid=data["process_group_id"]
 sha=data["launch_manifest_sha256"]
 expected_environment={
  "N0_TRACK31_INVOCATION_ID":invocation,
  "N0_TRACK31_CONTAINER_ID":container_id,
  "NODE_RANK":str(rank),
  "NNODES":str(node_count),
  "NGPU":str(expected_workers),
  "N0_TRACK31_SAVE_ROOT":root,
  "N0_TRACK31_LAUNCH_RECEIPT":manifest_path,
  "N0_TRACK31_LAUNCH_RECEIPT_SHA256":sha,
  "N0_TRACK31_LAUNCH_MANIFEST_SHA256":sha,
  "N0_TRACK31_ORCHESTRATED_PREFLIGHT":"track31-stage-a-v2",
 }
 group_members=[]; owned_workers=[]
 for proc_path in glob.glob("/proc/[0-9]*"):
  pid=int(os.path.basename(proc_path))
  try: state,_,process_pgid,_=process_stat(pid)
  except FileNotFoundError: continue
  try: argv=process_argv(pid)
  except FileNotFoundError:
   if process_pgid == pgid: raise SystemExit(1)
   continue
  worker=is_train_worker(argv)
  if process_pgid == pgid:
   if state == "Z" or not worker: raise SystemExit(1)
   group_members.append(pid)
  if not worker: continue
  try: environment=process_environ(pid)
  except FileNotFoundError:
   if process_pgid == pgid: raise SystemExit(1)
   continue
  if all(environment.get(key) == value for key,value in expected_environment.items()):
   owned_workers.append((pid,process_pgid))
 if not group_members and not owned_workers: return False
 if not 1 <= len(group_members) <= expected_workers: raise SystemExit(1)
 if len(owned_workers) != len(group_members): raise SystemExit(1)
 if {pid for pid,_ in owned_workers} != set(group_members): raise SystemExit(1)
 if any(process_pgid != pgid for _,process_pgid in owned_workers): raise SystemExit(1)
 return True

for path in sorted(glob.glob(os.path.join(root,"launch_manifests",f"job.*.rank{rank}.json"))):
 try: data=json.loads(immutable_bytes(path))
 except (FileNotFoundError,json.JSONDecodeError,UnicodeDecodeError): raise SystemExit(1)
 invocation=data.get("invocation_id")
 if not isinstance(invocation,str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}",invocation): raise SystemExit(1)
 if os.path.basename(path) != f"job.{invocation}.rank{rank}.json": raise SystemExit(1)
 if data.get("schema_version") != 1 or data.get("container_id") != container_id or data.get("node_rank") != rank: raise SystemExit(1)
 if expected and invocation != expected: continue
 leader=data.get("leader_pid"); pgid=data.get("process_group_id"); ticks=data.get("leader_start_ticks")
 if isinstance(leader,bool) or not isinstance(leader,int) or leader <= 1 or leader != pgid: raise SystemExit(1)
 if isinstance(ticks,bool) or not isinstance(ticks,int) or ticks <= 0: raise SystemExit(1)
 sha=data.get("launch_manifest_sha256")
 if not isinstance(sha,str) or not re.fullmatch(r"[0-9a-f]{64}",sha): raise SystemExit(1)
 manifest_path=os.path.join(root,"launch_manifests",f"launch_manifest.{invocation}.json")
 try: manifest_raw=immutable_bytes(manifest_path); manifest=json.loads(manifest_raw)
 except (FileNotFoundError,json.JSONDecodeError,UnicodeDecodeError): raise SystemExit(1)
 if hashlib.sha256(manifest_raw).hexdigest() != sha: raise SystemExit(1)
 nodes=manifest.get("nodes"); topology=manifest.get("topology")
 if manifest.get("schema_version") != 2 or manifest.get("invocation_id") != invocation: raise SystemExit(1)
 if not isinstance(nodes,list) or rank >= len(nodes) or nodes[rank] != host: raise SystemExit(1)
 if topology != {"node_count":len(nodes),"hcu_per_node":expected_workers,"world_size":len(nodes)*expected_workers}: raise SystemExit(1)
 try: state,_,leader_pgid,leader_ticks=process_stat(leader)
 except FileNotFoundError:
  if recoverable_orphan(data,manifest_path,len(nodes)): active.append((invocation,pgid,leader,sha))
  continue
 if state == "Z" or leader_pgid != pgid or leader_ticks != ticks: raise SystemExit(1)
 active.append((invocation,pgid,leader,sha))
if len(active) > 1: raise SystemExit(1)
print("NONE" if not active else " ".join(map(str,active[0])))'
    remote_exec "$host" docker exec "$container" python3 -c "$program" \
        "$CONTAINER_RUN_ROOT" "$rank" "$container_id" "$expected_invocation" \
        "$host" "$expected_workers"
}

discover_owned_invocation_group() {
    local host=$1 container=$2 rank=$3 container_id
    local expected_workers=${HCU_PER_NODE:-} program
    [[ "$expected_workers" == "8" && "$FORMAL_NODE_COUNT" =~ ^[1-9][0-9]*$ ]] || \
        return 1
    container_id=$(container_identity "$host" "$container") || return 1
    program='import hashlib,json,os,re,stat,sys
contract="formal_invocation_discovery_v1"
rank,host,roster,workers,container_id,root=sys.argv[1:]
rank=int(rank); workers=int(workers); nodes=roster.split(","); node_count=len(nodes)
if rank < 0 or rank >= node_count or nodes[rank] != host or workers != 8: raise SystemExit(1)
if not re.fullmatch(r"[0-9a-f]{64}",container_id): raise SystemExit(1)
base={"N0_TRACK31_CONTAINER_ID":container_id,"NODE_RANK":str(rank),"NNODES":str(node_count),"NGPU":str(workers),"N0_TRACK31_SAVE_ROOT":root,"N0_TRACK31_ORCHESTRATED_PREFLIGHT":"track31-stage-a-v2"}

def proc_stat(pid):
 with open(f"/proc/{pid}/stat","rb") as stream: raw=stream.read()
 parts=raw.rsplit(b")",1)
 if len(parts) != 2: raise SystemExit(1)
 fields=parts[1].split()
 if len(fields) <= 2: raise SystemExit(1)
 return fields[0].decode("ascii"),int(fields[2])

def proc_env(pid):
 with open(f"/proc/{pid}/environ","rb") as stream: raw=stream.read()
 result={}
 for item in raw.split(b"\0"):
  if not item: continue
  key,separator,value=item.partition(b"=")
  if not separator: raise SystemExit(1)
  try: key=key.decode("ascii"); value=value.decode("utf-8")
  except UnicodeDecodeError: raise SystemExit(1)
  if key in result: raise SystemExit(1)
  result[key]=value
 return result

snapshot={}; matching=[]
for name in os.listdir("/proc"):
 if not name.isdigit(): continue
 pid=int(name)
 try: state,pgid=proc_stat(pid)
 except FileNotFoundError: continue
 if state == "Z": continue
 snapshot[pid]=pgid
 try: environment=proc_env(pid)
 except FileNotFoundError: continue
 if not all(environment.get(key) == value for key,value in base.items()): continue
 invocation=environment.get("N0_TRACK31_INVOCATION_ID","")
 manifest_sha=environment.get("N0_TRACK31_LAUNCH_MANIFEST_SHA256","")
 receipt_sha=environment.get("N0_TRACK31_LAUNCH_RECEIPT_SHA256","")
 receipt=environment.get("N0_TRACK31_LAUNCH_RECEIPT","")
 if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}",invocation): raise SystemExit(1)
 if not re.fullmatch(r"[0-9a-f]{64}",manifest_sha) or receipt_sha != manifest_sha: raise SystemExit(1)
 expected_receipt=os.path.join(root,"launch_manifests",f"job.{invocation}.rank{rank}.json")
 if receipt != expected_receipt: raise SystemExit(1)
 matching.append((pid,pgid,invocation,manifest_sha,environment))
if not matching:
 print("NONE"); raise SystemExit(0)
identities={(invocation,manifest_sha) for _,_,invocation,manifest_sha,_ in matching}
groups={pgid for _,pgid,_,_,_ in matching}
if len(identities) != 1 or len(groups) != 1: raise SystemExit(1)
invocation,manifest_sha=identities.pop(); pgid=groups.pop()
leaders={pid for pid,_,_,_,_ in matching}
if pgid not in leaders:
 print("NONE"); raise SystemExit(0)
expected=dict(base)
expected.update({"N0_TRACK31_INVOCATION_ID":invocation,"N0_TRACK31_LAUNCH_RECEIPT":os.path.join(root,"launch_manifests",f"job.{invocation}.rank{rank}.json"),"N0_TRACK31_LAUNCH_RECEIPT_SHA256":manifest_sha,"N0_TRACK31_LAUNCH_MANIFEST_SHA256":manifest_sha})
members=[]
for pid,process_pgid in snapshot.items():
 if process_pgid != pgid: continue
 try: environment=proc_env(pid)
 except FileNotFoundError: raise SystemExit(1)
 if not all(environment.get(key) == value for key,value in expected.items()): raise SystemExit(1)
 members.append(pid)
if not members or pgid not in members: raise SystemExit(1)
manifest_path=os.path.join(root,"launch_manifests",f"launch_manifest.{invocation}.json")
metadata=os.lstat(manifest_path)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or metadata.st_mode & 0o222 or metadata.st_size > 1048576: raise SystemExit(1)
with open(manifest_path,"rb") as stream: raw=stream.read()
if hashlib.sha256(raw).hexdigest() != manifest_sha: raise SystemExit(1)
try: manifest=json.loads(raw)
except (json.JSONDecodeError,UnicodeDecodeError): raise SystemExit(1)
if manifest.get("schema_version") != 2 or manifest.get("invocation_id") != invocation or manifest.get("nodes") != nodes: raise SystemExit(1)
if manifest.get("topology") != {"node_count":node_count,"hcu_per_node":workers,"world_size":node_count*workers}: raise SystemExit(1)
print(invocation,pgid,manifest_sha)'
    remote_exec "$host" docker exec "$container" python3 -c "$program" \
        "$rank" "$host" "$N0_TRACK31_NODES" "$expected_workers" \
        "$container_id" "$CONTAINER_RUN_ROOT"
}

active_owned_invocation_group() {
    local host=$1 container=$2 rank=$3 invocation=$4 manifest_sha=$5 node_count=$6
    local discovered actual_invocation pgid actual_sha
    [[ "$node_count" == "$FORMAL_NODE_COUNT" \
        && "$manifest_sha" =~ ^[0-9a-f]{64}$ ]] || return 1
    discovered=$(discover_owned_invocation_group "$host" "$container" "$rank") || \
        return 1
    [[ "$discovered" != "NONE" ]] || { echo NONE; return 0; }
    read -r actual_invocation pgid actual_sha <<<"$discovered"
    [[ "$actual_invocation" == "$invocation" && "$actual_sha" == "$manifest_sha" ]] || \
        return 1
    echo "$pgid"
}

signal_owned_process_group() {
    local host=$1 container=$2 signal=$3 pgid=$4
    [[ "$signal" == "TERM" || "$signal" == "KILL" ]] || return 1
    [[ "$pgid" =~ ^[1-9][0-9]*$ ]] || return 1
    remote_exec "$host" docker exec "$container" bash -c \
        'kill -"$1" -- "-$2"' _ "$signal" "$pgid"
}

publish_atomic_file() {
    local host=$1 path=$2 payload=$3 encoded program
    encoded=$(python3 -c 'import base64,sys; print(base64.b64encode(sys.argv[1].encode()).decode())' "$payload")
    program='import base64,os,sys
p=sys.argv[1]; raw=memoryview(base64.b64decode(sys.argv[2])); tmp=p+".tmp"
fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
try:
 while raw: raw=raw[os.write(fd,raw):]
 os.fsync(fd)
finally: os.close(fd)
try: os.link(tmp,p)
finally: os.unlink(tmp)
fd=os.open(os.path.dirname(p),os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)'
    remote_exec "$host" python3 -c "$program" "$path" "$encoded"
}
