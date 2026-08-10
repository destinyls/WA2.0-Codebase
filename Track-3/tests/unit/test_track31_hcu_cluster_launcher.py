from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "script/track3_1/launch_hcu_stage_a.sh"

_FAKE_SSH = r"""#!/usr/bin/env python3
import base64, hashlib, json, os, shlex, sys
from pathlib import Path, PurePosixPath

host, raw = sys.argv[-2], sys.argv[-1]
cmd = shlex.split(raw)
state = Path(os.environ["MOCK_STATE"])
state.mkdir(parents=True, exist_ok=True)
with (state / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps({"argv": sys.argv[1:], "host": host, "cmd": cmd}) + "\n")

def flag(kind):
    return state / f"{kind}.{host}"

def existing():
    return os.environ.get("MOCK_EXISTING_CONTAINERS") == "1" or flag("created").exists()

def launch_manifest_sha():
    path = state / "launch_manifest_sha"
    return path.read_text() if path.exists() else "d" * 64

def materialize_delayed_registration():
    pending = flag("pending")
    if not pending.exists():
        return
    remaining = int(pending.read_text())
    if remaining > 0:
        pending.write_text(str(remaining - 1))
        return
    pending.unlink()
    flag("launched").touch()
    flag("orphan").unlink(missing_ok=True)

if not flag("initialized").exists():
    if os.environ.get("MOCK_INITIAL_LAUNCHED") == "1":
        flag("launched").touch()
    if os.environ.get("MOCK_INITIAL_ORPHAN") == "1":
        flag("launched").touch()
        flag("orphan").touch()
    if host == os.environ.get("MOCK_INITIAL_PENDING_HOST"):
        flag("pending").write_text(os.environ.get("MOCK_DELAYED_ACK_CHECKS", "1000"))
    flag("initialized").touch()

joined = " ".join(cmd)
image_id = os.environ.get("N0_TRACK31_IMAGE_ID", "sha256:" + "a" * 64)
train = os.environ.get("N0_TRACK31_TRAIN759_ROOT_HOST", "/data/train759")
parent = str(PurePosixPath(train).parent)
if cmd[:3] == ["docker", "image", "inspect"]:
    print(image_id)
elif cmd[:3] == ["docker", "container", "inspect"]:
    if not existing():
        sys.exit(1)
    if "--format" not in cmd:
        print("[]")
    else:
        fmt = cmd[cmd.index("--format") + 1]
        if fmt == "{{.Image}}": print(image_id)
        elif fmt == "{{.Id}}": print("f" * 64)
        elif fmt == "{{.State.Running}}": print("true")
        elif fmt == "{{.HostConfig.Privileged}}": print("false")
        elif fmt == "{{json .Config.Entrypoint}}": print('["bash"]')
        elif fmt == "{{json .Config.Cmd}}": print('["-lc","while :; do sleep 3600; done"]')
        elif fmt == "{{.Config.WorkingDir}}": print("/workspace/N0-TWAM")
        elif fmt == "{{.HostConfig.NetworkMode}}": print("host")
        elif fmt == "{{.HostConfig.IpcMode}}": print("host")
        elif fmt == "{{json .HostConfig.GroupAdd}}": print('["video"]')
        elif fmt == "{{json .HostConfig.CapAdd}}": print('["SYS_PTRACE"]')
        elif fmt == "{{json .HostConfig.SecurityOpt}}":
            print('["seccomp=unconfined","label=disable"]')
        elif fmt == "{{json .HostConfig.Devices}}":
            print('[{"PathOnHost":"/dev/kfd","PathInContainer":"/dev/kfd","CgroupPermissions":"rwm"},{"PathOnHost":"/dev/dri","PathInContainer":"/dev/dri","CgroupPermissions":"rwm"}]')
        elif "n0.track31.owner" in fmt: print("formal-stage-a-v1")
        elif "creation_token" in fmt:
            print(flag("label").read_text() if flag("label").exists() else "")
        elif ".Mounts" in fmt:
            mounts = [
                ("/opt/hyhal", "/opt/hyhal", "false"),
                ("/etc/hfm", "/etc/hfm", "false"),
                (os.environ["N0_TRACK31_SOURCE_ROOT_HOST"], "/workspace/N0-TWAM", "false"),
                (os.environ["N0_TRACK31_ARTIFACT_ROOT_HOST"], "/formal/artifacts", "false"),
                (train, train, "false"),
                (f"{parent}/.materialization_state.json", f"{parent}/.materialization_state.json", "false"),
                (os.environ["N0_TRACK31_MODEL_ROOT_HOST"], "/formal/model", "false"),
                (os.environ["N0_TRACK31_OVERLAY_ROOT_HOST"], "/formal/overlay", "false"),
                (os.environ["N0_TRACK31_RUN_ROOT_HOST"], "/formal/run", "true"),
            ]
            if host == os.environ.get("MOCK_MOUNT_MISMATCH_HOST"):
                mounts[-1] = (mounts[-1][0], mounts[-1][1], "false")
            print("\n".join("\t".join(("bind", *item)) for item in mounts))
elif cmd[:2] == ["docker", "run"]:
    if host == os.environ.get("MOCK_CREATE_FAIL_HOST"):
        sys.exit(1)
    token = cmd[cmd.index("--label") + 1].split("=", 1)[1]
    flag("label").write_text(token)
    flag("created").touch()
    if host == os.environ.get("MOCK_CREATE_GHOST_HOST"):
        sys.exit(1)
    print("container-id")
elif cmd[:3] == ["docker", "rm", "-f"]:
    flag("created").unlink(missing_ok=True)
    flag("label").unlink(missing_ok=True)
elif cmd[:2] == ["docker", "exec"]:
    if "-d" in cmd:
        if host == os.environ.get("MOCK_LAUNCH_FAIL_HOST"):
            sys.exit(1)
        if host == os.environ.get("MOCK_DELAYED_ACK_HOST"):
            flag("pending").write_text(os.environ.get("MOCK_DELAYED_ACK_CHECKS", "2"))
        else:
            flag("launched").touch()
            flag("orphan").unlink(missing_ok=True)
        if host == os.environ.get("MOCK_LEADER_DIES_AFTER_LAUNCH_HOST"):
            flag("orphan").touch()
    elif cmd[3:6] == ["ps", "-eo", "pid,ppid,pgid,args"]:
        print("PID PPID PGID COMMAND")
        if flag("launched").exists():
            nnodes = len(os.environ["N0_TRACK31_NODES"].split(","))
            orphan = flag("orphan").exists()
            if not orphan:
                print(f"100 1 100 /usr/bin/python3 /usr/local/bin/torchrun --nnodes={nnodes} -m n0_twam.train")
            replace = host == os.environ.get("MOCK_ORPHAN_REPLACES_DIRECT_HOST")
            if flag("partial").exists():
                count = int(flag("partial").read_text())
            else:
                count = 7 if host == os.environ.get("MOCK_UNHEALTHY_HOST") or replace else 8
            for rank in range(count):
                ppid = 1 if orphan else 100
                pgid = 101 + rank
                if (
                    orphan
                    and host == os.environ.get("MOCK_ORPHAN_MISMATCH_HOST")
                    and os.environ.get("MOCK_ORPHAN_MISMATCH_KIND") == "pgid"
                    and rank == count - 1
                ):
                    pgid = 200
                print(f"{101 + rank} {ppid} {pgid} /usr/bin/python3 -m n0_twam.train --rank={rank}")
            if host == os.environ.get("MOCK_EXTRA_WORKER_HOST") or replace:
                print("999 1 999 /usr/bin/python3 -m n0_twam.train --unowned")
            if orphan and host == os.environ.get("MOCK_ORPHAN_EXTRA_WORKER_HOST"):
                print("999 1 100 /usr/bin/python3 -m n0_twam.train --rank=8")
    elif cmd[3:] == ["hostname"]:
        print("mock-" + host)
    elif "device_names" in joined:
        print(json.dumps({"python":"3.10.0","platform":"linux","torch":"2.5.1","hip":"6.3","cuda_available":True,"device_count":8,"device_names":["BW"] * 8,"python_overlay":"/formal/overlay"}, sort_keys=True, separators=(",", ":")))
    elif "pip" in cmd and "freeze" in cmd:
        print("tiktoken==0.12.0\ntorch==2.5.1\ntransformers==4.49.0")
    elif "hy-smi --version" in joined:
        print("hy-smi mock-1")
    elif "formal_invocation_discovery_v1" in joined:
        if flag("pending").exists() or (
            flag("launched").exists() and not flag("orphan").exists()
        ):
            print(os.environ["N0_TRACK31_INVOCATION_ID"] + " 100 " + launch_manifest_sha())
        else:
            print("NONE")
    elif "glob.glob" in joined:
        materialize_delayed_registration()
        if flag("launched").exists() and flag("orphan").exists():
            required_orphan_checks = (
                "/proc/[0-9]*",
                "/environ",
                "N0_TRACK31_INVOCATION_ID",
                "N0_TRACK31_CONTAINER_ID",
                "NODE_RANK",
                "N0_TRACK31_LAUNCH_MANIFEST_SHA256",
                "expected_workers",
                "launch_manifest.",
            )
            if not all(check in joined for check in required_orphan_checks):
                sys.exit(98)
            if host == os.environ.get("MOCK_ORPHAN_MISMATCH_HOST"):
                sys.exit(1)
            if host == os.environ.get("MOCK_ORPHAN_EXTRA_WORKER_HOST"):
                sys.exit(1)
        if flag("launched").exists():
            print(os.environ["N0_TRACK31_INVOCATION_ID"] + " 100 100 " + launch_manifest_sha())
        else:
            print("NONE")
    elif "sha256sum" in cmd:
        print("f" * 64 + "  " + cmd[-1])
    elif "preflight_train.py" in joined:
        if host == os.environ.get("MOCK_PREFLIGHT_FAIL_HOST"):
            sys.exit(1)
    elif any(token in {"TERM", "KILL"} for token in cmd):
        if host == os.environ.get("MOCK_STOP_FAIL_HOST"):
            sys.exit(1)
        if cmd[-1] != "100":
            sys.exit(1)
        signal = "KILL" if "KILL" in cmd else "TERM"
        if signal == "TERM" and host == os.environ.get("MOCK_TERM_IGNORED_HOST"):
            pass
        elif signal == "TERM" and host == os.environ.get("MOCK_PARTIAL_ORPHAN_AFTER_TERM_HOST"):
            flag("launched").touch()
            flag("orphan").touch()
            flag("partial").write_text("3")
        else:
            flag("launched").unlink(missing_ok=True)
            flag("orphan").unlink(missing_ok=True)
            flag("partial").unlink(missing_ok=True)
            flag("pending").unlink(missing_ok=True)
elif cmd[:3] == ["docker", "ps", "-a"]:
    if existing():
        if cmd[cmd.index("--format") + 1] == "{{.Names}}":
            print(os.environ["N0_TRACK31_CONTAINER_NAME"])
        else:
            print(os.environ["N0_TRACK31_CONTAINER_NAME"] + "\tUp")
elif cmd and cmd[0] == "realpath":
    print(cmd[-1].rstrip("/") or "/")
elif cmd and cmd[0] == "hostname":
    print("mock-" + host)
elif cmd[:2] == ["python3", "-c"] and "base64.b64decode" in joined:
    published_path = cmd[3]
    if "/launch_manifests/launch_manifest." in published_path:
        (state / "launch_manifest_sha").write_text(
            hashlib.sha256(base64.b64decode(cmd[4])).hexdigest()
        )
elif cmd and cmd[0] == "sha256sum":
    if cmd[-1].endswith(".source_manifest.sha256"):
        print(os.environ["N0_TRACK31_SOURCE_MANIFEST_SHA256"] + "  " + cmd[-1])
    elif cmd[-1].endswith(".overlay_manifest.sha256"):
        print(os.environ["N0_TRACK31_OVERLAY_MANIFEST_SHA256"] + "  " + cmd[-1])
    elif cmd[-1].endswith("empty_emb.pt"):
        print(os.environ["N0_EMPTY_EMBEDDING_SHA256"] + "  " + cmd[-1])
    elif "/launch_manifests/launch_manifest." in cmd[-1]:
        print(launch_manifest_sha() + "  " + cmd[-1])
    else:
        print("d" * 64 + "  " + cmd[-1])
elif cmd[:2] == ["test", "-s"]:
    if not flag("launched").exists(): sys.exit(1)
elif cmd and cmd[0] in {"test", "find", "mkdir", "python3", "/usr/bin/python3", "bash"}:
    pass
else:
    print("unhandled fake ssh command: " + repr(cmd), file=sys.stderr)
    sys.exit(97)
"""


def _environment(tmp_path: Path, *, existing: bool = True) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    ssh = fake_bin / "ssh"
    ssh.write_text(_FAKE_SSH, encoding="utf-8")
    ssh.chmod(0o755)
    sleep = fake_bin / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sleep.chmod(0o755)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("n1 ssh-ed25519 AAAA\n", encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("N0_TRACK31_")
    }
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "LC_ALL": "C",
            "MOCK_STATE": str(tmp_path / "state"),
            "MOCK_EXISTING_CONTAINERS": "1" if existing else "0",
            "N0_TRACK31_KNOWN_HOSTS": str(known_hosts),
            "N0_TRACK31_NODES": "n1,n2,n3,n4,n5,n6",
            "N0_TRACK31_CONTAINER_NAME": "n0-track31-formal",
            "N0_TRACK31_IMAGE": "registry.example/n0:v1",
            "N0_TRACK31_IMAGE_ID": "sha256:" + "a" * 64,
            "N0_TRACK31_SOURCE_ROOT_HOST": "/src",
            "N0_TRACK31_SOURCE_MANIFEST_SHA256": "b" * 64,
            "N0_TRACK31_OVERLAY_MANIFEST_SHA256": "e" * 64,
            "N0_TRACK31_ARTIFACT_ROOT_HOST": "/artifact",
            "N0_TRACK31_TRAIN759_ROOT_HOST": "/data/train759",
            "N0_TRACK31_RAW_ROOT_HOST": "/raw",
            "N0_TRACK31_MODEL_ROOT_HOST": "/model",
            "N0_EMPTY_EMBEDDING_SHA256": "f" * 64,
            "N0_TRACK31_OVERLAY_ROOT_HOST": "/overlay",
            "N0_TRACK31_RUN_ROOT_HOST": "/run",
            "N0_TRACK31_COLLECTIVE_SMOKE_REPORT": (
                "/formal/run/preflight/collective_smoke.phase20-001.json"
            ),
            "N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256": "d" * 64,
            "N0_RELEASED_TRANSFORMER_SHA256": "c" * 64,
            "N0_TRACK31_RUN_ROLE": "development",
            "N0_TRACK31_STOP_AFTER_STEP": "20",
            "N0_TRACK31_INVOCATION_ID": "phase20-001",
            "N0_TRACK31_HEALTH_ATTEMPTS": "2",
            "N0_TRACK31_HEALTH_POLL_SECONDS": "0",
        }
    )
    return environment


def _run(
    tmp_path: Path, action: str, **updates: str
) -> subprocess.CompletedProcess[str]:
    environment = _environment(tmp_path, existing=updates.pop("existing", "1") == "1")
    environment.update(updates)
    return subprocess.run(
        ["bash", str(LAUNCHER), action],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _calls(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "state/calls.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_hcu_cluster_launcher_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_status_only_requires_operational_configuration(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    for key in tuple(environment):
        if key.startswith("N0_TRACK31_") and key not in {
            "N0_TRACK31_KNOWN_HOSTS",
            "N0_TRACK31_NODES",
            "N0_TRACK31_CONTAINER_NAME",
        }:
            environment.pop(key)
    result = subprocess.run(
        ["bash", str(LAUNCHER), "status"],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    assert len({call["host"] for call in calls}) == 6
    assert all(
        call["argv"][:2] == ["-p", "36000"]
        and "StrictHostKeyChecking=yes" in call["argv"]
        for call in calls
    )


def test_status_accepts_direct_workers_with_separate_process_groups(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "status",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_RUN_ROLE="final_refit",
        MOCK_INITIAL_LAUNCHED="1",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("torchrun=1 train=8 direct=8") == 2
    discovery_calls = [
        call
        for call in _calls(tmp_path)
        if "formal_invocation_discovery_v1" in " ".join(call["cmd"])
    ]
    assert discovery_calls == []


def test_formal_recipe_rejects_environment_drift_before_ssh(tmp_path: Path) -> None:
    result = _run(tmp_path, "launch", N0_TRACK31_BATCH_SIZE="3")
    assert result.returncode != 0
    assert "N0_TRACK31_BATCH_SIZE" in result.stderr
    assert _calls(tmp_path) == []


def test_formal_recipe_rejects_step_and_checkpoint_interval_drift_before_ssh(
    tmp_path: Path,
) -> None:
    invalid_recipes = (
        {"N0_TRACK31_NUM_STEPS": "5000"},
        {"N0_TRACK31_SAVE_INTERVAL": "500"},
    )
    for index, updates in enumerate(invalid_recipes):
        case_root = tmp_path / str(index)
        result = _run(case_root, "launch", **updates)
        assert result.returncode != 0
        assert _calls(case_root) == []


def test_formal_recipe_rejects_retired_accumulation_recipe_before_ssh(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "launch",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_BATCH_SIZE="1",
        N0_TRACK31_GRADIENT_ACCUMULATION_STEPS="12",
    )
    assert result.returncode != 0
    assert "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS" in result.stderr
    assert _calls(tmp_path) == []


def test_invalid_node_image_and_resume_are_fail_closed(tmp_path: Path) -> None:
    invalid_cases = (
        {"N0_TRACK31_NODES": "n1"},
        {"N0_TRACK31_NODES": "n1,n2,n3"},
        {"N0_TRACK31_RUN_ROLE": "evaluation"},
        {"N0_TRACK31_NODES": "n1,n2,n3,n4,n5,-oProxyCommand=bad"},
        {"N0_TRACK31_IMAGE": "REGISTRY.EXAMPLE/n0:v1"},
        {
            "N0_TRACK31_STOP_AFTER_STEP": "25",
            "N0_TRACK31_RESUME_FROM": "/tmp/checkpoint",
        },
        {
            "N0_TRACK31_STOP_AFTER_STEP": "25",
            "N0_TRACK31_RESUME_FROM": "/formal/run/checkpoints/checkpoint_step_19",
        },
        {
            "N0_TRACK31_STOP_AFTER_STEP": "1500",
            "N0_TRACK31_RESUME_FROM": "/formal/run/checkpoints/checkpoint_step_20",
        },
    )
    for index, updates in enumerate(invalid_cases):
        case_root = tmp_path / str(index)
        result = _run(case_root, "launch", **updates)
        assert result.returncode != 0
        assert _calls(case_root) == []


def test_rank_zero_foreground_preflight_gates_detached_launch(tmp_path: Path) -> None:
    result = _run(tmp_path, "launch", MOCK_PREFLIGHT_FAIL_HOST="n1")
    assert result.returncode != 0
    calls = _calls(tmp_path)
    preflights = [
        call for call in calls if "preflight_train.py" in " ".join(call["cmd"])
    ]
    launches = [call for call in calls if call["cmd"][:3] == ["docker", "exec", "-d"]]
    manifest_publications = [
        call
        for call in calls
        if call["cmd"]
        and call["cmd"][0] == "python3"
        and "launch_manifest.phase20-001.json" in " ".join(call["cmd"])
    ]
    assert [call["host"] for call in preflights] == ["n1"]
    assert launches == []
    assert manifest_publications == []


def test_health_failure_globally_terminates_training(tmp_path: Path) -> None:
    result = _run(tmp_path, "launch", MOCK_UNHEALTHY_HOST="n3")
    assert result.returncode != 0
    calls = _calls(tmp_path)
    term_hosts = {call["host"] for call in calls if "TERM" in call["cmd"]}
    assert term_hosts == {f"n{i}" for i in range(1, 7)}


def test_non_direct_or_extra_workers_fail_exact_health_contract(
    tmp_path: Path,
) -> None:
    for index, variable in enumerate(
        ("MOCK_EXTRA_WORKER_HOST", "MOCK_ORPHAN_REPLACES_DIRECT_HOST")
    ):
        result = _run(tmp_path / str(index), "launch", **{variable: "n3"})
        assert result.returncode != 0
        term_hosts = {
            call["host"]
            for call in _calls(tmp_path / str(index))
            if "TERM" in call["cmd"]
        }
        assert term_hosts == {f"n{i}" for i in range(1, 7)}


def test_leader_death_health_cleanup_allows_next_launch(tmp_path: Path) -> None:
    topology = {
        "N0_TRACK31_NODES": "n1,n2",
        "N0_TRACK31_RUN_ROLE": "final_refit",
        "N0_TRACK31_COLLECTIVE_SMOKE_REPORT": (
            "/formal/run/preflight/collective_smoke.phase20-001.json"
        ),
    }
    failed = _run(
        tmp_path,
        "launch",
        **topology,
        MOCK_LEADER_DIES_AFTER_LAUNCH_HOST="n2",
    )
    assert failed.returncode != 0
    first_calls = _calls(tmp_path)
    terms = [call for call in first_calls if "TERM" in call["cmd"]]
    assert {call["host"] for call in terms} == {"n1", "n2"}
    assert all(call["cmd"][-1] == "100" for call in terms)

    relaunched = _run(
        tmp_path,
        "launch",
        **{
            **topology,
            "N0_TRACK31_INVOCATION_ID": "phase20-002",
            "N0_TRACK31_COLLECTIVE_SMOKE_REPORT": (
                "/formal/run/preflight/collective_smoke.phase20-002.json"
            ),
        },
    )
    assert relaunched.returncode == 0, relaunched.stderr
    new_calls = _calls(tmp_path)[len(first_calls) :]
    launches = [
        call for call in new_calls if call["cmd"][:3] == ["docker", "exec", "-d"]
    ]
    assert {call["host"] for call in launches} == {"n1", "n2"}


def test_stop_recovers_exact_receipt_bound_orphans(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "stop",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_RUN_ROLE="final_refit",
        MOCK_INITIAL_ORPHAN="1",
    )
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    terms = [call for call in calls if "TERM" in call["cmd"]]
    assert {call["host"] for call in terms} == {"n1", "n2"}
    assert all(call["cmd"][-1] == "100" for call in terms)
    orphan_checks = [
        " ".join(call["cmd"])
        for call in calls
        if call["cmd"][:2] == ["docker", "exec"]
        and "glob.glob" in " ".join(call["cmd"])
    ]
    assert orphan_checks
    assert all("/proc/[0-9]*" in command for command in orphan_checks[:2])


def test_stop_kills_receipt_bound_partial_orphan_after_term(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "stop",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_RUN_ROLE="final_refit",
        MOCK_INITIAL_LAUNCHED="1",
        MOCK_PARTIAL_ORPHAN_AFTER_TERM_HOST="n2",
    )

    assert result.returncode == 0, result.stderr
    signals = [
        call
        for call in _calls(tmp_path)
        if any(token in {"TERM", "KILL"} for token in call["cmd"])
    ]
    assert {call["host"] for call in signals if "TERM" in call["cmd"]} == {
        "n1",
        "n2",
    }
    assert [call["host"] for call in signals if "KILL" in call["cmd"]] == ["n2"]
    ownership_source = (REPO_ROOT / "script/track3_1/hcu_stage_a_lib.sh").read_text(
        encoding="utf-8"
    )
    assert "1 <= len(group_members) <= expected_workers" in ownership_source


def test_orphan_recovery_rejects_identity_pgid_and_roster_mismatch(
    tmp_path: Path,
) -> None:
    cases = [
        {
            "MOCK_ORPHAN_MISMATCH_HOST": "n2",
            "MOCK_ORPHAN_MISMATCH_KIND": kind,
        }
        for kind in ("invocation", "container", "rank", "manifest", "pgid")
    ]
    cases.append({"MOCK_ORPHAN_EXTRA_WORKER_HOST": "n2"})
    for index, updates in enumerate(cases):
        case_root = tmp_path / str(index)
        result = _run(
            case_root,
            "stop",
            N0_TRACK31_NODES="n1,n2",
            N0_TRACK31_RUN_ROLE="final_refit",
            MOCK_INITIAL_ORPHAN="1",
            **updates,
        )
        assert result.returncode != 0
        signals = [
            call
            for call in _calls(case_root)
            if any(token in {"TERM", "KILL"} for token in call["cmd"])
        ]
        assert signals == []


def test_partial_container_creation_rolls_back_only_new_containers(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "containers", existing="0", MOCK_CREATE_FAIL_HOST="n3")
    assert result.returncode != 0
    removals = [
        call for call in _calls(tmp_path) if call["cmd"][:3] == ["docker", "rm", "-f"]
    ]
    assert [call["host"] for call in removals] == ["n1", "n2"]


def test_created_container_contract_failure_is_rolled_back(tmp_path: Path) -> None:
    result = _run(tmp_path, "containers", existing="0", MOCK_MOUNT_MISMATCH_HOST="n1")
    assert result.returncode != 0
    removals = [
        call for call in _calls(tmp_path) if call["cmd"][:3] == ["docker", "rm", "-f"]
    ]
    assert [call["host"] for call in removals] == ["n1"]


def test_uncertain_docker_ack_rolls_back_token_owned_container(tmp_path: Path) -> None:
    result = _run(tmp_path, "containers", existing="0", MOCK_CREATE_GHOST_HOST="n2")
    assert result.returncode != 0
    removals = [
        call for call in _calls(tmp_path) if call["cmd"][:3] == ["docker", "rm", "-f"]
    ]
    assert [call["host"] for call in removals] == ["n1", "n2"]


def test_successful_container_creation_uses_exact_nonprivileged_mounts(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "containers", existing="0")
    assert result.returncode == 0, result.stderr
    runs = [call for call in _calls(tmp_path) if call["cmd"][:2] == ["docker", "run"]]
    assert len(runs) == 6
    for call in runs:
        command = call["cmd"]
        assert "--privileged" not in command
        assert command.count("--security-opt") == 2
        assert "seccomp=unconfined" in command
        assert "label=disable" in command
        assert command[command.index("--entrypoint") + 1] == "bash"
        assert "type=bind,src=/data/train759,dst=/data/train759,readonly" in command
        assert not any("/raw" in token for token in command)


def test_detached_launch_failure_globally_terminates_started_ranks(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "launch", MOCK_LAUNCH_FAIL_HOST="n3")
    assert result.returncode != 0
    calls = _calls(tmp_path)
    terms = [call for call in calls if "TERM" in call["cmd"]]
    assert {call["host"] for call in terms} == {"n2"}


def test_delayed_registration_then_later_failure_kills_and_allows_restart(
    tmp_path: Path,
) -> None:
    topology = {
        "N0_TRACK31_NODES": "n1,n2",
        "N0_TRACK31_RUN_ROLE": "final_refit",
    }
    failed = _run(
        tmp_path,
        "launch",
        **topology,
        MOCK_DELAYED_ACK_HOST="n2",
        MOCK_DELAYED_ACK_CHECKS="3",
        MOCK_LAUNCH_FAIL_HOST="n1",
        MOCK_TERM_IGNORED_HOST="n2",
    )

    assert failed.returncode != 0
    first_calls = _calls(tmp_path)
    launches = [
        call for call in first_calls if call["cmd"][:3] == ["docker", "exec", "-d"]
    ]
    assert [call["host"] for call in launches] == ["n2", "n1"]
    rank_zero_launch_index = first_calls.index(launches[-1])
    assert any(
        call["host"] == "n2" and "glob.glob" in " ".join(call["cmd"])
        for call in first_calls[:rank_zero_launch_index]
    )
    signals = [
        call
        for call in first_calls
        if any(token in {"TERM", "KILL"} for token in call["cmd"])
    ]
    assert {call["host"] for call in signals if "TERM" in call["cmd"]} == {"n2"}
    assert [call["host"] for call in signals if "KILL" in call["cmd"]] == ["n2"]
    assert "stable process count is 0 0 on all nodes" in failed.stderr

    status = _run(tmp_path, "status", **topology)
    assert status.returncode == 0, status.stderr
    assert status.stdout.count("invocation=none torchrun=0 train=0 direct=0") == 2

    relaunched = _run(
        tmp_path,
        "launch",
        **topology,
        N0_TRACK31_INVOCATION_ID="phase20-002",
        N0_TRACK31_COLLECTIVE_SMOKE_REPORT=(
            "/formal/run/preflight/collective_smoke.phase20-002.json"
        ),
    )
    assert relaunched.returncode == 0, relaunched.stderr


def test_registration_timeout_kills_pre_receipt_group_before_return(
    tmp_path: Path,
) -> None:
    topology = {
        "N0_TRACK31_NODES": "n1,n2",
        "N0_TRACK31_RUN_ROLE": "final_refit",
    }
    failed = _run(
        tmp_path,
        "launch",
        **topology,
        MOCK_DELAYED_ACK_HOST="n2",
        MOCK_DELAYED_ACK_CHECKS="1000",
        MOCK_TERM_IGNORED_HOST="n2",
    )

    assert failed.returncode != 0
    calls = _calls(tmp_path)
    launches = [call for call in calls if call["cmd"][:3] == ["docker", "exec", "-d"]]
    assert [call["host"] for call in launches] == ["n2"]
    assert any(
        call["host"] == "n2"
        and "formal_invocation_discovery_v1" in " ".join(call["cmd"])
        for call in calls
    )
    assert [call["host"] for call in calls if "KILL" in call["cmd"]] == ["n2"]
    assert not (tmp_path / "state/pending.n2").exists()
    assert "stable process count is 0 0 on all nodes" in failed.stderr

    status = _run(tmp_path, "status", **topology)
    assert status.returncode == 0, status.stderr
    assert status.stdout.count("invocation=none torchrun=0 train=0 direct=0") == 2


def test_crash_window_pending_group_blocks_launch_and_is_stoppable(
    tmp_path: Path,
) -> None:
    topology = {
        "N0_TRACK31_NODES": "n1,n2",
        "N0_TRACK31_RUN_ROLE": "final_refit",
    }
    blocked = _run(
        tmp_path,
        "launch",
        **topology,
        MOCK_INITIAL_PENDING_HOST="n2",
        MOCK_DELAYED_ACK_CHECKS="1000",
    )

    assert blocked.returncode != 0
    assert "formal invocation already active or registering" in blocked.stderr
    assert not any(
        call["cmd"][:3] == ["docker", "exec", "-d"] for call in _calls(tmp_path)
    )

    status = _run(tmp_path, "status", **topology)
    assert status.returncode != 0
    assert "registration_pending invocation=phase20-001 pgid=100" in status.stdout

    stopped = _run(tmp_path, "stop", **topology)
    assert stopped.returncode == 0, stopped.stderr
    assert not (tmp_path / "state/pending.n2").exists()
    assert any(
        call["host"] == "n2" and "TERM" in call["cmd"] for call in _calls(tmp_path)
    )

    clean_status = _run(tmp_path, "status", **topology)
    assert clean_status.returncode == 0, clean_status.stderr
    assert clean_status.stdout.count("invocation=none torchrun=0 train=0 direct=0") == 2


def test_stop_aggregates_node_failures_without_short_circuit(tmp_path: Path) -> None:
    result = _run(tmp_path, "stop", MOCK_STOP_FAIL_HOST="n2", MOCK_INITIAL_LAUNCHED="1")
    assert result.returncode != 0
    terms = [call for call in _calls(tmp_path) if "TERM" in call["cmd"]]
    assert {call["host"] for call in terms} == {f"n{i}" for i in range(1, 7)}
    assert all(
        call["cmd"][-1] == "100" and "pgrep" not in " ".join(call["cmd"])
        for call in terms
    )


def test_successful_launch_writes_manifest_and_proves_exact_processes(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "launch")
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    manifest_calls = [
        call
        for call in calls
        if call["cmd"]
        and call["cmd"][0] == "python3"
        and "launch_manifest.phase20-001.json" in " ".join(call["cmd"])
    ]
    launches = [call for call in calls if call["cmd"][:3] == ["docker", "exec", "-d"]]
    assert len(manifest_calls) == 1
    manifest_raw = __import__("base64").b64decode(manifest_calls[0]["cmd"][-1])
    manifest_sha = __import__("hashlib").sha256(manifest_raw).hexdigest()
    manifest = json.loads(manifest_raw)
    assert manifest["ssh_port"] == 36000 and manifest["known_hosts_sha256"]
    assert manifest["environment_manifest"]["sha256"] == "d" * 64
    assert manifest["collective_smoke"] == {
        "path": "/formal/run/preflight/collective_smoke.phase20-001.json",
        "sha256": "d" * 64,
    }
    assert manifest["schema_version"] == 2
    assert manifest["run_role"] == "development"
    assert manifest["topology"] == {
        "node_count": 6,
        "hcu_per_node": 8,
        "world_size": 48,
    }
    assert manifest["recipe"]["batch_size"] == 1
    assert manifest["recipe"]["gradient_accumulation_steps"] == 1
    assert manifest["recipe"]["num_steps"] == 1500
    assert manifest["recipe"]["save_interval"] == 300
    assert manifest["recipe"]["val_interval"] == 100
    assert manifest["dataset_contract"] == {
        "physical_split": "train759",
        "train_view_id": "stage_a_dev719_v1",
        "train_episode_count": 719,
        "validation_view_id": "internal_dev40_v1",
        "normalizer_id": "qpos8_dev719_v1",
    }
    assert len(launches) == 6
    assert all(
        f"N0_TRACK31_LAUNCH_MANIFEST_SHA256={manifest_sha}" in " ".join(call["cmd"])
        for call in launches
    )
    assert all("phase20-001" in " ".join(call["cmd"]) for call in launches)
    assert all("track31-stage-a-v2" in " ".join(call["cmd"]) for call in launches)
    assert all(
        "N0_TRACK31_BATCH_SIZE=1" in " ".join(call["cmd"])
        and "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS=1" in " ".join(call["cmd"])
        for call in launches
    )
    assert all(
        "N0_EMPTY_EMBEDDING_SHA256" in " ".join(call["cmd"]) for call in launches
    )
    container_rosters = [
        call
        for call in calls
        if call["cmd"][:2] == ["docker", "exec"]
        and call["cmd"][-1] == ".source_manifest.sha256"
    ]
    assert len(container_rosters) == 6
    overlay_rosters = [
        call
        for call in calls
        if call["cmd"][:2] == ["docker", "exec"]
        and call["cmd"][-1] == ".overlay_manifest.sha256"
    ]
    assert len(overlay_rosters) == 6
    process_inventories = [
        call
        for call in calls
        if call["cmd"][:3] == ["docker", "exec", "n0-track31-formal"]
        and call["cmd"][3:6] == ["ps", "-eo", "pid,ppid,pgid,args"]
    ]
    assert len(process_inventories) >= 12
    assert not any(call["cmd"][:2] == ["docker", "top"] for call in calls)
    assert "launch_manifest_sha256=" in result.stdout


def test_four_node_final_refit_binds_32_hcu_and_step1500_recipe(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "launch",
        N0_TRACK31_NODES="n1,n2,n3,n4",
        N0_TRACK31_RUN_ROLE="final_refit",
    )
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    launches = [call for call in calls if call["cmd"][:3] == ["docker", "exec", "-d"]]
    manifest_calls = [
        call
        for call in calls
        if call["cmd"]
        and call["cmd"][0] == "python3"
        and "launch_manifest.phase20-001.json" in " ".join(call["cmd"])
    ]
    assert len(launches) == 4
    assert len(manifest_calls) == 1
    manifest_raw = __import__("base64").b64decode(manifest_calls[0]["cmd"][-1])
    manifest = json.loads(manifest_raw)
    assert manifest["topology"] == {
        "node_count": 4,
        "hcu_per_node": 8,
        "world_size": 32,
    }
    assert manifest["recipe"] == {
        "num_steps": 1500,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_interval": 300,
        "val_interval": 100,
        "max_latent_frames": 5,
        "load_worker": 0,
        "train_seed": 20260801,
        "action_init_seed": 0,
        "pythonhashseed": 20260801,
    }
    joined_launches = [" ".join(call["cmd"]) for call in launches]
    assert all("NNODES=4" in command for command in joined_launches)
    assert all("N0_TRACK31_NUM_STEPS=1500" in command for command in joined_launches)
    assert all("N0_TRACK31_SAVE_INTERVAL=300" in command for command in joined_launches)
    assert "healthy_process_contract=4x(1_torchrun+8_train)" in result.stdout


def test_hsdp_performance_recipe_is_bound_and_forwarded(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "launch",
        N0_TRACK31_NODES="n1,n2,n3,n4",
        N0_TRACK31_RUN_ROLE="final_refit",
        N0_FSDP_TOPOLOGY="hsdp",
        N0_FSDP_SHARD_SIZE="8",
        N0_FSDP_EXPERT_RESHARD_POLICY="after_backward",
        N0_TRACK31_SAMPLER_RANK_ALIGNMENT="shape_balanced",
        N0_MOT_CROSS_ATTENTION_BACKEND="flash_attn",
    )

    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    launches = [
        call for call in calls if call["cmd"][:3] == ["docker", "exec", "-d"]
    ]
    manifest_calls = [
        call
        for call in calls
        if call["cmd"]
        and call["cmd"][0] == "python3"
        and "launch_manifest.phase20-001.json" in " ".join(call["cmd"])
    ]
    manifest_raw = __import__("base64").b64decode(manifest_calls[0]["cmd"][-1])
    performance = json.loads(manifest_raw)["performance"]
    joined_launches = [" ".join(call["cmd"]) for call in launches]

    assert performance["fsdp"] == {
        "topology": "hsdp",
        "mesh_shape": [4, 8],
        "replicate_size": 4,
        "shard_size": 8,
        "expert_reshard_policy": "after_backward",
        "keep_expert_params_between_pre_post": True,
        "reduce_dtype": "bfloat16",
    }
    assert performance["sampler_rank_alignment"] == "shape_balanced"
    assert performance["mot_cross_attention_backend"] == "flash_attn"
    assert all("N0_FSDP_TOPOLOGY=hsdp" in command for command in joined_launches)
    assert all(
        "N0_FSDP_EXPERT_RESHARD_POLICY=after_backward" in command
        for command in joined_launches
    )
    assert all(
        "N0_TRACK31_SAMPLER_RANK_ALIGNMENT=shape_balanced" in command
        for command in joined_launches
    )
    assert all(
        "N0_MOT_CROSS_ATTENTION_BACKEND=flash_attn" in command
        for command in joined_launches
    )


def test_two_node_final_refit_binds_exact_topology_and_role(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "launch",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_RUN_ROLE="final_refit",
    )
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    launches = [call for call in calls if call["cmd"][:3] == ["docker", "exec", "-d"]]
    manifest_calls = [
        call
        for call in calls
        if call["cmd"]
        and call["cmd"][0] == "python3"
        and "launch_manifest.phase20-001.json" in " ".join(call["cmd"])
    ]
    assert len(launches) == 2
    assert len(manifest_calls) == 1
    manifest_raw = __import__("base64").b64decode(manifest_calls[0]["cmd"][-1])
    manifest_sha = __import__("hashlib").sha256(manifest_raw).hexdigest()
    manifest = json.loads(manifest_raw)
    assert manifest["schema_version"] == 2
    assert manifest["nodes"] == ["n1", "n2"]
    assert manifest["run_role"] == "final_refit"
    assert manifest["topology"] == {
        "node_count": 2,
        "hcu_per_node": 8,
        "world_size": 16,
    }
    assert manifest["multinode_hcu_preflight"] == "passed"
    assert manifest["recipe"]["batch_size"] == 1
    assert manifest["recipe"]["gradient_accumulation_steps"] == 1
    assert manifest["dataset_contract"] == {
        "physical_split": "train759",
        "train_view_id": "stage_a_final759_v1",
        "train_episode_count": 759,
        "validation_view_id": None,
        "normalizer_id": "qpos8_final759_v1",
    }
    joined_launches = [" ".join(call["cmd"]) for call in launches]
    assert all(
        "NNODES=2" in command and "NGPU=8" in command for command in joined_launches
    )
    assert all("N0_TRACK31_BATCH_SIZE=1" in command for command in joined_launches)
    assert all(
        "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS=1" in command
        for command in joined_launches
    )
    assert all("MASTER_ADDR=n1" in command for command in joined_launches)
    assert all(
        "N0_TRACK31_RUN_ROLE=final_refit" in command for command in joined_launches
    )
    assert all("track31-stage-a-v2" in command for command in joined_launches)
    assert any("NODE_RANK=0" in command for command in joined_launches)
    assert any("NODE_RANK=1" in command for command in joined_launches)
    runtime_probes = [
        " ".join(call["cmd"])
        for call in calls
        if "device_names" in " ".join(call["cmd"])
    ]
    assert len(runtime_probes) == 2
    assert all(
        "HIP_VISIBLE_DEVICES=0,1,5,4,2,3,7,6" in probe for probe in runtime_probes
    )
    assert all("source /opt/hyhal/env.sh" in probe for probe in runtime_probes)
    assert all("/usr/bin/python3 -B -c" in probe for probe in runtime_probes)
    foreground_preflights = [
        " ".join(call["cmd"])
        for call in calls
        if "preflight_train.py" in " ".join(call["cmd"])
    ]
    assert len(foreground_preflights) == 1
    assert (
        "PYTHONPATH=/formal/overlay:/workspace/N0-TWAM:/workspace/N0-TWAM/n0_twam"
        in foreground_preflights[0]
    )
    assert (
        f"N0_TRACK31_LAUNCH_MANIFEST_SHA256={manifest_sha}" in foreground_preflights[0]
    )
    assert all(
        f"N0_TRACK31_LAUNCH_MANIFEST_SHA256={manifest_sha}" in command
        for command in joined_launches
    )
    assert "healthy_process_contract=2x(1_torchrun+8_train)" in result.stdout


def test_two_node_health_failure_terminates_exact_roster(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "launch",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_RUN_ROLE="final_refit",
        MOCK_UNHEALTHY_HOST="n2",
    )
    assert result.returncode != 0
    term_hosts = {call["host"] for call in _calls(tmp_path) if "TERM" in call["cmd"]}
    assert term_hosts == {"n1", "n2"}
