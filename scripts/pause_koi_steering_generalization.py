"""Source-bound Linux episode-boundary pause; --pause sends one parent-only SIGINT.

No environment, Agent, frozen operator, or analyzer is imported. A held claims
directory flock excludes both the next parent intent and the next actual reset.
Ambiguous pre-reset children are allowed to finish with the lock released, never
classified as zero-reset from missing raw data. Unsupported kernel observations
fail closed. This is not a continuation or an extension of the frozen budget.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import stat
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = "koi-steering-generalization-v1"
FORMAT = "koi-steering-generalization-pause-v2"
PROOF_MODE = "kernel-flock-owner-waiter-v1"
PROTOCOL_SHA = "ccc6720f676814eb705e853792eceb31c6c5fb2b2aa695b1144641c2befcd7dd"
OPERATOR_SHA = "8f4e3a7929af83230f2b6bf8890df65e9b25abd78e17d0bdb14baca758d91833"
SUFFIXES = (".json", ".raw.jsonl", ".decisions.jsonl", ".process.json",
            ".bound-process.json", ".stdout.txt", ".stderr.txt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def ledger_rows(run):
    raw = (run / "reset-ledger.jsonl").read_bytes()
    require(raw.endswith(b"\n"), "empty or torn reset ledger")
    return [json.loads(line) for line in raw.splitlines()]


def slot_id(row):
    return f"{row['track_id']}:{row['seed']}:{row['mode']}"


def source_evidence(root, protocol_sha256):
    root = Path(root).resolve()
    run = root / "runs" / STUDY
    require(protocol_sha256 == PROTOCOL_SHA, "only the exact original frozen protocol is supported")
    pins = {}

    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, f"frozen evidence changed: {path}")
        pins[str(path)] = expected

    pin(root / "experiments" / f"{STUDY}.json", protocol_sha256)
    pin(run / "protocol.json", protocol_sha256)
    protocol = read_json(run / "protocol.json")
    require(protocol["study"] == STUDY and protocol["episodes"] == 144,
            "unsupported study or frozen schedule")
    schedule = []
    for seed in range(3184000001, 3184000025):
        for track in (1, 2, 3):
            arms = ["crossing_projection", "steering_release_v1"]
            if not (track + seed) % 2:
                arms.reverse()
            schedule.extend(dict(mode=arm, track_id=track, seed=seed, status="unrun") for arm in arms)
    require(protocol["schedule"] == schedule, "frozen 24-road/72-pair/144-slot schedule differs")
    require(protocol["operator_sha256"] == OPERATOR_SHA, "unsupported operator source")
    pin(root / "scripts/evaluate_koi_steering_generalization.py", OPERATOR_SHA)
    for group in ("helper_source_sha256", "environment_source_sha256", "root_agent_source_sha256"):
        for path, expected in protocol[group].items():
            pin(path, expected)
    for path, expected in protocol["reference_evidence"]["source_sha256"].items():
        pin(path, expected)
    for copy in protocol["source_copies"]:
        path = (run / copy["file"]).resolve()
        require(path.is_relative_to(run), "source copy escapes run")
        pin(path, copy["sha256"])
    for arm, name in (("crossing_projection", "crossing-projection-source-reconstruction.zip"),
                      ("steering_release_v1", "candidate.zip")):
        pin(run / name, protocol["model_hashes"][arm])
    pin(run / "candidate.manifest.json", protocol["candidate_manifest_sha256"])
    audit = protocol["freshness"]["audit_receipt"]
    pin(root / audit["path"], audit["sha256"])
    review_pin = read_json(run / "execution-review.json")
    pin(review_pin["path"], review_pin["sha256"])
    pins[str(run / "execution-review.json")] = sha(run / "execution-review.json")
    review = read_json(review_pin["path"])
    require(review["status"] == "passed" and review["environment_resets"] == 0,
            "independent review is not passing")
    for key, expected in (("protocol_sha256", protocol_sha256), ("operator_sha256", OPERATOR_SHA),
                          ("analyzer_sha256", protocol["analyzer_sha256"]),
                          ("audit_sha256", audit["sha256"])):
        require(review[key] == expected, f"review binding differs: {key}")
    pin(root / "scripts/finalize_koi_steering_generalization.py", review["finalizer_sha256"])
    pin(root / "experiments" / f"{STUDY}-evidence-guard.json", review["evidence_guard_sha256"])
    return protocol, pins


def process_identity(pid) -> dict[str, Any]:
    base = Path("/proc") / str(pid)
    stat = (base / "stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    require(fields[0] not in ("Z", "X"), "parent is no longer live")
    exe = (base / "exe").resolve(strict=True)
    return dict(pid=pid, start_ticks=int(fields[19]), ppid=int(fields[1]),
                cmdline=(base / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
                cwd=str((base / "cwd").resolve(strict=True)), exe=str(exe), exe_sha256=sha(exe),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def children(pid):
    # Children belong to individual threads; inspecting only the main thread is
    # insufficient for a no-live-child proof.
    result = set()
    for task in (Path("/proc") / str(pid) / "task").iterdir():
        try:
            result.update(int(p) for p in (task / "children").read_text().split())
        except FileNotFoundError:
            continue
    return sorted(result)


def verify_parent(identity, start_ticks, root, protocol):
    review = read_json(root / "runs" / STUDY / "execution-review.json")
    expected = ["-B", "-m", "scripts.evaluate_koi_steering_generalization", "--run",
                "--protocol-sha256", PROTOCOL_SHA, "--review", "--REVIEW--",
                "--review-sha256", review["sha256"]]
    args = identity["cmdline"]
    require(len(args) == len(expected) + 1, "target is not the exact Python operator (do not target Bash)")
    expected[7] = args[8]
    require(args[1:] == expected, "operator command differs")
    require((root / args[8]).resolve() == Path(review["path"]).resolve(), "operator review path differs")
    require(identity["start_ticks"] == start_ticks and identity["cwd"] == str(root),
            "parent identity/start time/cwd differs")
    require("python" in Path(identity["exe"]).name, "target executable is not Python")
    require(protocol["operator_sha256"] == OPERATOR_SHA, "parent source is not supported")
    status = (Path("/proc") / str(identity["pid"]) / "status").read_text().splitlines()
    masks = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in status if ":" in line}
    require(not int(masks["SigIgn"], 16) & 2 and int(masks["SigCgt"], 16) & 2,
            "Python SIGINT handler is not installed")


def wait_exit(pidfd, parent_pidfd=None, timeout: float = 180):
    ready, _, _ = select.select([pidfd] + ([] if parent_pidfd is None else [parent_pidfd]), [], [], timeout)
    require(pidfd in ready, "process exit not observed within bounded wait")
    require(parent_pidfd is None or parent_pidfd not in ready,
            "operator exited while waiting for child; no pause signal sent")


def blocked_flock(pid, descriptor, registry=None) -> dict[str, Any]:
    """Direct kernel owner/waiter proof; never read protected syscall data."""
    require(platform.system() == "Linux", "kernel FLOCK proof requires Linux")
    require(type(pid) is int and pid > 0 and pid != os.getpid(), "invalid parent waiter PID")
    registry = Path(registry) if registry is not None else ROOT / "experiments/train-seed-claims"
    info = os.fstat(descriptor)
    current = registry.stat()
    require(stat.S_ISDIR(info.st_mode) and stat.S_ISDIR(current.st_mode), "claims lock is not a directory")
    require((info.st_dev, info.st_ino) == (current.st_dev, current.st_ino),
            "claims directory inode changed")
    token = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    lines = Path("/proc/locks").read_text().splitlines()
    owners, waiters = [], []
    for line in lines:
        words = line.split()
        waiting = len(words) > 1 and words[1] == "->"
        fields = words[2:] if waiting else words[1:]
        if len(fields) == 7 and fields[:3] == ["FLOCK", "ADVISORY", "WRITE"] and fields[4:] == [token, "0", "EOF"]:
            if waiting and fields[3] == str(pid):
                waiters.append(line)
            elif not waiting and fields[3] == str(os.getpid()):
                owners.append(line)
    require(len(owners) == len(waiters) == 1, "kernel does not prove parent waiting on manager-owned directory flock")
    lock_id = owners[0].split()[0]
    require(lock_id.endswith(":") and lock_id[:-1].isdigit() and lock_id == waiters[0].split()[0],
            "kernel waiter is not linked to manager owner")
    require(not children(pid), "parent still has a live child")
    return dict(proof_mode=PROOF_MODE, path=str(registry), device=info.st_dev,
                inode=info.st_ino, kernel_lock_lines=owners + waiters,
                manager_pid=os.getpid(), parent_pid=pid)


def completed_evidence(run, protocol):
    rows = ledger_rows(run)
    require(len(rows) % 2 == 0 and 0 < len(rows) < 2 * len(protocol["schedule"]),
            "no strict incomplete completed-episode boundary")
    pins, completed = {}, []

    def pin(name, expected=None):
        path = (run / name).resolve()
        require(path.is_relative_to(run), "artifact escapes run")
        actual = sha(path)
        require(expected is None or actual == expected, f"completed artifact hash differs: {name}")
        pins[str(path)] = actual
        return path

    for index, slot in enumerate(protocol["schedule"]):
        output = run / f"{slot['track_id']}-{slot['seed']}-{slot['mode']}.json"
        if index >= len(rows) // 2:
            require(not any(output.with_suffix(s).exists() for s in SUFFIXES),
                    "unexecuted slot already has artifacts; no zero-reset inference")
            continue
        intent, row = rows[2 * index:2 * index + 2]
        identity = slot_id(slot)
        require(intent["status"] == "reset_intent" and row["status"] == "completed" and
                intent["slot_id"] == row["slot_id"] == identity and
                (intent["track"], intent["seed"], intent["arm"]) ==
                (slot["track_id"], slot["seed"], slot["mode"]), "ledger is not an exact completed prefix")
        require((row["track_id"], row["seed"], row["mode"]) ==
                (slot["track_id"], slot["seed"], slot["mode"]), "completed row identity differs")
        require(row["file"] == output.name and row["process_file"] == output.with_suffix(".bound-process.json").name,
                "completed artifact names differ")
        episode = read_json(pin(row["file"], row["sha256"]))
        require((episode["track_id"], episode["seed"], episode["mode"]) ==
                (slot["track_id"], slot["seed"], slot["mode"]), "episode identity differs")
        require(not episode["error"] and not episode["invalid_actions"] and not row["error"] and
                not row["invalid_actions"], "completed episode has operational failure")
        require(episode["retire_reason"] not in {"act_timeout", "reset_timeout", "agent_reset_error",
                "agent_setup_error", "evaluation_error", "invalid_action", "unknown", "max_steps"},
                "completed slot is censored/operationally invalid")
        for name, expected in ((episode["raw_trace_file"], episode["raw_trace_sha256"]),
                               (episode["partial_decisions_file"], episode["partial_decisions_sha256"])):
            pin(name, expected)
        process = read_json(pin(row["process_file"], row["process_sha256"]))
        require(process["status"] == "completed" and not process["error"] and
                process["slot_id"] == identity and process["operator_sha256"] == OPERATOR_SHA,
                "child process completion/source binding differs")
        original = read_json(pin(process["original_process_file"], process["original_process_sha256"]))
        require(original["status"] == "completed" and not original["error"], "child did not exit naturally")
        for suffix in SUFFIXES:
            pin(output.with_suffix(suffix).name)
        completed.append(row)
    return completed, pins, sha(run / "reset-ledger.jsonl")


def terminal_evidence(run, protocol, completed, ledger_sha256, review_pin) -> dict[str, Any]:
    report_path, failure_path = run / "episode-report.json", run / "operator-failure.json"
    report, failure = read_json(report_path), read_json(failure_path)
    require(report["protocol_sha256"] == PROTOCOL_SHA and not report["complete"] and
            report["operator_error"] == failure["error"] == "KeyboardInterrupt: ",
            "original operator did not finalize an immutable KeyboardInterrupt partial failure")
    require(report["reset_ledger_sha256"] == sha(run / "reset-ledger.jsonl") == ledger_sha256,
            "reset ledger changed after boundary SIGINT")
    require(report["review_receipt"] == review_pin, "terminal report review binding differs")
    rows = report["rows"]
    require(len(rows) == len(protocol["schedule"]), "terminal schedule length differs")
    require(rows[:len(completed)] == completed, "completed rows changed during finalization")
    for index, slot in enumerate(protocol["schedule"][len(completed):], len(completed)):
        row = rows[index]
        require((row["track_id"], row["seed"], row["mode"]) ==
                (slot["track_id"], slot["seed"], slot["mode"]), "terminal unexecuted identity differs")
        if index == len(completed):
            require(row["status"] == "operator_error" and row["error"] == "KeyboardInterrupt: " and
                    row.get("partial_artifacts") == [], "pre-intent failure has partial child evidence")
        else:
            require(row == slot, "later unexecuted row was modified")
    return dict(process_exited=True, report_path=str(report_path), report_sha256=sha(report_path),
                failure_path=str(failure_path), failure_sha256=sha(failure_path),
                reset_ledger_sha256=ledger_sha256)


def pause(pid, start_ticks, receipt_path, *, root=ROOT,
          settle_seconds: float = 30, wait_seconds: float = 180):
    root, receipt_path = Path(root).resolve(), Path(receipt_path).resolve()
    # Like process/execution receipts, this is run evidence, not a new allocation
    # descriptor. Root experiment metadata would collide with the frozen audit.
    require(receipt_path == root / "runs" / STUDY / "boundary-pause.json",
            "receipt must use the canonical run boundary-pause.json path")
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
            "interpreter lacks public pidfd APIs; use /usr/bin/python3")
    require(0 < settle_seconds <= 60 and 0 < wait_seconds <= 600, "invalid bounded wait")
    protocol, sources = source_evidence(root, PROTOCOL_SHA)
    require(str(receipt_path) not in sources, "receipt collides with pinned evidence")
    run, registry = root / "runs" / STUDY, root / "experiments/train-seed-claims"
    require(not (run / "episode-report.json").exists() and not (run / "operator-failure.json").exists(),
            "operator already finalized; do not signal it")
    identity = process_identity(pid)
    verify_parent(identity, start_ticks, root, protocol)
    parent_fd = os.pidfd_open(pid)
    lock_fd = None
    receipt: dict[str, Any] = dict(format=FORMAT, status="refused", coordinator_sha256=sha(__file__),
                   protocol_sha256=PROTOCOL_SHA, process=identity, source_sha256=sources,
                   signal=None, environment_resets_by_coordinator=0, model_updates=0,
                   official_action=False, observations=[])
    try:
        with receipt_path.open("x") as stream:
            try:
                for attempt in range(3):
                    lock_fd = os.open(registry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    current_registry, locked_registry = os.stat(registry), os.fstat(lock_fd)
                    require((current_registry.st_dev, current_registry.st_ino) ==
                            (locked_registry.st_dev, locked_registry.st_ino), "claims directory inode changed")
                    live = children(pid)
                    require(len(live) <= 1, "unexpected concurrent operator children")
                    if not live:
                        break
                    child_pid = live[0]
                    child_fd = os.pidfd_open(child_pid)
                    try:
                        child = process_identity(child_pid)
                        rows = ledger_rows(run)
                        require(rows[-1]["status"] == "reset_intent", "live child has no pending intent")
                        intent = rows[-1]
                        output = run / f"{intent['track']}-{intent['seed']}-{intent['arm']}.json"
                        command = (f"import sys;sys.path.insert(0,{str(root)!r});from scripts.evaluate_koi_steering_generalization "
                                   f"import worker;worker({intent['arm']!r},{intent['track']},{intent['seed']},{str(output)!r})")
                        require(child["cmdline"] == [protocol["python"], "-I", "-c", command] and
                                child["ppid"] == pid and not children(child_pid), "child identity is not source-bound")
                        raw = output.with_suffix(".raw.jsonl")
                        # A complete raw tick is a positive post-reset witness in
                        # this exact source. Absence/empty raw proves nothing.
                        post_reset = False
                        if raw.exists():
                            with raw.open("rb") as raw_stream:
                                first = raw_stream.readline()
                            if first.endswith(b"\n"):
                                tick = json.loads(first)
                                post_reset = type(tick.get("step")) is int and tick["step"] > 0
                        receipt["observations"].append(dict(child=child, post_reset_witness=post_reset,
                                                            lock_held_during_wait=post_reset))
                        if not post_reset:
                            os.close(lock_fd)
                            lock_fd = None
                        wait_exit(child_fd, parent_fd, wait_seconds)
                        if post_reset:
                            break
                    finally:
                        os.close(child_fd)
                require(lock_fd is not None, "no lock-held completed boundary acquired; rerun after review")
                # One bounded pidfd wait, not a timer/poll loop. The held lock
                # makes the parent's transition to its next intent harmless.
                ready, _, _ = select.select([parent_fd], [], [], settle_seconds)
                require(not ready, "operator exited naturally before pause proof")
                snapshots = []
                completed, artifacts, ledger_sha256 = [], {}, ""
                for observation in range(2):
                    require(not select.select([parent_fd], [], [], 0)[0], "parent pidfd is already exited")
                    require(process_identity(pid) == identity, "parent identity changed before signal")
                    lock_snapshot = blocked_flock(pid, lock_fd, registry)
                    completed, artifacts, ledger_sha256 = completed_evidence(run, protocol)
                    _, checked_sources = source_evidence(root, PROTOCOL_SHA)
                    require(checked_sources == sources, "source bindings changed while acquiring boundary")
                    require(sha(run / "reset-ledger.jsonl") == ledger_sha256, "ledger changed while locked")
                    require(process_identity(pid) == identity and not children(pid), "parent identity/children changed")
                    lock_snapshot.update(process=identity, no_live_children=True,
                                         completed_slots=len(completed), reset_ledger_sha256=ledger_sha256,
                                         source_state_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                                         artifact_state_sha256=hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest())
                    snapshots.append(lock_snapshot)
                first, second = snapshots
                require({k: v for k, v in first.items() if k != "kernel_lock_lines"} ==
                        {k: v for k, v in second.items() if k != "kernel_lock_lines"},
                        "boundary source/process/ledger/artifacts changed between kernel snapshots")
                lock = dict(proof_mode=PROOF_MODE, path=str(registry), device=second["device"],
                            inode=second["inode"], manager_pid=second["manager_pid"],
                            parent_pid=pid, snapshots=snapshots)
                receipt.update(lock=lock, artifact_sha256=artifacts,
                               boundary=dict(completed_slots=len(completed), reset_ledger_sha256=ledger_sha256,
                                             pending_reset_intents=0, no_live_children=True,
                                             no_new_reset=True, child_interrupted=False))
                # Persist the proof before the only allowed external action.
                receipt["status"] = "boundary_verified"
                stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                signal.pidfd_send_signal(parent_fd, signal.SIGINT)
                receipt["signal"] = dict(name="SIGINT", pidfd=True, pid=pid, sent_at=time.time())
                wait_exit(parent_fd, timeout=wait_seconds)
                uptime = float(Path("/proc/uptime").read_text().split()[0])
                hz = os.sysconf("SC_CLK_TCK")
                receipt["terminal"] = terminal_evidence(run, protocol, completed, ledger_sha256,
                                                        read_json(run / "execution-review.json"))
                receipt["terminal"].update(wall_time_s=uptime - identity["start_ticks"] / hz,
                                           wall_clock=dict(start_ticks=identity["start_ticks"],
                                                           clock_ticks_per_second=hz, observed_uptime_s=uptime,
                                                           endpoint="first pidfd exit observation",
                                                           precision="quantized /proc times plus observation latency; upper-bound forecast"))
                for path, expected in artifacts.items():
                    require(sha(path) == expected, "completed artifact changed after SIGINT")
                receipt["status"] = "paused"
            except BaseException as error:
                receipt["status"] = "failed_after_signal" if receipt["signal"] else "refused"
                receipt["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                stream.seek(0)
                stream.truncate()
                stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(parent_fd)
    return receipt


def verify_boundary(path, expected_sha256, *, root=ROOT):
    """Read-only authentication for a later separately reviewed continuation."""
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path == root / "runs" / STUDY / "boundary-pause.json", "noncanonical pause receipt path")
    require(sha(path) == expected_sha256, "pause receipt hash differs")
    receipt = read_json(path)
    require(receipt["format"] == FORMAT and receipt["status"] == "paused", "no successful pause receipt")
    require(receipt["coordinator_sha256"] == sha(__file__), "pause coordinator source changed")
    require(receipt["signal"]["name"] == "SIGINT" and receipt["signal"]["pidfd"] is True and
            receipt["signal"]["pid"] == receipt["process"]["pid"], "not a parent-only pidfd SIGINT")
    boundary = receipt["boundary"]
    require(boundary["pending_reset_intents"] == 0 and boundary["no_live_children"] is True and
            boundary["no_new_reset"] is True and boundary["child_interrupted"] is False,
            "pause does not prove a safe completed boundary")
    lock = receipt["lock"]
    registry = root / "experiments/train-seed-claims"
    info = registry.stat()
    require(lock["path"] == str(registry) and (lock["device"], lock["inode"]) ==
            (info.st_dev, info.st_ino), "pause lock inode/path differs")
    token = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    require(lock["proof_mode"] == PROOF_MODE and lock["parent_pid"] == receipt["process"]["pid"] and
            type(lock["manager_pid"]) is int and lock["manager_pid"] > 0 and
            lock["manager_pid"] != lock["parent_pid"] and len(lock["snapshots"]) == 2,
            "unsupported kernel owner/waiter proof mode or identity")
    for snapshot in lock["snapshots"]:
        require(all(snapshot[k] == lock[k] for k in
                    ("proof_mode", "path", "device", "inode", "manager_pid", "parent_pid")) and
                snapshot["process"] == receipt["process"] and snapshot["no_live_children"] is True and
                snapshot["completed_slots"] == boundary["completed_slots"] and
                snapshot["reset_ledger_sha256"] == boundary["reset_ledger_sha256"] and
                snapshot["source_state_sha256"] == hashlib.sha256(json.dumps(receipt["source_sha256"], sort_keys=True).encode()).hexdigest() and
                snapshot["artifact_state_sha256"] == hashlib.sha256(json.dumps(receipt["artifact_sha256"], sort_keys=True).encode()).hexdigest(),
                "kernel snapshot process/source/ledger/artifact state differs")
        lines = [line.split() for line in snapshot["kernel_lock_lines"]]
        require(len(lines) == 2 and len(lines[0]) == 8 and len(lines[1]) == 9 and
                lines[0][0].endswith(":") and lines[0][0][:-1].isdigit() and
                lines[0][0] == lines[1][0] and lines[0][1:] ==
                ["FLOCK", "ADVISORY", "WRITE", str(lock["manager_pid"]), token, "0", "EOF"] and
                lines[1][1:] == ["->", "FLOCK", "ADVISORY", "WRITE", str(lock["parent_pid"]), token, "0", "EOF"],
                "pause lacks consistent kernel parent-blocked-flock proof")
    protocol, sources = source_evidence(root, receipt["protocol_sha256"])
    require(sources == receipt["source_sha256"], "pause frozen source pins differ")
    run = root / "runs" / STUDY
    completed, artifacts, ledger_hash = completed_evidence(run, protocol)
    require(artifacts == receipt["artifact_sha256"] and len(completed) == boundary["completed_slots"] and
            ledger_hash == boundary["reset_ledger_sha256"], "pause completed evidence changed")
    terminal = terminal_evidence(run, protocol, completed, ledger_hash,
                                 read_json(run / "execution-review.json"))
    require(all(receipt["terminal"][k] == value for k, value in terminal.items()),
            "terminal pause evidence changed")
    clock = receipt["terminal"]["wall_clock"]
    require(clock["start_ticks"] == receipt["process"]["start_ticks"] and
            clock["clock_ticks_per_second"] > 0 and receipt["terminal"]["wall_time_s"] > 0 and
            receipt["terminal"]["wall_time_s"] ==
            clock["observed_uptime_s"] - clock["start_ticks"] / clock["clock_ticks_per_second"],
            "terminal wall-clock derivation differs")
    process = receipt["process"]
    if process["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        try:
            current = process_identity(process["pid"])
        except (FileNotFoundError, ProcessLookupError, ValueError):
            current = None
        require(current is None or current["start_ticks"] != process["start_ticks"],
                "original operator is still live")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", action="store_true", required=True,
                        help="explicitly execute reviewed parent-only boundary SIGINT")
    parser.add_argument("--parent-pid", type=int, required=True, help="Python PID, NOT tracked Bash wrapper")
    parser.add_argument("--parent-start-ticks", type=int, required=True, help="/proc/PID/stat field 22")
    parser.add_argument("--receipt", type=Path, required=True,
                        help="runs/koi-steering-generalization-v1/boundary-pause.json (exclusive creation)")
    parser.add_argument("--settle-seconds", type=float, default=30)
    parser.add_argument("--wait-seconds", type=float, default=180)
    args = parser.parse_args()
    result = pause(args.parent_pid, args.parent_start_ticks, args.receipt,
                   settle_seconds=args.settle_seconds, wait_seconds=args.wait_seconds)
    print(json.dumps(dict(status=result["status"], receipt=str(args.receipt.resolve()),
                          sha256=sha(args.receipt), completed_slots=result["boundary"]["completed_slots"])))


if __name__ == "__main__":
    main()
