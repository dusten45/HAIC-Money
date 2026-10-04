"""Source-bound fresh camera-policy comparison on a new official geometry grid.

The committed template contains no salt or geometry seeds. Explicit binding to
one committed agent.py creates both source snapshots and a new protocol. This
module never starts an episode before that protocol and its inputs validate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_camera_policy_generalization as previous
from tools import evaluate_bare_generalization as fresh

NAME = "camera-policy-generalization-v2"
TEMPLATE_PATH = ROOT / "experiments" / "camera-policy-generalization-v2.template.json"
PROTOCOL_PATH = ROOT / "experiments" / f"{NAME}.json"
OUTPUT_ROOT = ROOT / ".haic-artifacts" / NAME / "run"
SOURCE_PATHS = {arm: f".haic-artifacts/{NAME}/snapshots/{arm}_agent.py"
                for arm in fresh.ARMS}
CONTROL_CLASS = "_CompoundClearingBrakeCarryController"
TRACKS = [1, 2, 3, 4]
SEED_COUNTS = {"screen": 8, "confirmation": 16, "blind": 8}
SPOT_INDEX = {"screen": ((1, 0), (4, 7)),
              "confirmation": ((1, 0), (2, 1), (3, 2), (4, 3)),
              "blind": ((1, 0), (4, 7))}
DECISION_ENGINE_SHA256 = "756434beadd4eee9bffdc94237bb6abb80813975a5e611f4a8f0320e5d527917"
THRESHOLDS = {
    "minimum_net_finish_gain": {"screen": 3, "confirmation": 5, "blind": 3},
    "maximum_lost_control_finishes": {"screen": 1, "confirmation": 2, "blind": 1},
    "maximum_new_candidate_crashes": {"screen": 1, "confirmation": 2, "blind": 1},
    "moderate_both_dnf_progress_loss": .05,
    "maximum_moderate_both_dnf_losses": {"screen": 1, "confirmation": 2, "blind": 1},
    "severe_both_dnf_progress_loss": .15,
    "maximum_severe_both_dnf_losses": 0,
    "maximum_single_cell_contact_increase": 2,
    "maximum_single_cell_damage_increase": .4,
    "maximum_aggregate_shared_finish_time_ratio": 1.10,
    "maximum_single_shared_finish_time_ratio": 1.15,
    "maximum_aggregate_contact_increase": 0,
    "maximum_aggregate_damage_increase": 0.0,
}
compare_pairs = previous.compare_pairs
report_phase = previous.report_phase
require_phase = previous.require_phase
freeze_finalist = previous.freeze_finalist
seal_confirmation = previous.seal_confirmation


def validate_template(template: dict) -> None:
    expected = {"schema_version": 1, "name": NAME, "status": "UNBOUND",
                "control_class": CONTROL_CLASS, "track_ids": TRACKS,
                "seed_counts": SEED_COUNTS,
                "decision_engine_sha256": DECISION_ENGINE_SHA256,
                "spot_indices": {phase: [list(cell) for cell in SPOT_INDEX[phase]]
                                 for phase in fresh.PHASES}}
    if not isinstance(template, dict) or any(template.get(key) != value
                                             for key, value in expected.items()):
        raise ValueError("unbound template contract changed")
    if "partitions" in template or "seed_salt_hex" in template:
        raise ValueError("unbound template must contain no geometry seeds")


def derived_seeds(salt: str, phase: str) -> list[int]:
    if not isinstance(salt, str) or not re.fullmatch(r"[0-9a-f]{32}", salt):
        raise ValueError("seed salt must be 128-bit lowercase hex")
    if phase not in SEED_COUNTS:
        raise ValueError("unknown partition")
    return [int.from_bytes(hashlib.sha256(
        f"{NAME}:{salt}:{phase}:{index}".encode("ascii")).digest()[:4], "big")
        for index in range(SEED_COUNTS[phase])]


def _decision_engine_sha256() -> str:
    return fresh.digest(Path(previous.__file__))


def _require_v1_engine() -> None:
    if _decision_engine_sha256() != DECISION_ENGINE_SHA256 or previous.THRESHOLDS != THRESHOLDS:
        raise ValueError("frozen v1 decision engine changed")


def _environment_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "core").rglob("*.py")) + sorted((ROOT / "local_simulator").rglob("*.py"))
    paths += [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): fresh.digest(path)
            for path in paths}


def validate_protocol(protocol: dict, historical_seeds: set[int]) -> None:
    # This guard precedes any seed derivation or worker dispatch.
    if protocol.get("name") != NAME or protocol.get("status") != "PREREGISTERED":
        raise ValueError("unbound protocol: committed source and salt are required")
    salt = protocol.get("seed_salt_hex")
    if not isinstance(salt, str) or not re.fullmatch(r"[0-9a-f]{32}", salt):
        raise ValueError("bound protocol needs its one-time seed salt")
    _require_v1_engine()
    if protocol.get("decision_engine_sha256") != DECISION_ENGINE_SHA256:
        raise ValueError("registered v1 decision engine hash changed")
    if (protocol.get("runner_sha256") != fresh.digest(Path(__file__))
            or protocol.get("evaluation_harness_sha256") != fresh.digest(Path(fresh.__file__))):
        raise ValueError("registered runner or evaluation harness hash changed")
    if protocol.get("environment_sha256") != _environment_hashes():
        raise ValueError("registered environment hashes changed")
    fresh.validate_protocol(protocol, historical_seeds)
    if protocol.get("decision_thresholds") != THRESHOLDS:
        raise ValueError("strict v1 decision thresholds changed")
    if protocol.get("source_snapshots") != SOURCE_PATHS:
        raise ValueError("registered source snapshot paths changed")
    classes = protocol["controller_classes"]
    candidate_class = classes["candidate"]
    if (classes["control"] != CONTROL_CLASS or not re.fullmatch(r"_[A-Za-z0-9_]+", candidate_class)
            or candidate_class == CONTROL_CLASS):
        raise ValueError("control or candidate route class invalid")
    construction = protocol.get("source_construction", {})
    commit = construction.get("implementation_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source construction needs a full committed Git SHA")
    fresh._sha256_string(construction.get("agent_blob_sha256"), "agent_blob_sha256")
    if (construction["agent_blob_sha256"] != protocol["control_agent_sha256"]
            or construction.get("selector_from") != f"{CONTROL_CLASS}()"
            or construction.get("selector_to") != f"{candidate_class}()"):
        raise ValueError("source selector construction is inconsistent")
    if protocol.get("training") is not False:
        raise ValueError("only a fixed bare Agent may enter this evaluation")
    all_seeds = []
    for phase in fresh.PHASES:
        part = protocol["partitions"][phase]
        seeds = derived_seeds(salt, phase)
        all_seeds.extend(seeds)
        if part["track_ids"] != TRACKS or part["seeds"] != seeds:
            raise ValueError(f"{phase} registered track or seed grid changed")
        expected_spots = [[track, seeds[index]] for track, index in SPOT_INDEX[phase]]
        if part.get("spot_check_cells") != expected_spots:
            raise ValueError(f"{phase} deterministic spot checks changed")
    if len(set(all_seeds)) != sum(SEED_COUNTS.values()):
        raise ValueError("geometry seed collision across partitions")


def _committed_agent_source(commit: str, candidate_class: str) -> tuple[str, bytes]:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError("binding needs a committed Git SHA")
    if (not isinstance(candidate_class, str)
            or not re.fullmatch(r"_[A-Za-z0-9_]+", candidate_class)
            or candidate_class == CONTROL_CLASS):
        raise ValueError("binding needs a distinct candidate class")
    try:
        full = subprocess.run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
                              cwd=ROOT, capture_output=True, check=True).stdout.decode("ascii").strip()
        source = subprocess.run(["git", "show", f"{full}:agent.py"], cwd=ROOT,
                                capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise ValueError("binding needs an existing committed agent.py") from error
    old = f"{CONTROL_CLASS}()".encode("ascii")
    declaration = rb"(?m)^class " + candidate_class.encode("ascii") + rb"\b"
    if source.count(old) != 1 or re.search(declaration, source) is None:
        raise ValueError("committed agent.py must contain one control selector and candidate class")
    return full, source


def bind_source(template: dict, implementation_commit: str, candidate_class: str,
                model: Path, protocol_path: Path, control: Path, candidate: Path) -> dict:
    """Bind a committed candidate once; this is the only salt-generation path."""
    validate_template(template)
    if any(path.exists() for path in (protocol_path, control, candidate)):
        raise ValueError("bound protocol or source snapshot already exists")
    commit, source = _committed_agent_source(implementation_commit, candidate_class)
    _require_v1_engine()
    selected = source.replace(f"{CONTROL_CLASS}()".encode("ascii"),
                              f"{candidate_class}()".encode("ascii"), 1)
    model_hash = fresh.digest(model)
    helpers = {name: fresh.digest(ROOT / name) for name in
               ("action_smoothing.py", "action_representation.py")}
    environment_hashes = _environment_hashes()
    harness_hash = fresh.digest(Path(fresh.__file__))
    runner_hash = fresh.digest(Path(__file__))
    # All executable source hashes are known before any new geometry exists.
    salt = secrets.token_hex(16)
    seeds = {phase: derived_seeds(salt, phase) for phase in fresh.PHASES}
    all_seeds = [seed for phase in fresh.PHASES for seed in seeds[phase]]
    historical = fresh.historical_geometry_seeds(ROOT / "experiments")
    if len(set(all_seeds)) != len(all_seeds) or set(all_seeds) & historical:
        raise ValueError("fresh seed collision; do not open any cell; retry binding")
    partitions = {phase: {"track_ids": TRACKS, "seeds": seeds[phase],
                          "spot_check_cells": [[track, seeds[phase][index]]
                                               for track, index in SPOT_INDEX[phase]]}
                  for phase in fresh.PHASES}
    protocol = {
        "schema_version": 1, "name": NAME, "status": "PREREGISTERED",
        "hypothesis": "A distinct frozen camera-only controller improves unseen official-generator completion while satisfying the unchanged v1 safety and pace gates.",
        "evidence_scope": "New source-bound paired official-generator screen, confirmation and blind; all earlier cells are development only.",
        "control_agent_sha256": hashlib.sha256(source).hexdigest(),
        "candidate_agent_sha256": hashlib.sha256(selected).hexdigest(),
        "model_sha256": model_hash, "runtime_helper_sha256": helpers,
        "controller_classes": {"control": CONTROL_CLASS, "candidate": candidate_class},
        "source_snapshots": SOURCE_PATHS,
        "source_construction": {"implementation_commit": commit,
                                "agent_blob_sha256": hashlib.sha256(source).hexdigest(),
                                "selector_from": f"{CONTROL_CLASS}()",
                                "selector_to": f"{candidate_class}()",
                                "rule": "Candidate snapshot differs from committed agent.py solely in the Agent route selector."},
        "seed_salt_hex": salt, "seed_derivation": "First four bytes of SHA256(name:salt:phase:index), unsigned big-endian uint32; salt generated once after source validation.",
        "seed_freshness_audit": f"All 32 seeds are distinct and absent from {len(historical)} documented historical or reserved geometry seeds at binding.",
        "decision_engine_sha256": DECISION_ENGINE_SHA256,
        "runner_sha256": runner_hash, "evaluation_harness_sha256": harness_hash,
        "environment_sha256": environment_hashes,
        "max_steps": 2000, "frame_skip": 4, "training": False,
        "partitions": partitions, "decision_thresholds": THRESHOLDS,
        "decision_rule": "Unchanged camera-policy-generalization-v1 comparator, fixed before any v2 episode; screen and confirmation RETAIN seals unlock later phases.",
        "evaluation_budget": "32/64/32 canonical paired cells plus 2/4/2 paired deterministic spot checks; maximum 272 cold episodes.",
        "promotion_gate": "Only blind RETAIN after screen and confirmation RETAIN supports activating the identical frozen source.",
    }
    validate_protocol(protocol, historical)
    created = []
    try:
        for path, content in ((control, source), (candidate, selected)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
            created.append(path)
        # The registered protocol is written last; a failed bind opens no phase.
        fresh._atomic_json(protocol_path, protocol)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return protocol


def validate_source_pair(protocol: dict, control: Path, candidate: Path) -> None:
    source = control.read_bytes()
    selected = candidate.read_bytes()
    construction = protocol["source_construction"]
    old = construction["selector_from"].encode("ascii")
    new = construction["selector_to"].encode("ascii")
    if source.count(old) != 1 or selected != source.replace(old, new, 1):
        raise ValueError("candidate source must differ solely in the Agent route selector")
    if (fresh.digest(control) != protocol["control_agent_sha256"]
            or fresh.digest(candidate) != protocol["candidate_agent_sha256"]):
        raise ValueError("source snapshot SHA256 mismatch")


def verify_committed_source(protocol: dict, control: Path) -> None:
    commit = protocol["source_construction"]["implementation_commit"]
    completed = subprocess.run(["git", "show", f"{commit}:agent.py"], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if completed.stdout != control.read_bytes():
        raise ValueError("control snapshot differs from committed agent.py")


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path]) -> dict:
    identity = fresh.build_identity(protocol_path, protocol, paths["control"],
                                    paths["candidate"], paths["model"])
    return {**identity, "runner_sha256": fresh.digest(Path(__file__)),
            "decision_engine_sha256": _decision_engine_sha256(),
            "protocol_name": NAME,
            "candidate_route_class": protocol["controller_classes"]["candidate"]}


def check_frozen_inputs(identity: dict, protocol: dict, paths: dict[str, Path]) -> None:
    fresh.check_frozen_inputs(identity, paths)
    if (identity["runner_sha256"] != fresh.digest(Path(__file__))
            or identity["decision_engine_sha256"] != _decision_engine_sha256()
            or identity["decision_engine_sha256"] != DECISION_ENGINE_SHA256
            or identity["protocol_name"] != NAME
            or identity["candidate_route_class"] != protocol["controller_classes"]["candidate"]):
        raise ValueError("v2 runner, decision engine, or route changed")
    if (identity["runner_sha256"] != protocol["runner_sha256"]
            or identity["harness_sha256"] != protocol["evaluation_harness_sha256"]
            or identity["environment_sha256"] != protocol["environment_sha256"]):
        raise ValueError("bound protocol runtime hashes changed")
    validate_source_pair(protocol, paths["control"], paths["candidate"])


def _run_cold_episode(identity: dict, protocol: dict, paths: dict[str, Path],
                      arm: str, track: int, seed: int) -> dict:
    check_frozen_inputs(identity, protocol, paths)
    payload = {"source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
               "controller_class": protocol["controller_classes"][arm],
               "track_id": track, "seed": seed}
    command = [sys.executable, str(Path(fresh.__file__).resolve()),
               "--worker", json.dumps(payload, separators=(",", ":"))]
    try:
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True,
                                       capture_output=True, timeout=300, check=True)
            measured = json.loads(completed.stdout)
            if not isinstance(measured, dict):
                raise ValueError("cold worker returned a non-object result")
            return measured
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, ValueError) as error:
            stderr = getattr(error, "stderr", "") or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {"error": str(error), "stderr": stderr[-4000:]}
    finally:
        check_frozen_inputs(identity, protocol, paths)


def run_partition(root: Path, identity: dict, protocol: dict,
                  paths: dict[str, Path], phase: str, workers: int = 1) -> dict:
    previous.validate_worker_count(workers)
    require_phase(root, identity, protocol, phase)
    with fresh._run_lock(root):
        jobs = []
        for track, seed, repeat in fresh.expected_cells(protocol, phase):
            for arm in fresh.ARMS:
                prior = fresh.load_cell(root, identity, phase, arm, track, seed, repeat)
                if prior is not None:
                    if prior.get("error"):
                        raise RuntimeError(f"recorded operational failure: {phase}/{arm}/{track}/{seed}/{repeat}")
                    continue
                jobs.append((arm, track, seed, repeat))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            next_job = 0

            def fill_window() -> None:
                nonlocal next_job
                while len(pending) < workers and next_job < len(jobs):
                    arm, track, seed, _ = jobs[next_job]
                    pending[next_job] = pool.submit(_run_cold_episode, identity, protocol,
                                                    paths, arm, track, seed)
                    next_job += 1

            fill_window()
            for index, (arm, track, seed, repeat) in enumerate(jobs):
                measured = pending.pop(index).result()
                row = {"partition": phase, "arm": arm, "track_id": track,
                       "seed": seed, "repeat": repeat, **measured}
                fresh.record_cell(root, identity, row)
                if row.get("error"):
                    fresh._atomic_json(root / f"{phase}-summary.json",
                                       report_phase(root, identity, protocol, phase))
                    raise RuntimeError(f"worker failed at {phase}/{arm}/{track}/{seed}/{repeat}")
                print(json.dumps({"cell": [phase, arm, track, seed, repeat],
                                  "finished": row["finished"], "progress": row["progress"],
                                  "contacts": row["collision_count"]}), flush=True)
                fill_window()
    summary = report_phase(root, identity, protocol, phase)
    target = root / f"{phase}-summary.json"
    if target.exists() and fresh._read_json(target) != summary:
        raise ValueError(f"existing {phase} summary differs from frozen receipts")
    if not target.exists():
        fresh._atomic_json(target, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--control-agent", type=Path, default=ROOT / SOURCE_PATHS["control"])
    parser.add_argument("--candidate-agent", type=Path, default=ROOT / SOURCE_PATHS["candidate"])
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--candidate-class")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bind-source", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--partition", choices=fresh.PHASES)
    mode.add_argument("--freeze-finalist", action="store_true")
    mode.add_argument("--seal-confirmation", action="store_true")
    args = parser.parse_args()
    previous.validate_worker_count(args.workers)
    if args.protocol.resolve() != PROTOCOL_PATH.resolve() or args.output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("registered protocol and output root are required")
    for arm in fresh.ARMS:
        if getattr(args, f"{arm}_agent").resolve() != (ROOT / SOURCE_PATHS[arm]).resolve():
            raise ValueError(f"registered {arm} source path is required")
    paths = {"control": args.control_agent, "candidate": args.candidate_agent,
             "model": args.model, "protocol": args.protocol}
    if args.bind_source:
        if args.implementation_commit is None or args.candidate_class is None:
            parser.error("--bind-source needs --implementation-commit and --candidate-class")
        template = fresh._read_json(TEMPLATE_PATH)
        protocol = bind_source(template, args.implementation_commit, args.candidate_class,
                               args.model, args.protocol, args.control_agent, args.candidate_agent)
        print(json.dumps({"binding": "CREATED", "protocol": str(args.protocol),
                          "source_commit": protocol["source_construction"]["implementation_commit"]}, indent=2))
        return 0
    if args.implementation_commit is not None or args.candidate_class is not None:
        parser.error("source binding arguments require --bind-source")
    if not args.protocol.is_file():
        raise ValueError("unbound protocol: run --bind-source after the candidate source is committed")
    protocol = fresh._read_json(args.protocol)
    historical = fresh.historical_geometry_seeds(ROOT / "experiments", exclude=args.protocol)
    validate_protocol(protocol, historical)
    validate_source_pair(protocol, paths["control"], paths["candidate"])
    verify_committed_source(protocol, paths["control"])
    identity = build_identity(args.protocol, protocol, paths)
    check_frozen_inputs(identity, protocol, paths)
    if args.preflight_only:
        for arm in fresh.ARMS:
            fresh._load_agent(paths[arm], paths["model"], protocol["controller_classes"][arm])
        print(json.dumps({"preflight": "PASS", "identity": identity}, indent=2))
        return 0
    fresh.prepare_run(args.output_root, identity)
    if args.freeze_finalist:
        freeze_finalist(args.output_root, identity, protocol)
        summary = report_phase(args.output_root, identity, protocol, "screen")
    elif args.seal_confirmation:
        seal_confirmation(args.output_root, identity, protocol)
        summary = report_phase(args.output_root, identity, protocol, "confirmation")
    else:
        summary = run_partition(args.output_root, identity, protocol, paths,
                                args.partition, workers=args.workers)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
