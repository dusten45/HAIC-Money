"""Run a fixed four-run steering-L2 or augmentation-pad comparison without ML imports."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parent
TRAIN_TIMEOUT_SECONDS = 12 * 60 * 60
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
)}
TRAIN_SOURCES = (
    "train_drqv2.py", "drq_v2.py", "common_adapter.py", "train.py", "env_wrapper.py", "damage.py",
    "training/drq_demonstrations.py", "training/vision_teacher.py", "haic_agent/corridor_agent.py",
)
SOURCES = (*TRAIN_SOURCES, "run_drqv2_matched.py", "evaluate_policy.py", "agent.py",
           "tracking.py", "action_smoothing.py", "action_representation.py", "requirements.txt")
AXES = {"steering_logit_l2": {"control": 0.0, "steering_l2": 0.001},
        "augmentation_pad": {"control": 4, "augmentation_pad1": 1}}
SEEDS = [0, 1]
LIMITS = {"process_initialization_seconds": 10, "agent_reset_seconds": 5,
          "max_action_seconds": 5, "peak_rss_bytes": 1024**3}
SIGNATURE = ("status", "finished", "termination_class", "steps", "lap_time_ms", "progress",
             "damage", "collision_actions", "action_trace_sha256", "steering_delta_abs_mean",
             "action_smoothing_fingerprint", "action_control_fingerprint",
             "action_representation_fingerprint", "terminated", "truncated", "retire_reason",
             "loaded_archive_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def read_json(path):
    return json.loads(Path(path).read_bytes(), parse_constant=reject_constant)


def write_sealed(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with path.with_suffix(path.suffix + ".sha256").open("x") as handle:
        handle.write(sha256(path) + "\n")


def read_sealed(path):
    path = Path(path)
    require(sha256(path) == path.with_suffix(path.suffix + ".sha256").read_text().strip(),
            f"mutated sealed result: {path}")
    return read_json(path)


def file_hashes(paths):
    return {str(Path(path).absolute()): sha256(path) for path in paths}


def verify_files(files):
    for path, expected in files.items():
        require(sha256(path) == expected, f"mutated artifact: {path}")


def explicit_path(path):
    path = Path(os.path.abspath(path))
    require("_latest" not in path.parts and "_latest" not in path.resolve().parts,
            "explicit paths required; _latest is forbidden")
    return path


def reserved_seeds(protocol):
    return sorted(set(protocol.get("reserved_training_seeds", [])).union(
        *(matrix["seeds"] for matrix in protocol["partitions"].values())))


def matched_axis(protocol):
    matched = protocol["matched_training"]
    parameter = matched.get("arm_parameter", "steering_logit_l2")
    require(isinstance(parameter, str) and parameter in AXES, "unsupported matched arm_parameter")
    arms = AXES[parameter]
    value_types = (int,) if parameter == "augmentation_pad" else (int, float)
    require(matched.get("arms") == arms and all(type(value) in value_types for value in matched["arms"].values()),
            "fixed arms required; no extra arms or sweeps")
    unchanged = matched["unchanged_drq_config"]
    require(parameter not in unchanged, "active arm_parameter must not appear in unchanged_drq_config")
    fixed, expected = ("steering_logit_l2", 0.0) if parameter == "augmentation_pad" else ("augmentation_pad", 4)
    fixed_types = (int, float) if fixed == "steering_logit_l2" else (int,)
    require(type(unchanged.get(fixed)) in fixed_types and unchanged[fixed] == expected,
            f"inactive {fixed} must remain {expected}; compound changes are forbidden")
    return parameter, arms


def validate_protocol(protocol):
    require(type(protocol.get("frame_skip")) is int and protocol["frame_skip"] == 4
            and type(protocol.get("max_steps")) is int and protocol["max_steps"] > 0,
            "matched protocol requires frame_skip=4 and positive max_steps")
    require(protocol.get("training_track_ids") == [1, 2, 3, 4]
            and all(type(track) is int for track in protocol["training_track_ids"]), "wrong training track pool")
    matched = protocol["matched_training"]
    _, arms = matched_axis(protocol)
    require(matched.get("seeds") == SEEDS and all(type(seed) is int for seed in matched["seeds"]),
            "fixed training seeds required; no sweeps")
    for key in ("total_steps", "eval_freq", "batch_size", "warmup_steps", "replay_capacity", "updates_per_step"):
        require(type(matched[key]) is int and matched[key] > 0, f"invalid matched_training.{key}")
    require(type(matched["track_sampler_seed"]) is int and matched["track_sampler_seed"] >= 0,
            "invalid track sampler seed")
    require(matched["replay_capacity"] >= max(matched["batch_size"], matched["warmup_steps"]), "insufficient replay capacity")
    n_step = matched["unchanged_drq_config"]["n_step"]
    require(type(n_step) is int and n_step > 0
            and matched["warmup_steps"] >= matched["batch_size"] + n_step
            and matched["total_steps"] >= matched["warmup_steps"],
            "warmup must supply a complete n-step batch before fixed-budget updates")
    steps = sorted(set(range(matched["eval_freq"], matched["total_steps"] + 1, matched["eval_freq"])) | {matched["total_steps"]})
    require(matched["checkpoint_steps"] == steps, "checkpoint schedule disagrees with budget/eval_freq")
    require(set(protocol["partitions"]) == {"screen", "confirmation", "blind"},
            "three explicit partitions required")
    seen = set()
    for matrix in protocol["partitions"].values():
        for key, low, high in (("seeds", 0, 2**32), ("track_ids", 1, 2**32)):
            values = matrix[key]
            require(isinstance(values, list) and bool(values)
                    and all(type(value) is int and low <= value < high for value in values)
                    and len(set(values)) == len(values), f"invalid partition {key}")
        require(type(matrix["repeats"]) is int and matrix["repeats"] >= 2,
                "independent reload repeats required")
        require(not seen.intersection(matrix["seeds"]), "partition seed overlap")
        seen.update(matrix["seeds"])
    extra = protocol.get("reserved_training_seeds", [])
    require(isinstance(extra, list) and all(type(seed) is int and 0 <= seed < 2**32 for seed in extra)
            and len(set(extra)) == len(extra), "invalid reserved_training_seeds")
    gate = protocol["gates"]["confirmation_comparison"]
    require(gate["required_training_seeds"] == [0, 1]
            and gate["min_treatment_finishes_each_seed"] == 1
            and gate["min_paired_finish_delta_each_seed"] == 1, "unexpected confirmation criterion")
    require(protocol["frozen_selection"]["finalist_arm"] == next(arm for arm in arms if arm != "control"),
            "wrong finalist arm")


def child_environment():
    environment = dict(os.environ, **THREAD_ENV)
    for key in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(key, None)
    # In particular, do not change CUDA_VISIBLE_DEVICES in the GPU trainer.
    return environment


def training_jobs(root, protocol, execution):
    jobs = []
    matched = protocol["matched_training"]
    parameter, arms = matched_axis(protocol)
    for arm in arms:
        for seed in SEEDS:
            name = f"{arm}-seed{seed}"
            run = root / name
            options = {
                "name": name, "run-dir": run, "protocol-file": root / "protocol.json",
                "track-ids": ",".join(map(str, protocol["training_track_ids"])),
                "max-steps": protocol["max_steps"], "frame-skip": protocol["frame_skip"],
                "seed": seed, "steering-logit-l2": arms[arm] if parameter == "steering_logit_l2" else 0.0,
                "device": execution["device"], "eval-python": execution["eval_python"],
                "eval-workers": execution["eval_workers"], "evaluations-dir": run / "evaluations",
                "torch-threads": 1,
            }
            options.update({key.replace("_", "-"): matched[key] for key in (
                "total_steps", "eval_freq", "batch_size", "warmup_steps", "replay_capacity",
                "updates_per_step", "track_sampler_seed",
            )})
            # Preserve the exact commands already sealed by existing L2 studies.
            if parameter == "augmentation_pad":
                options["augmentation-pad"] = arms[arm]
            command = [execution["train_python"], "-B", str(root / "source/train_drqv2.py")]
            for key, value in options.items():
                command.extend([f"--{key}", str(value)])
            jobs.append({"name": name, "arm": arm, "seed": seed, "run_dir": str(run),
                         "command": command, "log": str(root / "logs" / f"{name}.log")})
    return jobs


def initialize(args):
    root = explicit_path(args.run_root).resolve()
    protocol_path = explicit_path(args.protocol_file)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes, parse_constant=reject_constant)
    validate_protocol(protocol)
    require(1 <= args.jobs <= 4 and args.eval_workers >= 1 and args.jobs * args.eval_workers <= 8,
            "require 1..4 jobs and at most eight concurrent CPU evaluation workers")
    require(args.device == "cuda" or args.device.startswith("cuda:"), "GPU training device required")
    execution = {"train_python": str(explicit_path(args.train_python)),
                 "eval_python": str(explicit_path(args.eval_python)), "device": args.device,
                 "jobs": args.jobs, "eval_workers": args.eval_workers,
                 "train_timeout_seconds": TRAIN_TIMEOUT_SECONDS, "thread_environment": THREAD_ENV}
    if args.evaluate_only:
        manifest = read_sealed(root / "manifest.json")
        require(manifest["execution"] == execution and manifest["run_root"] == str(root),
                "recovery execution arguments differ from frozen manifest")
        require(manifest["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest(),
                "protocol changed since launch")
        require(manifest["jobs"] == training_jobs(root, protocol, execution), "mutated frozen commands")
        verify_files(manifest["files"])
        return root, protocol, manifest
    root.mkdir(parents=False, exist_ok=False)
    (root / "source").mkdir()
    (root / "logs").mkdir()
    (root / "jobs").mkdir()
    with (root / "protocol.json").open("xb") as handle:
        handle.write(protocol_bytes)
    sources = [Path(name) for name in SOURCES] + sorted(path.relative_to(ROOT) for path in (ROOT / "core").rglob("*.py"))
    source_hashes = {}
    for relative in sources:
        data = (ROOT / relative).read_bytes()
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        source_hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    # Detect edits during the snapshot; jobs subsequently execute only these copies.
    require(all(sha256(ROOT / name) == digest for name, digest in source_hashes.items()),
            "source changed while freezing")
    manifest = {"format": "haic-drq-matched-v1", "run_root": str(root),
                "protocol_source": str(protocol_path), "protocol_sha256": sha256(root / "protocol.json"),
                "source_sha256": source_hashes, "execution": execution,
                "cuda_environment": {key: value for key, value in os.environ.items()
                                     if key.startswith(("CUDA", "CUBLAS"))},
                "jobs": training_jobs(root, protocol, execution),
                "evaluation_command_rule": "Freeze exact selected-actor commands in frozen_candidates.json before any confirmation.",
                "files": file_hashes([root / "protocol.json", *(root / "source" / name for name in source_hashes)])}
    write_sealed(root / "manifest.json", manifest)
    return root, protocol, manifest


def run_process(command, log, timeout, cwd):
    try:
        with Path(log).open("xb") as output:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                       cwd=cwd, env=child_environment(), timeout=timeout, check=False)
        return {"status": "ok" if completed.returncode == 0 else "error",
                "returncode": completed.returncode, "command": command, "log": str(log)}
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}",
                "command": command, "log": str(log)}


def train_all(root, manifest):
    def train(job):
        status = run_process(job["command"], job["log"], manifest["execution"]["train_timeout_seconds"], root / "source")
        write_sealed(root / "jobs" / f"{job['name']}.json", status)
        return {"name": job["name"], **status}

    with ThreadPoolExecutor(max_workers=manifest["execution"]["jobs"]) as pool:
        # Each process returns a status rather than cancelling the other complete-budget runs.
        return list(pool.map(train, manifest["jobs"]))


def validate_cpu_runtime(runtime, sources):
    packages = runtime["packages"]
    require(runtime["python_version"] == [3, 11] and runtime["sys_platform"] == "linux"
            and packages["torch"].split("+")[0] == "2.1.0"
            and packages["numpy"] == "1.26.0" and packages["gymnasium"] == "0.29.1"
            and packages["opencv-python"] == "4.8.1.78"
            and runtime["torch_cuda"] is None and runtime["cuda_available"] is False
            and runtime["torch_threads"] == runtime["torch_interop_threads"] == 1,
            "evaluation did not use pinned official CPU runtime")
    require(bool(runtime["runtime_sources"]) and all(sources.get(name) == digest
            for name, digest in runtime["runtime_sources"].items()), "evaluation runtime source mismatch")


def selection_score(result):
    require(result.get("eligible") is True and result.get("determinism_audited") is True
            and result.get("cpu_reload_matches") is True and result.get("operational_failures") == 0,
            "CPU operational/reload failure")
    metrics = result["summary"]
    finish, progress, lap = (metrics[key] for key in ("finish_rate", "avg_progress", "avg_lap_time_ms"))
    require(type(finish) in (int, float) and math.isfinite(finish) and 0 <= finish <= 1
            and type(progress) in (int, float) and math.isfinite(progress)
            and (lap is None or type(lap) in (int, float) and math.isfinite(lap) and lap >= 0),
            "invalid CPU selection metrics")
    return finish, progress, -lap if lap is not None else -math.inf


def evaluation_evidence(pointer_path, candidate, partition, protocol, manifest, diagnostic=False, previous=None):
    pointer_path = Path(pointer_path)
    pointer = read_json(pointer_path)
    directory = Path(pointer["evaluation_dir"])
    run = Path(candidate["run_dir"])
    require(directory.is_absolute() and directory.resolve().is_relative_to((run / "evaluations").resolve())
            and not directory.name.startswith(".pending"), "evaluation directory is not published run-local evidence")
    summary = read_json(directory / "summary.json")
    receipt = read_json(directory / "manifest.json")
    require(pointer["ranked"] == summary and len(summary) == 1, "pointer/immutable summary mismatch")
    result = summary[0]
    for record in (pointer, receipt):
        require(record["partition"] == partition and record["protocol_sha256"] == manifest["protocol_sha256"],
                "evaluation partition/protocol mismatch")
    require(all(record.get("diagnostic_only", False) is diagnostic for record in (pointer, receipt, result)),
            "diagnostic receipt mismatch")
    require(receipt["cell_matrix"] == protocol["partitions"][partition]
            and read_json(directory / "protocol.json") == receipt["cell_matrix"]
            and receipt["max_steps"] == protocol["max_steps"] and receipt["frame_skip"] == protocol["frame_skip"]
            and receipt["run_dir"] == str(run), "evaluation matrix/run mismatch")
    require(sha256(directory / "protocol_spec.json") == manifest["protocol_sha256"], "mutated evaluation protocol")
    require(result["archive_sha256"] == candidate["actor_sha256"]
            and result["source_path"] == candidate["actor"] and result["algorithm"] == "drq-v2"
            and result["run_config_sha256"] == sha256(run / "config.json"), "evaluation actor/config mismatch")
    exported = result.get("export_metadata", {})
    require(exported.get("format") == "haic-drq-v2-actor-v1"
            and exported.get("config") == read_json(run / "config.json")["config"]["drq_config"],
            "exported actor configuration/format mismatch")
    for relative, expected in ((result["evaluation_archive_path"], candidate["actor_sha256"]),
                               (result["evaluation_run_config_path"], result["run_config_sha256"])):
        target = directory / relative
        require(target.resolve().is_relative_to(directory.resolve()) and sha256(target) == expected,
                "immutable evaluation snapshot mismatch")
    selection_score(result)
    matrix = protocol["partitions"][partition]
    cells = {(track, seed, repeat) for track in matrix["track_ids"] for seed in matrix["seeds"]
             for repeat in range(matrix["repeats"])}
    count = len(matrix["track_ids"]) * len(matrix["seeds"])
    require(result["expected_cells"] == result["canonical_episodes"] == result["summary"]["n_episodes"] == count
            and result["expected_results"] == len(cells), "incomplete canonical evaluation")
    require(bool(receipt["worker_runtime"]), "missing worker runtime")
    for runtime in receipt["worker_runtime"]:
        validate_cpu_runtime(runtime, manifest["source_sha256"])
    observed, outcomes, signatures, progress, laps = set(), {}, {}, [], []
    with (directory / "episodes.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line, parse_constant=reject_constant)
            key = row["track_id"], row["seed"], row["repeat"]
            require(key in cells and key not in observed and row["candidate_id"] == result["candidate_id"],
                    "missing/duplicate/unexpected evaluation cell")
            observed.add(key)
            require(row["status"] == "ok" and row["loaded_archive_sha256"] == candidate["actor_sha256"]
                    and row["runtime"] in receipt["worker_runtime"], "invalid CPU cell")
            for field, limit in LIMITS.items():
                value = row[field]
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit,
                        f"CPU resource failure: {field}")
            signature = [row.get(field) for field in SIGNATURE]
            cell = key[:2]
            require(cell not in signatures or signatures[cell] == signature, "CPU reload outcome/trace mismatch")
            signatures[cell] = signature
            require(type(row["finished"]) is bool and math.isfinite(row["progress"]), "invalid cell outcome")
            if row["repeat"] == 0:
                outcomes[f"{key[0]}:{key[1]}"] = row["finished"]
                progress.append(row["progress"])
                if row["finished"]:
                    require(type(row["lap_time_ms"]) in (int, float) and math.isfinite(row["lap_time_ms"]),
                            "finish without completed lap time")
                    laps.append(row["lap_time_ms"])
    require(observed == cells, "partial evaluation cell matrix")
    audits = read_json(directory / "determinism.json")
    require(len(audits) == count and {(row["track_id"], row["seed"]) for row in audits} == set(signatures)
            and all(row["audited"] is True and row["matches_canonical"] is True
                    and row["repeats"] == matrix["repeats"] and row["candidate_id"] == result["candidate_id"]
                    for row in audits), "incomplete deterministic reload audit")
    expected_metrics = {"finish_rate": sum(outcomes.values()) / count, "avg_progress": sum(progress) / count,
                        "avg_lap_time_ms": sum(laps) / len(laps) if laps else None}
    for key, expected in expected_metrics.items():
        actual = result["summary"][key]
        require(actual is None if expected is None else actual is not None and math.isclose(actual, expected, abs_tol=1e-12),
                f"canonical summary mismatch: {key}")
    if previous is not None:
        prior = receipt["previous_evaluation"]
        require(prior["actor_sha256"] == candidate["actor_sha256"]
                and prior["partition"] == ("screen" if partition == "confirmation" else "confirmation")
                and prior["evaluation_dir"] == read_json(previous)["evaluation_dir"],
                "wrong preceding evaluation lineage")
        provenance = prior["files"]["previous_evaluation.json"]
        require(provenance["source_path"] == str(previous) and provenance["sha256"] == sha256(previous),
                "previous evaluation pointer hash mismatch")
        for filename, record in prior["files"].items():
            require(sha256(directory / filename) == record["sha256"] == sha256(record["source_path"]),
                    "mutated preceding evaluation receipt")
    files = file_hashes([pointer_path, *(path for path in directory.rglob("*") if path.is_file())])
    return {"pointer": str(pointer_path), "files": files, "result": result,
            "finishes": sum(outcomes.values()), "outcomes": outcomes, "diagnostic_only": diagnostic}


def episode_schedule(path, protocol):
    schedule, ended, global_step = [], True, 0
    reserved = set(reserved_seeds(protocol))
    budget = protocol["matched_training"]["total_steps"]
    with Path(path).open() as handle:
        for line in handle:
            event = json.loads(line, parse_constant=reject_constant)
            track, seed = event["track_id"], event["seed"]
            require(type(seed) is int and 0 <= seed < 2**32 and seed not in reserved,
                    "training reserved seed leak or invalid seed")
            require(type(track) is int and track in protocol["training_track_ids"], "unexpected sampled track")
            require(type(event["episode_id"]) is int, "invalid episode index")
            if event["event"] == "reset":
                require(ended and event["episode_id"] == len(schedule), "non-contiguous episode-index schedule")
                schedule.append((track, seed))
                ended = False
            else:
                require(event["event"] == "end" and not ended and event["episode_id"] == len(schedule) - 1
                        and (track, seed) == schedule[-1], "episode end/reset lineage mismatch")
                require(type(event["steps"]) is int and 0 < event["steps"] <= protocol["max_steps"]
                        and event["global_step"] == global_step + event["steps"] <= budget,
                        "invalid episode step budget")
                global_step = event["global_step"]
                ended = True
    require(bool(schedule) and not ended and 0 <= budget - global_step <= protocol["max_steps"],
            "partial training episode log")
    return schedule


def validate_runs(root, protocol, manifest):
    candidates, schedules, files, baseline = [], {}, {}, None
    matched = protocol["matched_training"]
    parameter, arms = matched_axis(protocol)
    source_map = {name: digest for name, digest in manifest["source_sha256"].items()
                  if name in TRAIN_SOURCES or name.startswith("core/")}
    for job in manifest["jobs"]:
        run = Path(job["run_dir"])
        wrapper, result = read_json(run / "config.json"), read_json(run / "result.json")
        config = wrapper["config"]
        require(shlex.split(wrapper["command_line"]) == job["command"][2:], "training command mismatch")
        expected = {"algorithm": "drq-v2", "seed": job["seed"], "track_ids": protocol["training_track_ids"],
                    "protocol": protocol, "max_steps": protocol["max_steps"], "frame_skip": 4,
                    "excluded_training_seeds": reserved_seeds(protocol), "resume_from": None, "resume_sha256": None,
                    "training_source_sha256": source_map, "observation_channels": 4,
                    "training_track_mode": "sampled", "training_obstacles": "official",
                    "reward_contract": {"reward_shaping": False, "norm_reward": False, "collision_penalty": 0.0},
                    "eval_python": manifest["execution"]["eval_python"], "eval_workers": manifest["execution"]["eval_workers"]}
        expected.update({key: matched[key] for key in ("total_steps", "eval_freq", "updates_per_step", "track_sampler_seed")})
        for key, value in expected.items():
            require(key in config and config[key] == value, f"{job['name']} config mismatch: {key}")
        expected_drq = {**protocol["matched_training"]["unchanged_drq_config"],
                        **{key: matched[key] for key in ("batch_size", "warmup_steps", "replay_capacity")},
                        "device": manifest["execution"]["device"], parameter: arms[job["arm"]]}
        require(config["drq_config"] == expected_drq, f"{job['name']} drq_config mismatch")
        require(config["action_smoothing"]["method"] == "none" and config["evaluation_runtime"]["status"] == "ok",
                "smoothing or invalid CPU preflight")
        validate_cpu_runtime(config["evaluation_runtime"]["runtime"], manifest["source_sha256"])
        require(config["runtime"]["torch_threads"] == 1 and config["runtime"]["deterministic_algorithms"] is True
                and config["runtime"]["cudnn_deterministic"] is True and config["runtime"]["cudnn_benchmark"] is False,
                "nondeterministic training runtime")
        normalized = copy.deepcopy(config)
        del normalized["seed"], normalized["drq_config"][parameter]
        require(baseline is None or baseline == normalized, "saved run configs differ beyond seed/active parameter")
        baseline = normalized
        require(type(result["environment_steps"]) is int and result["environment_steps"] == matched["total_steps"],
                "incomplete or exceeded exact training budget")
        expected_updates = (matched["total_steps"] - matched["warmup_steps"] + 1) * matched["updates_per_step"]
        require(type(result["gradient_steps"]) is int and result["gradient_steps"] == expected_updates,
                "incomplete or exceeded learner update budget")
        require(sha256(run / "protocol.json") == manifest["protocol_sha256"], "run protocol byte mismatch")
        schedules[job["name"]] = episode_schedule(run / "episodes.jsonl", protocol)
        with (run / "metrics.jsonl").open() as handle:
            metrics = [json.loads(line, parse_constant=reject_constant) for line in handle]
        training_metrics = [row for row in metrics if "replay_bytes" in row]
        expected_metric_steps = sorted(set(range(1000, matched["total_steps"] + 1, 1000)) | {matched["total_steps"]})
        require([row["step"] for row in training_metrics] == expected_metric_steps, "missing/partial training metrics")
        for row in training_metrics:
            updates = max(0, row["step"] - matched["warmup_steps"] + 1) * matched["updates_per_step"]
            require(type(row.get("gradient_steps")) in (int, float) and row["gradient_steps"] == updates,
                    "training metrics learner update budget mismatch")
            require(type(row.get("replay_size")) in (int, float)
                    and row["replay_size"] == min(row["step"], matched["replay_capacity"]), "training replay occupancy mismatch")
        files.update(file_hashes(run / name for name in ("config.json", "result.json", "selection.json", "episodes.jsonl", "metrics.jsonl", "protocol.json")))
        expected_dirs = [f"step-{step:09d}" for step in matched["checkpoint_steps"]]
        require(sorted(path.name for path in (run / "checkpoints").iterdir()) == expected_dirs,
                "partial or unexpected checkpoint budget")
        best, best_evidence = None, None
        for step in matched["checkpoint_steps"]:
            directory = run / "checkpoints" / f"step-{step:09d}"
            selection = read_json(directory / "selection.json")
            candidate = {"name": job["name"], "arm": job["arm"], "seed": job["seed"], "run_dir": str(run),
                         "step": step, "actor": str(directory / "actor.pt"), "actor_sha256": sha256(directory / "actor.pt")}
            require(selection["step"] == step and selection["actor"] == candidate["actor"]
                    and selection["actor_sha256"] == candidate["actor_sha256"]
                    and selection["checkpoint"] == str(directory / "checkpoint.pt")
                    and selection["checkpoint_sha256"] == sha256(directory / "checkpoint.pt")
                    and selection["protocol_sha256"] == manifest["protocol_sha256"], "checkpoint selection identity mismatch")
            updates = max(0, step - matched["warmup_steps"] + 1) * matched["updates_per_step"]
            require(type(selection.get("gradient_steps")) is int and selection["gradient_steps"] == updates
                    and type(selection.get("replay_size")) is int
                    and selection["replay_size"] == min(step, matched["replay_capacity"]), "checkpoint learner/replay budget mismatch")
            checkpoint_manifest = read_json(directory / "checkpoint.manifest.json")
            require(checkpoint_manifest["extra"]["environment_steps"] == step
                    and all(checkpoint_manifest["extra"][key] == value for key, value in config.items()),
                    "checkpoint manifest config/budget mismatch")
            parity = selection["parity"]
            require(all(type(parity[key]) in (int, float) and 0 <= parity[key] <= 1e-6
                        for key in ("restore_max_abs_error", "cpu_export_max_abs_error")) and parity["resets"] >= 2,
                    "failed checkpoint restore/CPU export parity")
            evidence = evaluation_evidence(directory / "evaluation.json", candidate, "screen", protocol, manifest)
            require(evidence["result"] == selection["cpu_result"]
                    and read_json(directory / "evaluation.json")["evaluation_dir"] == selection["evaluation_dir"],
                    "selection does not match immutable CPU evaluation")
            selected = best is None or selection_score(selection["cpu_result"]) > selection_score(best["cpu_result"])
            require(selection["selected"] is selected, "unstable checkpoint selection/tie")
            if selected:
                best, best_evidence = selection, {**candidate, "screen": evidence}
            files.update(evidence["files"])
            files.update(file_hashes(directory / name for name in ("selection.json", "actor.pt", "checkpoint.pt", "checkpoint.manifest.json")))
        require(best == result["selected_checkpoint"] == read_json(run / "selection.json"),
                "run selected actor differs from screen ranking")
        candidates.append(best_evidence)
    longest = max(schedules.values(), key=len)
    require(all(schedule == longest[:len(schedule)] for schedule in schedules.values()),
            "sampled episode-index prefixes differ across arms or seeds")
    return candidates, {name: len(schedule) for name, schedule in schedules.items()}, files


def evaluation_command(root, candidate, partition, protocol, execution):
    run = Path(candidate["run_dir"])
    previous = Path(candidate["actor"]).parent / "evaluation.json" if partition == "confirmation" else run / "confirmation.json"
    command = [execution["eval_python"], "-B", str(root / "source/evaluate_policy.py"),
               "--model", candidate["actor"], "--run-dir", str(run), "--protocol-file", str(root / "protocol.json"),
               "--partition", partition, "--previous-evaluation", str(previous),
               "--output", str(run / f"{partition}.json"), "--evaluations-dir", str(run / "evaluations"),
               "--max-steps", str(protocol["max_steps"]), "--frame-skip", str(protocol["frame_skip"]),
               "--workers", str(execution["eval_workers"])]
    if partition == "confirmation" and candidate["screen"]["finishes"] == 0:
        command.append("--diagnostic-confirmation")
    return command


def published_evaluations(run, partition):
    return sorted(path.parent for path in (run / "evaluations").glob("*/manifest.json")
                  if read_json(path).get("partition") == partition)


def freeze_candidates(root, protocol, manifest):
    path = root / "frozen_candidates.json"
    if path.exists():
        frozen = read_sealed(path)
        verify_files(frozen["files"])
    candidates, lengths, files = validate_runs(root, protocol, manifest)
    finalist = max((candidate for candidate in candidates if candidate["arm"] == protocol["frozen_selection"]["finalist_arm"]),
                   key=lambda candidate: (*selection_score(candidate["screen"]["result"]), -candidate["seed"]))
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["actor_sha256"], []).append(candidate["name"])
    expected = {"protocol_sha256": manifest["protocol_sha256"], "manifest_sha256": sha256(root / "manifest.json"),
                "candidates": candidates, "finalist": finalist["name"], "episode_prefix_lengths": lengths,
                "actor_lineage_groups": groups, "training_runs": len(candidates), "paired_training_seeds": SEEDS,
                "arm_counts": {arm: sum(candidate["arm"] == arm for candidate in candidates) for arm in protocol["matched_training"]["arms"]},
                "unique_selected_actors": len(groups),
                "lineage_note": "Every arm/seed requires its own complete training lineage. Equal actor hashes and reloads are not independent policy evidence.",
                "commands": {candidate["name"]: evaluation_command(root, candidate, "confirmation", protocol, manifest["execution"])
                             for candidate in candidates},
                "blind_command": evaluation_command(root, finalist, "blind", protocol, manifest["execution"]),
                "files": files}
    if path.exists():
        require(frozen == expected, "mutated frozen candidates or selection")
    else:
        for candidate in candidates:
            run = Path(candidate["run_dir"])
            require(not list(run.glob("confirmation*.json")) and not list(run.glob("blind*.json"))
                    and not published_evaluations(run, "confirmation") and not published_evaluations(run, "blind"),
                    "holdout exists before candidate freeze")
        write_sealed(path, expected)
    return expected


def evaluate_fixed(root, candidate, partition, command, protocol, manifest):
    run = Path(candidate["run_dir"])
    output, receipt_path, request_path = (run / f"{partition}{suffix}.json" for suffix in ("", ".receipt", ".request"))
    diagnostic = partition == "confirmation" and candidate["screen"]["finishes"] == 0
    previous = Path(command[command.index("--previous-evaluation") + 1])
    try:
        if receipt_path.exists():
            receipt = read_sealed(receipt_path)
            verify_files(receipt["files"])
            require(published_evaluations(run, partition) == [Path(read_json(output)["evaluation_dir"])],
                    "unexpected additional published evaluation")
            require(receipt["command"] == command, "reused evaluation command mismatch")
            request = read_sealed(request_path)
            require(request["frozen_candidates_sha256"] == sha256(root / "frozen_candidates.json")
                    and request["previous_pointer_sha256"] == sha256(previous), "reused evaluation lineage mismatch")
            evidence = evaluation_evidence(output, candidate, partition, protocol, manifest, diagnostic, previous)
            require(receipt["evidence"] == evidence, "mutated published evaluation")
            return receipt
        require(not output.exists() and not request_path.exists(),
                "evaluation already started/published without a sealed receipt; refusing to rerun fresh cells")
        require(not published_evaluations(run, partition),
                "evaluation already published without its pointer; refusing to rerun fresh cells")
        verify_files(candidate["screen"]["files"])
        require(sha256(candidate["actor"]) == candidate["actor_sha256"], "mutated frozen actor")
        write_sealed(request_path, {"command": command, "frozen_candidates_sha256": sha256(root / "frozen_candidates.json"),
                                    "previous_pointer_sha256": sha256(previous)})
        matrix = protocol["partitions"][partition]
        timeout = 120 + 300 * math.ceil(len(matrix["track_ids"]) * len(matrix["seeds"]) * matrix["repeats"]
                                       / manifest["execution"]["eval_workers"])
        status = run_process(command, root / "logs" / f"{candidate['name']}-{partition}.log", timeout, root / "source")
        require(status["status"] == "ok", f"evaluation subprocess failed: {status}")
        evidence = evaluation_evidence(output, candidate, partition, protocol, manifest, diagnostic, previous)
        receipt = {"status": "ok", "command": command, "evidence": evidence,
                   "files": {**evidence["files"], **file_hashes([request_path, request_path.with_suffix(".json.sha256")])}}
        write_sealed(receipt_path, receipt)
        return receipt
    except Exception as error:
        return {"status": "error", "command": command, "error": f"{type(error).__name__}: {error}"}


def confirmation_gate(candidates, confirmations, protocol):
    pairs = []
    for seed in SEEDS:
        control, treatment = (next(candidate for candidate in candidates if candidate["arm"] == arm and candidate["seed"] == seed)
                              for arm in ("control", protocol["frozen_selection"]["finalist_arm"]))
        reports = [confirmations[candidate["name"]] for candidate in (control, treatment)]
        if any(report["status"] != "ok" for report in reports):
            pairs.append({"seed": seed, "passed": False, "reason": "missing/invalid confirmation"})
            continue
        left, right = (report["evidence"] for report in reports)
        delta = right["finishes"] - left["finishes"]
        passed = treatment["screen"]["finishes"] > 0 and right["finishes"] > 0 and delta > 0 and not right["diagnostic_only"]
        pairs.append({"seed": seed, "control_finishes": left["finishes"], "treatment_finishes": right["finishes"],
                      "finish_delta": delta, "treatment_screen_finishes": treatment["screen"]["finishes"],
                      "paired_wins": sum(right["outcomes"][key] and not value for key, value in left["outcomes"].items()),
                      "paired_losses": sum(value and not right["outcomes"][key] for key, value in left["outcomes"].items()),
                      "passed": passed})
    return all(pair["passed"] for pair in pairs), pairs


def save_result(root, result):
    path = root / "result.json"
    index = 1
    while path.exists():
        path = root / f"result-recovery-{index:03d}.json"
        index += 1
    write_sealed(path, result)
    return path, result


def run_study(args):
    result = {"status": "error", "promotion": False, "next_stage_permitted": False,
              "official_submission_authorized": False,
              "confirmation_gate_passed": False, "blind": {"status": "not_run"}, "errors": [],
              "scope": "Positive results only permit consideration of a separately predeclared next stage, not automatic scale-up or official submission."}
    try:
        root, protocol, manifest = initialize(args)
    except Exception as error:
        root = explicit_path(args.run_root).resolve()
        if not args.evaluate_only or not (root / "manifest.json").is_file():
            raise
        # A refused recovery is evidence too; never replace a prior published verdict.
        with (root / "manifest.json").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result["errors"].append(f"{type(error).__name__}: {error}")
            return save_result(root, result)
    result["protocol_sha256"] = manifest["protocol_sha256"]
    with (root / "manifest.json").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for previous in sorted(root.glob("result*.json")):
                read_sealed(previous)
            if not args.evaluate_only:
                result["training_jobs"] = train_all(root, manifest)
                require(all(job["status"] == "ok" for job in result["training_jobs"]), "one or more training jobs failed")
            else:
                for job in manifest["jobs"]:
                    status_path = root / "jobs" / f"{job['name']}.json"
                    if status_path.exists():
                        status = read_sealed(status_path)
                        require(status["status"] == "ok", f"training job failed: {job['name']}")
            verify_files(manifest["files"])
            frozen = freeze_candidates(root, protocol, manifest)
            result["frozen_candidates_sha256"] = sha256(root / "frozen_candidates.json")
            result["finalist"] = frozen["finalist"]
            result["actor_lineage_groups"] = frozen["actor_lineage_groups"]
            def confirm(candidate):
                return evaluate_fixed(root, candidate, "confirmation", frozen["commands"][candidate["name"]], protocol, manifest)
            with ThreadPoolExecutor(max_workers=manifest["execution"]["jobs"]) as pool:
                confirmations = dict(zip((candidate["name"] for candidate in frozen["candidates"]), pool.map(confirm, frozen["candidates"])))
            result["confirmations"] = {name: {key: value for key, value in report.items() if key != "files" and key != "evidence"}
                                       | ({"pointer": report["evidence"]["pointer"], "finishes": report["evidence"]["finishes"],
                                           "diagnostic_only": report["evidence"]["diagnostic_only"]} if report["status"] == "ok" else {})
                                        for name, report in confirmations.items()}
            for report in confirmations.values():
                if report["status"] == "ok":
                    verify_files(report["files"])
            verify_files(manifest["files"])
            result["confirmation_gate_passed"], result["paired_comparison"] = confirmation_gate(frozen["candidates"], confirmations, protocol)
            result["errors"].extend(report["error"] for report in confirmations.values() if report["status"] != "ok")
            if result["confirmation_gate_passed"]:
                gate_path = root / "confirmation_gate.json"
                gate = {"frozen_candidates_sha256": result["frozen_candidates_sha256"],
                        "paired_comparison": result["paired_comparison"], "passed": True,
                        "confirmation_receipts": file_hashes(Path(candidate["run_dir"]) / "confirmation.receipt.json"
                                                              for candidate in frozen["candidates"])}
                if gate_path.exists():
                    require(read_sealed(gate_path) == gate, "mutated pre-blind confirmation gate")
                else:
                    write_sealed(gate_path, gate)
                # This identity was chosen before confirmation; no ranking is performed here.
                finalist = next(candidate for candidate in frozen["candidates"] if candidate["name"] == frozen["finalist"])
                blind = evaluate_fixed(root, finalist, "blind", frozen["blind_command"], protocol, manifest)
                result["blind"] = {"status": blind["status"]}
                if blind["status"] == "ok":
                    result["blind"].update(pointer=blind["evidence"]["pointer"], finishes=blind["evidence"]["finishes"])
                    result["promotion"] = blind["evidence"]["finishes"] > 0 and not blind["evidence"]["diagnostic_only"]
                else:
                    result["errors"].append(blind["error"])
            verify_files(manifest["files"])
            verify_files(frozen["files"])
            for report in confirmations.values():
                if report["status"] == "ok":
                    verify_files(report["files"])
            if result["confirmation_gate_passed"] and blind["status"] == "ok":
                verify_files(blind["files"])
            result["status"] = "error" if result["errors"] else "passed" if result["promotion"] else "rejected"
            result["next_stage_permitted"] = result["promotion"]
        except Exception as error:
            result.update(status="error", promotion=False, next_stage_permitted=False)
            result["errors"].append(f"{type(error).__name__}: {error}")
        return save_result(root, result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("protocol-file", "run-root", "train-python", "eval-python"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-only", action="store_true", help="validate complete existing runs and finish unstarted evaluations; never train or overwrite")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        path, result = run_study(parse_args(argv))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"result": str(path), "status": result["status"], "promotion": result["promotion"]}))
    return int(result["status"] == "error")


if __name__ == "__main__":
    raise SystemExit(main())
