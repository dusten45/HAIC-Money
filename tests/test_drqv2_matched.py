import ast
import json
from pathlib import Path
import shlex
import subprocess
import threading

import pytest

import run_drqv2_matched as runner


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def spec():
    return {
        "name": "matched-test", "max_steps": 6, "frame_skip": 4, "training_track_ids": [1, 2, 3, 4],
        "reserved_training_seeds": [7],
        "partitions": {name: {"track_ids": [100 + index], "seeds": list(range(90 + 10 * index, 94 + 10 * index)), "repeats": 2}
                       for index, name in enumerate(("screen", "confirmation", "blind"))},
        "matched_training": {"arms": {"control": 0.0, "steering_l2": 0.001}, "seeds": [0, 1],
                             "total_steps": 12, "eval_freq": 6, "checkpoint_steps": [6, 12],
                             "batch_size": 2, "warmup_steps": 6, "replay_capacity": 16,
                             "updates_per_step": 1, "track_sampler_seed": 917,
                             "unchanged_drq_config": {"gamma": 0.99, "n_step": 3, "device": "cuda"}},
        "frozen_selection": {"finalist_arm": "steering_l2"},
        "gates": {"confirmation_comparison": {"required_training_seeds": [0, 1],
                                               "min_treatment_finishes_each_seed": 1,
                                               "min_paired_finish_delta_each_seed": 1}},
    }


def runtime(manifest):
    return {"python_version": [3, 11], "sys_platform": "linux", "torch_cuda": None,
            "cuda_available": False, "torch_threads": 1, "torch_interop_threads": 1,
            "packages": {"torch": "2.1.0+cpu", "numpy": "1.26.0", "gymnasium": "0.29.1", "opencv-python": "4.8.1.78"},
            "runtime_sources": {name: manifest["source_sha256"][name] for name in ("evaluate_policy.py", "agent.py")}}


def publish_evaluation(root, run, actor, partition, pointer, finishes, diagnostic=False, previous=None):
    protocol = runner.read_json(root / "protocol.json")
    manifest = runner.read_json(root / "manifest.json")
    matrix = protocol["partitions"][partition]
    directory = run / "evaluations" / f"{partition}-{actor.parent.name}"
    directory.mkdir(parents=True)
    archive = directory / "candidates/actor.pt"
    archive.parent.mkdir()
    archive.write_bytes(actor.read_bytes())
    (directory / "candidates/config.json").write_bytes((run / "config.json").read_bytes())
    (directory / "protocol_spec.json").write_bytes((root / "protocol.json").read_bytes())
    put(directory / "protocol.json", matrix)
    digest = runner.sha256(actor)
    worker_runtime = runtime(manifest)
    candidate_id = digest[:16]
    rows, audits = [], []
    index = 0
    for track in matrix["track_ids"]:
        for seed in matrix["seeds"]:
            finished = index < finishes
            index += 1
            for repeat in range(matrix["repeats"]):
                rows.append({"candidate_id": candidate_id, "track_id": track, "seed": seed, "repeat": repeat,
                             "status": "ok", "finished": finished, "progress": 0.75, "lap_time_ms": 1000 if finished else None,
                             "action_trace_sha256": f"{digest}-{seed}", "steps": 4,
                             "termination_class": "finished" if finished else "off_track", "damage": 0.0,
                             "loaded_archive_sha256": digest, "runtime": worker_runtime,
                             "process_initialization_seconds": 0.2, "agent_reset_seconds": 0.01,
                             "max_action_seconds": 0.001, "peak_rss_bytes": 100000})
            audits.append({"candidate_id": candidate_id, "track_id": track, "seed": seed,
                           "repeats": matrix["repeats"], "audited": True, "matches_canonical": True})
    count = len(audits)
    result = {"candidate_id": candidate_id, "algorithm": "drq-v2", "source_path": str(actor),
              "export_metadata": {"format": "haic-drq-v2-actor-v1", "config": runner.read_json(run / "config.json")["config"]["drq_config"]},
              "archive_sha256": digest, "evaluation_archive_path": "candidates/actor.pt",
              "run_config_sha256": runner.sha256(run / "config.json"), "evaluation_run_config_path": "candidates/config.json",
              "diagnostic_only": diagnostic, "eligible": True, "determinism_audited": True,
              "cpu_reload_matches": True, "operational_failures": 0, "expected_cells": count,
              "canonical_episodes": count, "expected_results": len(rows),
              "summary": {"n_episodes": count, "finish_rate": finishes / count, "avg_progress": 0.75,
                          "avg_lap_time_ms": 1000 if finishes else None}}
    receipt = {"partition": partition, "protocol_sha256": manifest["protocol_sha256"], "diagnostic_only": diagnostic,
               "cell_matrix": matrix, "max_steps": protocol["max_steps"], "frame_skip": 4,
               "run_dir": str(run), "worker_runtime": [worker_runtime], "previous_evaluation": None}
    if previous:
        previous_pointer = runner.read_json(previous)
        prior = Path(previous_pointer["evaluation_dir"])
        provenance = {}
        for name, source in (("previous_evaluation.json", previous), ("previous_summary.json", prior / "summary.json"),
                             ("previous_manifest.json", prior / "manifest.json")):
            (directory / name).write_bytes(source.read_bytes())
            provenance[name] = {"source_path": str(source), "sha256": runner.sha256(source)}
        receipt["previous_evaluation"] = {"actor_sha256": digest, "partition": previous_pointer["partition"],
                                          "evaluation_dir": str(prior), "files": provenance}
    put(directory / "manifest.json", receipt)
    put(directory / "summary.json", [result])
    jsonl(directory / "episodes.jsonl", rows)
    put(directory / "determinism.json", audits)
    put(pointer, {"evaluation_dir": str(directory), "ranked": [result], "partition": partition,
                  "protocol_sha256": manifest["protocol_sha256"], "diagnostic_only": diagnostic})
    return result, directory


def publish_training(root, job, screens, alias=False):
    protocol = runner.read_json(root / "protocol.json")
    manifest = runner.read_json(root / "manifest.json")
    execution, matched = manifest["execution"], protocol["matched_training"]
    run = Path(job["run_dir"])
    run.mkdir()
    config = {"algorithm": "drq-v2", "seed": job["seed"], "track_ids": protocol["training_track_ids"],
              "protocol": protocol, "max_steps": protocol["max_steps"], "frame_skip": 4,
              "excluded_training_seeds": runner.reserved_seeds(protocol), "resume_from": None, "resume_sha256": None,
              "training_source_sha256": {name: digest for name, digest in manifest["source_sha256"].items()
                                        if name in runner.TRAIN_SOURCES or name.startswith("core/")},
              "observation_channels": 4, "training_track_mode": "sampled", "training_obstacles": "official",
              "reward_contract": {"reward_shaping": False, "norm_reward": False, "collision_penalty": 0.0},
              "eval_python": execution["eval_python"], "eval_workers": execution["eval_workers"],
              "action_smoothing": {"method": "none"},
              "evaluation_runtime": {"status": "ok", "runtime": runtime(manifest)},
              "runtime": {"torch_threads": 1, "deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False},
              "drq_config": {**matched["unchanged_drq_config"], "batch_size": matched["batch_size"],
                             "warmup_steps": matched["warmup_steps"], "replay_capacity": matched["replay_capacity"],
                             "device": execution["device"], "steering_logit_l2": matched["arms"][job["arm"]]}}
    config.update({key: matched[key] for key in ("total_steps", "eval_freq", "updates_per_step", "track_sampler_seed")})
    put(run / "config.json", {"config": config, "command_line": shlex.join(job["command"][2:])})
    (run / "protocol.json").write_bytes((root / "protocol.json").read_bytes())
    duration = 3 + job["seed"] + (2 if job["arm"] == "steering_l2" else 0)
    episodes = []
    global_step, episode_id = 0, 0
    while True:
        track, seed = 1 + episode_id % 4, 10000 + episode_id
        episodes.append({"event": "reset", "episode_id": episode_id, "track_id": track, "seed": seed})
        if global_step + duration > matched["total_steps"]:
            break
        global_step += duration
        episodes.append({"event": "end", "episode_id": episode_id, "track_id": track, "seed": seed,
                         "steps": duration, "global_step": global_step})
        episode_id += 1
    jsonl(run / "episodes.jsonl", episodes)
    updates = (matched["total_steps"] - matched["warmup_steps"] + 1) * matched["updates_per_step"]
    jsonl(run / "metrics.jsonl", [{"step": matched["total_steps"], "replay_bytes": 1000, "critic_loss": 0.1,
                                 "gradient_steps": float(updates), "replay_size": float(min(matched["total_steps"], matched["replay_capacity"]))}])
    best = None
    for step in matched["checkpoint_steps"]:
        directory = run / "checkpoints" / f"step-{step:09d}"
        directory.mkdir(parents=True)
        identity = "shared-treatment" if alias and job["arm"] == "steering_l2" else job["name"]
        actor, checkpoint = directory / "actor.pt", directory / "checkpoint.pt"
        actor.write_bytes(f"{identity}-{step}".encode())
        checkpoint.write_bytes(f"checkpoint-{job['name']}-{step}".encode())
        put(directory / "checkpoint.manifest.json", {"extra": {**config, "environment_steps": step}})
        result, evaluation = publish_evaluation(root, run, actor, "screen", directory / "evaluation.json", screens[job["name"]])
        selected = best is None or runner.selection_score(result) > runner.selection_score(best["cpu_result"])
        selection = {"step": step, "actor": str(actor), "actor_sha256": runner.sha256(actor),
                     "gradient_steps": max(0, step - matched["warmup_steps"] + 1) * matched["updates_per_step"],
                     "replay_size": min(step, matched["replay_capacity"]),
                     "checkpoint": str(checkpoint), "checkpoint_sha256": runner.sha256(checkpoint),
                     "protocol_sha256": manifest["protocol_sha256"], "cpu_result": result, "evaluation_dir": str(evaluation),
                     "selected": selected, "parity": {"restore_max_abs_error": 0.0, "cpu_export_max_abs_error": 0.0, "resets": 2}}
        put(directory / "selection.json", selection)
        if selected:
            best = selection
    put(run / "selection.json", best)
    put(run / "result.json", {"environment_steps": matched["total_steps"], "gradient_steps": updates, "selected_checkpoint": best})


@pytest.fixture
def study(tmp_path, monkeypatch):
    protocol_file = tmp_path / "protocol.json"
    put(protocol_file, spec())
    args = runner.parse_args(["--protocol-file", str(protocol_file), "--run-root", str(tmp_path / "study"),
                              "--train-python", "/explicit/gpu/bin/python", "--eval-python", "/explicit/cpu/bin/python"])
    screens = {f"{arm}-seed{seed}": (1 if arm == "control" else 2) for arm in runner.ARMS for seed in runner.SEEDS}
    confirms = {"control-seed0": 1, "control-seed1": 1, "steering_l2-seed0": 2, "steering_l2-seed1": 3}
    calls = []

    def process(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["timeout"] > 0 and kwargs["check"] is False
        assert kwargs["env"]["OMP_NUM_THREADS"] == kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        root = args.run_root
        manifest = runner.read_sealed(root / "manifest.json")
        if Path(command[2]).name == "train_drqv2.py":
            job = next(job for job in manifest["jobs"] if job["command"] == command)
            publish_training(root, job, screens)
        else:
            assert (root / "frozen_candidates.json").is_file()
            partition = command[command.index("--partition") + 1]
            if partition == "blind":
                assert runner.read_sealed(root / "confirmation_gate.json")["passed"]
            actor = Path(command[command.index("--model") + 1])
            run = Path(command[command.index("--run-dir") + 1])
            previous = Path(command[command.index("--previous-evaluation") + 1])
            output = Path(command[command.index("--output") + 1])
            count = confirms.get("blind", 1) if partition == "blind" else confirms[run.name]
            publish_evaluation(root, run, actor, partition, output, count, "--diagnostic-confirmation" in command, previous)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", process)
    return args, screens, confirms, calls


@pytest.fixture
def complete(study):
    args, screens, confirms, calls = study
    root, protocol, manifest = runner.initialize(args)
    for job in manifest["jobs"]:
        publish_training(root, job, screens)
    return root, protocol, manifest, args


def score(finish=0.0, progress=0.5, lap=None):
    return {"eligible": True, "determinism_audited": True, "cpu_reload_matches": True, "operational_failures": 0,
            "summary": {"finish_rate": finish, "avg_progress": progress, "avg_lap_time_ms": lap}}


def test_coordinator_has_only_stdlib_imports_and_preserves_cuda(monkeypatch):
    tree = ast.parse(Path(runner.__file__).read_text())
    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert imports <= {"argparse", "ast", "copy", "fcntl", "hashlib", "json", "math", "os", "pathlib", "shlex", "subprocess", "sys", "concurrent"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    environment = runner.child_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert "PYTHONPATH" not in environment
    assert all(environment[key] == "1" for key in runner.THREAD_ENV)


def test_interpreter_symlink_is_not_resolved(study, tmp_path):
    args, *_ = study
    target = tmp_path / "base-python"
    target.touch()
    python = tmp_path / "cpu-python"
    python.symlink_to(target)
    args.eval_python = python
    root, protocol, manifest = runner.initialize(args)
    assert manifest["execution"]["eval_python"] == str(python)
    assert str(python) in manifest["jobs"][0]["command"]
    assert (root / "protocol.json").read_bytes() == args.protocol_file.read_bytes()
    assert (root / "source/train_drqv2.py").is_file()


def test_ranking_finish_progress_lap_and_nonfinite():
    assert runner.selection_score(score(.1, .1, 100)) > runner.selection_score(score(0, 1))
    assert runner.selection_score(score(.1, .8, 300)) > runner.selection_score(score(.1, .1, 100))
    assert runner.selection_score(score(.1, .8, 100)) > runner.selection_score(score(.1, .8, 300))
    with pytest.raises(ValueError, match="metrics"):
        runner.selection_score(score(progress=float("nan")))
    broken = score()
    broken["cpu_reload_matches"] = False
    with pytest.raises(ValueError, match="operational/reload"):
        runner.selection_score(broken)


def test_four_complete_jobs_freeze_before_confirmation_and_never_rerank(study):
    args, screens, confirms, calls = study
    path, result = runner.run_study(args)
    assert result["status"] == "passed" and result["promotion"]
    assert result["next_stage_permitted"] and not result["official_submission_authorized"]
    frozen = runner.read_sealed(args.run_root / "frozen_candidates.json")
    assert [candidate["step"] for candidate in frozen["candidates"]] == [6] * 4
    assert frozen["finalist"] == "steering_l2-seed0"
    assert len(set(frozen["episode_prefix_lengths"].values())) > 1
    assert result["paired_comparison"][1]["treatment_finishes"] > result["paired_comparison"][0]["treatment_finishes"]
    assert len(calls) == 9
    blind = calls[-1][0]
    assert blind[blind.index("--partition") + 1] == "blind"
    assert "steering_l2-seed0" in blind[blind.index("--model") + 1]
    assert all("--previous-evaluation" in command for command, _ in calls[4:])
    original = path.read_bytes()
    args.evaluate_only = True
    recovered, repeated = runner.run_study(args)
    assert repeated["promotion"] and len(calls) == 9
    assert recovered != path and path.read_bytes() == original


@pytest.mark.parametrize("case", ["zero_screen", "tie", "regression", "zero_confirmation"])
def test_diagnostics_progress_and_one_seed_gain_cannot_promote(study, case):
    args, screens, confirms, calls = study
    if case == "zero_screen":
        screens["steering_l2-seed1"] = 0
        confirms["steering_l2-seed1"] = 4
    elif case == "tie":
        confirms["steering_l2-seed1"] = 1
    elif case == "regression":
        confirms["control-seed1"], confirms["steering_l2-seed1"] = 3, 2
    else:
        confirms["steering_l2-seed1"] = 0
    _, result = runner.run_study(args)
    assert result["status"] == "rejected" and not result["promotion"]
    assert result["blind"]["status"] == "not_run" and len(calls) == 8
    confirmation = next(command for command, _ in calls if "--partition" in command and "steering_l2-seed1" in command[command.index("--run-dir") + 1])
    assert ("--diagnostic-confirmation" in confirmation) == (case == "zero_screen")


def test_zero_screen_control_is_diagnostic_comparator_not_promoted(study):
    args, screens, confirms, calls = study
    screens["control-seed0"] = 0
    _, result = runner.run_study(args)
    assert result["promotion"]
    assert result["confirmations"]["control-seed0"]["diagnostic_only"]
    assert result["finalist"].startswith("steering_l2")


def test_zero_blind_completion_blocks_promotion_without_fallback(study):
    args, _, confirms, calls = study
    confirms["blind"] = 0
    _, result = runner.run_study(args)
    assert result["confirmation_gate_passed"] and not result["promotion"]
    assert result["status"] == "rejected" and result["blind"]["finishes"] == 0
    assert len(calls) == 9


def test_confirmation_failure_does_not_cancel_other_candidates_or_open_blind(study, monkeypatch):
    args, _, _, calls = study
    process = runner.subprocess.run

    def corrupt(command, **kwargs):
        completed = process(command, **kwargs)
        if "--partition" in command and "control-seed0" in command[command.index("--run-dir") + 1]:
            pointer = runner.read_json(Path(command[command.index("--output") + 1]))
            episodes = Path(pointer["evaluation_dir"]) / "episodes.jsonl"
            rows = [json.loads(line) for line in episodes.read_text().splitlines()]
            rows[1]["max_action_seconds"] = 6
            jsonl(episodes, rows)
        return completed

    monkeypatch.setattr(runner.subprocess, "run", corrupt)
    _, result = runner.run_study(args)
    assert result["status"] == "error" and not result["promotion"] and len(calls) == 8
    assert sum(receipt["status"] == "ok" for receipt in result["confirmations"].values()) == 3


def test_all_gpu_jobs_finish_and_failures_are_recorded(study, monkeypatch):
    args, _, _, calls = study
    process = runner.subprocess.run
    barrier = threading.Barrier(4, timeout=10)

    def fail_one(command, **kwargs):
        barrier.wait()
        if "control-seed0" in command:
            calls.append((command, kwargs))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return process(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fail_one)
    path, result = runner.run_study(args)
    assert len(calls) == 4 and result["status"] == "error" and not result["promotion"]
    assert len(result["training_jobs"]) == 4
    assert sum(job["status"] == "ok" for job in result["training_jobs"]) == 3
    assert len(list((args.run_root / "jobs").glob("*.json"))) == 4
    assert runner.read_sealed(path) == result


@pytest.mark.parametrize("mutation,message", [
    ("partial_budget", "exact training budget"), ("extra_budget", "exact training budget"),
    ("config", "config mismatch"), ("additional_config", "differ beyond"),
    ("schedule_arm", "prefixes differ"), ("schedule_seed", "prefixes differ"),
    ("leak", "seed leak"), ("checkpoint", "checkpoint selection identity"),
    ("missing_checkpoint", "checkpoint budget"), ("protocol", "protocol byte"),
    ("nan_loss", "non-finite"), ("partial_log", "partial training episode"),
    ("updates", "learner update budget"), ("metric_updates", "metrics learner update"),
    ("replay", "replay occupancy"), ("checkpoint_updates", "checkpoint learner/replay"),
])
def test_rejects_incomplete_or_unmatched_training(complete, mutation, message):
    root, protocol, manifest, args = complete
    run = root / ("steering_l2-seed1" if mutation == "schedule_arm" else "control-seed1")
    if mutation in ("partial_budget", "extra_budget", "updates"):
        record = runner.read_json(run / "result.json")
        record["gradient_steps" if mutation == "updates" else "environment_steps"] += 1 if mutation == "extra_budget" else -1
        put(run / "result.json", record)
    elif mutation in ("config", "additional_config"):
        record = runner.read_json(run / "config.json")
        record["config"]["track_sampler_seed" if mutation == "config" else "unexpected_axis"] = 99
        put(run / "config.json", record)
    elif mutation.startswith("schedule") or mutation == "leak":
        rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
        for row in rows:
            if row["episode_id"] == 1:
                row["seed"] = 7 if mutation == "leak" else 999
        jsonl(run / "episodes.jsonl", rows)
    elif mutation == "checkpoint":
        (run / "checkpoints/step-000000006/checkpoint.pt").write_bytes(b"changed")
    elif mutation == "missing_checkpoint":
        (run / "checkpoints/step-000000013").mkdir()
    elif mutation == "protocol":
        with (run / "protocol.json").open("a") as handle:
            handle.write(" ")
    elif mutation == "nan_loss":
        jsonl(run / "metrics.jsonl", [{"step": 12, "replay_bytes": 1000, "critic_loss": float("nan")}])
    elif mutation in ("metric_updates", "replay"):
        row = json.loads((run / "metrics.jsonl").read_text())
        del row["gradient_steps" if mutation == "metric_updates" else "replay_size"]
        jsonl(run / "metrics.jsonl", [row])
    elif mutation == "checkpoint_updates":
        path = run / "checkpoints/step-000000006/selection.json"
        row = runner.read_json(path)
        row["gradient_steps"] = 0
        put(path, row)
    else:
        jsonl(run / "episodes.jsonl", [{"event": "reset", "episode_id": 0, "track_id": 1, "seed": 10000}])
    with pytest.raises(ValueError, match=message):
        runner.freeze_candidates(root, protocol, manifest)
    assert not (root / "frozen_candidates.json").exists()


@pytest.mark.parametrize("mutation,message", [("pointer", "summary mismatch"), ("protocol", "protocol mismatch"),
                                             ("actor", "snapshot mismatch"), ("cell", "partial evaluation"),
                                             ("reload", "reload outcome"), ("resource", "resource failure"),
                                             ("export_config", "exported actor configuration"),
                                             ("export_missing", "exported actor configuration")])
def test_rejects_invalid_immutable_screen_evidence(complete, mutation, message):
    root, protocol, manifest, args = complete
    pointer = root / "control-seed0/checkpoints/step-000000006/evaluation.json"
    directory = Path(runner.read_json(pointer)["evaluation_dir"])
    if mutation == "pointer":
        value = runner.read_json(pointer)
        value["ranked"][0]["summary"]["finish_rate"] = 1.0
        put(pointer, value)
    elif mutation == "protocol":
        value = runner.read_json(directory / "manifest.json")
        value["protocol_sha256"] = "wrong"
        put(directory / "manifest.json", value)
    elif mutation == "actor":
        (directory / "candidates/actor.pt").write_bytes(b"wrong")
    elif mutation.startswith("export_"):
        value = runner.read_json(pointer)
        if mutation == "export_config":
            value["ranked"][0]["export_metadata"]["config"]["steering_logit_l2"] = 0.001
        else:
            del value["ranked"][0]["export_metadata"]
        put(pointer, value)
        put(directory / "summary.json", value["ranked"])
    else:
        rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
        if mutation == "cell":
            rows.pop()
        elif mutation == "reload":
            rows[1]["action_trace_sha256"] = "changed"
        else:
            rows[1]["peak_rss_bytes"] = 1024**3 + 1
        jsonl(directory / "episodes.jsonl", rows)
    with pytest.raises(ValueError, match=message):
        runner.freeze_candidates(root, protocol, manifest)


def test_evaluate_only_finishes_complete_explicit_runs_without_training(complete, study):
    root, protocol, manifest, args = complete
    args.evaluate_only = True
    _, result = runner.run_study(args)
    assert result["promotion"]
    assert len(study[3]) == 5
    assert all(Path(command[2]).name == "evaluate_policy.py" for command, _ in study[3])


def test_ambiguous_published_confirmation_is_never_silently_rerun(complete, study):
    root, protocol, manifest, args = complete
    frozen = runner.freeze_candidates(root, protocol, manifest)
    candidate = frozen["candidates"][0]
    run = Path(candidate["run_dir"])
    put(run / "confirmation.json", {"published": "without receipt"})
    before = (run / "confirmation.json").read_bytes()
    report = runner.evaluate_fixed(root, candidate, "confirmation", frozen["commands"][candidate["name"]], protocol, manifest)
    assert report["status"] == "error" and "refusing to rerun" in report["error"]
    assert not study[3] and (run / "confirmation.json").read_bytes() == before


@pytest.mark.parametrize("already_frozen", [False, True])
def test_orphan_published_confirmation_without_pointer_is_not_repeated(complete, study, already_frozen):
    root, protocol, manifest, args = complete
    if already_frozen:
        frozen = runner.freeze_candidates(root, protocol, manifest)
        candidate = frozen["candidates"][0]
    else:
        candidate = runner.validate_runs(root, protocol, manifest)[0][0]
    run, actor = Path(candidate["run_dir"]), Path(candidate["actor"])
    pointer = run / "lost-pointer.json"
    publish_evaluation(root, run, actor, "confirmation", pointer, 1, previous=actor.parent / "evaluation.json")
    pointer.unlink()
    if already_frozen:
        report = runner.evaluate_fixed(root, candidate, "confirmation", frozen["commands"][candidate["name"]], protocol, manifest)
        assert report["status"] == "error" and "already published" in report["error"]
    else:
        with pytest.raises(ValueError, match="holdout exists before"):
            runner.freeze_candidates(root, protocol, manifest)
        assert not (root / "frozen_candidates.json").exists()
    assert not study[3]


def test_recovery_rejects_mutated_published_confirmation(study):
    args, screens, confirms, calls = study
    runner.run_study(args)
    pointer = args.run_root / "steering_l2-seed0/confirmation.json"
    with pointer.open("a") as handle:
        handle.write(" ")
    args.evaluate_only = True
    path, result = runner.run_study(args)
    assert result["status"] == "error" and not result["promotion"] and len(calls) == 9
    assert any("mutated artifact" in error for error in result["errors"])


def test_recovery_rejects_protocol_source_and_result_mutation(study):
    args, _, _, calls = study
    runner.run_study(args)
    args.evaluate_only = True
    with args.protocol_file.open("a") as handle:
        handle.write(" ")
    _, result = runner.run_study(args)
    assert not result["promotion"] and "protocol changed" in result["errors"][0]
    args.protocol_file.write_bytes((args.run_root / "protocol.json").read_bytes())
    with (args.run_root / "result.json").open("a") as handle:
        handle.write(" ")
    _, result = runner.run_study(args)
    assert not result["promotion"] and "mutated sealed result" in result["errors"][0]
    assert len(calls) == 9


def test_recovery_rejects_mutated_frozen_source(study):
    args, _, _, calls = study
    runner.run_study(args)
    args.evaluate_only = True
    with (args.run_root / "source/evaluate_policy.py").open("a") as handle:
        handle.write("# mutated\n")
    _, result = runner.run_study(args)
    assert not result["promotion"] and result["status"] == "error" and len(calls) == 9
    assert "mutated artifact" in result["errors"][0]


def test_duplicate_actor_lineage_does_not_replace_required_training_seeds(study):
    args, screens, _, _ = study
    root, protocol, manifest = runner.initialize(args)
    for job in manifest["jobs"]:
        publish_training(root, job, screens, alias=True)
    frozen = runner.freeze_candidates(root, protocol, manifest)
    assert frozen["training_runs"] == 4 and frozen["unique_selected_actors"] == 3
    assert frozen["paired_training_seeds"] == [0, 1] and frozen["arm_counts"] == {"control": 2, "steering_l2": 2}
    assert ["steering_l2-seed0", "steering_l2-seed1"] in frozen["actor_lineage_groups"].values()
    config_path = root / "steering_l2-seed1/config.json"
    value = runner.read_json(config_path)
    value["config"]["seed"] = 0
    put(config_path, value)
    with pytest.raises(ValueError, match="mutated artifact"):
        runner.freeze_candidates(root, protocol, manifest)


def test_rejects_latest_existing_root_and_excess_workers(study):
    args, *_ = study
    args.jobs, args.eval_workers = 4, 3
    with pytest.raises(ValueError, match="eight"):
        runner.initialize(args)
    args.eval_workers = 2
    original = args.run_root
    args.run_root = original.parent / "_latest"
    with pytest.raises(ValueError, match="_latest"):
        runner.initialize(args)
    args.run_root = original
    runner.initialize(args)
    with pytest.raises(FileExistsError):
        runner.initialize(args)


@pytest.mark.parametrize("reserved", [[True], [1.0], None, [7, 7], [-1], [2**32]])
def test_reserved_seed_validation(reserved):
    protocol = spec()
    protocol["reserved_training_seeds"] = reserved
    with pytest.raises(ValueError, match="reserved_training_seeds"):
        runner.validate_protocol(protocol)
