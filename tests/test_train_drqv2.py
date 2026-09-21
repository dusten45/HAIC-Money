import json
import random
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from drq_v2 import DrQv2Agent, DrQv2Config
from train_drqv2 import (
    evaluate_checkpoint,
    main,
    selection_score,
    restore_collector,
    resume_selection,
    training_protocol,
    verify_checkpoint,
)


def result(finish=0.0, progress=0.5, lap=None, eligible=True):
    return {"eligible": eligible, "determinism_audited": True,
            "summary": {"finish_rate": finish, "avg_progress": progress,
                        "avg_lap_time_ms": lap}}


def test_selection_uses_cpu_finish_progress_completed_lap_order():
    assert selection_score(result(.1, .1, 20000)) > selection_score(result(0, 1))
    assert selection_score(result(.1, .8, 30000)) > selection_score(result(.1, .1, 20000))
    assert selection_score(result(.1, .8, 20000)) > selection_score(result(.1, .8, 30000))
    with pytest.raises(RuntimeError, match="CPU export"):
        selection_score(result(1, 1, 10000, eligible=False))
    with pytest.raises(ValueError, match="non-finite"):
        selection_score(result(progress=float("nan")))


def test_protocol_reserves_geometry_seeds_across_track_ids(tmp_path):
    protocol = {"name": "test", "frame_skip": 4, "max_steps": 2000, "partitions": {
        "screen": {"track_ids": [101], "seeds": [31001], "repeats": 2},
        "confirmation": {"track_ids": [111], "seeds": [31101], "repeats": 2},
    }}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    args = Namespace(protocol_file=path, frame_skip=4, max_steps=2000)
    assert training_protocol(args)[1] == [31001, 31101]
    protocol["partitions"]["confirmation"]["seeds"] = [31001]
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="partition seeds must be disjoint"):
        training_protocol(args)
    args.max_steps = 100
    with pytest.raises(ValueError, match="horizon"):
        training_protocol(args)


def test_protocol_rejects_silently_changed_training_track_pool(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"frame_skip": 4, "max_steps": 2000,
                               "training_track_ids": [1, 2, 3, 4]}))
    args = Namespace(protocol_file=path, frame_skip=4, max_steps=2000, track_ids="1,2,3")
    with pytest.raises(ValueError, match="training_track_ids"):
        training_protocol(args)


def test_checkpoint_evaluation_is_subprocess_and_checks_actor_identity(tmp_path):
    from train import file_sha256

    actor, protocol, output = [tmp_path / name for name in ("actor.pt", "protocol.json", "evaluation.json")]
    actor.write_bytes(b"actor")
    protocol.write_text("{}")
    args = Namespace(eval_python="/explicit/cpu/python", max_steps=2000, frame_skip=4,
                     evaluations_dir=tmp_path, eval_workers=1)
    cpu_result = {**result(), "archive_sha256": "not-this-actor"}
    with patch("train_drqv2.subprocess.run") as run:
        output.write_text(json.dumps({"evaluation_dir": str(tmp_path), "ranked": [cpu_result],
                                      "partition": "screen", "protocol_sha256": file_sha256(protocol)}))
        with pytest.raises(RuntimeError, match="does not match"):
            evaluate_checkpoint(actor, protocol, output, args)
        command = run.call_args.args[0]
        assert command[0] == "/explicit/cpu/python"
        assert str(actor.resolve()) in command
        assert "--partition" in command and "screen" in command
        assert "_latest" not in " ".join(command)
    with patch("train_drqv2.subprocess.run", side_effect=subprocess.CalledProcessError(1, "cpu")):
        with pytest.raises(subprocess.CalledProcessError):
            evaluate_checkpoint(actor, protocol, output, args)


def test_restore_and_cpu_export_check_preserves_training_rng(tmp_path):
    torch.set_num_threads(1)
    agent = DrQv2Agent(DrQv2Config(replay_capacity=8, batch_size=2, warmup_steps=2))
    checkpoint = agent.save_checkpoint(tmp_path / "checkpoint.pt")
    actor = agent.export_actor(tmp_path / "actor.pt")
    cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    report = verify_checkpoint(agent, checkpoint, actor, np.zeros((4, 84, 84), dtype=np.float32))
    assert report["restore_max_abs_error"] == 0
    assert report["cpu_export_max_abs_error"] == 0
    assert report["resets"] == 2
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert random.getstate() == python_rng


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_learned_cuda_checkpoint_uses_cpu_reference_for_export_parity(tmp_path):
    from common_adapter import Transition

    torch.set_num_threads(1)
    agent = DrQv2Agent(DrQv2Config(device="cuda", replay_capacity=8, batch_size=2, warmup_steps=2), seed=91)
    observation = np.zeros((4, 84, 84), dtype=np.float32)
    for step in range(4):
        agent.observe(Transition(observation, np.zeros(3, dtype=np.float32), 1., observation,
                                 step == 3, False, episode_id=0, step=step))
    agent.update()
    agent.update()
    checkpoint = agent.save_checkpoint(tmp_path / "checkpoint.pt")
    actor = agent.export_actor(tmp_path / "actor.pt")
    gpu_rng = torch.cuda.get_rng_state().clone()
    report = verify_checkpoint(agent, checkpoint, actor, observation)
    assert report["cpu_export_max_abs_error"] == 0
    assert report["restore_max_abs_error"] == 0
    assert report["training_device_vs_cpu_max_abs_error"] >= 0
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())


def test_training_evaluates_after_update_at_each_checkpoint_and_final(tmp_path):
    class Environment:
        def get_wrapper_attr(self, name):
            return np.random.default_rng(917)

        def reset(self, **kwargs):
            return np.zeros((4, 84, 84), dtype=np.float32), {"track_id": 1, "seed": 7}

        def step(self, action):
            return np.zeros((4, 84, 84), dtype=np.float32), 1.0, True, False, {"progress": .1}

        def close(self):
            pass

    calls = []

    def checkpoint(agent, observation, run_dir, config, args, best, trainer_state):
        calls.append((agent.environment_steps, agent.gradient_steps))
        assert trainer_state["format"] == "haic-drq-trainer-v1"
        return {"step": agent.environment_steps}

    args = ["train_drqv2.py", "--name", "test", "--run-dir", str(tmp_path / "run"),
            "--total-steps", "5", "--warmup-steps", "2", "--batch-size", "2",
            "--replay-capacity", "8", "--eval-freq", "2"]
    with patch("sys.argv", args), patch("train_drqv2.build_sampled_env", return_value=Environment()) as build:
        with patch("train_drqv2.save_and_select", side_effect=checkpoint), patch("tracking.pip_freeze", return_value=[]), patch("train_drqv2.check_evaluation_runtime", return_value={}):
            main()
    assert [step for step, updates in calls] == [2, 4, 5]
    assert all(updates > 0 for step, updates in calls)
    assert build.call_args.kwargs["excluded_seeds"] == [14001, 14002, 14003, 14004]
    recorded = json.loads((tmp_path / "run/config.json").read_text())["config"]
    assert recorded["updates_per_step"] == 1
    assert recorded["protocol"]["partitions"]["screen"]["repeats"] == 2
    events = [json.loads(line) for line in (tmp_path / "run/episodes.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event"] == "end"]) == 5
    original_protocol = tmp_path / "predeclared.json"
    original_protocol.write_text(json.dumps(recorded["protocol"], separators=(",", ":")) + "\n")
    with patch("sys.argv", args + ["--run-dir", str(tmp_path / "explicit"), "--protocol-file", str(original_protocol)]):
        with patch("train_drqv2.build_sampled_env", return_value=Environment()), patch("train_drqv2.save_and_select", side_effect=checkpoint):
            with patch("tracking.pip_freeze", return_value=[]), patch("train_drqv2.check_evaluation_runtime", return_value={}):
                main()
    assert (tmp_path / "explicit/protocol.json").read_bytes() == original_protocol.read_bytes()


def test_training_rejects_duplicate_run_directory_and_nonfrozen_frame_skip(tmp_path):
    args = ["train_drqv2.py", "--name", "test", "--total-steps", "1", "--run-dir", str(tmp_path)]
    with patch("sys.argv", args):
        with pytest.raises(FileExistsError):
            main()
    with patch("sys.argv", args + ["--frame-skip", "8"]):
        with pytest.raises(ValueError, match="frame_skip=4"):
            main()


def test_cpu_runtime_preflight_blocks_training_before_collection(tmp_path):
    run = tmp_path / "run"
    argv = ["train_drqv2.py", "--name", "test", "--total-steps", "1", "--run-dir", str(run)]
    with patch("sys.argv", argv), patch("train_drqv2.build_sampled_env") as build:
        with patch("train_drqv2.check_evaluation_runtime", side_effect=subprocess.CalledProcessError(1, "cpu")):
            with pytest.raises(subprocess.CalledProcessError):
                main()
    build.assert_not_called()
    assert not run.exists()


def test_restore_reconstructs_real_episode_prefix_without_replay_leakage():
    import copy
    from common_adapter import EpisodeCollector
    from train import build_sampled_env

    config = {"track_ids": [1, 2], "track_sampler_seed": 917, "max_steps": 40,
              "frame_skip": 4, "seed": 0, "updates_per_step": 1,
              "excluded_training_seeds": [31001], "protocol": {}, "reward_contract": {},
              "runtime": {"device": "cpu"}, "training_source_sha256": {}}
    env = build_sampled_env([1, 2], 917, 40, 4, False, [31001], True)
    other = build_sampled_env([1, 2], 917, 40, 4, False, [31001], True)
    try:
        collector = EpisodeCollector(env)
        before = copy.deepcopy(env.get_wrapper_attr("_rng").bit_generator.state)
        observation, info = collector.reset()
        actions = [np.array([0., 0., -1.], dtype=np.float32)] * 5
        for action in actions:
            observation = collector.step(action).next_observation
        warmup = np.random.default_rng(1)
        warmup.random(10)
        state = {"format": "haic-drq-trainer-v1", "run_config": config,
                 "sampler_before_reset": before, "episode_id": collector.episode_id,
                 "reset_info": info, "episode_actions": actions, "observation": observation,
                 "warmup_rng": copy.deepcopy(warmup.bit_generator.state)}
        restored = EpisodeCollector(other)
        restored_warmup = np.random.default_rng(2)
        actual, _ = restore_collector(restored, other.get_wrapper_attr("_rng"), restored_warmup, state, config)
        assert np.array_equal(actual, observation)
        assert restored.episode_id == collector.episode_id
        assert restored.step_index == collector.step_index
        assert restored_warmup.random() == warmup.random()
        next_action = np.array([.1, .5, -1.], dtype=np.float32)
        expected = collector.step(next_action)
        continued = restored.step(next_action)
        assert np.array_equal(continued.next_observation, expected.next_observation)
        assert continued.reward == expected.reward
        with pytest.raises(ValueError, match="full trainer state"):
            restore_collector(restored, other.get_wrapper_attr("_rng"), restored_warmup, None, config)
        with pytest.raises(ValueError, match="contract mismatch"):
            restore_collector(restored, other.get_wrapper_attr("_rng"), restored_warmup, state,
                              {**config, "track_sampler_seed": 918})
    finally:
        env.close()
        other.close()


def test_resume_selection_retains_better_prior_checkpoint(tmp_path):
    from train import file_sha256

    checkpoint, actor, protocol = [tmp_path / name for name in ("checkpoint.pt", "actor.pt", "protocol.json")]
    checkpoint.write_bytes(b"checkpoint")
    actor.write_bytes(b"actor")
    protocol.write_text("{}")
    record = {"checkpoint_sha256": file_sha256(checkpoint), "actor": str(actor),
              "actor_sha256": file_sha256(actor), "protocol_sha256": file_sha256(protocol),
              "cpu_result": {**result(.1, .5, 20000), "archive_sha256": file_sha256(actor)}}
    (tmp_path / "selection.json").write_text(json.dumps(record))
    best = {**record, "cpu_result": {**result(.2, .8, 25000), "archive_sha256": file_sha256(actor)}}
    assert resume_selection(checkpoint, {"selected_checkpoint": best}, protocol) == best
    actor.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="actor hash"):
        resume_selection(checkpoint, {"selected_checkpoint": best}, protocol)


def test_resume_keeps_prior_actor_on_equal_selection_scores(tmp_path):
    from train import file_sha256

    checkpoint, old_actor, new_actor, protocol = [tmp_path / name for name in (
        "checkpoint.pt", "old.pt", "new.pt", "protocol.json",
    )]
    for path in (checkpoint, old_actor, new_actor, protocol):
        path.write_bytes(path.name.encode())
    record = {"step": 8, "selected": False, "checkpoint_sha256": file_sha256(checkpoint),
              "actor": str(new_actor), "actor_sha256": file_sha256(new_actor),
              "protocol_sha256": file_sha256(protocol),
              "cpu_result": {**result(.1, .5, 20000), "archive_sha256": file_sha256(new_actor)}}
    prior = {**record, "step": 4, "selected": True, "actor": str(old_actor),
             "actor_sha256": file_sha256(old_actor),
             "cpu_result": {**result(.1, .5, 20000), "archive_sha256": file_sha256(old_actor)}}
    (tmp_path / "selection.json").write_text(json.dumps(record))
    assert resume_selection(checkpoint, {"selected_checkpoint": prior}, protocol) == prior


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_closed_loop_training_matches_uninterrupted_resume(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from train import file_sha256

    def evaluation(actor, protocol, output, args):
        step = int(actor.parent.name.split("-")[1])
        return {"evaluation_dir": str(tmp_path / "mock-cpu-evaluation"),
                "ranked": [{**result(.25 if step == 4 else 0, .5, 20000 if step == 4 else None),
                            "archive_sha256": file_sha256(actor)}]}

    def train(name, target, resume=None):
        argv = ["train_drqv2.py", "--name", name, "--run-dir", str(tmp_path / name),
                "--total-steps", str(target), "--max-steps", "6", "--device", device,
                "--warmup-steps", "5", "--batch-size", "2", "--replay-capacity", "32",
                "--updates-per-step", "2", "--eval-freq", "4", "--eval-track-ids", "9", "--eval-seeds", "9"]
        if resume:
            argv += ["--resume", str(resume)]
        with patch("sys.argv", argv), patch("train_drqv2.evaluate_checkpoint", side_effect=evaluation):
            with patch("train_drqv2.check_evaluation_runtime", return_value={}), patch("tracking.pip_freeze", return_value=[]):
                main()
        checkpoint = tmp_path / name / "checkpoints" / f"step-{target:09d}" / "checkpoint.pt"
        return checkpoint, torch.load(checkpoint, map_location="cpu", weights_only=False)

    _, uninterrupted = train("full", 12)
    checkpoint, _ = train("partial", 8)
    _, resumed = train("resumed", 12, checkpoint)
    for name in ("actor", "critic_one", "critic_two", "target_one", "target_two"):
        assert all(torch.equal(value, resumed[name][key]) for key, value in uninterrupted[name].items())
    assert uninterrupted["environment_steps"] == resumed["environment_steps"] == 12
    assert uninterrupted["gradient_steps"] == resumed["gradient_steps"] == 16
    assert torch.equal(uninterrupted["torch_rng_state"], resumed["torch_rng_state"])
    if device == "cuda":
        assert torch.equal(uninterrupted["torch_cuda_rng_state"], resumed["torch_cuda_rng_state"])
    left, right = uninterrupted["trainer_state"], resumed["trainer_state"]
    assert np.array_equal(left["observation"], right["observation"])
    assert left["episode_id"] == right["episode_id"]
    assert left["sampler_before_reset"] == right["sampler_before_reset"]
    assert left["selected_checkpoint"]["step"] == right["selected_checkpoint"]["step"] == 4
