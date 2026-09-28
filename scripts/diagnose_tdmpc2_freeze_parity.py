"""No-reset, source-bound offline TD-MPC2 final-model CPU/GPU diagnosis.

Run from the repository root: python -m scripts.diagnose_tdmpc2_freeze_parity
Only archived, completed TRAIN replay is read. Neither this module nor its
scorer imports or constructs an environment, runs an optimizer, or writes a
receipt. A .pt file is trusted for pickle deserialization ONLY after its SHA,
the completed result, both entire ledgers, and all frozen sources are checked.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/tdmpc2-long-20260928-v2"
PROTOCOL = "experiments/tdmpc2-long-reused-train-v2.json"
PROTOCOL_SHA = "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"
RESULT_SHA = "287470932dbba9a7eeed778a5be7762b736e6679d99fd29078c5103890a75a4b"
CHECKPOINT = f"{RUN}/checkpoint-at-least-100000-step-100354.pt"
CHECKPOINT_SHA = "aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295"
TRAIN_SHA = "84f06ee4dcc8568bd5d384d1811252321ad5dc6a98ce19d18badb448d09123d7"
STEP_SHA = "a94ce156afcb6e4f706474d6cf2d2ccb4754a868eb75a3eb4b53df5d2be37945"
# Reuse the strict full-cursor validator, but only after pinning its own bytes.
BINDER_SHA = "d2f3df9522257f2dfbffe5167571c5a491b7929acbbbcb3b84ebb0c2f8b91089"
DECISIONS, EPISODES, HORIZON = 100354, 307, 3
WINDOWS = (
    ("seed-random-ep0", 0, 16, (0, 0)),
    ("seed-random-ep1", 1, 16, (-1, 2)),
    ("late-finished-ep305", 305, 16, (2, -2)),
    ("late-nonfinish-ep306", 306, 16, (1, 1)),
)
# Absolute, predeclared float32 thresholds, applied independently to each output.
TOLERANCES = {
    "encoded_latent": 2e-4, "predicted_next_latent": 2e-4,
    "tanh_actor_prior_mean": 3e-4, "reward_logits": 2e-3,
    "decoded_reward": 2e-2, "q_logits_all_five": 2e-3,
    "decoded_q_all_five": 2.5e-1, "terminal_logits": 2e-3,
    "terminal_probability": 5e-4,
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_files(root: Path) -> dict:
    """Fail before deserialization on any protocol/source/whole-ledger drift."""
    root = root.resolve(strict=True)
    binder_path = root / "scripts/evaluate_tdmpc2_full_train.py"
    if binder_path.is_symlink() or binder_path.parent.is_symlink() or _digest(binder_path) != BINDER_SHA:
        raise ValueError("frozen evaluator cursor validator SHA mismatch")
    from scripts import evaluate_tdmpc2_full_train as binder

    for name, sha in ((PROTOCOL, PROTOCOL_SHA), (f"{RUN}/result.json", RESULT_SHA),
                      ("scripts/evaluate_tdmpc2_full_train.py", BINDER_SHA)):
        if _digest(binder._path(root, name)) != sha:
            raise ValueError(f"frozen SHA mismatch: {name}")
    source = {"protocol": {"path": PROTOCOL, "sha256": PROTOCOL_SHA},
              "result": {"path": f"{RUN}/result.json", "sha256": RESULT_SHA},
              "checkpoints": [{"target": 100000, "path": CHECKPOINT,
                               "sha256": CHECKPOINT_SHA, "training_cursor_sha256": TRAIN_SHA,
                               "step_cursor_sha256": STEP_SHA}]}
    p, result, run = binder._source(root, {"source": source})
    if (p["run_dir"] != RUN or result["decisions"] != DECISIONS or result["updates"] != DECISIONS
            or result["episodes"] != EPISODES or result["training_ledger_sha256"] != TRAIN_SHA
            or result["step_ledger_sha256"] != STEP_SHA
            or result["checkpoints"][-1]["sha256"] != CHECKPOINT_SHA):
        raise ValueError("not the completed final long-v2 source and ledger")
    pins = binder._ledger(root, p, result, run, source)
    if len(pins) != 1 or any(pins[0][k] != v for k, v in {
            "target": 100000, "decisions": DECISIONS, "updates": DECISIONS,
            "episodes": EPISODES, "path": CHECKPOINT, "sha256": CHECKPOINT_SHA,
            "training_cursor_sha256": TRAIN_SHA, "step_cursor_sha256": STEP_SHA}.items()):
        raise ValueError("final checkpoint does not have the complete ledger cursors")
    return {"root": root, "protocol": p, "pin": pins[0], "binder": binder}


def _load_checkpoint(bound: dict) -> dict:
    """Rehash the same open file descriptor passed to torch.load (no path race)."""
    path = bound["binder"]._path(bound["root"], CHECKPOINT)
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != CHECKPOINT_SHA:
            raise ValueError("checkpoint changed after the full ledger check")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)
    if (not isinstance(state, dict) or set(state) != {
            "format", "protocol_sha256", "source_sha256", "target", "decisions", "updates",
            "episodes", "action_dim", "learner", "optim", "pi_optim", "replay", "probe",
            "rng", "resume_supported"} or state["format"] != bound["protocol"]["format"]
            or state["protocol_sha256"] != PROTOCOL_SHA
            or state["source_sha256"] != bound["protocol"]["source_sha256"]
            or state["resume_supported"] is not False or state["action_dim"] != 3
            or any(state.get(k) != bound["pin"][k] for k in ("target", "decisions", "updates", "episodes"))
            or not isinstance(state["learner"], dict)
            or set(state["learner"]) != {"q_scale"} | {
                f"model.{k}" for k in _model_keys()}):
        raise ValueError("checkpoint payload or strict model weights differ from frozen source")
    return state


def _model_keys() -> set[str]:
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

    with torch.random.fork_rng(devices=[]):
        return set(WorldModel(TDMPC2ModelConfig(
            action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True)).state_dict())


def _windows(state: dict, bound: dict) -> list[dict]:
    """Bind selected complete replay episodes, pixel bytes and actions to TRAIN."""
    if (_digest(bound["binder"]._path(bound["root"], f"{RUN}/training.jsonl")) != TRAIN_SHA
            or _digest(bound["binder"]._path(bound["root"], f"{RUN}/steps.jsonl")) != STEP_SHA):
        raise ValueError("TRAIN ledger changed after checkpoint deserialization")
    replay = state["replay"]
    if (not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or replay.get("capacity") != 120000 or replay.get("horizon") != HORIZON
            or replay.get("action_dim") != 3 or replay.get("augmentation_pad") != 3
            or replay.get("observation_shape") != (4, 64, 64) or replay.get("active") is not None
            or replay.get("size") != DECISIONS or replay.get("next_episode_id") != EPISODES
            or not isinstance(replay.get("episodes"), list) or len(replay["episodes"]) != EPISODES):
        raise ValueError("replay is not the complete, untrimmed TRAIN snapshot")
    episodes = replay["episodes"]
    if any(ep.get("episode_id") != i or ep.get("start_step") != 0
           or len(ep.get("actions", ())) < 1 for i, ep in enumerate(episodes)):
        raise ValueError("replay episode IDs were trimmed or reordered")
    if sum(len(ep["actions"]) for ep in episodes) != DECISIONS:
        raise ValueError("replay does not contain all 100354 training decisions")
    wanted = {spec[1] for spec in WINDOWS}
    rows = {}
    with (bound["root"] / RUN / "training.jsonl").open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "episode" and row.get("episode") in wanted:
                rows[row["episode"]] = row
    if set(rows) != wanted:
        raise ValueError("selected TRAIN episode IDs missing from sealed ledger")
    selected_steps = {}
    decision_starts = {}
    for eid in wanted:
        ep, row = episodes[eid], rows[eid]
        length = row["length"]
        if (len(ep["actions"]) != length or len(ep["observations"]) != length + 1
                or ep["actions"].dtype != np.float32 or ep["actions"].shape != (length, 3)
                or ep["observations"].dtype != np.uint8
                or ep["observations"].shape != (length + 1, 4, 64, 64)
                or hashlib.sha256(ep["actions"].tobytes()).hexdigest() != row["action_trace_sha256"]
                or not np.isfinite(ep["actions"]).all() or np.any(np.abs(ep["actions"]) > 1)
                or row["track_id"] != 1
                or row["geometry_seed"] != bound["protocol"]["cells"][eid % 4]["geometry_seed"]
                or bool(ep["terminated"][-1]) != row["terminated"]
                or bool(ep["truncated"][-1]) != row["truncated"]
                or bool(ep["terminal"][-1]) != row["terminal"]
                or row["finished"] != (eid == 305)):
            raise ValueError(f"selected replay episode {eid} differs from the TRAIN ledger")
        decision_starts[eid] = row["decisions"] - length
    wanted_decisions = {decision_starts[eid] + start + t + 1: (eid, start + t)
                        for _, eid, start, _ in WINDOWS for t in range(HORIZON)}
    with (bound["root"] / RUN / "steps.jsonl").open("rb") as stream:
        for index, line in enumerate(stream, 1):
            if index in wanted_decisions:
                selected_steps[index] = json.loads(line)
    if set(selected_steps) != set(wanted_decisions):
        raise ValueError("selected action bytes missing from sealed step ledger")
    windows = []
    for label, eid, start, shift in WINDOWS:
        ep, row = episodes[eid], rows[eid]
        if start + HORIZON >= row["length"]:
            raise ValueError("fixed H3 window is not fully inside a completed episode")
        observations = np.ascontiguousarray(ep["observations"][start:start + HORIZON + 1])
        actions = np.ascontiguousarray(ep["actions"][start:start + HORIZON])
        for t in range(HORIZON):
            step = selected_steps[decision_starts[eid] + start + t + 1]
            if (step["episode"] != eid or step["geometry_seed"] != row["geometry_seed"]
                    or bytes.fromhex(step["action_f32_hex"]) != actions[t].tobytes()
                    or np.float32(step["reward"]) != ep["rewards"][start + t]
                    or any(bool(ep[key][start + t]) != step[key]
                           for key in ("terminated", "truncated", "terminal"))):
                raise ValueError("fixed replay action/reward/flag differs from complete step ledger")
        windows.append({"label": label, "episode_id": eid, "start_step": start,
                        "track_id": row["track_id"], "geometry_seed": row["geometry_seed"],
                        "finished_parent": row["finished"], "pixel_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
                        "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                        "pixel_bytes_hex": observations.tobytes().hex(), "action_bytes_hex": actions.tobytes().hex(),
                        "shift_yx": shift, "obs": observations, "actions": actions})
    return windows


def aligned_shift(pixels: torch.Tensor, y: int, x: int) -> torch.Tensor:
    """Explicit replicated integer crop; no CPU/GPU random-shift equivalence assumption."""
    if not isinstance(y, int) or not isinstance(x, int) or max(abs(y), abs(x)) > 3:
        raise ValueError("offset must lie in the frozen pad-3 range")
    image = F.pad(pixels.float(), (3, 3, 3, 3), mode="replicate")
    return image[..., 3 + y:3 + y + 64, 3 + x:3 + x + 64]


@contextmanager
def _shift_fixture(model, shift):
    layer = model._encoder["rgb"][0]
    if layer.__class__.__name__ != "ShiftAug" or layer.pad != 3:
        raise ValueError("expected frozen ShiftAug at encoder entry")
    original = layer.forward
    layer.forward = lambda pixels: aligned_shift(pixels, *shift)
    try:
        yield
    finally:
        layer.forward = original


def _forward(model, window: dict, device: str) -> dict[str, np.ndarray]:
    from haic.algorithms.tdmpc2.model import two_hot_inv

    obs = torch.from_numpy(window["obs"]).to(device)
    actions = torch.from_numpy(window["actions"]).to(device)
    with _shift_fixture(model, window["shift_yx"]), torch.inference_mode():
        z = model.encode(obs, None)
        pred = model.next(z[:-1], actions, None)
        actor = torch.tanh(model._pi(z[:-1]).chunk(2, dim=-1)[0])
        reward = model.reward(z[:-1], actions, None)
        qs = model.Q(z[:-1], actions, None, return_type="all")
        terminal = model.termination(pred, None, unnormalized=True)
        outputs = {"encoded_latent": z, "predicted_next_latent": pred,
                   "tanh_actor_prior_mean": actor, "reward_logits": reward,
                   "decoded_reward": two_hot_inv(reward, model.cfg),
                   "q_logits_all_five": qs, "decoded_q_all_five": two_hot_inv(qs, model.cfg),
                   "terminal_logits": terminal, "terminal_probability": torch.sigmoid(terminal)}
        result = {name: value.detach().cpu().numpy().copy() for name, value in outputs.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("nonfinite frozen-model output")
    return result


def difference_stats(left: np.ndarray, right: np.ndarray, tolerance: float) -> dict:
    if left.shape != right.shape or left.size == 0 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("comparison requires nonempty finite aligned tensors")
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    l2 = np.linalg.norm(diff.reshape(-1, left.shape[-1]), axis=-1)
    return {"elements": int(diff.size), "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()), "vector_l2_max": float(l2.max()),
            "vector_l2_mean": float(l2.mean()), "tolerance_abs": tolerance,
            "within_tolerance": bool(np.all(diff <= tolerance))}


def _paired_plan(planner, obs: torch.Tensor, *, seed: int = 4242) -> dict:
    """On ONE device replay identical pre-plan RNG/warm start; capture final elite."""
    if obs.device.type != "cpu" or type(seed) is not int:
        raise ValueError("planner mode pair uses only fixed CPU observation/seed")
    previous = planner.prev_mean.clone()
    original = planner._estimate_value
    reports = []
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(seed)
            before = torch.get_rng_state().clone()
            for eval_mode in (True, False):
                planner.prev_mean = previous.clone()
                torch.set_rng_state(before.clone())
                captured = {}

                def capture(z, actions):
                    values = original(z, actions)
                    captured.update(actions=actions.detach().clone(), values=values.detach().clone(),
                                    rng=torch.get_rng_state().clone())
                    return values

                planner._estimate_value = capture
                final = planner.plan(obs, t0=False, eval_mode=eval_mode).detach().clone()
                planner._estimate_value = original
                if not captured:
                    raise ValueError("planner returned without scoring trajectories")
                cfg = planner.config
                elite_value, indices = captured["values"].squeeze(-1).topk(cfg.num_elites)
                elites = captured["actions"][:, indices]
                score = torch.exp(cfg.temperature * (elite_value - elite_value.max()))
                score = score / score.sum()
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(captured["rng"])
                    gumbel = -torch.empty_like(score).exponential_().log()
                    selected_index = (score.log() + gumbel).argmax()
                    selected = elites[0, selected_index].clone()
                    std = ((score[None, :, None] * (elites - planner.prev_mean[:, None]).square()).sum(dim=1)
                           / (score.sum() + 1e-9)).sqrt().clamp(cfg.min_std, cfg.max_std)
                    noise = torch.randn(cfg.action_dim) if not eval_mode else None
                expected = selected if eval_mode else (selected + std[0] * noise).clamp(-1, 1)
                if not torch.equal(final, expected):
                    raise ValueError("mode pair differs before/beyond the final Gaussian")
                reports.append({"action": final, "elite": selected, "index": int(selected_index),
                                "prev_mean": planner.prev_mean.detach().clone()})
    finally:
        planner._estimate_value = original
        planner.prev_mean = previous
    eval_row, train_row = reports
    diff = (eval_row["action"] - train_row["action"]).float()
    return {"device": "cpu", "identical_initial_rng_and_prev_mean": True,
            "selected_elite_bit_identical": bool(torch.equal(eval_row["elite"], train_row["elite"])),
            "selected_elite_index_identical": eval_row["index"] == train_row["index"],
            "prev_mean_bit_identical": bool(torch.equal(eval_row["prev_mean"], train_row["prev_mean"])),
            "final_action_bit_identical": bool(torch.equal(eval_row["action"], train_row["action"])),
            "final_action_delta_l2": float(torch.linalg.vector_norm(diff)),
            "eval_action": eval_row["action"].tolist(), "training_action": train_row["action"].tolist(),
            "final_additive_noise_only_verified": True}


def score(*, root: Path = ROOT, planner_mode_pair: bool = False) -> dict:
    bound = _bound_files(root)
    # No loader or model import is reached until every source and ledger is sealed.
    has_gpu = bool(torch.cuda.is_available())
    devices = [torch.cuda.current_device()] if has_gpu else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(73301)
        state = _load_checkpoint(bound)
        windows = _windows(state, bound)
        from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

        weights = {key[6:]: value for key, value in state["learner"].items() if key.startswith("model.")}
        cfg = TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True)
        cpu = WorldModel(cfg).eval()
        cpu.load_state_dict(weights, strict=True)
        gpu = None
        if has_gpu:
            gpu = WorldModel(cfg).eval()
            gpu.load_state_dict(weights, strict=True)
            gpu.to("cuda")
            if not all(torch.equal(value, gpu.state_dict()[key].cpu()) for key, value in cpu.state_dict().items()):
                raise ValueError("CPU/GPU weights differ before inference")
        metadata = [{key: value for key, value in w.items() if key not in ("obs", "actions", "pixel_bytes_hex", "action_bytes_hex")}
                    for w in windows]
        # Exact bytes are SHA-bound in the checkpoint; emit bounded action bytes,
        # not 4 x 64KiB pixel hex strings, alongside per-window pixel digests.
        for row, window in zip(metadata, windows):
            row["action_bytes_hex"] = window["action_bytes_hex"]
        reports = []
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            for window in windows:
                base = _forward(cpu, window, "cpu")
                if gpu is not None:
                    other = _forward(gpu, window, "cuda")
                    metrics = {key: difference_stats(base[key], other[key], TOLERANCES[key]) for key in TOLERANCES}
                    reports.append({"window": window["label"], "metrics": metrics,
                                    "within_all_tolerances": all(v["within_tolerance"] for v in metrics.values())})
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
        paired = None
        if planner_mode_pair:
            from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner

            planner = TDMPC2Planner(cpu, PlannerConfig(action_dim=3, discount=.995, episodic=True))
            with _shift_fixture(cpu, windows[-1]["shift_yx"]):
                paired = _paired_plan(planner, torch.from_numpy(windows[-1]["obs"][:1].copy()))
    aggregate = None
    if has_gpu:
        aggregate = {}
        for name in TOLERANCES:
            values = [row["metrics"][name] for row in reports]
            aggregate[name] = {"elements": sum(v["elements"] for v in values),
                               "max_abs": max(v["max_abs"] for v in values),
                               "mean_abs": sum(v["mean_abs"] * v["elements"] for v in values)
                               / sum(v["elements"] for v in values),
                               "tolerance_abs": TOLERANCES[name],
                               "within_tolerance": all(v["within_tolerance"] for v in values)}
    return {"scope": "offline_archived_consumed_TRAIN_only", "environment_resets": 0,
            "optimizer_steps": 0, "protocol_sha256": PROTOCOL_SHA, "result_sha256": RESULT_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA, "training_ledger_sha256": TRAIN_SHA,
            "step_ledger_sha256": STEP_SHA, "binder_sha256": BINDER_SHA,
            "source_sha256": bound["protocol"]["source_sha256"], "decisions": DECISIONS,
            "completed_episodes": EPISODES, "windows": metadata, "torch_version": str(torch.__version__),
            "augmentation_fixture": "aligned replicated integer crop, fixed (y,x) per window; replaces stochastic ShiftAug grid_sample during scoring only; not a historical RNG trace",
            "tolerances": TOLERANCES, "cpu_gpu": {
                "status": "scored" if has_gpu else "unavailable_no_parity_proof",
                "gpu_name": torch.cuda.get_device_name() if has_gpu else None,
                "per_window": reports, "aggregate": aggregate,
                "within_all_tolerances": all(r["within_all_tolerances"] for r in reports) if has_gpu else None},
            "planner_eval_vs_training_one_device": paired,
            "limitations": "Fixed archived TRAIN states only; CPU full evaluation has action hashes, not action/pixel failure bytes. This cannot attribute its 0/8 finishes to device, noise, augmentation or damage."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-mode-pair", action="store_true", help="also compare same-RNG CPU eval/train MPPI modes")
    args = parser.parse_args()
    print(json.dumps(score(planner_mode_pair=args.planner_mode_pair), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
