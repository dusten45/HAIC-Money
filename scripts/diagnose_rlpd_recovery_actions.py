"""Zero-reset action retention/Q-rank diagnosis on privileged recovery TRAIN rows.

Only trusted local learner checkpoints may be supplied: torch.load uses pickle.
This is not a fresh generalization test, failure-causal test, or training run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
import torch

from haic.algorithms.rlpd.agent import RLPDConfig
from haic.algorithms.rlpd.model import PixelActor, PixelCritic
from haic.algorithms.rlpd.recovery import (
    FINISH_POLICY, file_sha256, frame_stack, load_recovery_replay,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-rlpd-recovery-action-diagnosis-v1"
BATCH_SIZE = 64
SOURCE_FILES = (
    "scripts/diagnose_rlpd_recovery_actions.py", "common_adapter.py",
    "haic/algorithms/rlpd/agent.py", "haic/algorithms/rlpd/model.py",
    "haic/algorithms/rlpd/augment.py", "haic/algorithms/rlpd/recovery.py",
    "haic/algorithms/rlpd/replay.py",
)
LIMITATIONS = (
    "Targets are actually executed, privileged teacher-selected paired-finish TRAIN actions; "
    "Oracle rows use privileged feedback, actor-handoff rows use the original actor. "
    "Action disagreement is not failure-causal evidence. Current-critic absolute Q values "
    "are not cross-critic causal comparisons. Within-critic target-minus-actor gaps test "
    "ranking only, not expert optimality. Repeated/selected rows are not fresh generalization. "
    "Lost-action retention and unchanged expert-Q-rank are hypotheses, not assertions."
)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rows(dataset: Path, manifest_sha256: str) -> tuple[dict, dict, dict]:
    if dataset.is_symlink():
        raise ValueError("symlinked dataset")
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("invalid manifest SHA256")
    cells = [{"track_id": 1, "geometry_seed": seed, "partition": "TRAIN", "obstacles": True}
             for seed in range(4272000001, 4272000013)]
    replay, receipt = load_recovery_replay(
        dataset, manifest_sha256=manifest_sha256, allowed_cells=cells, seed=0,
        required_policy=FINISH_POLICY,
    )
    del replay
    if file_sha256(dataset / "manifest.json") != manifest_sha256:
        raise ValueError("manifest changed after validation")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("dataset is not complete")
    rows = {key: [] for key in ("observations", "target_native", "target_official",
                                "episode_id", "step", "geometry_seed", "stratum", "role")}
    hashes = {}
    windows = 0
    for episode in manifest["episodes"]:
        path = dataset / episode["path"]
        # Hash every episode, including unmasked negative artifacts, before loading.
        hashes[episode["path"]] = file_sha256(path)
        if hashes[episode["path"]] != episode["sha256"]:
            raise ValueError("source NPZ changed after validation")
        with np.load(path, allow_pickle=False) as data:
            selected = np.flatnonzero(data["recovery_mask"])
            windows += bool(len(selected))
            for step in selected:
                rows["observations"].append(frame_stack(data["frames"], int(step)))
                rows["target_native"].append(data["executed_action"][step].copy())
                rows["target_official"].append(data["applied_action"][step].copy())
                for name in ("episode_id", "geometry_seed", "stratum"):
                    rows[name].append(episode[name])
                rows["step"].append(int(step))
                rows["role"].append(str(data["role"][step]))
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    if arrays["observations"].shape != (receipt["accepted_transitions"], 4, 84, 84):
        raise ValueError("masked stack count/shape mismatch")
    if not np.isin(arrays["role"], ["oracle", "actor"]).all():
        raise ValueError("unexpected masked role")
    counts = {"rows": len(arrays["step"]), "windows": windows,
              "failure": int((arrays["stratum"] == "failure").sum()),
              "finish_control": int((arrays["stratum"] == "finish-control").sum())}
    unique = {hashlib.sha256(obs.tobytes() + act.tobytes()).hexdigest()
              for obs, act in zip(arrays["observations"], arrays["target_native"])}
    counts["unique_rows"] = len(unique)
    if (counts["rows"] != manifest["counts"]["accepted_transitions"]
            or len(unique) != receipt["unique_accepted_transitions"]):
        raise ValueError("masked row identity counts disagree")
    if (file_sha256(dataset / "manifest.json") != manifest_sha256
            or any(file_sha256(dataset / name) != digest for name, digest in hashes.items())):
        raise ValueError("dataset changed during masked row reconstruction")
    return arrays, {"receipt": receipt, "counts": counts, "npz_sha256": hashes}, manifest


def load_models(path: Path, expected_sha256: str):
    if file_sha256(path) != expected_sha256:
        raise ValueError("checkpoint changed before trusted load")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(state, dict)
            or state.get("format") != "haic-rlpd-training-checkpoint-v1"
            or state.get("config") != asdict(RLPDConfig())):
        raise ValueError("checkpoint format/config differs from pinned V5 architecture")
    actor, critic = PixelActor(), PixelCritic(num_qs=10)
    actor.load_state_dict(state["actor"], strict=True)
    critic.load_state_dict(state["critic"], strict=True)
    metadata = {"config": state["config"], "environment_steps": state.get("environment_steps"),
                "gradient_steps": state.get("gradient_steps"), "critic": "current-not-target"}
    del state
    for model in (actor, critic):
        model.eval().requires_grad_(False)
    return actor, critic, metadata


def predict(rows: dict, actor, critic) -> dict:
    """Use uint8 exact source stacks; PixelEncoder performs the sole normalization."""
    values = {name: [] for name in ("actor_native", "q_actor_heads", "q_target_heads")}
    with torch.inference_mode():
        for start in range(0, len(rows["observations"]), BATCH_SIZE):
            obs = torch.from_numpy(rows["observations"][start:start + BATCH_SIZE])
            target = torch.from_numpy(rows["target_native"][start:start + BATCH_SIZE])
            action, _, _ = actor.sample(obs, deterministic=True)
            q_actor, q_target = critic(obs, action), critic(obs, target)
            if (action.shape != target.shape or q_actor.ndim != 2
                    or q_actor.shape != q_target.shape or q_actor.shape[0] != len(obs)):
                raise ValueError("invalid actor/critic output shapes")
            for name, tensor in zip(values, (action, q_actor, q_target)):
                # Independent clones: no retained model tensor views in audit arrays.
                array = tensor.detach().cpu().numpy().copy()
                if not np.isfinite(array).all():
                    raise ValueError("nonfinite actor/critic prediction")
                values[name].append(array)
    result = {name: np.concatenate(chunks) for name, chunks in values.items()}
    if (np.abs(result["actor_native"]) > 1).any():
        raise ValueError("actor native action out of bounds")
    official = result["actor_native"].copy()
    official[:, 1:] = (official[:, 1:] + 1) / 2
    result["actor_official"] = official
    for action in ("actor", "target"):
        heads = result[f"q_{action}_heads"]
        for statistic in ("mean", "min", "variance"):
            method = {"mean": np.mean, "min": np.min, "variance": np.var}[statistic]
            result[f"q_{action}_{statistic}"] = method(heads, axis=1)
    for statistic in ("mean", "min"):
        result[f"q_gap_{statistic}"] = result[f"q_target_{statistic}"] - result[f"q_actor_{statistic}"]
    return result


def summarize(rows: dict, predictions: dict) -> dict:
    groups = {}
    for stratum in ("all", "failure", "finish-control"):
        for role in ("all", "oracle", "actor"):
            mask = np.ones(len(rows["step"]), dtype=bool)
            if stratum != "all":
                mask &= rows["stratum"] == stratum
            if role != "all":
                mask &= rows["role"] == role
            n = int(mask.sum())
            report: dict = {"rows": n}
            if n:
                for space in ("native", "official"):
                    actor, target = predictions[f"actor_{space}"][mask], rows[f"target_{space}"][mask]
                    error = actor - target
                    strong = np.abs(target[:, 0]) >= .25
                    report[f"action_{space}"] = {
                        "steering_mae": float(np.abs(error[:, 0]).mean()),
                        "gas_signed_error_actor_minus_target": float(error[:, 1].mean()),
                        "brake_signed_error_actor_minus_target": float(error[:, 2].mean()),
                        "joint_mse": float(np.square(error).mean()),
                        "steering_opposition_denominator": int(strong.sum()),
                        "steering_opposition_fraction": float((actor[strong, 0] * target[strong, 0] < 0).mean()) if strong.any() else None,
                    }
                report["q"] = {key: float(value[mask].mean()) for key, value in predictions.items()
                               if key.startswith("q_") and value.ndim == 1}
                for statistic in ("mean", "min"):
                    gap = predictions[f"q_gap_{statistic}"][mask]
                    report["q"][f"target_preferred_fraction_{statistic}"] = float((gap > 0).mean())
                    report["q"][f"tie_fraction_{statistic}"] = float((gap == 0).mean())
            groups[f"{stratum}/{role}"] = report
    return groups


def run(dataset: Path, manifest_sha256: str, checkpoints: list[str], output: str,
        *, root: Path = ROOT, model_provider=load_models) -> dict:
    if (re.fullmatch(r"runs/[a-z0-9][a-z0-9-]*", output) is None
            or (root / "runs").is_symlink() or not (root / "runs").is_dir()):
        raise ValueError("output must be exclusive runs/<new-slug>")
    target = root / output
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    specs = {}
    for item in checkpoints:
        name, separator, raw_path = item.partition("=")
        if not separator or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None or name in specs:
            raise ValueError("require unique checkpoint name=PATH entries")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint must be a trusted local nonsymlink file")
        specs[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if not specs:
        raise ValueError("no checkpoints supplied")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    rows, evidence, _ = load_rows(dataset, manifest_sha256)
    source_hashes = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    provenance = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.resolve()), "manifest_sha256": manifest_sha256,
        "dataset_evidence": evidence, "checkpoints": specs, "source_sha256": source_hashes,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "torch": torch.__version__, "device": "cpu",
                    "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
        "batch_size": BATCH_SIZE, "environment_resets": 0, "learner_updates": 0,
        "eligibility_policy": FINISH_POLICY, "observation_shape": [4, 84, 84],
        "normalization": "uint8 exact frame_stack -> PixelEncoder.normalize /255; no augmentation",
        "learner_config_pin": asdict(RLPDConfig()), "limitations": LIMITATIONS,
    }
    target.mkdir()
    write_json(target / "provenance.json", provenance)
    try:
        audit = {key: value.copy() for key, value in rows.items() if key != "observations"}
        audit["observation_sha256"] = np.asarray([hashlib.sha256(obs.tobytes()).hexdigest() for obs in rows["observations"]])
        summary = {**provenance, "models": {}}
        for name, spec in specs.items():
            actor, critic, metadata = model_provider(Path(spec["path"]), spec["sha256"])
            prediction = predict(rows, actor, critic)
            summary["models"][name] = {"checkpoint": spec, "metadata": metadata,
                                       "groups": summarize(rows, prediction)}
            audit.update({f"{name}__{key}": value.copy() for key, value in prediction.items()})
            del actor, critic
        if (file_sha256(dataset / "manifest.json") != manifest_sha256
                or any(file_sha256(dataset / name) != digest for name, digest in evidence["npz_sha256"].items())
                or any(file_sha256(Path(spec["path"])) != spec["sha256"] for spec in specs.values())
                or any(file_sha256(ROOT / name) != digest for name, digest in source_hashes.items())):
            raise ValueError("inputs/source changed during diagnosis")
        with (target / "predictions.npz").open("xb") as handle:
            np.savez_compressed(handle, **audit)
            handle.flush()
            os.fsync(handle.fileno())
        summary["audit"] = {"path": "predictions.npz", "sha256": file_sha256(target / "predictions.npz")}
        write_json(target / "summary.json", summary)
        return summary
    except BaseException as error:
        write_json(target / "failure.json", {"status": "unsealed-diagnosis", "error_type": type(error).__name__,
                                             "error": str(error), "environment_resets": 0, "learner_updates": 0})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="trusted local learner checkpoints name=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.dataset, args.manifest_sha256, args.checkpoints, args.output)
        print(json.dumps({"output": args.output, "audit": result["audit"], "environment_resets": 0}, sort_keys=True))
        return 0
    except Exception as error:
        print(f"recovery diagnosis failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
