"""Offline, reused-TRAIN visual probe; no simulator, policy update or held-out claim."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec, ObservationSpec


ROOT = Path(__file__).resolve().parents[1]
SOURCE = {
    "protocol_path": "experiments/rlpd-g0-completion-v1.json",
    "protocol_sha256": "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b",
    "manifest_path": "runs/20260926-rlpd-g0-completion-v1/manifest.json",
    "manifest_sha256": "52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358",
    "cells_path": "runs/20260926-rlpd-g0-completion-v1/cells.jsonl",
    "cells_sha256": "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559",
}
ACTOR = {
    "id": "long-horizon-seed11",
    "path": "runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt",
    "sha256": "f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1",
    "source_sha256": "cf92e27db8e40eae3ca2c629f7db3c5bf833cb0b5d83a9621d1885b482b324e3",
    "export_protocol_sha256": "2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329",
}
CODE_FILES = (
    "scripts/probe_rlpd_visual_representation.py", "agent.py",
    "common_adapter.py", "haic/algorithms/rlpd/model.py",
    "scripts/__init__.py", "haic/__init__.py", "haic/algorithms/__init__.py",
    "requirements.txt",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ACTION_KEYS = (
    "proposed_native_action", "executed_native_action",
    "commanded_official_action", "raw_official_action",
)
BATCH = 256
BURN_IN = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unversioned-source-hashes-authoritative"


def protocol_template(*, source: dict = SOURCE, actor: dict = ACTOR,
                      code_root: Path = ROOT,
                      freeze_path: str = "experiments/rlpd-visual-representation-probe-v2.json",
                      output_root: str = "runs/20260926-rlpd-visual-representation-probe-v2") -> dict:
    """Printable proposal only: caller must separately review, save and hash a freeze."""
    return {
        "format": "haic-rlpd-visual-representation-probe-v1",
        "status": "frozen",
        "role": "reused-TRAIN-retrospective-diagnostic-not-generalization",
        "source": dict(source),
        "actor": dict(actor),
        "source_hashes": {name: sha256_file(code_root / name) for name in CODE_FILES},
        "execution": {
            "protocol_path": freeze_path,
            "output_root": output_root,
            "command": "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "
                       "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                       "python -B -m scripts.probe_rlpd_visual_representation "
                       f"--protocol {freeze_path} --protocol-sha256 <FROZEN_FILE_SHA256> "
                       f"--output-root {output_root}",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": "cpu", "max_torch_threads": 1,
            "source_revision": _source_revision(code_root),
            "working_tree_source_hashes_authoritative": True,
        },
        "split": {"fit_geometry_indices": list(range(1, 9)),
                  "diagnostic_geometry_indices": list(range(9, 13)),
                  "unit": "both actors on each G0 road geometry stay together"},
        "label": {"definition": "pre-decision speed_m_s[j-1] <= 2 for j>=20; exclude j<20",
                  "role": "retrospective-telemetry-target-only", "speed_m_s_max": 2.0,
                  "burn_in_decisions": BURN_IN},
        "coverage": {"min_positive_fit_geometries": 2,
                     "min_positive_diagnostic_geometries": 1,
                     "min_fit_decisions": BATCH, "require_both_fit_classes": True},
        "representation": {"input": "HUD-inclusive pre-action float32 CHW (4,84,84) uint8 stack / 255",
                           "module": "Agent.model.encoder", "feature_dim": 50,
                           "device": "cpu", "max_stack_batch": BATCH},
        "action_parity_atol": {"batched_native_vs_recorded": 1e-5,
                               "root_agent_official_vs_recorded": 1e-6},
        "head": {"type": "torch.nn.Linear(50,1)", "seed": 20260926,
                 "optimizer": "Adam", "lr": 0.001, "weight_decay": 0.0001,
                 "updates": 256, "batch_size": BATCH, "sampling": "train-decision-uniform-with-replacement",
                 "loss": "BCEWithLogitsLoss",
                 "positive_weight": "min(10,max(1,train_negative_decisions/train_positive_decisions))"},
        "report": {"cutoff": 0.5, "cutoff_rule": "sigmoid(logit)>=0.5",
                   "baseline": "constant fit-decision positive prevalence",
                   "diagnostic": "retrospective, previously viewed TRAIN roads 09..12"},
    }


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                       parse_constant=_bad_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file(root: Path, relative: str, prefix: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("unsafe frozen path")
    name = Path(relative)
    if (name.is_absolute() or len(name.parts) < 2 or name.parts[0] != prefix
            or name.as_posix() != relative or any(part in (".", "..") for part in name.parts)):
        raise ValueError(f"input must be a repository-relative {prefix}/ file")
    path = root
    for part in name.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"symlink in frozen input: {relative}")
    if not path.is_file():
        raise ValueError(f"missing frozen input: {relative}")
    return path


def _pinned(root: Path, relative: str, sha: str, prefix: str) -> Path:
    if not isinstance(sha, str) or SHA256.fullmatch(sha) is None:
        raise ValueError("invalid frozen SHA-256")
    path = _file(root, relative, prefix)
    if sha256_file(path) != sha:
        raise ValueError(f"frozen hash mismatch: {relative}")
    return path


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if value.device.type != "cpu" or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"actor parameter is non-CPU or nonfinite: {name}")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _inputs(root: Path, freeze_path: str, freeze_sha: str, output_root: str,
            *, source: dict, actor_info: dict, code_root: Path) -> tuple[dict, list[dict], list[str]]:
    name = Path(output_root) if isinstance(output_root, str) else Path()
    if (not isinstance(output_root, str) or "\\" in output_root or name.is_absolute()
            or len(name.parts) != 2 or name.parts[0] != "runs" or name.as_posix() != output_root
            or name.parts[1] in (".", "..") or (root / "runs").is_symlink()
            or not (root / "runs").is_dir() or (root / name).exists() or (root / name).is_symlink()):
        raise ValueError("output must be a new exclusive runs/<directory>")
    frozen = _json(_pinned(root, freeze_path, freeze_sha, "experiments"))
    expected = protocol_template(source=source, actor=actor_info, code_root=code_root,
                                 freeze_path=freeze_path, output_root=output_root)
    execution = frozen.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("probe execution contract is missing")
    recorded_revision = execution.get("source_revision")
    if (not isinstance(recorded_revision, str)
            or (re.fullmatch(r"[0-9a-f]{40}", recorded_revision) is None
                and recorded_revision != "unversioned-source-hashes-authoritative")):
        raise ValueError("probe source revision provenance is invalid")
    expected["execution"]["source_revision"] = recorded_revision
    if json.dumps(frozen, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("probe protocol differs from exact source-hashed template")
    original = _json(_pinned(root, source["protocol_path"], source["protocol_sha256"], "experiments"))
    manifest = _json(_pinned(root, source["manifest_path"], source["manifest_sha256"], "runs"))
    ledger = _pinned(root, source["cells_path"], source["cells_sha256"], "runs")
    if (original.get("format") != "haic-rlpd-g0-diagnostic-v1" or original.get("status") != "frozen"
            or original.get("partition") != "TRAIN" or original.get("frame_skip") != 4
            or original.get("interventions") is not False or original.get("learner_updates") != 0
            or manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
            or manifest.get("role") != "TRAIN-only-failure-diagnostic" or manifest.get("ranked") is not False
            or manifest.get("protocol_path") != source["protocol_path"]
            or manifest.get("protocol_sha256") != source["protocol_sha256"]
            or manifest.get("cells_sha256") != source["cells_sha256"]
            or manifest.get("cell_count") != 24 or manifest.get("geometry_count") != 12
            or manifest.get("geometry_audit_sha256") != original.get("geometry_audit_sha256")):
        raise ValueError("original G0 protocol/manifest identity differs")
    cells, actors = original.get("cells"), original.get("actors")
    if (not isinstance(cells, list) or len(cells) != 12 or not isinstance(actors, list) or len(actors) != 2
            or not all(isinstance(cell, dict) and cell.get("partition") == "TRAIN"
                       and cell.get("obstacles") is True and type(cell.get("track_id")) is int
                       and type(cell.get("geometry_seed")) is int for cell in cells)
            or len({(c["track_id"], c["geometry_seed"]) for c in cells}) != 12
            or not all(isinstance(a, dict) and a.get("action_mode") == "exported_tanh_mean"
                       for a in actors)
            or len({a.get("id") for a in actors}) != 2
            or len({a.get("sha256") for a in actors}) != 2
            or any(actors[0].get(key) != value for key, value in actor_info.items())):
        raise ValueError("G0 paired-cell or designated seed-11 actor identity differs")
    rows = [json.loads(line, object_pairs_hook=_unique, parse_constant=_bad_constant)
            for line in ledger.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 24 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("G0 ledger must contain exactly 24 complete rows")
    traces = []
    for index, row in enumerate(rows):
        cell, actor = cells[index // 2], actors[index % 2]
        relative = (Path(source["manifest_path"]).parent / "traces" /
                    f"seed-{cell['geometry_seed']}-track-{cell['track_id']}-{actor['id']}.npz").as_posix()
        if (row.get("partition") != "TRAIN" or row.get("track_id") != cell["track_id"]
                or row.get("geometry_seed") != cell["geometry_seed"]
                or row.get("actor_id") != actor["id"] or row.get("actor_sha256") != actor.get("sha256")
                or row.get("trace_path") != Path(relative).relative_to(Path(source["manifest_path"]).parent).as_posix()
                or type(row.get("steps")) is not int or not 2 <= row["steps"] <= 2000
                or not isinstance(row.get("summary"), dict)
                or row["summary"].get("outcome") not in {
                    "finished", "off_track", "crash", "out_of_bounds", "task_timeout", "unknown"
                } or not isinstance(row.get("road_centerline_sha256"), str)
                or SHA256.fullmatch(row["road_centerline_sha256"]) is None):
            raise ValueError("G0 ledger differs from frozen paired-cell schedule")
        if index % 2 and row["road_centerline_sha256"] != rows[index - 1]["road_centerline_sha256"]:
            raise ValueError("paired actors have different road geometry")
        traces.append(relative)
        _pinned(root, relative, row.get("trace_sha256"), "runs")
    if (len(set(traces)) != 24 or sum(row["steps"] for row in rows) != manifest.get("decisions_spent")
            or sum(row["summary"]["outcome"] == "finished" for row in rows) != 6
            or any(sum(row["summary"]["outcome"] == "finished" for row in rows[side::2]) != 3
                   for side in range(2))):
        raise ValueError("G0 ledger decision or six finished-control counts disagree")
    return frozen, rows, traces


def _trace_arrays(path: Path, steps: int) -> dict[str, np.ndarray]:
    # np.load is lazy; only one <=2000-decision archive is open at a time.
    with np.load(path, allow_pickle=False) as archive:
        required = {"initial_stack", "observation_frame", "next_frame", "speed_m_s",
                    "finished", "terminated", "truncated", *ACTION_KEYS}
        if len(archive.files) != len(set(archive.files)) or not required.issubset(archive.files):
            raise ValueError("G0 trace missing or duplicate columns")
        arrays = {key: archive[key] for key in required}
    if (arrays["initial_stack"].dtype != np.uint8 or arrays["initial_stack"].shape != (4, 84, 84)
            or any(arrays[key].dtype != np.uint8 or arrays[key].shape != (steps, 84, 84)
                   for key in ("observation_frame", "next_frame"))):
        raise ValueError("G0 trace has malformed pixel stacks")
    speed = arrays["speed_m_s"]
    if speed.shape != (steps,) or speed.dtype.kind != "f" or not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError("G0 trace has invalid speed labels")
    for key in ACTION_KEYS:
        values = arrays[key]
        if values.shape != (steps, 3) or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError(f"G0 action stream is invalid: {key}")
    for key in ("finished", "terminated", "truncated"):
        if arrays[key].shape != (steps,) or arrays[key].dtype.kind != "b":
            raise ValueError(f"G0 terminal stream is invalid: {key}")
    if (np.any(arrays["finished"][:-1] | arrays["terminated"][:-1] | arrays["truncated"][:-1])
            or not (arrays["terminated"][-1] or arrays["truncated"][-1])):
        raise ValueError("G0 trace ends without a terminal decision")
    return arrays


def _coverage(root: Path, rows: list[dict], traces: list[str]) -> dict:
    counts = {"fit": [], "diagnostic": []}
    per_episode = []
    for index, (row, relative) in enumerate(zip(rows, traces)):
        path = _pinned(root, relative, row["trace_sha256"], "runs")
        with np.load(path, allow_pickle=False) as archive:
            speed = archive["speed_m_s"]
        if (speed.shape != (row["steps"],) or speed.dtype.kind != "f"
                or not np.isfinite(speed).all() or np.any(speed < 0)):
            raise ValueError("malformed speed labels in coverage preflight")
        group = "fit" if index // 2 < 8 else "diagnostic"
        positives = int(np.count_nonzero(speed[BURN_IN - 1:-1] <= 2))
        counts[group].append(positives)
        per_episode.append({
            "geometry_seed": row["geometry_seed"], "actor_id": row["actor_id"],
            "split": group, "outcome": row["summary"]["outcome"],
            "eligible_decisions": max(0, row["steps"] - BURN_IN),
            "positive_decisions": positives,
        })
    fit, diagnostic = counts["fit"], counts["diagnostic"]
    if (sum(any(fit[i:i + 2]) for i in range(0, 16, 2)) < 2
            or sum(any(diagnostic[i:i + 2]) for i in range(0, 8, 2)) < 1):
        raise ValueError("insufficient positive geometry coverage: zero optimizer steps/output")
    eligible_fit = sum(max(0, row["steps"] - BURN_IN) for row in rows[:16])
    if eligible_fit < BATCH or sum(fit) == 0 or sum(fit) == eligible_fit:
        raise ValueError("fit requires >=256 decisions and both classes: zero optimizer steps/output")
    return {"fit_positive_geometries": sum(any(fit[i:i + 2]) for i in range(0, 16, 2)),
            "diagnostic_positive_geometries": sum(any(diagnostic[i:i + 2]) for i in range(0, 8, 2)),
            "per_episode": per_episode}


def _extract_episode(actor, row: dict, arrays: dict[str, np.ndarray], feature_digest,
                     designated_actor_id: str) -> np.ndarray:
    steps = row["steps"]
    adapter = ActionAdapter(ActionSpec())
    if (not np.allclose(arrays["proposed_native_action"], arrays["executed_native_action"], atol=1e-6, rtol=0)
            or not np.allclose(arrays["commanded_official_action"], arrays["raw_official_action"], atol=1e-6, rtol=0)
            or bool(arrays["finished"][-1]) != (row["summary"]["outcome"] == "finished")):
        raise ValueError("G0 executed action or terminal control parity failed")
    official = np.stack([adapter.to_official(value, clip=False)
                         for value in arrays["executed_native_action"]])
    if not np.allclose(official, arrays["commanded_official_action"], atol=1e-6, rtol=0):
        raise ValueError("G0 native/official action parity failed")
    stack = arrays["initial_stack"].copy()
    features = np.empty((max(0, steps - BURN_IN), 50), dtype=np.float32)
    for start in range(0, steps, BATCH):
        stop = min(start + BATCH, steps)
        pixels = np.empty((stop - start, 4, 84, 84), dtype=np.float32)
        for j in range(start, stop):
            if not np.array_equal(stack[-1], arrays["observation_frame"][j]):
                raise ValueError("G0 pre-action stack history disagrees")
            pixels[j - start] = stack.astype(np.float32) / 255.0
            stack[:-1] = stack[1:].copy()
            stack[-1] = arrays["next_frame"][j]
            if j + 1 < steps and not np.array_equal(stack[-1], arrays["observation_frame"][j + 1]):
                raise ValueError("G0 next frame does not precede next action")
        tensor = torch.from_numpy(pixels)
        with torch.no_grad():
            native = actor.model(tensor).numpy()
            latent = actor.model.encoder(tensor).numpy()
        if native.shape != (stop - start, 3) or latent.shape != (stop - start, 50) or (
                not np.isfinite(native).all() or not np.isfinite(latent).all()):
            raise ValueError("frozen actor emitted invalid action or 50-d features")
        if row["actor_id"] == designated_actor_id:
            if not np.allclose(native, arrays["proposed_native_action"][start:stop], atol=1e-5, rtol=0):
                raise ValueError("seed-11 actor action differs from G0 trace")
            for offset, sample in enumerate(pixels):
                agent_action = np.asarray(actor.act(sample))
                if (agent_action.shape != (3,) or agent_action.dtype != np.float32
                        or not np.allclose(agent_action, official[start + offset], atol=1e-6, rtol=0)):
                    raise ValueError("root Agent action parity differs from G0 trace")
        if stop > BURN_IN:
            first = max(BURN_IN, start)
            block = np.ascontiguousarray(latent[first - start:stop], dtype="<f4")
            features[first - BURN_IN:stop - BURN_IN] = block
            feature_digest.update(block.tobytes())
    return features


def _fit_head(features: np.ndarray, labels: np.ndarray,
              progress: dict | None = None) -> tuple[torch.nn.Linear, float]:
    positive = int(labels.sum())
    weight = min(10.0, max(1.0, (len(labels) - positive) / positive))
    torch.manual_seed(20260926)
    head = torch.nn.Linear(50, 1, device="cpu", dtype=torch.float32)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels.astype(np.float32))
    generator = torch.Generator(device="cpu").manual_seed(20260926)
    for _ in range(256):
        sample = torch.randint(len(labels), (BATCH,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(x[sample]).squeeze(-1), y[sample])
        loss.backward()
        optimizer.step()
        if progress is not None:
            progress["updates_completed"] += 1
    head.eval()
    return head, weight


def _recheck(root: Path, freeze_path: str, freeze_sha: str, rows: list[dict], traces: list[str],
             source: dict, frozen: dict, code_root: Path) -> None:
    _pinned(root, freeze_path, freeze_sha, "experiments")
    for key, prefix in (("protocol", "experiments"), ("manifest", "runs"), ("cells", "runs")):
        _pinned(root, source[f"{key}_path"], source[f"{key}_sha256"], prefix)
    _pinned(root, frozen["actor"]["path"], frozen["actor"]["sha256"], "runs")
    for row, relative in zip(rows, traces):
        _pinned(root, relative, row["trace_sha256"], "runs")
    for name, sha in frozen["source_hashes"].items():
        if sha256_file(code_root / name) != sha:
            raise ValueError(f"probe executable source drifted: {name}")


def preflight(root: Path, freeze_path: str, freeze_sha: str, output_root: str, *,
              source: dict = SOURCE, actor_info: dict = ACTOR, code_root: Path = ROOT) -> dict:
    root = root.resolve()
    frozen, rows, traces = _inputs(root, freeze_path, freeze_sha, output_root,
                                   source=source, actor_info=actor_info, code_root=code_root)
    _recheck(root, freeze_path, freeze_sha, rows, traces, source, frozen, code_root)
    coverage = _coverage(root, rows, traces)
    return {"status": "preflight-only", "protocol_sha256": freeze_sha,
            "source_actor_sha256": frozen["actor"]["sha256"],
            "eligible_fit_decisions": sum(ep["eligible_decisions"] for ep in coverage["per_episode"]
                                          if ep["split"] == "fit"),
            "eligible_diagnostic_decisions": sum(ep["eligible_decisions"] for ep in coverage["per_episode"]
                                                 if ep["split"] == "diagnostic"),
            "coverage": coverage, "optimizer_steps": 0, "output_created": False}


def probe(root: Path, freeze_path: str, freeze_sha: str, output_root: str, *,
          source: dict = SOURCE, actor_info: dict = ACTOR, code_root: Path = ROOT,
          actor_factory=None, fit_head=None) -> dict:
    """Only a separately frozen invocation may fit a diagnostic-only linear head."""
    root = root.resolve()
    torch.set_num_threads(1)
    frozen, rows, traces = _inputs(root, freeze_path, freeze_sha, output_root,
                                   source=source, actor_info=actor_info, code_root=code_root)
    _recheck(root, freeze_path, freeze_sha, rows, traces, source, frozen, code_root)
    coverage = _coverage(root, rows, traces)
    path = _pinned(root, actor_info["path"], actor_info["sha256"], "runs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format") != "haic-rlpd-pixel-actor-v1"
            or payload.get("source_sha256") != actor_info["source_sha256"]
            or payload.get("protocol_sha256") != actor_info["export_protocol_sha256"]
            or payload.get("config") != {"latent_dim": 50}
            or payload.get("observation_spec") != asdict(ObservationSpec())
            or payload.get("action_spec") != asdict(ActionSpec())):
        raise ValueError("seed-11 actor export identity/spec differs from G0")
    if actor_factory is None:
        from agent import Agent
        actor_factory = lambda actor_path: Agent(model_path=str(actor_path))
    actor = actor_factory(path)
    actor.model.cpu().eval().requires_grad_(False)
    before = _state_sha256(actor.model)
    episode_features, episode_labels = [], []
    feature_digest = hashlib.sha256()
    for row, relative in zip(rows, traces):
        arrays = _trace_arrays(_pinned(root, relative, row["trace_sha256"], "runs"), row["steps"])
        feature_digest.update(f"{row['geometry_seed']}:{row['actor_id']}:".encode("ascii"))
        episode_features.append(_extract_episode(actor, row, arrays, feature_digest, actor_info["id"]))
        episode_labels.append(np.asarray(arrays["speed_m_s"][BURN_IN - 1:-1] <= 2, dtype=np.uint8))
        del arrays
    if _state_sha256(actor.model) != before:
        raise ValueError("frozen actor parameters changed during feature extraction")
    _recheck(root, freeze_path, freeze_sha, rows, traces, source, frozen, code_root)
    fit_features = np.concatenate(episode_features[:16], axis=0)
    fit_labels = np.concatenate(episode_labels[:16], axis=0)
    if (coverage["fit_positive_geometries"] < 2 or coverage["diagnostic_positive_geometries"] < 1
            or len(fit_labels) < BATCH or not 0 < fit_labels.sum() < len(fit_labels)):
        raise ValueError("fit coverage drifted: zero optimizer steps/output")
    output = root / output_root
    output.mkdir(exist_ok=False)
    attempt = {
        "format": "haic-rlpd-visual-representation-probe-attempt-v1",
        "status": "reserved-head-only-attempt", "protocol_path": freeze_path,
        "protocol_sha256": freeze_sha, "actor_sha256": actor_info["sha256"],
        "actor_parameters_sha256_before": before, "feature_sha256": feature_digest.hexdigest(),
        "coverage": coverage, "fit_decisions": len(fit_labels),
        "fit_positive_decisions": int(fit_labels.sum()), "head_updates_planned": 256,
        "policy_updated": False, "no_retry_or_relabel": True,
    }
    with (output / "attempt.json").open("x", encoding="utf-8") as stream:
        json.dump(attempt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    progress = {"phase": "fitting_head", "updates_completed": 0}
    try:
        head, weight = (_fit_head(fit_features, fit_labels, progress)
                        if fit_head is None else fit_head(fit_features, fit_labels))
        head.cpu().eval()
        expected_weight = min(10.0, max(1.0, (len(fit_labels) - int(fit_labels.sum())) / int(fit_labels.sum())))
        if (not isinstance(head, torch.nn.Linear) or head.in_features != 50 or head.out_features != 1
                or any(param.device.type != "cpu" or param.dtype != torch.float32
                       or not torch.isfinite(param).all() for param in head.parameters())
                or type(weight) not in (int, float) or weight != expected_weight):
            raise ValueError("head must match frozen float32 CPU Linear(50,1) and positive weight")
        if _state_sha256(actor.model) != before:
            raise ValueError("actor changed after head-only fitting")
        progress["phase"] = "scoring_geometry_groups"
        prevalence = float(fit_labels.mean())
        geometries = []
        for i in range(12):
            episodes = []
            for k in (2 * i, 2 * i + 1):
                with torch.no_grad():
                    logits = head(torch.from_numpy(episode_features[k])).squeeze(-1).numpy()
                if logits.shape != episode_labels[k].shape or not np.isfinite(logits).all():
                    raise ValueError("head emitted invalid diagnostic logits")
                labels = episode_labels[k]
                prediction = logits >= 0.0  # sigmoid(logit) >= 0.5
                negative = labels == 0
                finished = rows[k]["summary"]["outcome"] == "finished"
                episodes.append({
                    "actor_id": rows[k]["actor_id"], "outcome": rows[k]["summary"]["outcome"],
                    "trace_sha256": rows[k]["trace_sha256"],
                    "decision_indices": list(range(BURN_IN, rows[k]["steps"])),
                    "labels": labels.tolist(), "logits": logits.astype(float).tolist(),
                    "positive_count": int(labels.sum()), "decision_count": len(labels),
                    "false_positives_at_0_5": int(np.count_nonzero(prediction & negative)),
                    "finished_control_false_positives_at_0_5": int(np.count_nonzero(prediction & negative)) if finished else None,
                    "finished_control_baseline_false_positives_at_0_5": int(np.count_nonzero(negative))
                    if finished and prevalence >= 0.5 else (0 if finished else None),
                })
            geometries.append({"partition": "fit" if i < 8 else "retrospective_diagnostic",
                               "geometry_index": i + 1, "geometry_seed": rows[2 * i]["geometry_seed"],
                               "track_id": rows[2 * i]["track_id"], "episodes": episodes})
        _recheck(root, freeze_path, freeze_sha, rows, traces, source, frozen, code_root)
        if _state_sha256(actor.model) != before:
            raise ValueError("actor parameter hash changed before exclusive output")
        progress["phase"] = "writing_diagnostic_head"
        with (output / "head.pt").open("xb") as stream:
            torch.save(head.state_dict(), stream)
        result = {
            "format": "haic-rlpd-visual-representation-probe-result-v1",
            "role": "reused-TRAIN-retrospective-diagnostic-not-generalization",
            "policy_updated": False, "controller_updated": False,
            "protocol_path": freeze_path, "protocol_sha256": freeze_sha,
            "source_revision_at_freeze": frozen["execution"]["source_revision"],
            "source_revision_at_run": _source_revision(code_root),
            "source": frozen["source"], "source_hashes": frozen["source_hashes"],
            "actor": frozen["actor"], "actor_parameters_sha256_before": before,
            "actor_parameters_sha256_after": _state_sha256(actor.model),
            "feature_sha256": feature_digest.hexdigest(),
            "head_path": "head.pt", "head_sha256": sha256_file(output / "head.pt"),
            "head_role": "diagnostic-only-not-a-policy-or-trigger",
            "integrity": {"source_rechecked_before_and_after_fit": True, "trace_hashes_rechecked": 24,
                           "seed11_action_parity_decisions": sum(row["steps"] for row in rows[::2]),
                           "bounded_stack_batch": BATCH, "excluded_startup_decisions_per_episode": BURN_IN,
                           "head_updates": progress["updates_completed"] if fit_head is None else 0},
            "coverage": coverage, "fit_decisions": len(fit_labels),
            "fit_positive_decisions": int(fit_labels.sum()),
            "fit_negative_decisions": int(len(fit_labels) - fit_labels.sum()),
            "positive_weight": weight, "fit_prevalence_baseline": prevalence,
            "cutoff": 0.5,
            "finished_controls": sum(ep["outcome"] == "finished" for group in geometries for ep in group["episodes"]),
            "finished_control_false_positives_at_0_5": sum(
                ep["finished_control_false_positives_at_0_5"] or 0
                for group in geometries for ep in group["episodes"]),
            "finished_control_baseline_false_positives_at_0_5": sum(
                ep["finished_control_baseline_false_positives_at_0_5"] or 0
                for group in geometries for ep in group["episodes"]),
            "geometries": geometries,
            "limitations": "The permitted observation contains a HUD speed gauge, so success may decode contemporaneous displayed speed, not optical flow or movement history. Decisions j<20 are excluded to avoid the tiled-reset/early-startup shortcut; slower later startup remains possible. All decisions within each of 12 previously viewed, track-1 TRAIN roads are serially correlated; two actors share each road. No fresh holdout, causal remedy, closed-loop benefit or official generalization claim.",
        }
        progress["phase"] = "writing_receipt"
        with (output / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except BaseException as exc:
        with (output / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({
                "format": "haic-rlpd-visual-representation-probe-failure-v1",
                "status": "invalid-head-only-attempt", "protocol_sha256": freeze_sha,
                "phase": progress["phase"], "head_updates_completed": progress["updates_completed"],
                "error": f"{type(exc).__name__}: {exc}", "no_retry_or_relabel": True,
            }, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-protocol-template", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--protocol")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    if args.print_protocol_template:
        if any((args.protocol, args.protocol_sha256, args.output_root)):
            parser.error("template printing cannot be combined with probe arguments")
        print(json.dumps(protocol_template(), sort_keys=True, indent=2))
        return
    if not all((args.protocol, args.protocol_sha256, args.output_root)):
        parser.error("--protocol, --protocol-sha256 and --output-root are required")
    if args.preflight_only:
        print(json.dumps(preflight(ROOT, args.protocol, args.protocol_sha256, args.output_root),
                         sort_keys=True))
        return
    result = probe(ROOT, args.protocol, args.protocol_sha256, args.output_root)
    print(json.dumps({key: result[key] for key in ("format", "fit_decisions", "finished_controls")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
