"""Read-only posterior/prior gap on consumed reused-TRAIN development episodes.

Run with ``python -m scripts.diagnose.dreamerv3_reused_posterior_gap`` only after
the separate protocol and this executable's SHA-256 have been frozen externally.
Posterior reconstruction conditions on the target image; it is NOT a forecast.
No simulator, learner, actor, or protected evaluation partition is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.diagnose import dreamerv3_reused_train_budget_probe as budget
from scripts.diagnose import dreamerv3_reused_train_open_loop as original


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-posterior-gap-v1"
PURPOSE = "reused-TRAIN-consumed-development-posterior-prior-diagnostic"
SOURCE = "scripts/diagnose/dreamerv3_reused_posterior_gap.py"
ACTION_SOURCE = "scripts/diagnose/dreamerv3_reused_action_probe.py"
ACTION_SOURCE_SHA256 = "b230439344be2adeb1548ff3c3053768646c5e9fbc9d92fdc4445ebdcc44b21c"
PROTOCOL = "experiments/dreamerv3-reused-posterior-gap-v1.json"
BUDGET_PROTOCOL_SHA256 = "ddcb33d19b36f6c0c17942263da835a45bf87f76271f550027e6113824ebf994"
BUDGET_RESULT = f"{budget.RUN_ROOT}/scoring-v2/score-result.json"
BUDGET_RESULT_SHA256 = "94bbd7457283d5ec8f0ad4a4ff2e654b97839d09a7eca764615de6fae625ff5d"
OUTPUT = f"{budget.RUN_ROOT}/posterior-gap-v1"
HORIZONS = (1, 8, 32)
SAMPLING = {
    "context_decisions": 8, "window_decisions": 32, "horizons": list(HORIZONS),
    "latent_seed": 120926, "context_mode": "reset-origin-full-prefix-terminal",
    "posterior_conditioning": "logged-observation-at-t-plus-1-target-conditioned",
    "one_step_conditioning": "posterior-at-t-and-logged-native-action-at-t",
}
PIXELS = 84 * 84


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty metric group")
    return original._finite(sum(values) / len(values))


def _image_row(previous: torch.Tensor, delta: torch.Tensor, target: torch.Tensor,
               baseline: torch.Tensor) -> dict[str, float | int]:
    if any(tensor.shape != (1, 1, 84, 84) for tensor in (previous, delta, target, baseline)):
        raise ValueError("residual decoder and latest frames must have shape (1,1,84,84)")
    raw = previous + delta
    prediction = raw.clamp(0.0, 1.0)
    residual = prediction - target
    low = int((raw < 0).sum().item())
    high = int((raw > 1).sum().item())
    return {
        "image_mse": original._finite(residual.square().mean().item()),
        "signed_residual_mean": original._finite(residual.mean().item()),
        "absolute_residual_mean": original._finite(residual.abs().mean().item()),
        "baseline_mse": original._finite((baseline - target).square().mean().item()),
        "clipped_below_zero_pixels": low, "clipped_above_one_pixels": high,
        "clipped_frames": int(low + high > 0), "target_frames": 1,
    }


def _image_summary(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("no image targets")
    frames = sum(int(row["target_frames"]) for row in rows)
    if frames != len(rows):
        raise ValueError("each image row must have one target frame")
    low = sum(int(row["clipped_below_zero_pixels"]) for row in rows)
    high = sum(int(row["clipped_above_one_pixels"]) for row in rows)
    clipped_frames = sum(int(row["clipped_frames"]) for row in rows)
    if low + high > frames * PIXELS or clipped_frames > frames:
        raise ValueError("invalid image clamp denominators")
    return {
        "target_frames": frames, "pixel_uses": frames * PIXELS,
        "image_mse": _mean([float(row["image_mse"]) for row in rows]),
        "baseline_mse": _mean([float(row["baseline_mse"]) for row in rows]),
        "signed_residual_mean": _mean([float(row["signed_residual_mean"]) for row in rows]),
        "absolute_residual_mean": _mean([float(row["absolute_residual_mean"]) for row in rows]),
        "clipped_below_zero_pixels": low, "clipped_above_one_pixels": high,
        "clipped_pixel_fraction": original._finite((low + high) / (frames * PIXELS)),
        "clipped_frames": clipped_frames,
    }


def _logits_kl(prior: torch.Tensor, posterior: torch.Tensor) -> float:
    """Raw categorical KL(post || prior): no unimix, epsilon clamp or free-nat floor."""
    if (not isinstance(prior, torch.Tensor) or not isinstance(posterior, torch.Tensor)
            or prior.shape != posterior.shape or prior.ndim != 3
            or prior.shape[0] != 1 or prior.shape[1] < 1 or prior.shape[2] < 2):
        raise ValueError("same-transition prior/posterior categorical logits required")
    if not torch.isfinite(prior).all() or not torch.isfinite(posterior).all():
        raise ValueError("nonfinite categorical logits")
    log_p = F.log_softmax(prior.double(), dim=-1)
    log_q = F.log_softmax(posterior.double(), dim=-1)
    value = original._finite((log_q.exp() * (log_q - log_p)).sum().item())
    if value < -1e-10:
        raise ValueError("negative categorical KL")
    return max(value, 0.0)


def _window(episode: Any, modules: tuple[Any, ...], states: torch.Tensor,
            kind: str, anchor: int, reference: dict[str, Any]) -> dict[str, Any]:
    encoder, rssm, decoder, _, _ = modules
    if (reference.get("kind") != kind or reference.get("context_anchor_decision") != anchor
            or reference.get("reset_origin_prefix_decisions") != anchor
            or reference.get("first_target_decision") != anchor
            or reference.get("last_target_decision") != anchor + 31
            or reference.get("label_uses") != 32):
        raise ValueError("window differs from pinned original scorer")

    # This is the original scorer's sampled-prior path, including its RNG position.
    rng_start = torch.random.get_rng_state()
    state = states[:, anchor].clone()
    stack = torch.from_numpy(episode.observation(anchor).copy()).unsqueeze(0)
    repeat = stack.clone()
    free_rows = []
    for index in range(anchor, anchor + 32):
        action = torch.from_numpy(episode.actions[index].copy()).unsqueeze(0)
        h, z = torch.split(state, [rssm.hidden_dim, rssm.stoch_dim], dim=-1)
        next_h, next_z, _, _ = rssm.step_prior(h, z, action)
        state = torch.cat((next_h, next_z), dim=-1)
        delta = decoder(state)
        target = torch.from_numpy(episode.observation(index + 1)[-1].copy()).view(1, 1, 84, 84)
        free_rows.append(_image_row(stack[:, -1:], delta, target, repeat[:, -1:]))
        latest = (stack[:, -1:] + delta).clamp(0.0, 1.0)
        stack = torch.cat((stack[:, 1:], latest), dim=1)
        repeat = torch.cat((repeat[:, 1:], repeat[:, -1:]), dim=1)
    after_free = torch.random.get_rng_state()
    horizons = {str(h): {"endpoint": _image_summary([free_rows[h - 1]]),
                         "prefix": _image_summary(free_rows[:h])} for h in HORIZONS}
    for name, calculated in (("image_mse", horizons["32"]["prefix"]["image_mse"]),
                             ("shifted_repeat_mse", horizons["32"]["prefix"]["baseline_mse"])):
        expected = reference.get("metrics", {}).get(name)
        if (type(expected) not in (int, float) or not math.isfinite(expected)
                or not math.isclose(calculated, expected, rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError(f"free-running {name} differs from pinned original score")

    # Rewind only for this separate comparison. At t, forecast with posterior
    # state(o[:t]) and action[t] BEFORE encoding the target observation[t+1].
    torch.random.set_rng_state(rng_start)
    h, z = torch.split(states[:, anchor].clone(), [rssm.hidden_dim, rssm.stoch_dim], dim=-1)
    one_rows, post_rows, kl_rows = [], [], []
    for index in range(anchor, anchor + 32):
        action = torch.from_numpy(episode.actions[index].copy()).unsqueeze(0)
        previous = torch.from_numpy(episode.observation(index)[-1].copy()).view(1, 1, 84, 84)
        next_h, next_z, prior_logits, _ = rssm.step_prior(h, z, action)
        prior_delta = decoder(torch.cat((next_h, next_z), dim=-1))
        target_stack = torch.from_numpy(episode.observation(index + 1).copy()).unsqueeze(0)
        target = target_stack[:, -1:]
        one_rows.append(_image_row(previous, prior_delta, target, previous))
        embed = encoder(target_stack)
        post_h, post_z, posterior_logits, _ = rssm.step_post(h, z, action, embed)
        if not torch.equal(next_h, post_h):
            raise ValueError("prior and posterior logits must share the same transition hidden state")
        post_delta = decoder(torch.cat((post_h, post_z), dim=-1))
        post_rows.append(_image_row(previous, post_delta, target, previous))
        kl_rows.append(_logits_kl(prior_logits, posterior_logits))
        h, z = post_h, post_z
    torch.random.set_rng_state(after_free)
    if not math.isclose(one_rows[0]["image_mse"], free_rows[0]["image_mse"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("anchored one-step prior differs from free-running horizon one")
    positives = sum(bool(episode.terminal[index]) for index in range(anchor, anchor + 32))
    if positives != reference.get("terminal_positive_label_uses"):
        raise ValueError("window terminal label count differs from pinned score")
    return {
        "kind": kind, "context_anchor_decision": anchor, "reset_origin_prefix_decisions": anchor,
        "first_target_decision": anchor, "last_target_decision": anchor + 31,
        "label_uses": 32, "terminal_positive_label_uses": positives,
        "posterior_target_conditioned": _image_summary(post_rows),
        "one_step_prior_teacher_forced_context": _image_summary(one_rows),
        "same_transition_raw_logits_kl": {
            "transition_uses": len(kl_rows), "mean_nats": _mean(kl_rows),
            "max_nats": max(kl_rows), "below_one_nat_uses": sum(value < 1.0 for value in kl_rows),
        },
        "free_running_prior": horizons,
    }


def _summary_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    if not windows:
        raise ValueError("no scored windows")
    def images(key: str, horizon: str | None = None, mode: str = "prefix") -> dict[str, Any]:
        rows = [window[key] if horizon is None else window[key][horizon][mode] for window in windows]
        count = sum(row["target_frames"] for row in rows)
        if count == 0:
            raise ValueError("empty image summary")
        return {
            "target_frames": count, "pixel_uses": count * PIXELS,
            **{name: _mean([float(row[name]) for row in rows]) for name in
               ("image_mse", "baseline_mse", "signed_residual_mean", "absolute_residual_mean")},
            "clipped_below_zero_pixels": sum(row["clipped_below_zero_pixels"] for row in rows),
            "clipped_above_one_pixels": sum(row["clipped_above_one_pixels"] for row in rows),
            "clipped_frames": sum(row["clipped_frames"] for row in rows),
            "clipped_pixel_fraction": original._finite(sum(row["clipped_below_zero_pixels"]
                                                           + row["clipped_above_one_pixels"] for row in rows)
                                                       / (count * PIXELS)),
        }
    kl = [window["same_transition_raw_logits_kl"] for window in windows]
    return {
        "window_count": len(windows), "window_label_uses": sum(row["label_uses"] for row in windows),
        "terminal_positive_label_uses": sum(row["terminal_positive_label_uses"] for row in windows),
        "posterior_target_conditioned": images("posterior_target_conditioned"),
        "one_step_prior_teacher_forced_context": images("one_step_prior_teacher_forced_context"),
        "same_transition_raw_logits_kl": {
            "transition_uses": sum(row["transition_uses"] for row in kl),
            "mean_nats": _mean([row["mean_nats"] for row in kl]),
            "max_nats": max(row["max_nats"] for row in kl),
            "below_one_nat_uses": sum(row["below_one_nat_uses"] for row in kl),
        },
        "free_running_prior": {str(h): {mode: images("free_running_prior", str(h), mode)
                                         for mode in ("endpoint", "prefix")} for h in HORIZONS},
    }


def _episode(episode: Any, modules: tuple[Any, ...], latent_seed: int,
             reference: dict[str, Any]) -> dict[str, Any]:
    length = episode.steps
    if (length < 41 or episode.actions.shape != (length, 3)
            or not np.isfinite(episode.actions).all() or np.any(np.abs(episode.actions) > 1)):
        raise ValueError("two 8+32 windows with bounded logged native actions required")
    anchors = (("reset", 8), ("terminal", length - 32))
    if (reference.get("episode_id") != episode.episode_id or reference.get("decisions") != length
            or (reference.get("track_id"), reference.get("geometry_seed"))
            != (episode.track_id, int(episode.geometry_id)) or len(reference.get("windows", [])) != 2):
        raise ValueError("episode identity/count differs from pinned score")
    encoder, rssm = modules[:2]
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(latent_seed)
        max_anchor = anchors[-1][1]
        embeds = []
        for start in range(0, max_anchor + 1, 32):
            stop = min(start + 32, max_anchor + 1)
            embeds.append(encoder(torch.from_numpy(np.stack(
                [episode.observation(i) for i in range(start, stop)])).float()).unsqueeze(0))
        first = torch.zeros((1, max_anchor), dtype=torch.bool)
        first[:, 0] = True
        states, _, _ = rssm.observe_sequence(
            torch.cat(embeds, dim=1), torch.from_numpy(episode.actions[:max_anchor].copy()).unsqueeze(0), first)
        if states.shape != (1, max_anchor + 1, rssm.hidden_dim + rssm.stoch_dim):
            raise ValueError("reset-origin posterior prefix length or latent dimension differs")
        windows = [_window(episode, modules, states, kind, anchor, reference["windows"][index])
                   for index, (kind, anchor) in enumerate(anchors)]
    scored = set(range(8, 40)) | set(range(length - 32, length))
    result = {
        "episode_id": episode.episode_id, "track_id": episode.track_id,
        "geometry_seed": int(episode.geometry_id), "decisions": length,
        "terminal_event": bool(episode.terminal[-1]), "finished": bool(episode.finished[-1]),
        "unique_scored_decisions": len(scored), "duplicate_label_uses": 64 - len(scored),
        "unique_terminal_positive_labels": sum(bool(episode.terminal[i]) for i in scored),
        "windows": windows, "summary": _summary_windows(windows),
    }
    for key in ("terminal_event", "finished", "unique_scored_decisions",
                "duplicate_label_uses", "unique_terminal_positive_labels"):
        if result[key] != reference.get(key):
            raise ValueError(f"{key}: episode denominators differ from pinned score")
    return result


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("no independent episodes")
    windows = [window for episode in episodes for window in episode["windows"]]
    summary = _summary_windows(windows)
    return {
        "independent_episode_count": len(episodes),
        "independent_road_count": len({(row["track_id"], row["geometry_seed"]) for row in episodes}),
        "independent_terminal_episode_count": sum(row["terminal_event"] for row in episodes),
        "unique_scored_decisions": sum(row["unique_scored_decisions"] for row in episodes),
        "duplicate_label_uses": sum(row["duplicate_label_uses"] for row in episodes),
        "unique_terminal_positive_labels": sum(row["unique_terminal_positive_labels"] for row in episodes),
        **summary,
    }


def preflight(protocol_path: Path, protocol_sha256: str, source_sha256: str,
              output_dir: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Fail closed before reading any development archive bytes or making outputs."""
    root = Path(repo_root).resolve()
    if original._relative(root, protocol_path) != PROTOCOL:
        raise ValueError("a separate fixed posterior-gap protocol path is required")
    if original._relative(root, output_dir) != OUTPUT:
        raise ValueError("output must be the new fixed posterior-gap sibling directory")
    document = original._json(budget._fixed_file(root, PROTOCOL, PROTOCOL, protocol_sha256, cap=1024 * 1024))
    if (set(document) != {"format", "purpose", "study_id", "budget_score_protocol",
                          "budget_score_result", "source_sha256", "sampling", "resources"}
            or document["format"] != FORMAT or document["purpose"] != PURPOSE
            or document["study_id"] != original.DEVELOPMENT_ID
            or document["sampling"] != SAMPLING):
        raise ValueError("invalid separate posterior-gap protocol schema/sampling")
    if document["budget_score_protocol"] != {"path": "experiments/dreamerv3-reused-train-budget-probe-v1.json",
                                             "sha256": BUDGET_PROTOCOL_SHA256}:
        raise ValueError("exact frozen 64/256 budget protocol required")
    if document["budget_score_result"] != {"path": BUDGET_RESULT, "sha256": BUDGET_RESULT_SHA256}:
        raise ValueError("exact frozen 256-update score result required")
    parent = budget._ref(root, document["budget_score_protocol"],
                         document["budget_score_protocol"]["path"], BUDGET_PROTOCOL_SHA256)
    v2 = budget._ref(root, document["budget_score_result"], BUDGET_RESULT, BUDGET_RESULT_SHA256)
    if document["resources"] != parent.get("resources"):
        raise ValueError("resource bounds differ from frozen budget scorer")
    sources = document["source_sha256"]
    if (not isinstance(sources, dict)
            or sources != {**parent["source_sha256"],
                           ACTION_SOURCE: ACTION_SOURCE_SHA256, SOURCE: source_sha256}):
        raise ValueError("exact scorer, model, action-probe and new executable SHA map required")
    for name, digest in sources.items():
        budget._fixed_file(root, name, name, digest, cap=1024 * 1024)
    if original._sha256(Path(__file__)) != original._hash(source_sha256):
        raise ValueError("executing posterior-gap script differs from frozen source")

    checked = budget.preflight(root / document["budget_score_protocol"]["path"],
                               BUDGET_PROTOCOL_SHA256, root / OUTPUT, repo_root=root)
    if (v2.get("format") != budget.FORMAT or v2.get("purpose") != budget.PURPOSE
            or v2.get("status") != "iterative_tuning_descriptive_only"
            or v2.get("score_protocol_sha256") != BUDGET_PROTOCOL_SHA256
            or v2.get("original_score_result_sha256") != budget.ORIGINAL_SCORE_RESULT_SHA256
            or v2.get("source_sha256") != checked["protocol"]["source_sha256"]
            or v2.get("sampling") != checked["protocol"]["scoring"]
            or any(v2.get(key) is not False for key in (
                "fresh_claim", "p1b_claim", "promotion_eligible", "student_actor_trained"))
            or set(v2.get("strata", {})) != {"random", "teacher"}):
        raise ValueError("pinned 256-update score identity/claims differ")
    for source_arm in ("random", "teacher"):
        old = checked["original"]["strata"][source_arm]
        newer = v2["strata"][source_arm]
        if (old.get("development_archive_sha256")
                != checked["checked_v1"]["development"][source_arm]["row"]["archive_sha256"]
                or {key: val for key, val in old.items() if key != "models"}
                != {key: val for key, val in newer.items() if key != "models"}
                or len(old.get("models", [])) != 4 or len(newer.get("models", [])) != 4
                or old.get("scored_episode_count") != 4
                or old.get("scored_independent_geometries") != 4
                or old.get("scored_independent_terminal_episodes") != 4):
            raise ValueError("exact four independent scored development roads/episodes required")
        for models, training in ((old["models"], checked["checked_v1"]["training"]),
                                 (newer["models"], checked["training"])):
            for index, model in enumerate(models):
                arm, seed = ("random", "teacher")[index // 2], index % 2
                sha = (training[arm][seed]["checkpoint_sha256"] if models is old["models"]
                       else training[arm][seed]["sha256"])
                if (model.get("model_arm") != arm or type(model.get("learner_seed")) is not int
                        or model["learner_seed"] != seed or model.get("checkpoint_sha256") != sha
                        or model.get("aggregate", {}).get("episode_count") != 4
                        or len(model.get("episodes", [])) != 4):
                    raise ValueError("eight exact model/checkpoint/episode identities required")
    return {**checked, "v2_result": v2, "gap_protocol": document,
            "gap_protocol_sha256": protocol_sha256, "source_sha256": source_sha256}


def score(protocol_path: Path, protocol_sha256: str, source_sha256: str,
          output_dir: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, source_sha256, output_dir, repo_root=repo_root)
    root, resources = checked["root"], checked["protocol"]["resources"]
    cgroup = original._cgroup()
    if (cgroup["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or cgroup["available_bytes"] < checked["required_available_bytes"]):
        raise ValueError("cgroup lacks frozen archive-load headroom")
    torch.set_num_threads(1)
    datasets = {}
    for arm in ("random", "teacher"):
        dev = checked["checked_v1"]["development"][arm]
        dataset = original._load_dataset(dev["archive"], dev["row"], resources,
                                         checked["checked_v1"]["development_cells"],
                                         original.DEVELOPMENT_ID, arm)
        receipt = dev["receipt"]
        rows = [row for row in receipt["episode_rows"] if row.get("status") == "complete"]
        if (dataset.transition_count != receipt["stored_decisions"]
                or len(dataset.episodes) != receipt["complete_episode_count"]
                or len(dataset.episodes) != len(rows) or len(rows) != 4
                or {(ep.track_id, int(ep.geometry_id)) for ep in dataset.episodes}
                != set(checked["checked_v1"]["development_cells"])):
            raise ValueError("exact sealed development episode/road/decision counts required")
        for ep, row in zip(dataset.episodes, rows, strict=True):
            if (type(row.get("episode_id")) is not int or ep.episode_id != str(row["episode_id"])
                    or ep.metadata.get("attempt") != row.get("attempt")
                    or (ep.track_id, int(ep.geometry_id)) != (row.get("track_id"), row.get("geometry_seed"))
                    or ep.steps != row.get("decisions") or ep.steps < 41
                    or bool(ep.terminal[-1]) != row.get("terminal") or not ep.terminal[-1]
                    or bool(ep.finished[-1]) != row.get("finished")):
                raise ValueError("sealed development episode differs from frozen receipt")
        datasets[arm] = dataset

    report: dict[str, Any] = {
        "format": FORMAT, "purpose": PURPOSE, "status": "iterative_tuning_descriptive_only",
        "protocol_sha256": protocol_sha256, "source_sha256": checked["gap_protocol"]["source_sha256"],
        "pinned_inputs_sha256": {
            "v1_score_protocol": budget.ORIGINAL_SCORE_PROTOCOL_SHA256,
            "v1_score_result": budget.ORIGINAL_SCORE_RESULT_SHA256,
            "v2_score_protocol": BUDGET_PROTOCOL_SHA256, "v2_score_result": BUDGET_RESULT_SHA256,
            "v1_offline_protocol": original.OFFLINE_SHA256,
            "v2_offline_protocol": budget.V2_OFFLINE_SHA256,
            "development_collection_protocol": checked["protocol"]["development_collection_protocol"]["sha256"],
        },
        "sampling": SAMPLING, "fresh_claim": False, "p1b_claim": False,
        "promotion_eligible": False, "student_actor_trained": False,
        "interpretation": "All eight frozen model-only checkpoints on the SAME already-consumed reused-TRAIN "
                          "development episodes. Posterior reconstruction sees its target o[t+1] and is NOT a "
                          "forecast or predictive generalization. One-step prior sees observed posterior at t "
                          "and logged action[t], never o[t+1]; free-running prior sees only the scorer's "
                          "reset-origin anchor and future logged actions. Raw-logit KL has no unimix or "
                          "free-nat clamp. Posterior/one-step repeat uses the logged frame at t; "
                          "free-running shifted-repeat uses the frozen anchor only. Two windows and eight "
                          "models per road do not increase independent episode/geometry denominators. "
                          "No actor, simulator counterfactual, fresh holdout, blind, confirmation or official claim.",
        "strata": {},
    }
    for source_arm in ("random", "teacher"):
        dataset = datasets[source_arm]
        stratum: dict[str, Any] = {
            "development_archive_sha256": checked["checked_v1"]["development"][source_arm]["row"]["archive_sha256"],
            "source_id": checked["checked_v1"]["development"][source_arm]["row"]["source_id"],
            "independent_episode_count": 4, "independent_road_count": 4, "models": [],
        }
        for updates in (64, 256):
            for arm in ("random", "teacher"):
                for seed in (0, 1):
                    index = 2 * (arm == "teacher") + seed
                    reference = (checked["original"] if updates == 64 else checked["v2_result"])["strata"][source_arm]["models"][index]
                    row = (checked["checked_v1"]["training"][arm][seed] if updates == 64
                           else checked["training"][arm][seed])
                    sha = row["checkpoint_sha256"] if updates == 64 else row["sha256"]
                    budget._fixed_file(root, original._relative(root, row["checkpoint"]),
                                       original._relative(root, row["checkpoint"]), sha,
                                       cap=resources["max_checkpoint_bytes"])
                    if updates == 64:
                        modules = original._checkpoint_modules(
                            row["checkpoint"], arm=arm, seed=seed,
                            dataset_digest=checked["checked_v1"]["offline"]["datasets"][arm]["dataset_digest"],
                            config=checked["checked_v1"]["offline"]["learner"]["config"])
                    else:
                        modules = budget._modules(row["checkpoint"], arm=arm, seed=seed,
                                                  offline_sha=budget.V2_OFFLINE_SHA256,
                                                  dataset_digest=checked["offline"]["datasets"][arm]["dataset_digest"],
                                                  config=checked["offline"]["learner"]["config"])
                    episodes = [_episode(ep, modules,
                                         SAMPLING["latent_seed"] + (source_arm == "teacher") * 1_000_003
                                         + i * 1009 + seed, reference["episodes"][i])
                                for i, ep in enumerate(dataset.episodes)]
                    del modules
                    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
                    for ep in episodes:
                        grouped[(ep["track_id"], ep["geometry_seed"])].append(ep)
                    aggregate = _aggregate(episodes)
                    old = reference["aggregate"]
                    if (aggregate["independent_episode_count"] != old["episode_count"]
                            or aggregate["window_count"] != old["window_count"]
                            or aggregate["window_label_uses"] != old["window_label_uses"]
                            or aggregate["unique_scored_decisions"] != old["unique_scored_decisions"]
                            or aggregate["duplicate_label_uses"] != old["duplicate_label_uses"]
                            or aggregate["terminal_positive_label_uses"] != old["terminal_positive_label_uses"]
                            or aggregate["unique_terminal_positive_labels"] != old["unique_terminal_positive_labels"]):
                        raise ValueError("aggregate labels/counts differ from pinned original score")
                    stratum["models"].append({
                        "model_arm": arm, "learner_seed": seed, "updates": updates,
                        "checkpoint_sha256": sha, "aggregate": aggregate,
                        "roads": [{"track_id": track, "geometry_seed": road, **_aggregate(items)}
                                  for (track, road), items in sorted(grouped.items())],
                        "episodes": episodes,
                    })
        report["strata"][source_arm] = stratum

    # All checks, including post-computation drift/collision, precede the only output.
    budget._fixed_file(root, PROTOCOL, PROTOCOL, protocol_sha256, cap=1024 * 1024)
    for name, digest in checked["gap_protocol"]["source_sha256"].items():
        budget._fixed_file(root, name, name, digest, cap=1024 * 1024)
    if original._sha256(Path(__file__)) != source_sha256:
        raise ValueError("executing posterior-gap script changed while scoring")
    for name, digest in ((budget.ORIGINAL_SCORE_PROTOCOL_PATH, budget.ORIGINAL_SCORE_PROTOCOL_SHA256),
                         (budget.ORIGINAL_SCORE_RESULT_PATH, budget.ORIGINAL_SCORE_RESULT_SHA256),
                         ("experiments/dreamerv3-reused-train-budget-probe-v1.json", BUDGET_PROTOCOL_SHA256),
                         (BUDGET_RESULT, BUDGET_RESULT_SHA256),
                         (original.OFFLINE_PATH, original.OFFLINE_SHA256),
                         (budget.V2_OFFLINE_PATH, budget.V2_OFFLINE_SHA256)):
        budget._fixed_file(root, name, name, digest, cap=1024 * 1024)
    for ref in (checked["checked_v1"]["offline"]["collection_protocol"],
                checked["protocol"]["development_collection_protocol"],
                checked["checked_v1"]["training_collection"]["r6_protocol"]):
        budget._fixed_file(root, ref["path"], ref["path"], ref["sha256"], cap=1024 * 1024)
    for arm in ("random", "teacher"):
        for protocol in (checked["checked_v1"]["protocol"], checked["protocol"]):
            for row in protocol["training"][arm]:
                for key, cap in (("result", 1024 * 1024), ("checkpoint", resources["max_checkpoint_bytes"])):
                    budget._fixed_file(root, row[f"{key}_path"], row[f"{key}_path"],
                                       row[f"{key}_sha256"], cap=cap)
        for row in (checked["checked_v1"]["offline"]["datasets"][arm],
                    checked["protocol"]["development"][arm]):
            for key, cap in (("receipt", 1024 * 1024), ("archive", resources["max_archive_bytes"])):
                budget._fixed_file(root, row[f"{key}_path"], row[f"{key}_path"], row[f"{key}_sha256"], cap=cap)
    output = original._path(root, OUTPUT, "output", exists=False)
    output.mkdir(exist_ok=False)
    with (output / "posterior-gap-result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha256", required=True,
                        help="externally frozen SHA-256 of this executable before any real archive read")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = score(args.protocol, args.protocol_sha256, args.source_sha256, args.output, repo_root=args.repo_root)
    print(json.dumps({"status": result["status"], "output": str(args.output / "posterior-gap-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
