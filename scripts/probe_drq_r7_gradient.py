"""TRAIN-only preflight: pre-fix one r7b preservation weight from two r6 controls."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from drq_v2 import DrQActor, DrQCritic, Uint8Replay, load_exported_actor, random_shift
from haic.algorithms.drq_v2.retention import RetentionReplaySampler
from scripts import train_drq_geometry_mix as r6
from scripts.train_drq_retention_r7 import R6_SHA, RUN_ROOT, load_source_replay, pinned, sha


@torch.no_grad()
def _load_models(checkpoint: Path, actor_path: Path) -> tuple[DrQActor, DrQCritic, DrQActor]:
    payload = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=False)
    actor = DrQActor().eval()
    critic = DrQCritic().eval()
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic_one"])
    teacher, _, _ = load_exported_actor(actor_path)
    teacher.requires_grad_(False)
    return actor, critic, teacher


def _norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    return float(torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients)))


def probe(root: Path) -> dict:
    reference_path = pinned(root, "experiments/drqv2-geometry-mix-v1-r6.json", R6_SHA)
    protocol = json.loads(reference_path.read_text(encoding="utf-8"))
    r6._validate_protocol(protocol)
    r6._validate_catalog(protocol, root)
    catalog = json.loads(pinned(root, protocol["catalog"]["path"], protocol["catalog"]["sha256"]).read_text())
    excluded = {int(row["geometry_seed"]) for row in catalog["train_diagnostic"]}
    rows = []
    torch.set_num_threads(1)
    for seed in (0, 1):
        arm = r6._arm(protocol, seed, "uniform")
        source = next(item for item in protocol["source_actors"] if item["learner_seed"] == seed)
        source_checkpoint = pinned(root, source["checkpoint_path"], source["checkpoint_sha256"])
        source_actor = pinned(root, source["actor_path"], source["actor_sha256"])
        source_replay = load_source_replay(root, source, expected_sha=source["checkpoint_sha256"],
                                           expected_seed=seed, excluded_roads=excluded)
        result = json.loads((root / arm["run_dir"] / "result.json").read_text(encoding="utf-8"))
        final = next(item for item in result["candidates"] if item["checkpoint_online_step"] == 32768)
        checkpoint = pinned(root, final["checkpoint_path"], final["checkpoint_sha256"])
        payload = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=False)
        online = Uint8Replay(capacity=100000, action_dim=3, n_step=3, gamma=.99)
        online.load_state_dict(payload["replay"])
        if online.size != 32768 or payload["trainer_state"]["source_seed"] != seed:
            raise ValueError("r6 uniform replay checkpoint identity mismatch")
        sampler = RetentionReplaySampler(online, source_replay, seed=arm["replay_rng_seed"])
        batch = sampler.sample(64)
        actor, critic, teacher = _load_models(checkpoint, source_actor)
        observations = torch.from_numpy(batch["observation"])
        shifted = random_shift(observations, pad=4, seed=4242)
        q_loss = -critic(shifted, actor(shifted)).mean()
        q_norm = _norm(q_loss, list(actor.parameters()))
        original = observations[torch.from_numpy(batch["source"] == 1)]
        with torch.no_grad():
            reference_action = teacher(original)
        preserve = F.mse_loss(actor(original), reference_action)
        p_norm = _norm(preserve, list(actor.parameters()))
        if not all(math.isfinite(value) and value > 0 for value in (q_norm, p_norm, float(preserve))):
            raise ValueError("r6 gradient scale is zero or non-finite; cannot pre-fix lambda")
        rows.append({"source_seed": seed, "r6_variant": "uniform", "r6_checkpoint_sha256": sha(checkpoint),
                     "source_checkpoint_sha256": sha(source_checkpoint), "source_actor_sha256": sha(source_actor),
                     "sampled_source": 32, "sampled_online": 32,
                     "source_indices": batch["source_indices"][batch["source"] == 1].tolist(),
                     "online_indices": batch["source_indices"][batch["source"] == 0].tolist(),
                     "q1_actor_gradient_l2": q_norm, "preservation_gradient_l2": p_norm,
                     "unweighted_preservation_loss": float(preserve),
                     "zero_lag_teacher": False})
        del payload, online, source_replay, sampler, actor, critic, teacher
    quotients = [row["q1_actor_gradient_l2"] / row["preservation_gradient_l2"] for row in rows]
    # One power-of-two lambda, closest to one quarter of the median Q1 gradient.
    lam = float(2.0 ** round(math.log2(.25 * float(np.median(quotients)))))
    if not .125 <= lam <= 128:
        raise ValueError("predeclared bounded gradient rule yielded unreasonable lambda")
    return {"format": "haic-drq-retention-gradient-probe-v1", "partition": "TRAIN source replay + r6 TRAIN online replay",
            "r6_protocol_sha256": R6_SHA, "r6_comparator_variant": "uniform", "source_seed_count": 2,
            "probe_rule": "batch 32:32; r6 Q1 actor on pad4 seed4242; source raw action MSE; lambda=nearest power of 2 to 0.25*median(Q_grad/preserve_grad)",
            "lambda_preserve": lam, "weighted_to_q_gradient_ratios": [lam / quotient for quotient in quotients],
            "rows": rows, "train_diagnostic_used": False, "environment_decisions": 0,
            "learner_updates": 0, "source_replay_mutated": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "gradient-scale-v1.json")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() or not output.parent.is_dir() or not output.is_relative_to(root / RUN_ROOT):
        raise ValueError("gradient receipt path must be a new file inside existing r7 run root")
    result = probe(root)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(output), "sha256": sha(output), "lambda_preserve": result["lambda_preserve"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
