import json
import math
from pathlib import Path
import numpy as np


def analyze_pilot(run_dir: Path, output_path: Path | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    episodes_file = run_dir / "episodes.jsonl"
    metrics_file = run_dir / "metrics.jsonl"
    result_file = run_dir / "result.json"

    episodes = []
    if episodes_file.is_file():
        for line in episodes_file.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("event") == "end":
                    episodes.append(row)
            except json.JSONDecodeError:
                continue

    metrics = []
    if metrics_file.is_file():
        for line in metrics_file.read_text().splitlines():
            try:
                row = json.loads(line)
                if "loss_wm" in row:
                    metrics.append(row)
            except json.JSONDecodeError:
                continue

    eval_dirs = sorted(run_dir.glob("checkpoints/step-*/evaluation.json"))
    evaluations = {}
    for eval_file in eval_dirs:
        step = int(eval_file.parent.name.split("-")[1])
        try:
            report = json.loads(eval_file.read_text())
            evaluations[str(step)] = report["ranked"][0]["summary"]
        except Exception as e:
            evaluations[str(step)] = {"error": str(e)}

    # Separate warmup episodes from policy episodes
    warmup_episodes = [ep for ep in episodes if ep["global_step"] <= 1000]
    policy_episodes = [ep for ep in episodes if ep["global_step"] > 1000]

    warmup_progress = [ep["progress"] for ep in warmup_episodes] if warmup_episodes else [0.0]
    policy_progress = [ep["progress"] for ep in policy_episodes] if policy_episodes else [0.0]

    warmup_rewards = [ep["reward"] for ep in warmup_episodes] if warmup_episodes else [0.0]
    policy_rewards = [ep["reward"] for ep in policy_episodes] if policy_episodes else [0.0]

    # Action saturation
    policy_saturations = [
        ep.get("native_saturation_fraction", [0, 0, 0])[0] for ep in policy_episodes
    ]
    policy_steering_means = [
        ep.get("native_action_mean", [0, 0, 0])[0] for ep in policy_episodes
    ]

    report = {
        "run_dir": str(run_dir),
        "total_episodes_closed": len(episodes),
        "warmup_episodes": len(warmup_episodes),
        "policy_episodes": len(policy_episodes),
        "warmup_mean_progress": float(np.mean(warmup_progress)),
        "policy_mean_progress": float(np.mean(policy_progress)),
        "policy_max_progress": float(np.max(policy_progress)) if policy_progress else 0.0,
        "warmup_mean_reward": float(np.mean(warmup_rewards)),
        "policy_mean_reward": float(np.mean(policy_rewards)),
        "policy_steering_saturation_fraction_mean": float(np.mean(policy_saturations)) if policy_saturations else 0.0,
        "policy_steering_mean": float(np.mean(policy_steering_means)) if policy_steering_means else 0.0,
        "checkpoints_evaluated": evaluations,
        "world_model_losses": {
            "loss_wm_initial": metrics[0]["loss_wm"] if metrics else None,
            "loss_wm_final": metrics[-1]["loss_wm"] if metrics else None,
            "loss_obs_final": metrics[-1]["loss_obs"] if metrics else None,
            "loss_reward_final": metrics[-1]["loss_reward"] if metrics else None,
            "loss_continue_final": metrics[-1]["loss_continue"] if metrics else None,
            "loss_kl_final": metrics[-1]["loss_kl"] if metrics else None,
        },
        "policy_learning_verdict": {
            "progress_improved_over_warmup": bool(np.mean(policy_progress) > np.mean(warmup_progress)),
            "steering_saturation_severe": bool(np.mean(policy_saturations) > 0.8) if policy_saturations else False,
            "finishes_achieved": sum(ep.get("finished", False) for ep in policy_episodes),
        },
    }

    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    import sys
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    res = analyze_pilot(run, out)
    print(json.dumps(res, indent=2))
