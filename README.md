# HAIC-Money

HAIC-Money is a research repository for building a reliable single-model agent for
the 2026 HAIC CarRacing AI Challenge. The target is not a collection of per-track
best records: one immutable candidate must generalize across the conditions that
matter for final evaluation.

## Quick Start

Use the current official participant repository as the authority for runtime and
submission requirements. The local development path is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python local_runner.py --track-id 1 --seed 42
python -m unittest discover -s tests -v
```

The local runner is an internal tool. A successful local run or test does not prove
official package acceptance or competition performance.

## Main Entry Points

| Purpose | Entry point |
|---|---|
| Submission agent interface | `agent.py` |
| Local official-environment mirror | `local_runner.py`, `core/`, `env_wrapper.py`, `damage.py` |
| Native DrQ-v2 research | `train_drqv2.py`, `drq_v2.py`, `run_drqv2_matched.py` |
| Native DreamerV3 research | `train_dreamerv3.py`, `dreamer_v3.py` |
| Legacy/Track Lab visual PPO research | `training/`, `haic_agent/` |
| Local evaluator | `evaluate_policy.py` |
| Submission packaging | `package_submission.py`, `training/package_submission.py` |
| Custom-track simulator and replay | `local_simulator/`, `web_simulator/` |

## Documentation

- Current research state and next gate:
  [`docs/context/current-state.md`](docs/context/current-state.md)
- Official schedule, rules, restrictions, and local submission ledger:
  [`docs/competition/`](docs/competition/)
- Architecture boundaries:
  [`docs/architecture/overview.md`](docs/architecture/overview.md)
- Internal evaluation and generalization policy:
  [`docs/evaluation/`](docs/evaluation/)
- Research workflows:
  [`docs/workflows/`](docs/workflows/)
- Active plan:
  [`docs/plans/active/`](docs/plans/active/)
- Durable experiment evidence and decisions:
  [`docs/experiments/INDEX.md`](docs/experiments/INDEX.md) and
  [`docs/decisions/INDEX.md`](docs/decisions/INDEX.md)
- Candidate, submission, and confirmation status:
  [`docs/results/MODEL_STATUS.md`](docs/results/MODEL_STATUS.md)

`AGENTS.md` is the stable router and operating constitution for coding/research
agents. It defines which documents to read for a task and separates internal
research from official external actions.

## Official Sources

- [Competition website](https://ships-duo-ethical-saver.trycloudflare.com/)
- [Official Participants repository](https://github.com/2026-HAIC/Participants)

The official sources override this repository's local mirrors whenever rules,
environment behavior, package restrictions, submission state, or schedule differ.
Refresh them before official submission, model confirmation, mock evaluation, final
deadline work, or first official evaluation after a new public track is released.

## Research Boundary

Training reward, local progress, damage, smoothness, and custom robustness metrics
are internal proxies. They are not official HAIC scores. A promising local result is
an internal candidate until it completes the explicit packaging, external submission,
and model-confirmation workflows.
