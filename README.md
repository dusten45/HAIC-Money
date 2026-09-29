# HAIC-Money

## 전체 로컬 작업 보존

[2026-09-30 작업 스냅샷과 원자료 복원 안내](snapshots/haic-local-20260930/README.md) — 코드·문서는 Git, 대용량 원자료는 같은 저장소의 GitHub Release에 보존합니다.

## 제출 후보 (2026-09-30)

[Arrival-speed 제출 ZIP과 소스·검증 결과](releases/arrival-speed-20260930/README.md)

별도 검증에서 기존과 같은 10/12 완주, 완주 기록 중앙값 19.34초 → 18.46초. 대회 제출 완료를 의미하지 않습니다. 저장소 루트 연구용 agent 대신 링크된 고정 ZIP을 사용하세요.


HAIC-Money is an experimental research fork of the official Participants template
for building a reliable single-model agent for the 2026 HAIC CarRacing AI Challenge.
The target is not a collection of per-track best records: one immutable candidate
must generalize across the conditions that matter for final evaluation.

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
- Multi-agent collaboration and asynchronous research discussion:
  [`talk/README.md`](talk/README.md)

`AGENTS.md` is the stable router and operating constitution for coding/research
agents. It defines which documents to read for a task and separates internal
research from official external actions.

## Official Sources

- [Competition website](https://scholarships-hardwood-headers-influenced.trycloudflare.com/)
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
