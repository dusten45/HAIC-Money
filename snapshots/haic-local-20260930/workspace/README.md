# 2026 HAIC CarRacing AI Challenge

## HAIC research quick start

1. Read [PROJECT_INFO.md](PROJECT_INFO.md) for the goal and the current workflow entry points.
2. Read [AGENTS.md](AGENTS.md) for research states, approvals, agent handoffs, and completion-first selection.
3. Check [COMPETITION_INFO.md](COMPETITION_INFO.md) and [RESTRICTIONS.md](RESTRICTIONS.md) before designing an experiment; verify current official sources before any external action.
4. Use [harness.config.json](harness.config.json) for the HAIC-only v2 paths, protected splits, search limits, and registered local operations. The default is plan-only.

| Need | Document |
|---|---|
| New experiments | [docs/experiments/INDEX.md](docs/experiments/INDEX.md) |
| Agent handoff | [docs/handoffs/REPORT_TEMPLATE.md](docs/handoffs/REPORT_TEMPLATE.md) |
| Official source ledger and legacy path map | [docs/sources/INDEX.md](docs/sources/INDEX.md) |
| Pinned official Participants README and LICENSE | [docs/sources/official-participants/README.md](docs/sources/official-participants/README.md), [LICENSE](docs/sources/official-participants/LICENSE) |
| Strategy decisions and current synthesis | [docs/strategy-history.md](docs/strategy-history.md), [docs/report.md](docs/report.md) |
| Historical result and local SOTA references | [RESULTS.md](RESULTS.md), [SOTA.md](SOTA.md) |

## Supported local CLI

Use `python -m haic_research.cli` from the repository root. These examples show the syntax; create reviewed metadata first and replace `example-001` with its manifest `run_id`. The files below are operation inputs, not bundled sample experiments.

```powershell
# Register a plan without launching training, evaluation, or packaging.
python -m haic_research.cli plan --manifest metadata/manifest.json --research metadata/hypothesis.json --profile train_policy --arguments metadata/arguments.json
# Explicit plan-only preview; no approval or subprocess is needed.
python -m haic_research.cli run example-001
python -m haic_research.cli status example-001
# Validate repository structure without running an operation.
python -m haic_research.cli validate
```

The manifest records the run/cycle, candidate and control revisions and package hashes, data/map/split identities, versions, resource and permission limits, and source hashes. The hypothesis supplies every field of `Hypothesis` in [haic_research/models.py](haic_research/models.py). Arguments are a JSON object with the exact registered keys in [harness.config.json](harness.config.json), without leading `--`; `train_policy` requires `train-only-site-map-split`, `defer-tune: true`, and positive `total-steps`. The harness supplies the plan hash and v2 output destination. Metadata must stay in this repository outside historical data roots.

Record each approval only after the user authorizes that stage for the persisted plan. Changed code, inputs, profiles, or plan metadata require a new run and approvals. The following lifecycle syntax is for an authorized local operation:

```powershell
python -m haic_research.cli approve example-001 design --source-ref "explicit user design approval reference"
python -m haic_research.cli approve example-001 execution --source-ref "explicit user execution approval reference"
python -m haic_research.cli run example-001 --execute
python -m haic_research.cli report example-001 --outcome REJECT --report metadata/gates.json
# For a separately registered and executed evaluation run:
python -m haic_research.cli report evaluation-001 --outcome ADVANCE --report metadata/gates.json --result metadata/candidate.json --control metadata/control.json --promote-sota
```

Choose one `report` command for the actual outcome after successful execution. Its report is an `IntegrationReport` with exactly the three registered gate results and evidence paths; optional candidate/control inputs are complete `ExperimentResult` records matching the preregistered comparison. These commands do not constitute official submission or model confirmation. The [project guide](PROJECT_INFO.md) describes operation inputs and limits; the [legacy path map](docs/sources/legacy-path-map.md) records retired behavior.

## 현재 작업 범위

완주율 관련 네 독립 방향을 [중앙 인계 문서](docs/handoffs/completion-first-discovery-20260926/INDEX.md)에 기록하고 탐색·가설 통합을 완료했습니다. corridor 전용 아이디어는 후보 배치에서 제외했습니다. 현재는 배치 설계를 작성·수정하는 단계이며, plan revision/hash·split·seed·denominator·자원 예산·설계 승인은 아직 확정하지 않았습니다. 실험 구현 전 설계 승인, 로컬 실험 실행 전 별도 실행 승인을 기록합니다. 구현 승인 단계는 요구하지 않습니다.

Task 7 결과 기록·SOTA 승격 코드와 Task 8 구조 검증기가 커밋됐습니다. Task 9 문서 전환과 새 하네스 전체 단위·구조 검증은 완료했으며, 기존 파일의 실제 삭제는 자동 정책 차단으로 남아 있습니다. 검증 수치와 한계는 [운영 요약](docs/report.md)에 기록합니다.

이번 구현과 현재 탐색에서 새로운 주행 성능 개선이나 SOTA 승격을 주장하지 않습니다. [한국어 운영 요약과 실제 완료 상태](docs/report.md)를 확인하세요.
