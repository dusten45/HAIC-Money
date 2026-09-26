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

## 현재 작업 범위

하네스를 바탕으로 작업을 재개했습니다. 현재 여러 에이전트가 독립 방향 4개에서 가설을 탐색하는 DISCOVER/HYPOTHESIZE 단계이며, corridor 경로의 제출 적격성은 아직 미확인입니다. 구체적인 실험은 중앙 통합과 설계·구현·실행 승인 게이트를 거칩니다.

Task 7 결과 기록·SOTA 승격 코드가 `02be453`에 커밋됐습니다. 해당 인계 보고서의 검증 결과는 총 131개 중 130개 통과, 1개 건너뜀입니다. Task 8 구조 검증기와 Task 9 기존 오케스트레이터 퇴역은 남아 있습니다.

이번 구현과 현재 탐색에서 새로운 주행 성능 개선이나 SOTA 승격을 주장하지 않습니다. [한국어 운영 요약과 실제 완료 상태](docs/report.md)를 확인하세요.
