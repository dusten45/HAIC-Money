# HAIC 폴더 재배치 설계 — 2026-09-28

## 목적과 성공 조건

현재 저장소의 코드, 문서, 실험 스크립트를 역할별로 찾을 수 있게 재배치한다. 사용자가 선택한 방식은 기능별 재배치이며, 기존 경로의 원본을 로컬 Git 기록에 보존한 뒤 이동한다. 완료 기준은 새 경로의 import와 문서 링크가 맞고, 제출 계약과 지원 중인 V2 진입점이 유지되며, 이동 전 파일의 경로·해시·Git 보존 커밋을 확인할 수 있는 것이다.

이 작업은 저장소 정리다. 학습, 주행 평가, 후보 패키징, SOTA 승격, 공식 제출을 실행하지 않는다. 기존 연구 결과를 새 결과로 해석하거나 점수를 다시 계산하지 않는다.

## 보존 경계

- `agent.py`와 현재 제출 ZIP이 요구하는 최상위 계약, `haic_research`의 지원 진입점, 등록된 네 V2 프로필의 실행 모듈(`training/train_policy.py`, `training/evaluate_closed_loop.py`, `training/package_submission.py`, `training/benchmark_corridor.py`), `harness.config.json`은 경로를 유지한다.
- `runs/`, `artifacts/haic/`, `submissions/`, `artifacts/haic-research-v2/`, `results/`, `tmp/`, `training/maps/`, `.worktrees/`, `model.pt`, `report.pdf`, `RESULTS.md`, `SOTA.md`의 내용과 위치를 유지한다. 기존 ZIP과 체크포인트도 수정하지 않는다.
- 삭제가 차단된 Task 9의 열 개 파일은 이동하거나 삭제하지 않는다. `research_ops/`, `training/improvement_loop.py`, `RULES.md` 등 해당 경로는 현재 상태 그대로 둔다.
- 과거 계획서·인계문·실험 기록의 본문은 역사적 근거로 보존한다. 현재 문서의 링크와 코드 참조만 새 위치에 맞춘다.
- 기존 사용자의 다른 미커밋 변경은 복구·초기화·일괄 스테이징하지 않는다.

## 목표 구조와 이동 범위

| 현재 위치 | 새 위치 | 처리 |
|---|---|---|
| `haic_agent/*_runtime.py` (`runtime_config.py` 제외) | `haic_agent/variants/` | 실험용 actor 변형을 한 패키지에 모은다. 내부 import, 연구 스크립트, 테스트, 향후 패키지 생성 코드의 참조를 갱신한다. |
| `training/curve_brake_screen.py`, `evaluate_hazard_potential_tune.py`, `evaluate_lagrangian_tune.py`, `preflight_hazard_potential_screen.py`, `preflight_lagrangian_screen.py` | `training/diagnostics/` | 등록 프로필이 아닌 진단 코드를 묶고 import 및 `__file__` 기반 저장소 루트 계산을 갱신한다. |
| `research/*_20260928.py` | `research/scripts/2026_09_28/` | 날짜별 직접 진단·비교·로컬 패키지 생성 스크립트를 묶는다. 스크립트 간 import, 상대 경로, 문서의 현재 실행 안내를 갱신한다. |
| 최상위 `scratch_*.py`와 `scratch_*.json` | `research/scripts/legacy_scratch/` | 과거 임시 스크립트와 그 직접 출력물을 함께 둔다. 참조가 있는 과거 계획서 자체는 고치지 않고 이동 지도에서 이전 위치를 찾게 한다. |
| `docs/strategy-history.md` | `docs/history/strategy-history.md` | 역사 문서를 현재 문서와 분리한다. 현재 문서의 링크를 갱신한다. |

`haic_agent/corridor_agent.py`, 정책·관측·동역학 모듈, `training`의 지원 학습·평가·패키지 모듈, `research`의 날짜 없는 공용 모듈과 역사적 노트·JSON은 제자리에 둔다. `docs/report.md`는 현재 상태의 안내 문서로 유지하고 `docs/README.md`에 문서 영역과 새 위치를 안내한다. 이동할 패키지에는 필요한 `__init__.py`를 둔다.

## 원본 보존과 참조 갱신

1. 이동 대상의 실제 파일 목록, Git 추적 상태, SHA-256을 기록한다. 기존 `runs/`, `artifacts/haic/`, `submissions/`을 훑지 않는다.
2. 이동 대상의 현재 바이트만 정확히 스테이징해 로컬 보존 커밋을 만든다. 다른 변경은 스테이징하지 않는다. 보존 커밋을 확인한 뒤 파일을 이동한다.
3. `docs/history/path-migration-2026-09-28.md`에 각 이전 경로, 새 경로, 이동 전 SHA-256, 보존 커밋을 기록한다. 이 문서는 과거 계획서의 경로를 해석하기 위한 지도이며 원본 결과를 복사하지 않는다.
4. 활성 코드·테스트의 import, 정적 파일 목록, `Path(__file__)` 기준 루트 계산, 현재 안내 문서의 링크를 새 위치에 맞춘다. 현재 패키지 생성 스크립트가 ZIP 안에 넣는 모듈 이름도 새 경로와 일치시킨다. 이미 생성된 ZIP과 매니페스트의 파일명·해시는 그대로 둔다.
5. 이전 경로에 실행 가능한 호환 코드나 복제본을 남기지 않는다. 과거 실행의 재현에는 보존 커밋과 이동 지도를 사용한다. V2의 새 실행은 이후 별도 계획과 승인에 따른다.

## 검증과 실패 처리

- 이동 전후의 파일 목록과 해시를 대조하고 누락·중복 파일이 없는지 확인한다.
- Python import 및 구문 검사, 경로 참조 검색, 관련 기존 단위 검증과 V2 구조 검증을 수행한다. 실험 실행이나 실제 제출 패키지 생성은 검증에 포함하지 않는다.
- 정적 패키지 파일 목록과 import 경로가 일치하는지 확인한다. 이미 보존된 ZIP을 다시 생성하거나 수정하지 않는다.
- 경로 의존성을 해결하지 못한 파일은 이동을 완료했다고 표시하지 않는다. 해당 파일은 원래 위치로 되돌리고 이동 지도에 미이동 사유를 남긴다. 사용자 작업의 다른 변경에는 복구 명령을 적용하지 않는다.

## 구현 순서

먼저 이동 목록과 참조 검색을 확정하고 원본 보존 커밋을 만든다. 이어 코드와 실험 스크립트를 이동하며 import·경로 참조를 수정한다. 마지막으로 문서와 이동 지도를 갱신하고 검증한다. 각 단계의 파일 범위를 분리해 기존의 대규모 미커밋 변경과 섞이지 않게 한다.
