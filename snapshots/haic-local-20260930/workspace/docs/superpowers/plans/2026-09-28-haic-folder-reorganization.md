# HAIC Folder Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기능별 코드·문서·실험 스크립트 위치를 정리하고 기존 원본과 실험 근거를 보존한다.

**Architecture:** 지원 중인 루트 계약과 V2 프로필 모듈은 고정한다. 실험용 actor, 비등록 진단 모듈, 날짜별 조사 스크립트와 루트 scratch 파일을 각자의 패키지로 옮기고 import·경로 참조를 함께 바꾼다. 이동 전 바이트는 범위를 한정한 로컬 Git 커밋에 보존하고 이전·새 경로와 SHA-256을 문서화한다.

**Tech Stack:** Python 3.11 (`.venv/Scripts/python.exe`), PowerShell, Git, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-haic-folder-reorganization-design.md`

## Global Constraints

- `agent.py`, 등록된 V2 프로필의 네 모듈, `haic_research` 진입점과 `harness.config.json` 경로를 유지한다.
- `runs/`, `artifacts/haic/`, `submissions/`, `artifacts/haic-research-v2/`, `results/`, `tmp/`, `training/maps/`, `.worktrees/` 및 기존 ZIP·checkpoint·역사 문서를 수정하지 않는다.
- Task 9의 삭제 차단 파일 열 개를 이동하거나 삭제하지 않는다.
- 이전 원본 바이트를 정확한 경로만 스테이징한 로컬 Git 커밋에 남긴 후 이동한다. 기존 사용자의 다른 변경은 스테이징하지 않는다.
- 학습, 주행 평가, 후보 패키징, SOTA 승격, 공식 제출을 실행하지 않는다. 과거 점수를 재계산하지 않는다.
- `docs/history/path-migration-2026-09-28.md`에는 이전·새 경로, 이동 전 SHA-256, 원본 보존 커밋을 기록한다. 이전 경로에 코드 복제본을 남기지 않는다.

## File Structure

- `haic_agent/variants/__init__.py`와 현재 `haic_agent/*_runtime.py` 13개: 실험용 actor 변형. `runtime_config.py`와 `corridor_agent.py`는 현 위치에 둔다.
- `training/diagnostics/__init__.py`와 설계 문서에 명시한 비등록 진단 모듈 5개. 등록된 `training/benchmark_corridor.py`는 현 위치에 둔다.
- `research/scripts/__init__.py`, `research/scripts/y2026_09_28/__init__.py`, 현재 `research/*_20260928.py` 47개: 날짜별 조사·로컬 패키지 생성 스크립트.
- `research/scripts/legacy_scratch/__init__.py`, 최상위 `scratch_*.py` 10개와 `scratch_*.json` 4개: 역사적 임시 스크립트와 직접 출력.
- `docs/history/strategy-history.md`, `docs/history/path-migration-2026-09-28.md`, `docs/README.md`: 역사 문서, 이동 지도, 문서 안내.
- 참조 수정 대상: 위 파일을 import하는 `tests/test_*runtime.py`, 관련 진단 테스트, `training/train_policy.py`, `research`의 패키지 생성 스크립트, 현재 안내 문서. 과거 계획서와 인계문 본문은 그대로 둔다.

## Review Focus

1. 등록된 `benchmark_corridor_diagnostic`가 이동 때문에 끊기지 않아야 한다. Task 3에서 설정과 import 경로를 확인한다.
2. 새 날짜 패키지 이름이 Python import에 유효하고 스크립트의 저장소 루트가 실제 루트여야 한다. Task 4에서 모듈 import와 루트 값을 확인한다.
3. 이동한 actor를 참조하는 로컬 패키지 생성 스크립트의 `SOURCES` 및 `ENTRY` 경로가 모두 존재해야 한다. Task 2에서 파일과 import를 확인한다.
4. 루트 scratch trace의 새 위치와 스크립트 출력 위치가 일치해야 한다. Task 4에서 경로를 확인한다.
5. 기존 결과 파일과 현재 문서 링크가 보존되어야 한다. Task 5에서 이름 지정된 결과의 해시와 변경 문서의 로컬 링크를 확인한다.

---

### Task 1: 이동 대상 원본 보존

**Files:**
- Create: `docs/history/path-migration-2026-09-28.md` (초기 이동 목록과 해시)
- Snapshot only: 설계 문서의 이동 대상 80개

**Interfaces:**
- Produces: 각 파일의 `old_path`, `new_path`, `sha256_before`, 보존 커밋 해시. Tasks 2–5가 이 목록을 이동 범위로 사용한다.

- [ ] **Step 1: 이동 목록을 고정한다.** `haic_agent/*_runtime.py` 13개, 비등록 진단 모듈 5개, `research/*_20260928.py` 47개, 최상위 `scratch_*.py` 10개·`scratch_*.json` 4개, `docs/strategy-history.md` 1개의 현재 파일 존재와 SHA-256을 확인한다. 기존 결과 루트는 훑지 않고, 문서에 명시된 현재 후보 ZIP `artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip` 하나의 SHA-256만 전후 비교용으로 기록한다.
- [ ] **Step 2: 사용자 스테이징을 확인한다.** `git diff --cached --name-only`가 비어 있지 않으면 스테이징을 변경하지 않고 범위를 재검토한다.
- [ ] **Step 3: 원본을 보존한다.** 이동 대상 현재 바이트만 정확한 파일 경로로 `git add`하고 `git diff --cached --name-only`와 `git diff --cached --check`를 검토한 다음 `chore: snapshot HAIC sources before reorganization` 커밋을 만든다. `git add .`는 사용하지 않는다.
- [ ] **Step 4: 이동 지도의 첫 버전을 쓴다.** 80개 경로·해시와 `git rev-parse HEAD` 결과를 기록한다. 기록된 해시가 원본 파일 해시와 모두 같아야 한다.

### Task 2: Actor 변형 패키지

**Files:**
- Create: `haic_agent/variants/__init__.py`
- Move: `haic_agent/*_runtime.py` 13개 → `haic_agent/variants/`
- Modify: 옮긴 모듈의 상호 import, `research/*_20260928.py`의 actor import·`SOURCES`·`ENTRY`, `tests/test_*runtime.py`의 import와 patch 대상

**Interfaces:**
- Produces: `haic_agent.variants.<기존 모듈명>` import 경로. Task 4의 연구 스크립트가 이를 사용한다.

- [ ] **Step 1: 기존 actor 단위 검증의 import를 새 패키지 경로로 바꾸고 해당 테스트를 실행한다.** `& .\.venv\Scripts\python.exe -m pytest tests/test_hybrid_runtime.py tests/test_obstacle_commit_runtime.py tests/test_obstacle_commit_late_runtime.py tests/test_obstacle_road_guard_runtime.py tests/test_obstacle_side_evidence_runtime.py tests/test_obstacle_full_road_guard_runtime.py tests/test_obstacle_guard_brake_runtime.py tests/test_obstacle_slow_runtime.py tests/test_obstacle_slew_runtime.py tests/test_obstacle_stuck_recovery_runtime.py tests/test_obstacle_visibility_reset_runtime.py tests/test_straight_sprint_runtime.py tests/test_straight_boost_runtime.py -q`는 이동 전 `ModuleNotFoundError`로 실패해야 한다.
- [ ] **Step 2: 13개 파일을 이동하고 `haic_agent/variants/__init__.py`를 만든다.** 파일 본문의 상대·절대 actor import를 `haic_agent.variants.*`로 갱신한다.
- [ ] **Step 3: 2026-09-28 연구 스크립트에서 actor import, `ROOT / "haic_agent/..._runtime.py"`, `SOURCES`와 ZIP `ENTRY` 문자열을 새 경로로 갱신한다.** ZIP의 `SOURCES`에 `haic_agent/variants/__init__.py`를 포함한다. 이미 생성된 ZIP은 건드리지 않는다.
- [ ] **Step 4: Step 1의 actor 단위 검증과 `& .\.venv\Scripts\python.exe -m pytest tests/test_submission_layout.py -q`를 다시 실행한다.** 예상: 모두 통과. `research/package_side_evidence_20260928.py`, `package_full_road_guard_20260928.py`, `package_hybrid_20260928.py`, `package_straight_sprint_20260928.py`의 모든 `SOURCES`가 실제 파일을 가리키고 `ENTRY` import가 새 모듈을 가리키는지 정적으로 확인한다.
- [ ] **Step 5: 이 Task의 이동·참조 수정만 스테이징하고 확인 후 커밋한다.** 이미 수정돼 있던 파일은 필요한 hunk만 스테이징한다.

### Task 3: 비등록 학습 진단 모듈

**Files:**
- Create: `training/diagnostics/__init__.py`
- Move: `training/{curve_brake_screen,evaluate_hazard_potential_tune,evaluate_lagrangian_tune,preflight_hazard_potential_screen,preflight_lagrangian_screen}.py` → `training/diagnostics/`
- Modify: `training/train_policy.py`의 진단 import, 옮긴 모듈의 상호 import·루트 계산, 관련 `tests/test_curve_brake_screen.py`, `tests/test_evaluate_hazard_potential_tune.py`, `tests/test_hazard_potential_preflight.py`, `tests/test_lagrangian_screen_preflight.py`

**Interfaces:**
- Produces: `training.diagnostics.<기존 모듈명>` import 경로. 등록된 네 V2 프로필 모듈 경로는 그대로다.

- [ ] **Step 1: 관련 단위 검증의 import·patch 대상 문자열을 새 경로로 바꾸고 `& .\.venv\Scripts\python.exe -m pytest tests/test_curve_brake_screen.py tests/test_evaluate_hazard_potential_tune.py tests/test_hazard_potential_preflight.py tests/test_lagrangian_screen_preflight.py -q`를 실행한다.** 예상: 이동 전 `ModuleNotFoundError`로 실패한다.
- [ ] **Step 2: 5개 파일을 이동하고 패키지 초기화 파일을 만든다.** 상호 import와 `training/train_policy.py`의 진단 import를 갱신한다. 옮긴 파일의 저장소 루트 계산은 새 깊이에 맞춰 `Path(__file__).resolve().parents[2]`로 바꾼다.
- [ ] **Step 3: Step 1의 단위 검증을 다시 실행한다.** 예상: 통과. `harness.config.json`과 `haic_research/config.py`의 `benchmark_corridor_diagnostic` 모듈이 여전히 `training.benchmark_corridor`인지 확인한다.
- [ ] **Step 4: 이 Task의 파일과 필요한 hunk만 스테이징·검토·커밋한다.** `training/train_policy.py`의 기존 미커밋 변경을 함께 커밋하지 않는다.

### Task 4: 연구 스크립트와 루트 scratch 정리

**Files:**
- Create: `research/scripts/__init__.py`, `research/scripts/y2026_09_28/__init__.py`, `research/scripts/legacy_scratch/__init__.py`
- Move: `research/*_20260928.py` 47개 → `research/scripts/y2026_09_28/`
- Move: 최상위 `scratch_*.py` 10개·`scratch_*.json` 4개 → `research/scripts/legacy_scratch/`
- Modify: 이동한 연구 스크립트의 상호 import·`ROOT` 계산, scratch trace 출력 경로, 현재 문서의 실행 명령

**Interfaces:**
- Produces: `research.scripts.y2026_09_28.<기존 모듈명>` 및 `research.scripts.legacy_scratch.<기존 모듈명>` 경로.

- [ ] **Step 1: 옮길 스크립트의 구문 검사와 기존 `research.<이름>_20260928` import 목록을 기록한다.** 이동 후 이 목록이 활성 코드에 남지 않아야 한다.
- [ ] **Step 2: 파일을 이동하고 패키지 초기화 파일을 만든다.** 날짜별 스크립트 간 import를 새 패키지 경로로 바꾸며, 원래 `Path(__file__).resolve().parents[1]`로 저장소 루트를 계산한 스크립트는 `parents[3]`을 사용한다. 고정된 `/workspace` 경로는 바꾸지 않는다.
- [ ] **Step 3: scratch 스크립트의 출력 경로를 같은 `legacy_scratch` 폴더로 맞춘다.** 이동된 네 JSON과 스크립트가 참조하는 이름을 정적으로 확인한다.
- [ ] **Step 4: `& .\.venv\Scripts\python.exe -m compileall -q haic_agent/variants training/diagnostics research/scripts`를 실행하고, `& .\.venv\Scripts\python.exe -c "from pathlib import Path; from research.scripts.y2026_09_28.score_probe_20260928 import ROOT; assert ROOT == Path.cwd().resolve()"`로 루트 계산을 확인한다.** 훈련·주행·패키징 함수는 호출하지 않는다.
- [ ] **Step 5: 이동과 참조 수정만 스테이징·검토·커밋한다.** 과거 run·ZIP·결과 파일은 스테이징하지 않는다.

### Task 5: 문서 안내, 경로 지도, 최종 확인

**Files:**
- Move: `docs/strategy-history.md` → `docs/history/strategy-history.md`
- Create: `docs/README.md`
- Complete: `docs/history/path-migration-2026-09-28.md`
- Modify: `docs/report.md`와 현재 문서 중 `strategy-history.md` 또는 이동한 실행 파일을 가리키는 링크

**Interfaces:**
- Produces: 현재 탐색용 문서 색인과 80개 이전·새 경로·원본 해시·보존 커밋을 담은 이동 지도.

- [ ] **Step 1: 역사 문서를 이동하고 `docs/README.md`에 현재 상태·실험·계획·인계·출처·역사 자료의 경로를 적는다.** 과거 계획서·인계문 본문은 고치지 않는다.
- [ ] **Step 2: 현재 문서의 링크를 갱신하고 이동 지도를 완성한다.** 각 파일의 새 경로가 존재하고 원본 SHA-256이 보존 커밋의 바이트와 같은지 확인한다.
- [ ] **Step 3: 변경 문서의 상대 링크, 남은 옛 활성 import·경로, `git diff --check`를 검사한다.** Task 1에서 기록한 현재 후보 ZIP의 SHA-256이 같고, 보호 경로가 `git status --short`의 이번 작업 변경 목록에 없는지 확인한다.
- [ ] **Step 4: Tasks 2–3의 단위 검증과 `& .\.venv\Scripts\python.exe -m haic_research.cli validate`를 실행한다.** 예상: 모두 통과. 이는 실제 주행 성능이나 공식 규칙 적합성을 검증하지 않는다.
- [ ] **Step 5: 이 Task의 문서·지도 수정만 스테이징·검토·커밋하고 최종 변경 범위를 보고한다.**
