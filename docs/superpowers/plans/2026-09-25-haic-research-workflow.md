# HAIC Research Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** HAIC 전용 연구 운영 체계를 제공된 하네스 문서에 맞춰 구현하고, 규칙을 통과한 후보의 완주율을 최우선으로 비교하며, 구형 오케스트레이션을 검증 후 퇴역시킨다.

**Architecture:** `AGENTS.md`와 HAIC 전용 `harness.config.json`이 정책의 단일 기준이 된다. `haic_research/`는 상태·가설·완주율 정책·기록·허용 명령을 담당하고, 중앙 오케스트레이터는 독립 에이전트의 보고와 승인 게이트를 관리한다. 과거 데이터는 자동 검색하지 않고, 새 실행은 별도 경로에만 쓴다.

**Tech Stack:** Python `>=3.11,<3.12`, Python 표준 라이브러리(JSON, dataclasses, subprocess, unittest), Windows PowerShell validator shim.

**Spec:** `docs/superpowers/specs/2026-09-25-haic-research-workflow-redesign.md`

## Global Constraints

- HAIC 전용으로 구현하며 다른 프로젝트용 portable package, generic core/adapter, arbitrary project CLI는 만들지 않는다.
- 제공 문서의 파일 역할, 상태 머신, 가설 형식, 중앙 조정, 보고 구분, 4/8/2 탐색 한도, 3회 pivot 기준, 세 판정 게이트를 반영한다.
- `rule_compliance` PASS는 승격의 필수조건이며, 제출 가능한 후보는 같은 registered split에서 **완주율을 가장 먼저** 비교한다.
- 완주율이 같을 때만 중앙 완주 시간, 미완주 진행도, P90, 충돌·손상, 추론 비용 순으로 비교한다. 공식 score와 내부 우선순위를 분리한다.
- 설계·구현·실행은 각각 승인 기록을 요구하고, CLI의 기본 동작은 plan-only다.
- 기존 `runs/`, `artifacts/haic/`, `submissions/`의 데이터·manifest·hash는 복사·이동·수정·자동 색인하지 않는다.
- 새 run은 `runs/haic-research-v2/<run-id>/`, 새 artifact는 `artifacts/haic-research-v2/<run-id>/` 아래에 둔다.
- `research_ops`와 `training/improvement_loop.py`는 새 코드의 의존성이나 지원 진입점으로 남기지 않는다. 퇴역은 새 경로 검증 후 수행한다.
- 외부 제출·모델 확인·대회 사이트 업로드는 CLI에서 실행하지 않는다.
- 현재 체크아웃에는 사전 변경과 미추적 파일이 많다. 커밋은 계획에 적힌 파일만 stage하고 `git add -A`는 사용하지 않는다. 브랜치가 원격보다 93개 커밋 뒤이므로 별도 조정 전에는 push하지 않는다.
- 학습·시뮬레이션·대회 평가는 구조 전환의 검증에 사용하지 않는다.

## Review Focus

- completion rate가 더 낮은 빠른 후보가 이기지 않음 — Task 4의 순위·승격 테스트.
- 평가 split, map/seed, denominator가 다른 결과는 matched 비교로 취급하지 않음 — Task 4의 비교 호환성 테스트.
- `UNKNOWN`/`FAIL` 규칙 준수 또는 메커니즘 게이트가 SOTA 승격으로 이어지지 않음 — Task 4와 7의 판정 테스트.
- infra invalid cycle은 3회 pivot 연속 실패 횟수에서 제외됨 — Task 4의 pivot 테스트.
- 경로 이탈, legacy 경로 쓰기, 미승인 명령 또는 임의 shell이 차단됨 — Task 2, 5, 6의 경로·실행 테스트.

---

## File Structure

| File | Responsibility |
|---|---|
| `AGENTS.md` | source precedence, state machine, central coordinator, agent report, approval/search rules |
| `PROJECT_INFO.md` | HAIC goals, official source pointers, success endpoint, completion-rate objective, operation entry points |
| `RESTRICTIONS.md` | competition/runtime, privacy/leakage, split, execution and submission limits |
| `harness.config.json` | HAIC-only path, metric, split, search, budget, and command-profile settings |
| `haic_research/config.py` | load and validate the single HAIC configuration |
| `haic_research/models.py` | hypothesis, approval, result, gate, manifest, event, five-field report and coordinator envelope schemas |
| `haic_research/state.py` | legal workflow transitions and approval checks |
| `haic_research/policy.py` | comparable-result checks, completion-first ranking, promotion and pivot rules |
| `haic_research/coordinator.py` | batch validation, non-overlapping assignments, report consolidation |
| `haic_research/records.py` | new-run directory, manifest/event/report persistence |
| `haic_research/commands.py` | registered argv profiles and approval-gated subprocess execution |
| `haic_research/results.py` | append-only experiment entries and gated SOTA promotion |
| `haic_research/validation.py` | structural/document/config validation |
| `haic_research/cli.py` | HAIC-only validate, plan, approval, status, execute and report entry points |
| `scripts/harness/validate.ps1` | PowerShell wrapper around the Python validator |
| `docs/experiments/`, `docs/handoffs/`, `docs/sources/` | experiment records, agent handoffs and cited source ledger |
| `docs/strategy-history.md`, `docs/report.md` | strategy decision history and current evidence summary |
| `tests/test_haic_research_*.py` | isolated tests for config, workflow, policy, records, commands, and reporting |
| `.gitignore` | ignore generated v2 run/artifact and PDF outputs while retaining the directory marker |
| `output/pdf/.gitkeep` | preserve the requested report-output directory without tracking generated PDFs |

`docs/sources/official-participants/README.md` and `LICENSE` are a pinned, verbatim local mirror of the official Participants repository documentation and license; `docs/sources/INDEX.md` records the exact commit and check date. `COMPETITION_INFO.md` remains a local historical summary, never the source mirror or current authority. Existing `RESULTS.md` generated rows are frozen historical data. `RULES.md` is migrated into `AGENTS.md` before removal. Existing training/evaluation modules are reviewed per command profile and may be reused behind the new interface; old orchestration is not.

### Test conventions

Named test builders (`make_config`, `messages`, `result`, `approval`, `config`, `config_with_run_root`, `valid_args`, `FakeRunner`, `write_fixture`, `experiment_file`, and `project_fixture`) are defined in the test module that uses them; they are not production APIs. Filesystem tests use `tempfile.TemporaryDirectory`. Command tests use an injected `FakeRunner` that records argv, cwd, timeout, and shell settings, and never starts a real process.

## Task 1: Establish the HAIC document system

**Files:**
- Create: `AGENTS.md`
- Create: `PROJECT_INFO.md`
- Create: `harness.config.json`
- Create: `docs/experiments/INDEX.md`
- Create: `docs/handoffs/REPORT_TEMPLATE.md`
- Create: `docs/sources/INDEX.md`
- Create: `docs/sources/legacy-path-map.md`
- Create: `docs/sources/official-participants/README.md`
- Create: `docs/sources/official-participants/LICENSE`
- Create: `docs/strategy-history.md`
- Create: `docs/report.md`
- Modify: `README.md`
- Modify: `COMPETITION_INFO.md`
- Modify: `RESTRICTIONS.md`
- Modify: `RESULTS.md`
- Modify: `SOTA.md`
- Modify: `.gitignore`
- Create: `output/pdf/.gitkeep`

**Interfaces:**
- Produces: canonical operating documents and a HAIC config with keys `schema_version`, `project`, `paths`, `official_sources`, `workflow`, `search`, `metrics`, `splits`, and `commands`.

- [ ] **Step 1: Inventory existing doc links and unique content**

Run:
```powershell
rg -n "COMPETITION_INFO|RESTRICTIONS|RULES|RESULTS|SOTA|research_ops|improvement_loop|artifacts/haic|runs/" README.md COMPETITION_INFO.md RESTRICTIONS.md RULES.md RESULTS.md SOTA.md docs research_ops training/improvement_loop.py
```
Expected: every current operational entry point and generated-results reference is listed before moving document content.

- [ ] **Step 2: Write `AGENTS.md` with the provided operating contract**

Include source precedence (official website, Participants repository, pinned local README/license mirror, then historical evidence), the exact STOPPED-to-RELEASE state machine, central coordinator duties, independent agent scope, `fact/inference/unknown/recommendation/source_paths` report fields, 4/8/2 search limits, no threshold sweep before activation, and three-valid-cycle pivot.

- [ ] **Step 3: Write `PROJECT_INFO.md` and `harness.config.json`**

Set `metrics.primary` to `completion_rate`; set tie-breakers to `median_finished_lap_ms`, `mean_incomplete_progress`, `p90_finished_lap_ms`, `collisions`, `damage`, and `act_latency_p95_ms`. Record official source precedence (current competition site, Participants repository, `docs/sources/official-participants/README.md` pinned mirror, historical local summary/evidence), legacy data map, new run/artifact/PDF roots, protected split identifiers, pivot count 3, search limits, plan-only default, and only allowlisted local operations. Do not add credentials.

- [ ] **Step 4: Reassign current document responsibilities without deleting evidence**

Reduce `README.md` to quick start and document map. Preserve its official template content in a verbatim, pinned Participants README snapshot under `docs/sources/official-participants/`; move the existing user-authored corridor benchmark narrative, including every condition/result/caveat, to a pre-v2 historical entry in `docs/strategy-history.md` and do not present it as a registered submission result. Keep competition facts in `COMPETITION_INFO.md` as a historical local summary and point to it from `PROJECT_INFO.md`. Move hard restrictions into `RESTRICTIONS.md`; move provisional S1/S2/S3 strategy choices from restrictions into historical strategy records. Freeze the existing generated `RESULTS.md` section and append a dated migration note above it. Preserve the current `SOTA.md` record as a historical local reference until a new candidate passes the new gates. Copy no run/artifact data.

- [ ] **Step 5: Configure generated-output ignores and create the requested output directory**

Add `runs/haic-research-v2/`, `artifacts/haic-research-v2/`, and `output/pdf/*.pdf` to `.gitignore`, then add `!output/pdf/.gitkeep`. Create `output/pdf/.gitkeep`; do not move the existing root `report.pdf`.

- [ ] **Step 6: Add source, experiment, handoff, strategy and report templates**

`docs/experiments/INDEX.md` lists only new v2 experiment records; `docs/handoffs/REPORT_TEMPLATE.md` has the five report fields; `docs/sources/INDEX.md` records URL, version/check date, pinned commit, official-source status, and supported claim. Preserve the official Participants README and LICENSE verbatim under `docs/sources/official-participants/` at commit `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`; note that the competition website was inaccessible on 2026-09-26 and must be rechecked before external action. `legacy-path-map.md` maps old paths without reading their contents; `strategy-history.md` is append-only and receives the moved pre-v2 benchmark with unregistered/ineligible status; `report.md` separates verified facts from inference and unknowns.

- [ ] **Step 7: Review and commit only the document-system files**

Run `git diff --check`, review `git diff -- README.md COMPETITION_INFO.md RESTRICTIONS.md RESULTS.md SOTA.md AGENTS.md PROJECT_INFO.md harness.config.json .gitignore docs output/pdf`, stage exact intended hunks only (the README already contains user changes), and commit with:
```powershell
git add -p README.md
git add AGENTS.md PROJECT_INFO.md harness.config.json COMPETITION_INFO.md RESTRICTIONS.md RESULTS.md SOTA.md .gitignore output/pdf/.gitkeep docs/experiments docs/handoffs docs/sources docs/strategy-history.md docs/report.md
git commit -m "docs: establish HAIC research operating contract"
```
Before staging, inspect the original README diff and preserve the user's benchmark content in `docs/strategy-history.md`; the root README should contain only the quick-start/document map. Expected: unrelated user source-code changes, historical experiment data, and run data are absent from the staged path list.

## Task 2: Implement the HAIC config loader and structural config checks

**Files:**
- Create: `haic_research/__init__.py`
- Create: `haic_research/config.py`
- Create: `tests/test_haic_research_config.py`
- Modify: `harness.config.json`

**Interfaces:**
- Produces: `load_config(root: Path) -> HarnessConfig`; `validate_config(config: HarnessConfig) -> list[ConfigIssue]`.
- `HarnessConfig` exposes `repo_root` plus resolved `run_root`, `artifact_root`, `legacy_paths`, `primary_metric`, `tie_breakers`, `search_limits`, `pivot_after`, `split_ids`, and `command_profiles`.

- [ ] **Step 1: Write config validation tests**

```python
class ConfigTests(unittest.TestCase):
    def test_completion_rate_is_required_primary_metric(self):
        config = make_config(primary_metric="lap_time")
        self.assertIn("metrics.primary must be completion_rate", messages(validate_config(config)))

    def test_new_roots_are_project_relative_and_distinct(self):
        config = make_config(run_root="../../outside", artifact_root="artifacts/haic")
        self.assertTrue(validate_config(config))
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m unittest tests.test_haic_research_config -v`
Expected: FAIL because the config model and validator do not exist.

- [ ] **Step 3: Implement `HarnessConfig`, `ConfigIssue`, `load_config`, and `validate_config`**

Parse JSON with the standard library. Reject absent required keys, unsupported schema version, unknown metric names, incorrect 4/8/2 or pivot values, path escapes, output roots equal to or inside `artifacts/haic/` or `submissions/`, unregistered commands, and secret-like config keys. Resolve relative paths from repository root.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `python -m unittest tests.test_haic_research_config -v`
Expected: PASS, including malformed JSON, missing path, primary metric and forbidden-root cases.

- [ ] **Step 5: Commit**

```powershell
git add haic_research/__init__.py haic_research/config.py harness.config.json tests/test_haic_research_config.py
git commit -m "feat: validate HAIC research configuration"
```

## Task 3: Add schemas and approval-gated workflow state

**Files:**
- Create: `haic_research/models.py`
- Create: `haic_research/state.py`
- Create: `tests/test_haic_research_state.py`

**Interfaces:**
- `WorkflowState` enum contains every state in `AGENTS.md`, including all four `GATE_REVIEW_<OUTCOME>` substates.
- `Approval` has `stage`, `plan_hash`, `approved_at`, and `source_ref`.
- `Hypothesis` has all eleven fields from the spec, with `source_paths: tuple[str, ...]` for cited evidence.
- `GateResult` contains gate name, one of `PASS/FAIL/UNKNOWN/NOT_APPLICABLE`, rationale, and evidence paths. `RunManifest` contains run ID, purpose, hypothesis/approval hashes, candidate/control revisions and package hashes, tool/runtime versions, data/map/split IDs, resource/permission limits, output paths and source hashes. `RunEvent` contains UTC timestamp, event kind, workflow state, approval reference, execution status/error, resource usage, mechanism signal, endpoint, and correction reference.
- `IntegrationReport` contains each gate result and cited evidence paths. `AgentReport` contains exactly `fact`, `inference`, `unknown`, `recommendation`, and `source_paths`. `CycleSummary` carries validity, protocol-match, improvement, and infra-invalid status. `WorkAssignment` carries direction, hypothesis IDs, owner, allowed read/write scope, and handoff path. These are immutable records in `models.py`.
- `transition(current: WorkflowState, requested: WorkflowState, *, plan_hash: str, approvals: Sequence[Approval], gates: Sequence[GateResult]) -> WorkflowState`.

- [ ] **Step 1: Write transition and schema tests**

```python
class WorkflowTests(unittest.TestCase):
    def test_cannot_execute_without_matching_execution_approval(self):
        with self.assertRaises(ApprovalError):
            transition(WorkflowState.EXECUTE_PENDING_APPROVAL, WorkflowState.EVALUATE,
                       plan_hash="new", approvals=[], gates=[])

    def test_every_evaluation_outcome_enters_gate_review(self):
        for outcome in (WorkflowState.ADVANCE, WorkflowState.REJECT,
                        WorkflowState.REVISE, WorkflowState.PIVOT):
            self.assertEqual(transition(outcome, WorkflowState[f"GATE_REVIEW_{outcome.value}"],
                                        plan_hash="p1", approvals=[], gates=all_gates()),
                             WorkflowState[f"GATE_REVIEW_{outcome.value}"])

    def test_only_advance_with_all_pass_gates_can_release(self):
        with self.assertRaises(GateError):
            transition(WorkflowState.GATE_REVIEW_ADVANCE, WorkflowState.RELEASE_IF_GATE_PASS,
                       plan_hash="p1", approvals=[], gates=unknown_gates())
        with self.assertRaises(TransitionError):
            transition(WorkflowState.GATE_REVIEW_PIVOT, WorkflowState.RELEASE_IF_GATE_PASS,
                       plan_hash="p1", approvals=[], gates=passing_gates())
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m unittest tests.test_haic_research_state -v`
Expected: FAIL because the state and schema modules do not exist.

- [ ] **Step 3: Implement typed schemas and explicit transition map**

Represent gate states as `PASS`, `FAIL`, `UNKNOWN`, `NOT_APPLICABLE`; model the exact state sequence in the spec. Each evaluation outcome enters only its matching `GATE_REVIEW_<OUTCOME>` substate and records exactly the three registered gates. Remove the caller-supplied `evaluation_outcome` override. Only `GATE_REVIEW_ADVANCE` with all three statuses `PASS` may enter `RELEASE_IF_GATE_PASS`; all other review substates may only stop without release. The ADVANCE review may also stop without release. Test the chained REJECT-to-review-to-release bypass. The current config defines no `NOT_APPLICABLE` exemptions; any future exemption requires an explicit validated config value and transition API support, and can never exempt rule compliance. A revised or pivoted effort starts a new cycle at `DISCOVER` with a new plan hash and fresh approvals; pivot preserves the prior checkpoint. Reject illegal transitions and stage/hash mismatch. `UNKNOWN` is never a pass. Store approvals as events; never infer approval from the presence of a plan file.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `python -m unittest tests.test_haic_research_state -v`
Expected: PASS for every valid transition and rejection case.

- [ ] **Step 5: Commit**

```powershell
git add haic_research/models.py haic_research/state.py tests/test_haic_research_state.py
git commit -m "feat: model HAIC workflow states and approvals"
```

## Task 4: Implement hypothesis batches, central coordination and completion-first policy

**Files:**
- Create: `haic_research/policy.py`
- Create: `haic_research/coordinator.py`
- Create: `tests/test_haic_research_policy.py`
- Create: `tests/test_haic_research_coordinator.py`
- Modify: `haic_research/models.py`

**Interfaces:**
- `ExperimentResult` carries `candidate_id`, `comparison_id`, `split_id`, `map_ids`, `seed_ids`, `completion_count`, `episode_count`, `median_finished_lap_ms`, `mean_incomplete_progress`, `p90_finished_lap_ms`, `collisions`, `damage`, `act_latency_p95_ms`, `rule_compliance`, and `mechanism_activation`.
- `ExperimentResult` also carries `official_score` as a separate reporting field and an eligibility label so teacher/smoke/diagnostic rows cannot enter submission-candidate ranking.
- `rank_candidates(records: Sequence[ExperimentResult]) -> list[ExperimentResult]` accepts one matched comparison group only and returns eligible rows only; ineligible observations remain in the experiment ledger.
- `promotion_decision(candidate: ExperimentResult, control: ExperimentResult) -> PromotionDecision`.
- `validate_batch(hypotheses: Sequence[Hypothesis]) -> None`; `should_pivot(cycles: Sequence[CycleSummary], limit: int = 3) -> bool`.
- `CycleSummary` and `WorkAssignment` are immutable `models.py` records. `assign_work(batch_id: str, hypotheses: Sequence[Hypothesis]) -> list[WorkAssignment]` gives each direction one owner and non-overlapping scope; it records assignments but does not spawn external agents itself.
- `AgentReport` remains an immutable record with exactly five fields. `AgentReportEnvelope` is a separate immutable record with `handoff_path`, `owner`, `edited_paths`, and `report: AgentReport`.
- `integrate_reports(assignments: Sequence[WorkAssignment], reports: Sequence[AgentReportEnvelope]) -> IntegrationReport` verifies complete scope ownership, unique handoff/owner matching, that each edited path is within the assignment's allowed write scope, and all five report fields before central synthesis.

- [ ] **Step 1: Write completion-priority and gate tests**

```python
class CompletionPriorityTests(unittest.TestCase):
    def test_higher_completion_rate_beats_faster_lap(self):
        reliable = result("reliable", completion_count=8, episode_count=10, median_finished_lap_ms=24000)
        fast = result("fast", completion_count=6, episode_count=10, median_finished_lap_ms=18000)
        self.assertEqual([r.candidate_id for r in rank_candidates([fast, reliable])],
                         ["reliable", "fast"])

    def test_unknown_rule_gate_cannot_promote(self):
        self.assertFalse(promotion_decision(result(rule_compliance="UNKNOWN"), result()).eligible)
```

Also test mismatched `comparison_id`, split/map/seed sets and episode denominator, invalid cycles, 3/8/2 batch boundaries, missing handoff fields, duplicate or overlapping assignment ownership, and rejection of teacher/smoke rows from candidate ranking.
Require compliance and mechanism activation statuses to each reject both `UNKNOWN` and `FAIL`.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m unittest tests.test_haic_research_policy tests.test_haic_research_coordinator -v`
Expected: FAIL because ranking, batch validation and coordination do not exist.

- [ ] **Step 3: Implement candidate ordering and promotion decisions**

Compute completion rate as `completion_count / episode_count`. Require identical comparison/split/map/seed protocol and episode denominator before comparing; raise `ComparisonMismatchError` for unmatched data. `rank_candidates` filters ineligible results and ranks eligible ones only, while the ledger retains every observation. Require both compliance and activation `PASS` for promotion, and promote only if the eligible candidate outranks the eligible control under the full completion-first ordering. Sort by descending completion rate, then ascending median finish time, descending incomplete progress, ascending p90, collisions, damage and latency. Keep official score as a separately reported field.

- [ ] **Step 4: Implement batch validation, pivot counter and work assignments**

Reject fewer than four independent directions, more than eight total hypotheses, or more than two per direction. `should_pivot` counts only valid comparable non-improving cycles; `infra_invalid` entries are ignored and do not increment the streak. Assignments contain direction, hypothesis IDs, owner label, allowed read/write scope and required handoff path. Wrap each unchanged five-field `AgentReport` in an `AgentReportEnvelope` carrying owner, handoff path and edited paths. `integrate_reports` rejects missing/duplicate or mismatched envelopes, out-of-scope edits, or reports missing any required field, then emits an integration report with all three gate results and evidence paths.

- [ ] **Step 5: Run the focused test to verify it passes**

Run: `python -m unittest tests.test_haic_research_policy tests.test_haic_research_coordinator -v`
Expected: PASS, including cases where a faster but less reliable candidate loses.

- [ ] **Step 6: Commit**

```powershell
git add haic_research/models.py haic_research/policy.py haic_research/coordinator.py tests/test_haic_research_policy.py tests/test_haic_research_coordinator.py
git commit -m "feat: prioritize completion in HAIC research policy"
```

## Task 5: Add isolated run records and immutable event history

**Files:**
- Create: `haic_research/records.py`
- Create: `tests/test_haic_research_records.py`
- Modify: `haic_research/models.py`
- Modify: `tests/test_haic_research_state.py`

**Interfaces:**
- `create_run(config: HarnessConfig, manifest: RunManifest, *, previous_run_dir: Path | None = None) -> Path` creates one directory only under configured `run_root`; a continuation may read only the explicitly named predecessor inside the new v2 run root.
- `append_event(run_dir: Path, event: RunEvent, *, config: HarnessConfig) -> None` appends one JSON line.
- `write_integration_report(run_dir: Path, report: IntegrationReport, *, config: HarnessConfig) -> None` writes the report once; corrections become new `events.jsonl` entries.
- `read_manifest(run_dir: Path, *, config: HarnessConfig) -> RunManifest` and `read_events(run_dir: Path, *, config: HarnessConfig) -> tuple[RunEvent, ...]` validate the configured persistence boundary before reading.
- Extend `RunManifest` with `plan_hash`, `cycle_id`, optional `checkpoint_ref`, `predecessor_run_id`, and `predecessor_decision_ref`; extend `RunEvent` with a stable `event_id`, structured approval fields, and optional `checkpoint_ref`. Add an immutable `CheckpointRef(path, sha256)` record. Approval events belong to exactly one run and identify the stage and approved plan hash.

- [ ] **Step 1: Write path and append-only tests**

```python
class RunRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config = config_in_temp_root(Path(self.temp_dir.name))
        self.manifest = manifest("run-001", cycle_id="cycle-001")
        self.run_dir = create_run(self.config, self.manifest)

    def test_run_directory_stays_under_configured_root(self):
        path = self.run_dir
        self.assertEqual(path.parent.name, "haic-research-v2")

    def test_events_append_without_replacing_prior_lines(self):
        append_event(self.run_dir, event("DISCOVER"), config=self.config)
        append_event(self.run_dir, event("HYPOTHESIZE"), config=self.config)
        self.assertEqual(len((self.run_dir / "events.jsonl").read_text().splitlines()), 2)

    def test_legacy_and_outside_paths_are_refused(self):
        with self.assertRaises(PathSafetyError):
            create_run(config_with_run_root(Path(self.temp_dir.name) / ".." / "artifacts" / "haic"), manifest("bad"))

    def test_new_cycle_requires_distinct_plan_and_parent_decision_event(self):
        decision = event("CYCLE_DECISION", state=WorkflowState.REVISE)
        append_event(self.run_dir, decision, config=self.config)
        append_event(self.run_dir, event("GATE_REVIEW", state=WorkflowState.GATE_REVIEW_REVISE), config=self.config)
        write_integration_report(self.run_dir, passing_report(), config=self.config)
        append_event(self.run_dir, event("STATE", state=WorkflowState.STOPPED), config=self.config)
        revised = linked_manifest("run-002", parent=self.manifest,
                                  decision_ref=decision.event_id, outcome=WorkflowState.REVISE)
        create_run(self.config, revised, previous_run_dir=self.run_dir)

    def test_pivot_carries_forward_the_exact_checkpoint_hash(self):
        checkpoint = CheckpointRef("artifacts/haic-research-v2/run-001/checkpoint.pt", "sha256:abc")
        prior = manifest("run-001", checkpoint_ref=checkpoint)
        prior_dir = create_run(self.config, prior)
        decision = event("CYCLE_DECISION", state=WorkflowState.PIVOT, checkpoint_ref=checkpoint)
        append_event(prior_dir, decision, config=self.config)
        append_event(prior_dir, event("GATE_REVIEW", state=WorkflowState.GATE_REVIEW_PIVOT), config=self.config)
        write_integration_report(prior_dir, passing_report(), config=self.config)
        append_event(prior_dir, event("STATE", state=WorkflowState.STOPPED), config=self.config)
        pivot = linked_manifest("run-002", parent=prior, decision_ref=decision.event_id,
                                outcome=WorkflowState.PIVOT, checkpoint_ref=checkpoint)
        create_run(self.config, pivot, previous_run_dir=prior_dir)

    def test_record_serializer_handles_frozen_mappings(self):
        record = read_manifest(self.run_dir, config=self.config)
        self.assertEqual(dict(record.tool_versions), {})

    def test_record_serializer_handles_nested_frozen_event_data(self):
        append_event(self.run_dir, event("RESOURCE", resource_usage={"cpu": {"seconds": 3}}), config=self.config)
        row = json.loads((self.run_dir / "events.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["resource_usage"]["cpu"]["seconds"], 3)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m unittest tests.test_haic_research_records -v`
Expected: FAIL because the record store does not exist.

- [ ] **Step 3: Implement confined run creation and append-only event writes**

Resolve and check all output paths against configured roots. Refuse path traversal, reused run IDs, symlinks escaping the root, writes into `artifacts/haic/` or `submissions/`, and writes to legacy manifests. Serialize dataclasses, enums, UTC timestamps, frozen mappings and tuples explicitly to stable JSON; do not use `dataclasses.asdict()` on `MappingProxyType` values. For a continuation, read only the exact predecessor named by the caller and verify it is a direct child of the configured v2 run root. Require matching predecessor run ID, a referenced `CYCLE_DECISION` event with outcome `REVISE` or `PIVOT`, a later `GATE_REVIEW` event in the matching `GATE_REVIEW_REVISE` or `GATE_REVIEW_PIVOT` substate and integration report, and a later `STOPPED` event. Require distinct run ID, cycle ID, and plan hash. The referenced parent decision must be the latest cycle decision. A `PIVOT` must carry the same non-empty selected checkpoint path and hash, taking the decision event checkpoint in preference to the initial manifest. Validate checkpoint confinement without reading/copying checkpoint data. Do not enumerate, import, or inspect legacy `runs/`, `artifacts/haic/`, or `submissions/`. Task 6 must source approvals only from the current run's events and require the exact current plan hash.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `python -m unittest tests.test_haic_research_records -v`
Expected: PASS; tests use temporary directories and do not touch project data.

- [ ] **Step 5: Commit**

```powershell
git add haic_research/records.py tests/test_haic_research_records.py
git commit -m "feat: add isolated HAIC run records"
```


**Persistence boundary addendum:** Every append/read/report call requires `config` and proves the run directory is an exact direct child of its run root, has a matching manifest run ID, and has no symlink, junction, hardlink, or path alias. Validate config before creating directories. Reject reused IDs, Windows reserved names, traversal and separators before writes. Use a per-run exclusive lock, failing closed on an existing or stale lock; manifest and report creation are exclusive. Event IDs are unique, timestamps are nondecreasing UTC, and corrections reference an existing event or immutable report. APPROVAL events contain `approval_stage`, `approved_plan_hash`, and optional authorization `approval_ref`; downstream `Approval.source_ref` identifies their unique event ID. Reports contain every registered gate exactly once, including FAIL/UNKNOWN/NOT_APPLICABLE observations without granting release permission. Continuations read only their explicitly named v2 predecessor and never import approvals.

## Task 6: Add typed command profiles and plan-only CLI execution

**Scoped files:** `haic_research/commands.py`, `haic_research/cli.py`, `haic_research/config.py`, `haic_research/records.py`, `harness.config.json`, `tests/test_haic_research_commands.py`, `tests/test_haic_research_cli.py`, `tests/test_haic_research_config.py`, and this Task 6 section.

**Execution sequence:** Write focused lifecycle/schema/CLI tests, observe the missing implementation failure, implement the contract below, run only focused fake-runner tests, inspect the exact scoped diff, and commit a coherent unit. Task 6's original illustrative snippets are superseded by this authorization contract. No real local operations, network, historical data inspection, or external submission actions are part of implementation verification.

### Task 6 authorization and lifecycle contract


This controller ruling supplements Task 6 after the independent approval audit. Preserve the user-approved HAIC-only scope, plan-only default, three separate approval stages, completion-first decisions and historical-data boundary.

1. One registered operation per new run. `plan` accepts explicit manifest metadata plus a profile and arguments, validates and normalizes them, writes a run through Task 5 and an immutable `execution_plan.json`, and prints its hash. No subprocess. The plan payload contains run/cycle IDs, all relevant manifest metadata (excluding plan_hash and approval_hash), profile ID, normalized arguments, profile/module/argument-schema/timeout fingerprint, source revisions, and the run-specific artifact destination. SHA-256 over canonical JSON is the plan_hash stored in the manifest and approved at every stage. Initial approval_hash may be empty; that means no approval exists, never implicit authorization. Actual approvals are recorded in events.
2. Execution reloads only that direct child of configured v2 run_root, validates manifest and complete event log, recomputes the canonical plan hash, checks the manifest metadata and current command-profile fingerprint, and uses only the persisted command. If the existing execute_approved signature retains profile/arguments/hash, they must exactly match the persisted normalized plan. Changed arguments, profile, source revisions or output path require a new plan/run and approvals.
3. Specify structured Task 5 approval fields explicitly: event_id, approval_stage, approved_plan_hash, timestamp and workflow_state. Approval.source_ref is the unique current-run event_id. Reject malformed JSON lines, duplicate event IDs, wrong stages/hashes, invalid UTC or decreasing timestamps, references absent from this exact log, or a referenced event that is not an APPROVAL. Never load approval lists from another run.
4. Replay workflow history through the state transition primitive. Plan registration creates STOPPED/DISCOVER/HYPOTHESIZE/DESIGN_PENDING_APPROVAL events from the supplied complete research metadata. `approve design` is legal only at DESIGN_PENDING_APPROVAL and advances to IMPLEMENT_PENDING_APPROVAL; implementation approval advances to EXECUTE_PENDING_APPROVAL; execution approval is legal only there and stays pending until run. All approve commands use the persisted current hash and require a source reference explaining the user's authorization. Reject bypassed or reordered stages and duplicate stage approvals.
5. Execution is at most once per run/approval: atomically reserve execution and append EXECUTION_STARTED before invoking the runner, then append finish/failure. A second or concurrent invocation is refused, including after failure/timeout. Retry requires a new run and approvals. Keep tests entirely on FakeRunner.
6. Add concrete typed argument schemas to config profiles: flag name, type, required status, permitted values/ranges where needed, and input/output path role. Validate schema/positive timeout in config.py and focused config tests (added to Task 6 file scope). Inspect real HAIC argparse definitions without executing modules. Force all output paths under artifact_root/run_id, and validate input paths explicitly rather than treating shell metacharacter rejection as the main boundary.
7. CLI plan/approve/status/run/report must be usable together; run defaults to preview and explicit --execute is required. report records the selected evaluation outcome and three gate results, uses matching gate-review substates, and only ADVANCE plus all PASS enters release. It never submits or confirms a competition model. Validation implementation arrives in Task 8; an unavailable validator must fail clearly instead of claiming success.

Tests must cover a complete valid lifecycle plus stale/cross-run approval, changed args with the same hash, profile drift, malformed/duplicate/out-of-order events, skipped approval stages, wrong output roots, repeated/concurrent execution, failed execution, default preview, and non-ADVANCE non-release. Use temporary directories and a fake runner; no historical data reads or actual training/evaluation/packaging.

Source inspection notes: train_policy and evaluate_closed_loop --output are directories; package_submission --output is a ZIP file; benchmark_corridor --output is a JSON file. Evaluation and packaging require both policy-checkpoint and dynamics-checkpoint. Benchmark repeated flags are --track, --site-map and --profile; do not invent unsupported flags. Training has --train-only-site-map-split and --defer-tune for a deliberately TRAIN-only operation. Register a useful explicit subset of actual arguments, not every historical research flag. Keep required read paths explicit; never discover old checkpoints automatically. Expose a reusable current-run history replay helper so Task 7 reporting can require an actual ADVANCE release path instead of promoting a REVISE/PIVOT report whose gates happen to PASS.

Task6 scope may extend records.py with confined immutable execution-plan read/write, read_integration_report, and execution-reservation helpers, reusing the same path/lock boundary. Do not introduce unguarded arbitrary file writes for plan/claim files. Plan source/hypothesis/resource metadata must remain reviewable in the hashed payload; if actual Hypothesis input is accepted validate its required fields. Synchronize Task6 plan text with this lifecycle before implementation. Do not redesign already-reviewed persistence or state beyond necessary integration helpers.

**Task 6 review correction:** Caller revision labels are insufficient executable identity. Include in the canonical plan the actual SHA-256 bytes and exact bounded inventory policy for non-recursive Python files in `training/`, `haic_agent/`, `core/`, and `core/vendor/`, plus explicit root `agent.py`, `env_wrapper.py`, and `damage.py`. Include the selected interpreter path/version/implementation identity and packaging's explicitly selected inference sources when `source-root` differs. Reject source redirections and hardlinks before opening them. Recompute this inventory before execution reservation; added, removed, or changed source files require a new run/plan/approvals. Do not require a clean Git checkout or inspect legacy/data roots. Installed third-party package binaries are outside this local source-byte inventory.

For split inputs, read only the exact split JSON and validate referenced map paths through the historical/no-redirection boundary before any runner call, without opening forbidden map targets or scanning directories. TRAIN-only validates only its loaded TRAIN group; evaluation and benchmark split loaders validate all three groups because the modules load all three. Revalidate split references after approval and before reservation so changed split contents cannot introduce a historical target.

## Task 7: Add result, experiment and SOTA reporting

**Files:**
- Create: `haic_research/results.py`
- Create: `tests/test_haic_research_results.py`
- Modify: `RESULTS.md`
- Modify: `SOTA.md`
- Modify: `docs/experiments/INDEX.md`
- Modify: `docs/report.md`

**Interfaces:**
- `append_experiment_summary(results_path: Path, experiment_path: Path, result: ExperimentResult) -> None` adds a dated record before the frozen generated block and links its full experiment file.
- `promote_sota(sota_path: Path, candidate: ExperimentResult, control: ExperimentResult, report_path: Path) -> PromotionDecision` updates only after all gates pass and completion-first comparison selects the candidate.

- [ ] **Step 1: Write append-only and completion-promotion tests**

```python
class ResultReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "SOTA.md"
        self.path.write_text("# SOTA\n", encoding="utf-8")
        self.report = Path(self.temp_dir.name) / "integration_report.json"

    def test_append_preserves_frozen_generated_section(self):
        before = "# Results\n\n<!-- BEGIN GENERATED RESULTS -->\nold rows\n<!-- END GENERATED RESULTS -->\n"
        path = write_fixture(before)
        append_experiment_summary(path, experiment_file(), result("r1"))
        after = path.read_text()
        marker = "<!-- BEGIN GENERATED RESULTS -->"
        self.assertEqual(after[after.index(marker):], before[before.index(marker):])

    def test_sota_does_not_promote_lower_completion_candidate(self):
        decision = promote_sota(self.path, result("fast", completion_count=6, episode_count=10), result("control", completion_count=8, episode_count=10), self.report)
        self.assertFalse(decision.promoted)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m unittest tests.test_haic_research_results -v`
Expected: FAIL because new-only result reporting is not implemented.

- [ ] **Step 3: Implement append-only records and SOTA gate**

Write full experiment detail to `docs/experiments/<run-id>.md` with manifest/report paths. Append a short row to the human-managed section of `RESULTS.md` without regenerating or rewriting the frozen JSON block. Update `SOTA.md` only for rule-compliant, mechanism-activated, matched candidates selected first by completion rate.

- [ ] **Step 4: Verify historical section preservation and run focused tests**

Run: `python -m unittest tests.test_haic_research_results -v`
Expected: PASS with byte-identical generated section and no data-directory reads.

- [ ] **Step 5: Commit**

```powershell
git add haic_research/results.py tests/test_haic_research_results.py RESULTS.md SOTA.md docs/experiments/INDEX.md docs/report.md
git commit -m "feat: record HAIC results with completion-first promotion"
```

## Task 8: Implement the harness-aligned structural validator

**Files:**
- Create: `haic_research/validation.py`
- Create: `scripts/harness/validate.ps1`
- Create: `tests/test_haic_research_validation.py`
- Modify: `haic_research/cli.py`

**Interfaces:**
- `validate_project(root: Path) -> list[ValidationIssue]`.
- `python -m haic_research.cli validate --root <repo-root>` returns exit code 0 only when all required documents/config entries and invariants pass.

- [ ] **Step 1: Write validator tests**

```python
class ProjectValidationTests(unittest.TestCase):
    def test_missing_required_handoff_template_is_reported(self):
        issues = validate_project(project_fixture(without="docs/handoffs/REPORT_TEMPLATE.md"))
        self.assertIn("docs/handoffs/REPORT_TEMPLATE.md", [issue.path for issue in issues])

    def test_plan_only_and_completion_priority_are_required(self):
        issues = validate_project(project_fixture(plan_only=False, primary_metric="lap_time"))
        self.assertGreaterEqual(len(issues), 2)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m unittest tests.test_haic_research_validation -v`
Expected: FAIL because project validation is not implemented.

- [ ] **Step 3: Implement document/config invariants and PowerShell wrapper**

Validate required paths from the reference tree, including the pinned local Participants README/LICENSE mirror; config schema and mirror path; official-source references; completion-first metric order; search/pivot limits; split list; allowed profile IDs; no secrets; and plan-only default. `scripts/harness/validate.ps1` must contain `$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path`, run `python -m haic_research.cli validate --root $RepoRoot`, then `exit $LASTEXITCODE`. Do not inspect `runs/`, `artifacts/haic/`, or `submissions/` contents.

- [ ] **Step 4: Run the validator and focused test**

Run:
```powershell
python -m unittest tests.test_haic_research_validation -v
.\scripts\harness\validate.ps1
```
Expected: tests pass and validator exits 0 on the completed document/config structure.

- [ ] **Step 5: Commit**

```powershell
git add haic_research/validation.py haic_research/cli.py scripts/harness/validate.ps1 tests/test_haic_research_validation.py
git commit -m "feat: validate HAIC harness operating structure"
```

## Task 9: Cut over and retire the former orchestration path

**Files:**
- Modify: `README.md`
- Modify: `COMPETITION_INFO.md`
- Modify: `AGENTS.md`
- Remove after unique rules are migrated: `RULES.md`
- Remove after new-flow verification: `research_ops/__init__.py`, `research_ops/index.py`, `research_ops/policy.py`, `research_ops/orchestrator.py`, `research_ops/cli.py`, `research_ops/build_report.py`
- Remove after new-flow verification: `training/improvement_loop.py`
- Remove obsolete tests after replacement coverage passes: `tests/test_research_ops.py`, `tests/test_research_ops_cli.py`

**Interfaces:**
- New supported entry point: `python -m haic_research.cli`.
- The former `python -m research_ops.cli improve` and `python -m training.improvement_loop` paths have no compatibility shim.

- [ ] **Step 1: Enumerate all references to the retired orchestration APIs**

Run:
```powershell
rg -n "research_ops|training\.improvement_loop|sync-results|improve .*--execute|--command" --glob '!docs/superpowers/plans/**' --glob '!docs/superpowers/specs/**' .
```
Expected: references are classified as active links to update or historical provenance to retain.

- [ ] **Step 2: Move unique operating rules and command details into canonical docs**

Transfer still-valid content from `RULES.md` to `AGENTS.md`/`PROJECT_INFO.md`; keep old generated-results provenance in the frozen `RESULTS.md` block. Replace active README commands with `python -m haic_research.cli` examples and link from `COMPETITION_INFO.md` to canonical policy docs.

- [ ] **Step 3: Delete the old orchestrators and their obsolete tests**

Remove the listed source modules and tests only after Task 2–8 tests pass. Do not delete or regenerate JSON runs, checkpoints, submission ZIPs, source notes or result history.

- [ ] **Step 4: Run new unit suite and structural validator**

Run:
```powershell
python -m unittest tests.test_haic_research_config tests.test_haic_research_state tests.test_haic_research_policy tests.test_haic_research_coordinator tests.test_haic_research_records tests.test_haic_research_commands tests.test_haic_research_cli tests.test_haic_research_results tests.test_haic_research_validation -v
.\scripts\harness\validate.ps1
```
Expected: all harness unit tests pass; validator exits 0; no train/evaluate/package/submit profile runs.

- [ ] **Step 5: Confirm source-tree boundary and commit the cutover**

Run:
```powershell
rg -n "research_ops\.cli|training\.improvement_loop|sync-results" README.md AGENTS.md PROJECT_INFO.md RESTRICTIONS.md haic_research
 git status --short
```
Expected: no active reference remains; historical citations are confined to `RESULTS.md`, `docs/strategy-history.md`, and `docs/sources/legacy-path-map.md`; the data roots were not staged.

Commit with:
```powershell
git add -p README.md
git add -u COMPETITION_INFO.md AGENTS.md
git commit -m "refactor: retire legacy HAIC research orchestration"
```

`research_ops/`, `training/improvement_loop.py`, their tests, and `RULES.md` are currently untracked in the selected checkout. After replacement coverage passes and unique rules are migrated, remove those files from the working tree as authorized by the cutover, but do not add their removals to the commit as tracked removals. Before staging README hunks, inspect the diff and retain only intended command/document-map changes; leave pre-existing user edits unstaged.

## Execution Notes

- Execute tasks in order; each task is a reviewable commit.
- During implementation use `superpowers:executing-plans` for native execution or `superpowers:subagent-driven-development` if the user selects delegated execution.
- Completion rate is the promotion objective; official rule compliance remains a hard gate.
- The plan authorizes local harness unit tests and document validation because the user asked to check compliance. It does not authorize training, simulation, official evaluation, package generation, submission, or model confirmation.
