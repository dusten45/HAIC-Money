# Competition-Aligned Camera Evaluation V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the already implemented camera controller should replace the live bare `Agent` route under a fresh evaluation aligned with the official finish-first ranking.

**Architecture:** Freeze the unchanged `_BoundedSideHoldController` selector-only source and current control source as distinct byte snapshots. A new immutable protocol binds fresh geometry seeds, environment/source/runner/decision hashes and fixed phase gates before any v3 episode. Cold CPU workers run screen, confirmation and blind sequentially, with source checks and deterministic repeats; the selector changes only if all three phases retain.

**Tech Stack:** Python 3.11, NumPy 1.26, Torch 2.1 CPU, local official-generator simulator, pytest, Git.

**Spec:** `CONTEXT.md`, `experiments/camera-policy-generalization-v2-result.json`, and the [official Participants README](https://github.com/2026-HAIC/Participants#readme), whose ranking compares completed lap times first and places all unfinished laps behind finishes.

## Global Constraints

- V2 screen was rejected under its original per-cell safety gates. Its confirmation and blind phases stay sealed. V3 answers a distinct competition-aligned utility question and never counts a v1/v2 geometry as fresh.
- Candidate algorithm is byte-identical to the committed `f12c5b62c5f02bff4053e6f4775d202d79917938` source except the bare selector. No new tuning is permitted after v3 binding.
- Cross the same geometry seeds with track IDs 1–4; use 8/16/8 seeds for screen/confirmation/blind and 2/4/2 repeat pairs. Seed derivation is SHA256 of name, one-time random salt, phase and index. Confirm freshness against all documented experiment seeds before binding.
- Phase RETAIN requires net completed laps at least 3/6/3, lost control finishes at most 2/4/2, candidate total crashes no greater than control, candidate contacts and damage no greater than control, candidate mean progress at least control, and candidate shared-finish total time no more than 1.10 times control. Empty shared-finish sets skip only that time ratio. All cells and repeats must validate without operational error. The combined 128 canonical pairs must have at least 18 net additional finishes before activation.
- Within a phase, reject a geometry seed if the candidate loses at least two control finishes across its four track IDs and its net finish delta on that seed is negative. Across all three phases, require the candidate finish count to be at least the control count separately for each of track IDs 1–4. These subgroup guards are fixed before v3 salt generation to protect reliability across track layouts.
- The metrics and source checks are fixed before seed generation. Per-cell contacts and rare both-DNF progress losses remain reported, but do not veto an otherwise better aggregate competition result. This is an explicit decision change from v2, justified by the official finish-first ranking and applied only to new data.
- The consumed v2 screen would retrospectively pass these v3 screen gates. State that post-data fact plainly in the final result; it is not v3 evidence and does not reverse v2's REJECT decision.
- Require the exact f12 agent blob SHA256 `291d93081a64d49f18507ec7baaba411e06510bb533221ce07ae585bc4e77b61`, `_BoundedSideHoldController`, and root `model.pt` path before generating salt. Import no unpinned executable v1/v2 runner helpers. Validate duplicate/unexpected receipt coordinates and every repeat's runtime metrics.
- Official limits: import/init 10 s, reset/action 5 s, process 1,024 MiB; action shape `(3,)`, finite steer `[-1,1]`, gas/brake `[0,1]`; max steps 2,000, frame skip 4. Never edit `core/`, `env_wrapper.py`, or `damage.py`.
- Preserve user untracked videos, old submissions, and `submission.zip`. Commit/push each coherent verified checkpoint. Do not create a new worktree or top-level task for delegation.

## Review Focus

- Missing, duplicated or invalid cold receipts must make a phase incomplete or rejected; a summary cannot silently omit a cell.
- A single new crash can be allowed only if total candidate crashes do not exceed control; the report must still expose its cell and progress.
- A phase must expose finish counts by geometry seed and track ID so the subgroup guards and every lost control finish can be audited.
- Confirmation and blind cannot run before their predecessor retains under the fixed v3 gate, regardless of how strong the aggregate looked.
- A source byte, model, helper, environment, runner or decision-engine change after binding must abort before and after every phase.
- A repeat pair must compare action hashes and outcome fields but must never count as a new statistical observation.
- A fresh checkout must reconstruct ignored source snapshots from the pinned Git blob and selector without changing the bound salt or overwriting a mismatched existing snapshot.

---

### Task 1: Fixed competition comparator

**Files:** Create `tools/competition_camera_gate_v3.py`; create `tests/test_competition_camera_gate_v3.py`.

**Interfaces:** `compare_pairs(rows: list[dict], cells: list[tuple[int,int,int]], phase: str) -> dict` consumes the existing cold receipt schema and produces all gate metrics, flagged pairs, reasons and `RETAIN`/`REJECT`/`INCOMPLETE`. `combined_decision(phase_summaries: dict[str,dict]) -> dict` applies the 18-finish combined gate.

- [ ] Write failing tests for each exact phase threshold, aggregate crash/contact/damage and progress/pace gates, geometry-seed loss clusters, combined track noninferiority, empty shared finishes, invalid/missing rows, and repeat exclusion.
- [ ] Run the focused tests and confirm they fail for the missing module.
- [ ] Implement only the fixed comparator and combined decision functions.
- [ ] Run focused tests and confirm pass; inspect comparator source and its Git-byte hash.
- [ ] Commit the comparator and tests with a one-line message.

### Task 2: Source-bound v3 runner and sealed protocol

**Files:** Create `tools/compare_camera_policy_competition_v3.py`; create `experiments/camera-policy-competition-v3.template.json`; create `tests/test_camera_policy_competition_v3.py`.

**Interfaces:** The runner reuses v2's cold worker and source-binding shape but imports Task 1's comparator. CLI subcommands `bind`, `preflight`, `run-phase`, `report-phase`, `freeze-finalist`, `seal-confirmation` accept only the v3 protocol and output root. `bind` writes new source snapshots and the protocol exactly once.

- [ ] Write failing tests for template validation, fresh unique seed generation, selector-only source construction, sealed phase ordering, byte tampering and exact repeat checks.
- [ ] Run tests and confirm the expected failures.
- [ ] Implement the v3 runner with new name/path/hash identity and fixed Task 1 gate; keep v2 files immutable.
- [ ] Implement `restore-snapshots` from the pinned committed source, and verify source/model/helper/environment/runner/gate hashes again after report and seal paths, including resumed phases with no new jobs.
- [ ] Run focused tests and preflight; review diff and run `git diff --check`.
- [ ] Commit and push the runner, tests and unbound template.
- [ ] Bind a one-time v3 salt and 8/16/8 new seeds to the committed source and runner; verify no historical collision and commit/push the immutable protocol before any v3 simulation.

### Task 3: Cold evaluation and promotion decision

**Files:** Create `experiments/camera-policy-competition-v3-result.json`; modify `CONTEXT.md`; modify `agent.py` only if all gates retain.

**Interfaces:** Each `run-phase` writes 32/64/32 canonical paired receipts and 2/4/2 deterministic repeat pairs. `report-phase` independently recomputes summaries from validated receipts. The result file records all hashes, phase decisions, per-track completion, aggregate safety/pace and runtime maxima.

- [ ] Run screen cold workers and audit every receipt, input hash, repeat and summary.
- [ ] If screen RETAIN, freeze finalist; run confirmation and audit it. If confirmation RETAIN, seal and run blind, then audit it.
- [ ] Apply combined 18-finish gate only after all three phases retain; if any gate fails, record rejection and keep active route unchanged.
- [ ] Recompute all three summaries from validated receipts, verify both predecessor seals and matching identities, and report per-track and per-geometry outcomes plus the retrospective v2-screen disclosure.
- [ ] If all gates retain, change only the bare selector to the exact frozen candidate class; verify Agent action parity on registered repeat cells and static submission constraints.
- [ ] Run focused and broad available tests, inspect staged diff, commit and push result and any validated selector change. Do not overwrite the user's existing `submission.zip`.
