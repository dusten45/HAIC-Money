# Autonomous Camera Upgrade Implementation Plan

> **For agentic workers:** Use parallel in-session implementation and independent review. The user explicitly requested autonomous analysis, upgrading, and prospective relaxation after repeated performance failures.

**Goal:** Promote and package a faster, more reliable camera policy with completed, source-bound evaluation across all four track IDs.

**Architecture:** Retain the official simulator and camera/action contract. Correct candidate-only stale obstacle confidence after road loss, audit the complete consumed development grid, then evaluate frozen source on new geometry. Preserve historical decisions and sealed blind data.

**Tech Stack:** Python 3.11, Torch 2.1 CPU, NumPy 1.26, existing cold episode workers.

**Spec:** The current user request and `AGENTS.md`; this plan prospectively supersedes the obsolete unbound V4 threshold requirement in `CONTEXT.md`.

## Constraints and decisions

- Keep `core/`, `env_wrapper.py`, and `damage.py` unchanged. Official participant HEAD remains `dfb7a2de2178825ca5c5ce20bab01ba67052ba31`.
- Work on the existing `feature/DG-Codex` branch; preserve user videos, submission directories and `submission.zip`.
- Do not reopen historical blind phases or reinterpret historical rejected studies as passes.
- V1, V2 and V3 are three documented consecutive generalization rejections. Select the practical V4 profile before binding any V4 seeds: phase finish gains at least 2/4/2; lost control finishes at most 3/6/3; combined finish gain at least 12.
- Preserve per-track finish noninferiority, seed-cluster floors, nonincreasing aggregate crashes/contacts/damage, mean progress noninferiority, exact repeats, complete receipts, source pins, and shared-finish time ratio at most 1.10.
- Development audit: all 96 consumed cells, no lost previous candidate finishes, both documented road-exit rescues, no added crashes/contacts, shared-finish ratio at most 1.10. After two failed development revisions, a new preregistered revision may permit at most one extra contact in 96 cells; it must still satisfy the other floors.
- Bound search to two development implementations and two fresh protocols. Aim for a 60-minute experiment window, allow at most 90 minutes for completed receipts and packaging. If V4 rejects, a separate fresh protocol may require gains 1/2/1 and combined gain 8, retaining all other floors. Never rescore the rejected dataset under a new profile.
- Report observed completion and pace honestly; finite testing cannot guarantee completion on every possible shape.

## Task 1: Correct candidate obstacle freshness

**Files:** `agent.py`, `tests/test_clear_road_row42_dropout.py`.

- [x] Reproduce stale confidence with a visible obstacle, lost-road decisions, and low-speed reacquisition.
- [x] Add failing observation-sequence tests for road loss, invalid current camera frames and reset.
- [x] Add candidate-only confidence state; invalidate on road loss/invalid current camera frames and reset; arm on visible-road obstacle evidence. Preserve inherited avoidance state and the existing three visible missing-decision window.
- [x] Run focused policy/submission tests and independently review the diff.
- [x] Commit and push the isolated fix (`7df85ae`).

## Task 2: Complete consumed development audit

**Artifacts:** `.haic-artifacts/clear-road-row42-v4/freshness-audit/`; tracked result under `experiments/`.

- [x] Freeze actual candidate source with one bare `Agent` selector swap.
- [x] Validate every archived baseline receipt and action-exact baseline replay against the consumed V3 summaries.
- [x] Run 96 cold candidate episodes, retaining source/runtime hashes and complete metrics.
- [x] Recompute aggregate/per-track results independently; inspect every changed outcome and both known rescue cells.
- [x] Persist a concise source-bound development result.

## Task 3: Preregister and run fresh practical V4

**Files:** V4 gate, runner, template and their tests; bound V4 protocol.

- [x] Add tests where practical thresholds accept valid paired results but safety/integrity violations reject.
- [x] Implement practical thresholds only in V4; preserve V3 bytes and decisions.
- [x] Pin candidate commit/blob and decision engine bytes, test, commit and push (`71bb2f4`).
- [x] Bind fresh disjoint seeds, commit and push the protocol before episodes (`0d11229`).
- [x] Run screen, confirmation and blind only after their predecessor gates pass. Validate exact cold repeats and final combined decision (all retained; independent 272-receipt audit).

## Task 4: Promote, package and report

**Files:** `agent.py`, policy selector test, `CONTEXT.md`, V4 result.

- [x] Promote only the source-bound passing candidate and update the actual-route test.
- [ ] Run the Windows-compatible suite and submission static/CPU checks; report the known Windows `fcntl` exclusion.
- [ ] Verify promoted source equals the evaluated selected candidate bytes exactly; the single selector swap is relative to pinned base commit `7df85ae`. Test cold package action parity.
- [ ] Build an immutable package under ignored `.haic-artifacts/submissions/`, preserve user packages, and record archive/source/model hashes.
- [ ] Update concise project state, commit and push coherent progress. Deliver measured outcomes, package path and validation limitations.

## Review focus

- Stale obstacle confidence after missing/invalid current road frames. Official observations provide all four finite frames in range; older-frame corruption is outside this study's input contract.
- Fresh obstacles rearming after road reacquisition and resets clearing the new confidence.
- Pace summaries using common finishes rather than comparing unmatched completed laps.
- Integrity/runtime errors never becoming acceptable through statistical relaxation.
- Promotion/package bytes matching the exact evaluated source route.
