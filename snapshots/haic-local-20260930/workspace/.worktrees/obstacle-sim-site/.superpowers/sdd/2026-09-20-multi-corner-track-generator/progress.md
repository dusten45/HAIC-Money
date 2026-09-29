# SDD ledger — plan: docs/superpowers/plans/2026-09-20-multi-corner-track-generator.md

## Setup

- Spec: `docs/superpowers/specs/2026-09-20-multi-corner-track-generator-design.md` (user-approved).
- Worktree: `C:/Users/koi/Coding/HAIC/.worktrees/obstacle-sim-site`, branch `codex/obstacle-sim-site`; linked worktree confirmed.
- Pre-existing index: seven staged Task 7 web/API files; do not rewrite or discard them. `git diff --cached --check` passed.
- Commit signing: `commit.gpgsign=true`, `gpg.format=ssh`, configured SSH key present. Previous turn reported a 1Password signer failure; signed preflight retry is required before implementation. Never bypass signing.
- Baseline before feature work: `unittest discover -s tests -v` passed (57 tests, 1 existing server-repository skip); `node --check web_simulator/app.js` passed.

## Shared-interface preflight

| Producer → consumer | Interface | Finding |
|---|---|---|
| Task 1 → Task 2 | Python template IDs, profile metadata, candidate builder, geometry validator | Sequential by design: Task 1 creates deterministic generation; Task 2 adds clearance validation and bounded retry around that generator. Keep candidate construction separate from retries. |
| Task 1 → Task 3 | PRNG, recipe token format, rounded centerline, generator metadata | Port exact draw order, zero-seed normalization, quantization, and JSON-compatible metadata to JavaScript; parity tests compare exact output. |
| Task 2 → Task 3 | Curvature, road-boundary, retry validity rules | Mirror the same thresholds and first-valid-candidate rule in browser fallback; obstacles remain on their existing independent RNG. |
| Tasks 1 and 3 → Task 4 | `generate_custom_map()` output, API/CLI serialization, preview summary | Preserve schema 2 and current request contract; document the new template without adding template-specific API/CLI branches. |

Pre-flight ruling: none. The task interfaces are consistent with the approved spec and plan.

## Task checklist

- [x] Task 1: Versioned Python corner-profile generation (commit `e431492`; BASE `f5c5a947cc2c74af2befc1335bb5d3c1fc66efd2`).
- [x] Task 2: Width-aware geometry safety and Box2D smoke coverage (commit `a89802e`; BASE `e4314921ec9803545ba1e236f2b50e74f92531b6`).
- [x] Task 3: Browser template selection, summary, and generator parity (commit `fdc3ca7`; BASE `a89802e1180b08d47455635192d49ee2ef628eb6`).
- [x] Task 4: CLI/API coverage, user documentation, and end-to-end verification (commit `89aa387`; BASE `fdc3ca701aa493cba5edb32526a25bb0d20e40ec`).

## Commit preflight

- [x] Commit the pre-existing staged Task 7 change using normal SSH signing: `240acfa`.
- [x] Commit the user-approved spec and this plan separately using normal SSH signing: `f5c5a94`.
- [x] Verify the index is clear before Task 1. `git status --short`, `git diff --cached --quiet`, and `git diff --quiet` confirmed a clean worktree at `f5c5a94`.

Preflight complete. No unsigned commit or index rewrite was used.

## Task 1 evidence

- RED: the new focused profile test failed because `technical` was absent; full generator tests also showed the missing-template error and a 4.8668-unit segment violating the 4-unit sampling limit. Existing invalid-seed and unknown-template behaviors already passed and are retained as regressions.
- Debug evidence: normalized gap weights can fall outside 20°–100° for deterministic seeds, and a 32-draw retry still exhausted for `oval, seed=13`. A deterministic experiment projected 500 generated gap vectors (5 templates × seeds 0–99) toward equal spacing by the largest common factor; all 500 met both bounds.
- Ruling: replace gap rejection/redraw with that bounded projection, and mirror it exactly in JavaScript. This is a narrow plan deviation: no candidate budget is spent on a constraint that can be satisfied by projection, while seeded angular differences remain unless a bound requires compression. Cost if wrong: gap variance may be narrower than intended; template-specific anchor radii and profile recipes still provide variation.
- GREEN: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator -v` passed (10 tests).
## Task 2 evidence

- RED: tight-radius test failed before the radius check was added. Its first circle fixture was rejected earlier by the existing minimum-length rule; replaced it with a long, non-self-intersecting loop containing a radius-2.12 corner so the intended curvature behavior is isolated.
- Debug evidence: seed 73 initially produced no valid chicane candidate in 32 attempts. Near-distance diagnostics showed consecutive centerline sections on a single local bend were being counted as overlapping road, and geometry.width is used by `CustomCarRacing` as each side's offset distance (half-width), not the full road width. Using the actual half-width for curvature clearance and ignoring along-track pairs within `max(4 * width, 8)` removed those false positives. Class-specific target radii now size quadratic fillets; retries still reject genuine tight curvature and nonlocal overlap.
- GREEN: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator -v` passed (15 tests, including five templates × 100 seeds and width endpoints 0.5/100).
- GREEN: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_custom_environment -v` passed (4 tests; every template created, reset, and stepped in Box2D).
- Ruling: retain the implementation's half-width semantics and local arc-separation exclusion, and mirror both in Task 3 JavaScript. Cost if wrong: an overly permissive local exclusion could miss a very tight self-approach, mitigated by the independent turn-radius, boundary-intersection, and nonlocal distance checks.

Task 1: complete (commit `e431492`, tests: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator -v` → OK)

## Task 3 evidence

- RED: asset checks showed the `technical` option and summary target absent; old fallback output used only the older 48-point generator and metadata omitted generator version/corner recipe. Unknown templates and invalid seeds were not rejected consistently. The test subprocesses initially hit Windows CP949 decoding on Node's Unicode output; new checks explicitly decode UTF-8.
- GREEN: `node --check web_simulator/app.js` passed; `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets tests.test_web_simulator_client -v` passed (13 tests), including exact Python/JavaScript centerline and metadata parity for every template at seeds 0, 42, and 4294967295, plus width endpoints 0.5 and 100.
- Ruling: increase the browser loader's centerline cap to the generator's 4096-point ceiling; keep the existing 256-point manual-edit cap and replace the large editable table with an explanatory message for denser generated tracks. Clear recipe summary metadata when users manually edit centerline geometry so the displayed profile never claims a stale recipe.
Task 2: complete (commits e431492..a89802e, tests: D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator tests.test_custom_environment -v → OK)
Task 3: complete (commits a89802e..fdc3ca7, tests: D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets tests.test_web_simulator_client -v → OK)

## Task 3 integrated browser review

- The worktree version was served separately at `http://127.0.0.1:8766/`; the pre-existing process on port 8765 was left untouched. The same seed (42) produced distinct point counts and coordinate fingerprints: oval 78 / `ea62b963`, S-curve 88 / `6bc2e08b`, hairpin 86 / `e7aaf839`, chicane 91 / `58f70e82`, and technical 94 / `a0e1846e`.
- Changing only the template did not alter the current map until generation was requested. The summary then updated to match the generated profile. No map-save action was used, so the browser review added no user map artifacts.

## Task 4 evidence

- API and CLI regression tests verify `technical` keeps map schema 2 and writes generator version/corner metadata without template-specific API branches.
- README documents all five styles, seeded reproducibility, a generation command, optional obstacle separation, and that these local maps do not replace official evaluation tracks.
- Visual review exposed that a fresh official map initially displayed the legacy text `직접 제작 맵`. A regression test first failed on that mismatch; the page now initializes the summary from the selected map and displays `공식 트랙` by default. Focused asset tests passed afterward.
- GREEN: `D:/HAIC/haic-env/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v` passed (75 tests; one pre-existing skip because the server repository is unavailable).
- GREEN: `node --check web_simulator/app.js` and `git diff --check` passed. The integrated worktree page at port 8766 generated the `technical` seed-42 track with 94 control points and its expected 10-corner summary.
- Task 4: complete (commits fdc3ca7..89aa387, tests: `D:/HAIC/haic-env/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v` → 75 passed, 1 skipped).

## Final whole-branch review

- Fresh reviewer ran read-only against review package `50ee204..89aa387`; no Critical findings. Reviewer found five Important issues: recipe directions/classes do not match measured turns; width-100 seed 73 starts outside the simulator playfield; API map-save/run paths bypass the custom-geometry safety validator; the UI truncates fractional/blank design seeds; and manual driving does not render the live vehicle state. These are verified findings and enter the single fix pass.
- Deferred minor: switching from a generated custom map to an official map can leave the previous custom corner summary visible. This is a UI-label issue and is not entering the fix pass; cost if wrong: users may briefly see a stale summary until the next map generation/reload.
- Reviewer declined to assess older security, legacy simulator, and README issues as regressions from this branch; these are not certified safe by this review. The earlier 50ee204..pre-240acfa history and live browser manual session were not exhaustively reviewed/repeated.
- Commit `89aa387` contains an SSH `gpgsig` and `commit.gpgsign=true`; local signature verification cannot run because `gpg.ssh.allowedSignersFile` is not configured. No signing configuration was changed.
- Fix-pass outcomes (each finding received a reproducing regression test before the correction):
  1. Corner recipe metadata now records measured turn angle and radius; generated signed turns, hairpin angle, and radius class are validated. Python and browser fallback both use generator version 3 and exact parity tests. Cost if wrong: the summary could still mischaracterize a curve; measured-geometry assertions cover every template at widths 8 and 9 across four boundary/representative seeds.
  2. The approved design explicitly rejects silently replacing a requested multi-corner track with an oval. Generated half-width is therefore capped at 9, and larger requests return a clear range error; candidate retry budget is 64. A 5-template × 100-seed width-9 sweep passed, including reset and first-step checks for all templates. Cost if wrong: this restricts generated road width, but preserves the requested shape and stays inside the fixed playfield; the UI, Python API, CLI, README, and spec use the same limit. Imported/saved maps are still subject to runtime geometry safety validation.
  3. Map saving and environment creation now validate custom geometry at their API boundaries. Tests confirm unsafe geometry returns an error and is not written or run. Cost if wrong: an unsafe legacy map might still be inspectable, but cannot pass these save/execute boundaries.
  4. Browser design-seed input now rejects blank and fractional values rather than truncating them. Cost if wrong: an invalid seed could make the user's requested track differ; input-level regression tests cover both cases and Python/JavaScript generator parity remains exact.
  5. Manual-run start returns the selected track snapshot and each action redraws the vehicle at its returned position/orientation. Renderer tests verify the live marker path. Cost if wrong: the view could lag the simulator; API snapshot and browser canvas tests exercise the same manual-run state flow.
- GREEN: `D:/HAIC/haic-env/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v` passed (84 tests, 1 pre-existing skip because the server repository is unavailable; 244.745s). `node --check web_simulator/app.js` and `git diff --check` passed. The integrated worktree site at `http://127.0.0.1:8766/` visually generated the technical 10-corner map with the updated maximum-width label. A browser manual run was not started, so no run artifact was written to the user's data directory.
- Remaining minor: switching from a generated custom map to an official map can leave the previous custom corner summary visible. This was not part of the Important-finding fix pass; cost if wrong: a stale label may briefly confuse the user until regeneration/reload. Track shape, execution, and saved data are unaffected.
- Scope caveat: the independent review did not assess older security, legacy simulator, or README issues as regressions from this branch; those are not certified safe by that review. No second independent review was run after this focused fix pass.
- Fix pass: complete. All five Important findings have regression coverage; no fallback that changes the selected template is present. SSH signing configuration was left unchanged.
Task 4: complete (commits fdc3ca7..fdc3ca7, tests: D:/HAIC/haic-env/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v → custom obstacles: 0)
