# SDD ledger — plan: docs/superpowers/plans/2026-09-18-custom-tracks-and-run-simulator.md

Recovered progress: Tasks 1–6 are committed on this branch as `7b7fe65`, `4755f7e`, `0163a77`, `c8f07f0`, `6786c62`, and `d3bb863`. The prior session summary reports the complete suite passed with 51 tests and one expected skip. No prior SDD ledger was present, so these completed tasks are recovered from git history and session summary rather than rerun.

Pre-flight interfaces for Task 7:
- Task 1 → Task 7: schema-2 official/custom map contracts and schema-1 compatibility are available in `map_from_dict`/`map_to_dict`; browser normalization must preserve both kinds.
- Task 2 → Task 7: deterministic custom generator accepts `map_id`, `design_seed`, `template`, `width`, `max_steps`, and `frame_skip`; local `/api/maps/generate` has the matching JSON contract.
- Task 3 → Task 7: previews use `track.points` entries `(progress, normal_angle, x, y)` and world-unit width; browser rendering can reuse world-coordinate projection.
- Task 5 → Task 7: run logs carry schema 2, embedded map, track snapshot, steps, summary, optional frames; action order is `[steer, gas, brake]`.
- Task 6 → Task 7: same-origin health/generate/save/agent/start/action/finish APIs exist. The agent endpoint is synchronous; manual sessions are stateful. Baseline automatic execution can use the existing `/api/runs/start` policy and action route without adding a server endpoint.

Task 7: in progress (BASE `d3bb86306b88515f9c25e5dd85d0e2bfb8d40316`).
Task 7: RED — `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets -v` → 6 tests, 2 expected failures for missing custom-map/run-control DOM and API/manual client behavior.
Task 7: Ruling: add `POST /api/runs/auto` for synchronous baseline execution — Task 7's browser acceptance expects a baseline run to produce a complete log, while Task 6 exposes only the Agent batch route; one bounded API route avoids thousands of per-step HTTP calls and only allows `agent` or `baseline` — cost if wrong: a route outside the original Task 7 file list must be maintained.
Task 7: API RED — `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_api.TestLocalSimulatorApi.test_baseline_auto_run_returns_and_saves_run_log -v` → expected 404 for the not-yet-implemented `/api/runs/auto`.
Task 7: Static fallback defect reproduced in the browser — custom generation raised `design_seed is not defined`; the inline payload used an undefined snake_case variable while the function parameter was `designSeed`.
Task 7: Fallback RED — `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_client -v` → expected missing `makeBrowserCustomMap` behavior.
Task 7: Fallback GREEN — `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_client tests.test_web_simulator_assets -v` → 7 tests passed; `node --check web_simulator/app.js` passed; static browser server generated a 48-point chicane with no console errors.
Task 7: Browser GREEN — integrated site generated/validated/saved `custom-track-ui-check` (48 centerline points) to `D:\HAIC\maps`; Agent, baseline, and manual runs produced logs and loaded them into replay; pause/step/finish and static file-only fallback were exercised without console errors. Test run outputs are preserved under `D:\HAIC\runs`.
Task 7: GREEN — `node --check web_simulator/app.js`, `git diff --check`, and `D:/HAIC/haic-env/Scripts/python.exe -W ignore::DeprecationWarning -m unittest tests.test_web_simulator_assets tests.test_web_simulator_client tests.test_local_simulator_api -v` → 13 tests passed.
