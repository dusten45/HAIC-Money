# Project Context

## Current State

The goal is a fast, reliable, rule-compliant 2026 HAIC agent. The live bare
`Agent` now selects `_ClearRoadRow42DropoutController`; neural weights in
`model.pt` are bypassed after road detection. Other diagnostic controllers in
`agent.py` are not the active submission route.

The `_ClearRoadRow42DropoutController` candidate now invalidates its own
obstacle confidence after lost-road or invalid current-frame decisions while
preserving inherited obstacle latches. Commit `7df85ae` is pinned as its base;
the actual selected source SHA256 is
`d77b27e4489564f98df985b138d6c8ba2d74ff9ed275aeb62cd61477f2b46c18`.
Its complete 96-cell consumed V3 development audit independently passed:
75 to 81 finishes, no previous finish lost, contacts 43 to 43, crashes 2 to 2,
damage 8.6 to 8.6, common-finish time ratio 0.999857. Both known no-contact
track-3 road-exit rescues remain. See
`experiments/camera-upgrade-development-20261004-result.json`; full development
receipts are under `.haic-artifacts/clear-road-row42-v4/freshness-audit/`.

V1, V2 and V3 camera studies remain rejected under their original fixed gates.
V3 screen had 29/32 versus 18/32 finishes; confirmation had 46/64 versus 31/64
but lost five control finishes against its fixed allowance of four. V3 blind
remains sealed; earlier unopened phases also remain protected. Its 24 opened
geometries are development data. Known remaining failures include post-contact
on-road stalls and multi-contact crashes.

At the user's explicit request to relax criteria after repeated failures, a
new practical V4 profile was committed before binding any new geometry:
phase finish gains at least 2/4/2, lost control finish budgets 3/6/3, combined
gain at least 12. Runtime, source/repeat integrity, aggregate safety, progress,
pace ratio at most 1.10, seed-cluster and combined per-track floors are unchanged.
The search is capped at two candidate implementations and two fresh studies;
any further relaxation requires a distinct protocol and unused geometry.
V4 is now source-bound in `experiments/camera-policy-competition-v4.json`,
with 8/16/8 new geometries across four tracks and 2/4/2 exact paired repeats.
All three phases and the combined gate independently retained: screen 26/32
versus 16/32 finishes, confirmation 51/64 versus 24/64, blind 23/32 versus 17/32.
Across 128 paired cells, finishes increased from 57 to 100, contacts fell from
226 to 89 and crashes from 21 to 1. Per-track candidate/control finishes are
26/17, 22/11, 29/17 and 23/12 (32 cells each). Six control finishes were lost;
28 candidate cells remain DNF. The 51 common finishes took 4.79% longer in
aggregate, within the fixed 10% pace allowance. All 272 cold receipts, eight
exact paired repeats, source pins and predecessor seals passed independent
recomputation. Runtime maxima: init 4.67 s, reset 0.20 ms, action 139.26 ms,
RSS 231.55 MiB. Finite testing does not guarantee completion on every geometry.
Next controller priority: 25 of the 28 remaining DNF cells retired after
contacts without a sampled full off-road event. Investigate post-contact loss
of motion and partial road contact before choosing a further recovery change.
The promoted source exactly matches the evaluated selected source above.
See `experiments/camera-policy-competition-v4-result.json`. Final ZIP:
`.haic-artifacts/submissions/20261004T104340018489Z_clear-road-camera-practical-v4-final/submission.zip`
(6,298,260 bytes; SHA256
`3508100c4f700de4fdeb1d7f9e66a5e0eeb79c697b9762a57d6987d2bce05fe6`).
Its four member hashes match V4; CPU loading/reset/action checks passed, and
extracted cold driving exactly reproduced a fresh V4 action trace/outcome.
Post-promotion Windows-compatible full suite passed 972 tests, skipped 10;
the submission ZIP also passed independent extracted-file cold driving.

## Frozen Competition Contract

- Do not modify `core/`, `env_wrapper.py`, or `damage.py` for experiments.
- Input: four 84x84 grayscale frames, CHW float32 in `[0,1]`. Output:
  continuous `[steer, gas, brake]` with steer in `[-1,1]` and pedals in `[0,1]`.
  There is one decision per step; frame skip 4 is inside the environment.
- Use raw official reward. Pure time limits bootstrap; finish, crash and
  off-track endpoints do not. Finish needs at least 95% progress and a valid
  forward finish crossing. Geometry is seed-driven; track ID changes obstacles.
  Reserve a geometry seed across every track ID.
- Official ranking prioritizes completion, then DNF progress or completed lap
  time. Require source-bound cold CPU receipts and exact reload checks before
  promotion. Never select on confirmation/blind or alter gates after data.
- Inference gate: Python 3.11, Torch 2.1 CPU, NumPy 1.26; init 10 s,
  reset/action 5 s, worker 1,024 MiB, ZIP 500 MiB. No trainer, SB3 or prohibited
  imports in submission inference. The official participant source checked on
  2026-10-04 KST was commit `dfb7a2de2178825ca5c5ce20bab01ba67052ba31`.

## Other Evidence and Infrastructure

- The completed matched DrQ-v2 steering-logit L2 study rejected coefficient
  0.001 in both training seeds: paired confirmation finish deltas were -3 and
  -7. The separately planned augmentation-padding comparison has no verified
  completion in this Windows checkout. Demonstration replay is implemented but
  not trained or performance-evaluated. Read the corresponding
  `experiments/drqv2-*-result.json` and execution records before resuming ML.
- Earlier PPO variants did not establish competitive completion. Video-recovered
  maps `(1,516237)`, `(2,644062)`, `(3,1007)` and untracked `evaluation_videos/`
  are diagnostic, not fresh evaluation. Track Lab can replay camera, action,
  trajectory and collision telemetry.
- Local `.venv`: Windows Python 3.11.15, Torch 2.1 CPU, NumPy 1.26. Windows
  lacks `fcntl` for `tests/test_drqv2_matched.py`; the latest post-promotion
  full suite excluding that file passed 972 tests and skipped 10 (430.49 s).
  The unchanged local web API has an intermittent Windows TCP abort on rejected
  POST requests (two of 360 diagnostic requests; final full suite passed).
  This does not establish an inference defect. Official-like Linux
  submission-container certification is unavailable here: Docker is stopped
  and WSL lacks the required runtime/dependencies.
- Preserve untracked user videos, prior submission directories and
  `submission.zip`; none is the current candidate package. Keep large traces,
  videos, models and run artifacts out of Git. Commit and push coherent,
  validated progress to the current upstream branch.
