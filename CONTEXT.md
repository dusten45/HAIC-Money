# Project Context

## Current State

The goal is a fast, reliable, rule-compliant 2026 HAIC agent. The live bare
`Agent` still selects `_ClearRoadRow42DropoutController`; neural weights in
`model.pt` are bypassed after road detection. Other diagnostic controllers in
`agent.py`, including the rejected speed candidate, are not the submission
route. Use the immutable V4 ZIP below for the exact verified source; the live
file also contains an unselected experimental class and is not byte-identical.

The `_ClearRoadRow42DropoutController` candidate now invalidates its own
obstacle confidence after lost-road or invalid current-frame decisions while
preserving inherited obstacle latches. Commit `7df85ae` is pinned as its base;
the evaluated and packaged V4 source SHA256 is
`d77b27e4489564f98df985b138d6c8ba2d74ff9ed275aeb62cd61477f2b46c18`.
Its complete consumed V3 development audit passed; both known no-contact
track-3 road-exit rescues remain. See
`experiments/camera-upgrade-development-20261004-result.json`.

V1, V2 and V3 camera studies remain rejected under their original fixed gates.
V3 confirmation exceeded its fixed lost-finish allowance. Its blind phase and
earlier unopened phases remain protected; its 24 opened geometries are
development data. Remaining failures include post-contact stalls and crashes.

At the user's explicit request to relax criteria after repeated failures, a
new practical V4 profile was committed before binding any new geometry, with
bounded lost-finish allowances and a 1.10 combined pace floor. Runtime,
source/repeat integrity, safety, progress and per-track floors stayed fixed.
See the immutable protocol for its complete prospective criteria.
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
The preserved V4 package source exactly matches the evaluated source above.
See `experiments/camera-policy-competition-v4-result.json`. Final ZIP:
`.haic-artifacts/submissions/20261004T104340018489Z_clear-road-camera-practical-v4-final/submission.zip`
(6,298,260 bytes; SHA256
`3508100c4f700de4fdeb1d7f9e66a5e0eeb79c697b9762a57d6987d2bce05fe6`).
Its four member hashes match V4; CPU loading/reset/action checks passed, and
extracted cold driving exactly reproduced a fresh V4 action trace/outcome.
Post-promotion Windows-compatible full suite passed 972 tests, skipped 10;
the submission ZIP also passed independent extracted-file cold driving.

## Bounded Camera Speed Study

The latest request adds a no-aggregate-slowdown constraint and 10–13 s targets
for `(1,516237)`, `(2,644062)` and `(3,1007)`. Exact V4 cold laps are
23.98/31.00/27.84 s, all without contacts or sampled off-road events. The main
limits are low gas on clear curves and conservative obstacle/corridor speed;
more gas changes the trajectory before hazards become visible.

The unrestricted acceleration probe lost four old finishes and raised contacts
15 to 24 on the consumed 32-cell V4 screen. Localized production revision 1
lost four and raised contacts to 30. Revision 2 restricts extra gas to HUD at
most 35 with aligned actual steering; its target laps are 23.20/30.96/27.76 s,
only 1.09% faster in total. Its complete consumed-grid audit had 23/32 versus
26/32 finishes, lost three old finishes, and raised contacts 15 to 27 and
damage 3.0 to 5.4 (common-finish time ratio 0.994633). It is rejected. Faster
preview probes also failed the fixed
completion floor; their best target laps were 20.88/26.64/24.08 s. The requested
10–13 s performance is not achieved.

After recorded failures, the target-speed floor was prospectively relaxed
from 13 s to 20% improvement, then 3%, then any positive gain for the final
revision. Development completion/safety and no-aggregate-slowdown floors stay
fixed. The two-revision budget is exhausted: keep V4. New V5 strict/fallback
protocol tools are tested, but no V5 geometry has been bound or opened. Older
unopened partitions remain protected. See the plan and
`experiments/camera-speed-v5-development-result.json`; consumed data must never
be described as fresh validation.

Exact V4 replay of `(1,1493128875)` confirms a post-contact stall despite
continuously visible road and obstacle; this is not a missing-row42 failure.
Six-step gas pulses of 0.14 and 0.35 each failed to recover it. Next controller
priority is preventing contact with better camera path/corridor prediction,
rather than adding global acceleration or weakening safety gates. Large
source-bound telemetry/receipts remain under `.haic-artifacts/camera-speed-v5/`.
The preserved V4 ZIP passed a second independent extracted cold replay on
confirmation `(1,2841112300)`: identical action/outcome, 25.76 s, no contacts,
damage or sampled off-road events; init 7.06 s and other runtime limits pass.
The final Windows-compatible full suite passed 1,043 tests, skipped 10
(630.22 s); it excludes only the Linux `fcntl` matched-study test file.
Three original cold child failures lack their original stderr; two later
initialization-limit failures (19.54/11.55 s) are preserved explicitly. The
last serial resume completed all 32 valid receipts without further failures;
this does not erase earlier invalid attempts or qualify the rejected source.

## Independent Agent Checkpoint

The separate new-agent study is preserved in `agents/apex_2026/` on branch
`codex/apex-2026-independent-agent`; the active root agent and official physics
were preserved. Hybrid mandatory laps are 19.62/26.68/22.74/21.78 s with zero
contacts, but only 10/16 additional development cells finish. The 10–13 s target
and relaxed 18 s profile remain unmet; no independent candidate is promoted.
Holdout geometries remain unopened. See its README and source-bound benchmark
summary. Development was paused at the user's request for a GitHub checkpoint.

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
  full suite excluding that file passed 1,043 tests and skipped 10 (630.22 s).
  The unchanged local web API has an intermittent Windows TCP abort on rejected
  POST requests (two of 360 diagnostic requests; final full suite passed).
  This does not establish an inference defect. Official-like Linux
  submission-container certification is unavailable here: Docker is stopped
  and WSL lacks the required runtime/dependencies.
- Preserve untracked user videos, prior submission directories and
  `submission.zip`; none is the current candidate package. Keep large traces,
  videos, models and run artifacts out of Git. Commit and push coherent,
  validated progress to the current upstream branch.
