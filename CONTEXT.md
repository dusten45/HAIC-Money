# Project Context

## Current State

The goal is a fast, reliable, rule-compliant 2026 HAIC agent. The live bare
`Agent` route remains `_CompoundClearingBrakeCarryController`; neural weights in
`model.pt` are bypassed after road detection. Diagnostic controllers in
`agent.py` are not active or submission-ready.

The `_ClearRoadRow42DropoutController` candidate now invalidates its own
obstacle confidence after lost-road or invalid current-frame decisions while
preserving inherited obstacle latches. Commit `7df85ae` is pinned as its base;
the actual selected source SHA256 is
`d77b27e4489564f98df985b138d6c8ba2d74ff9ed275aeb62cd61477f2b46c18`.
Its complete 96-cell consumed V3 development audit independently passed:
75 to 81 finishes, no previous finish lost, contacts 43 to 43, crashes 2 to 2,
damage 8.6 to 8.6, common-finish time ratio 0.999857. Both known no-contact
track-3 road-exit rescues remain. See
`experiments/camera-upgrade-development-20261004-result.json`; full receipts
and cold ZIP action-parity evidence are under
`.haic-artifacts/clear-road-row42-v4/freshness-audit/`. This is development
evidence, not fresh generalization or activation.

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
Screen independently retained: 26/32 versus 16/32 finishes, one lost control
finish, contacts 15 versus 57, crashes zero versus seven, common-finish time
ratio 1.055276; both exact repeat pairs passed. Confirmation is running on
16 separate geometries. See `experiments/camera-policy-competition-v4-result.json`.
Only promote after retained phases, valid seals and combined decision.
The old V4 progress checkpoint is superseded
by `docs/superpowers/plans/2026-10-04-autonomous-camera-upgrade.md`.

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
  lacks `fcntl` for `tests/test_drqv2_matched.py`; the latest full suite
  excluding that file passed 907 tests and skipped 10. Official-like Linux
  submission-container certification is unavailable here: Docker is stopped
  and WSL lacks the required runtime/dependencies.
- Preserve untracked user videos, prior submission directories and
  `submission.zip`; none is the current candidate package. Keep large traces,
  videos, models and run artifacts out of Git. Commit and push coherent,
  validated progress to the current upstream branch.
