# Project Context

## Current State

The goal is a fast, reliable, rule-compliant 2026 HAIC agent. The live bare
`Agent` route remains `_CompoundClearingBrakeCarryController`; neural weights in
`model.pt` are bypassed after road detection. Diagnostic controllers in
`agent.py` are not active or submission-ready.

The source-bound `camera-policy-competition-v3` study tested the unchanged
`_BoundedSideHoldController` on new geometry under a competition-aligned gate.
Screen retained (29/32 finishes versus 18/32), but confirmation **REJECTED**:
46/64 versus 31/64 finishes and five lost control finishes against its fixed
limit of four. All 136 confirmation receipts and four exact repeats validated;
the saved summary and independent recomputation agree. Candidate
crashes/contacts/damage were 2/38/7.6 versus 8/102/20.4; shared finishes were
about 6% slower. The V3 blind phase is sealed. See
`experiments/camera-policy-competition-v3-result.json`. The 24 opened V3
geometries are development data.

Earlier V2 and V1 camera candidates were also rejected under their own fixed
screen gates despite aggregate finish gains. Their confirmation/blind phases
remain sealed. V2 lost-control mechanisms include on-road post-contact stalls,
side-switch vetoes and a crash without the proposed intervention; static
camera-pixel corridor widths did not certify physical clearance. The prior
results and consumed-cell diagnostics remain in `experiments/`.

V3 action-exact replays found two clear-road exits, two post-contact on-road
stalls, and one five-contact crash among its five confirmation losses. An
isolated clear-road row-42 dropout correction rescued the two road exits on
consumed data. Its full 96-cell audit improved finishes 75 to 82 without losing
a prior finish, but contacts rose 43 to 44, failing the development gate;
crashes stayed at two and shared-finish time ratio was 0.999857. A refined
recent-obstacle-or-HUD-speed gate was tested on nine changed-action cells: six finish gains,
no contact increase, and both known road-exit rescues retained. Its full
96-cell source-bound replay was interrupted at 57/96 receipts; it has not
passed a gate.
Independent review found that remembered obstacle state may survive lost-road
frames, so that predicate needs correction and a complete audit before any
fresh V4 seed bind. The inactive `_ClearRoadRow42DropoutController` in `agent.py`
contains the refined predicate but has not passed the full development gate;
the stale-memory edge remains. V4 runner/gate/template are committed but
unbound with pending source pins. No V4 fresh seeds or episodes exist; V3 blind
must stay sealed. See `experiments/camera-policy-competition-v4-progress.md`.

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
