# Project Context

## Current State

The goal is a fast, reliable, rule-compliant 2026 HAIC agent. The live bare
`Agent` route remains `_CompoundClearingBrakeCarryController`; neural weights in
`model.pt` are bypassed after road detection. Diagnostic controllers in
`agent.py` are not active or submission-ready.

The latest source-bound fresh study, `camera-policy-generalization-v2`, is
**REJECTED on its screen**. Its committed `_BoundedSideHoldController` candidate
finished 27/32 previously unseen cells versus 16/32 for the live route, with
31 versus 65 contacts. It lost one control finish, introduced two new crashes,
had one severe both-DNF progress loss, and exceeded per-cell contact/damage
limits on three cells. These failures trigger fixed gates despite the aggregate
finish gain. All 68 cold receipts and two spot-check pairs validated;
confirmation and blind remain sealed. Source commit: `f12c5b6`. Protocol and
outcome: `experiments/camera-policy-generalization-v2.json` and
`experiments/camera-policy-generalization-v2-result.json`. The eight screen
geometries are now consumed development data.

The prior fresh `camera-policy-generalization-v1` screen was also **REJECTED**:
its `_ImpactAwareSparseRoadController` finished 25/32 versus the live route's
21/32 but lost five control finishes and breached contact/damage and pace
gates. Its confirmation and blind remain sealed. Exact loss traces showed four
post-contact on-road stalls and one high-speed road exit. See
`experiments/camera-policy-generalization-v1-result.json` and
`experiments/camera-policy-screen-loss-mechanisms-v1-result.json`.

On the consumed v1 screen, a high-speed bend priority and three-decision
obstacle-side hold produced 28/32 finishes and rescued three lost finishes,
without losing any previous candidate finish. The old metric gates still fail
on two lost control finishes and per-cell contact/damage/pace. This is
development evidence only (`experiments/bounded-side-hold-camera-dev-v1-result.json`).
Raising the global no-corridor speed cap 18 to 24 lost three previously finished
cells on that reused screen and was rejected. Camera-triggered post-contact gas
pulses and a wider-side recovery attempt did not rescue the remaining stalls.
The three-decision camera-motion predictor has wide and sparse residuals; it
does not certify physical passing reachability. Related `experiments/*dev-v1-result.json`
files retain the evidence.

Current priority: diagnose v2's severe loss at track 2 / geometry 1918128940
and crash/contact increases at tracks 3/4 on geometry 416046940 and track 4
on geometry 3602985103. Initial exact replays show the short side hold made no
applied steering change in these failures. Inherited no-path obstacle steering,
side switching and braking require a source-bound causal analysis. Test
distinct repairs only on consumed data, then preregister new geometry; never
open v2 confirmation/blind or activate the rejected source.

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
  excluding that file passed 863 tests and skipped 10. Official-like Linux
  submission-container certification is unavailable here: Docker is stopped
  and WSL lacks the required runtime/dependencies.
- Preserve untracked user videos, prior submission directories and
  `submission.zip`; none is the current candidate package. Keep large traces,
  videos, models and run artifacts out of Git. Commit and push coherent,
  validated progress to the current upstream branch.
