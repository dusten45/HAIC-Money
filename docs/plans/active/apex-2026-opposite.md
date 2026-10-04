# Apex independent opposite-strategy plan

Status: completed research cycle; original timing goal and holdout gate failed.
See [final report](../../../agents/apex_2026/REPORT.md).

User-directed research, 2026-10-05 KST. Original main root agent and simulator
are protected. The user removed the clock deadline; no time-triggered fallback.
The prospective source of truth is [DESIGN.md](../../../agents/apex_2026/DESIGN.md)
and all required/development cells are in [benchmark.json](../../../agents/apex_2026/benchmark.json).

1. Completed: restore Python 3.11, reproduce current root/release baselines and
   existing tests; preserve pre-existing missing-artifact and memory-test errors.
2. Completed: independently improve continuous-path planning and joint-action
   trajectory rollout from pixel observations. Keep negative results and exact
   source snapshots. No imported old policy, per-track dispatch, or telemetry input.
3. Tested hypothesis: optical-flow dropouts incorrectly erase instantaneous yaw
   at critical turns. Calibrated HUD yaw should improve tracking while retaining
   original speed/path settings. Matched required four and all twelve declared
   development cells determine benefit; synthetic decoding accuracy alone does not.
4. Diagnose remaining speed limits using completed traces and isolated physical
   calibration before further changes. Change one causal component at a time,
   test runtime behavior, preserve failures, commit and push meaningful work.
5. Select by the prospective completion/speed gates. Freeze a single source and
   configuration, commit and push, rerun required/development and repeat required.
   Only then allocate the twelve unseen holdout cells and evaluate once without
   subsequent tuning. Report original 10–13-second achievement separately from
   weaker completion-first qualification. No official submission/confirmation.

Results and reproducible commands live in the lane README and result ledgers.
Consumed development performance is not an independent generalization estimate.

Final outcome: required4/4, development12/12, repeat4/4; then unseen holdout10/12.
No original-window laps. Candidate frozen in cdf675e and left experimental,
not adopted; no post-holdout code change. All five work stages completed,
including negative result reporting. This file records closure, not a live queue.
