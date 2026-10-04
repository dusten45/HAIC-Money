# Normal V1 T2 rejection attribution

One exact173-action diagnostic replay of frozen Normal V1 saved cameras at
`.haic-artifacts/apex-speed-20261005/normal-line-v1-probe-track2/`.
The replay and offline Inspector both assert every emitted float32 action
against the source-bound benchmark trace. No second candidate/benchmark,
frozen source edit, parameter grid or holdout occurred.

First trajectory divergence17 initially helps: Normal leads0.16s at progress0.6.
The actual regression occurs across progress0.6–0.7, where that lead becomes a
1.12s deficit. New optimized references select148/149 near the next hairpin.
Even though selected path corners have supported camera clearance, the final
long-preview pursuit action is not the same trajectory. The inherited ridge
branch replaces the already checked confidence action and bypasses ArcGuard's
near emitted-action body check.

At149 speed82.8m/s, actual emitted steer−0.147 predicts only1.38m corner depth
over8m, below1.9, and kinematic lateral demand314m/s². The same exact prestate
confidence branch gives steer−0.103/brake0.533 versus ridge brake0.324,
target44.8 versus59.7m/s, and body depth1.92m. That is a supported fallback at
the earliest locally unsupported emitted action in this onset sequence.

At150 actual ridge depth0.43m/demand416; at151 depth1.69/demand476. At154 the
discarded confidence ArcGuard selects steer−0.4 with body depth2.12m, whereas
ridge emits−0.326/brake0.04 with depth1.01. At155–157 ridge depth0 despite road
available; actual track-center gap grows1.7→4.7m. At158 road becomes unavailable
at6.1m gap and inherited recovery brakes to near zero. Multiple slow recovery
actions explain the time loss. Zero contacts/full-offtrack counts conceal13
partial-offtrack samples; this is not clean road tracking.

The concrete next intervention is to validate the ACTUAL slew-limited ridge
action's near body arc before replacing the confidence action, retaining
confidence when it passes and ridge does not. This targets action selection,
not tail trimming. Existing force/road/hazard limits should remain. An emitted
force speed cap alone is insufficient at149: its final target59.7 is already
below the approximately64m/s emitted-force cap. Delayed braking and geometric
tracking matter, and later frames have no supported fallback.

All arc checks assume constant emitted steer over8m. This is a camera heuristic,
not a dynamic safety certificate. Physical track/pose/velocity are offline labels
only. Compact `normal-line-recovery-probe.json` binds source, helper, frames and
trace hashes, lists selected references and all140–172 onset diagnostics,
including same-prestate confidence controls. No new candidate is implemented
by this report.
