# Fixed-speed missing-road and post-impact recovery

User requests diagnosis and implementation that keeps progressing after road loss/collision, retaining high target speed. User accepts the previously measured brief road excursions; prior 1%/8-tick limits are diagnostic fields only, not acceptance gates.

## Evidence and causal hypotheses

Original damping TRAIN2:38200 at decisions112–113 has near-road centers left of42 but missing row42. Default far=42 makes steering positive, opposite to the previously visible bend. At114 no centers remain and steering becomes0. TRAIN3:38201 has analogous all-missing fallback at100. In TRAIN1:38201, collision at100 drops actual speed58.7→29; final gas already rises to0.6 at101. Steering swings to+0.7 then−0.7 by107 while progress stalls near45.6%. Lack of throttle is not the observed first bottleneck. Frozen source and traces establish code behavior; each proposed recovery still needs intervention evidence.

## Batch: four independent directions, one each

All candidates wrap original damping, preserve final target60 governor and first10 actions. Pixel-only inputs; evaluator geometry never enters policy.

1. Partial-row geometry: when row42/54 is missing but at least2 rows remain, linearly extrapolate from nearest visible rows instead of substituting center42. Recompose road+existing obstacle steering, clip once.
2. Last-visible road memory: on loss of row42, use last trustworthy road direction for at most30 decisions, bound magnitude0.18–0.55. Resume original on reacquisition. This specifically tests keeping turn direction rather than straight-zero fallback.
3. Wide road reacquisition: on missing key rows, search full-width asphalt runs at42/54, selecting the run nearest last visible center and requiring4-45pixels. Use reconstructed geometry only when both rows found.
4. Post-impact slew: pixel speed below50 and drop>10 from last4 readings activates12 decisions of steering-change bound0.18 per decision. Final pedals unchanged. This tests whether discontinuous steering after speed loss prevents recovery; speed-drop proxy may include non-collision events.

TRAIN consumed tracks1–3 × seeds38200–38201, four candidates ×6=24 full episodes, max1200 decisions each, 2CPU/2GiB/1800s. Immutable original damping is control. Verify first10 action/state hashes against it. Record effective modifications and first changed decision; this proves shared history before intervention, not all subsequent driving states. No threshold sweep.

Success for advancing to fresh comparison: completion above3/6 while preserving all three original finishes, invalid0, no explicit target reduction. Compare time only on tied completion. Report collisions/damage and road departures rather than zero-collision or strict-road hard rejection. A recovery result must show progress after the intervention/collision, not just event detection. Any infrastructure failure invalidates scoring.

If a candidate meets the TRAIN gate, freeze it and run a separately registered fresh matched comparison against exact original damping before offering a replacement ZIP. No site action or automatic SOTA promotion. A non-improving branch remains available for diagnosis; do not lower target to pass. Outcome always records three gates.

Implementation: add recovery runtime; focused synthetic checks of missing-row sign, timeout, unchanged pedals and post-impact slew; register local diagnostic runner; exact-hash design/execution events;24 episodes; integration and evidence summary. Standing local authorization2026-09-29 and current user instruction cover this work.

Additional environment finding before execution: off_track retirement is the wrapper counter for101 consecutive negative-reward decisions (about8.08s at4/50 cadence), not direct center-road containment. One collision adds0.2damage and reduces grip/engine/steering, but does not itself terminate. Recovery must resume new forward progress before this unchanged environment deadline. No environment modifications.
