# Fixed60 completion causal repair probe

The initial four-direction batch is preserved. Review found a saturation residual in continuous_road and an off-by-one impact-clear window in continuous_road/offset_path. Near_servo1:38211 overrides the parent's inward steering at233 while the near-road error is already shrinking. Control1:38210 and offset_path1:38210 turn the wrong way when row42 is missing, even though rows54/50/46 still describe the bend. These are concrete source/trace observations, not a new coefficient sweep.

Standing authorization2026-09-29 and current all-completion request cover this repair. Four second candidates (eight total in batch), no pedal/target change:
- offset_repair: fix impact window; continue offset-path geometry through a missing middle row by linear extrapolation from the nearest two visible rows. Temporal correction uses the same reconstructed row in the previous observation. This addresses its explicit wrong-sign command at147.
- servo_unwind: preserve near-servo coefficients but suppress its extra correction when near-road error shrinks and the correction opposes that inward motion.
- continuous_repair: exact unclipped obstacle term, correct current-action impact window. No other perception/feedback changes.
- boundary_repair: maintain boundary projection but replace false-center road steering on missing row42 with observed-row extrapolation and temporal feedback; far-row30 visibility is not needed for this correction.

Probe all four plus rerun control on tracks1 × seeds38200,38210,38211, three consumed TRAIN cells,15 episodes. These test a valid control finish, missing geometry, and delayed unwinding. Target is3/3 with all control finishes preserved, or clear causal repair that merits the remaining full12-cell check. Never label this selected subset a full completion rate or generalization. Record intervention/trace evidence, same first10prefix, invalid0. 2CPU2GiB900s,1200decisions/episode. Report all three gates, no release/site action. If a full candidate remains weaker, preserve baseline. After three comparable non-improving cycles, pivot architecture rather than more coefficient changes.
