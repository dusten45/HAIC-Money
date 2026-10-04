# Camera Speed Upgrade Plan

The user explicitly authorizes autonomous implementation and validation. The
new requirement is faster ordinary laps without paying an aggregate time
penalty for stability on sharp curves combined with obstacles. Development
targets are `(1,516237)`, `(2,644062)` and `(3,1007)`, aiming at 10–13 seconds.

## Constraints and prospective decisions

- Control is the promoted V4 source at commit
  `1d45611bef833a52e3042d847c984b744b9513d9`, SHA256
  `d77b27e4489564f98df985b138d6c8ba2d74ff9ed275aeb62cd61477f2b46c18`.
- Keep official environment, model, existing evaluation tools and historical
  results unchanged. Never open earlier sealed evaluation partitions.
- Submission decisions must use camera observations and internal state only;
  true vehicle speed/geometry are permitted solely in diagnostic artifacts.
- Diagnose actual throttle, braking and preview limits before changing them.
  Compare a localized clear-road pace change with generic speed-envelope
  variants; do not encode the three requested map IDs or seeds in inference.
- Development: all three target maps must finish without added contacts,
  every current finish in the consumed 32-cell V4 screen must remain, aggregate
  crash/contact/damage/progress must be noninferior, common-finish total time
  must not increase. Initially require all three target laps at most 13 s.
- After two distinct documented failed development candidates, prospectively
  relax only the absolute target-lap acceptance to at least 20% aggregate
  improvement over control with no individual target slowdown. Continue to
  report actual lap times and any unmet 10–13 s goal. Safety and the no-overall-
  slowdown condition do not relax.
- A separate source-bound V5 study uses four fresh geometries per phase across
  all four tracks, two exact paired repeats per phase (72 cold episodes total).
  Select and commit its profile/source/seeds before opening fresh episodes.
- Strict fresh profile: preserve every control finish; phase and combined
  per-track completion floors; aggregate safety/progress noninferiority;
  phase common-finish total-time ratio at most 1.00; combined at most 0.90.
  Require at least one common finish per phase and complete valid receipts.
- If two candidate performance failures justify further relaxation, only a
  distinct study with unused geometries may permit one total lost finish,
  nonnegative net finishes and combined pace ratio at most 1.00. All other
  floors stay fixed. Never rescore already-opened data under weaker gates.
- Bound retained implementation revisions to two and fresh studies to two.
  Aim for 60 minutes, allow at most 90 minutes for completed verification and
  packaging. Retain current V4 if no new candidate verifies; preserve useful
  source-bound findings rather than silently promoting failed experiments.

## Execution

- [ ] Measure current policy and per-decision limitations on the three targets.
- [ ] Run bounded ignored exploratory variants and select one generic design.
- [ ] Add failing behavioral tests, implement the minimal speed controller,
  review and commit/push it while the live selector remains the V4 control.
- [ ] Complete the consumed development grid and record failures/relaxation
  explicitly if applicable.
- [ ] Commit and bind the independent V5 profile, source/runtime pins and fresh
  seeds before episodes. Run screen before unlocking confirmation.
- [ ] Independently recompute receipts, exact repeats, seals and speed metrics.
- [ ] Promote only verified exact candidate bytes; update route tests, run the
  compatible suite and independently test immutable ZIP cold-driving parity.
- [ ] Update concise project state/results and commit/push coherent checkpoints.

## Review focus

HUD saturation at 80; acceleration out of gentle curves; early sharp-bend
braking; obstacles or remembered hazards blocking clear-road acceleration;
reset and invalid-camera state; preserving explicit neural export routes;
speed ratios on common finishes rather than unmatched lap averages; immutable
fresh thresholds and exact promoted/package source identity.
