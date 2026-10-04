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
- Progress is completion-aware: a valid official finish (raw progress at
  least 0.95) scores 1.0; a DNF retains raw progress. Also report raw means.
  This definition is fixed before the first V5 binding.
- Bound retained implementation revisions to two and fresh studies to two.
  Aim for 60 minutes, allow at most 90 minutes for completed verification and
  packaging. Retain current V4 if no new candidate verifies; preserve useful
  source-bound findings rather than silently promoting failed experiments.

## Execution

- [x] Measure current policy and per-decision limitations on the three targets.
- [x] Run bounded ignored exploratory variants and select one generic design.
- [x] Add failing behavioral tests, implement the minimal speed controller,
  review and commit/push it while the live selector remains the V4 control.
- [x] Complete the consumed development grid and record failures/relaxation
  explicitly if applicable.
- [x] Commit and test the independent V5 profiles and source/runtime binding
  tools. Binding and fresh episodes are intentionally skipped: no candidate
  passed the fixed consumed-development completion/safety gates.
- [x] Independently recompute consumed development receipts and speed metrics;
  verify the preserved original V4 gates/seals and exact extracted replay.
- [x] Keep the verified V4 selector/package after rejecting the speed sources;
  run the compatible suite and independently verify preserved ZIP cold parity.
- [x] Update concise project state/results and commit/push coherent checkpoints.

## Review focus

HUD saturation at 80; acceleration out of gentle curves; early sharp-bend
braking; obstacles or remembered hazards blocking clear-road acceleration;
reset and invalid-camera state; preserving explicit neural export routes;
speed ratios on common finishes rather than unmatched lap averages; immutable
fresh thresholds and exact promoted/package source identity.

## Prospective bounded-time relaxation before V5 binding

Complete source-bound propulsion and target-76 probes missed the 13-second
target (three-map ratios 0.921758 and 0.871770); the two preview revisions
also missed a 20% aggregate gain (best ratio 0.864525). The unrestricted
propulsion probe additionally lost four of the control's 26 consumed-screen
finishes and increased contacts 15 to 24, so it is rejected for safety.
The localized production candidate therefore uses a further prospective
development pace floor of 3% aggregate improvement, with no individual target
slowdown. Preserve all existing development completion and aggregate safety
floors. Use the already-preregistered fallback fresh profile (combined time
ratio at most 1.00, all other stated floors) only with two committed exact
source/report failure proofs. No V5 geometry has been bound or opened yet.
Actual target laps must still be reported; this does not establish the
requested low-teens performance.

Before any fresh binding, two global study slots may each select the fixed
fallback profile. A slot is single-use across profiles; total bindings remain
at most two, with distinct candidate source hashes and disjoint geometry.
The second slot pins its predecessor's exact protocol/source. Neither slot
may rescore an earlier study or weaken the fallback safety/time floors.

The first localized production revision lost two prior track-2 finishes in
its consumed grid. An action-exact diagnostic on `(1,1274277667)` also showed
extra gas at HUD 47–52 immediately before the hazard detector armed, followed
by three contacts. The final revision therefore limits additional gas to
HUD at most 35 and requires actual steering to have caught up to the request.
Its source, three-map outcomes and complete grid must verify independently.
For this final revision, before observing its development outcomes, accept any
strictly positive three-map total-time improvement with no individual target
slowdown. The repeated rejected faster variants justify this bounded-time
pace relaxation; all completion, aggregate safety and fresh no-slowdown
floors stay fixed. A safety failure still keeps the verified V4 route live.

## Final disposition

The two production revisions both failed the fixed consumed-development
completion/safety gates. The final revision completed all 32 comparisons:
23 versus 26 finishes, three old finishes lost, contacts 27 versus 15,
damage 5.4 versus 3.0, and lower completion-aware progress. Its 23 common
finishes were 0.54% faster in total; that does not outweigh the gate failures.
The designated maps were 23.20/30.96/27.76 s versus V4's
23.98/31.00/27.84 s (1.09% total gain). The 10–13 s goal remains unmet.

No V5 study was bound or run. No failed source was promoted and no submission
was rebuilt. The default V4 route and original immutable d77 ZIP are retained;
the live module contains only one extra unused speed class. Independent static
AST comparison verified this distinction, and the preserved extracted ZIP
reproduced the original 25.76 s confirmation action/outcome exactly. The
compatible full suite passed 1,043 tests with 10 skips (630.22 s).

Three original operational child failures had no original stderr. Two later
initialization-limit errors are recorded explicitly; the final serial resume
then completed valid receipts without further errors. Failed attempts remain
part of the record. See `experiments/camera-speed-v5-development-result.json`
and `.haic-artifacts/camera-speed-v5/final-verification.json`.
