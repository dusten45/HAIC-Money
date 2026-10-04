# Retain current-motion footprint protection on minimum targets
- Message ID: 20260930T155336Z-m7c3-koi-clearance-projection-correction
- Type: coordination
- Author/session: m7c3
- Written: 2026-09-30T15:53:36Z
- Reply to: 20260930T152943Z-m7c3-koi-clearance-r2-start
- Evidence: completed r2 primary result and completed3/38301 decision trace
- Status: separate correction; no adoption

All48 r2 episodes completed; experiments/koi-minimum-clearance-ab-r2-result.json
reports kept20/lost1/gained0 and failed damage/collision/new-hit/window gates.
Keep v2 ZIP64cecc0b..., r2 source copies and result unchanged.

Completed3/38301 candidate contains exactly one actual override. Atstep31 the
projected obstacle center38.6338028 is only3.3661972px from42, crossing_active true,
but the feasible geometric target displaces+4.25px and replaces+.257611 baseline
with+.118087. A full desired lane is not the present motion projection. The later
collision is not proof this early obstacle itself was clipped; detailed contact
diagnosis and independent full-result audit are in progress.

Direct correction adds a separate current-motion projection footprint gate:
minimum-target overrides require a finite available projection on the opposite
side of the selected pass flank, separated by physical wheel/hull half-width plus
detected box extent and the existing raster allowance. Do not use a new steer
scale or speed target. Both full target-lane geometry and.08s applied-command
checks stay. Newv3 package/explicit flag, tests, and separate same48-slot r3 run;
v1 default and v2 configuration semantics remain preserved. Rejecting the observed
unsafe-projection intervention is a hypothesis for safety restoration, not prior
proof of shorter/safe trajectories. No new/protected/official action or Git write.
