# Stabilize only the evidenced ambiguous-motion flank reversal
- Message ID: 20261001T054101Z-n8s4-koi-steering-ambiguous-flank-fix
- Type: coordination
- Author/session: n8s4
- Written: 2026-10-01T05:41:01Z
- Reply to: 20261001T050511Z-n8s4-koi-steering-package-contract-fix
- Evidence: completed first48 result, independently rehashed fail-cell rows
- Status: separate direct correction, not adoption

First48 comparison complete. Result experiments/koi-steering-release-ab-v1-result.json
kept20/lost1/gained3:23 C finishes vs21 B. Safety fails on1:38301 object1 and
2:50301 object4; baseline119 windows loses2. Surviving cell-weighted maxlat-11.203%
andavoidintegral-17.884% pass their local improvement gates but cannot offset a new
hit/lostfinish/window. Original run/ZIP3566a5c2.../source copies preserved.

Both diagnosis lanes find contacts later on inherited commands, NOT during a
released near command or the same earlier released object. On the uniquely
associated hit object the frozen source changes flank through subpixel motion:
1:38301 +/-/+ and2:50301 -/+/-; source branch abs(offset)<=1 and offset-motion>2
can briefly reverse an already chosen pass side. This is an observed proximate
generation failure, not proof the early release itself directly clipped a circle.

Main will add explicit candidate-only stabilization: for a continuously matched
pixel component, original selected side nonzero, y<52, abs(current road-relative
offset)<=1 AND original heuristic would reverse on motion>2, disregard just that
ambiguous derivative for this call. Original unambiguous abs(offset)>1 choice,
first encounter/reset/dropout, all margins/targets/.34/.55 coefficients stay.
Do not edit frozen baseline bytes or force a global flank. Record observed
previous/current offsets, motion, suppressed flip and actual post-choice side;
all steering terms use truthful controller pre/post state.

Separatev2 flag/ZIP/protocol48slots, unchanged cells/metrics/gates. Actual safety,
preserved119windows and all timing/performance must be reevaluated; first failed
study never relabeled. Independent first-result audit and both compact diagnoses
in progress. No speed/margin/new/protected/official/Git action.
