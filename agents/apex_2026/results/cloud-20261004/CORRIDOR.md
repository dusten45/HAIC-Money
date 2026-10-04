# Observed curved corridor experiment

`corridor_agent.py` is a self-contained camera-only hybrid derivative. It
rejects elongated components using symmetric world-scaled aspect, separates
road turning from lane-offset change, and follows measured visible asphalt
midpoints through a finite obstacle approach/pass/exit. Every required row
must be observed. An oriented hull polygon is checked at the measured road
breakpoints, together with circle clearance and full-hull viewport bounds.
The five-row near midpoint extension supplies a steering anchor, not road
clearance. Longitudinal portions beyond the finite observed horizon remain
uncertified. The ineffective stationary recovery extension is omitted.

V3 replaced the artificial center ±18-pixel edge search with contiguous runs
over the full camera width. A span touching x0/x83 bounds only visible asphalt;
it invents no offscreen road. Separate runs join only for measured obstacle
occlusion, and missing rows are never interpolated. V4 requires a compact
component peak compatible with official orange RGB(255,165,0), which the
official grayscale transform renders as 171/255. Only an actual .52–.54 pixel
within one pixel of the measured bright component can extend its occlusion.
This fixes saved-frame curb competition and antialias topology without
padding road edges or relaxing hull margins.

## Fresh benchmark screens

All cells below were newly simulated against evaluator SHA256 `39a28061…`.
They are benchmark screens, **not complete development rejections**. Required
cells are `(1,516237)`, `(2,644062)`, `(3,1007)`, `(4,18800)` in that order.

| Version | Required lap seconds | Required finishes | Target `(1,1764402399)` |
| --- | --- | --- | --- |
| V1 | 19.70 / worker error / 23.36 / 21.88 | 3, one operational failure | DNF, progress .33916, one contact |
| V2 | 19.70 / 25.92 / 23.36 / 21.88 | 4/4, zero contacts | DNF, progress .33916, one contact |
| V3 | 19.44 / 25.84 / 23.68 / DNF | 3/4; track 4 stalled at .52330 with one contact | DNF, progress .33916, one contact |
| V4 | 19.98 / 24.78 / 22.68 / 21.76 | 4/4, zero contacts | DNF, progress .33916, one contact |

V1's track-2 exception came from an unobserved partial-path anchor. V2 adds
a clean rejection plus hull-side intrusion checks between corners; its full
fresh screen is preserved separately. V3's track-4 failure is a performance
regression, not an operational error. V4 restores those four completions but
still misses the original 10–13-second goal and the 15/18-second ceilings.
Target retirement has zero offroad samples: the official reward watchdog
ends a stationary obstacle contact, rather than a loss of road-wheel contact.

## Complete development result

`development/r6-corridor.json` binds frozen V4, parameters `{}`, evaluator
`39a28061…`, and the five preceding complete development rejections. This
fresh 20-cell trial used the prospective **15-second** profile; the original
13-second verdict remains separate and false.

- Required laps: **19.98 / 24.78 / 22.68 / 21.76 seconds**, zero contacts.
- Extra finishes: **10/16**, per track **2/4, 4/4, 1/4, 3/4**. This matches
  the fresh hybrid's total extra coverage while redistributing its failures.
- No lap passes either 13 or 15 seconds. The complete profile fails; V4 is
  not promoted. It is the sixth consecutive complete rejection, so DESIGN's
  next prospective profile is 18 seconds without rewriting earlier verdicts.
- Maximum measured initialization/reset/action: **3.292 / .098 / 124.055ms**;
  peak worker RSS **95.781MiB**. These measured runs meet the runtime limits.

V4 source is frozen at SHA256
`43e163dfb4de301c85403d43bbaf7fa6a4de3267727badfb36273be5f1c2dd0e`.
Twenty corridor tests pass, including failing-first cases for the curved
pass, missing near rows, hull intrusion, measured full spans, clipped visible
envelopes, viewport escape, curb competition and bounded antialias occlusion.
The current Apex test directory passes 88 tests. These establish the scoped
behavior, not a successful pass policy or full-repository test result.

## Limits and provenance

Saved-frame V4 replay keeps obstacle identity through steps 80–82, where V3
changed identity twice after selecting a small white curb fragment. At step
80, row 44's span changes from x15–47 to its measured visible x0–47; step 84,
row 45 changes from x25–37 to x0–37. Neither replay produces a verified pass.
The inherited partial planner still charges an absolute straight-image shift
and centroid-side heuristic; the measured midpoint of a clipped span also
does not identify the true road center. These remain separate hypotheses,
requiring a reachable trajectory anchored at the current camera pose and
checked against observed swept-body clearance before any further correction.

A subsequent read-only replay found no certified correction to that gate.
Full passes at steps 80–83 lack required exit rows; at steps 84–85 both
curve-relative lane shifts exceed the existing budget. At step 86 the right
shift fits, but the oriented hull fails near row 41. The partial left shift
is 1.88–2.39 pixels per row. A current-pose/heading-anchored 15m circular left
arc at step 80 clears the observed footprint by only .124 pixels and folds
forward image y after 90 degrees; a single-valued x(y) path cannot describe
that arc. It also has 31 initial poses with unobserved near-body road and a
46.64m/s lateral ceiling below the 52.42m/s camera speed. It is **not** a safe
or reachable-pass counterexample. No gate was relaxed on this evidence.

Independent source review also found that converting the quantized bright
bbox directly to a circle radius can underestimate the official 1.2m radius;
for example, a 2×3-pixel core yields .882m. The analytic oriented-hull check
therefore needs a conservative physical radius before it can independently
certify circle clearance. The existing center margins do not establish that
this estimate is correct for every rotated pose. Frozen V4 retains this
known limitation; the proposed correction belongs to a separate candidate.

Compact exact receipts are `probes/corridor-v{1,2,3,4}-{target,mandatory}.json`.
`probes/corridor-provenance.json` binds their hashes and source versions.
Reversible source patches `corridor-v{1,2,3}-to-v4.patch` reconstruct each
earlier source exactly; V2/V3 test patches preserve their test suites. Source,
test and package snapshots and full traces remain in ignored cloud artifacts.
The submission ZIP contains only `agent.py`, passes syntax/member integrity,
and is 27,493 bytes; this is not a Linux cold-import certification.

Root and official simulator sources remain unchanged. Holdout has not been
evaluated. V4 remains a rejected research candidate and is not promoted.
