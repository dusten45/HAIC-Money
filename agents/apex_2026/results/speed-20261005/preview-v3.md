# Committed camera pass V3

## Fresh mandatory screen

Source `fast_preview_commit_agent.py`, SHA256
`6b5ec35a93a1facd9879f36a5cecd813fdba6f3626a8c0b10bc4bb88b77a92d7`,
parameters `{}`. Exactly one new four-cell benchmark used workers 1 and
700 maximum actions. Every receipt is source-bound; errors are null.

| Track | Seed | V2 result | V3 result | Contacts | Full / partial off-road samples |
|---|---:|---|---|---:|---:|
| 1 | 516237 | 20.20 s | 19.42 s, finished | 0 | 0 / 6 |
| 2 | 644062 | 26.08 s | 25.22 s, finished | 0 | 0 / 0 |
| 3 | 1007 | 22.82 s | 21.50 s, finished | 0 | 0 / 3 |
| 4 | 18800 | DNF, progress 0.5197, one contact | 22.74 s, finished | 0 | 13 / 21 |

Track 4 has a valid forward finish at progress 0.971326. Its 13 full off-road
samples remain a material limitation. Zero contacts does not establish a safe
vehicle path or reliable completion outside these four cells.

The three common V2 finishes improve from 69.10 to 66.14 s total (4.28%).
All four V3 finishes total 88.88 s versus the fresh hybrid benchmark 90.54 s
(1.83% faster). Fresh hybrid track 4 was 21.82 s, so V3 is 0.92 s slower there.
These are distinct whole-source results, not a composite of best per-track
values. No required lap is at most 13 seconds. **Original goal: false;
adopted: false.** Mandatory benchmark screens do not advance the prospective
development rejection count. No additional development or holdout was opened
by this lane.

## Change and causal regression

The entire frozen V2 source is retained byte-for-byte except its public class
is renamed `_PassAgent`. A new public subclass adds camera-local commitment
and tracking. V1, V2, root inference and the official simulator are unchanged.
The reversible zero-context patch and full lineage record accompany this
report.

The causal regression contains only three official camera observations from
an exact 113-action replay of the consumed V2 track-4 benchmark. V2 resets
its pass from ego each frame: peak curvature 0.02451, 0.07147, then a rejected
0.15667 /m at steps 110–112. The last exceeds the 0.13049 /m physical joint
limit. V3 preserves the original local curve under a rigid camera transform.
Its peak remains 0.024512 /m on all three observations; rechecked road margins
are 2.949/2.949/2.830 m, circle-hull margin 1.168 m and centerline separation
4.007 m. Targets are 77.05/64.92/70.38 m/s. These are camera regression
diagnostics, not new driving validations or proof of causal isolation on a
changed vehicle trajectory.

The transform uses legal red yaw-HUD pixels and the currently associated
circle displacement. Rightward ego yaw rotates previous (x-right,y-forward)
points with standard positive rotation, moving a stationary ahead point left.
Circle displacement corrects translation rather than assuming that body
heading equals velocity. A 2 m consistency gate compares that circle with the
HUD speed/yaw turning-arc prediction. Passed reference points are trimmed at
current ego, and signed parametric curvature is retained.

Every transported acceptance retains the V2 road support/cubic ambiguity
gate, the full-vehicle half-width 1.6 m, road margin 0.30 m, official obstacle
radius 1.2 m, obstacle margin 0.75 m and centerline separation 3.5 m. Actual
ego footprint is checked separately. Non-monotone support, missing circle,
inconsistent motion, lateral reference error over 1.5 m or heading error over
0.45 rad rejects commitment. Local signed curvature, heading, lateral and yaw
feedback track the accepted curve with inherited joint, force, emitted
steering slew and rear-wheel propulsion limits. Clear-road defaults are V1's
100 m/s, lateral budget 190 m/s² and preview time 0.12 s; pass cruise is 85 m/s
and separation is 4.0 m.

## Verification and artifacts

- New behaviors first produced 10 expected RED tests. Initial implementation
  passed 35; added rejection cases passed 38. Independent review found the
  missing current cubic ambiguity check; its behavioral test was RED before
  the bounded correction. Final focused suite: 39 passed in 1.71 s.
- Entire Apex test suite: 307 passed in 7.96 s before the benchmark.
- Independent read-only review cleared the exact final source before driving.
- Maximum initialization 3.154 ms, reset 0.0775 ms, action 71.738 ms and
  worker RSS 94.176 MiB, within the published limits for this screen.
- Deterministic ZIP: 31,070 bytes, SHA256
  `1f169dbadf2cde772de7e848c8ba810d53c63a092c8ff753e9a398f36d4c194d`.
  Sole member `agent.py` exactly matches the evaluated source; package AST and
  prohibited-import/call checks passed. This is not official container
  certification or an extracted ZIP driving repeat.
- Exact compact receipt: `preview-v3-mandatory.json`. Source, tests, ZIP and
  complete traces are preserved under `.haic-artifacts/apex-speed-preview-v3/`.
  Camera fixture, probe, trace summary, lineage and patch are separately bound.

## Remaining limits

Current-road geometry still assumes a single-valued cubic: branches/hairpins
can reject to inherited target 18 m/s. That target persists for 6/25/8/35
actions by track. HUD median speeds are 51.31/46.42/51.31/44.02 m/s, far below
the pace required for the original target. Circle association, grayscale
raster bias and quantization remain heuristic measurements. Reference
clearance and feedback do not certify tire grip, steering lag, sideslip,
dynamic reachability, visibility or arbitrary obstacle identity. Track 4's
off-road samples make this distinction concrete. Further research needs
source-bound development coverage and better camera path/vehicle tracking;
this source is a measured research checkpoint only.
