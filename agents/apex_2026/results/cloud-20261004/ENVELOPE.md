# Heading-change curvature envelope study

`envelope_agent.py` replaces the short quadratic second derivative with heading
changes between overlapping 10 m linear road fits, separated by 4 m. Dividing
the angle change by the distance between fit centroids estimates physical road
curvature. The approach envelope permits braking distance after reserving
2.6 m plus one 0.08-second decision's travel. Pursuit and actual steering speed
ceilings remain active. Sparse support falls back to a global quadratic fit.

The source SHA256 is
`80a8feedf4b2feeabaed86f029498a3c7d1967289ba0f95f88a6efed0efdf087`
(126,584 bytes). It imports NumPy and allowed standard library modules only.
Eighteen focused tests and the combined 96-test Apex/preview contract pass.

## Quantified camera geometry checks

- Straight roads with slopes 0, 0.10, 0.25 and -0.35, rounded to camera pixels
  with at most half a pixel error, retain the 100 m/s curvature target.
- A radius-30 m circle predicts 65.83, 65.86 and 66.14 m/s at camera rotations
  0, 0.35 and 0.60 radians, versus the physical estimate 65.95 m/s.
- The same quadratic bend near the car limits the target to 50.61 m/s. Moving
  it 18 m ahead permits 68.75 m/s at current speed 80; stationary planning
  permits 77.50 m/s. A five-point sparse quadratic curve limits speed to 45.51.

These controlled checks support smoother curvature estimation. They do not
establish driving completion or race pace on unseen geometries.

## Fresh required-lap screens

Both receipts under `probes/` are exact copies of new cold benchmark receipts;
their source, parameters, environment hashes and action hashes are retained.
They are mandatory screens and do not count as complete development rejections.
Source bytes are also preserved in the ignored artifact directories.

| Source/parameters | Track 1 | Track 2 | Track 3 | Track 4 |
|---|---:|---:|---:|---:|
| Fresh starting hybrid | 19.62 s | 26.20 s | 22.90 s | 21.82 s |
| Envelope default: 100/.19/brake100 | 20.16 s | DNF | 23.30 s | DNF |
| Envelope conservative: 95/.23/brake60 | 19.90 s | 25.68 s | 23.24 s | 22.20 s |

Default track-2 DNF reaches 0.68155 progress without contact; track-4 reaches
0.51971 with one contact. Conservative parameters complete all four without
contacts, damage or full off-road samples. Partial off-road samples are
0/10/0/2. Compared with the fresh hybrid, track 2 gains 0.52 s, while the total
time increases from 90.54 to 91.02 s (0.53%). Aggregate pace gain is unproven.
Historical hybrid track-2 time 26.68 s is not the fresh baseline comparison.

Maximum measured cold runtime across these screens: initialization 20.91 ms,
reset 0.068 ms, action 46.62 ms, and RSS 96.68 MiB. These local measurements
meet the stated bounds; they are not official container certification.

The conservative defaults are passed explicitly as
`{"cruise_speed":95.0,"preview_time":0.23,"braking_accel":60.0}`.
Lateral acceleration remains 145, heading window 10, heading step 4. The
current source's embedded defaults remain 100/.19/brake100.

## Independent review gaps

The inherited guided route retains robust corridor steering but can replace
its gas/brake commands with metric pedals and report the faster metric target.
Safety V2's target, crawl-gas and brake protection has not been combined into
this measured source. A valid corridor therefore does not establish safe
guided propulsion for this lane.

Review also reproduced a localized sparse bend whose global quadratic fallback
permits 81.96 m/s at current speed 70, rather than respecting its local curve
and approach distance. The visible-horizon braking cap still assumes 100 even
when `braking_accel=60` is passed. A separate guarded variant investigates these
defects without changing the source during its full development run.

## Fresh complete development result

`development/r5-envelope.json` evaluates the same frozen source and conservative
parameters on all 20 development cells. Its prospective limit remains
**15 seconds**, bound after four complete prior rejections; the original
13-second goal remains unchanged.

- Required laps reproduce **19.90 / 25.68 / 23.24 / 22.20 seconds**, zero contacts.
- Additional completion is **12/16**. Per-track finishes are **1/4, 4/4, 3/4,
  4/4**; track 1 fails the fixed 75% floor.
- The 15-second profile fails, and the original target remains unmet. This is
  the fifth consecutive complete development rejection.

The receipt retains each DNF and its progress/contact data. Neither mandatory
screen meets the original 13-second goal or the 15/18-second alternatives.
**Original target met: false.** This source is not promoted. The separate
guarded variant has no measured development outcome here. Holdout remains
unopened.
