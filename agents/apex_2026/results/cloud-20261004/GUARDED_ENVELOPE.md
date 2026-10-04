# Guarded envelope correction study

`guarded_envelope_agent.py` combines the measured heading estimator with
corridor V4's camera topology, obstacle classification and curved hull checks.
It preserves robust hazard gas, braking and speed targets in the guided route.
Its embedded defaults match the selected explicit envelope parameters:
cruise 95, preview 0.23, braking 60, lateral acceleration 145, heading window 10,
heading step 4. Inference imports NumPy and allowed standard-library modules;
it does not import another candidate or the simulator.

Frozen source SHA256:
`c0e0443f35257deb6cf69c6b868df1f12955771c918ef4431df7b547e3398098`.
`probes/guarded-envelope-v1-lineage.json` records hashes and embedded constants.
The exact ignored snapshot is
`.haic-artifacts/apex-cloud-guarded-envelope-v1/source.py`.

## Source derivation

The definitions before `_MetricAgent` were copied from corridor V4 source
`43e163dfb4de301c85403d43bbaf7fa6a4de3267727badfb36273be5f1c2dd0e`.
The copied original prefix hash is
`6de78470cd4595129136ed6f04d417a1c89e20a93324d36df319fd09ef1b9032`.
Its two changes are recorded exactly in
`probes/corridor-v4-to-guarded-prefix.patch`:

- Obstacle radius is at least the official 1.2 m, even when a dim bright core
  gives a smaller camera bounding box.
- A comment distinguishes the saved raster's observed 171/255 grayscale peak
  from pure RGB orange's theoretical 173/255. The classifier band covers both.

The metric/public suffix derives from envelope source `80a8feed…`. It adds
sparse local geometry and braking corrections, plus safety V2's hazard pedal
protection. Frozen envelope and corridor source files remain unchanged.

## Reproduced defects and corrections

Eleven new behavioral regressions were observed failing before their fixes.
All now pass, along with all 20 corridor behaviors rebound to this standalone
candidate; combined original/envelope/preview/guarded checks passed 127 tests.

Independent read-only review confirmed these exact physical-gauge camera cases:

| Localized bend | Previous envelope target | Guarded target | Guarded action |
|---|---:|---:|---|
| Sparse camera bands | 81.96 m/s | 57.42 m/s | gas 0, brake 0.186 |
| Dense camera rows | 62.96 m/s | 54.26 m/s | gas 0, brake 0.242 |

The decoded gauge speed is 67.73 m/s. Sparse consecutive geometric segment
headings and centroid distances supplement the retained conservative global
fit when local heading supports are insufficient. Both sparse and dense
envelopes reserve braking distance from the support's observed near edge,
instead of its centroid. A prospective 1.2 curvature-peak factor accounts for
averaging within supports; driving experiments must determine its tradeoff.
The decision reserve is 2.6 m plus 0.08 times current speed. Declared braking 60
also controls the active visible-horizon budget.

Guided hazards retain robust steering and steering memory, cap gas at the
robust crawl request, retain explicit robust brake, and brake against the
minimum effective robust/metric target. Ragged/nonnumeric inputs produce
finite bounded recovery actions.

The exact small-core front-bar overlap test previously returned a positive
0.23096-pixel clearance. The official fixed-radius circle overlaps the physical
front bar by 0.19541 m; the new 1.2 m radius floor correctly rejects that pose.

## Remaining limits and evaluation status

The corridor check is a geometric test over observed poses, rather than a
complete dynamic trajectory certificate. Review identified an unchecked
first 2.94 m connection from the ego pose to the checked near horizon; steering
lag and curvature reachability are not certified by that checker. A diagonal
bright curb pixel can still merge with and invalidate an orange component.
These specific limitations remain. No complete-path safety claim is made.

## Fresh bounded driving probes

Exact receipts `probes/guarded-envelope-v1-mandatory.json` and
`probes/guarded-envelope-v1-target.json` bind the source above and parameters
`{}` to cold workers, environment hashes and action hashes. Embedded defaults
provide the selected 95/0.23/brake60 settings without evaluator overrides.

| Required track | Result | Contacts |
|---|---:|---:|
| 1, seed 516237 | 21.18 s | 0 |
| 2, seed 644062 | 32.12 s | 0 |
| 3, seed 1007 | 24.32 s | 0 |
| 4, seed 18800 | DNF, progress 0.54839 | 2 |

The previously consumed development failure `(1,1764402399)` also remains
DNF: progress 0.33916, one contact, damage 0.2, official retirement reason
`off_track`. There were no sampled full/partial off-road events in that worker;
sampled telemetry does not exclude transient events between decisions.

The fixes pass their camera/geometry regressions, but driving completion and
pace worsen. These bounded probes do not count as complete development
rejections. No promotion or speed gain is supported. A full development
run is rejected for this direction. Maximum measured local runtime is
initialization 3.78 ms, reset 0.072 ms, action 106.04 ms and RSS 95.45 MiB;
all five worker errors are null. **Original target met: false;
promoted: false.** Holdout remains unopened.
