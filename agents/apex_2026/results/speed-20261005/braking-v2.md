# Braking feedforward V2: correct saturated turn reserve

The camera regression gives 240.508 m/s² estimated lateral demand, above the
210 m/s² working tire-force budget. V1 clipped that demand to 209 before
calculating longitudinal reserve, adding brake 0.0430588 even at saturation.
V2 uses the actual demand and clamps the squared remaining reserve to zero.
The existing proportional brake remains; the correction removes only the
additional feedforward term when no reserve is available.

The new regression failed on V1 (1 failed, 3 passed), then the braking and path
focused tests passed together (23 passed). The zero-context source patch
reconstructs exact V1 bytes when reversed. The source stayed unchanged
through one fresh mandatory benchmark, defaults, 700 steps.

| Track | Seed | V2 official result | Contacts | V1 official result |
| --- | --- | --- | --- | --- |
| 1 | 516237 | 16.56 s | 0 | 16.60 s |
| 2 | 644062 | DNF, progress 0.6875, off_track | 0 | DNF, progress 0.6875 |
| 3 | 1007 | DNF, progress 0.3851132686, off_track | 4 | DNF, progress 0.3851132686 |
| 4 | 18800 | 16.76 s | 0 | 17.04 s |

The correction improves the two completed laps slightly but leaves both other
tracks unfinished. No early 10-second or original 13-second goal is met. This
is an experimental bug correction, not an adopted candidate. No new holdout
was opened. Privileged simulator state is absent from inference.

Source SHA256: `b65ccebae577adb6be8b8b78b1f414986b90bdbb502ee666c487ba60e0004900`.
Exact source, parent, test, patch, freeze and receipt binding are in
`braking-v2-lineage.json`; previous results remain in `braking-v1.json`.
