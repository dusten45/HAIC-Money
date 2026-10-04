# Rear clear V1: release confirmed post-pass cooldown

The exact T4 prefix showed a supported ridge blocked by cooldown after a
completed rear pass. This candidate embeds exact Arc clear V2 definitions
under a class alias. Its route override calls the parent once and releases
cooldown only when a prior active pass clears, missing count stays at most 4,
no current circles exist, road is valid, and the unchanged parent camera
transport predicts `y < -5 m`. Missing-object expiry retains the old phase.
All ridge, near-body, force, circle-shape and road-support thresholds remain.
No simulator state, map, seed or actual rear label enters inference.

The actual input-101 regression failed on the parent (confidence phase), then
selected the inherited supported ridge. The yaw-transport release regression
also failed before implementation. Six new tests cover these transitions,
expiry, current hazard, not-yet-clear rear, invalid inputs and reset. Combined
rear-clear/arc-clear/arc-guard/early-circle/clear-supported tests: 25 passed.

One fresh mandatory benchmark used default parameters and 700 steps:

| Track | Seed | Official lap | Contacts |
| --- | --- | --- | --- |
| 1 | 516237 | 15.54 s | 0 |
| 2 | 644062 | 18.72 s | 0 |
| 3 | 1007 | 16.56 s | 0 |
| 4 | 18800 | 18.14 s | 0 |

All four finish crossings are official; track 4 coverage is 0.9928315412.
The preserved parent receipt reports 15.44/19.08/16.52/20.92 s with contacts
0/0/0/1. The cooldown fix improves track 4 and removes its contact, improves
track 2, and slightly slows tracks 1/3. Parent results were not repeated here.
No early 10-second or original 13-second goal is met. This remains research,
not an adopted candidate; additional reliability has not been validated.

Source SHA256:
`093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc`.
The source stayed frozen through evaluation. `rear-clear-v1-lineage.json`
binds source, parent, exact reversible patch, tests, actual camera fixture,
freeze, receipt and trace-only comparison. Official files and all earlier
frozen sources were preserved. No new holdout was opened.
