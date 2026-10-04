# Bounded camera motion study

No control change was accepted. `fast_motion_agent.py` is an observer with the
exact frozen V5 controller embedded as its base. It receives only camera arrays;
simulator velocity is read exclusively by the development diagnostic after
inference has returned. No new candidate lap screen or holdout was run.

The study replayed one frozen mandatory track-4/seed-18800 prefix of 120
decisions. Actions matched the saved V5 trace exactly. A second identical replay
cached camera arrays; the second model then used that cache without simulation.

| Model | Confident samples at >=20m/s | Heading MAE / P90 | Lateral velocity MAE / P90 | Supported large-slip samples |
| --- | ---: | --- | --- | ---: |
| V1 center registration | 69 | 1.04 / 1.79 degrees | 1.10 / 2.08 m/s | 1 of 5 |
| V2 two-edge consistency | 65 | 0.84 / 1.43 degrees | 0.95 / 1.84 m/s | 0 of 5 |

Confidence means >=0.7. Large slip means absolute actual travel heading above
0.20 radians, with speed >=20m/s. V1 falsely reported -14.7 degrees at confidence
0.82 while the actual heading was -1.5 degrees. V2 rejected that false confidence,
but also rejected every large-slip sample, including its onset. This evidence
does not support feeding the estimate into steering or propulsion.

The estimator registers static road geometry with HUD-yaw rotation compensation
and a HUD-speed distance prior. The road alone cannot resolve longitudinal
translation on a straight. Clipped edges, road branch changes and rapid rotation
reduce support. Endpoint heading correction assumes roughly steady yaw over the
camera interval. Low motion confidence also occurs in benign frames, so it is
not established as a braking trigger.

Eight focused motion regressions pass, covering translation, rotation, curved
registration, incompatible boundaries, missing data and reset. The 19 frozen V5
tests also pass. This is 27 focused tests, not a claim that the entire repository
suite is green.

Current source SHA256:
`97fe8719664b329cac389bcb52c690138e3d17a47d4ff4a4e652b4708b49073c`.
The tracked reverse patch reconstructs V1 exactly, and `motion-lineage.json`
binds source hashes, receipts, the camera cache and the unchanged V5 prefix.
The former study's holdout was already consumed; no new holdout was opened.
The four-track 13-second goal remains unmet, with no formal adoption.
