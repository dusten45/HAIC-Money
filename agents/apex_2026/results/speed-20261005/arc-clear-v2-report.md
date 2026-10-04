# Arc clear V2: preserve turn intent before shortening

Exact prefix camera fixtures corrected the V1 diagnosis: track 3 action 54 and
track 4 action 62 have no active pass or detected current circles. A pass-only
gate cannot prevent their harmful turn reversals. V2 excludes active passes
and current circles, and also rejects shortening that reverses the original
emitted turn sign. The original curvature/target state is restored on that
rejection. This conservatively bounds the camera heuristic; turn sign is not
a general safety proof.

The two reference-action regressions failed on frozen V1, then passed on V2.
Saved clear bend cameras 77/78 still select the supported near vehicle turn.
The combined arc-clear, arc-guard, early-circle and clear-supported focused
tests passed: 19 passed. The two fixture prefixes replay 55/63 exact earlier
actions and post-step speeds; they are camera collection, not lap validation.

One fresh mandatory benchmark used default parameters and 700 steps:

| Track | Seed | Official lap | Contacts |
| --- | --- | --- | --- |
| 1 | 516237 | 15.44 s | 0 |
| 2 | 644062 | 19.08 s | 0 |
| 3 | 1007 | 16.52 s | 0 |
| 4 | 18800 | 20.92 s | 1 |

All four official finishes include the required finish crossing. Track 4
coverage is 0.9749103943, above the official 0.95 threshold. Compared with the
preserved V1 receipt (15.44/19.50/19.76/19.78 s, contacts 0/0/2/1), track 3
improves and loses its two contacts, while track 4 becomes slower and retains
a contact. V1 results are historical reference data, not repeated controls.
No 13-second or early 10-second goal is met; V2 is not adopted.

The source stayed frozen throughout this benchmark. Exact Arc V1 definitions
are embedded under a class alias, and the zero-context source patch reverses
to the exact parent bytes. No memory-corridor controller was combined here.
No official source, existing frozen candidate or holdout was changed.

Source SHA256:
`15478c179ff1c8fa2c7dfe12bf65eaba91ad215fc250bde893ce65d7eebcd5ba`.
`arc-clear-v2-lineage.json` binds source, parent, patch, tests, actual hazard
cameras, freeze and receipt; model and dynamic limits remain those of V1.
