# Fresh cloud baseline at 14bb967

These are new Python 3.11.16 Linux runs of the committed checkpoint, not copies
of historical benchmark rows. NumPy 1.26.0, Torch 2.1.0+cu121, Gymnasium 0.29.1,
OpenCV 4.8.1.78, Box2D 2.3.5; `pip check` reports no broken requirements.

| Track / seed | Root | Robust | Hybrid | MPC |
|---|---:|---:|---:|---:|
| 1 / 516237 | 23.98 s | 22.30 s | 19.62 s | DNF, progress 0.90357 |
| 2 / 644062 | 31.00 s | 27.88 s | 26.20 s | DNF, progress 0.66071 |
| 3 / 1007 | 27.84 s | 26.72 s | 22.90 s | 31.88 s |
| 4 / 18800 | 24.82 s | 24.30 s | 21.82 s | DNF, progress 0.52330 |

All finished laps and all MPC DNFs had zero contacts. All MPC retirements were
`off_track`; increasing the 2000-step cap would not repair them. Hybrid used
700 steps; the other baselines used 2000. All four receipts are stored in
`baseline/`, including source/parameter/environment/action hashes and runtime
measurements. All candidates fail even an 18-second mandatory screen.

The 56 existing Apex tests passed in 1.63 seconds. The isolated existing
submission-package tests passed 11/11 in 15.89 seconds. The remaining full
suite reported **1159 passed, 10 skipped, 15 failed** in 447.11 seconds, for
**1170 passed, 10 skipped, 15 failed** combined. Four old V4 protocol tests
require Box2D 2.3.10, while the official Gymnasium extra and repository lock
install box2d-py 2.3.5. Eleven old consumed-screen provenance tests reject
LF bytes where their frozen protocol hash corresponds to CRLF; their ignored
historical run artifacts are also absent. No old source/test was changed to
conceal these failures. Packaging ran in a fresh process to avoid inheriting
the memory-heavy suite's Linux RSS high-water mark.

A fresh 20-cell hybrid benchmark in `baseline/hybrid-development.json` also
finished only **10/16** extra cells: track completion counts were 1/4, 4/4,
2/4, and 3/4. No extra finish was within 13 seconds. The benchmark role does
not count toward prospective candidate rejections.

## Provenance and limits

The official participant repository still points to
`dfb7a2de2178825ca5c5ce20bab01ba67052ba31`; the seven official environment,
finish, contact, and physics files match byte for byte. The official website
is inaccessible here (proxy CONNECT 403; direct DNS fails), so website-only
announcements and submission details remain unverified.

The old ignored receipts are absent from this cloud checkout. The preserved
robust/MPC source hashes in the old benchmark equal the current files encoded
with CRLF rather than LF. This explains the byte hash discrepancy, but does
not establish identical numerical behavior across platforms. The hybrid's
historical evaluated hash differs by a trailing newline, as already recorded.
Fresh Linux receipts bind the actual current bytes. Root and robust laps
reproduce the prior times; hybrid and especially MPC do not reproduce all
historical times or completion outcomes.

Root `agent.py` SHA256 remains
`2b2248f0d8fc2d35eddf20a6b9cca0302b95a428c04a3459b564ec6e2181c988`.
No root agent or official simulator source was changed. Holdout simulations
have not been opened.
