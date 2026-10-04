# Cloud continuation final research report

**The requested 10–13-second laps are not achieved. The prospective 18-second profile also fails. No independent candidate is adopted.**

Work continued from `14bb967` on `codex/apex-2026-independent-agent` in Python 3.11.16 Linux. Root `agent.py` and official simulator files are unchanged. The official participant revision `dfb7a2de2178825ca5c5ce20bab01ba67052ba31` was rechecked byte for byte; website-only announcements were inaccessible.

## Frozen research artifact

Source: `agents/apex_2026/guarded_preview_agent.py`, SHA256 `76863032117175870fbf7ef1a18b49fb2b966ca6b53dd58adcd79a1d2ee46bb1`. Parameters are `{}`; embedded defaults are cruise 100 / preview 0.19 / braking 100 / lateral 145. [Freeze manifest](final/freeze.json) was committed and pushed in `b43ef98` before holdout started. Its hash remains `b1915752995d689e1c07177ed3d1a67f366f9b66aa34eab427ee838db894e594`.

The reviewed change preserves robust hazard gas, brake and effective speed targets during guided routing. Metric/robust definitions match preview V4. Earlier heading and curved-path integrations lost mandatory finishes and remain rejected, with exact receipts and reversible patches.

This artifact is a reproducible current research snapshot. Preview V4 remains the balanced development completion reference. The guarded revision is slower on common finishes and has more development contacts; freezing it establishes neither adoption nor an aggregate performance improvement.

## Fresh required laps

Times use the official forward finish crossing with at least 95% progress. Warmup is excluded; FPS is 50 and frame skip is 4. DNF is never counted as a lap.

| Track | Seed | Lap | Contacts |
|---|---:|---:|---:|
| 1 | 516237 | 21.92 s | 0 |
| 2 | 644062 | 27.64 s | 0 |
| 3 | 1007 | 26.14 s | 0 |
| 4 | 18800 | 23.74 s | 0 |

All four action traces and outcomes match the independent cold source repeats, extracted ZIP runs and holdout mandatory repeats.

## Complete prospective development trials

| Trial | Limit | Required lap seconds | Extra finishes | Per-track extra finishes | Verdict |
|---|---:|---|---:|---|---|
| r1-safety | 13 s | 19.62 / 26.46 / 23.96 / 21.82 | 10/16 | 1 / 4 / 2 / 3 | reject |
| r2-pace | 13 s | 18.76 / 25.14 / 21.78 / DNF | 9/16 | 1 / 3 / 3 / 2 | reject |
| r3-recovery | 13 s | 19.62 / 26.20 / 22.90 / 21.82 | 10/16 | 1 / 4 / 2 / 3 | reject |
| r4-preview | 15 s | 21.94 / 27.16 / 25.12 / 24.44 | 12/16 | 3 / 3 / 3 / 3 | reject |
| r5-envelope | 15 s | 19.90 / 25.68 / 23.24 / 22.20 | 12/16 | 1 / 4 / 3 / 4 | reject |
| r6-corridor | 15 s | 19.98 / 24.78 / 22.68 / 21.76 | 10/16 | 2 / 4 / 1 / 3 | reject |
| r7-guarded-preview | 18 s | 21.92 / 27.64 / 26.14 / 23.74 | 12/16 | 2 / 4 / 3 / 3 | reject |

Only complete candidate trials covering the exact 20 development cells advance the counter. Mandatory screens and operational failures do not. R1–R3 used 13 seconds, R4–R6 used 15; R7 used 18 after six fresh complete rejections. No past outcome was relabeled. Every original-target verdict is false.

R7 finishes 12/16 extras (75%); track 1 finishes 2/4. No extra finish is within 13 or 18 seconds. Compared with preview V4, it loses `(1,359018627)` and gains `(2,4029571806)`, is 1.37% slower on 11 common extra finishes and records 15 versus 8 contact events across 20 cells. Compared with fresh hybrid, it gains two finishes but is 14.85% slower on 10 common extra finishes. See [development comparison](final/development-comparison.json).

## Holdout opened after freeze

Holdout is now consumed and must not be described as unopened in future work. Source, parameters and selection were unchanged after it opened.

- Extra completion: **15/16 (93.75%)**; per-track counts **4/4, 4/4, 4/4, 3/4**.
- Extra finish times: **20.00–25.18 s**. Zero finishes satisfy 13 or 18 seconds.
- The only DNF is `(4,2376840915)`: progress 0.409594, five contacts, crash retirement. Extra cells have six total contact events.
- The holdout does not rescue the failed development/profile verdict or qualify the source for adoption. No held-out retuning was performed.

## Reproducibility and execution limits

- Source cold repeats: 4/4 exact action hashes and semantic outcomes.
- Extracted ZIP driving: 4/4 exact action hashes and semantic outcomes.
- Independent audit recomputed 28 final trace hashes from finite bounded float32 actions and checked finish timing, receipt/source/environment binding and freeze chronology.
- Maximum import/initialization including isolated cold NumPy import: **84.06 ms**; reset **0.094 ms**; action **100.66 ms**; worker RSS **97.36 MiB**. Limits remain 10 s / 5 s / 5 s / 1024 MiB.
- Research ZIP: **26,704 bytes**, sole `agent.py`, SHA256 `12e68a101b6fb7456fd26e035f3dc1f0fc5e9a3f779d4e9a5f33bb9c35f6be12`. Its source/defaults match evaluation. Rebuild with `agents.apex_2026.package` from the frozen source.
- Isolated Python `-I` cold import succeeds without repository or Torch imports. Static prohibited-import/call, member/CRC/size/compression checks pass. These local cloud measurements are not official container certification.

## Tests and preserved failures

Fresh current Apex and new behavior tests: **213 passed in 5.73 s**. The starting 56 Apex tests passed before implementation. The pre-existing full baseline reports **1170 passed, 10 skipped, 15 failed**: four old protocol tests expect Box2D 2.3.10 while the official lock installs 2.3.5, and eleven historical provenance checks reject LF/CRLF hashes or missing ignored receipts. No old test or root/official source was changed to conceal these failures. `pip check` found no broken requirements. See [baseline](BASELINE.md).

## Remaining work

The fallback for sparsely sampled bends still underestimates some visible curvature, and legacy corridor geometry and dynamic reachability are incomplete. Post-contact stalls remain; stationary gas/steer recovery did not solve them. The privileged teacher study is labeled diagnostic and establishes neither a legal camera result nor impossibility of 10–13-second laps. Faster, reliable planning from the current pose and earlier hazard handling remain necessary; the existing receipts provide no competition-winning claim.

Large traces and ZIPs remain in ignored `.haic-artifacts/`; compact source-bound receipts, provenance and this report are tracked.
