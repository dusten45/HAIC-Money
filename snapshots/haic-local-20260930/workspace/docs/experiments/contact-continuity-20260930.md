# Completion diagnosis and contact continuity — 2026-09-30 04:28 KST heartbeat

## Frozen TRAIN discovery

Registered `completion-discovery-20260930`, plan `7a8a7a81bc1f8ed089cc7ba4cbc500d84885e1b3601797f28c16f8e7d1622d76`. Tracks1–3 × new TRAIN50300–50303, 24 episodes, 140.007 s. Historical9/12 versus current arrival10/12. Diagnostic coverage, not a new candidate batch or promotion. Prior ADVANCE cannot be a CLI continuation: initial registration attempt rejected before execution; final manifest starts a new root. The design's formal-predecessor sentence is superseded by batch_audit registration_note. REVISE/STOPPED, gates PASS/PASS/UNKNOWN. All24 hashes valid/errors0/firstten equal. Source policies unchanged.

Current failures2:50301 and2:50302 accrue five damage increments in five consecutive decisions near the same nearest obstacle index4 (distance~3.8–4.0). At2:50301 steps171–175, speed collapses43.6→14.2, then gas0.6 and alternating steer; at2:50302 steps136–140 analogous11.1→3.7. Damage20%→100% over0.32s. Prior-brake veto is still active during collapse, so inherited impact recovery can be suppressed. This is observed action/state association, not proof that every brake veto causes a failure. Historical1:50301 loses road rows and commandszero steering; arrival policy completes it.

## Four independent mechanisms

Run `contact-continuity-20260930`, plan `adf61af9da52b3a243bf797e9d82bb4e5d15c6484a0230216e4cf13734da9fb8`. Same consumed TRAIN12cells, 6arms/72episodes, 427.805s. New module contact_continuity_runtime.py. Pre-freeze read-only review caught altered steering clipping order; repaired to preserve parent ordinary two-stage clipping and reconstructed-road semantics. No hidden simulator input. Separate standing-authorized design/execution records.

| Arm | Finished | Median finished lap |
|---|---:|---:|
| Historical | 9/12 | 19.86s |
| Current arrival | 10/12 | 19.04s |
| Impact classification override | 10/12 | 19.04s |
| Contact propulsion release | 10/12 | 19.04s |
| Avoidance side persistence | 11/12 | 19.12s |
| Projected image crossing | 11/12 | 18.86s |

Allfour mechanisms actually alter actions; exact24reference trajectories reproduce. All72 hashes valid/errors0/firsttenequal. Full p50/p95/max latency, RSS, create/reset and distributions in batch_audit. Import time not separately measured by screen.

Side persistence changes only obstacle4 decisions133–136 on2:50302, preserving left avoidance rather than reversing near the object, and finishes without collision. Crossing rescues2:50301; it changes decisions55(obstacle1) and169(obstacle4), so do not attribute the rescue solely to the last intervention. Contact recovery/coasting do not restore either failed finish. Crossing selected by preregistered completion-first then median rule; side candidate remains preserved in the frozen source and traces. Screen REVISE/STOPPED, UNKNOWN/PASS/UNKNOWN pending package validation.

## Exact package and untouched held-out validation

`contact-package-20260930`, plan `62fd7b9e2b8738d2be2c74caa542ec6af4974c856325db02d266e202a041e670`. No policy edits. CandidateZIP12TRAIN replay trajectories and terminal outcomes exactly match screen. New held_out tracks1–3 ×51300–51303, candidate/current/historical each12episodes. Total48episodes382.190s; allthree runs144episodes950.002s, each2CPU/2GiB, below total1200s. All48 raw hashes valid, errors/invalidactions0; sameprefix across all three controls. Cold import/reset/action/RSS and ZIP structural/static limits pass.

| New held-out arm | Finished | Median finished lap | Collisions |
|---|---:|---:|---:|
| Current arrival ZIP | 11/12 | 19.06s | 3 |
| Crossing candidate ZIP | 12/12 | 18.96s | 0 |
| Original historical ZIP | 12/12 | 19.67s | 0 |

Current3:51302 fails; candidate and historical complete. Candidate's median cannot be called a speed gain over current because finish populations differ. On eleven common finishes, candidate is slower in7, tied1, faster3 (deltas+140,+220,+40,+360,0,+20,+40,-100,-40,+160,-60ms). Completion-first wins despite small time costs. Against historical, all12 finish and median improves0.71s(3.61%). No universal completion or half-lap-time goal claim. Prior held_out49300–49303 was not rerun or tuned; new candidate performance on that older set remains unmeasured. New51300–51303 is consumed held_out, never fresh/tuning data. Confirmation/blind untouched.

Candidate ZIP: `artifacts/haic-research-v2/contact-package-20260930/submission-crossing-projection.zip`, SHA256 `4447c8dfad1392ddbbb83f0b3d23b8d5237af2b9be785cdfd08edaf6c4623f64`.
Previous current ZIP preserved at far-hazard-package-20260930/submission-arrival-speed.zip, SHA256 `b168a17dac5fe5d6b28a265fb4adeeba6b3346391b35acd4a5818e7dc0b9693b`.
Original historical checkpoint preserved at fast-completion-row-repair-20260929/submission-fast-completion.zip, SHA256 `7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835`.

## Next bounded work

Preserve the two successful TRAIN mechanisms separately; do not tune from held-out geometry. Next investigate interactions on TRAIN using a preregistered four-direction batch or validate frozen side candidate independently. Current crossing does not solve every consumed TRAIN failure (11/12). Continue optimizing rather than ending the automation. Official source compliance evidence reuses previous heartbeat's current public repository audit; no official-source conflict found, no external actions. Local release is not official model confirmation/submission/upload.

Final integration: local ADVANCE, PASS/PASS/PASS, RELEASE_IF_GATE_PASS then STOPPED; independent read-only review found no blocker. Not official submission/model confirmation. Both controls and exact selected ZIP hashes checked again.
