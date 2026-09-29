# Far-hazard acceleration and frozen ZIP validation — 2026-09-30

## Search protocol and results

Four independent mechanisms over the frozen sprint actor, plus unchanged stable and sprint references. Registered run `far-hazard-20260930`, plan `70eefa094cb3738357c5153a47d601cfebd4e37353ac938f514aec5e632988ce`. Separate design/execution records cite standing local authorization. Tracks 1–3 × consumed TRAIN seeds 38300–38303: 72 episodes, 459.821 s, 2 CPU / 2 GiB.

| Arm | Finished | Median finished lap | Collisions |
|---|---:|---:|---:|
| Original stable | 12/12 | 20.52 s | 1 |
| Frozen sprint | 10/12 | 19.15 s | 6 |
| Far-object pedal veto | 10/12 | 19.61 s | 4 |
| Temporal hazard | 10/12 | 19.61 s | 4 |
| Clearance preview | 8/12 | 19.86 s | 16 |
| Arrival-speed envelope | 12/12 | 20.13 s | 3 |

Arrival-speed peak 75.270 internal distance units/s. High straight acceleration remains; approach speed depends on observed object distance, with brake capped at 0.6. All 72 raw hashes valid, no errors/invalid actions, identical first ten decisions. Both references exactly reproduced all 12 prior trajectories. Search integrated REVISE/STOPPED with UNKNOWN/PASS/UNKNOWN pending package verification.

Mechanism evidence: arrival envelope changed 181 decisions relative to sprint, selecting far objects for 31 and near objects for 150. The result does not isolate the causal benefit of far detection alone. Temporal tracking added no distinct action beyond far veto; no claim of tracking benefit. Screen audit field `memory_only_changes` means action changes with no current far detection; for arrival_speed its 148 events mostly represent ordinary near-object decisions, not memory. This clarification preserves the original audit. Asphalt extent and positional matching remain heuristics, not certified path clearance/object identity.

## Frozen package validation

Run `far-hazard-package-20260930`, plan `e8a5d6debacaf9b52f4b475ea2744f7384ffeebcbf7a7353525ed89d4ddfa4ff`. No policy changes. 12 exact candidate TRAIN replays followed by candidate/control on tracks 1–3 × new held_out seeds 49300–49303 (24 episodes). Separate exact-hash design/execution approvals. All 12 TRAIN traces and terminal outcomes reproduced from extracted ZIP; all 36 raw hashes, imports, resets, actions and resources passed. Prefix equality held for all matched pairs. Runtime 243.247 s; both runs total 703.068 s. Peak RSS 284,217,344 bytes; cold import+creation max 0.2683 s; reset max 0.000104 s; act max 12.575 ms. Per-episode p50/p95 latency and distributions in batch_audit.json.

| Held-out arm | Finished | Median finished lap |
|---|---:|---:|
| Original ZIP | 10/12 | 19.34 s |
| Arrival-speed ZIP | 10/12 | 18.46 s |

All ten common finishes improved by 0.66–1.16 s. Median reduction 4.55%. Both failed the same two cells: 2:49303 and 3:49300, off_track. Candidate had zero held-out collisions; original had one. No 100% completion or global-optimum claim. Aggregate unique development+held-out completion is 22/24 for each, but split medians must remain separate. These held-out identities are now consumed and must not become tuning data or be presented as fresh again. Confirmation/blind untouched.

Original historical ZIP remains unchanged at `artifacts/haic-research-v2/fast-completion-row-repair-20260929/submission-fast-completion.zip`, SHA-256 `7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835`.

Candidate: `artifacts/haic-research-v2/far-hazard-package-20260930/submission-arrival-speed.zip`, SHA-256 `b168a17dac5fe5d6b28a265fb4adeeba6b3346391b35acd4a5818e7dc0b9693b`.

## Rules and remaining work

Current official Participants README checked via public source; priority website inaccessible. ZIP static checks and CPU resource contracts passed. Official core physics/damage files equal local copies after newline normalization; wrapper differs only by additive diagnostic info fields, not policy inputs or simulation behavior. Source URLs and hashes in official_source_audit.json and docs/sources/INDEX.md. No external actions, upload, official submission, or model confirmation.

Next iteration preserves this frozen candidate as a separate current local reference and retains the original historical benchmark. Use new registered TRAIN cells to study completion robustness and predictive steering/control; do not tune on held-out failure geometry. At least four independent mechanism directions for a new search batch. A valid matched improvement interrupts the prior non-improvement streak; this is not a third failed cycle. Overall optimization continues.

Final integration: package run ADVANCE with PASS/PASS/PASS, local RELEASE_IF_GATE_PASS then STOPPED. Independent read-only review found no remaining blocker. This is a local candidate release, not official submission or model confirmation.
