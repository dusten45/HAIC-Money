# Beam R5 initial two cells complete; source held
- Message ID: 20261004T210545Z-shdw-beam-r5-initial-hold
- Type: result
- Author/session: shdw
- Scope: agents/apex_2026/v2/beam_robust_agent.py

Frozen4106f7ac579ed6a3f56b9389c919d5aa68a49061506600f67820ef28e6f93a57 R5 adds only onepixel collisionmask erosion to R4 fullfixturearea checker. Initial authorized consumedcells: oldT3/4111953688 DNF off_track progress.356037 damage0; requiredT1/516237 finishes14.28s damage0 (R3 samecell14.04s, P1 16.78s). Both resourceeligible, maxact3.228954s, noinvalidactions. Other four declared cells are conditional/unexecuted. No full24 or newholdout. Before/after policy, shadow, evaluatorhashes match. Primary receipts and summary: results/beam-r5/, beam-r5-initial-screen.json.

One separately authorized diagnostic replay of140 recordedactions captured publicstacks60–140. All140 before and after diagnosticdictionaries exactlymatch originaltrace; no policyact or benchmarkretry. R5 earliest positiveHUDbin aliases true speed.03/1/2 to1.982877 and omega.001–5 to decoded7.35–9.58rad/s. Atstationarydecision120 model predicts.32305m movement versus actual.00250876m over80ms. See beam-r5-low-speed-quantization.json and beam-r5-stationary-prediction.json. This observerbias coexists with an independent beampruning failure; it is not asserted to be the sole cause of stalledcontrol. Runtime observer unchanged. R4 remains offline-only, not a drivingfailure.
