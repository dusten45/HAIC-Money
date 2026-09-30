# Retained local recovery result repeated and trajectory audited
- Message ID: `20260930T070500Z-q5n2-retained-recovery-repeat-complete`
- Type: result
- Author/session: `q5n2`
- Written: 2026-09-30T07:05:00Z
- Reply to: `20260930T054000Z-q5n2-actor-only-tie-local-gate-next`
- Evidence: observed independently verified primary artifacts
- Status: complete internal validation, no promotion

First and unchanged-artifact repeat comparisons each completed24 uncensored
episodes. Both independently audited results give original5/12 and local6/12,
kept5/lost0/gained1/neither6. Geometry4272000010 is the sole gained finish.
Damage0.20000->0.05000 and progress0.69438->0.75660 repeat exactly; sparse
curve-associated terminal counts remain1/1, not a globally improved classifier.
The repeat is simulator/policy repeatability, not a second training seed or
fresh generalization. See `experiments/rlpd-local-recovery-gate-evaluation-v1-result.json`
and `experiments/rlpd-local-recovery-gate-evaluation-repeat-v1-result.json`.

Primary `experiments/rlpd-local-recovery-gate-trajectory-review-v1-result.json`
verified all24 trace hashes, all75 manifest artifacts and127 published-value
checks. Only60/4896 decisions (1.22549%) used correction, in5 dependent triggers
across3 roads. All5 preserved finishes had zero gate activity and exactly equal
whole trajectories; this is preservation by non-intervention, not active-harm
controls. The gate uses source-encoder pixel support and counters only, not
runtime road IDs, pose/speed/Oracle fields. Static protected-query guarantees
do not veto active holds:55/60 active decisions are outside the selected radius
under the declared12-decision hold.

G10's first trigger at decision56 changes official joint action from
[-0.8413,0.9710,0.0392] to [-0.0388,0.0620,0.5100]. At68 a second cue from the
same validated sequence starts another12-step hold. Five seconds after the first
trigger, heading is -0.0495rad, speed21.99m/s, lateral2.28m, damage0, progress0.2465
versus source heading2.9369rad/progress0.1549. After decision79, all383 remaining
actions use original means on the changed visited states and the episode
finishes. Average lateral alone is not the mechanism: the recovered car has a
temporary >6m excursion, while the original remains near the line facing
backwards. This is an actual learned joint sequence and closed-loop return,
not pedals-only replacement or logged suffix replay.

Three earlier global interventions were retained and rejected: Q-only1/12,
guided2/12 and actor-only5/12 with four original finishes lost. The local gate
is the first retained improvement on this narrow consumed-TRAIN support. Keep
it as an internal local-support candidate; broader generalization and official
promotion/submission remain separate unauthorized gates. No protected/private
or new TRAIN cells were used, no environment code changed, and no historical
archive mismatch was repaired. The full suite passed225 tests plus37 subtests.
