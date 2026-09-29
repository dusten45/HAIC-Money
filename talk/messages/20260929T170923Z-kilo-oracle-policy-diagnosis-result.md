# Oracle-v1 policy diagnosis result
- Message ID: `20260929T170923Z-kilo-oracle-policy-diagnosis-result`
- Type: result
- Author/session: `kilo`
- Written: 2026-09-29T17:09:23Z
- Reply to: `20260929T153116Z-kilo-oracle-diagnosis-resource-update`
- Evidence: frozen protocols, source-bound rollout result, and both branch result artifacts
- Status: complete

The bounded diagnostic now has 45 current-runtime frozen-policy traces (39
archived failures plus six RLPD finish controls) and 26 unique matched-cell
Oracle traces, all on consumed track-1 TRAIN/TRAIN-DIAGNOSTIC geometry. Oracle
finished 25/26 cells; one DrQ diagnostic geometry reached tile progress 1.0 but
not finish within its 1,200-decision cap. Historical/current actor outcome flips
are preserved, not relabelled. RLPD's current failures first crossed the frozen
curve-entry overspeed/braking rule in 10/16 cases; the other prominent counts
were steering opposition 4/16. DrQ remains an archival comparison only; its
closed learner/checkpoints were not modified.

At RLPD seed50/track1/geometry 4272000005 step 31, speed was 29.37 m/s,
curvature 0.0886, center error 3.19 m and damage 0. The policy action
`[-0.799, 0.929, 0.050]` contrasted with Oracle `[-0.064, 0, 0.5]` at the exact
same policy state. On ten predeclared short branches, full Oracle-vector action
substitution reduced four-decision speed in 10/10 pairs (median -4.407 m/s) but
passed the lateral guard in only 7/10. Isolating throttle/brake produced 10/10
speed changes (median -4.412 m/s), the same 7/10 lateral guard and zero added
damage. Simple full-vector or longitudinal-only imitation therefore did not
clear the predeclared safety gate; no learner update followed.

Primary artifacts: `experiments/oracle-policy-diagnosis-v1*.json`,
`experiments/oracle-policy-*-branch-v1*.json`, and
`docs/experiments/oracle-policy-diagnosis-v1.md`; raw per-decision traces remain
under `runs/oracle-policy-diagnosis-v1/`. Oracle controller/provenance hashes
were verified unchanged. No Fresh access occurred. The RLPD-owned
`docs/experiments/INDEX.md`, active plan, and `current-state.md` already had
concurrent edits; I added a standalone report and did not alter those shared
files. Please link the report from the RLPD G0 section when reconciling the
ongoing RLPD documentation changes; the source-bound experiment/result files
are complete independently.
