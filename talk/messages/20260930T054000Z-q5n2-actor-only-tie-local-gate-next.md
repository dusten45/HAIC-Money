# Global actor correction ties completion but loses preservation
- Message ID: `20260930T054000Z-q5n2-actor-only-tie-local-gate-next`
- Type: result
- Author/session: `q5n2`
- Written: 2026-09-30T05:40:00Z
- Reply to: `20260930T003700Z-q5n2-guided-negative-frozen-actor-next`
- Evidence: observed primary fit and verified full evaluation
- Status: open

The user explicitly continued this lane at 05:29 UTC. The actor-only fit finished
2,048 updates with zero environment/SAC updates and every declared frozen encoder,
critic, target, temperature/reference/std-head component hash unchanged. Native
guide MSE0.00491, protected retention MSE0.00262; the full suite passed178 tests
plus37 subtests. Export/checkpoint SHA are bound in the current summary.

Primary `experiments/rlpd-recovery-actor-only-evaluation-v1-result.json` independently
checks 24 uncensored episodes. Original and correction both finish5/12, but only
road9 is shared: correction gains roads2/5/7/10 and loses original1/6/8/12. Damage
increases0.2->0.28333 and curve-associated terminal failures1->2. This fails the
net-improvement/preservation gate; the outcome-selected union9 is not a policy.

The next isolated hypothesis leaves original V5 output exactly unchanged when
inactive outside pixel-feature support. A calibrated gate near87 proved failure
Oracle images invokes the learned joint correction for a held12-decision window.
Each radius is half its closest protected feature distance, zero-radius support
disabled. Calibration references verified ordinary prior plus original initial
and source-finished snapshots, not road IDs or runtime geometry. Protected-query
no-trigger does not guarantee no correction during active holds or preservation
of whole successful paths; all24 full episodes and trigger/hold timelines remain
the outcome gate. Pixel-only local-support/memorization limitations will be
explicit, not framed as fresh generalization or privileged deployment.

Builder/runtime and evaluator implementations own separate new files. Existing
source-bound runs/core policy entry point stay unchanged. No new cells, protected
data, official action or promotion are in scope. Main will build and execute
only after tests, relevant source/data checks and artifact calibration pass.
