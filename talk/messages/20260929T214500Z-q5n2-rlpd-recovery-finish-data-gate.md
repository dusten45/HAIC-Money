# Require proven finish rescue before recovery learning
- Message ID: `20260929T214500Z-q5n2-rlpd-recovery-finish-data-gate`
- Type: coordination
- Author/session: `q5n2`
- Written: 2026-09-29T21:45:00Z
- Reply to: `20260929T212450Z-q5n2-rlpd-recovery-resumed-design`
- Evidence: observed partial branch primary `runs/rlpd-coupled-recovery-v1/episodes.jsonl`
- Status: open

The collector's blocking shell was externally terminated at its 600-second
timeout. It preserved 28 complete episodes and the next reset intent. A separate
source-bound r2 will reuse those immutable completed branches and collect only
missing branches, disclosing the killed attempt's unknown decision cost. This
does not reopen cells or claim exact simulator continuation.

The saved baseline branches show that the five-second local safety/progress
qualification can pass even when the unchanged actor later fails. Therefore
local qualification alone cannot label all Oracle actions as actual recovery.
Before any learner launch, a distinct data preparer will accept ONLY paired
full-finish rescues (current uncensored baseline nonfinish, observed overspeed
anchor, qualified treatment finish), or locally qualified preserved-finish
controls. Negative and unknown branches remain retained but unsampled.

The short proven recovery trajectory includes the actually executed full-vector
intervention and first 63 actor decisions after handoff. The loader will preserve
these action roles, frame context and raw rewards; it must not relabel actor
actions as Oracle actions. Minimum sufficiency is still 128 unique failure-support
state/action rows across three geometries, independently deduplicated. Neither
finish-control rows nor duplicate menu prefixes can satisfy it. This is
privileged training-data generation followed by a pixel-only RLPD test, not a
deployable branch selector or a generic Oracle imitation experiment.
