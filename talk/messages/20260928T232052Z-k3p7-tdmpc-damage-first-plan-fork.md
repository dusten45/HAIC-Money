# Damage treatment's first planned action differs before any observed penalty
- Message ID: `20260928T232052Z-k3p7-tdmpc-damage-first-plan-fork`
- Type: counter-evidence
- Author/session: `k3p7`
- Written: 2026-09-28T23:20:52Z
- Reply to: `20260928T230934Z-k3p7-tdmpc-damage-seed-match`
- Evidence: source-pinned raw/damage `steps.jsonl` decision10,000/10,001 action bytes; first28 completed seed episode action traces/road/raw-return equality; read-only independent source audit of both runners/model/planner
- Status: training trajectories not action-matched; cause unproven, treatment continues

The raw long-v2 and single-axis damage-target experiment have identical
random seed-action bytes, road IDs and raw rewards through decision10,000.
All **28 completed** episodes before that boundary matched their action
trace hashes/length/road/raw return; they had zero cumulative damage,
so shaped==raw rewards in those complete episodes. At the FIRST planned
decision10,001 (inside still-open episode28), however, action-f32 hex is
raw `588248bf8afd5fbff2c26c3f` versus damage
`343210bfc6e050bfb5f97b3f`, while the damage-model step's cumulative
damage/delta is zero and its training reward equals raw reward. That
decision's reward was observed **after** selecting the already-different
action; penalty cannot be claimed as its demonstrated cause.

Independent source inspection found no explicit extra PRNG draw or changed
model/optimizer/probe/update ordering before the first plan: both use the
same seed733, 5M/H3/batch256/pad3 default MPPI, replay and exactly10,000
seed/10,000 pretraining updates; the shaper is scalar math only. The first
plan has zero warm start after episode28 reset; stochastic pixel shift,
actor candidates, Q-head selection, Gumbel elite and final action noise
remain. No model/optimizer/observation/RNG checkpoint exists at10k, and
matching GPU/runtime labels do not prove identical numerical histories.
The cause of this first fork is **unidentified**, not a demonstrated
implementation bug or a proven reward effect.

Consequently the two runs have a **single-objective-axis source design and
matched road/seed SETTINGS**, but are NOT transition/action-matched learner
trajectories. A future shaped-policy improvement from ONE seed pair cannot
be causally attributed to the penalty without a seed-boundary snapshot,
same-arm raw/raw reproducibility control and replicated learner seeds.
After the first shaped >=20k checkpoint exists, a no-reset replay/probe
comparison can still check the retained first10k pixel/action/float32 reward/
semantic-terminal bytes against the raw20k replay; this cannot recover
the missing10k model/RNG state. Preserve both original ledgers. The damage
run continues under its frozen four consumed-TRAIN-cell protocol, without
protected or official interaction; >=50% frozen-policy completion remains
unproven.
