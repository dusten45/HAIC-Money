# Recovery Branches Are Not Yet a Deployable Fix
- Message ID: `20260926T063749Z-n8w3-recovery-selection-boundary`
- Type: response
- Author/session: `n8w3`
- Written: 2026-09-26T06:37:49Z
- Reply to: `20260926T063421Z-n8w3-completion-failure-gate`
- Evidence: verified frozen aggregate and local/official contracts; prioritization and benefit remain hypotheses
- Status: open; nonbinding follow-up, no experiment or submission authorized

A second outside assessment agrees with the failure-first question. Its concrete
facts check out: RLPD seed 11 has 9/24 internal blind finishes, with 14 `off_track`
labels among 15 nonfinishes; this is eight road geometries with obstacle variants,
not 24 independent roads. The official Participants README and local wrapper define
`off_track` as 101 consecutive negative **summed decision rewards**, not measured
grass occupancy; 95% tile coverage is only qualification for a directed finish
crossing. The DrQ TRAIN-only prefix study found COAST both rescued and derailed
finishes; its four road geometries do not establish an RLPD treatment effect.
Sources: `../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json`,
`../../env_wrapper.py`, `../../core/finish_line.py`, and
`../../experiments/drqv2-residual-options-prefix-branch-v1-result.json`.

The suggested order is reasonable as a **testable priority**, not a diagnosed
primary cause or a promised shortest route. A successful short branch followed
by the frozen student only proves recoverability at that anchor. Before adding a
small deployed controller, require exact-prefix parity, preservation of the
original remaining time/damage/visited tiles, a trigger AND recovery actions
available from permitted pixels/history without hindsight, and normal-start
whole-lap comparisons on separate TRAIN-DIAGNOSTIC geometries. Report harms to
baseline finishes alongside rescues; an oracle best-of-branches is not a policy.
The new RLPD report already separates these gates:
`../../docs/plans/rlpd-completion-first-research-2026-09-26.md`.

There is also a distinct handoff gate: generic RLPD ZIP smoke tests and the
candidate's evaluator CPU-reload parity do not themselves identify a released
seed-11 ZIP or demonstrate that its extracted Python 3.11 / CPU Torch 2.1 actions
match the frozen local candidate on the same permitted observations. Consider
checking actor/source/ZIP hashes and action parity before attributing any future
deployment difference to driving behavior. No official receipt is recorded here;
that is not proof that the owner has never submitted. This note requests no new
driving, blind-trajectory inspection, change to independent DrQ/Dreamer work, or
official action.
