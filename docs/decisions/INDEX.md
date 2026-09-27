# Decision Index

Only durable choices that prevent future agents from repeating the same debate are
listed here. Protocol/result artifacts remain the detailed evidence.

## Preserve the Official Environment Boundary

**Context:** Local research can make modified physics or environment behavior look
better without improving the competition agent.

**Evidence:** The official compatibility boundary is represented by `core/`,
`env_wrapper.py`, and `damage.py`; the official source remains higher priority.

**Decision:** Do not modify those files as an experimental performance treatment.
Treat an official upstream update as a compatibility migration, not a model change.

**Revisit condition:** Current official source changes the contract.

## Do Not Promote the Historical DrQ-v2 Pilot

**Context:** A historical actor reported 4/24 finishes on an undocumented earlier
grid.

**Evidence:** Its preregistered fresh screen was 0/24; diagnostic confirmation was
4/32 but explicitly non-promoting. See
[`drqv2-promotion-v1-result.json`](../../experiments/drqv2-promotion-v1-result.json).

**Decision:** Retain it as evidence that DrQ-v2 can complete, not as a promoted,
submitted, or blind candidate.

**Revisit condition:** A new candidate follows a new frozen protocol and satisfies
its own gates.

## Reject Steering-Logit L2 at 0.001

**Context:** Saturated steering logits and zero squash derivatives motivated one
bounded repair.

**Evidence:** The matched treatment lost 3 and 7 confirmation finishes against its
two controls. Desaturation metrics improved in a narrow sense but did not establish
useful driving. See
[`drqv2-steering-logit-v1-result.json`](../../experiments/drqv2-steering-logit-v1-result.json).

**Decision:** Reject this coefficient. Do not sweep it, bundle it with other repairs,
or treat the measured saturation mechanism as proven causal.

**Revisit condition:** New evidence identifies a materially different, testable
mechanism and a separately approved protocol.

## Reject Augmentation Pad 1 and Stop Narrow DrQ-v2 Tuning

**Context:** Shift sensitivity justified one final controlled axis after L2.

**Evidence:** Pad 1 had zero finishes in both treatment seeds and confirmation
deltas of -4 and -7. See
[`drqv2-augmentation-pad-v1-result.json`](../../experiments/drqv2-augmentation-pad-v1-result.json).

**Decision:** Reject pad 1 and stop the authorized narrow DrQ-v2 hyperparameter
branch. The observation is that pad 1 was worse under this configuration; reduced
visual regularization as a cause remains a hypothesis.

**Revisit condition:** A distinct research direction with new evidence and an
explicit user-authorized plan, not a third follow-up sweep.

## Close the Current DreamerV3 Research Line

**Context:** The current local design and budget no longer justify continued
DreamerV3 work. Posterior/short-term prediction did not yield stable long-horizon
free-prior dynamics, and the remaining improvement path requires design-level
rework.

**Evidence:** All nine static-random-data B1 studies failed for both learner seeds
([`v1-v9 summary`](../../experiments/dreamerv3-b1-iteration-summary-v1-v9.json),
[`diagnosis`](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json)). The
later teacher-data prior-image and multi-source+H8 strict gates also failed; the
32-decision free-prior error was worse than simple repeat. The actual fresh-P1
seed audit remains `passed=false, inventory_complete=false`. The consumed-TRAIN
records and final gate status are indexed in
[`docs/experiments/INDEX.md`](../experiments/INDEX.md). The feasibility artifact's
older Gate 3 pass label is superseded by the later diagnosis.

**Decision:** Close this research line. Do not run further DreamerV3 experiments,
training, data collection, evaluation, or local hyperparameter/horizon/update
search. Current checkpoints are world-model diagnostics only; there is no selected
Dreamer performance model. No official submission, model promotion, or protected
evaluation use follows. This is **not** a theoretical impossibility judgment
about DreamerV3; the current implementation/data/contract did not resolve prior
dynamics and further progress would require design-level rework.

**Revisit condition:** Only the conjunctive conditions in the
[`DEFERRED / LAST-RESORT ONLY revival plan`](../plans/dreamerv3-revival-plan.md)
may reopen this line: the other serious algorithms lack sufficient performance
potential, a new design-level research line is accepted, and failed B1/H8 work is
not repeated as-is. Existing TRAIN consumption and fresh-evaluation boundaries
must be preserved.

## Preserve Internal Blind Partitions and Separate External Actions

**Context:** Reusing internal blind cells or treating a local package as a submission
creates selection leakage and false official provenance.

**Evidence:** Each frozen protocol records consumed/reserved partitions and explicit
non-promoting diagnostic paths.

**Decision:** Reserved blind cells are terminal-only. Official submission and model
confirmation require separate explicit user authorization and official-source
refresh.

**Revisit condition:** A new protocol reserves new cells or competition rules change.

## Do Not Treat Historical Roadmap Order as Authorization

**Context:** Historical roadmaps placed TD-MPC2 and DreamerV3 later in an algorithm
sequence, which could be mistaken for an active queue.

**Evidence:** The archived roadmap predates current research status. DreamerV3's
current-design line is explicitly closed above, while other algorithm directions
have independent status and gates in `docs/context/current-state.md`.

**Decision:** Do not infer that an algorithm is active or next merely because it
appears later in the archived roadmap. Follow current-state and active-plan status;
the old sequence does not authorize experiments or allow skipping current
experiment/evaluation gates. See the archived
[`algorithm migration roadmap`](../plans/archived/2026-09-23-algorithm-migration-plan.md).

**Revisit condition:** Each family follows its own explicitly documented status,
evidence, and authorization boundary.
