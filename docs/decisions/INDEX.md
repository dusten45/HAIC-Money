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

## Do Not Scale the Current DreamerV3 Formulation

**Context:** Pre-repair native DreamerV3 passed packaging/recurrent and short training
gates.

**Evidence:** The pre-repair pilot policy saturated steering and achieved no screen
finishes; its diagnostics did not establish a world model suitable for control.
See [`dreamerv3-feasibility-gate.json`](../../experiments/dreamerv3-feasibility-gate.json).
A1-A3 repairs and B1 tooling were subsequently implemented, but all nine
source-pinned random-data world-model studies v1-v9 failed their frozen B1 gates
in both learner seeds. See the
[`v1-v9 summary`](../../experiments/dreamerv3-b1-iteration-summary-v1-v9.json)
and [`B1 diagnosis`](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json).
No post-repair policy was trained in those studies.

**Decision:** Do not launch the 131k matched run for the pre-repair formulation
or the B1-failed repaired studies. Keep policy training blocked under the active
recovery plan; this is not a family-wide rejection of DreamerV3.

**Revisit condition:** The active plan's correctness, open-loop, counterfactual,
and renewed pilot gates pass.

## Preserve Internal Blind Partitions and Separate External Actions

**Context:** Reusing internal blind cells or treating a local package as a submission
creates selection leakage and false official provenance.

**Evidence:** Each frozen protocol records consumed/reserved partitions and explicit
non-promoting diagnostic paths.

**Decision:** Reserved blind cells are terminal-only. Official submission and model
confirmation require separate explicit user authorization and official-source
refresh.

**Revisit condition:** A new protocol reserves new cells or competition rules change.

## Keep Future Algorithm Families as Candidates, Not Active Work

**Context:** The prior algorithm roadmap listed TD-MPC2 and Dreamer 4 after DrQ-v2
and DreamerV3.

**Evidence:** No current artifact establishes a feasible CPU/package path or a
matched improvement for either family. DreamerV3 A1-A3 repairs and B1 tooling
are implemented, but all nine two-seed v1-v9 B1-only studies failed; no
post-repair actor performance is known.

**Decision:** Keep those families as strategic possibilities in the archived roadmap
([`2026-09-23-algorithm-migration-plan.md`](../plans/archived/2026-09-23-algorithm-migration-plan.md)),
not as active tasks. Do not skip evaluation/compatibility gates merely because they
appear later in an old sequence.

**Revisit condition:** The active DreamerV3 decision closes or new evidence justifies
a separately scoped plan.
