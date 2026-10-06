# Decision Index

Only durable choices that prevent future agents from repeating the same debate are
listed here. Protocol/result artifacts remain the detailed evidence.

## Close the Current H4-Cost Successor

**Context:** At 2026-10-06T02:04:20Z, the user ended the current H4-cost-based
successor after the matched single-intervention feedback branch. Preserve the
existing champion for this submission rather than continue local successor tuning.

**Evidence:** The [frozen ABC result](../../experiments/joint-single-branch-v1-result.json),
[primary summary](../../runs/joint-single-branch-v1/summary.json) and
[actual-feedback H4 costs](../../runs/joint-single-branch-v1/feedback-cost.json)
show a real short benefit, including under B's own champion feedback, but worse
lap time. In this consumed TRAIN case, cost favors C then B then A, whereas lap
time favors A then B then C. B advances at H4 but is behind by one second and
further behind at two seconds. This is a counterexample to the current selector,
not evidence that the first short benefit failed to reproduce.

**Interpretation boundary:** The earlier explanation that changed future actions
immediately erase the short benefit does not describe this first intervention.
The current cost, short horizon and terminal assessment fail to account adequately
for later feedback-driving loss in the measured case. Which steering, braking,
return or acceleration mechanism produces that loss remains unisolated. This
does not reject joint-control in general, identify the observer as the sole
bottleneck, or provide a safety/generalization theorem.

**Decision:** Close further tuning, driving, diagnostic branch extensions,
retries and candidate promotion of the current H4-cost successor. Keep it excluded
from submission candidates. Preserve exact champion ZIP `c9e376a0...`, root Agent,
distillation and all prior experiment evidence without rewriting pinned artifacts.
The temporal observer, compact physics predictor and associated validation tools
remain reusable research assets; do not delete or reset them. Other independently
authorized research lanes are not stopped or redirected by this decision.

**Future-work boundary:** Estimating how an H4 terminal state affects subsequent
champion feedback and long-term task performance is a separate research objective,
not an active plan or authorization to implement, train, collect or evaluate it.
Retrospectively changing this road's weights is not evidence that the problem is
solved. The user's MPC cost-design reference supplies context, not a transferred
guarantee or a uniquely proven mechanism in this experiment.

**Revisit condition:** Explicit new user direction is required to reopen research.
Candidate consideration must rest on measured actual-feedback lap-time improvement,
with matched provenance and observed failures retained, not lower H4 cost or more
interventions alone. No official upload, model replacement or server confirmation
is authorized by this closure.

## Stop Local Optimization of the KOI Champion

**Context:** On2026-10-04T15:23:01Z, after the sprint72-relief rejection and the
previous closed avoidance/recovery/corner directions, the user stopped further
micro-optimization of the current hand-crafted champion. This is broader than
closing only the latest coefficient or gate.

**Evidence:** The [fixed sprint72 experiment](../../experiments/koi-sprint72-relief-v1-result.json)
and [independent audit](../../experiments/koi-sprint72-relief-v1-audit.json) show a
source-correct local intervention, a real local undershoot reduction and changed
subsequent trajectories, alongside a lost finish, longer offroad exposure and
slower common-finish laps. Preserve both the rescued and lost cells. The
[full report](../architecture/koi-baseline-analysis-2026-09-30.md#32-fixed-sprint72-handback-relief-2026-10-04)
distinguishes local, unmatched and complete matched measurements.

**Interpretation boundary:** This supports sensitivity of the closed-loop system
to local changes, not a theorem that all heuristic improvements must fail or that
MPC will outperform it. The6.825->2.218 undershoot pair is one2/0006 window with
brake-to-gas1->0; repeated cycles3->0 is an own-window aggregate, not that same
matched window or a complete-cohort matched improvement. Brake history is coupled
to impact logic, but all observed triggers remain vetoed and genuine impact-clear
is zero in both arms; no unique impact-mediated cause of the DNF is established.

**Decision:** Preserve exact champion ZIP`c9e376a0...`, shield`ad772bde...`, all11
frozen sources and root Agent. End autonomous small-rule/threshold/margin/pedal/
steering/recovery/corner optimization, diagnostic searches and A/B extensions on
this champion. Do not reopen previously closed variants or relax their gates.
No candidate promotion or new external action follows. Other independent model
lanes are not implicitly stopped, redirected or authorized by this KOI decision.

**Future-work boundary:** The user raised retaining the champion through the
competition OR assessing a separately designed joint steering/pedal short-horizon
controller. Neither is an instruction to implement, train or evaluate a successor
now. A later authorized feasibility/design gate would need an observation-valid
state/prediction model, uncertainty and latency budget, minimal coupled-action
scope, and a tested interface to any reused shield. Evaluation-only physical
heading/progress/map state cannot silently become policy input. Reusing unchanged
shield bytes does not transfer a safety guarantee to a new nominal controller.
The user's four-track/overall-sixth summary remains user-reported, not independent
verification or a reason to consume official submissions.

**Revisit condition:** Only explicit new user direction may reopen this campaign
or commission a separate successor design. No reset, fresh/generalization/private/
official evaluation, upload or model confirmation is authorized by this record.

## Close KOI Overavoidance Optimization

**Context:** On2026-10-02 the user authorized one separate reaction-based bounded
nominal steering-magnitude hypothesis, preserving frozen crossing_projection plus
collision-shield v1. The user explicitly required discarding the candidate and
ending overavoidance optimization if meaningful safe efficiency was absent or a
regression appeared. Margin reduction, steering-release, persistent planning and
long recovery/release-state extensions were excluded.

**Evidence:** The [frozen four-pair consumed ordinary TRAIN A/B](../../runs/koi-avoidance-magnitude-v1/protocol.json)
completed all8 episodes/two roads. Its [primary result](../../experiments/koi-avoidance-magnitude-v1-result.json)
is REJECTED: finishes4->3, damage0->1.4, collision decisions0->7 and hit objects0->3
among all24 objects/arm. Five baseline windows and return followups are lost.
Conditional three-retained-cell lateral/path/steering means and kept-lap-120ms
cannot offset safety/coverage regressions; road0013 also increases path, steering
integral/variation and lap+80ms. Detailed boundaries are in
[architecture section27](../architecture/koi-baseline-analysis-2026-09-30.md#27-stateless-bounded-avoidance-magnitude-2026-10-02).
The [independent primary audit](../../experiments/koi-avoidance-magnitude-v1-audit.json)
verifies the same rejection and all17 gates without policy execution or reset.

**Decision:** Discard bounded magnitude v1 as FAILED / NOT ADOPTED and close
overavoidance optimization. Preserve source/protocol/raw/result and prior negative
evidence; no retuning, repeats, controller extension or gate relaxation. The exact
submitted champion`c9e376a0...`, shield`ad772bde...` and root Agent remain unchanged.
This closes the user-scoped research direction, not a theoretical impossibility
claim about all obstacle avoidance. Other research lanes are unaffected.

**Revisit condition:** Only an explicit new user instruction can reopen this
direction. No model promotion, upload or official confirmation is authorized.

## Designate Crossing Plus Shield v1 as Submission Baseline

**Context:** At2026-10-02T01:36:47Z the user reported that the current official
crossing_projection + collision-shield v1 submission finished Track4 in18.4s and
ranked6th overall, and explicitly designated it the new submission baseline.

**Evidence boundary:** This is a user-reported official result with user-bound ZIP
SHA-256`c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801`,
not independent package-to-result verification. No site submission ID, server
receipt or model-confirmation receipt was supplied. See the
[submission ledger](../competition/submissions.md#current-submission-baseline-2026-10-02).

**Decision:** Adopt this exact package as the **submission baseline**, separately
from the original research [NOT_ADOPTED result](../../experiments/koi-collision-shield-v1-result.json)
and failed+20ms efficiency gate, which remain unchanged. Preserve it under
`submissions/20261002-crossing-projection-collision-shield-v1-baseline/`; exact
ZIP/member checks and standalone restoration/source rebuild reproduce its pinned
hash without policy execution. Retain the original experimental
ZIP/receipt, standalone crossing, all previous candidates and root Agent unchanged.
This records selection, not a new upload, server confirmation or research-gate pass.

**Boundary:** No retuning or evaluation is authorized here. A future baseline
replacement requires a separate explicit designation; official external actions
retain their separate authorization and source-recheck requirements.

## Close KOI Minimum-Clearance and Margin Reduction

**Context:** On 2026-10-01 the user explicitly ended all minimum-clearance
variants as failed/nonadopted and stopped safety-margin reduction alongside the
already closed speed-target direction.

**Evidence:** The preserved [v1](../../experiments/koi-minimum-clearance-ab-v1-result.json),
[r2](../../experiments/koi-minimum-clearance-ab-r2-result.json) and
[r3](../../experiments/koi-minimum-clearance-ab-r3-result.json) consumed-TRAIN
comparisons do not meet their improvement gates. R2 loses a baseline finish;
r3 restores safety but not meaningful path/lap/steering improvement, verified by
its [independent audit](../../experiments/koi-minimum-clearance-ab-r3-audit.json).

**Decision:** All minimum-clearance variants are FAILED / NOT ADOPTED; preserve
all frozen sources/artifacts and stop margin shrinking. Crossing_projection
remains fixed. The separately authorized avoidance-steering generation/hold/
release hypothesis changes only steering lifecycle after source/trace diagnosis,
not margin or speed target, and does not automatically promote any candidate.

**Revisit condition:** Only an explicit new user instruction may reopen the
speed-target or margin-reduction directions. No official action is authorized.

## Close KOI Adaptive-v1 and Fix the Crossing Baseline

**Context:** On 2026-09-30 the user explicitly rejected adaptive-v1 and ended
the speed-target adjustment direction, while authorizing a separate obstacle
minimum-clearance/path hypothesis.

**Evidence:** The [unchanged consumed-TRAIN A/B](../../experiments/koi-adaptive-ab-v1-result.json)
and [independent primary audit](../../experiments/koi-adaptive-ab-v1-audit.json)
preserve completion and contact outcomes but fail the declared improvement gate.
The [KOI analysis](../architecture/koi-baseline-analysis-2026-09-30.md#14-첫-unchanged-ab-결과-채택하지-않음)
records its measurement limitations and missing original crossing ZIP boundary.

**Decision:** Adaptive-v1 is FAILED / NOT ADOPTED. Preserve all sources, ZIPs
and negative evidence; do not pursue further speed-target tuning. Fix
`ContactContinuityAgent('crossing_projection')` as the KOI improvement baseline.
A separate footprint/obstacle/road-boundary steering candidate must demonstrate
preserved finishes and safety plus shorter avoidance paths before internal
candidate judgment; it does not automatically replace the baseline or authorize
official action. Other model lines are unaffected.

**Revisit condition:** Only an explicit new user instruction may reopen the
speed-target direction. The minimum-clearance iteration was subsequently closed
by the decision above; the new steering-lifecycle hypothesis is not reopening.

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

## Close the DrQ-v2 Research Line

**Context:** The user ended DrQ-v2 work on 2026-09-29 and directed that no more
resources be spent on this idea. This supersedes the revisit conditions and
possible follow-ups in the historical DrQ decisions and plans below.

**Evidence:** The pad-4 controls remain historical internal baselines, not an
official model. Later geometry-mix, source-retention, residual-option and
speed-only studies did not establish a promotable improvement. The final
[`speed-only result`](../../experiments/drqv2-speed-reused-development-v1-result.json)
lost four previously completed local development cells across tracks 1-2;
track 3 had no paired completed lap. The experiment
[`index`](../experiments/INDEX.md) links the earlier studies and their limits.

**Decision:** **CLOSED.** Do not spend further resources on DrQ-v2 research,
training, data collection, diagnostics, evaluation, speed or completion tuning,
or candidate packaging/submission work. Do not promote the existing local ZIP
or relabel its internal results as official. Preserve all frozen protocols,
negative results, checkpoints and local package artifacts as historical evidence;
the closure is a resource-priority decision, not a proof that DrQ-v2 cannot
work in principle. Other algorithm lines are unaffected.

**Revisit condition:** Only a new explicit user instruction reopening DrQ-v2;
historical conditional follow-up language is not authorization.

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
