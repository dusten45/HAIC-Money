# Predictive batch standalone, V1

`fast_predictive_batch_agent.py` reduces physics-planning compute cost while
preserving the frozen predictive V2 perception and control formula. Its source
SHA256 is `93a6b4db3484a2621407a410af64fea6f88465cb49173f6c8234e94e5c240547`.
No new episode has been run for this source; no pace improvement or adoption
is claimed. The early-10-second goal remains unmet.

The standalone starts with every byte of frozen V2
`b70af66e0ba39ea975d53909ec6d7a19df2a73005a036465d71dd7ac8caeff8a`.
It captures the original `Agent` and `FixedControlPlanner`, appends exact
NumPy batch-model exports from `8ae7239b...`, and appends the batch planner
class from `263909e1...` using the captured original scalar class as its base.
The offline `build_predictive_batch.py` pins all three full hashes and
reproduces this source exactly.

The public appended Agent inherits V2's `act`, `reset`, camera road/hazard
perception, supported-prefix extraction, three state scenarios, and strict
1.9 m body/road, circle, and first-0.08-second full-stop requirements. Its only
planning changes are the batch callback and explicit owner budget method.
Inherited `act` resolves the global planner alias to a budgeted batch class;
the global scalar `predict_step` stays unchanged. In particular,
`CameraObserver.advance` still executes four scalar ticks after the final
emitted action, including fallback. The inherited parent visual update and
observer advance each occur once.

The original deadline starts before parent perception and is never restarted.
The callback checks before and after every batch model operation; the planner
checks before and after every trajectory geometry operation and once more
before returning a successful final selection. Private deadline fields are
read only by the owning Agent's explicit public method. A timeout becomes
`planning_error` with `_PlanningBudgetExceeded`, so inherited V2 selects its
normal timeout fallback and still advances actual action history. Constructor
timeout checks do not throw outside inherited error handling.

`predictive_model_calls` now counts actual raw row predictions, matching
`batched_model_rows`, rather than batch callback invocations. Planner
`model_steps` retains scalar logical visited-scenario work. The eager batch
computes additional rows compared with scalar early rejection, so actual work
counts need not match even when every selected control and policy metric does.

Ten focused tests passed in 16.34 seconds. They cover scalar observer/model
namespace preservation, exact batch recurrence, reproducible append-only
assembly and allowed inference imports, final-action-only history, geometry
and final-score deadlines, timeout/reset, grass islands and unknown input,
detached scenario copies, and three causal actual-camera differentials.
Before implementation, three tests failed against V2 because its batch model,
batch planner, and owner budget callback were absent. An independent read-only
audit also passed: exact source prefix, AST-exact exports, inherited method
identities, model/geometry post-call expiry checks, and four scalar observer
ticks after timeout with a deliberately failing batch sentinel. That audit
took 0.148 seconds and made no edits.

The fixed full-action profiles use only existing camera observations and
their past logged legal controls. They reproduce the same history separately
for V2 scalar and the new standalone. No truth, future controls, map or seed
initializes or updates either agent. The old clear-supported prefix is not a
new trajectory of either predictive candidate.

| Old camera action | Scalar V2 full `act` | Batch full `act` | Speedup | Feasible controls |
| --- | ---: | ---: | ---: | ---: |
| 20 | 2.06373 s | 0.90236 s | 2.29x | 56 |
| 40 | 1.37307 s | 0.74483 s | 1.84x | 26 |
| 60 | 1.65725 s | 0.80641 s | 2.06x | 40 |

All three selected predictive control. Their emitted commands, every scalar
plan field, sensor diagnostics and final observer state match. Timers include
parent perception, camera field, ridge, scenarios, model rollout, backup and
geometry, and final scalar observer advance. Prefix warmup and module loading
are excluded. The committed 378,971-byte fixture contains 61 old observations
and 60 preceding legal controls; it omits all truth/progress/contact labels.

These are single fixed measurements on a shared host, not a worst-case
latency certificate. Successful finite planning within the original budget
has the verified policy equivalence; deadline behavior can differ because
batching finishes sooner and the new guards cover geometry/final scoring.
The pure finite-callback assumption and eager-error behavior of the batch
helper remain explicit: an error on a scalar-unvisited row can cause fallback.
The uniform-asphalt predictor, sampled scenarios, camera support and
emergency trajectory provide no general grass/contact safety certificate.

`predictive-batch-integration-v1-lineage.json` pins sources, builder, tests,
camera/legal-action fixture, profile and this report. A zero-context append
patch and ignored source freeze preserve the exact V2 predecessor and child.
