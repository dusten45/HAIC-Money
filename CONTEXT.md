# Project Context

## Current Objective

Develop a submission-capable DrQ-v2 candidate, not a PPO improvement project.
The current user request authorizes implementation of a separate, permanent
demonstration replay for DrQ-v2. This implementation-only task must not train a
candidate or run any performance evaluation; model quality remains the user's
decision. The running padding study and its frozen artifacts remain untouched.

## Frozen Contract

- Do not modify `core/`, `env_wrapper.py`, or `damage.py` for experiments.
- Four `84x84` grayscale frames, CHW float32 `[0,1]`; official continuous action
  `[steer,gas,brake]` in `[-1,0,0]..[1,1,1]`. No fifth plane or smoothing.
- Raw official reward, no normalization, shaping, finish bonus or collision penalty.
- One decision per environment step; frame skip 4 only inside the environment.
  Pure time limits bootstrap; finish/crash/off-track endpoints do not.
- Finish requires 95% progress AND a valid forward finish crossing. Geometry is
  seed-driven; `track_id` changes obstacles. Reserve geometry seeds across ALL IDs.
- Exported CPU actor results are authoritative. Rank finish rate, then progress,
  then completed lap time. Freeze actor hashes before confirmation; never select
  on confirmation/blind. Diagnostic receipts cannot promote or unlock blind.
- CPU gate: Python 3.11, Torch 2.1 CPU, NumPy 1.26; init10s, reset/action5s,
  process1,024MB, ZIP500MB. Participant reference commit
  `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`. No trainer/SB3/prohibited imports in
  submission inference. Official submission-container execution remains separate.

## Completed L2 Study

Protocol: `experiments/drqv2-steering-logit-v1.json`. All four runs completed in
`runs/20260922-drq-steering-l2-v1-fast/`, frozen source `8e5fa46`:
control/L2 coefficients0/0.001, training seeds0/1, **131,072 decisions and121,073
updates each**, replay100,000, batch64, warmup10,000, sampler917, tracks1--4.
Every run selected its131,072 checkpoint over65,536 using the same CPU screen.

| Arm / Seed | Screen Finishes | Fresh Confirmation Finishes | Confirmation Progress |
| --- | --- | --- | --- |
| control / 0 | 7/24 | 6/32 | 0.735757 |
| control / 1 | 5/24 | 7/32 | 0.638823 |
| L2 / 0 | 3/24 | 3/32 | 0.519913 |
| L2 / 1 | 0/24 | 0/32, diagnostic-only | 0.422361 |

**Reject L2 at0.001:** paired finish deltas are **-3 and-7**. No promotion,
scale-up, submission or blind evaluation. Do not retrospectively nominate a
control for blind after seeing confirmation. Controls themselves reproduce
nonzero completion across both training seeds; this is not DrQ family rejection.

- Screen: IDs101--103/seeds31001--31008. Confirmation: IDs211--214/seeds32101--32108,
  now consumed. Blind IDs221--223/seeds32201--32208 remains untouched/reserved.
- Independent final audit verified all8 checkpoints, matching episode-index
  streams/exclusions, frozen actor/config/source lineage, and all640 CPU executions.
  Exact paired traces, zero operational failures; max action4.064ms,
  initialization1.218s, whole-worker RSS344.5625MiB.
- Those640 executions represent320 canonical actor/checkpoint-cell observations,
  56 distinct cells and16 geometry seeds, NOT640 independent trials.
- Full result: `experiments/drqv2-steering-logit-v1-result.json`.
  Mechanics: `experiments/drqv2-steering-logit-v1-diagnostics.json`.
  Operational history: `experiments/drqv2-l2-execution.json`.

## Final Controlled Follow-Up

**RUNNING:** augmentation padding **4 versus1**,
`steering_logit_l2=0` in BOTH arms. No combined repair. This is attempt2 of2.
Protocol: `experiments/drqv2-augmentation-pad-v1.json`; explicit root
`runs/20260922-drq-augmentation-pad-v1/`, frozen source `a28ef02`. All four arm/seed combinations run
from scratch with the same131,072 budget and all other settings unchanged.
Operator state is in `experiments/drqv2-pad-execution.json`. All222 related tests,
CPU21 preflight, four-job GPU/CPU smoke and idempotent recovery passed before launch.

- Reuse only the consumed development screen101--103/31001--31008.
- Fresh confirmation311--314/33101--33108; fresh blind321--323/33201--33208,
  two CPU reloads each. Reserve180 seeds including every previous reservation.
- Keep strict positive confirmation finish gain in BOTH seeds, nonzero treatment
  screen/confirmation and all CPU gates. Finalist fixed by screen before confirmation.
- If this final trial does not reproduce improvement, **stop further DrQ tuning**.
  No third axis, coefficient search, reward/frame-skip/architecture bundle, or new
  algorithm. Positive evidence permits only a separately declared next stage.

Why this one axis: two2,048-state control replay pools, each shared across all four
actors, show current controls' pad4 view-pair steering sign disagreement at
9.03--9.33%, versus3.32--3.71% for pad1. L2 removes exactly-zero squash derivatives
in both seeds, but seed1 becomes MORE saturated and both seeds lose completion.
The penalty also changes longitudinal policy on identical inputs. History is used;
no new step/terminal bug was found. These are mechanisms/associations, not causes.
Smaller test shifts naturally change actions less; **only matched fresh completion
can accept weaker training augmentation**. L2 seed0 already reduced shift sensitivity
while losing finishes, so lower sensitivity or saturation alone is not success.

Additional observations: critic gradients are heavily clipped yet finite, with
Adam confounding naive scale interpretations; negative-reward tails dominate replay,
but L2 seed0 had MORE finish support than control0. Finishing policies also oscillate.
Do not attribute failure simply to insufficient finish data, gas/brake overlap,
noise decay, or steering sign changes. Pre-study diagnostics are preserved in
`experiments/drqv2-pre-l2-diagnostics.json`.

## Demonstration Replay Implementation

Implemented, not trained or performance-evaluated. A training-only corridor
teacher artifact stores raw-reward, terminal-safe DrQ transitions with official
three-dimensional actions converted once to native symmetric coordinates. DrQ
keeps this artifact in a replay separate from online experience and draws an
exact fixed number of demonstration rows per learner batch. The demonstration
replay, sampling RNG and source lineage are checkpointed; exported actors remain
unchanged and contain no teacher or replay. Correctness coverage uses only fake
environments and synthetic transitions.

## Local Visual Diagnostics

The local simulator can run an `agent.py` against a reproducible map with camera
frames enabled. `local_simulator.diagnostic` writes the run log, a standalone
HTML replay with synchronized camera/action/trajectory telemetry, and optionally
an annotated MP4. The report's saturation, gas/brake overlap, and collision
signals are observational flags only; no performance evaluation is performed.
The Track Lab web server now exposes a local Agent catalog under
`<artifact-root>/agents`, lets the browser choose a ready model, and passes that
selection to the automatic run endpoint. Map files, official seeds, and generated
custom tracks remain selectable in the same UI.

## Forward-Stability Runtime Repair

The legacy bare `model.pt` baseline now uses an inline forward-corridor safety
controller at inference time. It estimates the visible road centerline and asphalt
spans, rate-limits steering, snaps near-centered straight segments to zero steering,
regulates speed conservatively from the HUD (cruise target 48, throttle cap 0.08),
and emits mutually exclusive throttle or brake.
Because the official RGB input is converted to grayscale, green background and orange
obstacles are treated as the same bright non-road hazard. Narrowing or disappearing
near-road spans, small edge clearance, and bright pixels inside the road corridor all
cut throttle; the controller brakes when the corridor is blocked and biases only
toward the visible road. Compact bright objects still trigger bounded avoidance, but
never override the road-width safety gate. After the road has been seen, a temporary
visual dropout brakes and decays steering instead of falling back to an unconstrained
turn. Opposite steering must first pass through zero, and a one-step steering change
is limited to 0.07. No simulation or performance evaluation was run for this repair.
Explicit action-contract payloads, DrQ actors, and the HAIC visual-policy runtime
keep their recorded model paths; no simulation or performance evaluation was run
for this repair.

## Runtime And Infrastructure

- Preserve `.venv` -> `/venv/main`: Python3.11.14, Torch2.11.0+cu128, RTX5070Ti.
  Actual GPU learning works. Never lock-sync back to Torch2.1 or redesign the image.
- CPU gate interpreter: `/tmp/kilo/haic-cpu21/bin/python`, Torch2.1.0+cpu,
  NumPy1.26.0, Gymnasium0.29.1, OpenCV4.8.1.78. Preserve its symlink path when
  invoking it: resolving to the base executable loses venv isolation.
- `train_drqv2.py`: explicit run/checkpoint paths, CPU selection, replay/optimizer/
  CPU+CUDA RNG, sampler/warmup state, exact partial-episode reconstruction, incumbent
  preservation, source/runtime/protocol validation. No `_latest` dependency.
- `run_drqv2_matched.py`: executes immutable source/command snapshots, verifies
  budgets and episode prefixes, freezes actors, validates sealed CPU receipts,
  and supports non-mutating evaluation recovery. No retuning during a study.
- `diagnose_drqv2.py`: CPU21-only shared-replay offline actor comparisons, no env
  cells or optimizer updates. Raw large artifacts remain local; check existence
  after moving servers. Compact protocols/results and code are committed.
- Initial slow L2 run was stopped at last-logged22k per arm, before ANY checkpoint
  or evaluation. A CUDA-only gather preserves all scalar RNG draws/output and
  tested learner state, removing128 per-view host synchronizations. All arms
  restarted equally; live learning metrics matched through17k, throughput about3x.
  Sealed restart receipt and prefix proof are in the run roots/execution JSON.
- CPU/GPU restore/export tests, batch64 smokes, full four-job harness and idempotent
  recovery passed. RNG parity covers288 augmentation cases and successive updates.

## Earlier Evidence And Caveats

- Historical actor `runs/20260921-043514_drqv2-pilot-131072/actor.pt` recorded4/24,
  progress0.807, damage0.45. Exact historical cells/source provenance are incomplete.
  Its subsequent frozen CPU21 screen was0/24 and diagnostic confirmation4/32 on
  both reloads; no promotion. Details: `experiments/drqv2-promotion-v1-result.json`.
  Old confirmation31101--31108 is consumed; blind31201--31208 remains untouched.
- This is a native DrQ-v2 variant, not an exact author implementation: encoder
  sharing/strides and actor/augmentation details differ. These stayed fixed in L2.
- Sampled PPO at1,048,576 decisions failed0/24 development,0/32 confirmation,0/24
  blind; curriculum, collision, EMA/Markov and action-codebook variants did not
  establish completion. They remain historical, not active alternatives.
- Historical continuous PPO uses5 channels, EMA, shaping/normalization and a
  different effective sampler. It is budget evidence, not a matched algorithm-only
  control. The archived official submission is still an old PPO baseline.
- Concurrent upstream `b1ad528` introduced separate policy/planner/site work,
  preserved via merge9649c62. It is not used/trained here; official environment
  sources and the functioning GPU stack remain unchanged.
- Known unrelated merged-suite failure: `tests/test_submission_layout.py` reads
  inherited `ru_maxrss` in `training/package_submission.py` after CUDA-heavy tests;
  it passes alone. DrQ uses process-local `/proc/self/status` and passes its gates.
