# Project Context

## Current Objective And Status

Transitioned from frozen DrQ-v2 baseline to Stage 2: **DreamerV3** per `PLAN.md`.
DrQ-v2 baseline is frozen as benchmark (4~7/32 confirmation finishes).
Blind partition remains reserved and untouched.
Implemented native PyTorch DreamerV3 core, sequence replay, recurrent CPU export,
and executed feasibility gates 1-4 before any 131k matched training decision.

## DreamerV3 Feasibility Gates (Stage 2)

Per user instructions, DreamerV3 was audited through 4 progressive feasibility gates:

1. **Gate 1: Interface, Recurrent Reset/Carry, and CPU 2.1 Export (PASSED)**
   - Self-contained native PyTorch implementation in `dreamer_v3.py` and `agent.py`.
   - All 9 unit tests passed in both host and isolated Torch 2.1 CPU environments.
   - Recurrent inference latency on CPU: **0.69 ms** (far below 5.0 s ceiling).
   - Peak RSS: **330.1 MiB** (far below 1,024 MiB ceiling).
   - Packaging smoke passed (`init: 0.445s, act: 2.3ms, reset_matches_first: True`).
   - Bit-identical action trace determinism verified across independent episode resets.

2. **Gate 2: Short GPU Training Smoke (PASSED)**
   - 2,000 decisions, 1,501 updates completed on RTX 5070 Ti in 146.9 s (`runs/20260922-dreamerv3-gpu-smoke-2k/`).
   - Checkpoints saved at 1,000 and 2,000 steps; parity verified.
   - Both checkpoints exported to CPU actor and evaluated in isolated CPU 2.1 environment.

3. **Gate 3: World Model Learning Diagnosis (PASSED)**
   - Image reconstruction MSE: **0.0077 (2k) -> 0.0086 (5k)** (sharp visual dynamics).
   - Continuation / terminal prediction accuracy: **99.22%** (BCE loss 0.044).
   - Reward prediction: symlog MSE 0.056, MAE 0.135, reward correlation **+0.626**.
   - Categorical KL divergence: **0.0024** (stable latent transitions, no collapse).
   - Latent imagination: 15-step horizon rollouts completely finite, bounded, and stable.

4. **Gate 4: Small Pilot Policy Learning (FAILED)**
   - Pilot run: `runs/20260922-dreamerv3-pilot-10k/`.
   - Observation: policy progress collapsed from **0.077** (warmup random actions) to **0.019** (policy steps).
   - Action saturation: policy steering saturation fraction **0.954 (95.4%)**, hard-locked at -0.99999.
   - Screen evaluation at step 5,000: **0/6 finishes (0.0% finish rate)**, all episodes off-track at step 109.
   - **Failure Attribution:**
     - World Model learning: **Functioning properly** (MSE 0.0086, continue 99.2%, reward correlation +0.626).
     - CPU recurrent state handling: **Functioning properly** (0.69 ms latency, 330 MB RSS, exact determinism).
     - Algorithmic failure mechanism: Tanh-Gaussian continuous actor optimization collapsed into premature tanh saturation (steering -0.99999) under dynamics backpropagation, squashing action variance to zero. The car immediately leaves the track in 11 steps, filling the sequence replay with 95%+ negative-reward grass driving. Imagination from grass states predicts only negative returns, trapping the policy in a degenerate saturation loop.

**Verdict:** Feasibility gate **FAILED** at Gate 4. Do NOT launch the 131,072-decision matched training.
The frozen DrQ-v2 control baseline remains the reigning champion benchmark.
Details: `experiments/dreamerv3-feasibility-gate.json`.

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

## Completed Final Controlled Follow-Up: Padding Study

Protocol: `experiments/drqv2-augmentation-pad-v1.json`. All four runs completed in
`runs/20260922-drq-augmentation-pad-v1-restart/`, frozen source `a28ef02`:
control (pad 4) versus treatment (pad 1), `steering_logit_l2=0` in BOTH arms,
training seeds 0/1, **131,072 decisions and 121,073 updates each**, replay 100,000,
batch 64, warmup 10,000, sampler 917, tracks 1--4. Selection at 65,536/131,072.

| Arm / Seed | Screen Finishes | Fresh Confirmation Finishes | Confirmation Progress |
| --- | --- | --- | --- |
| control (pad 4) / 0 | 7/24 | 4/32 | 0.633857 |
| control (pad 4) / 1 | 5/24 | 7/32 | 0.693897 |
| pad 1 / 0 | 0/24 | 0/32, diagnostic-only | 0.340640 |
| pad 1 / 1 | 0/24 | 0/32, diagnostic-only | 0.340333 |

**Reject augmentation padding=1:** paired finish deltas are **-4 and -7**.
Both treatment seeds collapsed to 0 finishes across all evaluation cells.
Under the current setup, pad=1 performed substantially worse than pad=4.
While this outcome is consistent with reduced visual regularization or generalization degradation under this configuration,
the exact causal mechanism remains an interpretive hypothesis rather than direct proof (pad=2, 3 등 다른 강도를 전면 탐색하지 않았으므로 pad=4만이 필수라고 단정하지 않고, 현재 설정에서 pad=1이 pad=4 대비 크게 열등했음을 확인).

- Screen: IDs 101--103/seeds 31001--31008. Confirmation: IDs 311--314/seeds 33101--33108,
  now consumed. Blind IDs 321--323/seeds 33201--33208 remains untouched/reserved.
- Exact paired traces, zero operational failures across all 640 CPU executions;
  max action 2.855 ms, initialization 1.196 s, whole-worker RSS 361.5 MiB.
- Full result: `experiments/drqv2-augmentation-pad-v1-result.json`.
  Diagnostics: `experiments/drqv2-augmentation-pad-v1-diagnostics.json`.
  Execution history: `experiments/drqv2-pad-execution.json`.

## Stop Rule And Synthesis

Per user instructions ("1~2개의 근거 있는 후속 실험에서도 개선이 재현되지 않으면 DrQ-v2 추가 튜닝을 중단해라")
and protocol stop rules (attempt 2 of 2):

1. **Both authorized single-variable follow-ups failed to improve completion:**
   - Attempt 1 (steering-logit L2=0.001): confirmation finish deltas **-3 and -7**.
   - Attempt 2 (augmentation_pad=1): confirmation finish deltas **-4 and -7**.
2. **Further DrQ-v2 hyperparameter tuning is STOPPED.** No third controlled axis,
   coefficient sweep, padding search, reward/frame-skip/architecture bundle, or new algorithm.
3. **Control baseline reproducibility:** Across four independent control training runs
   (two in the L2 study, two in the padding study), standard DrQ-v2 (pad=4, raw reward,
   no L2) consistently achieves 4 to 7 finishes out of 32 on fresh confirmation
   (12.5% to 21.9% finish rate, mean progress 0.63 to 0.74). The completion signal
   is real and reproducible across seeds, but narrow modifications (L2, pad 1) consistently
   harm it.
4. No blind cells were consumed; no submission was released; no promotion was granted.

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
