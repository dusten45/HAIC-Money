# DreamerV3 실패 진단 기반 발전 전략

## Active Plan Status

| Field | Current plan |
|---|---|
| Status | A1-A3 correctness paths and B1 tools are implemented. Nine source-pinned, B1-only pretraining iterations failed their frozen gates; a post-hoc failure diagnosis found a v9 wrong-class loss and B1 comparator/metric flaws. No policy screen or later partition was opened. |
| Goal | Repair the identified DreamerV3 fidelity defects, then determine whether a faithful implementation can pass feasibility before matched training. |
| Rationale | The current formulation failed the small policy-learning gate, while its prior Gate 3 interpretation was withdrawn as insufficient world-model validation. |
| Evidence | `experiments/dreamerv3-feasibility-gate.json`, `experiments/dreamerv3-b1-iteration-summary-v1-v9.json`, and `experiments/dreamerv3-b1-failure-diagnosis-v1.json`; see `docs/experiments/INDEX.md`. |
| Hypothesis | Correct transition alignment, actor/critic objectives, action distribution, replay/pretraining, and world-model gates can remove implementation-induced collapse. |
| Scope | Correctness repair and feasibility only. No 131,072-decision matched run, official submission, or confirmation follows automatically. |
| Steps | Test Phase A repairs, pass Phase B1 open-loop, run a diagnostic Phase C 10k pilot, pass Phase B2 counterfactual reproducibility before any larger treatment, then seek two-seed completion at Phase D. |
| Evaluation | Frozen study-specific contract, CPU recurrent/export checks, B1 held-out-development prediction, B2 action ranking only after prefix replay parity, C stability, and D repeated screen completion. |
| Acceptance | Phase A, B1, B2, and C operational gates pass; Phase D records nonzero screen completion for both learner seeds without operational failure. This is feasibility, not proof of improvement over DrQ-v2. |
| Rejection | A faithful repair still fails the renewed stability/completion gates or violates CPU/package constraints. Record failure or inconclusive tooling separately instead of scaling it. |
| Stop condition | Do not start matched scale-up until all preceding gates pass. Official actions require their separate workflows and explicit user authorization. |

The detailed diagnosis and phased technical work below are retained from the
pre-migration recovery strategy.

The dated [validation and implementation specification](#2026-09-24-검증-및-구현-명세)
at the end of this document resolves earlier overclaims and adds the missing data,
state, and evaluation contracts. Read it before implementing the phases below.

### Implementation Progress (2026-09-24)

- Implemented in `dreamer_v3.py`: replay batches contain `T` transitions and
  `T+1` observations. Ordinary successor frames reuse the next replay row;
  terminal/truncation result frames are retained sparsely by monotonic sequence
  ID, removed on ring overwrite, and serialized in replay checkpoints. Reward
  and continuation heads are trained on the posterior state reached after each
  action. Legacy replay state dictionaries lacking a boundary frame load, but
  any sequence that needs the missing frame fails closed rather than using a
  reset observation. Scalar-head full checkpoint v1/v2 and full-stack decoder
  checkpoints are rejected by the residual-frame checkpoint v3 learner.
- Added fixed-length same-episode burn-in to replay sampling and RSSM rollout.
  Burn-in gradients are stopped; only the following configured transitions
  contribute reconstruction, reward, continue, and KL losses.
- Implemented in `dreamer_v3.py` and `train_dreamerv3.py`: externally selected
  warmup actions advance online posterior state, the executed native action
  replaces the proposed action in recurrent carry, policy observations are not
  processed twice, and checkpoint verification restores live recurrent carry
  and Python/NumPy/Torch RNG state.
- Implemented the A3 learner contract: bounded-normal sampling around
  `tanh(mean)` with detached-sample score log-prob, analytic entropy, clipped
  executed actions in imagination, two-hot raw reward/value heads, separate
  dynamics/representation free-nat floors, cumulative continuation weights,
  percentile-scaled advantages, imagination slow-critic regularization, and a
  replay λ-return value anchor. Adam replaces default-decay AdamW. The
  residual-frame decoder predicts the newest image change rather than all four
  stack channels; overshooting prior KL and its free-floor are explicit controls.
  Return percentiles are checkpointed in the current learner format.
- Updated `scripts/diagnose/dreamerv3.py` to use the same `T+1` posterior rollout
  and score reward/terminal on result states. Its legacy `diagnostics_passed`
  field remains a basic finite/reconstruction smoke signal, not the B1 gate.
- Added `agent.update(model_only=True)` and `--replay-pretrain-updates`; a
  `--skip-screen` run is allowed only when policy actions are not reached and
  still saves/reloads the CPU actor without calling the screen evaluator.
- Added a protocol-checked random development collector and B1 prior-only
  open-loop diagnostic. Development cells must be exact, unique, source-pinned,
  disjoint from evaluator partitions, and included in
  `reserved_training_seeds`. The gate reports episode-cluster bootstrap image
  errors, cumulative reward rank, and terminal metrics against trivial baselines.
- Added marker, terminal/truncation, ring overwrite, burn-in-gradient,
  bounded-normal score-gradient, two-hot, free-nat, terminal-weight,
  warmup-handoff, checkpoint-isolation, pretrain-only, source-hash, and
  development-isolation, overshooting, and residual-frame tests. The existing
  frozen scalar-head/stack-decoder pilot checkpoints cannot load in this learner.
- The full selected regression command currently passes **139 tests** across
  Dreamer, trainer/protocol, B1 tooling, inference contracts, and map baselines.
- The bounded first study is frozen in
  `experiments/dreamerv3-b1-world-model-local-v1.json`: two learner seeds,
  1,024 random decisions, 100 replay-only model updates, 32 training-excluded
  development geometries, separate screen/confirmation/blind partitions, and
  source hashes. No seed from its evaluation partitions is opened in this stage.
- The protocol v1 and v2 pretraining-only runs completed as planned, but their
  B1 gates failed; the v3-v9 treatments are follow-up development iterations.
  Policy training remains blocked by B1. B2 prefix reproducibility, CPU screen,
  confirmation, blind, custom-map integration, and all performance claims
  remain open.
- The follow-up B1 iterations v1-v9 are summarized in
  [`experiments/dreamerv3-b1-iteration-summary-v1-v9.json`](../../../experiments/dreamerv3-b1-iteration-summary-v1-v9.json).
  More random data alone and terminal-balanced sampling did not fix prior
  dynamics; v8's residual decoder produced a small replicated image-baseline
  gain but terminal/reward gates still failed. Frozen v9 used `pos_weight=56`
  on `continue=1`, not rare terminal events: the old phrase "positive terminal
  weight" was incorrect. Its worsened BCE is an observation, not evidence that
  terminal-positive weighting was tested or causally rejected. The B1 scorer
  also compares sampled terminal windows against a natural-training prevalence
  constant and ranks episode-average returns rather than fixed-state actions.
  The [`B1 diagnosis`](../../../experiments/dreamerv3-b1-failure-diagnosis-v1.json)
  records an actual four-decision transition trace and same-anchor 5/8/10/full
  context tests. Results remain local and do not select a policy.

## 핵심 결론

관측된 종전 실패는 **복구 전 DreamerV3 구현의 파일럿 실패**였습니다. A1-A3 수리 후에도 이번 B1 세계모델 게이트는 아홉 처리에서 통과하지 못했고, 새 정책은 아직 학습하지 않았습니다. 종전 파일럿 결함이 단독 원인이었다는 증거도 없습니다.

관측된 steering saturation은 사실이지만, 원인을 단순히 `tanh`로 돌릴 수 없습니다. 현재 구현에는 이를 유발하거나 악화할 수 있는 여러 fidelity defect가 있습니다.

따라서 먼저 검증할 가치가 높은 경로는 다음입니다. 뒤의 데이터·불확실성 처리는 성능이 입증된 구성이 아니라 조건부 가설입니다.

> **충실한 DreamerV3 복원 → prefill 후 learner pretraining → episode-balanced replay → training-only DrQ teacher 데이터로 world model과 actor를 warm-start → 필요할 때만 uncertainty-aware imagination**

## 핵심 결함

| 사전 복구 결함 | 2026-09-24 구현 상태 | 남은 판정/위험 |
|---|---|---|
| Actor가 symlog critic 출력 평균을 직접 최대화 | stopped advantage `log_prob` score loss로 교체 | 폐회로 주행 이득은 아직 미검증 |
| Reward/continue target의 행동 시점이 한 칸 어긋남 | 결과 관측 상태에 reward/terminal 정렬 | B1 open-loop 예측 성능은 미검증 |
| Entropy objective 누락 | `actor_entropy_coeff=3e-4` 사용; std entropy gradient 단위시험 추가 | action 분포와 완주 효과는 미검증 |
| Raw return percentile과 symlog critic 값 혼합 | Raw-domain two-hot reward/value와 percentile-scaled advantage | 향후 두-hot 출력 및 재적재 parity 유지 필요 |
| `tanh(mu + sigma*eps)` 분포 | `Normal(tanh(mean), bounded_std)`로 교체; 원표본 log-prob와 실행 action 분리 | 표본은 무한 범위이며 환경 clip이 필요; clipped control 성능 미검증 |
| KL free-nat를 계산만 하고 미적용 | dynamics/representation KL 각각 free-nat clamp; 기울기 시험 추가 | 학습에서 collapse가 해소되는지는 미검증 |
| Scalar MSE reward/value heads | 255-bin symlog-spaced two-hot heads 및 finite loss 시험 | 기존 scalar-head checkpoint v1은 새 learner와 비호환 |
| Imagination continuation weighting 누락 | Actor·imagination critic에 cumulative discount/continue weight 적용 | 폐회로 예측 타당성은 B1/B2 대상 |
| Critic slow regularization/replay anchor 누락 | 두-hot slow-value regularizer와 replay λ-return critic loss 추가 | replay data 분포와 scale은 개발 gate 대상 |
| Mid-episode sequence zero initialization | 같은 episode prefix burn-in; prefix gradient 차단 시험 | 8-step 근사 문맥이며 전체 과거 hidden state 복원은 아님 |
| Random prefill 뒤 즉시 정책 제어 | bounded-normal learner-only pretraining 및 `policy_start_step` 구현 | 100-update 효과는 아직 실행·검증되지 않음 |
| Reconstruction loss scale이 공식 구현보다 작을 가능성 | 수치 조정은 하지 않음 | baseline open-loop와 gradient scale을 새 실행에서 측정해야 함 |

공식 DreamerV3 actor objective는 다음 형태입니다.

\[
L_\pi =
-\sum_t w_t
\left[
\operatorname{sg}
\left(
\frac{R^\lambda_t-V(s_t)}
{\max(1,S)}
\right)\log\pi(a_t|s_t)
+\eta H[\pi(\cdot|s_t)]
\right]
\]

복구 전 구현의 `-mean(critic(imagined_state))`는 최종 논문 방식도, 2023년 초기 코드 방식도 아니었습니다. 현재 작업트리의 A3 손실은 이를 score-function 목적함수로 대체했지만, 실행 결과가 개선되었음을 뜻하지는 않습니다.

## Gate 재해석

기존 Gate 3의 “world model 정상” 판정은 철회하는 것이 맞습니다.

- Reconstruction MSE는 posterior teacher-forced reconstruction일 뿐 open-loop dynamics 정확도가 아닙니다.
- Continue accuracy `99.21875%`는 512개 중 terminal이 4개라면 항상 continue만 예측해도 나오는 값입니다.
- Continue BCE `0.04437`도 **terminal이 실제로 512개 중 4개일 경우** constant-prevalence baseline 약 `0.0457`보다 거의 낫지 않습니다. 진단에서 양성 개수는 별도로 계수하지 않았습니다.
- Reward correlation은 action-reward 시점이 어긋난 상태에서도 road/grass 시각 상관 때문에 높게 나올 수 있습니다.
- KL `0.0024`는 free-nat bug 아래에서는 latent collapse 경고입니다.
- 15-step imagination이 finite하다는 것은 NaN이 없다는 뜻일 뿐입니다.

따라서:

- Gate 1 CPU recurrent/export: 유효
- Gate 2 GPU 실행: 유효
- Gate 3 world-model learning: **미입증**
- Gate 4 pilot collapse: 해당 구현의 경험적 실패는 유효
- Faithful DreamerV3 feasibility: 아직 미검증

## 전략 1: Fidelity 복원

이 작업은 hyperparameter 실험이 아니라 correctness repair로 한 번에 처리해야 합니다.

1. Transition convention을 `state-action-resulting observation/reward` 기준으로 완전히 정렬합니다.
2. Final DreamerV3의 stopped normalized advantage + REINFORCE objective를 사용합니다.
3. `log π(a|s)`와 entropy `3e-4`를 실제 actor loss에 적용합니다. Score-function 항에서는 sampling 경로를 끊습니다.
4. Actor와 critic loss에 cumulative discount/continue weight를 적용합니다.
5. Action distribution을 공식 `bounded_normal`로 바꿉니다.
6. Actor output layer를 `outscale=0.01`로 초기화합니다.
7. Reward/value를 255-bin symexp two-hot distribution으로 변경합니다.
8. Reward/value output layer를 zero initialization합니다.
9. KL dynamics/representation 항에 free-nat 1을 각각 적용합니다.
10. KL 가중치는 공식 `1.0/0.1`을 기준으로 사용합니다.
11. Current critic으로 λ-return을 계산하고 slow critic regularization을 추가합니다.
12. Replay value loss로 critic을 실제 transition return에 고정합니다.
13. Implement fixed 8-decision same-episode burn-in with stopped prefix gradients; a full persisted recurrent state remains unnecessary and unsupported.
14. Replaced default-decay AdamW with Adam (`weight_decay=0`) for world model, actor, and critic.

Action 관련 직접 권장 설정:

- `mean = tanh(raw_mean)`
- action-space Gaussian noise
- `std ∈ [0.1,1.0]`
- 환경 경계에서 단 한 번 clip; 모델의 이전 행동과 replay에는 실제 실행 행동을 동일 좌표로 기록
- evaluation은 deterministic mode
- per-axis mean/std/entropy/boundary mass 기록

현재 `tanh(mu + sigma*eps)`를 유지하려면 최소한 SAC 방식의 정확한 transformed log-probability와 Jacobian correction이 필요합니다. 하지만 공식 `bounded_normal`을 먼저 검증하는 것이 더 적절합니다.

참고:

- [DreamerV3 논문](https://arxiv.org/html/2301.04104v2)
- [공식 DreamerV3 actor/critic](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py)
- [공식 DreamerV3 config](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml)
- [공식 bounded-normal head](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/jax/heads.py)
- [SAC entropy 및 squashed Gaussian](https://arxiv.org/abs/1812.05905)

## 전략 2: Prefill 후 Pretraining

현재처럼 1,000 random steps 직후 actor에게 운전을 맡기면 안 됩니다.

권장 순서:

1. 1,000 decisions random prefill
2. 환경 제어 없이 replay-only learner update 100~500회
3. World model open-loop gate 통과
4. Actor entropy와 action distribution gate 통과
5. 이후 actor takeover

첫 비교에서 여전히 collapse하면 5,000-decision prefill을 단일 treatment로 시험합니다.

DreamerV2 공식 driver도 replay를 먼저 채우고 learner pretraining 후 환경 제어를 시작합니다.

- [DreamerV2 driver](https://github.com/danijar/dreamerv2/blob/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/train.py#L102-L130)
- [DreamerV2 config](https://github.com/danijar/dreamerv2/blob/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/configs.yaml)

## 전략 3: DrQ Teacher Bootstrap

도로 주행 데이터를 제공할 수 있는 별도 가설적 처리입니다. 교사 데이터가 완주율을 높이는지는 정합성 복구 후 통제된 실험으로 확인해야 합니다.

1. Frozen DrQ actor로 training IDs `1–4`에서 fresh trajectory를 수집합니다.
2. Screen/confirmation/blind seed는 모두 제외합니다.
3. Pixel, 실제 applied action과 이를 역변환한 executed native action, raw reward, 결과 관측, terminal/truncation을 함께 기록합니다. Dreamer replay/BC target에는 executed native action을 사용합니다.
4. Dreamer world model을 teacher sequence로 사전학습합니다.
5. Actor는 teacher action likelihood로 짧게 behavior cloning합니다.
6. BC coefficient는 actor takeover 후 점진적으로 제거합니다.
7. Replay batch는 teacher/online을 예를 들어 50/50으로 구성합니다.
8. 최종 제출 actor에는 DrQ가 포함되지 않습니다.

이 전략은 Dreamer가 첫 정책 단계부터 도로 상태를 경험하고, DrQ가 이미 발견한 완주 가능 주행 분포를 world model에 제공합니다.

공정성상 결과는 `DreamerV3 + DrQ bootstrap`으로 별도 표기해야 합니다. 순수 Dreamer와 동일한 것으로 보고하면 안 됩니다.

근거:

- [DQfD](https://arxiv.org/abs/1704.03732)
- [DDPG from Demonstrations](https://arxiv.org/abs/1707.08817)
- [AWAC](https://arxiv.org/abs/2006.09359)
- [RLPD](https://arxiv.org/abs/2302.02948)

## 전략 4: Episode-Balanced Replay

현재 uniform-step replay는 긴 실패 episode를 상대적으로 자주 뽑습니다. 그것이 학습 실패의 원인인지는 아직 검증되지 않았습니다.

권장 구성:

- 50% uniform-step
- 50% uniform-episode
- warmup/teacher episode 보호
- success/high-progress stratum은 최대 25~50%
- 어떤 특수 stratum도 batch 전체를 독점하지 않음
- world model은 uniform component를 항상 유지

DreamerV2도 episode를 먼저 고른 뒤 sequence를 샘플링합니다.

이 방식은 109-step grass failure가 반복될 때 grass frame 개수가 곧 학습 확률이 되는 문제를 줄입니다.

## 전략 5: World Model 신뢰성 Gate

다음 지표를 Gate 3의 새 기준으로 사용해야 합니다.

### Open-loop prediction

- 5~10 posterior context steps
- recorded action으로 prior만 rollout
- horizon `1,3,5,10,15`
- observation, reward, continue, future posterior latent를 비교

비교 baseline:

- repeat-last-frame
- mean image
- constant reward
- constant continue
- deterministic-h-only RSSM

권장 합격 기준:

- 모든 horizon에서 trivial baseline보다 우수
- 5-step cumulative reward rank correlation ≥ 0.5
- terminal recall ≥ 0.8
- balanced accuracy ≥ 0.8
- terminal PR-AUC가 prevalence의 최소 5배
- constant-prevalence BCE보다 최소 10% 개선
- \(z\) shuffle/ablation 시 성능이 유의하게 저하
- imagined action ranking과 simulator short branch ranking이 일치

### Counterfactual action test

같은 동적 상태에서 여러 steer/gas/brake action을 실제 환경과 world model에서 분기합니다. 현재 로컬 실행기에는 완전한 차량 상태 clone/restore가 없으므로, 먼저 동일 reset과 실행 행동 prefix를 재생해 상태 재현성을 검증해야 합니다. 재현성이 없으면 이 비교는 미실행/미입증으로 기록합니다.

- reward ordering
- terminal risk ordering
- progress ordering
- 1/5/15-step cumulative reward

을 비교합니다.

이 테스트가 통과하지 않으면 actor가 extreme action을 선택해도 world-model exploitation인지 실제 개선인지 구별할 수 없습니다.

## 전략 6: Conservative Imagination

Faithful Dreamer에서도 actor가 model error를 악용한다면 latent ensemble disagreement를 사용합니다.

\[
\tilde r_t = \hat r_t - \beta U(s_t,a_t)
\]

- penalty는 imagination actor loss에서만 사용
- replay reward는 raw reward 유지
- evaluation/export actor에는 ensemble 불필요
- 먼저 disagreement와 실제 prediction error의 상관을 검증

관련 연구:

- [MOPO](https://arxiv.org/abs/2005.13239)
- [MBPO](https://arxiv.org/abs/1906.08253)
- [LOMPO](https://arxiv.org/abs/2012.11547)

모델 오차가 큰 초기에는 imagination horizon을 `3 → 5 → 15`로 늘리는 방식도 유효합니다.

## 전략 7: Plan2Explore

별도의 exploration actor가 latent ensemble disagreement를 최대화하도록 합니다.

HAIC 적용 규칙:

- Task actor와 explorer를 분리
- Episode 경계에서만 전환
- Stored reward는 항상 raw reward
- Task actor objective에 intrinsic reward를 섞지 않음
- Evaluation/export에는 explorer를 포함하지 않음
- 2k random calibration 후 3k explorer pilot

주의점은 grass와 collision이 높은 novelty를 보일 수 있다는 점입니다. 따라서 DrQ teacher bootstrap과 episode-balanced replay보다 우선순위가 낮습니다.

- [Plan2Explore](https://arxiv.org/abs/2005.05960)

## 전략 8: 보조 예측 Head

Raw reward를 변경하지 않고 다음 training-only auxiliary target을 추가할 수 있습니다.

- progress delta
- tile acquisition
- off-track/crash/finish 다중분류
- time-to-terminal
- road/grass 상태
- action-conditioned future progress ordering

이는 representation에 completion-relevant 정보를 남기지만 actor reward를 직접 shaping하지 않습니다.

다만 환경 `info`를 사용하므로 privileged training signal로 명시해야 하며, fidelity baseline 이후 별도 treatment로 평가해야 합니다.

## 실행 로드맵

### Phase A: Correctness

환경 interaction 전에 다음을 전부 검증합니다.

- Synthetic action/reward marker의 timestep alignment 오류 0건
- Free-nat 이하 KL gradient 0
- Actor loss가 λ-return advantage와 log-prob에 실제 의존
- Analytic Normal entropy gradient가 std head에 전달; mean head의 학습 신호는 detached-sample advantage/log-prob 항에서 확인
- Terminal 이후 imagination weight 0
- Episode 끝의 결과 관측 보존, 실행된 native action과 replay/online recurrent carry의 일치
- Checkpoint 검증 전후 살아 있는 학습 agent의 carry/다음 행동 일치
- Two-hot encode/decode 왕복
- Bounded mean 포화 상태에서도 action variance 유지
- CPU recurrent reset/carry parity 유지

### Phase B: World Model

1k prefill + learner pretraining을 수행한 뒤 훈련과 분리된 **훈련 범위 내 개발 자료**의 open-loop gate를 통과해야 합니다. 확인 또는 blind 셀을 이 게이트에 사용하지 않습니다.

Posterior reconstruction MSE만으로 통과시키지 않습니다.
Open-loop를 B1, 동일 동적 상태의 행동 분기를 B2로 구분합니다. B1 통과
후 B2 도구가 아직 재현성을 입증하지 못해도 10k **진단 목적** 파일럿은
실행할 수 있지만 B2가 통과하지 못한 상태에서 교사·replay 규모 확대,
Phase D 또는 131k로 진행하지 않습니다. 이때 B2는 모델 실패가 아닌
`inconclusive: branch tooling`입니다.

### Phase C: 10k Pilot

- 두 training seed 사용
- 동일 off-track-at-109 trace 반복 금지
- steering `|a|≥0.99` 비율을 잠정 50% 미만으로 제한
- action-space std/entropy가 0으로 붕괴하지 않음
- 같은 사전등록 개발 조건에서 수집한 별도 대조 자료 대비 policy progress 기록(보조 지표); 기존 random warmup `0.077`은 비매칭이므로 합격 기준으로 사용하지 않음
- 두 seed 모두 finite/stable
- CPU export/reset gate 유지

### Phase D: 20k~32k Completion Gate

순수 faithful Dreamer가 느리면 DrQ teacher bootstrap을 사용합니다.

합격 기준:

- 두 seed 모두 nonzero screen completion
- progress만으로 통과 금지
- actor hash를 confirmation 전에 고정
- evaluation cell을 replay에 넣지 않음

### Phase E: 131k Matched Run

오직 Phase D 통과 후 수행합니다.

- 동일 131,072 decisions
- 최소 두 training seed
- DrQ-v2와 호환되는 새로 동결한 CPU 평가 계약과 새 평가 셀; 과거 confirmation 셀 재사용 금지
- 같은 새 셀에서 고정 DrQ actor와 비교하고 훈련 시드별 재현성 확인; 과거 `4~7/32`는 역사적 기준이지 새로운 매칭 수치가 아님
- 한 seed의 단일 `8/32`만으로 승격 금지
- Blind는 두 seed 개선 재현 후에만 사용

## 하지 말아야 할 것

- 현재 actor에 steering L2만 추가
- Entropy 계수만 임의로 키우기
- External noise만 추가
- Reconstruction MSE가 낮다는 이유로 131k 실행
- `continue accuracy=99%`를 정상 학습으로 해석
- Reward alignment를 고치지 않고 world model을 확장
- Grass-heavy replay에 PER만 적용
- 정합성 수리에 필요한 action/target/value 수정과 구별 없이 **선택적** action/replay/reward 처리를 한 번에 묶어 비교
- 평가 trajectory를 teacher data로 재사용
- CEM planner나 TD-MPC2로 우회

## 최종 권고

검증 순서상 우선순위가 높은 후보 구성은 다음입니다.

> **공식 DreamerV3 actor/world-model 계약 복원 + prefill 후 learner pretraining + bounded-normal policy + training-only DrQ teacher bootstrap + episode-balanced replay + 검증된 uncertainty penalty**

이 순서는 현재 알려진 정합성 결함을 먼저 제거하고, 이후 데이터 분포와 교사 사용의 효과를 분리해 시험하려는 설계입니다. 포화·고착이 해소되거나 교사 이득이 발생한다는 주장은 아직 검증되지 않았습니다.

어떤 방법도 완주를 보장하지 않습니다. 아래의 구현 명세는 결함의 존재와 실험 방법을 검증한 결과이며, 아직 새 정책의 성능 향상을 입증한 결과가 아닙니다.

## 2026-09-24 검증 및 구현 명세

### 판정 범위와 출발점

이 명세는 코드 revision `1904ded87500`과 저장된 실험 기록을 읽고,
작은 메모리 내 진단 및 기존 단위 시험으로 앞의 아이디어를 검증한 결과다.
**구현 타당성**과 **완주율 개선의 실증**은 다르다. 아래의 결함은 고쳐야
하지만, 고쳤을 때 포화가 사라지거나 완주가 늘어난다는 인과관계는 아직
확인되지 않았다. 이 문서 작성 과정에서는 새 학습, 로컬 주행 평가, 공식
제출, 모델 확정을 하지 않았다.

| 제안 | 확인된 사실 | 아직 검증되지 않은 성능 주장 | 판단 |
|---|---|---|---|
| DreamerV3 시간축·상태 복구 | 결과 관측이 replay에서 누락되고, reward/continue target의 행동 시점이 어긋나며, warmup과 checkpoint 검사에서 online carry가 불연속이다. 현재 actor도 최종 DreamerV3 목적함수와 다르다. | 이 결함들이 조향 포화의 단독 원인이라는 주장 | 먼저 고쳐야 할 정합성 작업. 작은 합성 시험부터 시작한다. |
| 제어에 유용한 세계모델 게이트 | 기존 진단은 posterior reconstruction 및 자기 replay 기반이다. 512개 표본의 종료가 4개라면 다수 클래스 기준선도 정확도 99.2%다. 실제 양성 수는 별도 계수가 필요하다. | 추천된 open-loop 수치 문턱이 실제 완주를 예측한다는 주장 | 학습 범위 내 별도 개발 자료에서 기준선과 비교하되, 수치 문턱은 실행 전에 동결하고 데이터 부족은 `inconclusive`로 기록한다. |
| Episode-balanced replay와 DrQ teacher | 현재 replay에는 episode/source 균형 표집이 없고, DrQ-v2에는 내부 완주 사례가 있다. | 균형 표집·모방이 Dreamer 일반화와 완주를 높인다는 주장 | 충실한 순수 Dreamer의 게이트 통과 이후에만 각각 독립 처리로 시험한다. |
| 단일 모델·다양한 트랙 검증 | 현재 custom held-out은 하나의 도로 형상에 환경 seed만 다섯 개다. Native CPU evaluator는 custom-map manifest를 직접 받지 않는다. | custom-map 점수가 공식 비공개 트랙 결과를 예측한다는 주장 | 형상 단위 진단은 별도 개발 도구로 취급하고, 공식 근거와 혼합하지 않는다. |

근거: `common_adapter.py:293-320`, `dreamer_v3.py:267-360,575-635,726-746`,
`train_dreamerv3.py:129-160,360-411`, `scripts/diagnose/dreamerv3.py`,
`experiments/dreamerv3-feasibility-gate.json`, `docs/experiments/INDEX.md`,
`docs/results/MODEL_STATUS.md`, `docs/evaluation/generalization-policy.md`,
`training/maps/site/site_map_split.json`.

기존 Dreamer feasibility JSON의 Gate 3 `PASSED`는 **당시의 진단 기록**이다.
제어에 유용한 세계모델이라는 해석은 `docs/experiments/INDEX.md`에서
철회되었다. 기존 파일럿의 warmup progress `0.07719`와 policy progress
`0.01938`은 `scripts/analyze/dreamerv3_pilot.py`가 종료 episode의
`global_step`으로 분류한 값이다. 1,000-step 경계를 가로지르는 episode가
있으므로 두 값은 독립된 같은 조건의 무작위/정책 비교가 아니다. 2k smoke의
reconstruction `0.00773`과 5k 파일럿의 reward correlation `0.62618`도
서로 다른 checkpoint다. 포화와 해당 파일럿의 실패 자체는 관측 사실이다.
DrQ-v2의 내부 confirmation `4~7/32`는 **고유 학습 actor 두 개**의 기록이며
Dreamer의 6-cell smoke와 매칭되지 않는다.

구현 전 기존 시험 기준: `tests.test_dreamerv3 tests.test_train_dreamerv3`
13개, `tests.test_evaluate_policy` 29개,
`tests.test_local_contract tests.test_agent_inference` 26개가
`python -m unittest`에서 통과했다.
`tests.test_site_map_training tests.test_track_generator` 30개도 통과했다.
이번 구현은 종료 관측·전이 정렬·ring wrap·실행 행동 carry 및 checkpoint
검증 상태 보존 회귀를 추가했다. Score-function/KL 목적함수, middle-sequence
burn-in, same-state counterfactual 재현성, 세계모델의 폐회로 효과는 아직
검사하지 않았다. 지도 생성·로드 시험도 독립적인 코스 형상 일반화를
입증하지 않는다. 별도의 학습 성능 시험은 하지 않았다.

### Gate 0: 계약과 실험 경계 동결

**목표:** 결함 수리와 알고리즘 처리, 학습 데이터와 평가 데이터를 혼동하지
않는다. 주 작업 파일은 기존 호환성 경계인 `dreamer_v3.py`, `train_dreamerv3.py`,
`common_adapter.py`, `agent.py`이며, 진단 CLI는 `scripts/diagnose/` 또는
`scripts/analyze/`, 새 시험은 `tests/`에 둔다. `core/`, `env_wrapper.py`,
`damage.py`는 성능 처리로 수정하지 않는다.

1. 우선 기존 소스 revision과 의존성·local official mirror, `frame_skip`,
   관측·행동 범위, 두 학습 RNG seed, 학습 track ID와 geometry seed 배제를
   설계 초안에 기록한다. **수리 후 최종 소스 revision/config**는 A의 코드
   수정과 시험이 끝난 뒤, 최초 환경 자료 수집·학습 전에 확정한다.
   DrQ의 이전 JSON은 형식 참고자료이지 새 Dreamer 프로토콜 또는 새
   평가 셀이 아니다.
2. 훈련, 훈련 범위 내 개발 진단, screen, 새 confirmation, 예약 blind의
   `(track_id, geometry_seed, obstacle setting)`과 목적을 서로 명시한다.
   과거에 소비된 confirmation을 fresh로 다시 부르지 않고, 다른 연구의
   예약 blind도 열지 않는다. 개발 진단 geometry seed **전부**를
   protocol의 `reserved_training_seeds`에 넣고 screen/confirmation/blind
   제외 집합과 합쳐 학습 ID 전체에서 금지되는지 `train_dreamerv3.py`의
   sampler를 테스트한다. `evaluate_policy.py`에는 임의의 `development`
   partition이 없으므로 개발 진단은 별도 훈련 범위 도구로만 기록한다.
3. 학습 decision은 환경 `step()` 한 번(`frame_skip=4`의 기본 local cadence)으로
   센다. 교사 수집, random prefill, 학생 on-policy, 조건부 DAgger의 각각과
   합계를 기록한다. Replay-only update, BC epoch, 교사 질의, 벽시계 시간,
   기존 DrQ 교사의 선행 학습비용은 별도 열로 기록한다.
4. **A의 코드 수리와 시험이 끝나고 각 연구의 첫 자료 수집·학습/화면 전에**
   최종 소스/config revision과 protocol JSON 한 개에 가설, 변경 변수,
   대조군, 환경 계약, screen/confirmation/blind 전체 셀과 개발 제외 집합,
   확인·중단 기준, 지표와 CPU 계약을 동결한다. 동일 연구에서 screen과
   confirmation은 같은 protocol SHA를 사용한다. 별도 replay/교사
   treatment나 131k 연구를 시작한다면 **새 연구의 학습/첫 screen 전에**
   전체 계약을 다시 동결하고 새 protocol에서 screen부터 다시 시작한다.
   이미 본 screen을 새 SHA의 confirmation 앞단으로 가져오지 않는다.
   DrQ 교사 hash는 해당 수집 전에 정하지만, 아직 존재하지 않는 학생
   actor/checkpoint hash는 protocol에 미리 채우지 않는다. 그것은 학습 후
   불변 selection/screen 영수증에 기록하고 confirmation 전에 고정한다.

**통과 조건:** A 작업 전에는 분할 초안과 셀 예약·중복 감사가 끝나야 한다.
최초 환경 자료 수집·학습 전에는 수정 완료된 소스/config를 포함한
protocol이 동결되고 개발 제외 누락이 0건이어야 한다. 신설 학생 actor
hash는 확인 단계 전에 영수증에서 고정한다. 이 계획으로 공식 사이트
작업을 시작하지 않는다.

### Gate A1: 행동-결과 전이와 replay 보존

**전이 규약:** 한 결정은 `(o_t, proposed_native_a_t,
executed_official_a_t, executed_native_a_t, r_{t+1}, o_{t+1}, terminated,
truncated, terminal, is_first, is_last, episode_id)`로 구분한다.
`EpisodeCollector.step()`은 native 행동을 공식 `[steer, gas, brake]` 범위로
변환·클리핑해 실행하고, 실행된 행동을 다시 native로 만든
`transition.action`과 공식 좌표 `transition.applied_action`을 제공한다.
**Dreamer RSSM/replay/BC에는 `transition.action` 좌표를 넣는다.** 물리
감사에는 `applied_action`을 함께 남긴다. 공식 좌표 값을 `step()`의 native
입력으로 다시 넣거나 replay target으로 곧바로 쓰면 페달을 이중 변환한다.

`dreamer_v3.py:300-360`의 replay는 현재 `o_t`만 저장하고 `o_{t+1}`을
버린다. 비종료 전이는 다음 행의 `o_t`로 결과를 복원할 수 있어도 종료·절단
후에는 다음 행이 새로운 episode의 첫 프레임이다. 단순히 reward/continue
인덱스만 한 칸 밀어 그 첫 프레임을 결과 관측으로 쓰면 안 된다. 경계의
`o_{t+1}`를 별도로 저장하거나 전이 구조를 명시적으로 바꾸고,
`is_last`와 `is_terminal`을 분리한다. 시간제한 절단은 episode 경계지만
환경 terminal과 동일한 continuation target이라고 가정하지 않는다.
첫 상태의 이전 행동은 dummy/mask이며, 중간 sequence는 burn-in 또는
보존된 recurrent state로 문맥을 복원한다.

메모리 설계도 작업에 포함한다. `(4,84,84)` uint8 관측은 칸당 28,224
바이트이므로 기본 100,000칸 replay의 관측 배열은 약 2.629 GiB다.
모든 전이에 `next_observation`을 중복 저장하면 비슷한 크기가 추가된다.
경계 결과 영상만 보존하면 통상 더 작지만 실제 episode 빈도에 따라 다르다.
이는 예약 배열 규모이지 그대로 상주 RSS라는 뜻은 아니며, batch 변환,
autograd 및 checkpoint 복사 비용은 별도로 계측한다.

**추가할 시험 (`tests/test_dreamerv3.py` 중심):**

1. 고유한 `o0, o1, terminal_o2, reset_o0` 픽셀과 `a0, a1, r1, r2`
   마커로 2-step 종료 episode 및 다음 episode를 만든다. 모델 입력이
   `o_t, executed_native_a_t`일 때 결과 `o_{t+1}, r_{t+1}, terminal`만
   목표가 되고 `reset_o0`가 결과로 섞이지 않아야 한다.
2. `is_last=True, is_terminal=False`인 절단도 같은 방식으로 검사한다.
   `is_first` 마스킹, ring wrap, sequence 마지막 칸과 중간 sequence
   burn-in에서 잘못된 episode 간 전이와 정보 누출이 없어야 한다.
3. 범위 밖 정책 표본, 실제 클리핑 행동, DrQ 교사 행동에 대해
   `executed_native == ActionAdapter.to_native(executed_official)`을 확인한다.
   replay와 online `prev_a`가 그 실행값을 가리켜야 하며, score 계산에
   필요한 원래 표본은 별도로 식별한다.

**거절/중단:** 한 marker라도 이전 행동이 아닌 현재 행동의 결과 상태에
정렬되지 않거나 episode 경계를 넘으면 환경 interaction 전에 중단한다.
훈련 데이터를 더 넣어 이 결함을 덮지 않는다.

### Gate A2: online recurrent carry와 검증 부작용 제거

현재 random warmup은 `agent.act()`를 호출하지 않는다. 정책이 episode
중간에 제어권을 넘겨받으면 RSSM은 그 앞의 관측·행동 이력을 보지 못한다.
더구나 `verify_checkpoint()`는 살아 있는 학습 agent를 reset하고 가짜
관측으로 전진시킨 후 원래 carry를 복원하지 않는다. 메모리 내 mock 검사는
export 행동 오차가 0인데도 학습 agent carry가 `9 -> 3`으로 바뀌는 사례를
재현했다. 이는 export parity 통과가 *실제 주행의 상태 보존*을 보증하지
않음을 뜻한다.

1. Random/teacher/학생 중 누가 행동을 골랐든 매 결정마다 관측을 정확히
   한 번 받아들인 다음 실제 실행된 native 행동으로 `prev_a`를 갱신하는
   계약을 정의한다. Warmup 끝에 episode 경계까지 기다리거나, 같은
   episode의 prefix를 현재 가중치로 replay해 carry를 준비하는 방식 중
   하나를 선택하고 학습 decision 계산을 함께 명시한다.
2. Checkpoint parity는 독립 복제본에서 수행하거나 원본의 `_online_h`,
   `_online_z`, `_online_prev_a`, `_online_first`와 필요한 RNG 상태를
   완전히 복원한다. 검증이 실제 학습 agent의 다음 행동을 바꿔서는 안 된다.
3. 현재 checkpoint가 online carry와 return-percentile 상태를 저장하는
   것은 아니며 `--resume`도 실제 복구 경로가 아니다. 중간 episode 재개를
   지원하려면 환경 prefix, carry, optimizer/normalizer/RNG까지 별도
   연속성 계약과 테스트를 추가한다. 그전에는 export parity를 완전한
   학습 재개 증명이라고 쓰지 않는다.

**추가할 시험 (`tests/test_train_dreamerv3.py`):** 고정 모델·관측·실행
행동 시퀀스에서 uninterrupted reference와 episode 중간 warmup-to-policy
handoff의 `h/z/prev_a` 및 다음 행동이 일치한다. Episode 경계에서의
handoff도 별도 검증한다. 중간 checkpoint 전후 live carry와 다음 행동은
동일해야 한다. 입력 parity를 위해 실행 행동 대신 정책 제안 행동을
넣는 시험은 허용하지 않는다. 실제 파일럿 포화를 전부 이 결함으로
설명한다는 주장은 이 시험만으로 성립하지 않는다.

### Gate A3: 목표함수와 분포의 수학적 계약

위 A1/A2를 만족한 뒤 기존 계획의 advantage, value/reward two-hot,
free-nat, continuation weighting, slow critic 및 optimizer 정합성을 한
**정확성 복구 단위**로 다룬다. 알고리즘의 성공 주장과는 분리한다.

1. 복구 전 actor는 critic 출력 평균을 직접 최대화하고
   `actor_entropy_coeff`를 사용하지 않았다. 구현된 복구에서는 stopped
   percentile-scaled lambda-return advantage의 `log pi` 항, entropy,
   cumulative continuation weight를 사용한다. 합성 advantage 부호 반전,
   entropy, terminal-weight 시험이 통과했다.
2. 복구 전 sampler는 `Normal(...).rsample()` 다음 `tanh(sample)`이었다.
   Score-function `log_prob`에 이 재매개 표본을 **그대로** 대입하면
   `sample = mean + std * epsilon`의 두 미분 경로가 상쇄되어 Normal의
   mean score gradient가 0이 된다. 이는 **현재 손실에서 일어나는 버그가
   아니라 앞으로 손실을 고칠 때의 위험**이다. 고정 표본에 대한
   `log_prob(sample.detach())` 또는 동등한 비재매개 표본으로 평균의
   gradient와 advantage 부호 반전을 검증한다. Analytic Normal entropy는
   std head에 직접 gradient를 주지만 mean head에는 직접 주지 않는다.
3. 복구된 actor는 공식 pinned `bounded_normal`에 따라 `tanh(raw_mean)`을 평균으로 하는
   Normal이다. **표본 자체를 tanh로 제한하지 않으며** 범위 밖 표본이
   가능하다. 기존 계획의 경계 1회 클리핑과 결합할 때 raw 표본,
   `log_prob`의 표본, 환경 실행 행동, RSSM 이전 행동, replay 행동의
   각 용도를 별도로 정의한다. Squashed Gaussian의 Jacobian을 이
   Normal에 기계적으로 붙이거나 클리핑된 값을 raw 표본의 밀도라고
   간주하지 않는다.
4. KL free-nat는 dynamics/representation 경로에 각각 적용되고 아래
   gradient가 0인지, episode terminal 뒤 imagination 가중치가 0인지,
   two-hot raw-return head 및 replay value anchor와 단위가 일치하는지
   검사한다. score/two-hot/KL/λ-return 통합 단위시험은 통과했다.

**통과 조건:** 합성 표본의 mean score gradient가 기대 방향으로 유한하며
0이 아니고 advantage 부호를 바꾸면 방향이 반전한다. Std entropy
gradient가 존재한다. Terminal 뒤 가중치 0, KL free-nat·two-hot 왕복,
배포 행동 범위·실행 행동 정합성, CPU recurrent reset/export parity가
모두 통과한다. 임의의 entropy 계수 확대나 steering L2는 대체안이 아니다.

**낮은 우선순위의 별도 가설:** 학습 RSSM은 categorical latent를
샘플링하지만 CPU/export actor는 argmax를 사용한다. 차이는 코드상
확인되지만 주행 저하의 증거는 아니다. 같은 **훈련용 개발 시퀀스**에서
posterior 표본 반복과 argmax에 대해 deterministic actor 행동의 축별
차이·경계 비율을 측정하고, 사전에 선언한 허용 오차를 벗어난 경우에만
배포 방식의 별도 처리를 검토한다. 실제 완주 효과는 독립된 폐회로
비교가 필요하다.

### Gate B: 세계모델의 예측 및 행동순위 검증

**선행 자료:** A1의 완전한 전이와 분할 정보가 있는 훈련 범위 내
episode를 수집한다. 학습 episode와 개발 진단 episode는 geometry seed
단위로 분리하고, 완료된 연구의 confirmation/blind를 재활용하지 않는다.
현재 `scripts/diagnose/dreamerv3.py`의 자기 replay posterior 재구성만으로
이 게이트를 통과시키지 않는다. 1k prefill 뒤 환경 제어 없이 learner
pretraining은 `agent.update(model_only=True)`와
`train_dreamerv3.py --replay-pretrain-updates`로 실행할 수 있다. 다만 실제
데이터에서 이 경로가 open-loop gate를 개선하는지는 아직 실험되지 않았다.

1. 서로 겹치지 않는 개발 episode에서 **각각 5개와 10개** 관측/이전 실행 행동으로
   posterior 문맥을 만든다. 이후에는 **기록된 실행 행동만** 사용하여
   prior로 1, 3, 5, 10, 15-step을 전개한다. 미래 관측/posterior를
   rollout 입력으로 재사용하지 않는다. 고정 RNG에서 여러 latent draw를
   집계하고 가장 좋아 보이는 draw만 골라 보고하지 않는다. 각 horizon의
   최신 영상 채널과 전체 stack, 단일·누적 raw reward, continue와
   실제 미래 영상의 posterior latent를 비교한다.
2. Frame stack을 제대로 한 칸씩 옮기는 repeat-last-frame, mean-frame,
   last/constant reward, constant continue 및
   필요한 단순 RSSM 기준선을 같은 셀에 계산한다. 자연 episode의
   terminal prevalence와 positive episode 수를 먼저 보고하고,
   terminal recall, balanced accuracy, PR-AUC, BCE를 prevalence/상수
   기준선과 함께 기록한다. 양성 episode가 부족하면 99% accuracy나
   PR-AUC 한 값으로 통과시키지 말고 `inconclusive: insufficient events`로
   기록한다. 시간 상관이 강한 프레임을 독립 표본으로 세지 않는다. 도로
   영역 및 최신 영상에서 오차를 따로 측정해 정적인 배경/중첩 프레임만
   잘 복원한 경우와 구분한다.
3. 원래 계획의 5-step reward rank correlation `>=0.5`, terminal recall
   `>=0.8`, balanced accuracy `>=0.8`, PR-AUC `>=5x prevalence`,
   BCE `>=10%` 개선은 **사전 제안된
   문턱**이지 효과가 실증된 범용 기준이 아니다. 통계 단위, 충분한 양성
   사례 수, 분류 확률 문턱, 기준선과 사전 고정 판정법을 새 protocol에 적고
   실행 전에 동결한다. 제안 예시는 양성/음성 **각각 서로 다른 episode
   20개 이상**이지만 통계적 충분성이 입증된 숫자는 아니다. 부족하면
   사전 동결 수집 상한까지 훈련 범위의 새 데이터만 늘리거나 판정 보류한다.
   `5x prevalence > 1`이면 그 기준은 정의상 불가능하므로 분포를
   확정할 때 다른 규칙을 미리 정한다. 종료 인접 창을 과표집했다면 자연
   prevalence와 분리 보고한다. 사후에 문턱을 낮추어 합격시키지 않는다.

시간 제한 때문에 미래 15-step 결과를 관측할 수 없는 창은 음성 종료
사례로 바꾸지 않고 그 horizon에서 검열 처리한다. `finished=True`인
완주는 raw 환경의 `truncated` 값과 무관하게 collector의 terminal
규약으로 판정한다. 실제 terminal 이후의 reward는 누적하지 않는다.
관측창 단위가 아니라 서로 다른 episode 및 geometry seed 단위로
신뢰구간을 구한다. 같은 geometry의 장애물 변형을 새 도로로 세지 않는다.

**Counterfactual의 선결 조건:** 현재 `local_simulator`의 `TrackSnapshot`은
도로 형상만 보관하며 차량·Box2D·손상·RNG의 전체 동적 상태를 복원하지
않는다. 따라서 '같은 관측'만으로 동일 시뮬레이터 상태의 분기를 주장하면
안 된다. 공식 local mirror의 물리를 바꾸거나 보호된 환경 파일에 snapshot
기능을 밀어 넣는 대신, 훈련 전용 `(track_id, geometry_seed)`에서 두 독립
환경을 같은 공식 지도·장애물·`frame_skip=4`·최대 결정 수로 reset한 뒤
**공식 좌표로 실제 실행했던 행동 prefix**를 재생하는 대안을 먼저
시험한다. 각 단계의 frame-stack hash, reward, 차량 위치·각도·속도,
손상, 진행률, terminal 및 off-track counter를 대조한다. 영상·플래그는
동일해야 하고 물리량의 허용 오차는 실행 전에 선언한다. 한 번이라도
불일치하면 그 branch를 폐기한다. 이 재현성 자체는 아직 시험되지
않았고 픽셀 같음만으로 숨은 접촉 상태의 일치를 보장할 수도 없다.
통과한 비종료 anchor에서만 사전 고정된 후보 steer/gas/brake **첫 한
decision**을 바꾼 뒤 동일한 기록 행동 접미를 14 decisions까지 적용해
1/5/15-step 누적 reward·progress·terminal 위험의 모델 순위와 실제
순위를 대조한다. Clip 후 동일한 후보, 전부 종료하지 않는 위험 비교,
동률 progress는 유효 순위 표본에서 별도 집계한다. Anchor 내 쌍별
순위 일치율과 geometry 단위 신뢰구간이 우연 수준을 넘어야 한다는 것은
사전 제안이지 아직 검증된 문턱이 아니다. 이는 첫 행동 한 번의 개입
효과이지 분기마다 새로 계획하는 정책의 효과가 아니다.

**판정:** B1 open-loop가 사전 기준선을 이기지 못하면 세계모델 학습 게이트
실패로 분류한다. 양성 사례 부족이나 환경 prefix 비결정성은 모델 실패가
아닌 **검증 불충분/도구 실패**다. B1 통과, B2 도구 미완성이면
안정성 확인만을 위한 Phase C 10k까지 진행할 수 있다. B2가 실행
불가능하면 행동순위가 검증됐다고 쓰지 않고, 선택적 teacher/replay
처리와 Phase D/E 확장도 보류한다. 모델/시뮬레이터의 action 좌표와
frame cadence가 어긋났다면 먼저 A1과 재현성 시험으로 돌아간다.

### Gate C: 작은 정책 파일럿과 단계별 데이터 처리

**순수 복구 버전:** A와 B1을 통과한 구성에서 두 독립 학습 seed의 같은
예산·환경·개발 평가 계약을 사용한다. Warmup을 매 **transition**에
`controller=random/teacher/student`, 실행 행동, source, episode,
`global_step`으로 기록한다. Warmup과 정책이 섞인 episode를 둘 중
하나의 독립 대조군으로 분류하지 않는다. Checkpoint 선택 시점,
CPU 재적재 시험, screen 셀과 중단 조건을 사전 동결한다. 10k 파일럿의
조향 `|a| >= .99` 비율 `50%` 미만은 계획의 **잠정 진단 문턱**이며,
완주 증거를 대신하지 않는다. 두 seed의 finite 실행, 실제 행동 분산,
CPU carry/export, 개발 screen의 종료 사유를 함께 보고한다.
B2 미판정 상태의 C는 **진단 목적 파일럿**이며 C의 안정성만으로
Phase D, 성능 우열 또는 행동순위 개선을 주장하지 않는다.

**규모의 예시이지 새 셀 지정은 아님:** 새 훈련 범위 내 개발 자료는
4개 학습 ID 각각 geometry 8개(32 episode)를 초기 조사 규모로,
종료 사례가 부족할 때의 상한 64 episode를 사전에 제안할 수 있다.
분기 구현의 첫 타당성 검사는 분리된 8 episode, episode마다 미리 정한
anchor 두 개와 행동 최대 다섯 개로 제한한다. 10k screen은 별도
3개 개발 ID x geometry 4개인 12 canonical cell/actor와 CPU 재적재
두 번으로 설계할 수 있다. **이 수치는 아직 fresh를 확인하지 않은
설계 예시**이며, 기존 실행·예약 목록을 조사해 배제 조건을 확정하고
protocol에 동결하기 전에는 실행하지 않는다. 두 reload를 24개의 독립
도로라고 세지 않는다.

**데이터 처리 1: replay 분포.** 순수 복구 버전이 안정적이고 B2가
통과한 경우에만
고정된 episode-balanced 대 uniform-step 샘플러를 하나의 변경 변수로
비교한다. 시작 제안은 `50/50` 혼합이며 parameter sweep이 아니다.
`Uint8SequenceReplay`에 episode/source 색인, 경계·ring overwrite,
유효 sequence 길이·burn-in, teacher/초기 episode 보존을 명시하고
배치별 source 비율, 고유 episode·geometry 수, terminal/finish 빈도를
로그한다. Uniform 부분을 남겨 실패 데이터가 사라지지 않게 한다.
같은 기존 자료/환경 decision/학습 update/두 seed에서 비교하고,
재표집 빈도가 바뀌었다는 것만으로 성능 향상이라고 부르지 않는다.

**데이터 처리 2: 동결 DrQ teacher.** 순수 C 파일럿은 안정적이지만
완주가 나타나지 않거나, 순수 Phase D 시험이 완주 관문에 실패하고,
B2가 통과했을 때 별도 처리로 검토한다. 이미 사용한 screen을 새
처리의 confirmation으로 재사용하지 않는다.
DrQ control의 actor hash·소스·
checkpoint를 정하고 학습 track ID `1-4`와 새 protocol이 허용한
geometry seed에서만 독립 trajectory를 수집한다. Screen/confirmation/
blind 및 다른 연구의 예약 geometry seed 합집합을 배제한다. 각 전이의
`o_t, o_{t+1}`, 교사 제안 행동, 실제 공식/역변환 native 행동, raw reward,
종료/절단, episode·track·geometry, 교사 hash와 수집 원천을 저장한다.
World model pretraining 후 actor BC를 짧게 적용하되 50/50 등 교사/온라인
혼합률과 BC 종료 조건은 별도 protocol에서 먼저 고정한다. 최종 배우는 actor는
**학생 단독**으로 평가한다. BC loss가 낮아도 완주 증거가 없으면 합격
아니다. 현재 PPO DAgger 경로는 다른 교사와 2차원 페달 좌표를 사용하므로
그대로 Dreamer에 재사용하지 않는다. 결과명은 반드시
`DreamerV3 + DrQ bootstrap`이다.

**조건부 처리 3: on-policy DAgger.** 올바른 전이·교사 BC 후에도 학생이
방문하는 훈련 상태에서 조기 이탈이 재현되고, 그 상태에서 동결 교사가
더 나은 행동을 제안한다는 훈련 전용 증거가 있을 때만 별도 처리로
검토한다. `teacher_label`과 `executed_action`을 구분하며 혼합 제어의
교사 실행 비율을 기록한다. 교사가 학생 대신 운전한 episode를 학생
단독 완주로 세지 않는다. 교사도 해당 상태에서 불확실하거나 실패하면
DAgger를 강행하지 않는다.

**예산과 선택:** 각 처리에서 teacher 수집 + random prefill + 학생
online + DAgger의 합산 환경 decision, replay-only update, BC epoch 및
DrQ 교사 선행 학습 131,072 decision을 별도 공개한다. Teacher를 쓴
Dreamer와 순수 Dreamer를 '동일 총 학습 자원의 알고리즘-only 비교'라고
표현하지 않는다. 새 처리마다 변경에 영향받는 A의 계약 시험과 B1,
B2, 두 seed의 C 10k pilot 및 CPU 검사를 **다시** 통과해야 한다.
Phase D의 20k~32k 완주 게이트에서는 두 학습 seed
모두 사전 동결한 screen에서 실제 nonzero finish가 있어야 한다.
Progress만으로 131k로 확장하지 않으며, A/B/CPU package 게이트가
깨지면 그 처리의 추가 학습을 멈춘다.

### Gate D: 새 평가 셀, 단일 모델, 형상 다양성

**공식 local mirror 평가:** `evaluate_policy.py`로 같은 후보 actor를
고정한 새 study-specific protocol에서 평가한다. 비교 DrQ actor도
동일 셀에서 고정한다. CPU 두 재적재는 결정성 검사이지 새로운 두 개의
독립 학습 seed나 도로가 아니다. Actor/checkpoint/source/protocol/환경
해시를 확인 전에 동결하고, `repeat == 0`만 결과 집계에 포함한다.
완주 우선, 미완주 진행률과 완주 랩타임은 동결 protocol의 순서로
해석하며 reward·damage·조향 부드러움은 내부 진단 지표다. Dreamer+
교사와 DrQ의 학습 비용·자료 분포가 다르면 같은 평가 셀이라도 훈련
방법까지 매칭되었다고 주장하지 않는다. 과거 DrQ confirmation 수치나
Dreamer 0/6 smoke를 새로운 matched outcome으로 재사용하지 않는다.
기존 `evaluate_policy.py`의 수치형 CPU latency/RSS eligibility 검사는
DrQ 경로에만 적용되므로, 새로운 Dreamer actor에는 별도로 Python 3.11
CPU 환경에서 생성 `<=10s`, 각 reset/act `<=5s`, 전체 프로세스
peak RSS `<=1024 MB`를 계측·판정한다. 과거 Gate 1과 로컬 패키지
smoke는 **옛 구현**의 운영 증거이지 수정 actor의 공식 서버 통과
증명이 아니다. 공식 pinned Dreamer의 `policy()`와 이 저장소의 CPU
deterministic 선택은 같은 mode 계약이라고 가정하지 않는다.

**Custom-map 스트레스(별도 도구):** 현행 `training/maps/site/site_map_split.json`
의 held-out은 도로 **하나**와 seed 다섯 개다. 현 생성기에는 oval,
s-curve, hairpin, chicane, technical, extreme-technical 형식이 있으나
다른 seed 또는 정확히 다른 fingerprint가 독립적인 코너 구조를
보장하지 않는다. 새 지도를 만들 경우 map ID뿐 아니라 centerline
fingerprint, generator version/template, 길이·폭, 코너 순서/반경,
장애물 형상과 회전·시작 위치·방향을 고려한 유사성을 분할 전에 검사한다.
원래 held-out 지도를 개발 튜닝에 재사용하지 않는다. Native CPU
`evaluate_policy.py`는 custom-map manifest를 직접 받지 않고,
`training/evaluate_closed_loop.py`는 별도 PPO/CEM 경로다. 동일한
native actor를 custom 환경에서 검증하는 연결과 parity 시험이 먼저
필요하다. 그 결과는 항상 **로컬 custom 환경 proxy**로 별도 표기한다.

**공개/비공개와 외부 행동:** 2026-09-24 10:16 UTC의
[공식 공개 트랙 API](https://ships-duo-ethical-saver.trycloudflare.com/api/tracks)
읽기 전용 확인에서는 공개 트랙 1·2·3이 조회됐다. 이 스냅샷은 바뀔 수 있고
`docs/competition/info.md`의 전날 1·2 기록보다 새로운 시점의
정보다. 하나의 동일한 불변 package로 모든 당시 접근 가능한 공개
트랙 결과를 기록해야 하며, 트랙별로 서로 다른 제출의 최고값을 합쳐
한 모델의 성능으로 해석하지 않는다. 공개 트랙은 반복적인 지도별
튜닝셋이 아니고 공식 private 트랙은 접근 가능한 내부 holdout이
아니다. 공식 제출/모델 확정은 이 계획에 포함되지 않는다. 실행
직전에 공식 사이트와 Participants 규칙을 다시 확인하고 사용자에게
그 **별도 외부 행동**에 대한 명시적 승인을 받아야 한다.

### 구현 결과물의 최소 기록 계약

`experiments/`의 새 protocol은 아래 필드와 제외 규칙을 먼저 동결한다.
`runs/`의 episode/transition 파일 및 `evaluations/`의 영수증은 각각
protocol ID와 hash를 참조한다. 기존 프로토콜의 셀 번호를 그대로
복사하여 새로 쓰라는 뜻이 아니다.

| 단위 | 최소 필드 및 판정 이유 |
|---|---|
| 연구 protocol | 수리 후 `source_revision`, source/environment/dependency fingerprints, `algorithm/config_hash`, 각 학습·개발·screen·confirmation·blind의 track/geometry 목록과 제외 집합, 사용 시 **기존 교사** hash, 예산·중단선·지표 정의, CPU runtime 계약. 미래 학생 actor hash를 요구하지 않으며 결과 후 threshold 변경 금지. |
| 수집 episode | `run_id`, `learner_seed`, `track_id`, `geometry_seed`, `episode_id`, `partition`, `obstacle_mode`, `frame_skip`, 지도/hash(있을 때), `collector_kind`, 시작/종료 결정, `finish`, `retire_reason`, 데이터 출처와 교사 hash. |
| 전이 | `decision_index`, `controller_kind`, `observation_hash`, `next_observation_hash`, 정책 raw sample/교사 label, `executed_native`, `applied_official`, raw reward, `terminated`, `truncated`, `terminal`, `is_first`, `is_last`, `progress`, `damage`. 표본과 실행 행동은 서로 다른 열. |
| Open-loop 창 | `episode_id/anchor_t`, 5 또는 10-step context, horizon, RNG draw, 실제·예측 최신 영상/전체 stack 오차, trivial baselines, 단일/누적 reward, continue·terminal/검열 이유, 미래 posterior 지표, terminal 발생 episode·geometry 수. |
| Counterfactual 분기 | prefix 실행 행동 hash 및 parity 오차, anchor/후보 행동의 native·official 좌표, clipping 후 중복 여부, 고정 접미 hash, 1/5/15-step 모델/실제 reward·종료·progress, 동률·조기 종료 사유. |
| CPU screen·확인 | **학습 후 선택·고정한** 모델/checkpoint/actor/package/source/protocol hash, 고유 학습 seed, canonical `repeat==0` 셀과 reload 반복 구분, 완주/랩타임/미완주 progress, latency/RSS, 행동 유효성 및 episode 종료 이유. |

영상 자체를 매 step 영구 보관하는 것은 큰 저장 비용을 초래한다.
전이 내용이 필요한 학습/진단 세트는 해시뿐 아니라 실제 픽셀을
replay 또는 분리된 훈련 전용 아티팩트에 보존하고, 일반 episode 원장은
해시·행동·레이블로 충분한지 실험 전에 정한다. 해시만으로 미래
open-loop 영상 오차를 다시 계산할 수는 없다. 실제 모델 입력에
공식 평가에서 주어지지 않는 위치·맵·물리 레이블을 섞지 않는다.

**구현자가 바로 확인할 원본:** [현재 상태](../../context/current-state.md),
[실험 색인](../../experiments/INDEX.md), [평가 절차](../../evaluation/protocol.md),
[분할 이력](../../evaluation/generalization-policy.md),
[모델 장부](../../results/MODEL_STATUS.md),
[Dreamer 동결 진단](../../../experiments/dreamerv3-feasibility-gate.json),
[공식 Dreamer actor/critic](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py),
[bounded-normal head](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/jax/heads.py),
[Normal 출력](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/jax/outs.py).

### 구현 순서와 작업 완료의 증거

| 순서 | 소유 파일/기능 | 환경 실행 전·후 산출물 | 다음 단계 허용 조건 |
|---|---|---|---|
| 0 | 연구별 분할·예산 설계 및 셀 예약 | `reserved_training_seeds` 포함 geometry 감사와 screen/confirmation/blind 초안; 이후 코드 수정은 허용 | A 수리 전 셀 사용/누출 방지; 최종 protocol SHA는 행 4에서 확정 |
| 1 | `dreamer_v3.py` replay/sequence, `common_adapter.py` 행동 좌표, `tests/test_dreamerv3.py` | marker 전이, terminal/truncation, ring/burn-in, 클리핑 시험 | 관측·행동·reward·종료 정렬 오류 0건 |
| 2 | `train_dreamerv3.py` prefill/checkpoint, `tests/test_train_dreamerv3.py` | 중간 episode 전환 및 검증 전후 live carry/다음 행동 parity | 검증이 주행·RNG를 오염하지 않음 |
| 3 | `dreamer_v3.py` objective/분포, `agent.py` Dreamer export, 관련 시험 | detached score, entropy, free-nat/two-hot/terminal weight, CPU parity | 합성 loss와 배포 행동 계약을 만족 |
| 4 | `train_dreamerv3.py` replay-only pretraining 경로와 시험; **수리된 소스의 연구별 protocol 동결** | 1k prefill 뒤 100-500 update의 환경 무제어 경로 시험, 최종 source/config hash와 전체 분할 SHA | 코드 시험 통과와 최종 protocol 동결 후에만 B1 데이터 수집·prefill/학습 시작 |
| 5 | `scripts/diagnose/`의 B1 open-loop와 조건부 B2 prefix-branch | 분리 개발 episode, horizon별 기준선·종료 prevalence·분기 재현성 | B1 통과 시 진단 C 가능, B2 통과 전에는 treatment/Phase D 보류 |
| 6 | `train_dreamerv3.py` 순수 복구 버전 두 seed 10k 파일럿 | controller별 prefix 로그, CPU screen, run/result/selection 기록 | Phase C의 안정성 게이트; 완주는 아직 필수 아님 |
| 7 | 조건부 episode-balanced replay 또는 DrQ teacher/BC, 필요 시 DAgger | 처리마다 행 0·4의 설계/동결, 관련 A/B1/B2/C 재검증, source·예산·교사 hash | B2 및 학생 단독 C 안정성; 별도 처리로만 Phase D 진입 |
| 8 | 순수 또는 별도 처리의 Phase D 20k-32k 완주 게이트 | 동일 연구에 사전 동결된 screen, 두 seed 실제 완주와 CPU 검사 | 두 seed 모두 nonzero finish일 때만 후속 연구 후보 |
| 9 | 별도 131k 연구의 설계·전체 계약 동결(행 0·4 재수행), 별도 실행 결정 후 두 seed 131,072-decision 학습·CPU screen·후속 확인 | 새 protocol SHA의 screen과 **같은 SHA** confirmation·조건부 blind의 영수증, 단일 actor/package hash | 미래 별도 규모 연구의 내부 후보 판단만 가능; 이 문서는 실행 승인 아님 |

행 7의 새 처리와 행 9의 신규 규모 연구는 각각 **새 연구**다. 행 0에서
해당 연구의 screen, confirmation, blind를 설계하고, 행 4에 준해 최종
소스/config와 전체 분할 protocol을 **첫 수집·학습 전에** 동결한다.
그 protocol SHA로 학습부터 첫 screen·confirmation까지 이어 간다.
이전 연구의 screen을 다른 SHA의 confirmation과 연결하지 않는다.

구현할 때마다 관련 신설 테스트와 기존 회귀 시험을 실행한다.

```bash
python -m unittest tests.test_dreamerv3 tests.test_train_dreamerv3 tests.test_evaluate_policy tests.test_local_contract tests.test_agent_inference
```

Source·환경·checkpoint/protocol 해시, command, seed,
screen/confirmation 사용 이력, CPU latency/RSS 및 결과를 남긴다.
`docs/workflows/run-experiment.md`의 동결·중단·사후 기록 순서를 따른다.
Custom-map 작업이 있을 때는 별도로
`python -m unittest tests.test_site_map_training tests.test_track_generator`를
실행한다. 이 구현에서 지정한 107개 회귀 시험이 통과했지만, 이는
**새로운 Gate A/B의 학습·세계모델 통과 증거가 아니다.**
정합성에서 실패하면 알고리즘 튜닝을 멈추고 정합성을 수정한다.
Phase D 반복 완주 실패 또는 CPU/패키지 제한 실패는 131,072-decision
규모 확장의 거절 조건이며, DreamerV3 계열 전체의 기각은 아니다.
