# DreamerV3 실패 진단 기반 발전 전략

## Active Plan Status

| Field | Current plan |
|---|---|
| Status | Active research plan; no recovery implementation or new recovery run has been recorded. |
| Goal | Repair the identified DreamerV3 fidelity defects, then determine whether a faithful implementation can pass feasibility before matched training. |
| Rationale | The current formulation failed the small policy-learning gate, while its prior Gate 3 interpretation was withdrawn as insufficient world-model validation. |
| Evidence | `experiments/dreamerv3-feasibility-gate.json` and the linked run diagnostics; see `docs/experiments/INDEX.md`. |
| Hypothesis | Correct transition alignment, actor/critic objectives, action distribution, replay/pretraining, and world-model gates can remove implementation-induced collapse. |
| Scope | Correctness repair and feasibility only. No 131,072-decision matched run, official submission, or confirmation follows automatically. |
| Steps | Implement and unit-test Phase A correctness repairs, pass Phase B open-loop/counterfactual gates, then run the Phase C small pilot. |
| Evaluation | Frozen local contract, CPU recurrent/export checks, explicit open-loop and counterfactual diagnostics, then repeated pilot screen completion. |
| Acceptance | The plan's Phase A/B gates pass and the renewed pilot achieves the declared nonzero repeated screen-completion criterion without operational failure. |
| Rejection | A faithful repair still fails the renewed pilot or violates CPU/package constraints. Record the result instead of scaling it. |
| Stop condition | Do not start matched scale-up until all preceding gates pass. Official actions require their separate workflows and explicit user authorization. |

The detailed diagnosis and phased technical work below are retained from the
pre-migration recovery strategy.

## 핵심 결론

현재 실패는 **DreamerV3 자체의 실패가 아니라, DreamerV3와 다른 actor/world-model objective를 사용한 구현의 실패**로 보는 것이 정확합니다.

관측된 steering saturation은 사실이지만, 원인을 단순히 `tanh`로 돌릴 수 없습니다. 현재 구현에는 saturation을 직접 유발할 수 있는 여러 fidelity defect가 있습니다.

따라서 가장 성공 확률이 높은 경로는 다음입니다.

> **충실한 DreamerV3 복원 → prefill 후 learner pretraining → episode-balanced replay → training-only DrQ teacher 데이터로 world model과 actor를 warm-start → 필요할 때만 uncertainty-aware imagination**

## 핵심 결함

| 우선순위 | 현재 구현 문제 | 영향 |
|---:|---|---|
| Critical | Actor가 λ-return advantage가 아니라 symlog critic 출력 평균을 직접 최대화 | model exploitation과 극단 action 유도 |
| Critical | Reward/continue target이 action과 한 스텝 어긋남 | action 결과를 올바르게 학습할 수 없음 |
| Critical | `actor_entropy_coeff`가 loss에 사용되지 않음 | 포화 정책에 복원력이 없음 |
| Critical | Raw return percentile과 symlog critic 값을 혼합 | actor gradient scale이 비일관적 |
| High | `tanh(mu + sigma*eps)` 사용 | 큰 `mu`에서 탐색 분산과 gradient가 함께 소멸 |
| High | KL free-nat 값을 계산하고 버림 | posterior-prior collapse 유도 |
| High | Reward/value가 scalar MSE | DreamerV3의 scale-free two-hot 분포형 학습이 아님 |
| High | Continuation weighting 누락 | 사실상 종료된 imagined state도 동일 가중치 |
| High | Critic slow regularization/replay value anchor 누락 | reward-model 오차가 critic과 actor로 직접 전파 |
| Medium | Mid-episode sequence가 항상 zero recurrent state에서 시작 | sequence context 손실 |
| Medium | 1,000 random steps 직후 미학습 actor가 바로 제어 | grass-only replay lock-in |
| Medium | Reconstruction loss scale이 공식 구현보다 매우 작음 | 공유 표현에서 image gradient가 약할 가능성 |

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

현재 구현의 `-mean(critic(imagined_state))`는 최종 논문 방식도, 2023년 초기 코드 방식도 아닙니다.

## Gate 재해석

기존 Gate 3의 “world model 정상” 판정은 철회하는 것이 맞습니다.

- Reconstruction MSE는 posterior teacher-forced reconstruction일 뿐 open-loop dynamics 정확도가 아닙니다.
- Continue accuracy `99.21875%`는 512개 중 terminal이 4개라면 항상 continue만 예측해도 나오는 값입니다.
- Continue BCE `0.04437`도 constant-prevalence baseline 약 `0.0457`보다 거의 낫지 않습니다.
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
3. `log π(a|s)`와 entropy `3e-4`를 실제 actor loss에 적용합니다.
4. Actor와 critic loss에 cumulative discount/continue weight를 적용합니다.
5. Action distribution을 공식 `bounded_normal`로 바꿉니다.
6. Actor output layer를 `outscale=0.01`로 초기화합니다.
7. Reward/value를 255-bin symexp two-hot distribution으로 변경합니다.
8. Reward/value output layer를 zero initialization합니다.
9. KL dynamics/representation 항에 free-nat 1을 각각 적용합니다.
10. KL 가중치는 공식 `1.0/0.1`을 기준으로 사용합니다.
11. Current critic으로 λ-return을 계산하고 slow critic regularization을 추가합니다.
12. Replay value loss로 critic을 실제 transition return에 고정합니다.
13. Sequence 시작에 burn-in 또는 저장된 recurrent carry를 사용합니다.
14. AdamW 기본 weight decay를 제거하고 공식 optimizer 규약을 맞춥니다.

Action 관련 직접 권장 설정:

- `mean = tanh(raw_mean)`
- action-space Gaussian noise
- `std ∈ [0.1,1.0]`
- 환경 경계에서 단 한 번 clip
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

완주 확률을 가장 직접적으로 높이는 방안입니다.

1. Frozen DrQ actor로 training IDs `1–4`에서 fresh trajectory를 수집합니다.
2. Screen/confirmation/blind seed는 모두 제외합니다.
3. Pixel, 실제 applied action, raw reward, terminal만 저장합니다.
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

현재 uniform-step replay는 긴 실패 tail이 학습 분포를 독점하게 합니다.

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

같은 관측 상태에서 여러 steer/gas/brake action을 실제 환경과 world model에서 분기합니다.

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
- Entropy gradient가 mean/std head에 전달
- Terminal 이후 imagination weight 0
- Two-hot encode/decode 왕복
- Bounded mean 포화 상태에서도 action variance 유지
- CPU recurrent reset/carry parity 유지

### Phase B: World Model

1k prefill + learner pretraining을 수행한 뒤 held-out open-loop gate를 통과해야 합니다.

Posterior reconstruction MSE만으로 통과시키지 않습니다.

### Phase C: 10k Pilot

- 두 training seed 사용
- 동일 off-track-at-109 trace 반복 금지
- steering `|a|≥0.99` 비율을 잠정 50% 미만으로 제한
- action-space std/entropy가 0으로 붕괴하지 않음
- policy progress ≥ random warmup 0.077
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
- Frozen DrQ-v2와 동일 CPU protocol
- seed별로 DrQ confirmation `4~7/32`를 반복적으로 초과
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
- 여러 action/replay/reward 변경을 한 번에 적용
- 평가 trajectory를 teacher data로 재사용
- CEM planner나 TD-MPC2로 우회

## 최종 권고

완주 가능성이 가장 높은 구성은 다음입니다.

> **공식 DreamerV3 actor/world-model 계약 복원 + prefill 후 learner pretraining + bounded-normal policy + training-only DrQ teacher bootstrap + episode-balanced replay + 검증된 uncertainty penalty**

이 순서는 현재 포화 원인을 직접 제거하고, 초기 grass-only 데이터 고착을 방지하며, 이미 검증된 DrQ의 도로 주행 능력을 Dreamer의 초기 상태 분포로 이전합니다.

어떤 방법도 완주를 수학적으로 보장할 수는 없습니다. 그러나 현재 artifact와 문헌 근거를 종합하면 위 구성이 **완주 확률을 최대화하면서 실패 원인을 단계적으로 식별할 수 있는 최적 전략**입니다.

이번 작업은 조사와 설계만 수행했으며 파일 수정이나 학습 실행은 하지 않았습니다.
