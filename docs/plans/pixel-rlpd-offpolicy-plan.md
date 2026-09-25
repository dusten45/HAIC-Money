# Pixel RLPD: HAIC native 관측을 위한 독립 오프폴리시 SAC 계획

> Historical proposal (2026-09-24). The v2 pilot stopped at its two-seed gate;
> a separately frozen long-horizon follow-up produced an internal candidate, and
> a fresh entropy-target v4 ablation is running. The proposed budgets and
> authorization below describe that dated plan, not a current protocol or
> permission for an official action. See `docs/experiments/INDEX.md` and
> `docs/context/current-state.md` for current outcomes and restrictions.

## 상태와 판단 범위

| 항목 | 계획 |
|---|---|
| 상태 | **bounded local-study execution authorized.** 2026-09-24 사용자가 이 계획을 실행하고 수집·학습·평가·피드백·조건부 반복까지 진행하라고 명시했다. 이는 등록된 repository-wide active plan을 대체하지 않으며, 공식 제출이나 모델 확정 승인도 아니다. 기존의 “의향은 실행 승인이 아님” 표기는 이 명시적 지시에 의해 superseded 되었다. |
| 질문 | 고정된 훈련 전용 DrQ-v2 주행 자료를 50:50으로 섞는 pixel RLPD가 동일한 pixel SAC의 **offline 자료 없는** 대조군보다 새 내부 도로에서 일관되게 완주할 수 있는가? |
| 주가설 | 동일한 SAC 구성과 학생의 온라인 상호작용/업데이트 예산에서, 고정 prior data의 균형 표집이 초기 탐색과 나중의 완주를 개선할 수 있다. **가설**이며 HAIC에서 검증된 사실이 아니다. |
| 귀무/실패 | 오프라인 자료가 학생의 새로운 상태 방문을 대체하거나 Q의 분포 밖 행동 과대추정/교사 의존을 키워 완주가 늘지 않거나 악화될 수 있다. |
| 대상 | 독립적인 `haic/algorithms/rlpd/` PyTorch pixel SAC, 훈련 전용 frozen teacher dataset, 순수 학생 actor의 CPU export, 기존 native 평가/패키지 경계의 명시적 확장. Dreamer, DrQ-v2 학습 구현이나 공식 환경 물리를 수정하는 계획이 아니다. |
| 성공의 뜻 | 사전 동결한 내부 셀에서 두 **고유 학생 학습 seed**의 매칭된 SAC 대비 차이가 재현되고, 순수 학생 CPU/패키지 계약을 통과한 **내부 후보**에 한한다. 공식 HAIC 점수/제출/확정이 아니다. |

이 연구는 별도의 DrQ teacher-replay 처리 성공/실패를 선행 조건으로 두지 않는다. 두 계획은 각각 독립 study ID, source/protocol, 사전 자료 수집 및 fresh 분할을 가지며, 한 계획의 화면이나 blind 결과를 다른 계획의 새로운 확인 자료로 가져오지 않는다. 기존 DrQ actor를 **고정된 prior-data 생성기**로 쓰는 것과 그 actor의 기존 DrQ critic/replay를 계속 학습시키는 것은 다른 실험이다.

현재 DrQ-v2는 `docs/context/current-state.md`와 `docs/results/MODEL_STATUS.md`상 내부 검증 대조선이며, 두 고유 control actor가 과거 fresh confirmation에서 32셀당 4~7회 완주했다. 그 수치는 이 연구의 새 매칭 관측치가 아니다. 기존 DrQ 좁은 L2/pad 조정은 종료되었고 이 제안은 새 알고리즘 계열이다. 아래 예산, seed 번호, 체크포인트 시점, 셀 크기, 문턱 중 **공식 RLPD 소스 또는 HAIC 제약이라고 명시하지 않은 숫자는 모두 실행 전 검증·동결할 제안값**이다.

## 논문과 저자 구현에서 가져올 것, 가져오지 않을 것

검증 가능한 기준은 [RLPD 논문 v4](https://arxiv.org/html/2302.02948v4), [저자 코드 `c90fd4baf28c9c9ef40a81460a2e395092844f88`](https://github.com/ikostrikov/rlpd/tree/c90fd4baf28c9c9ef40a81460a2e395092844f88)이다. 다음은 **저자 pixel 경로의 실제 설정/동작**과 HAIC 이식 선택을 분리한 것이다.

| 항목 | 논문/저자 pixel 소스의 실제 근거 | 이 계획의 HAIC 처리 |
|---|---|---|
| prior data | 논문 4.1절의 온라인/오프라인 **50:50 symmetric sampling**. [pixel trainer](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/train_finetuning_pixels.py)의 `offline_ratio=0.5`, 별도 dataset/online replay를 interleave. | 훈련 전용 frozen DrQ trajectory 저장소와 학생 online buffer에서 각 배치 절반씩, source별 통계 기록. SAC 대조군은 동일 배치 크기를 전부 online으로 채운다. |
| 사전학습 | 논문은 offline-only pretraining과 명시적 BC를 쓰지 않는다. pixel trainer는 초기 무작위 온라인 수집 뒤 혼합 업데이트를 시작한다. | 기본 비교에는 offline-only 업데이트, BC, 교사 행동 강제, DAgger **없음**. 안정화용 추가 pretraining은 별도 이름·새 protocol의 후속 가설이다. |
| critic | [pixel config](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/configs/rlpd_pixels_config.py): `num_qs=10`, `num_min_qs=1`, `critic_layer_norm=True`, `backup_entropy=False`. [SAC 업데이트](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/sac/sac_learner.py)는 무작위 target subset, 전체 ensemble actor-Q 평균, soft target update를 사용. | 10개 독립 Q head, 매 업데이트 무복원 추첨 target Q 한 개, 모든 Q head를 같은 TD target으로 학습. Critic hidden LayerNorm 사용. `num_min_qs=1`이면 `min`은 선택된 1개 값이다. 이를 모든 10개 최소값 또는 고정 Q1로 바꾸지 않는다. |
| pixel와 분포 | [pixel base](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/configs/pixel_config.py)는 D4PG CNN `(32,64,128,256)`, 3x3/stride 2, latent 50, MLP `(256,256)`. [DrQ learner](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/drq/drq_learner.py)는 critic CNN을 actor에 복사하고 actor CNN의 정책 gradient를 막으며, [pixel multiplexer](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/networks/pixel_multiplexer.py)는 `uint8 / 255`, latent LayerNorm/Tanh를 적용. [augmentations](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/drq/augmentations.py)는 edge pad 4의 독립 random crop. | 같은 처리 순서를 Torch로 옮기되 입력 4장 회색 84x84와 native 3행동으로 적응. 원본 V-D4RL의 64px/3 stack, JAX/Flax/TFP/Gym/MuJoCo 체크포인트와 학습 의존성을 복사하지 않는다. |
| 엔트로피/업데이트 | [pixel trainer](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/train_finetuning_pixels.py)의 기본 `utd_ratio=1`; [기본 pixel DrQ config](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/configs/drq_config.py)의 `gamma=.99`, `tau=.005`, actor/critic/temp LR `3e-4`, 초기 alpha `.1`. [DrQ learner](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/drq/drq_learner.py)의 기본 target entropy는 `-action_dim/2`. | 한 학생 env decision당 critic/actor/alpha 각각 1회, target EMA 1회: 최초 비교의 UTD=1, gamma=.99, tau=.005, LR `3e-4`, alpha 시작 `.1`, native 3축 목표 entropy `-1.5`는 **저자 기본을 적용하는 제안**이다. 수치를 변경하려면 새 연구로 취급. |

논문의 pixel 실험은 **V-D4RL/DMC**에서 이루어졌고 HAIC CarRacing 4x84, 3차원 비대칭 공식 행동, 장애물·은퇴 규칙의 성능을 보증하지 않는다. 논문 4.5절은 target subset 2도 검토하라고 권하지만 **저자 pixel config는 1**이다. 또한 `backup_entropy=False`는 Bellman target에서만 `-alpha log pi`를 빼는 선택을 끄며, actor loss와 학습 온도까지 제거한다는 뜻이 아니다. 논문 의사코드의 선택적 entropy 줄만으로 구현 부호를 추측하지 말고 위 코드의 분기와 손실을 기준으로 이식한다. 원본 저자 실행은 Python 3.9/JAX 계열이고 공식 제출은 CPU Python 3.11/Torch 2.1.0 계열이다. 독립 Torch 포트와 원본의 수학적 차이는 source 매핑과 단위시험에 남긴다.

## 환경, 데이터, 물리적 행동 계약

- 현재 로컬 공식 제약 전사는 `docs/competition/restrictions.md`: `variables-6`, Linux CPU/Python 3.11, `gymnasium[box2d]==0.29.1`, CPU `torch==2.1.0`, `numpy==1.26.0`, `opencv-python==4.8.1.78`이다. 학습기는 별도 CUDA Torch 환경을 써도 **제출 추론에는 GPU/JAX/Flax/TFP/teacher/critic/replay를 넣지 않는다**. 공식 외부 행동 전에는 당시 사이트와 [Participants](https://github.com/2026-HAIC/Participants)의 최신 규칙을 다시 확인한다.
- `common_adapter.ObservationSpec` 입력은 이미 프레임 4장, CHW `(4,84,84)`, `float32 [0,1]`; 학생 CNN은 학습 시 `uint8`로 저장했다가 `/255`하여 동일 관측을 복원한다. 채널별 서로 다른 crop 또는 별도 grayscale/리사이즈/제어 plane/`VecNormalize`를 도입하지 않는다. 학습 random shift는 4채널 stack에 동일 공간 이동, 각 샘플과 현재/다음 stack에는 독립 이동, **평가에는 증강 없음**. 범위/채널·오프셋 검사로 DrQ-v2의 `drq_v2.random_shift`가 하는 edge replicate와 의미가 같은지 확인하되 구현은 별도 학습 모듈에 둔다.
- 정책 좌표 `a_n=(steer,gas_n,brake_n) in [-1,1]^3`. `ActionAdapter.to_official`은 조향 그대로, pedal `(a_n+1)/2`로 공식 `[-1,1]x[0,1]^2`에 보낸다. `EpisodeCollector.step()`은 clip한 공식 실행값과 역변환한 `transition.action`(실행 native)을 제공한다. **Q/replay에는 실제 실행 native 행동**, 감사에는 정책 제안 native(pre-squash `u`와 squashed `a`), 실행 native 및 공식 `applied_action`을 구별하여 보존한다. 공식 pedal을 native로 다시 `step`하거나 teacher의 공식 행동을 이중 변환하지 않는다. 학생 단독 평가에는 교사, 지도 ID, 위치, 손상, progress, privileged `info`가 입력되지 않는다.
- 한 `EpisodeCollector.step()` = 학생 또는 교사 **환경 결정 1개**, `frame_skip=4`는 내부에서만 수행. 첫 비교는 DrQ 대조 실행과 같은 제안 조건인 학습 track IDs 1~4, 공식 장애물, `max_steps=2000`, `frame_skip=4`, raw reward(성형/정규화/추가 collision penalty 없음), 기본 무평활 행동을 동결한다. 로컬 기본값을 공식 evaluator의 내부 상세라고 주장하지 않는다.
- 전이 `(o_t, proposed_a_t, executed_native_a_t, applied_official_a_t, r_{t+1}, o_{t+1}, terminated, truncated, terminal, episode_id, step, track_id, geometry_seed, source)`를 원본 결과 시점에 기록한다. `Transition.done = terminated or truncated`는 reset 경계이고 `Transition.terminal`은 종료/완주/충돌성 retire를 포함하는 **부트스트랩 금지** 플래그다. 시간제한처럼 `truncated=True, terminal=False`면 최종 `o_{t+1}`에서 bootstrap하고 다음 reset `o_0`는 target으로 사용하지 않는다. 보상은 실제 환경 출력이며 `d_t=gamma*(1-terminal)`인 **1-step** target을 먼저 채택한다(DrQ-v2의 n-step=3과 다름).
- Pixel SAC가 tanh로 내보낸 유한 native 행동은 이론상 범위 내다. float32 경계 반올림, adapter clip, 향후 smoothing 등으로 `proposed_a != executed_native_a`가 생기면 그 차이를 축별로 계측하고 Q에는 실행값을 쓴다. **현재 정책 sample의 log probability를 실행/clip된 action의 log probability로 재평가하지 않는다.** SAC actor는 자기 분포에서 새로 생성한 action/log probability에만 의존하며 replay teacher action은 BC/importance weight 없이 Q 회귀 입력이다. 의미 있는 불일치가 생기면 정책과 물리의 분포가 다르므로 가정을 실패로 처리하고 데이터 생성 경로를 고친 뒤 새 protocol로 재시작한다.

## Torch 학습기의 정확한 손실

`o`는 증강한 현재 관측, `o'`는 별도 증강한 결과 관측이다. actor는 `mu_phi(o)`와 `log_sigma_phi(o)`를 내고 `u=mu+exp(log_sigma)*epsilon`, `epsilon~N(0,I)`, `a=tanh(u)`로 **native 3축** action을 샘플링한다. 저자 [TanhNormal](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/distributions/tanh_normal.py)처럼 `log_sigma`는 `[-20,2]` 범위를 제안한다. 정책의 정확한 density는

```text
log pi_phi(a|o) = sum_j [Normal(mu_j, sigma_j).log_prob(u_j)
                         - log(1 - tanh(u_j)^2)]
log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2*u))
```

이다. `u`를 유지한 안정형 Jacobian을 사용하고 `atanh(clamp(a))`로 경계 행동을 거꾸로 추정하지 않는다. 이 log probability와 target entropy는 **native 좌표** 기준이다. 공식 pedal 좌표의 density는 두 축의 affine Jacobian 때문에 달라지므로 공식 action과 native `log pi`를 섞지 않는다. 결정적 CPU action은 저자 [mode](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/distributions/tanh_transformed.py)의 `tanh(mu)`를 공식 행동으로 1회 변환한다. `tanh(mu+sigma*epsilon)`을 *Dreamer의 bounded-normal*이나 DrQ-v2의 deterministic actor+외부 Gaussian noise로 취급하지 않는다.

10개 critic `Q_i(o,a)`는 하나의 pixel encoder/latent trunk를 공유하되 **10개 독립 Q head**(각각 독립 초기화/hidden LayerNorm)로 계산한다. 별도 actor CNN은 업데이트 시작 시 critic CNN으로 복사하고 actor CNN에는 정책 gradient를 전파하지 않는 저자 경로를 우선 설계한다. Actor latent/MLP와 mean/log-std head에는 정상적으로 gradient가 흘러야 한다. Target에는 critic encoder와 10개 head의 독립 EMA 복사본을 둔다. 원본의 공유 범위와 복사 시점(critic 업데이트 *전* actor CNN 동기화), 피처 정규화, LN 위치, Q ensemble loss 평균을 PyTorch 포트 시험으로 명시한다. 메모리 최적화로 공유 참조 구조를 바꾸면 동등성 시험 없이 같은 설정이라고 부르지 않는다. optimizer는 저자 설정에 맞춰 Adam(기본 weight decay 0); 이미지 `/255` 외에 reward scaling, symlog, 자동 관측 정규화는 넣지 않는다. Q hidden의 LayerNorm은 DrQ encoder에 이미 있던 LN과 **별개**로 추가하며 저자의 Q 정규화가 이식되었다고 주장하기 전에 위치/gradient를 검사한다.

```text
J: 매 업데이트 독립 RNG로 {1,...,10}에서 균등 무복원 추첨, |J|=1
a' ~ pi_phi(.|o')
y = r + gamma * (1 - terminal) * min_{j in J} Q^-_j(o', a')   # no alpha/log pi
L_Q = mean_{i=1..10, batch} (Q_i(o, executed_native_a) - stopgrad(y))^2
a_new, logp_new ~ pi_phi(.|o)
L_actor = mean_batch [alpha.detach() * logp_new - mean_i Q_i(o, a_new)]
alpha = exp(beta) > 0
L_beta = mean_batch [exp(beta) * ((-logp_new).detach() - H_target)]
H_target = -3/2  # 저자 DrQLearner의 기본을 native 3축에 적용하는 제안
theta_target <- (1-tau)*theta_target + tau*theta_Q   # encoder 및 10개 head 전부
```

`L_actor`에서는 Q 파라미터를 고정하되 Q의 **action 입력을 통한 gradient**는 actor로 흘린다. `L_beta`는 actor에 역전파하지 않고 `beta` 자신의 optimizer만 갱신한다. 평상시 `backup_entropy=False`: target에 entropy 없음, 그러나 actor `alpha*log pi`와 alpha 학습은 유지한다. 선택적 entropy backup의 정확한 대비식은 `y_ent = r + gamma*(1-terminal)*(min_{j in J}Q^-_j(o',a') - alpha*log pi(a'|o'))`이다. 이것을 기본과 섞지 않는다. 저자 구현은 critic 업데이트 후 actor, 그 후 온도, critic 뒤 target EMA를 수행한다; 이 순서를 고정하고 초기 target은 critic과 동일 가중치로 만든다. UTD=1일 때는 유효 배치 하나로 각 1회 업데이트하며 10개의 head 업데이트를 UTD=10으로 세지 않는다. Q 최대값/ensemble 분산/target 오차, gradient·log-std·alpha·clip 비율을 기록해 유한성과 발산을 점검한다. 논문의 LayerNorm 분석이나 DMC 결과를 HAIC의 발산 방지/완주 증거로 바꾸지 않는다.

## 두 버퍼, 사전 자료와 전환

1. **교사 고정 및 학습용 자료 수집**: 제안 교사는 pad-4 DrQ control의 **원래 학습 seed 1** actor(`runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/checkpoints/step-000131072/actor.pt`; 장부의 actor SHA256 `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954`) 한 개다. 실행 전 actor/checkpoint와 source/config/protocol을 다시 해시·검증한다. 새 연구의 학습 track 1~4, 예약된 학습용 geometry seed에서만 frozen 교사로 주행해 이미지/실제 행동/raw reward/종료 사유를 새로 기록한다. 기존 screen/confirmation/blind **주행 로그나 평가 trajectory를 수입하지 않는다**. 연구자에게 허용된 사전 자료이지 대회 공식 오프라인 데이터가 아니다. 학습 시 학생은 혼자 제어하고 DrQ에는 추가 질의하지 않는다. 공유 교사 dataset hash를 두 학생 seed가 참조하면 seed가 둘이어도 교사 자료의 독립 복제는 하나다. 교사 131,072-decision 원본 훈련의 `episodes.jsonl` 및 reset sampler 원장을 검사하여 **새 평가 geometry가 교사의 과거 훈련 geometry와도 교차하지 않게** 해야 한다. DrQ teacher-replay와 동시에 계획할 경우 두 DrQ 원본 source의 원장 모두와 교차 검증한다.
2. **불변 offline store**: 에피소드/프레임/seed/track/source/교사 hash가 검증된 후 봉인하고 학습 중 insert/overwrite하지 않는다. train/evaluation geometry를 seed 단위로 교차 배제한다. 학생 online buffer는 별도 ring으로 생성하며 학생 무작위 warmup 및 정책 자료만 저장한다. 가득 찬 ring에서도 offline 50%는 유지한다. Batch 제안값 `64 = offline 32 + online 32`이고 SAC 대조군은 `64 online`; 실제 source별 정수 개수와 unique episode/geometry를 기록한다. 작은 offline store에서 복원추출은 가능하되 추첨 편향/중복율을 로깅하고 두 buffer 모두 terminal-safe valid index만 사용한다.
3. **프레임 메모리**: `drq_v2.Uint8Replay`의 아이디어를 채용하되 해당 DrQ n-step sampler나 checkpoint format을 그대로 SAC의 1-step/offline 50:50 sampler라고 가정하지 않는다. 전이당 최신 회색 frame `84*84=7,056` bytes를 저장하고 해당 episode 첫 frame으로 stack 시작을 pad한다. 본래의 4-channel 결과 stack에서 종료/절단 경계 frame은 full stack으로 별도 저장하고 ring overwrite 시 함께 삭제한다. 100,000 online 최신 frame의 예약량은 **705,600,000 bytes (약 673 MiB)**, offline 최대 8,192/16,384 frame은 각각 **57,802,752/115,605,504 bytes (약 55/110 MiB)**이며 메타데이터, 경계 full stack(1개당 28,224 bytes), batch, 저장 복제본, CUDA 활성화는 추가다. 전체/두 장의 float stack을 전이별 상주시켜 CPU RAM을 급증시키지 않는다. ring 최초 3개가 덮여 필요한 prefix를 잃으면 유효하지 않은 row로 거절한다. `o_{t+1}`는 연속 전이의 다음 최신 frame에서 복원하고, 경계는 봉인된 실제 결과 stack을 사용한다. 구성 메모리와 GPU OOM/RSS를 초기 gate에서 계측하고 상한에 맞춰 **실험 전** 용량을 동결한다.
4. **온라인 출발과 takeover**: 저자 pixel 원형은 offline-only pretraining이 없다. 여기서는 proposed warmup `2,000` **학생 online 결정** 중 첫 `1,000`은 순수 무작위 수집, 그 뒤에도 `2,000`까지 무작위 제어를 유지하면서 online/offline 혼합 learner 업데이트 `1회/결정`을 수행해 정책을 준비한다. `2,000` 이후 학생의 재매개 샘플만 제어에 사용한다. 원형의 시작 시점보다 늦은 takeover는 **HAIC 전용 제안이며 별도의 적응 차이**로 기록한다. 동결 이후 각 군의 기대 학습 업데이트 수는 16,384 결정 파일럿에 **15,384**, 131,072 결정 full에 **130,072**(batch 부족/비유한 실패 시 조용히 건너뛰지 않고 중단)이다. SAC 대조군도 동일 무작위 episode, learner 시작/전환 시점 및 업데이트 횟수(전부 online)를 사용한다. 첫 정책 행동 전 actor/alpha/Q 유한성과 CPU 행동 범위 검사를 수행한다. episode 중 전환은 feed-forward 정책이므로 recurrent carry는 없지만 `episode_id/step`, 실제 실행 행동, source를 계속 기록한다. 위험 상태라고 offline-only 100~500 업데이트/BC를 몰래 추가하지 않는다. 초기 붕괴 시 원인 분석 후 별도 protocol에서 **양쪽 arm에 동일한** 추가 replay-only 업데이트 등의 처리로 다시 시험한다.

## 공정한 연구 설계와 제안 예산

1차 **인과적으로 비교 가능한 축**은 `online-only SAC (RLPD와 동일한 10-Q/LayerNorm/actor/증강/alpha/target 설정)` 대 `RLPD = 동일 SAC + frozen offline dataset 50:50`이다. 그러므로 결과명은 단순 '기존 표준 2-Q SAC 대비 RLPD'가 아니라 **'동일 pixel SAC의 prior-data 처리 효과'**다. 대조군은 교사 dataset을 로드/학습하지 않는다. 두 arm은 각각 학생 seed `0,1`(제안), 초기 가중치/학생 RNG 정책, 학습 track/geometry 샘플러의 episode-index 스트림, batch 64, online 결정 수, 유효 critic/actor/temperature update 횟수, checkpoint 시점, CPU 런타임, 화면 선택 기준을 맞춘다. 에피소드 길이가 달라 실제 방문 상태·소비된 geometry는 달라질 수 있으므로 같은 episode-index seed가 곧 같은 주행 데이터는 아니다. 오프라인 가용 여부와 그 결과 replay 분포의 변화가 **바로 treatment**다.

| 제안 단계 (이번 지시는 로컬 계획의 사전 gate 종속 실행을 승인) | 학생 online 예산/seed/체크포인트 | 별도 DrQ 교사 자료·평가 | 다음 단계의 사전 gate |
|---|---|---|---|
| 합성/CPU/GPU correctness | 실제 환경 결정 **0**; 합성 데이터로만 테스트 | 실제 교사 수집 없음 | 아래 A gate 전부와 CUDA/CPU export/restart 시험 완료 |
| 파일럿 (새 study/protocol) | arm당 seed 0,1 각각 **16,384 학생 결정**, warmup 2,000 포함; **8,192 / 16,384** CPU export screen 후보 | 연구 전체에 상한 **8,192 학습용 교사 결정**을 사전 고정; 완료 episode 중 **서로 다른 geometry에서 최소 2회 완주**를 자료 적격성 기준으로 동결. 별도 fresh 파일럿 screen 제안 `3 track IDs x 4 geometry = 12 canonical cells/actor`, 각 2 CPU reload. 필수 confirmation/blind grid는 사전에 예약하지만 파일럿에서는 열지 않음 | 자료 적격성 + 두 학생 seed 모두 정책 제어 구간/선택된 actor가 finite, 자원·재적재 통과, **각 RLPD seed가 적어도 1개 실제 screen 완주**, 각 seed의 RLPD 완주가 대응 SAC보다 적지 않음. 미달·불충분이면 정지/보류, 임의 수치 재조정 금지 |
| 조건부 full (새 연구·새 protocol, 처음부터 재훈련) | arm당 seed 0,1 각각 **131,072 학생 결정**, warmup 포함; **65,536 / 131,072** CPU export screen 후보 | 연구 전체에 별도 hash의 상한 **16,384 학습용 교사 결정**, **서로 다른 geometry에서 최소 4회 완주**를 자료 적격성 기준으로 동결(파일럿 자료를 fresh로 재라벨하지 않음); 새 screen `3x8=24`, confirmation `4x8=32`, 예약 blind `3x8=24` canonical cells/actor, 각 2 reload 제안 | 파일럿 게이트 + 별도 실행 결정 + 새 연구의 source/config/분할/예산 사전 동결 뒤에만 착수; confirmation/blind 승격은 아래 규칙대로 |

파일럿 수치, full 규모 및 셀 수는 **사전 제안**이지 저자 HAIC 권장 설정/확정 seed 목록이 아니다. 새 셀의 실제 `(track_id, uint32 geometry_seed, obstacle mode)`는 실행 전에 과거 run/evaluation/예약 기록과 **교사 원본의 훈련 episode 원장**에 겹치지 않는지 감사하고 다른 연구의 blind를 침범하지 않게 지정한다. `reserved_training_seeds`의 모든 geometry는 **학습 ID 전체**에서 학생/교사 수집에 금지한다. 기존 DrQ promotion confirmation `111-114 x 31101-31108`, L2 `211-214 x 32101-32108`, pad `311-314 x 33101-33108`은 이미 소비됐고 관련 `121-123 x 31201-31208`, `221-223 x 32201-32208`, `321-323 x 33201-33208` blind는 해당 연구용으로 예약되어 있다(`docs/evaluation/generalization-policy.md`). 새 grid라 해도 전역 미사용이 증명된 것은 아니며 사용 기록 누락 발견 시 신선도 주장과 gate를 철회한다. 교사 수집 한도가 episode 중간에 차면 **가짜 terminal을 만들지 않고** 부분 episode를 버리되 소모한 결정을 비용에 포함한다. 적격 완주가 부족해도 같은 연구 안에서 교사 상한이나 학습 셀을 사후 확대하지 않는다; 결과는 `교사 자료 불충분`으로 남긴다.

학생 online 예산은 두 arm이 같지만 RLPD의 **추가 교사 환경 결정 상한 8,192/16,384**와 선행 DrQ 교사 학습(선택된 control run의 **131,072 결정**), 교사 선정/수집 시간과 저장, 총 gradient/FLOP/벽시계 시간을 별도 원장에 계산한다. 같은 CPU 평가 셀에서 DrQ control actor(학습 비용과 모델 hash 고정)를 나란히 평가하는 것은 유용한 **성능 대조**이나, 교사에서 배운 RLPD vs 교사 본인 또는 SAC vs DrQ는 동일 자료/총 환경 비용의 알고리즘-only 매칭이 아니다. 추가 자원까지 맞추고 싶다면 SAC에 동등한 online 상호작용을 별도 부여하는 **다른 budget-normalized 연구**로 사전 동결해야 하며 1차 one-axis 결과와 혼동하지 않는다.

## 평가 분할, 선택, 승격 및 중단

1. **첫 실제 수집/학습 전에** pilot 연구의 수정 완료 소스·의존성·환경 contract/해시, 교사 actor/data 수집 계획, 알고리즘 버전, 두 arm/seed와 고정 RNG, CPU runtime, 정확한 training 제외 집합, screen/confirmation/blind 전체 grid, 제외·중단·선택 규칙을 `experiments/`의 **불변 study JSON**에 동결한다. 아직 존재하지 않는 학생 actor hash는 넣지 않고 첫 export 후 영수증에 기록한다. full은 파일럿 screen을 재활용하지 않고 다른 source/config hash와 fresh 분할을 가진 **새 protocol**을 첫 full 수집 전 별도 동결한다. 학습용 사전 자료 재사용 여부/중복 제한도 full protocol에서 미리 결정한다.
   학습 재현성은 Git HEAD만으로 대체하지 않는다. `haic/algorithms/rlpd/`, `scripts/`의 해당 collector/trainer, `common_adapter.py`, 환경 `core/`/wrapper, `agent.py`, `evaluate_policy.py`, `package_submission.py` 및 잠금 의존성의 실제 사용 bytes/hash를 protocol/run snapshot에 남긴다. dirty worktree면 source별 해시와 해당 소스 사본을 봉인하고 비관련 변경을 학습 코드에 섞지 않는다. Offline dataset은 episode manifest와 content SHA, 교사 actor/export SHA, 수집 source/geometry 목록이 일치해야 소비한다. `runs/`의 checkpoint/export/episode 영수증, `evaluations/`의 runtime/actor snapshot, `experiments/`의 JSON 및 결과 영수증을 각자 자기 경로에 보관한다.
2. `evaluate_policy.py`의 **custom `--protocol-file`**로 학생 CPU export를 모델별 screen에 평가한다. 현재 `.pt` 형식은 DrQ/Dreamer만 인식하고 `.pt`면 기본적으로 `drq-v2`라고 표기하며, 숫자형 CPU latency/RSS 적격성 검사는 `drq-v2`에서만 켜진다. RLPD 태그와 알고리즘 구별, 정책·source hash, CPU whole-worker gate(`init <=10 s`, `reset <=5 s`, `act <=5 s`, `VmHWM <=1,024 MiB`), 평가 런타임 snapshot/의존성 검사를 먼저 늘리고 관련 테스트를 통과시킨다. `.pt` 평가에 legacy `checkpoint-v1-*` 사용 금지. 두 reload는 결정성 감사이지 독립 road 2개가 아니며 `repeat == 0`만 집계한다.
3. 각 arm/seed의 두 시점 중 eligible + 두 reload 일치 + operational fail 0인 checkpoint를 **screen 완주율 -> 진행률 -> 완주 lap time이 낮음 -> 완전 동률은 이전 checkpoint** 순서로 미리 정해 고른다. 모든 4개 student run과 screen이 끝나기 전에는 treatment finalist를 확정하지 않는다. 순수 RLPD finalist **한 actor**도 screen 정보만으로 사전 선택한다. 선정 모델의 CPU actor SHA, learner checkpoint SHA, 학습/source/protocol/environment SHA, 화면 영수증을 confirmation 전에 봉인한다. confirmation은 후보 재선택, 추가 학습 또는 threshold 수정 단계가 아니다.
4. Full screen에서 RLPD 두 seed 모두 진짜 완주와 자원 게이트를 만족할 때만 frozen 두 seed의 RLPD와 짝지은 online-only SAC를 **같은 새로운 confirmation cell**에 평가한다. 현행 `evaluate_policy.previous_evaluation_metadata()`는 screen 완주 0 actor의 보통 confirmation을 거절한다. SAC/DrQ comparator가 screen 완주 0이면 현행 `--diagnostic-confirmation`으로만 해당 고정 모델의 confirmation 자료를 수집하고 그것은 **비승격 comparator 영수증**으로 명시한다. 그런 대조군 결과로 그 모델의 blind를 열 수 없으며 treatment의 normal confirmation 영수증과 섞어 재표기하지 않는다. frozen DrQ actor도 동일 protocol/새 셀로 모델별 screen 영수증을 만든 후, 필요 시 같은 규칙으로 confirmation을 실행한다. 현행 evaluator는 confirmation에 모델 **1개**와 그 모델의 single-candidate screen receipt를 요구하므로 각 모델 영수증을 따로 연결한다.
5. **제안 promotion gate**: 각 독립 학생 seed에서 RLPD의 confirmation canonical 완주 수가 짝 SAC보다 **엄격히 많고** 적어도 1회 완주, 모든 arm 자원·결정성 검사 통과, screen에서 봉인한 finalist에 대한 선택 편향이 없을 때만 '동일 SAC에서 prior-data가 도움된 내부 후보'로 기록한다. 차이·불확실성은 `track_id, geometry_seed`별 paired 결과와 seed별 분포로 제시한다. 2 seed만으로 광범위한 효과의 확정적 통계 증거라고 주장하지 않는다. 비교 우위가 한 seed에만 있거나 0 finish이면 실패/판정 불충분으로 정지한다. 사전 확인 gate 통과 후 **같은 단일 finalist**에 대해서만 예약한 그 연구의 blind를 1회 열고 사전 동결 기준으로 최종 내부 일반화 판단; blind로 모델을 바꾸거나 재훈련하지 않는다. DrQ 대비 차이는 별도 보고하며 교사/학습예산 confound를 명시한다.
6. Raw CarRacing reward, progress, damage, 조향 변화/경계 비율과 무작위 shift 민감도는 내부 진단 proxy이다. 완주 우선/미완주 progress·lap time의 순서는 protocol에서 고정하고, 장애물 variant가 같은 geometry면 독립 학습 seed/도로로 세지 않는다. 공식 private track은 접근 가능한 holdout이 아니다. 공식 public 결과나 사이트 모델 확인은 이 문서의 gate가 아니며 **실행 직전 별도 명시적 사용자 승인 및 최신 공식 규칙 재확인**이 필요하다.

## 구현 시 파일 지도 (이 문서에서는 만들거나 변경하지 않음)

| 미래 파일/경계 | 역할과 누출 방지 |
|---|---|
| `haic/algorithms/rlpd/{__init__,model,agent,replay}.py` | Torch tanh-normal actor/10-Q critic/temperature·EMA, 불변 offline 및 ring online source-balanced sampler; `common_adapter`의 관측/실행 native 전이 contract를 사용한다. 추론용 module을 훈련 module에서 import하지 않는다. |
| `scripts/collect_rlpd_prior.py`, `scripts/train_rlpd.py`, 필요 시 `scripts/run_rlpd_matched.py` | frozen 교사만 학습용 cell에 호출, 두 arm 별 protocol/예산 검증, run config/selection/재시작 영수증 작성; repo root에서 `python -m scripts.collect_rlpd_prior`, `python -m scripts.train_rlpd` 등으로 호출한다. 학생 훈련용 새 root Python 파일을 만들지 않는다. |
| `agent.py` | 작은 **추론 전용** `haic-rlpd-pixel-actor-v1` 태그의 `(4,84,84) -> tanh(mu) -> [steer,gas,brake]` 분기, `torch.load(..., map_location="cpu", weights_only=True)`, 태그·spec·config·state shape/finite 확인, 무상태 `reset`. 기존 baseline/DrQ/Dreamer dispatch 보존; 허용 inference-only 구현만 패키지에 들어간다. |
| `evaluate_policy.py`, `tests/test_evaluate_policy.py` | RLPD 태그/알고리즘 provenance 지원, custom protocol 요구, `.pt` 공통 전역 적격성/isolated CPU reload, 반복 행동 trace와 자원 수치·hash 검사. 0-screen comparator는 기존 진단 확인 규칙을 보존하고 blind/promotion에 쓰지 않는다. Snapshot에 새 inference module이 꼭 필요하다면 source 포함·hash 감사와 패키저 지원을 선행한다. |
| `package_submission.py`, `tests/test_submission_package.py` | 현행 packager는 root의 `agent.py`, `model.pt`와 인지된 `action_smoothing.py`, `action_representation.py`만 포함하고 추가 trainer module/임의 requirements는 자동으로 싣지 않는다. RLPD actor-only export가 이 root-only 경로를 통과하는지 시험하고 tag/export provenance, 금지 API·용량·CPU ZIP smoke의 빠진 개별 한계를 보강한다. 새 inference module이 필요하면 기존 packager에 명시적 허용/정적 검사 추가 후 시험하며 원본 JAX 의존성은 넣지 않는다. |
| `tests/test_rlpd.py`, `tests/test_train_rlpd.py`, `tests/test_agent_inference.py` | 다음 합성 수학·버퍼·RNG/CPU export 테스트와 기존 inference 회귀. frozen protocol/result는 각각 `experiments/`, `runs/`, `evaluations/`의 기존 자리에; 기록이 생기기 전에는 문서 상태나 모델 장부를 바꾸지 않는다. |

## 검증 게이트 (실행 전 작성할 테스트 명세)

| Gate | 재현 가능한 실패 신호와 필요 증거 |
|---|---|
| A1: 수학/파라미터 | 고정 `u, mu, sigma`의 수동 Normal+Jacobian logp와 Torch 일치, 극단 `|u|`에서도 finite 및 mu/log-std 정책 gradient 0 아님. alpha=0/1, `backup_entropy=False/True`의 합성 TD target 차이를 검사하고 기본 false에서 target alpha gradient 없음. terminal target=`r`, time-limit truncation target=`r+gamma*Q(next_final)` 및 reset frame 혼입 0. `num_min_qs=1`의 무작위 index가 여러 head를 방문, 10개 Q 모두 업데이트, 평균 actor Q 사용, EMA 극단 tau=0/1, LayerNorm과 actor CNN gradient stop 검사. |
| A2: 이미지/replay | 4개 채널에 동일 shift와 `pad=4` 범위, current/next 별도 RNG, eval 무증강. 한 episode 첫 frame pad, 현재/다음 stack, 마지막 실제 결과 frame, `terminated`/`truncated` 경계, ring overwrite와 유효 index의 의미를 유일 픽셀/보상 marker로 시험. Offline store 불변과 online ring의 source 혼합 `32:32`, 빈 버퍼 시 **조용한 offline-only 업데이트 금지**. geometry seed 중복/예약 유출, **원본 교사 학습 주행과 새 평가 geometry의 교차**, teacher/action 좌표·hash 불일치 거절. |
| A3: 제어/CPU | 합성 관측에 대해 Torch 학습 actor의 `tanh(mu)`와 새 CPU export 및 `Agent`가 만든 native/공식 행동 parity(축별 허용오차 사전 고정), reset/재적재 2회 동일 trace. 경계/비유한/잘못된 spec/checkpoint는 fail closed. `.pt` evaluator가 RLPD를 DrQ로 오표기하지 않고 CPU 적격성 검사, protocol 해시 및 `repeat==0`만 집계. CPU-only Python 3.11/Torch 2.1.0 clean load와 **전체 evaluator/패키지**의 init/reset/act 개별 시간 및 RSS 계측. |
| A4: CUDA/복구 | 실제 사용 GPU에서 학습 1업데이트 전후 finite loss/10-Q gradient/메모리·속도, same-seed CPU export 일치(다른 커널의 GPU↔CPU bitwise 동등성은 가정하지 않음). Checkpoint에는 actor, 전체 Q/target, alpha, 세 optimizer, 두 replay 및 유효 index/경계 frame, offline dataset hash, 환경 결정·업데이트 수, Python/NumPy/Torch/CUDA/증강/샘플러 RNG, 학습 에피소드 prefix/실행 공식 행동과 누적 reward를 저장. 독립 복제에서 저장/재적재 후 다음 배치 index/target/행동 및 연속 실행 결과를 비교하고 살아 있는 학습기의 RNG·정책은 검증으로 변하지 않아야 한다. 물리 상태 직접 직렬화가 없으므로 중간 episode는 동일 reset+실행 행동 prefix 재생·관측/보상/종료 parity가 입증될 때만 **정확 재시작**; 불일치하면 checkpoint 이후 다음 **새 episode 경계 재시작** 또는 run 무효로 기록하고 동일 실행의 연속이라고 주장하지 않는다. |
| B: 학생/패키지 | 프로토콜 동결 후에만 제안 파일럿 수집/학습·CPU screen; 안정성/실제 완주로만 full 진입 결정. ZIP root `agent.py`, `model.pt`와 필요한 허용 Python만 포함, teacher/critic/replay/CUDA 파일 없음. 로컬 ZIP 검사는 공식 수락이 아니며 사이트 업로드는 금지 상태. |

공식 제약의 로컬 전사에 따라 ZIP 압축 크기 `<=500 MB`, `<=1,000`파일, 풀린 총량 `<=2 GB`, 단일 파일 `<=500 MB`, 단일 파일 압축비 `<=100x`를 각각 확인한다. ZIP에 필요한 모델을 전부 넣고 실행 중 인터넷에 의존하지 않는다. 모든 제출 `.py`에서 금지 import(`ctypes`, `importlib`, `multiprocessing`, `os`, `pathlib`, `resource`, `shutil`, `signal`, `socket`, `subprocess`, `sys`), 금지 동적 호출(`compile`, `eval`, `exec`, `__import__`) 및 금지 실행/네이티브 확장자를 검사한다. 추가 추론 의존성이 꼭 필요하면 공식 PyPI의 Linux/Python 3.11 바이너리 wheel과 root `requirements.txt`의 정확한 pin, 설치 한도를 공식 원문으로 재검증한다. ROOT 패키저의 합산 smoke는 각 호출의 제한이나 whole-worker RSS를 스스로 증명하지 못한다(`docs/workflows/prepare-submission.md`). 로컬 게이트는 제출 권한도 서버 수락도 아니다.

금지/중단: official environment(`core/`, `env_wrapper.py`, `damage.py`)를 성능 때문에 고치지 않는다; 확인/예약 blind를 개발 자료로 되돌리지 않는다; teacher가 평가 셀에서 운전한 결과를 학생 완주라 하지 않는다; CUDA OOM, NaN/Inf, terminal leakage, clipping 불일치, 잘못된 alpha/target, CPU 계약 실패, source/hash/분할 불일치가 나면 학습량 증대로 덮지 않고 단계 A로 돌아간다. 실패한 결과는 성공한 새 checkpoint로 사후 대체하지 않고 해당 연구의 불일치/거절로 봉인한다. 내부 연구 기록이 실제 생길 때만 `docs/experiments/INDEX.md`와 현재 상태/해당 계획의 출처를 갱신하고 `docs/results/MODEL_STATUS.md`는 실제 후보 상태 전이 시에만 갱신한다.

## 실행 결과: 파일럿 v2 (2026-09-24)

- v1 protocol은 collector import 이후 dependency inventory가 protocol lock과
  다르다는 사전 gate에서 중단되었다. `experiments/pixel-rlpd-offpolicy-pilot-v1-preflight.json`
  에 zero teacher/student/evaluation decisions를 기록했다. 이를 숨기거나 기존
  geometry를 소비된 셀로 표기하지 않고, v2에서 새 seed allocation과 안정된
  import-order runtime hash를 동결했다.
- v2 teacher dataset은 고정 8,192 decisions 중 8,170 transitions, 19 complete
  episodes, 3 distinct-geometry finishes를 기록하여 최소 2 finish gate를 넘었다.
  동일 pixel SAC/RLPD 4개 run 각각 16,384 decisions 및 15,384 updates를 완료했다.
- v2 screen은 동일한 새 12-cell grid, 2 repeat/actor, CPU Python 3.11/Torch
  2.1.0+cpu에서 8 checkpoint 전체 192 evaluations/96 canonical episodes를
  수행했다. All candidate reload/determinism/resource gates passed, but exactly
  one canonical finish occurred: RLPD seed 1 at 8,192 decisions. RLPD seed 0의
  선택된 8,192 actor는 0/12이고, 짝 SAC seed 0도 0/12였다. Seed 1은 RLPD 1/12,
  SAC 0/12였다.
- 따라서 이 계획의 **사전 pilot gate는 실패/stop-hold**다. 이 수치는 작은 내부
  CarRacing screen의 기술 결과이며 공식 점수나 두 seed로 재현된 향상 증거가
  아니다. v2 full, confirmation, blind cells는 실행하지 않았고 계속 닫혀 있다.
  평가 manifest와 selection 결과는 `experiments/pixel-rlpd-offpolicy-pilot-v2-result.json`
  및 `evaluations/20260924T184536689584Z_pixel-rlpd-offpolicy-pilot-v2-screen/`에
  고정되어 있다.
- v2 run orchestrator가 post-evaluation candidate metadata의 step field 위치를
  잘못 가정하는 harness bug를 selection 도중 발견했다. 공식 evaluator 결과에는
  영향이 없었고 metrics는 frozen per-run candidate index와 평가 receipt를
  사용해 이 계획의 checkpoint 순서대로 재계산했다. 이 harness는 후속 source
  revision에서 시험을 보강한다.
- 사용자는 결과 피드백/재학습까지의 반복을 명시적으로 요청했다. v2의 실패는
  v2 범위 내 학습 확대나 holdout 개방을 허가하지 않는다. 이어지는 연구는
  짧은 예산과 작은 prior data가 불안정한 내부 완주의 **가능한 원인이라는
  가설**을 별도로 시험하는 새 protocol이다. 다른 teacher geometry, 131,072
  matched student decisions, 16,384 fresh prior decisions 및 완전히 새 screen/
  confirmation/blind allocation을 동결하고, v2 data와 v2 screen/예약 seed를
  재사용하지 않는다. 온라인 예산과 prior-data 예산을 함께 바꾸므로 원인 하나를
  분리하는 비교가 아니라 fuller recipe의 새로운 내부 시험으로 기록한다.

## 실행 결과: Long-Horizon Follow-up v1

- 별도 source-hashed protocol은
  `experiments/pixel-rlpd-long-horizon-followup-v1.json` (SHA-256
  `2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329`)이다. 이
  freeze 당시 이 계획의 SHA-256은
  `bca60b19ede9a8d2de76edd6bd47a2c6c158aa1c1193f0889f38b52ea955d86c`; 해당 바이트는
  `runs/20260924-pixel-rlpd-long-horizon-followup-v1/source/docs/plans/`에 보존했다.
- 새 teacher data는 16,384 decisions를 사용해 16,305 transitions/36 episodes/
  6 distinct-geometry finishes를 저장했다. RLPD/SAC에서 learner seed 10,11의
  네 run은 각각 131,072 decisions, 130,072 updates를 완료했다.
- 24 canonical screen cell/actor에서 seed 10 RLPD/SAC가 7/24 대 3/24, seed 11이
  12/24 대 3/24를 기록해 두 screen gate를 통과했다. 네 confirmation actor도
  32 canonical cells에서 RLPD/SAC seed 10은 3/32 대 2/32, seed 11은 12/32 대
  7/32를 기록해 두 strict matched gate를 통과했다.
- Screen 순위만으로 사전 선택한 RLPD seed-11, 131,072-step actor
  (`f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1`)는 한 번의
  final internal blind에서 9/24 canonical finishes/0.679 mean progress를 얻었다.
  이 결과는 내부 CarRacing proxy이지 공식 HAIC 점수/경기/제출이 아니다. 확인,
  blind, 재적재, 결정성, 자원 결과는
  `experiments/pixel-rlpd-long-horizon-followup-v1-result.json` 및 연결된 CPU21
  evaluation artifact를 따른다.
- 최초 confirmation runner는 8후보 screen pointer를 단일후보 preceding-receipt로
  전달해 validation에서 멈췄다. 감사는 confirmation/blind cell 0회를 확인했다.
  Original screen의 정확한 per-actor summary/archive/hash를 selection projection으로
  묶고 evaluator lineage preflight를 통과한 뒤, 미소비 confirmation 4개 actor와
  사전 선택 blind 1개 actor만 실행했다. Screen, protocol, learner가 변경되거나
  재실행되지는 않았다. 완료 뒤 duplicate recovery 호출은 existing receipt guard로
  차단되어 환경 cell을 추가 소비하지 않았다.
- RLPD seed 11, step 131,072를 최고 현지 내부 candidate로 기록하되 official
  model status와 혼동하지 않는다. 다음 separate hypothesis는 author-faithful
  `target_entropy=-1.5`와 `+1.5`를 비교하는 한 가지 변인만의 entropy ablation이다.
  이를 위해 새로운 teacher data, source/protocol hash, student seed, screen,
  confirmation, blind allocation을 사용하고 이 v1의 data/model/evaluation cell은
  재사용하지 않는다.
- 첫 entropy ablation protocol v1은
  `experiments/pixel-rlpd-entropy-target-ablation-v1.json` (SHA-256
  `cbbe3039d8bec0564d3673f3e920540348c1e6bf5ae9a419a2c641e0b3a160a9`)로 freeze됐지만,
  helper가 fixed `decisions/minimum_finishes` 외 accounting metadata를 거부해
  `read_entropy_protocol` preflight에서 중단됐다. Teacher/student decisions 및
  모든 geometry interaction은 0; record는
  `experiments/pixel-rlpd-entropy-target-ablation-v1-preflight.json`이다. V1
  training/screen/confirmation/blind allocation은 미소비여도 retire되었고 이후
  새 split에 재사용하지 않는다. Exact source snapshot은
  `runs/20260925-pixel-rlpd-entropy-target-ablation-v1-preflight/source/`에 있다.
- Entropy protocol v2의 filename/execution template는 `v2`였지만 내부 study name은
  `v1`로 남았다. Source/protocol validation이 환경 생성 전 이를 거절했다. 이 역시
  0 decisions/cells로 기록해 retired allocation으로 두고
   `experiments/pixel-rlpd-entropy-target-ablation-v2-preflight.json`에 봉인했다.
   두 preflight seed allocations 모두 소비되지 않았지만 재활용하지 않는다.
- Entropy protocol v3 (`experiments/pixel-rlpd-entropy-target-ablation-v3.json`,
  SHA-256 `541b86c8726efdb4c5c9c4c445ec6c0313e3f7ee6e32e5a5f53a2d7920a02f8f`)도
  learner seed label이 reader와 다르고 source hash가 drift하여 `read_entropy_protocol`
  에서 거절됐다. Training `4000026001-4000026064`, screen `4000027001-7008`,
  confirmation `4000027011-7018`, blind `4000027021-7028` 모두 0 interaction으로
  retired; record는 `experiments/pixel-rlpd-entropy-target-ablation-v3-preflight.json`.
- 다음 coherent entropy protocol v4는 correction된 validator, 새 study name/source
  hash와 새 candidate geometry를 독립 freshness audit 후에만 동결한다.
  One-factor design은 동일한 새로운 teacher dataset, learner initialization,
  architecture, offline/online mix, update budget 및 replay/augmentation RNG stream
  아래 RLPD target `-1.5`와 `+1.5`를 비교한다. `+1.5`는 실험할 higher
  differential-entropy target이지 저자의 부호 오류라는 가정이나 V3 결과에서
  유도한 causal fix가 아니다. Entropy v1/v2/v3 retired allocations, V1/V2 pixel-
  pilot geometry, v3 follow-up data/actors/cells를 모두 training/evaluation
  exclusion에 포함하고, blind는 confirmation rule에 따른 한 actor에게만 허용한다.

## 참고 및 실증 경계

- [논문: Ball et al., *Efficient Online Reinforcement Learning with Offline Data*, arXiv:2302.02948v4](https://arxiv.org/html/2302.02948v4) (4.1/4.2/4.3/4.5, pixel 실험, 한계)
- [저자 공개 구현 고정 revision](https://github.com/ikostrikov/rlpd/tree/c90fd4baf28c9c9ef40a81460a2e395092844f88), [pixel config](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/configs/rlpd_pixels_config.py), [pixel trainer](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/train_finetuning_pixels.py), [pixel learner](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/drq/drq_learner.py), [SAC 손실](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/sac/sac_learner.py)
- [이 저장소의 DrQ 데이터·증강·export](../../drq_v2.py), [실제 실행/재개 경계](../../train_drqv2.py), [공유 전이/행동](../../common_adapter.py), [추론 분기](../../agent.py), [평가·영수증](../../evaluate_policy.py), [패키지 검사](../../package_submission.py), [평가 원칙](../evaluation/protocol.md), [공식 제약 로컬 전사](../competition/restrictions.md)

이 계획은 소스 근거 설계이며 2026-09-24 요청으로 bounded local execution을 시작했다. 저자 논문의 V-D4RL 수치, DrQ의 과거 확인 결과 또는 합성·CPU 시험 통과 어느 것도 새로운 RLPD 학생의 HAIC 완주/공식 점수로 기술하지 않는다.
