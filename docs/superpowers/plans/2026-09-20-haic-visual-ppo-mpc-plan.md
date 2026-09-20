# Implementation Plan: HAIC 시각 PPO와 학습 기반 행동 계획기

> 승인된 설계: docs/superpowers/specs/2026-09-20-haic-visual-model-based-rl-design.md
> 구현 전 계획 검토용 문서입니다. 검토와 실행 방식 선택 전에는 구현을 시작하지 않습니다.

**목표:** 84×84 흑백 프레임 네 장만 받는 에이전트에 PPO 시각 정책을 학습시키고, 학습된 잠재 전이 모델을 사용해 각 행동 호출에서 제한시간 내 CEM 계획을 수행한다. 주행 행동은 학습된 정책과 예측 모델이 정하고, 규칙 코드는 유효성 검사와 실패 시 fallback만 맡는다.

**구조:** 수집기는 로컬 시뮬레이터 상태를 보조 학습 레이블로만 읽는다. 정책과 계획기는 프레임 스택만 본다. 프레임 전체 CNN에는 고정 위치 HUD 막대 영역을 처리하는 시각 branch를 결합한다. PPO actor-critic은 즉시 행동과 행동 사전분포를 내고, 작은 앙상블 전이 모델과 CEM은 짧은 미래 행동열을 평가한다. agent.py는 평가 환경의 진입점으로 남긴다.

**런타임:** Python 3.11, PyTorch CPU 추론, 추가 런타임 패키지 없음. 각 act 호출의 전체 예산은 입력 검증, encoder, actor, 후보 생성, dynamics 및 CEM 계산을 포함해 기본 4.5초다. 호출 시작 시 단조 시계 deadline을 잡고 bounded batch 크기로 남은 시간 내 중단한다. 빠른 반복 평가에는 더 작은 예산을 지정할 수 있지만, 최종 보고에는 기본 설정 결과와 전체 에피소드 시간도 포함한다.

## 적용 제약과 설계 결정

- 공식 인터페이스는 Agent(), reset(observation), act(observation)이다. 관측은 float32 (4, 84, 84), 행동은 유한한 [steer, gas, brake] 세 값이다.
- 학습기만 진행률, 충돌, 손상, 차량 속도·방향 등의 시뮬레이터 레이블을 읽는다. agent.py와 제출 패키지의 추론 경로에서는 트랙 ID, 시드, 지도, 보상, 충돌 상태, 원시 물리 상태에 접근하지 않는다.
- 한 행동이 기본 4개 원본 physics tick 동안 유지된다. 전이 학습, 보상, 계획 horizon은 이 decision cadence에 맞춘다.
- HUD crop은 전처리 후 84×84 좌표에서 자른다. 전체 이미지 branch를 항상 유지하고, HUD branch가 보조 상태 추정 검증을 통과하지 못하면 정책의 필수 입력으로 의존하지 않는다.
- 개발용 .venv에는 Windows에서 설치되는 Box2D 2.3.10 wheel이 설치되어 있다. 저장소의 공식 requirements.txt는 평가 환경과의 호환성을 위해 바꾸지 않는다.
- 평가 순위에 맞춰 체크포인트를 완주율, 완주 랩타임, 미완주 진행률 순으로 고른다. PPO 단독과 PPO+CEM을 같은 검증 에피소드에서 비교한다.
- 규칙 기반 정상 주행, 평가 시뮬레이터 상태 읽기, 평가 중 인터넷, 검증되지 않은 큰 탐색 모델은 범위 밖이다.

## 검토가 필요한 실패 유형

아래 항목은 기존 계약 테스트가 충분히 다루지 않으므로 각각 소유 작업의 검증에 포함한다.

1. 84×84 회색조 축소에서 HUD 막대 신호가 유지되는 정도와 HUD branch가 상태 추정에 실제로 기여하는지.
2. 네 프레임 스택과 4-tick frame skip이 학습 전이와 평가 추론에서 어긋나는 경우.
3. 최대 네 번 충돌한 손상 차량에서 정책 행동과 잠재 모델 예측이 무손상 데이터에만 과적합되는 경우.
4. 음수 보상 101 decision 연속, 95% 타일 자격, 정상 방향 결승선 통과를 혼동하는 경우.
5. CEM이 예측 모델 오차를 악용하거나 5초를 넘겨 호출 스레드가 계속 실행되는 경우.
6. PPO 학습·튜닝 시드가 보류 검증 시드와 겹쳐 성능을 과대평가하는 경우.

## Task 1: 학습 데이터와 시뮬레이터 계약을 고정한다

**파일**

- 생성: training/__init__.py
- 생성: training/env_factory.py
- 생성: training/labels.py
- 생성: haic_agent/observation.py
- 생성: tests/test_training_data.py
- 생성: tests/test_hud_features.py
- 수정: .gitignore

- [x] 개발 가상환경에서 gymnasium, Box2D, torch, numpy, cv2, pygame import와 CarEnvironment reset/step을 확인한다. 이 설치 확인에만 Windows용 Box2D wheel을 쓰며 공식 requirements.txt는 수정하지 않는다.
- [x] env_factory에 학습용 CarEnvironment 생성을 구현한다. warmup 50 raw tick, frame skip 4, stack 4와 전처리를 공식 로컬 경로와 맞춘다.
- [x] labels에 수집기 전용 레이블을 정의한다: 속도, 네 바퀴 회전 신호, 조향각, yaw 속도, 타일 진행률, 충돌, 손상, off-track, finish. 현재 decision observation/action과 다음 observation 및 합산 reward를 한 transition으로 반환한다.
- [x] HUD ROI 추출은 84×84 이미지에서만 동작하게 하고, 각 ROI 출력과 전체 프레임을 함께 보존한다. 원본 시뮬레이터 상태값은 ROI 보정과 보조 학습 레이블로만 쓴다.
- [x] haic_agent/observation.py에 inference-safe HUD ROI 인터페이스를 두고 training/labels.py도 같은 함수를 사용한다.
- [x] .gitignore에는 학습 checkpoint, 평가 보고서, 제출 zip을 모을 artifacts/haic/만 추가한다. 이후 작업은 이 파일을 수정하지 않는다.
- [x] 결정론적 track_id/seed 목록을 학습, 튜닝, 보류 검증으로 분리하는 split 함수를 추가하고, 세트 중복을 거부한다.
- [x] 테스트 우선 작성: observation shape/range, reset 시 네 프레임 반복, 한 transition당 action 한 번과 raw tick 네 번, HUD crop 좌표와 레이블 시점 정렬, seed set 교집합 없음.
- [x] 96개 서로 다른 decision frame에서 속도·휠·조향·yaw 보조 레이블 오차를 기록했다(`artifacts/haic/task5-hud-validation-20.json`). HUD branch의 오차 변화는 미미했고 steering/yaw는 약간 악화되어, HUD branch를 선택적으로 끌 수 있게 유지하고 전체 프레임 경로만으로도 추론 가능하게 했다. 이 pilot은 track 3/seed 201 한 주행에서만 측정했고 ROI별 feature mask는 추가하지 않았다.

**통과 기준:** 기존 local contract 테스트와 새 수집기 테스트가 통과한다. 학습 로그에는 원시 상태 레이블과 관측/행동 전이의 시점 대응이 남고, 추론 Agent에 레이블 객체를 넘기는 경로가 없다.

## Task 2: 시각 PPO 정책과 학습 루프를 구현한다

**파일**

- 생성: haic_agent/__init__.py
- 생성: haic_agent/networks.py
- 생성: training/rollout.py
- 생성: training/ppo.py
- 생성: training/train_policy.py
- 생성: tests/test_policy_network.py
- 생성: tests/test_ppo_update.py

- [x] 네 프레임 전체를 받는 작은 CNN encoder와 고정 HUD ROI encoder를 구현하고, feature를 하나의 latent vector로 융합한다. HUD ROI branch는 전체 장면 branch를 대체하지 않는다.
- [x] actor-critic이 연속 조향·가속·제동 분포, value, 속도/휠/조향/yaw 보조 예측을 출력하도록 한다. 행동 변환은 steer [-1,1], gas/brake [0,1]을 보장하고 PPO log probability 계산과 일치시킨다.
- [x] rollout 저장소, terminal/truncation 처리, GAE, advantage 정규화, PPO clipped objective와 entropy/value/보조 손실을 구현한다.
- [x] 수집 전용 레이블과 reward를 구성한다. 새 도로 진행과 완주를 높이고 충돌, 오프트랙, 시간 경과를 낮춘다. 완료 순위와 독립적으로 공식 로컬 환경의 종료 조건도 함께 기록한다.
- [x] training/train_policy.py에 고정 seed, train/tune/held-out 분리, 저장 재개, 체크포인트 평가를 넣는다. smoke 실행은 짧게, 전체 학습 step은 인자로 조정 가능하게 한다.
- [x] 테스트 우선 작성: 출력 shape와 유한성, 행동 범위, 분포 log probability 유한성, 한 rollout minibatch에서 PPO 파라미터와 손실이 실제 갱신되는지, 종료 transition에서 GAE가 다음 에피소드로 새지 않는지.
- [x] HUD branch 포함/제외 ablation을 튜닝 split에서 측정해 보조 추정과 주행 성공에 기여하는지 확인한다.

**통과 기준:** 짧은 CPU smoke 학습에서 loss와 gradient가 유한하고 checkpoint를 저장·다시 읽을 수 있다. 학습 중 추론 입력 형식은 공식 관측과 일치하며, 기준 모델 선택 지표는 누적 reward가 아니라 완주율 우선이다.

## Task 3: 짧은 미래를 예측하는 학습 모델 앙상블을 구현한다

**파일**

- 생성: haic_agent/dynamics.py
- 생성: training/train_dynamics.py
- 생성: tests/test_latent_dynamics.py

- [x] 정책 encoder latent와 현재 action을 받아 다음 latent의 변화량을 예측하는 작은 모델 여러 개를 구현한다.
- [x] 보조 head가 다음 시점 진행량/reward, collision 위험, off-track 위험을 예측한다. collision과 off-track은 희귀하므로 샘플 가중치 또는 균형 배치를 지원한다.
- [x] 정책 수집 transition으로 앙상블을 학습하고, holdout에서 latent 예측 오차와 위험 분류 지표를 기록한다. 레이블은 학습 과정에서만 읽는다.
- [x] 앙상블 간 예측 차이를 불확실성으로 산출하고, 학습 데이터 범위 밖에서 불확실성이 커지는지 검사한다.
- [x] 테스트 우선 작성: batch/horizon 차원, action cadence 정렬, finite next latent 및 위험 확률, 모델 간 불일치 점수, 작은 합성 transition 집합에서 손실 감소.

**통과 기준:** 평가 시 simulator import 없이 모델이 프레임에서 만든 latent/action만으로 예측한다. holdout 예측이 PPO 단독보다 명백히 비정상적인 위험 행동을 추천하지 않도록 불확실성 비용이 실제 후보 점수에 반영된다.

## Task 4: 제한시간 CEM과 Agent 진입점을 연결한다

**파일**

- 생성: haic_agent/planner.py
- 수정: agent.py
- 생성: tests/test_planner.py
- 생성: tests/test_agent_inference.py
- 수정: local_runner.py

- [x] CEM은 PPO 행동 분포와 이전 계획을 초기 후보로 사용하고, 학습 전이 모델의 짧은 action sequence rollout으로 진행량, 시간 비용, collision/off-track 위험, ensemble uncertainty를 평가한다.
- [x] 기본 horizon과 population을 CPU에서 조정 가능하게 두고, 최고 시퀀스의 첫 행동만 실행한다. 다음 act에서는 이전 계획을 한 칸 이동해 재사용한다.
- [x] act 진입 직후부터 전체 호출을 재는 monotonic clock deadline을 둔다. 기본 예산은 4.5초이고 5초 전에 끝낸다. CEM candidate batch는 CPU에서 호출당 제한시간을 넘지 않는 크기로 고정한다. 수렴 시 조기 반환하며, 유효 계획이 없거나 시간이 끝나면 PPO의 즉시 행동을 반환한다.
- [x] Agent.reset은 계획 캐시와 모든 episode state를 초기화한다. Agent.act은 입력 shape/range를 확인하고 torch inference mode로 계산하며 유한한 범위 내 행동만 반환한다.
- [x] 추론 코드에서 시뮬레이터 모듈과 원시 state label import가 없는지 검사한다. import 금지 목록과 정적 제출 검사 규칙은 README의 제출 규약에 맞춘다.
- [x] local_runner에 계획 예산 선택 옵션을 더해 짧은 예산의 빠른 폐쇄루프 점검을 지원한다. 기본값은 Agent와 동일하게 4.5초로 유지한다.
- [x] 테스트 우선 작성: warm start 이동, reset 격리, 정책 단독 fallback, invalid/NaN 모델 출력 fallback, plan horizon shape, 예산 소진 전 유효 행동 반환, 4.5초 미만의 act 상한, 반복 episode 간 캐시 오염 없음.

**통과 기준:** 모든 경로에서 행동은 유효하고 5초 제한을 넘지 않는다. 정상적인 주행 방향 선택은 정책과 학습 모델에서 오며, 코드 규칙은 결과 검증과 fallback만 수행한다.

## Task 5: 폐쇄루프 평가, 모델 선택, 제출 묶음을 만든다

**파일**

- 생성: training/evaluate_closed_loop.py
- 생성: training/package_submission.py
- 생성: tests/test_submission_layout.py
- 수정: README.md

- [x] 비렌더링 CarEnvironment로 동일한 seed 목록에서 PPO-only와 PPO+CEM을 모두 주행시키고 episode JSONL과 요약 JSON을 남긴다. 0.1초 budget, 2,000 decision cap의 paired comparison 10개 held-out tuple을 끝까지 실행했고, 모두 자연 off_track DNF를 기록했다(`artifacts/haic/task5-eval-fast-fullcap/`).
- [x] 튜닝 seed 결과로만 planner horizon/population/uncertainty weight를 선택하고 held-out seed는 최종 비교에만 사용하도록 구현했다. 10개 held-out tuple을 고정하고 중복을 거부한다.
- [x] 빠른 반복용 짧은 budget 결과와 기본 4.5초 budget 결과를 구분해 기록했다. 기본 4.5초, 2,000 cap PPO-only episode는 268 decision 후 off_track DNF로 종료했다.
- [x] 제출 ZIP은 현재 planner를 비활성화한다. 2,000-cap held-out 결과에서 PPO+CEM 진행률이 PPO-only보다 낮고 두 모드 모두 0/10 완주였으므로 PPO-only를 선택했고, ZIP runtime config가 이 선택을 강제한다.
- [x] 최종 CPU 검증은 import/Agent 생성 10초, reset/act 각 5초, finite action, 2회 연속 reset 격리와 프로세스 RSS를 점검했다. package smoke는 init 1.328초, act 최대 0.015초, RSS 201,940,992 bytes였다.
- [x] package_submission.py는 agent.py, 필요한 haic_agent 모듈, CPU 가중치만 ZIP 루트에 넣고 학습 코드·labels·dataset·.venv를 제외한다. 깨끗한 임시 디렉터리 static 검사와 strict CPU checkpoint load를 통과했다.
- [x] 사용자 실행법에 venv 생성, Windows 전용 Box2D 설치 우회, 학습, 보류 시드 평가, 제출 묶음 생성을 적었다. 공식 requirements.txt와 공식 평가용 의존성은 그대로 뒀다.

**통과 기준:** 기본 설정 폐쇄루프 결과가 저장되고 두 에이전트 모드가 같은 seed에서 비교된다. 최종 가중치는 held-out 완주율 우선 기준으로 선택하고, 제출 ZIP 내부 경로와 모델 로드가 CPU에서 정상 동작한다.

## 전체 완료 검증

- [x] 개발 가상환경에서 기존 테스트와 새 단위 테스트 실행: 79 passed, 1 expected server-parity skip.
- [x] PPO 256 decision update와 같은 checkpoint 기반 dynamics 128 update를 실행하고 checkpoint를 저장했다.
- [x] 0.05초/2-decision 및 0.1초/300-decision fast-budget 폐쇄루프를 기록했다.
- [x] 기본 4.5초, 2,000 cap PPO-only episode의 실제 종료와 호출/전체 시간을 기록했다.
- [x] 최소 10개 보류 seed의 2,000-cap PPO-only/PPO+CEM paired 비교: 0.1초 fast budget에서 두 모드 모두 0/10 완주 및 자연 off_track 종료. 평균 진행률 PPO-only 0.077411, PPO+CEM 0.074530. 기본 4.5초 예산에서 PPO-only 전체 에피소드도 별도로 기록했으며, 4.5초 CEM paired 비교는 미실행.
- [x] 제출 ZIP을 깨끗한 임시 디렉터리로 풀어 정적 검사와 strict CPU inference smoke를 수행했다.

계획 검토자는 특히 per-act 4.5초 상한, 실제 전체 episode 실행시간, HUD branch의 유효성 기준, 손상/off-track 및 결승선 경계 사례, 검증 시드 분리 기준을 확인한다.
