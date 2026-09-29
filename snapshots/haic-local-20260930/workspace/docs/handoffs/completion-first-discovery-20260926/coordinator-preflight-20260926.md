# 중앙 조정자 사전 검토 — 2026-09-26

**범위:** 읽기 전용 소스 검토. 에이전트가 HZ1–HZ3, 페달 범위, 대진 점수·평가 경로·하네스 계약을 각각 확인했고 중앙 조정자가 HZ4 및 후보 대체안을 검토했다. 후보 성능이나 실제 TRAIN 씬 활성률을 확인한 것이 아니다. 이 사전 검토 중 파일, 체크포인트, 맵 payload를 수정·실행·평가하지 않았다.

## 가설별 준비도

| 가설 | 코드에서 확인된 사실 | 현재 해석과 주요 위험 | 등록 전 필요한 점검 |
|---|---|---|---|
| HZ-PERCEPT-01 | PPO의 시각 특징에는 도로 중심 기준 차량 offset과 장애물 lateral offset이 이미 있다. 학습 코드에서 이 둘을 결합한 signed 상대 offset을 위험 shaping에 쓰지만 actor 입력에는 별도 채널로 넣지 않는다. 시각 특징 경로는 기본 비활성이다. (`haic_agent/pixel_features.py:18-27,153-180`; `haic_agent/networks.py:181-195,231-264`; `training/train_policy.py:172-189`) | 새 채널은 새 픽셀 정보보다 기존 정보의 결합을 유도하는 inductive bias에 가깝다. 시각 입력 설정을 후보에서만 켜면 대조군과 혼입된다. 채널 수 변경은 checkpoint encoder shape에도 영향을 준다. | 합성 픽셀 fixture에서 offset 부호·정규화·좌우 반전을 확인하고, 허용된 TRAIN 자료에서 eligible 상태 빈도와 지연을 측정한다. 기존 시각 경로 설정은 모든 비교군에서 동일하게 고정한다. |
| HZ-PERCEPT-02 | 관측은 4개의 84×84 grayscale frame stack이며 기존 temporal feature는 직전 obstacle side를 다룬다. 마지막 두 policy observation은 인접 관측이다. (`env_wrapper.py:13-27,42-56`; `haic_agent/pixel_features.py:78-125,153-180,193-218`) | `clip((y_now-y_prev)/s, 0, 1)`은 가능한 pixel-only 단서지만, 물체 추적이나 물리적 time-to-collision은 아니다. detector가 각 프레임에서 가장 가까운 blob을 고르므로 물체 교체, 도로 굴곡, 카메라 움직임이 오탐을 만들 수 있다. `s`와 matching 기준은 정해지지 않았다. | 동일 프레임은 0, 아래쪽 이동은 양수, 사라짐·후퇴·불일치는 0이 되는지 고정 fixture로 확인한다. TRAIN-only에서 activation/false-match 빈도와 batch-one latency를 확인한 뒤 `s`와 matching 기준을 등록한다. |
| PPO-PEDAL-RANGE-COMPLETION-01 | `--throttle-expansion` 경로가 있고 정책 학습·추론은 branch별 `PedalScale`을 사용한다. 기본 gas 한계는 0.12이며, 체크포인트는 throttle/brake scale을 기록한다. (`haic_agent/networks.py:29-54,294-311`; `training/train_policy.py:1645-1655`; `agent.py:162-185`) | 네 방향 중 구현 마찰이 가장 적지만, 포화가 실제 미완주 전에 일어나는지는 모른다. scale을 바꾸면 같은 checkpoint의 절대 gas 출력이 즉시 달라질 수 있고, 시각 특징을 켠 상태에서는 gas shaping의 정규화도 바뀔 수 있다. | 한 `K` 값을 사전 등록하고 TRAIN-only 기록에서 포화 빈도와 absolute gas를 확인한다. warm start가 필요하면 optimizer 상태를 재사용하지 않는 actor-only 초기화 경로와 시작 출력 차이를 함께 고정한다. 다른 reward·feature 설정은 두 arm에서 동일하게 유지한다. |
| CEM-UNCERTAINTY-FALLBACK-01 | actor action fallback은 이미 있다. Dynamics의 `uncertainty`는 latent/progress/reward/collision/off-track spread를 더한 값이고, `PlanResult`는 선택 action·sequence·score만 반환한다. (`agent.py:243-280`; `haic_agent/dynamics.py:112-130`; `haic_agent/planner.py:21-28,117-159,184-245`) | 선택 plan의 불확실성은 현재 외부에 보이지 않으며, 서로 다른 단위의 spread를 합친 값은 보정되지 않은 threshold에 적합하지 않다. 동역학 모델 학습·threshold 보정은 runtime·데이터 의존성이 가장 크다. | TRAIN-only transition 오차와 uncertainty의 보정을 먼저 등록한다. 선택 sequence의 uncertainty를 넘기는 구현 경로와 시간 예산을 사전 검토한다. threshold는 tune/held-out을 보지 않고 한 번 고정한다. |

위 표의 위험·아이디어는 source-grounded inference 또는 제안이다. **네 가설 모두 completion gain은 미측정**이다.

## CEM 대체 후보 검토

독립 에이전트는 `road_curve_magnitude`와 기존 PPO reward shaping을 이용한 **curve-conditioned braking reward**를 대안으로 찾았다. `training/train_policy.py`에 `--curve-brake-reward` 경로가 있고 actor는 시각 특징으로 곡률을 볼 수 있다. 이 후보는 새 dynamics 모델 학습이나 planner 선택 불확실성 전달이 필요하지 않아 CEM 후보보다 구현·실행 경로가 단순하다 (`haic_agent/pixel_features.py:142,153`; `training/train_policy.py:248,319,1946`; `haic_agent/networks.py:127`; `agent.py:152`).

이는 성능 순위가 아니라 **첫 대진의 실행 가능성에 따른 제안**이다. 곡률 추정이 제동보다 충분히 앞서는지, shaping이 braking을 바꾸는지, 완주율을 높이는지는 측정되지 않았다. 기존 학습에는 curve에서 target-speed shortfall을 완화하는 `CURVE_SPEED_RELIEF`가 이미 있으므로, 이 후보가 검증하는 것은 그 위에 더하는 brake reward의 추가 효과다. 후보로 채택하려면 control과 candidate 모두 시각 특징을 켠 동일 architecture/초기화/훈련 조건을 쓰고, control의 brake reward 계수는 0, candidate의 계수는 설계 전에 고정해야 한다. 새 보상은 불필요한 제동도 유발할 수 있어 activation에서 실제 행동 변화를 확인해야 한다. `--enable-visual-features`와 `--curve-brake-reward`는 현재 train allowlist에 없으므로 별도 구현 승인이 필요하다. CEM direction은 dynamics 준비·보정이 갖춰질 때까지 다음 주기로 미루는 것을 권고한다.

## 하네스와 실행 준비 상태

- `runs/haic-research-v2/`와 `artifacts/haic-research-v2/`는 아직 없다. 새 경로에서 사용할 control PPO 및 CEM dynamics checkpoint도 확인되지 않았다. 과거 artifact를 자동으로 찾아 control로 삼지 않는다.
- 등록된 `train_policy` allowlist에는 `--initialize-from`, `--enable-visual-features`, `--enable-temporal-features`, `--throttle-expansion`이 없다. 실제 학습 parser에는 관련 flag가 존재한다. CEM용 dynamics 학습도 별도 등록된 command profile이 없다.
- 정확한 manifest/run hash, control과 checkpoint identity, train/tune/held-out split·seed·분모, 학습 단계 수와 시간/compute 상한이 고정되지 않았다.
- 하네스 규칙상 최소 네 개의 독립 방향이 필요하며, 후보가 활성 조건을 못 맞추면 미활성 후보를 포함한 세 방향 실험으로 축소하지 말고 등록 전 교체·수정해야 한다.

## 권고 다음 단계

1. HZ1, HZ2, throttle 범위, curve-brake 대체안의 activation gate와 공통 control protocol을 설계한다. CEM은 dynamics checkpoint와 보정 경로가 갖춰질 때까지 미룬다.
2. 현재 하네스의 split 인자·Windows 경로·held-out winner 검증은 source상 구현돼 있다. 기능을 새로 구현할 대상으로 보지 말고, 허가된 검증 단계에서 확인해야 한다.
3. 새 v2 control 생성 방식, training seed·step budget, 허용 TRAIN split, tune/held-out cell 수·evaluation seed, 최대 compute와 평가기 불변조건을 하나의 plan revision/hash로 묶는다.
4. 설계·구현·실행을 각각 같은 revision/hash에 결속해 승인한다. activation이 확인된 후보만 fixed map/seed tune에 넣고, tune의 strict winner 한 개만 별도 승인 뒤 held-out으로 보낸다.

현재에는 계획 값이 고정되지 않아 tune run을 등록하거나 실행할 수 없다. 에이전트 한 명이 초기 범위 제한 검색에서 `research/artifacts/`와 `tmp/` 텍스트 일부를 우연히 출력했지만 의도적으로 열거나 분석하지 않았고, 권고 근거로 사용하지 않았다. 이후 검토는 소스와 문서로 제한했다. held-out map payload, checkpoint, training/evaluation 결과는 열지 않았다.

## 토너먼트 하네스 에이전트 검토

점수와 데이터 경로 검토는 비교 분모, map/seed identity, 예산 및 실행 입력 해시 확인이 대체로 작동한다고 보았다. 점수 검토에서는 malformed action과 누락된 collision/damage 계측이 후보 실패와 일반 계측 오류에 맞게 구분되는 것으로 확인됐다. 초기 계약 리뷰는 P1/P2 결함을 지적했지만 이는 그때 본 소스 상태를 반영한다. 현재 dirty working tree를 다시 읽은 결과, 아래 source-level guard가 이미 존재한다.

- `haic_research/commands.py`는 `candidate-manifest`와 `split-group` 조합 및 `site-map-split` 존재를 split traversal보다 먼저 검증한다.
- 같은 파일은 split map 경로를 `normcase(...).casefold()`로 비교해 Windows 대소문자 별칭을 같은 identity로 취급한다.
- `training/evaluate_tournament.py`의 `_verify_frozen_tune`은 held-out 전에 완료·승인된 tune run, run/plan/summary hash, 후보/control/runtime identity, ranking 상태와 실제 winner를 확인한다.

위 guard는 현재 working tree에 존재하지만 uncommitted 상태이며, 이번에는 테스트나 대진으로 동작을 검증하지 않았다. 따라서 구현 결함이 해결됐다고 단정하지 않고 source-level 근거와 behavior verification 상태를 구분한다. 개별 호출 강제 종료는 지원되지 않는다. 호출이 반환하지 않으면 바깥 timeout이 전체 run을 실패 처리하며 부분 기록만 남는다.
