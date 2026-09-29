# HAIC PPO 페달·조향 학습 수정 계획

**목표:** 사이트 맵 첫 코너에서 PPO 정책의 브레이크가 앞바퀴의 조향 효과를 억제하는 문제를 줄이고, 완주 성능을 여러 학습 시드로 비교한다.

## 추가 진단: 첫 페달 수정 실험

- 시드 `8101/8102/8103`은 각각 8,192 transition을 모은 뒤 PPO update를 한 번만 수행했다. tune 완주율은 전부 `0/2`, 진행률은 각각 `12.35%`, `2.47%`, `3.29%`였다.
- tune 상위 시드 `8101`은 held-out 첫 episode에서 600 decision 뒤 off-track으로 종료했다. 평균 속도 `19.67`, 평균 gas `0.00432`, brake 사용 `0%`, 평균 절대 yaw `0.599 rad/s`였다. 따라서 이 수정 뒤에는 차가 느려서가 아니라 시각 정책이 코스를 오래 추종하지 못해서 완주하지 못했다.
- 행동은 거의 일정한 좌조향과 저가속에 머물렀다. 원인은 8,192 transition이 충분한 on-policy 반복을 만들지 못한 점, 그리고 수집기 cap인 500 decision이 실제 truncation으로 표시되지 않아 GAE가 다음 reset episode까지 이어진 점으로 좁혀졌다.
- GAE 경계 수정 후 4 update로 나눈 8,192-step 실험도 tune 완주 `0/2`, 최고 진행률 `3.70%`에 그쳤다. 최선 정책은 첫 tune episode에서 decision 158에 off-track 되었고, steer 평균 `-0.034` (표준편차 `0.0093`), 평균 속도 `10.59`, 평균 gas `0.00544`였다. truncation bug는 수정했지만 반복 횟수만 늘리는 것으로는 해결되지 않았다.
- 조작 보정은 튠 지도의 첫 코너에서 steer `-0.12`, gas `0.005`로 200 decision 동안 off-track 없이 진행률 `9.05%`; gas `0.01`로 평균 속도 `17.02`, 진행률 `9.47%`를 보였다. steer `0`는 148 decision, steer `+0.12`는 135 decision에서 off-track 되었다. 다음 실험은 초기 평균을 이 코너 값에 가깝게 둔다.
- 입력축을 분리한 추가 보정은 같은 tune map seed에서 steer `-0.30`부터 `+0.30`까지, gas `0`부터 `0.02`까지, brake `0`부터 `0.20`까지 비교했다. 80 decision에서 steer `-0.12`가 진행률 `7.41%`, yaw 평균 `+0.361 rad/s`였고, steer `0`은 `2.88%`, 반대 steer `+0.12`는 `2.06%`였다. 이는 외부 steer 부호가 내부 앞바퀴 각도와 반대인 것을 포함해 초기 방향을 확인한다.
- gas `0.01`은 같은 80 decision에서 속도 `15.98`, 진행률 `7.41%`; gas `0.02`는 속도 `23.49`, 진행률 `7.41%`였다. 같은 50 decision 접근 주행 후 brake `0.01`은 속도를 `12.14`에서 `7.26`으로 낮췄고, brake `0.03`은 20 decision 안에 정지시켰다. 그래서 signed pedal의 음수 구간을 기존처럼 brake `0~1`에 대응하지 않고, 물리 측정값에 맞춰 최대 brake `0.03`으로 축소하고 역변환/Jacobian도 같은 스케일을 쓰도록 했다. 세부 기록은 `artifacts/haic/action-axis-calibration-v1.json`이다.
- 보정 초기값으로 학습한 checkpoint는 첫 두 update 뒤 tune `31.3%`, 세 번째 update 뒤 `50.6%`에 도달했지만, 마지막 update에서 성능이 급락했다. 보존된 최선 정책도 decision 448에서 tune 지도의 progress `50.6%` 장애물 구간 뒤 off-track으로 끝났다. 기존 보고의 lateral error `5.79` 비교는 도로 반폭을 `4`로 잡았으나, 시뮬레이터는 custom map `geometry.width=8`을 반폭으로 사용한다. 그 값만으로 도로 경계 이탈을 입증할 수 없어 원인 근거에서는 제외한다. 평균 속도 `15.94`, 최고 `33.49`, brake 사용 `0%`였다.
- 이 학습의 critic loss는 최대 `88`, 보조 MSE는 최대 `402`였고 policy loss는 `0.15` 수준이었다. 따라서 보상 합계와 물리 보조 타깃 크기를 정규화해 가치·보조 회귀가 정책 최적화를 압도하는 문제를 줄인다.
- reward/auxiliary normalization을 쓴 다음 run은 8,192 transition, 4 update에 `686.4s` 걸렸고 tune selection 진행률은 `20.99% → 8.64% → 8.23% → 7.82%`로 떨어졌다. loss는 안정됐지만 성능이 오르지 않아 learning rate를 낮춰 추가 비교한다. update 사이 selection 평가는 800 decision cap으로 제한하고, 최종 tune은 2,000 cap으로 확인하며 단계별 runtime을 기록하도록 추가했다.
- lower-LR 재개도 최선 tune `20.58%`, 완주 `0/2`로 개선되지 않았다. trace는 steer 평균 절대값 `0.083`, gas 평균 `0.0107`, brake `0%`, 최고 속도 `44.84`로 progress `20.58%`에서 off-track을 보였다. 긴 주행에서 policy 평균이 거의 일정했고 초기 signed pedal 분산 `0.15`와 longitudinal mean `0.55` 조합은 brake 샘플 확률이 사실상 0이었다.
- 다음 실험은 초기 분산을 steer `0.35`, signed pedal `0.4`로 넓혀 brake를 포함한 대안 행동을 실제 PPO rollout에 제공하고, 각 rollout의 pedal/steer 사용률을 기록한다.
- 이 분산과 brake 상한 `0.03`으로 새로 학습한 seed `8101`은 8,192 transition, 4 update, `493.3s`였다. tune 완주는 `0/2`, 최고 진행률은 update 1의 `9.47%`; update 2–4는 `8.23%`, `8.64%`, `8.23%`였다. rollout brake 비율은 `3.9–5.5%`, pedal overlap은 항상 `0%`였다. 새 페달 스케일은 정상 적용됐지만 완주 성능은 없었다.
- 선택 정책 trace는 steer 평균 `-0.1268`, 표준편차 `0.0052`, gas 평균 `0.01148`, brake `0`으로 사실상 고정 출력이었다. 최고 속도 `37.62`; decision 70에서 네 바퀴가 도로 접촉을 잃었을 때 속도 `15.91`, 진행률 `6.58%`였고, off-track 종료는 decision 370에서 발생했다. 충돌/손상은 없었다. 부족한 점은 모터 부호나 크기보다 상태에 따라 조향을 조절하는 정책 학습이다.
- 그에 따라 collector가 지도 중심선 기준 lateral/heading 오차를 훈련 전용 label로 기록하고, PPO reward에 차선 중심·방향 오차의 dense penalty를 더한다. 제출 추론 입력과 행동 선택에는 지도 정답을 전달하지 않는다.
- v7의 상세 trace에서도 route reward만으로 actor가 입력별 조향을 배우지 못했다. 더 확인해 보니 PPO observation은 행동 전 프레임인데 auxiliary target은 행동 뒤 상태에서 와서, 빠른 코너에서 화면과 정답이 한 decision 어긋났다. `CollectedTransition`에 행동 전 `observation_labels`를 보존하고 기존 7개 센서 목표에 정규화 lateral offset 및 heading sine/cosine을 추가해, actor와 공유하는 visual encoder가 조향에 필요한 경로 상태를 같은 프레임에서 예측하도록 수정했다. geometry는 계속 학습 전용이며 runtime으로 전달하지 않는다.
- v8은 8,192 transition / 4 update / seed 8101에서 tune 선택 진행률 `25.93%`, 전체 tune 진행률 `28.40%`, 완주 `0/2`였다. deterministic trace의 steer 표준편차는 `0.00498`로 여전히 상태 반응이 부족했다.
- 같은 사이트 tune map 두 시드에서 `VisionCorridorAgent`가 decision 530에 완주했고 충돌·off-track 없이 주행했다. 이를 runtime 주행 모드로 쓰지 않고, 사용자가 승인한 훈련 전용 교사로 활용해 actor를 워밍업한 뒤 imitation loss 없이 PPO를 fine-tune한다.

## 훈련 전용 교사 워밍업 설계

- 시연 수집기는 `TrainingSplit.train`만 순회한다. tune/held-out 데이터는 teacher 수집과 BC 업데이트에 전달하지 않는다.
- 시연 행동 `[steer, gas, brake]`를 actor의 `[steer, signed longitudinal]` pre-transform 좌표로 역변환하고, 픽셀 encoder와 actor mean에만 기본 3 epoch MSE를 적용한다.
- 워밍업 뒤 PPO rollout/update에는 teacher loss나 teacher controller를 포함하지 않는다. 최종 Agent는 strict-load된 visual actor를 사용하고, submission ZIP에서 teacher 모듈과 controller mode를 제외한다.
- 시연 픽셀은 `uint8`로 보관한 뒤 minibatch에서 `[0,1]`로 변환한다. 각 train episode 시연 상한은 기본 800 decision이며, 수집량·저장 byte·초기/최종 imitation MSE·경과 시간을 기록한다.
- 8-step 사이트 맵 smoke에서 train split 4개 episode로 시연 16개를 수집했고 MSE가 `0.06660 → 0.05963`으로 감소했다. 전체 회귀 테스트는 111개 통과, 1개 skip이었다.
- v9은 seed 8101, 8,192 transition, 4 update로 학습했다. train split 4 episode에서 2,138개 teacher sample을 모으는 데 `88.62s`, uint8 저장량은 `60,360,016` bytes였고, 3-epoch BC action MSE는 `0.07353 → 0.000843`으로 낮아졌다.
- BC 직후 tune 평가는 완주 `0/2`, 진행률 `8.23%`였다. PPO 이후도 완주 `0/2`, 진행률 `8.23%`로, v8의 `28.40%`보다 낮았다. BC-only trace는 decision 274, steer 표준편차 `0.00995`; PPO checkpoint trace는 decision 301, steer 표준편차 `0.00477`에서 off-track했다. 이번 BC는 train 데이터에서 teacher 행동을 맞췄지만 별도 tune map으로 일반화하지 못했다.
- 다음 진단은 DAgger식으로 train split에서만 warm-start actor가 방문한 상태를 teacher로 라벨링해 BC를 재학습하고, PPO 전에 tune 성능을 확인한다. tune map은 끝까지 라벨 수집에 쓰지 않는다. 근거: Ross, Gordon, Bagnell, AISTATS 2011, https://proceedings.mlr.press/v15/ross11a.html

## 확인된 원인

- 기존 정책은 조향과 함께 가속 약 `0.5`, 브레이크 약 `0.35`를 자주 출력한다.
- 로컬 물리에서는 브레이크가 네 바퀴 모두에 걸린다. 같은 조향·가속에서 브레이크 `0.35`는 차체 회전을 거의 없앴다.
- 브레이크를 끄면 조향으로 차체가 돌았지만 속도가 빠르게 과도해졌다. 따라서 브레이크 제거만으로 해결하지 않는다.
- 조향·가속·제동 입력은 튜닝 지도에서 따로 보정했다. 가속 `0.005`, 무제동에서 첫 코너를 낮은 속도로 통과할 수 있는 것을 확인했다.

## 설계

1. 시각 PPO actor는 내부적으로 `[steer, longitudinal]` 두 값을 학습한다. 양의 longitudinal은 가속, 음수는 제동, 0은 코스팅이다.
2. 액션 변환기가 내부 값을 HAIC의 기존 `[steer, gas, brake]` 3값으로 바꾼다. 한 시점에 가속과 제동이 동시에 나오지 않으며, 외부 `Agent.act()` 계약과 학습 전이 모델의 3값 행동 입력은 유지한다.
3. 사이트 맵 보정 결과에 맞춰 최대 가속 `0.02`, 최대 제동 `0.03`으로 제한한다. 첫 코너에 맞춘 초기 평균은 steer `-0.12`, gas 약 `0.01`로 두며, 훈련 때만 쓰는 route-tracking 보상으로 PPO가 영상 상태에 따른 조향 변화를 학습하도록 돕는다.
4. CEM은 내부 정책 좌표에서 후보를 탐색하고, 전이 모델에는 변환된 3값 행동을 전달한다.
5. tune 지도에서 후보를 비교한 뒤 세 개의 독립 학습 시드 결과를 평가한다. held-out 지도는 최종 성능 확인에만 사용한다. 검증 전에는 새 ZIP을 제출 후보로 취급하지 않는다.

## 구현·검증 작업

- [x] 설계된 2차원 PPO 분포, 상호 배타적 페달 변환, 역변환 로그확률, 안전한 초기 행동을 테스트한다.
- [x] Agent fallback과 CEM의 내부/외부 액션 차원을 테스트한다.
- [x] 기존 3차원 전이 모델 입력과 제출용 3차원 출력이 유지되는지 테스트한다.
- [x] 수집기 episode cap을 GAE truncation 경계로 전달하는 테스트와 반복 PPO update 테스트를 추가한다.
- [x] tune 첫 코너에서 조향 방향과 안전한 가속 크기를 개별 보정한다.
- [x] 학습 보상과 보조 물리 정답을 스케일링해 critic/auxiliary loss를 안정화하는 테스트를 추가한다.
- [x] tune selection cap과 전체 episode cap을 분리하고 rollout/PPO/evaluation 경과시간을 기록한다.
- [x] 정책 초기 표준편차가 signed pedal 제동 행동을 표본화하는지 검증하고 update별 액션 비율을 기록한다.
- [x] 같은 출발 상태에서 steer/gas/brake 축을 따로 보정하고 제동 상한을 물리 응답에 맞춘다.
- [x] collector에 지도 기준 lateral/heading label을 추가하고 dense route-tracking reward를 검증한다.
- [x] 시각 보조 타깃을 policy observation과 같은 상태에서 만들고 lateral/heading 신호를 visual encoder에 학습시킨다.
- [x] train-only visual teacher demonstration과 actor-only BC warm-up을 추가하고, 이후 PPO에서 teacher loss를 쓰지 않는지 검증한다.
- [x] 제출 Agent 및 ZIP에서 corridor teacher 실행 경로를 제거하고 learned actor strict loading을 검증한다.
- [x] 8-step 사이트 맵 smoke에서 시연 수집, MSE 감소, PPO rollout/update와 체크포인트 저장을 확인한다.
- [x] v9 전체 학습과 BC-only tune 진단을 끝내고, 두 경로 모두 tune 진행률 `8.23%`, 완주 `0/2`로 개선이 아님을 기록한다.
- [ ] train split에서만 DAgger 상태를 수집해 BC-only tune을 평가한다. 성능이 개선되지 않으면 PPO 전체 학습을 재실행하지 않는다.
- [ ] BC-only 성능이 검증된 경우에만 같은 예산의 PPO fine-tune을 수행하고 결정론적 조향 변화와 완주율을 확인한다.
- [ ] 세 시드로 동일한 사이트 train/tune split에서 4회 이상 on-policy update, 보정된 초기 평균, 정규화된 학습 신호를 사용해 학습하고 시간·tune 결과를 기록한다.
- [ ] tune 기준 상위 정책을 held-out 맵의 전체 episode cap과 기존 공식 트랙에서 평가한다.
- [ ] 같은 맵에서 브레이크 사용량, 차체 yaw, 조향 입력, 진행률을 비교한다.
- [ ] 성능 개선이 확인될 때만 새 제출 ZIP 후보를 만든다.

## 수용 기준

- 출력 행동은 항상 3개 값이며, gas는 `0.02`, brake는 `0.03`을 넘지 않고 두 페달이 동시에 양수일 수 없다.
- 초기 결정 행동은 작은 조향과 약한 전진 페달을 보인다.
- PPO likelihood 재계산은 rollout에서 저장한 pre-transform action과 일치한다.
- CEM 후보가 전이 모델에 유효한 3차원 행동으로 전달된다.
- 성능은 세 학습 시드와 분리된 held-out 평가 결과로 보고한다. smoke test나 액션 형식 검증만으로 주행 성공을 주장하지 않는다.

## 추가 설계: 픽셀 기반 위험·HUD 특징을 PPO actor에 결합

### 진단 근거

- 최신 actor는 tune 지도를 `2/2` 완주했지만, 중앙에 가까운 50% 장애물이 있는 held-out 지도는 `0/5` 완주했다. 같은 시뮬레이터 seed를 teacher는 충돌 없이 완주했다.
- 같은 teacher 주행 픽셀을 비교했을 때 teacher는 35.8% progress에서 장애물을 감지하고 36.6%부터 제동했지만, PPO actor는 37.9%까지 가속을 유지했다. actor가 작은 장애물을 초기에 읽지 못하는 시각 특징 문제가 확인됐다.
- 지도 장애물 분포도 달랐다. train의 50% 장애물 lateral은 `-0.2541`, `+0.4991`, tune은 `+0.4395`, held-out은 `+0.0927`이었다. train/tune에 중앙 장애물이 거의 없었다.

### 결정

- 제출 runtime에서 행동을 고르는 주체는 PPO actor 하나로 유지한다. teacher/corridor action fallback은 넣지 않는다.
- 같은 84x84 픽셀에서만 HUD 속도 bar, 가까운/먼 차선 중심, 휘어짐 크기, 가장 가까운 밝은 장애물의 유무·상대 좌우 위치·화면상 접근 정도를 계산한다. map, simulator state, 좌표 정답은 입력하지 않는다.
- 이 특징은 learned encoder와 함께 PPO actor의 입력으로만 사용한다. 특징 추출기가 steer/gas/brake를 결정하지 않는다. 훈련과 제출에서 동일한 extractor를 쓴다.
- 기존 checkpoint 호환성을 위해 checkpoint metadata에 feature version을 기록하고, field가 없는 기존 checkpoint는 feature extractor를 비활성화한다.
- 장애물 lateral domain randomization은 train 지도에서만 생성한다. tune/held-out 원본 geometry, obstacle labels, pixels는 augmentation이나 PPO update에 포함하지 않는다.

### 검증 기준

- synthetic pixel fixtures에서 속도 bar·차선 center·장애물 위치가 알려진 값으로 검출되는지 단위 테스트한다.
- actor output은 기존 3값 action 계약과 gas/brake mutual exclusion을 유지하고, feature-enabled checkpoint strict load 및 ZIP smoke를 검증한다.
- held-out 지도를 보지 않은 채 train/tune을 평가한다. tune candidate가 기존 기준을 넘은 경우에만 held-out 5 seeds를 최종 평가한다.
- actor `act()` latency p95는 5초 제한 아래인지 기록한다. held-out의 완료·충돌·lap time에서 개선이 확인될 때까지 제출 ZIP은 만들지 않는다.

## 최신 속도·공식 맵 실패 진단 및 다음 PPO 실험 (2026-09-22)

### 측정 결과

- 최신 PPO actor는 Track Lab held-out 장애물 맵(seed `20260923`, 장애물 5개)을 `5/5` 완주했다. 각 기록은 `19.66s`, `246` decision, 평균 속도 `39.69`, 충돌·손상 `0`이다.
- 같은 map/seed의 오프라인 `sprint_guarded` 교사 기준은 `17.32s`, `217` decision, 평균 속도 `46.76`, 충돌·손상 `0`이다. 교사는 속도 비교에만 쓰고 제출 action 경로에는 넣지 않는다.
- 랩타임은 `246 × 4 raw frame / 50 FPS = 19.68s`와 일치한다. PPO `act()` p95 `16ms`는 5초 호출 제한보다 훨씬 짧으므로 주행 랩타임의 병목은 추론 대기시간이 아니다.
- 동일한 PPO가 official track 2/seed `101`은 `25.80s`에 완주했지만, official track 1/seed `42`는 6개 내장 장애물에서 5회 연속 충돌 후 progress `16.6%`에서 종료했다. 같은 트랙에 사용자 장애물 하나를 추가한 map은 충돌 전 progress `14.1%`에서 이탈했다.
- track 1 충돌 trace에서 actor의 픽셀 추출기는 15.7m 거리부터 장애물을 검출하고 urgency `0.96–1.0`을 냈다. 그러나 4.1m에서 충돌한 action은 gas `0.076`, brake `0`이었다. 따라서 이 사례는 장애물 미검출만의 문제가 아니라, 해당 시각 상태에서 제동·회피 행동을 PPO가 학습하지 못한 문제다.
- PPO는 custom-only 사이트 맵 split으로 학습했고 official 내장 장애물을 포함한 학습 episode는 없었다. 공식 obstacle 이미지·배치 분포의 훈련 노출 부족이 track 1 실패의 주요 원인으로 판단된다.
- 같은 held-out 맵의 직선 구간에서 PPO gas는 대체로 `0.04–0.09`, 반면 비교 교사는 최대 gas `0.24`를 사용했다. PPO의 현재 `MAX_GAS=0.12`와 75% teacher warm-up cap은 빠른 추격의 상한을 낮춘다. 학습 보상에는 목표 속도 항이 없고, 랩타임 개선은 타일 진행과 step/time 비용을 통해 간접적으로만 유도된다.
- 최신 4,096-step PPO continuation은 약 `356s`가 걸렸다. 이 중 rollout `173s`, PPO update `83s`, tune/HUD 평가 약 `99s`다. 이는 학습 반복 시간이며 차량의 실제 시뮬레이션 랩타임과는 별도 지표다.

### 다음 실험: 공식 obstacle 분포에 대한 PPO 학습

- **가설:** custom-only train 분포가 official 내장 장애물의 시각·주행 분포를 대표하지 못해, detector가 위험을 읽어도 actor가 대응하지 못한다.
- action scale과 reward는 우선 그대로 두고 train에 official track 1 seeds `43–46`, track 2 seeds `102–105`를 추가한다. 같은 custom train maps 및 train-only obstacle-lateral variants도 유지한다.
- tune에는 custom tune episode와 official track 1 seeds `47–48`, track 2 seeds `106–107`을 사용한다. held-out에는 기존 custom map, official track 1/seed `42`, track 2/seed `101`, track 1/seed `42`에 추가 장애물을 둔 stress map을 유지한다. split 간 track/seed 및 map ID 겹침을 금지한다.
- 최신 actor에서 teacher 수집 없이 PPO를 4,096 transition / 2 update / learning rate `1e-5`로 이어간다. actor-only action 경로와 gas/brake 상한은 이번 실험에서 바꾸지 않아 official 분포 추가 효과를 분리한다.
- tune가 현재 baseline보다 나아질 때만 held-out을 다시 평가한다. official track 완주가 개선되어도 held-out lap time이 교사 기준과 크게 벌어지면, 다음 독립 실험에서 action gas 범위와 PPO 속도 학습 신호를 조정한다.

## 낮은 학습률 PPO actor-only continuation (2026-09-22)

- 동일한 source actor, 4,096 transition, 2 update, seed `8101`, round-robin custom/official rollout, CPU thread `1` 설정에서 learning rate만 `1e-5 → 2e-6`으로 낮춰 비교했다. update 1이 tune 4/6을 완료해 보존됐고 update 2는 3/6으로 떨어져 선택되지 않았다.
- 실제 제출 Agent의 tune 결과는 source `3/6`에서 후보 `4/6`으로 늘었다. 후보는 custom tune 두 건을 각각 `20.34s`에 완주했고, Track 1 seed `48`은 `23.78s`, Track 2 seed `107`은 `30.40s`에 완주했다. Track 1 seed `47`은 progress `80.2%`에서 off-track, Track 2 seed `106`은 `11.1%`에서 crash했다. source보다 완주율은 높지만, 추가 완주가 느려 tune 완주 랩타임 중앙값은 `20.46s → 22.06s`로 악화됐다.
- 실제 PPO-only held-out은 후보 `6/8`, source도 `6/8`이었다. 커스텀 장애물 맵은 후보 `5/5`를 충돌 없이 `19.62s`에 완주했다(source `19.66s`). Track 2 seed `101`은 후보 `25.68s`, source `25.80s`로 완주했다. Track 1 seed `42`는 둘 다 off-track으로 끝났고, 추가 장애물 stress map도 후보 progress `14.5%`에서 off-track되어 기존 실패가 남았다.
- 앞선 `1e-5` interleaved 후보가 held-out `2/8`에 그친 것에 비해 `2e-6`은 custom obstacle 주행을 회복했고 held-out 완주 수를 source 수준으로 보존했다. 다만 source 대비 held-out 완주율은 개선되지 않았으므로 이 ZIP을 SOTA로 확정하지 않고 검토 후보로 둔다. 독립적인 새 맵/seed에서 추가 확인이 필요하다.
- 결과는 `artifacts/haic/site-map-official-domain-interleaved-ppo4096-lr2e6-seed8101-cpu1/training-result.json`, `tune-episode-results.json`, `heldout-episode-results.json`에 저장했다. 수집 중 실행된 구버전 train evaluator는 `finish_time_s` 원값을 사용해 lap time이 `1.02s` 길게 기록됐으므로 절대 랩타임은 실제 Agent 평가 결과를 기준으로 한다.
- train evaluator가 reset 시각을 빼도록 수정하고 회귀 테스트를 추가했다. 새 테스트는 수정 전 `12.0s`를 보고해 실패했고 수정 후 `10.75s`를 보고해 통과했다. 전체 회귀 테스트는 `136 passed, 1 skipped, 5 subtests passed`였다.
- 검토용 제출 후보 `artifacts/haic/submission/haic-ppo-actor-lr2e6-seed8101.zip`을 만들었다. ZIP smoke에서 planner 비활성화, strict checkpoint loading, 유한 action, 초기화 `1.5s`, RSS 약 `194 MiB`를 확인했다. ZIP에는 corridor 실행 코드가 없으며 PPO actor만 action을 결정한다. 이 후보는 held-out 성능 우위를 입증하지 못했으므로 최종 제출/최고기록으로 표시하지 않는다.

## 속도가 개선되지 않은 원인과 적용한 변경 (2026-09-23)

### 측정으로 확인한 원인

1. **행동 변환의 가속 상한이 이미 포화였다.** 선택된 actor의 rollout `training_action_metrics`는 `gas_mean 0.06222`, `gas_max 0.11526`이고 `MAX_GAS`는 `0.12`다. 즉 표본 최대 가속이 상한의 `96.1%`로, PPO는 허용된 가속을 거의 전부 요청하고 있었다. 같은 맵을 `17.32s`에 도는 오프라인 비교 교사는 `max_gas 0.24`를 쓴다. 랩타임을 제한한 것은 정책이 아니라 action 변환의 천장이었다.
2. **보상에 랩타임 기울기가 사실상 없었다.** 기존 `SAFE_SPEED_REWARD_WEIGHT * min(speed/50, 1)`은 결정마다 속도에 비례해 더하는 항이다. 한 랩에 대해 합하면 `Σ speed × Δt = 주행 거리`이고 랩 거리는 고정이므로, 빠른 랩과 느린 랩의 총합이 같다. 속도 보상이라는 이름과 달리 빠른 랩을 전혀 선호하지 않았다.
3. **교사 시연 라벨이 교사보다 느렸다.** `_teacher_action_tensor`는 교사 gas를 인스턴스 값이 아니라 클래스 상수 `MAX_TEACHER_GAS = 0.12`로 clip한 뒤 정규화했다. `sprint_guarded`(`max_gas 0.24`)로 수집해도 `0.12` 위는 전부 잘렸고, 여기에 `TEACHER_POLICY_CAP_FRACTION = 0.75`가 곱해져 imitation 목표의 최대 가속은 `0.09`였다. 워밍업이 처음부터 actor를 느린 지점에 고정했다.
4. **`PPOConfig.gamma`와 `gae_lambda`가 학습에 연결되어 있지 않았다.** `collect_rollout`이 `gamma=0.99, gae_lambda=0.95`를 직접 써서 config 값은 무시됐다. 실험에서 credit assignment 지평을 바꿀 수 없었다.

### 적용한 변경

- **pedal expansion `k`** (`haic_agent/networks.py`): 종방향 좌표를 `tanh(u / k)`로 두고 페달 스케일을 `MAX_GAS * k`, `MAX_BRAKE * k`로 곱한다. `u`가 작은 순항 구간에서는 두 인자가 상쇄되어 기존 checkpoint가 내던 가속이 유지되고, 포화 구간에서만 천장이 `k`배로 열린다. `k = 1.0`은 기존 변환과 완전히 같고, checkpoint metadata의 `pedal_expansion`이 없는 기존 checkpoint는 자동으로 `1.0`이다. brake 천장이 시뮬레이터의 `1.0`을 넘지 않도록 `MAX_PEDAL_EXPANSION = 3.5`로 제한했다. 역변환과 log-Jacobian도 `k`를 반영한다(`MAX * k * tanh(u/k)`의 도함수는 `MAX * sech²(u/k)`로 `k`가 상쇄된다).
- **속도 부족 비용** (`training/train_policy.py`): 랩타임에 무관했던 속도 보너스를 제거하고, 결정마다 목표 순항 속도에 못 미치는 만큼 `SPEED_SHORTFALL_PENALTY = 0.6`을 청구한다. 느린 랩은 더 많은 결정 동안 더 큰 비용을 내므로 같은 거리라도 빠른 랩이 엄격히 유리하다. 충돌·이탈 상태에서는 기존과 같이 청구하지 않는다. 화면에 위험이 보이면 `HAZARD_SPEED_RELIEF = 0.5`만큼 목표 속도를 낮춰 정당한 제동을 벌하지 않되, 목표를 0으로 만들지는 않아 장애물 앞에서 정지해 비용을 회피하는 경로를 막는다.
- **교사 페달 범위 수정** (`training/imitation.py`): `teacher_pedal_reference()`가 교사 인스턴스의 실제 `max_gas`/`max_brake`를 읽고, 라벨을 actor의 확장된 페달 범위로 매핑한다. 기본 교사(`max_gas 0.12`)의 라벨은 그대로이고, 빠른 프로필이 더 이상 잘리지 않는다. `cap_fraction`은 인자로 노출했다(기본값 `0.75` 유지).
- **진단 지표**: rollout에 `throttle_limit`, `gas_saturation_fraction`, `mean_speed`, `max_speed`를 남기고 평가에도 `mean_speed`, `max_speed`를 추가했다. "느린 이유가 신중함인지 상한인지"를 다음 실험에서 바로 읽을 수 있다.
- **`gamma`/`gae_lambda` 연결**과 `--pedal-expansion`, `--gamma`, `--gae-lambda` CLI를 추가했다. 기본값은 기존과 동일하다.
- 제출 경로(`agent.py`), 패키징(`training/package_submission.py`), closed-loop 평가(`training/evaluate_closed_loop.py`), CEM planner가 모두 checkpoint의 `pedal_expansion`을 따라간다. 행동 계약(3값, gas/brake 상호 배타, 유한값)과 metadata 없는 기존 checkpoint 동작은 바뀌지 않는다.

### 측정 결과

**1) 가중치를 고정한 채 상한만 연 zero-shot 확인** (`scratch_eval_expansion.py`, tune 맵 4개, 600 decision cap). 완주율은 `3/4`로 동일했고 중앙 랩타임만 줄었다.

| expansion | throttle 상한 | 완주 | 중앙 랩 | P90 랩 |
|---|---:|---:|---:|---:|
| 1.0 | 0.12 | 3/4 | 19.88s | 22.58s |
| 1.5 | 0.18 | 3/4 | 19.70s | 22.28s |
| 2.0 | 0.24 | 3/4 | 19.40s | 22.04s |
| 2.5 | 0.30 | 1/4 | 22.76s | 22.76s |
| 3.0 | 0.36 | 3/4 | 19.76s | 22.06s |
| 3.5 | 0.42 | 3/4 | 19.36s | 22.03s |

`2.5`에서 한 episode가 이탈해 완주율이 무너졌다. 재학습하지 않은 가중치로는 `2.0` 위가 불안정하므로 이번 실험의 상한은 비교 교사와 같은 `0.24`(`k = 2.0`)로 둔다.

**2) 전체 tune split zero-shot** (6 episode, 400 decision cap). `k = 1.0`은 checkpoint metadata에 기록된 tune 결과(`4/6`, 중앙 `21.57s`, P90 `27.88s`, 진행률 `0.9014`)를 그대로 재현했다. 변환 변경이 `k = 1.0`에서 동작을 바꾸지 않는다는 확인이다. `k = 2.0`은 완주한 랩은 훨씬 빨랐지만(중앙 `19.40s`, P90 `22.04s`) 완주가 `3/6`으로 하나 줄었다. 상한만 열고 재학습하지 않으면 비교 순서상 개선이 아니다.

**3) 확장된 페달 공간에서 PPO continuation** (`artifacts/haic/pedal-expansion2-speed-reward-ppo4096-lr2e6-seed8101`, source는 obstacle-risk seed 8101 checkpoint, 4,096 transition / 2 update / lr `2e-6` / seed `8101`, `k = 2.0`, 새 속도 보상, wall `883s`).

| 지표 | source (`k=1.0`) | candidate update 1 (`k=2.0`) |
|---|---:|---:|
| tune 완주 | 4/6 | 4/6 |
| tune 중앙 랩 | 21.57s | **21.02s** |
| tune P90 랩 | 27.88s | **24.19s** |
| tune 평균 진행률 | 0.9014 | 0.8037 |
| tune 평균 속도 | 36.40 | 39.02 |
| rollout `gas_max` | 0.11526 (상한의 96%) | 0.18479 |
| rollout `gas_saturation_fraction` | — (신규 지표) | 0.0 |

완주율이 같고 중앙·P90 랩타임이 줄었으므로 `RULES.md`의 비교 순서(완주율 → 중앙 랩 → P90 랩 → 진행률)에서는 tune 기준 개선이다. 평균 진행률은 네 번째 키이고 낮아졌다. update 2는 tune 완주 `1/6`으로 떨어져 저장되지 않았다(`save_best_checkpoint`가 거부).

`gas_saturation_fraction`이 `0.0`이 된 것이 이번 진단의 직접 확인이다. 가속 상한이 더 이상 정책의 선택을 자르지 않는다.

**4) held-out 확인 결과: 이 후보는 개선이 아니다** (`scratch_eval_heldout_expansion.py`, held-out 8 episode, 600 decision cap).

| 지표 | source (`k=1.0`) | candidate (`k=2.0`) |
|---|---:|---:|
| 완주 | 6/8 | 2/8 |
| 중앙 랩 | 19.32s | 26.06s |
| P90 랩 | 22.42s | 27.26s |
| 평균 진행률 | 0.7880 | 0.4288 |
| 평균 속도 | 35.24 | 34.97 |

tune에서 중앙·P90 랩타임이 좋아졌지만 held-out에서는 완주가 `6/8 → 2/8`로 무너졌고 완주한 랩도 더 느렸다. tune 평균 진행률이 `0.9014 → 0.8037`로 떨어진 것이 이미 경고였다. 4,096 transition / 2 update는 `k = 2.0`만큼의 행동 분포 변화를 흡수하기에 부족했다고 본다. **SOTA 포인터는 그대로 두고 제출 ZIP도 만들지 않는다.** 이 후보는 `rejected-heldout`으로 보존한다.

**5) 더 작은 확장으로 재시도해도 완주율은 회복되지 않았다** (`artifacts/haic/pedal-expansion1p5-speed-reward-ppo8192-u4-lr2e6-seed8101`, `k = 1.5`, 8,192 transition / 4 update / lr `2e-6`, wall `1,040s`).

| update | tune 완주 | 중앙 랩 | P90 랩 | 진행률 | 평균 속도 | rollout `gas_max` |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3/6 | 19.56s | 22.20s | 0.8613 | 37.11 | 0.1551 |
| 2 (선택) | 3/6 | 19.52s | 21.94s | 0.8891 | 41.23 | 0.1603 |
| 3 | 3/6 | 19.82s | 22.04s | 0.7850 | 33.39 | 0.1609 |
| 4 | 3/6 | 19.84s | 22.03s | 0.7867 | 36.22 | 0.1587 |

랩타임은 source(`21.57s`)보다 일관되게 `2s` 빨랐지만 완주가 `4/6 → 3/6`으로 떨어졌다. 완주율이 비교 순서의 첫 키이므로 이 후보도 tune을 통과하지 못했고 held-out으로 보내지 않았다. 네 update 모두 `gas_saturation_fraction`이 `0.0`이었다.

### 정리

진단은 측정으로 확인됐다. `gas_max`가 상한의 `96%`였고, 상한만 열어도 완주한 랩이 빨라졌다(`19.88s → 19.40s`). 보상의 속도 항이 랩타임에 무관했다는 것과 교사 라벨이 `0.09`로 잘려 있었다는 것도 코드에서 확인해 고쳤다.

그러나 **두 번의 재학습 모두 랩타임과 완주율을 맞바꿨다.** `k = 2.0`은 tune 완주를 지켰지만 held-out에서 `6/8 → 2/8`로 무너졌고, `k = 1.5`는 랩타임이 `2s` 빨라진 대신 tune 완주가 `4/6 → 3/6`으로 떨어져 held-out에 가지도 못했다. 가속 상한이 랩타임을 제한한 것은 사실이지만, 상한을 여는 것만으로는 더 빠른 제출 후보가 되지 않는다. 지금 정책이 잃는 것은 속도가 아니라 더 높은 속도에서의 코너 진입·장애물 회피 능력이고, 그건 짧은 fine-tune으로 회복되지 않았다. **SOTA와 제출 후보는 바뀌지 않았다.**

### 다음 실험

- `k`를 학습 중 점진적으로 키우는 커리큘럼(예: update마다 `1.0 → 1.25 → 1.5 → 2.0`)을 넣는다. 한 번에 바꾸면 policy가 이미 배운 코너 진입 속도를 전부 다시 배워야 하는데, 이번 두 실험의 예산으로는 부족했다.
- `sprint_guarded` 교사(`max_gas 0.24`)로 시연을 다시 모아 확장된 페달 공간에서 BC 워밍업을 한 뒤 PPO를 잇는다. 라벨 clipping 버그를 고쳤으므로 이제 교사의 실제 가속이 그대로 전달된다. 빠른 속도에서의 코너 진입을 학습 데이터로 직접 주는 경로다.
- 학습 예산을 늘린다. 이번 두 실험은 4,096/8,192 transition으로, 확장 전 checkpoint가 누적한 `19,456` step에 비해 매우 짧다.
- 속도-완주 trade-off를 선택 단계에서 보이도록 tune 기록에 `mean_speed`를 함께 남겼다. 완주율이 같을 때만 랩타임으로 고르는 현재 순서는 유지한다.

## 학습 예산이 부족한 것인가 (2026-09-23)

앞의 두 continuation이 실패한 뒤 "학습 시간이 부족했을 뿐"인지 측정했다. 답은 **예산도 부족하지만, 예산을 늘리는 것만으로는 해결되지 않는다**이다. 선택 신호가 먼저 고장나 있다.

### 1. actor는 사실상 움직이지 않았다

| 대상 | `k = 2.0` / 4,096 step | `k = 1.5` / 8,192 step |
|---|---:|---:|
| 전체 `‖Δθ‖ / ‖θ‖` | 0.1306% | 0.2701% |
| `policy_mean.weight` | 0.0519% | 0.0654% |
| `policy_mean.bias` | 0.0137% | 0.0194% |
| `policy_log_std` | 0.0071% | 0.0091% |
| `value_head.weight` | 0.6598% | 1.5737% |

움직인 것은 critic뿐이다. 새 보상에 value function이 적응하는 중이었고 actor는 거의 그대로였다. source checkpoint의 누적 학습량은 `19,456` decision = `77,824` raw frame으로, 일반적인 CarRacing PPO 참고 예산(수백만 frame)의 2% 수준이다. lr `2e-6`은 과거 `1e-5`가 불안정해서 낮춘 값이라, 단위 wall-clock당 학습량이 의도적으로 매우 작다.

같은 tune 픽셀 400장(source 정책이 실제로 방문한 상태)에서 두 후보를 source와 **같은 pedal expansion으로** 비교한 행동 차이도 미미했다.

| 후보 (expansion 1.0에서 비교) | 평균 \|Δsteer\| | 평균 \|Δgas\| | 평균 \|Δbrake\| |
|---|---:|---:|---:|
| `k=2.0` 가중치 | 0.00222 | 0.00037 | 0.00038 |
| `k=1.5` 가중치 | 0.00841 | 0.00048 | 0.00050 |

기준 규모는 평균 `|steer| ≈ 0.21`, `gas ≈ 0.06`이다. 즉 조향이 평균 1~4%, 가속이 1% 미만 바뀌었다.

### 2. 그런데 그 미세한 변화가 tune 점수를 무너뜨린다

같은 가중치를 expansion 1.0으로 되돌려 tune을 다시 평가했다.

| | tune 완주 | 중앙 랩 | 진행률 |
|---|---:|---:|---:|
| source | 4/6 | 21.57s | 0.9014 |
| `k=2.0` 가중치 @1.0 | 2/6 | 24.57s | 0.8065 |
| `k=1.5` 가중치 @1.0 | 1/6 | 23.04s | 0.7056 |

평균 조향이 `0.008` 바뀌었는데 완주가 `4/6 → 1/6`이 된다.

### 3. 대조 실험: 같은 크기의 무작위 잡음도 똑같이 흔든다

`‖Δθ‖ = 0.0495`(= `k=1.5` 실험의 실제 변화량)만큼 **무작위 방향으로** source 가중치를 흔들고 같은 tune을 돌렸다.

| | tune 완주 | 중앙 랩 | 진행률 |
|---|---:|---:|---:|
| source | 4/6 | 21.57s | 0.9014 |
| random seed 101 | **5/6** | 23.26s | 0.8417 |
| random seed 202 | 3/6 | 20.06s | 0.7806 |
| random seed 303 | 3/6 | 20.36s | 0.7822 |

**무작위 잡음이 `3/6 ~ 5/6` 범위를 만든다.** 학습한 두 후보(`2/6`, `1/6`)도, source(`4/6`)도 이 띠 안이거나 그보다 나쁘다. seed 101은 아무것도 배우지 않고 source보다 완주가 높다(대신 랩타임은 더 느리다).

### 결론

- **예산은 분명히 부족하다.** actor 가중치가 0.3% 미만 움직였고 누적 학습량은 참고 예산의 2% 수준이다.
- **그러나 지금 예산을 늘려도 얻는 것이 없다.** tune 6 episode의 완주율은 해상도가 `1/6 = 0.167`인데, 무작위 잡음의 진폭이 그보다 크다. `save_best_checkpoint`가 이 지표로 고르면 사실상 잡음을 고른다. 계획 문서에 기록된 "tune에서 이겼는데 held-out에서 무너진" 후보들의 패턴이 정확히 이것이다.
- 즉 순서가 틀렸다. **선택 신호를 먼저 고치고, 그다음에 예산을 늘려야 한다.**

### 시간 배분

| | `k=2.0` 4,096 step | `k=1.5` 8,192 step |
|---|---:|---:|
| rollout | 250.8s (28.4%) | 372.2s (35.8%) |
| PPO update | 141.0s (16.0%) | 169.0s (16.3%) |
| tune selection eval | 187.8s (21.3%) | 284.6s (27.4%) |
| full tune eval | 108.6s (12.3%) | 70.4s (6.8%) |
| HUD ablation | 194.5s (22.0%) | 143.6s (13.8%) |
| rollout 처리량 | 16.3 decision/s | 22.0 decision/s |

평가가 wall-clock의 `40~46%`를 쓴다. 그중 HUD ablation은 이 checkpoint가 `use_hud=False`라 두 조건의 결과가 완전히 동일한 무의미한 측정이었다. **`hud_ablation()`이 HUD branch 없는 actor에서는 건너뛰도록 고쳤다**(회귀 테스트 2개 추가). 이것만으로 continuation 한 번당 `14~22%`의 wall-clock이 rollout으로 돌아온다.

전체 파이프라인 처리량은 `4.6~7.9 decision/s`다. 이 속도로 `200,000` decision(= `800,000` raw frame)을 돌리면 약 `7시간`, `1,000,000` decision이면 약 `35시간`이다.

### 다음 순서

1. **선택 신호부터.** tune episode 수를 늘려 완주율 해상도를 잡음 진폭보다 작게 만든다(tune 맵에 train/held-out과 겹치지 않는 seed 추가). 그 전에는 어떤 후보도 "개선"이라고 부를 수 없다.
2. 선택 기준에 잡음 폭을 반영한다. 위 대조 실험의 `3/6~5/6`이 현재 잡음 띠이므로, 완주율 `+1`은 근거가 되지 못한다. 연속 지표(진행률·랩타임)를 잡음 폭과 함께 본다.
3. 평가 비용을 더 줄인다(선택 평가를 매 update가 아니라 N update마다).
4. 그다음에 예산을 늘린다. 최소 수만 decision 규모, 그리고 lr `2e-6`보다 큰 값으로. 지금 설정은 한 번의 continuation이 무작위 잡음보다 작은 변화를 만든다.

## 현황 점검: 완주와 속도 (2026-09-23)

### 채점 규칙이 요구하는 것

`COMPETITION_INFO.md` 기준이다.

- 완주 = 고유 타일 95% 이상 방문 + 정방향 결승선 통과.
- **완주자는 랩타임 짧은 순, 미완주자는 진행도 순.** 완주는 순위의 하드 게이트이고, DNF 하나는 모든 완주자 아래로 간다.
- 손상 100%(충돌 5회), 101회 연속 음수 보상, 영역 이탈, max-steps, invalid action 10회 연속이 미완주 조건이다. **충돌 예산은 4회뿐이다.**

### 지금 어디에 있는가

최신 checkpoint(`site-map-official-domain-steer-recovery-fullsplit-ppo8192-u8-lr2e6-seed8101`)의 `ppo_only` 기록을 맵 종류로 나눴다.

| 분류 | 완주 | 중앙 랩 | 평균 진행률 |
|---|---:|---:|---:|
| 커스텀 맵 (7 episode) | **7/7** | 19.36s | 1.000 |
| 공식 트랙 (7 episode) | **3/7** | 25.40s | 0.679 |

공식 트랙 개별 내역이다.

| episode | 결과 | 진행률 | 충돌 | 손상 | 랩 |
|---|---|---:|---:|---:|---:|
| track1 seed47 | DNF `off_track` | 0.853 | 2 | 0.4 | — |
| track1 seed48 | 완주 | 0.996 | 0 | 0.0 | 23.08s |
| track2 seed106 | DNF `crash` | 0.607 | 5 | 1.0 | — |
| track2 seed107 | 완주 | 0.965 | 2 | 0.4 | 29.98s |
| track1 seed42 | DNF `crash` | 0.166 | 5 | 1.0 | — |
| track2 seed101 | 완주 | 1.000 | 0 | 0.0 | 25.40s |
| track1 seed42 + 사용자 장애물 | DNF `off_track` | 0.163 | 1 | 0.2 | — |

**커스텀 맵은 사실상 풀렸고(7/7, 충돌 0), 채점 대상인 공식 트랙은 3/7이다.** 실패 4건 중 2건은 충돌 5회로 손상이 100%에 닿아 탈락했고, 2건은 코스 이탈이다. 커스텀 맵에서는 충돌이 한 번도 없다.

### 속도는 이미 문제가 아니다

제출할 수 없는 규칙 기반 corridor 교사를 같은 공식 트랙에서 돌린 원본 기록(`artifacts/haic/vision-racing-official-v1/results.json`)이다.

| | 완주 | 충돌 | 랩 | 평균 속도 |
|---|---:|---:|---:|---:|
| corridor teacher, track1 seed42 | ✅ | 0 | 47.28s | 20.68 |
| corridor teacher, track1 seed43 | ✅ | 0 | 49.52s | 21.19 |
| corridor teacher, track2 seed101 | ✅ | 0 | 49.04s | 20.93 |
| corridor teacher, track3 seed201 | ✅ | 0 | 45.76s | 20.38 |
| **S1 PPO actor, track2 seed101** | ✅ | 0 | **25.40s** | — |

같은 트랙·시드에서 우리 actor는 교사보다 **약 1.9배 빠르다**. 교사는 느린 대신 4/4 무사고 완주다. 즉 지금 부족한 것은 속도가 아니라 **그 속도를 완주까지 유지하는 신뢰도**다.

> **정정.** `RESULTS.md`와 `report.pdf`가 교사 기준선을 "공식 Track1/2 22.40/20.92초"로 적어 왔는데, `20.92`는 랩타임이 아니라 track2 seed101의 `mean_speed` 값이다. 실제 랩은 `591`/`613` decision, 즉 `47.28s`/`49.04s`다. 이 표기 때문에 교사가 20초대에 도는 빠른 기준선처럼 읽혔고, 우리 actor의 `25.40s`가 뒤처진 것처럼 보였다. 두 문서를 측정값으로 고쳤다.

### 원인으로 보이는 것

1. **학습 분포가 채점 분포와 어긋나 있다.** `full_site_map_split.json`의 train 22 episode 중 공식은 8개뿐이고 14개가 커스텀·증강 맵이다. 그 14개는 이미 7/7로 푼 분포이고 채점되지도 않는다.
2. **충돌 예산 인식이 없다.** 충돌 5회면 손상 100%로 즉시 탈락인데, 보상은 충돌 1회마다 고정 페널티를 줄 뿐 남은 예산을 상태로 보지 않는다. 공식 트랙 실패 2건이 정확히 5회 충돌로 끝났다.
3. **seed 42는 공식 평가 시드가 아니다**(`README.md`). 실제 평가 시드는 공개되지 않았으므로 seed 42의 16% 지점을 국소 수리하는 접근은 순위로 이어지지 않는다. 필요한 것은 시드 전반의 강건성이다.

### 우선순위 판정

직전 작업(페달 상한 확장, 속도 보상)은 **우선순위가 틀렸다.** 채점 규칙상 공식 완주율 `3/7`이 순위를 결정하고 있고, 완주하는 구간에서 우리는 이미 교사의 두 배 속도다. 페달 확장은 기본값 `k = 1.0`이라 현재 제출 후보 동작에는 영향이 없으니 그대로 두고, 완주율이 회복된 뒤에 다시 꺼내는 것이 맞다.

### 이번에 준비한 것

- `tools/make_official_split.py`: 공식 맵 descriptor는 `track_id`와 `seed`뿐이고 트랙·장애물은 시뮬레이터가 만든다. 새 공식 시드 48개를 생성하고 opt-in 분할 `training/maps/site/official_heavy_split.json`을 만든다. 기존 분할 파일은 건드리지 않는다.

| 분할 | 기존 | 신규 (`official_heavy_split.json`) |
|---|---|---|
| train | 22 (공식 8 / 커스텀 14) | **62 (공식 48 / 커스텀 14)** |
| tune | 6 (공식 4 / 커스텀 2) | **14 (공식 12 / 커스텀 2)** |
| held_out | 8 (공식 3 / 커스텀 5) | 8 (변경 없음) |

train/tune/held_out 시드 중복은 생성 시점에 검사하고, held-out은 이전 결과와 비교 가능하도록 그대로 뒀다. tune을 14로 늘린 것은 앞 절에서 측정한 선택 잡음(완주율 해상도 `1/6`이 무작위 섭동 진폭보다 컸다) 대응이기도 하다.

### 권고 순서

1. `official_heavy_split.json`으로 재학습해 **공식 완주율**을 목표 지표로 삼는다. 커스텀 맵은 이미 포화이므로 더 배울 것이 없다.
2. 선택 지표를 공식 트랙 완주율 우선으로 본다. tune 14 episode면 해상도가 `1/14`로 좁아진다.
3. 충돌 예산을 상태로 다룬다. 남은 손상 여유(`1 - damage`)에 따라 위험 회피 강도를 조절하는 보상·특징을 검토한다. 5회째 충돌은 4회째와 비용이 다르다.
4. 속도(`pedal_expansion`, 속도 부족 비용)는 공식 완주율이 교사 수준에 근접한 뒤에 다시 올린다.

## 목표: 공식 Track 1·2를 15초 안에 완주 (2026-09-23)

### 15초가 요구하는 것

시뮬레이터가 생성한 트랙의 타일 중심 좌표로 길이를 직접 쟀다(`scratch_eval_lap_time_ceiling.py`).

| 트랙 | 타일 | 트랙 길이 | 15초에 필요한 평균 속도 |
|---|---:|---:|---:|
| track1 seed42 | 283 | 994 | **66.3** |
| track1 seed48 | 274 | 962 | **64.2** |
| track2 seed101 | 303 | 1,064 | **70.9** |
| track2 seed107 | 341 | 1,197 | **79.8** |

같은 트랙에서 고정 페달로 직선 가속을 측정한 결과다.

| gas | 40 decision 시점 속도 | 직선 최고 속도 |
|---:|---:|---:|
| 0.12 (현재 상한) | 42.5 | 59 ~ 71 |
| 0.24 | 60.7 | 75 ~ 90 |
| 0.42 | 80.0 | 91 ~ 100 |
| 1.00 | 89 ~ 100 | **100** |

차량 최고 속도는 `100`에서 하드 클리핑된다. 필요한 평균 속도가 `64~80`이므로 **15초는 물리적으로 가능하다.** 다만 코너를 포함한 랩 전체 평균이 최고 속도의 `64~80%`여야 한다.

### 진단 정정: 가속 상한은 지금의 병목이 아니다

앞 절에서 나는 rollout의 `gas_max = 0.1153`이 상한 `0.12`의 `96%`라는 것을 근거로 "상한이 랩타임을 막고 있다"고 적었다. **그 수치는 탐험 잡음이 섞인 표본 행동이고, 실제 주행을 결정하는 결정론적 정책의 명령이 아니다.**

공식 track2 seed101에서 같은 가중치로 상한만 바꿔 결정론적 주행을 추적했다.

| 변환 | 랩 | gas 평균 | gas 최대 | 상한 사용률 | 평균 속도 |
|---|---:|---:|---:|---:|---:|
| gas 상한 0.12 | 25.52s | 0.0613 | 0.0922 | 76.9% | 40.7 |
| gas 상한 1.00 | 24.80s | 0.0712 | **0.1214** | **12.1%** | 41.9 |

상한을 8배 열어 줘도 정책은 **가용 가속의 12%만 요청한다.** 공식 트랙 2개(track1 seed48, track2 seed101)에서 상한을 넓히는 sweep도 같은 결론이다.

| gas / brake 상한 | 완주 | 중앙 랩 | 평균 속도 |
|---|---:|---:|---:|
| 0.12 / 0.28 | 2/2 | 24.24s | 41.27 |
| 0.24 / 0.28 | 2/2 | 23.93s | 42.08 |
| 0.42 / 0.28 | 2/2 | 23.69s | 42.34 |
| 0.42 / 0.98 | 2/2 | 23.75s | 42.35 |
| 0.72 / 0.84 | 2/2 | 23.72s | 42.40 |
| 1.00 / 1.00 | 2/2 | 23.72s | 42.33 |

상한을 전부 열어도 `24.24s → 23.72s`, **2%**다. 즉 **15초는 변환의 문제가 아니라 정책의 문제다.** 지금 정책은 평균 속도 `41~42`로 달리는 주행 스타일을 학습했고, 목표는 `64~80`이다. 약 `1.7배`가 필요하다.

### 이번에 적용한 것

1. **페달 분기 분리** (`haic_agent/networks.py`). 기존 대칭 확장은 brake 상한이 먼저 `1.0`에 닿아 gas를 `0.42` 위로 열 수 없었다. throttle과 brake가 각자의 확장 계수를 갖도록 바꿔 `gas 1.0`(`throttle_expansion = 1/MAX_GAS`), `brake 1.0`(`brake_expansion = 1/MAX_BRAKE`)까지 도달한다. 순항 명령 보존, 역변환, log-Jacobian은 분기별로 맞췄고 회귀 테스트를 추가했다. 확장 `1.0`은 여전히 기존 변환과 동일하고 기존 checkpoint는 그대로 동작한다. **이것은 필요조건이지 충분조건이 아니다** — 정책이 더 밟기를 원하게 된 뒤에야 의미가 있다.
2. **속도 목표를 트랙에서 다시 잡았다** (`training/train_policy.py`). `SPEED_TARGET`이 `50.0`이었는데, 이는 15초 요구치 `64~80`보다 낮다. 평균 속도 `50`으로 달리면 랩이 약 `21초`인데 보상은 한 푼도 청구하지 않았다. 측정치에 맞춰 `70.0`으로 올렸다.
3. `--speed-target`, `--speed-shortfall-penalty`, `--throttle-expansion`, `--brake-expansion`을 CLI로 열어 실험마다 파일을 고치지 않고 sweep할 수 있게 했다. 값은 checkpoint metadata에 기록된다.

### 정직한 평가

- 15초는 물리적으로 가능하지만, 현재 정책에서 **평균 속도를 1.7배 올리는 일**이다. 노브 조정이 아니라 새 주행 행동(직선 전개 가속, 코너 전 강한 제동, 레이싱 라인) 학습이다.
- 동시에 공식 트랙 완주율은 `3/7`이다. 속도를 밀면 완주가 더 어려워진다. 두 목표가 정면으로 충돌하는 구간에 있다.
- 앞 절에서 측정한 학습 처리량은 파이프라인 전체 `4.6~7.9 decision/s`이고, continuation 한 번이 actor 가중치를 `0.3%` 미만 움직였다. 이 예산으로는 주행 스타일이 바뀌지 않는다.

### 15초를 목표로 한다면 필요한 것

1. **속도 보상이 타일 보상과 견줄 크기여야 한다.** 시뮬레이터 자체 보상은 랩당 약 `+1000`으로 고정이고 시간 비용은 결정당 약 `0.45`다. `25초 → 15초`(125 decision 절약)는 약 `56`, 즉 랩 보상의 `5.6%`에 불과하다. `SPEED_SHORTFALL_PENALTY`를 올려 이 차이를 수십 %로 키우는 sweep이 필요하다(`--speed-shortfall-penalty`).
2. **고속 코너링을 데이터로 준다.** `sprint_guarded` 교사는 커스텀 맵에서 평균 속도 `46.8`, 랩 `17.32s`를 낸다. 교사 라벨 clipping 버그를 고쳤으므로 이제 확장된 페달 공간으로 그대로 전달된다. 다만 이 교사도 공식 track1에서는 `14.5%`에서 실패하므로, 공식 트랙용 빠른 교사 프로필 보정이 선행되어야 한다.
3. **예산.** 목표 주행 스타일 자체가 바뀌어야 하므로 수만~수십만 decision 규모가 필요하다. 측정 처리량 기준 `200,000` decision이 약 `7시간`이다.

## 제출 가능 여부 점검 (2026-09-23)

### 기존 ZIP은 지금 그대로 제출 가능하다

`artifacts/haic/submission/haic-obstacle-risk-ppo-actor.zip`을 오늘 코드로 다시 검증했다. ZIP은 자기완결형이라 이번 변환 변경의 영향을 받지 않는다.

| 항목 | 측정 | 제한 |
|---|---:|---:|
| import + 생성 | 2.08s | 10s |
| reset | 0.0s | 5s |
| act | 0.0s | 5s |
| 프로세스 RSS | 193.3 MiB | 1,024 MB |
| 아카이브 | 3.85 MB / 10 파일 | 500 MB / 1,000 파일 |

레이아웃(루트 `agent.py`), 정적 검사(금지 import·동적 실행 없음), planner 비활성, strict checkpoint loading, 유한 action 모두 통과했다.

### 어느 checkpoint를 넣을지

train에 없는 공식 에피소드 7개를 같은 프로토콜(800 decision cap, 결정론적 행동)로 다시 돌렸다(`scratch_eval_official_candidates.py`).

| checkpoint | 공식 완주 | 중앙 랩 | 평균 진행도 | held-out 공식 3개 진행도 |
|---|---:|---:|---:|---:|
| obstacle-risk (현재 ZIP) | 2/7 | 25.46s | 0.642 | 0.435 |
| steer-recovery-fullsplit | 2/7 | 25.48s | 0.644 | 0.443 |
| **brake-reward-fullsplit** | 2/7 | 25.38s | **0.762** | **0.716** |

완주 수와 랩타임은 셋 다 사실상 같다. 차이는 **미완주 에피소드의 진행도**이고, 채점 규칙상 미완주자는 진행도로 순위가 매겨지므로 이는 점수에 직접 영향을 준다. 다만 차이의 대부분이 `official-track1-seed42-plus-obstacle` 한 건(`0.152 → 0.989`, 완주 직전까지 감)에서 나온다. 앞서 측정한 선택 잡음 폭을 감안하면 단일 에피소드 근거로 SOTA를 옮길 수 없어 포인터는 유지한다.

기록: `artifacts/haic/official-candidate-ranking/official-candidate-ranking.json`.

### 주의: 이전 기록과의 불일치

`artifacts/haic/eval-steer-recovery-fullsplit-heldout/episodes.jsonl`은 `official-track2-seed107`을 `29.98s` 완주로 남겼지만, 이번 재실행에서는 진행도 `0.724`, 손상 `1.0`으로 미완주였다. 평가 경로(`evaluate_closed_loop` 대 `evaluate_policy`)나 cap 차이일 수 있어 원인을 아직 확정하지 못했다. 제출 후보를 확정하기 전에 두 경로를 같은 설정으로 맞춰 재확인해야 한다.
