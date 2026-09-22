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
