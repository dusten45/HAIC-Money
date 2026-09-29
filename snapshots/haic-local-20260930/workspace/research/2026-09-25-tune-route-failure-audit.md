# TUNE 초반 코스 이탈 분석 — 2026-09-25

## 원시 증거

- 잠금 평가: `artifacts/haic/lagrangian-fresh-runtime-architecture-20260925/tune-evaluation.json`
- 평가된 정책: fresh PPO `seed8104/8105 × Lagrangian off/adaptive`, 각 8,192 결정·8회 PPO 업데이트
- 입력: `use_hud=false`, pixel-derived visual features 활성, temporal feature 비활성, throttle expansion 3.5 (gas limit 0.42), teacher warm-up 0, PPO learning rate `2e-6`
- 여덟 에피소드 전부 완주 0, 13초 이내 완주 0, 충돌 0, damage 0. 전부 `off_track`으로 종료했고 평균 진행도는 `0.090535`였다. 네 팔의 평균 속도는 40.03–40.45 m/s, 최고 속도는 58.58–58.68 m/s였다.
- 두 환경 seed 표시는 같은 `custom-track-haic-tune-20260922` 지도 형상 하나를 반복한다. 따라서 8회는 독립 지도 8개가 아니다.

## 실패 지점과 행동

TUNE 지도에는 중심선 243점, 폭 8.0, 장애물 5개가 있다. 첫 장애물은 진행도 0.2467에 있다. `seed8104-off` 기록에서 진행도 0.0658인 decision 36에는 차와 가장 가까운 중심선 점 사이 거리가 2.85였지만, decision 41에는 진행도 0.0741에서 11.02로 벌어졌다. 폭 8.0의 반폭 4.0보다 훨씬 크므로 첫 장애물보다 앞선 첫 굽이에서 이미 차로를 벗어났다. 진행도는 이후 0.090535에 멈췄다.

평가 중 결정적 행동은 사실상 상수였다. 네 팔 모두 steer 표준편차가 `0.00008–0.00016`이었고, steer 평균은 `−0.12085`에서 `−0.12125`, gas 평균은 `0.06588–0.06602`였다. 이는 새 actor의 초기 bias인 steer `−0.12`, gas 약 `0.066`과 거의 같다. 장애물 위험에 반응할 구간까지 도달하지도 못했다. 이 자료는 장애물 탐지보다 **초기 경로 추종 행동을 actor가 학습하지 못한 것**을 우선 원인으로 지목한다.

학습 기록도 같은 방향이다. seed8104 adaptive는 update 1→8에서 gas 평균 `0.0691→0.0634`, 평균 속도 `40.54→37.60 m/s`였고, update 8 rollout의 평균 진행도도 0.134–0.147에 머물렀다. 이 실행은 랜덤 초기 actor, teacher warm-up 0, 전체 PPO learning rate `2e-6`, 총 8회 업데이트였다. 학습률 하나만 원인이라고 단정할 수는 없지만, 학습 뒤 결정적 actor 행동이 초기 bias에서 거의 변하지 않았다는 직접 관측과 일치한다.

## 속도 병목

- 13초 완주에 필요한 평균 속도는 등록된 트랙 길이에 따라 약 74–92 m/s다. 이 평가의 평균 40 m/s는 목표의 대략 절반이며, 최고도 약 59 m/s다.
- 코드상 gas 상한은 0.42지만 actor의 gas 명령은 약 0.066에 머물렀고 훈련 rollout 최대 gas도 약 0.20, 상한 포화율은 0이었다. 당장은 최대 gas 상한보다 정책이 가속 명령을 학습하지 못한 점이 병목이다.
- 추론 `act` 지연 p95는 약 16 ms로 5초 제한보다 충분히 작다. 훈련은 8,192 결정당 약 303–381초였고 측정상 rollout이 전체의 약 77–78%를 차지했다. 따라서 랩 속도 문제와 훈련 wall time 문제는 다르다. 실행 계약 때문에 학습/평가는 공유 시뮬레이터에서 하나씩 실행한다.

## 판별된 다음 단계

1. 첫 굽이에서 시각 입력을 조향으로 바꾸는 능력을 먼저 회복한다. 원인상 충돌 penalty, hazard shaping, global throttle 증가는 첫 개입으로 부적절하다.
2. 다음 단일변수 PPO screen은 **TRAIN-only corridor-teacher 행동 warm-up 유무**다. warm-up 뒤에는 PPO만으로 학습하고 제출 actor에는 teacher나 privileged state를 넣지 않는다. 양쪽 모두 동일한 2개 model seed, 동일한 8,192 PPO 결정·8 update, 동일한 `2e-6` PPO learning rate와 reward/속도 설정을 쓴다. 현재 no-warm-up paired 결과가 대조군이다. 이전 30-epoch teacher warm-up 시도는 PPO learning rate `3e-4`에서 첫 TUNE update 이후 붕괴했고 높은 KL/gradient를 기록했으므로, 이번 비교는 그 높은 PPO learning rate를 재사용하지 않는다.
3. 판정 기준은 TRAIN의 진행도/완주와 조향의 입력 반응성, 동결 후 TUNE 완주율·미완주 진행도·속도다. 같은 TUNE 형상만으로 일반화했다고 결론내리지 않는다. 경로 완주가 회복되면 별도 단계에서 clear-straight 구간의 PPO pace 학습을 검증한다.

이 메모는 결과 해석이며 새 학습이나 평가를 실행하지 않았다. 현재 정책은 제출 후보로 승격하지 않는다.
