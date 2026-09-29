# 속도 상태 학습 주기 — 2026-09-28

## 목적과 가설

기존 두 고속 정책 결합은 새 3트랙 tune에서 속도가 빨라졌지만 각각 완주율이 1셀 낮았다. 픽셀만 보는 조향·가속 정책 자체가 빠른 코너의 상태와 복구를 배워야 한다. 기존 선택 PPO의 학습 상태를 읽기 전용으로 이어받고, TRAIN 맵만 열어 스로틀 표현 범위를 4.0배로 늘린다. 기본 속도 부족 보상(목표 70, 계수 0.6)과 커브 속도 완화는 현재 학습 코드 그대로 쓴다. 보상 항목을 동시에 변경하지 않는다. 학습용 simulator labels는 보상과 보조 목표에만 쓰고 추론 입력은 픽셀이다.

## 고정 프로토콜

- 초기 체크포인트: `artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt`, SHA-256 `3ceb5ebbe2a4bd96944649a693b8ef924c0252fa620b0fd547c7fb494afff199`. 실행 프로필은 이 정확한 SHA만 읽기 전용 허용한다. 원본을 복사하거나 바꾸지 않는다.
- TRAIN: `training/maps/site/official_heavy_split.json`의 `train` 그룹만 로드한다. TUNE·held-out 맵 파일은 학습 동안 열지 않는다.
- 로컬 등록 명령: `train_policy`, `defer-tune=true`, 4,096 environment decisions, 업데이트 4회, model seed 8106, episode당 최대 800 decision, learning rate `2e-6`, visual pixel features 활성, throttle expansion `4.0`. Linux/Python 3.11 CPU에서 실행하며 3,600초 상한을 적용한다.
- 첫 게이트: 학습 원본의 단계별 페달 사용량, 평균·최대 속도, TRAIN 완주/실패, 충돌·손상, 시간·메모리, 생성 checkpoint를 확인한다. 실제 가속 사용과 주행 속도가 바뀌지 않거나 조기 실패만 늘면 새 tune을 열지 않는다.
- 첫 게이트를 통과하면 기존 full-road-guard ZIP을 대조군으로 고정해 트랙 1·2·3 × seed 284–287(12셀/arm)에서 `finish_time_s` 완주율 우선으로 비교한다. 통과 시 새 held-out 288–291, 패키지 그대로 confirmation 292–295를 차례로 한 번만 사용한다. 기존 264–283 예약 셀은 열지 않는다. 새 seed 범위는 실행 직전 충돌을 재확인한다.

이 주기는 속도 학습 메커니즘의 첫 활성 검사다. 4,096 decision 한 번의 학습이 최종 성능을 보장하지 않으며, TRAIN 결과를 후보 완주율로 집계하지 않는다. 대회 사이트 업로드·공식 제출·모델 확정은 하지 않는다.

## 인프라 수정 revision 2

첫 등록 실행은 기존 체크포인트가 `use_hud=false`인데 생성할 모델의 기본값이 `true`라서 학습 시작 전에 중단됐다. 원인과 세 게이트를 원래 run에 `REVISE`로 기록했다. Revision 2는 학습 알고리즘·seed·resource·분리 셀을 바꾸지 않고 `--disable-hud-branch`만 명시하여 체크포인트 구조를 맞춘다. 새 run ID와 plan hash로 설계·구현·실행 기록을 다시 남긴다.
