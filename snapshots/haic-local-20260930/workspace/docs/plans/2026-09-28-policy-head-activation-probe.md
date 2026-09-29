# 정책 행동층 학습 활성 진단 — 2026-09-28

사용자의 이번 대화 지시(“그냥 여기서 계속 반복… 그냥 알아서 할”)에 따라 로컬 구현·TRAIN 학습을 이어간다. 이는 대회 사이트 업로드, 모델 확정 또는 공식 제출 지시가 아니다. 이 진단은 네 방향 r2 배치 전의 **TRAIN 전용 활성 조사**이며 제출 후보 비교나 SOTA 증거로 쓰지 않는다.

## 관찰과 단일 가설

직전 등록 학습은 가속 상한을 0.12에서 0.48로 늘렸지만 마지막 TRAIN 평균 gas 0.0674, 평균 속도 30.32였다. `policy_mean.weight` 상대 변화는 0.0601%였다. 당시 CLI는 전체 학습률 `2e-6`만 노출했고, 학습 함수 안에 이미 구현된 `policy_mean_learning_rate`는 사용할 수 없었다. **가설:** 행동층 갱신률이 너무 낮아 현재 속도 목표가 페달 행동으로 전파되지 않았다. 이 가설은 아직 원인 확정이 아니다.

## 고정 진단

- 출발 actor: 직전 v2 TRAIN 결과 `artifacts/haic-research-v2/speed-state-ppo-continuation-hudfix-20260928/training/policy.pt`. 두 팔이 같은 파일의 actor 가중치만 읽고 optimizer는 새로 시작한다. 원본은 수정하지 않는다.
- TRAIN 파일: `training/maps/site/official_heavy_split.json`의 train 그룹만. `defer-tune=true`; TUNE/held-out/confirmation/blind는 열지 않는다.
- 두 팔 모두 seed 8106, 4,096 decisions, 4 PPO updates, episode 상한 800, **교사 예열 0회**, 기본 학습률 `2e-6`, throttle expansion 4.0, visual features 활성, HUD branch 비활성. 한 팔은 행동층 학습률도 `2e-6`; 다른 팔만 `2e-4`로 바꾼다.
- 각 팔 3,600초 상한, 순차 Linux Python 3.11 CPU 실행. 입력 checkpoint·split·source 해시를 등록하고 사용자 지시를 근거로 설계·구현·실행 기록을 별도로 남긴다.
- 메커니즘 활성: 같은 고정 TRAIN 프로토콜에서 후보의 후반 rollout 평균 gas가 대조군보다 최소 0.02 높고, 행동층 가중치 변화가 대조군보다 커야 한다. 완주/손상 악화는 별도로 보고하며, 활성만으로 개선을 선언하지 않는다. 이 기준 미달이면 새 TUNE 셀을 열지 않는다.
- 활성 시에도 네 방향 r2 배치와 신규 TUNE 셀 비교는 별도 등록 후 진행한다. 학습 로그의 속도·보상은 공식 점수가 아니다.

첫 대조군 실행은 `initialize-from`의 기본 교사 예열 3회가 계획에 없는 시범 수집을 시작한 것을 확인하여 중단했다. 이 인프라 무효 실행은 성능 주기로 세지 않는다. revision 2는 예열 0회를 등록 프로필의 필수값으로 고정하고 새 run ID/plan hash를 사용한다.
