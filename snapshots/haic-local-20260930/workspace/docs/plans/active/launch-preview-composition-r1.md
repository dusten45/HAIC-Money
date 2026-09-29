# 출발 가속과 전방 커브 조향 결합 — revision 1

## 근거와 목적

출발 가속 ZIP `4043993d9901d58b126980b7bafe4f0154305c6a0bf01e1e7e5bc02aacec2389`은 소비된 진단 셀에서 첫 10% 도달 시간을 약 1초 단축했으나, 다른 소비된 셀에서 완주율이 10/12로 대조군 11/12보다 낮았다. 앞보기 조향 및 속도 조절은 과거 별도 실험에서 행동을 바꿨지만 출발 가속과 결합한 결과는 미상이다. 이 계획은 출발 가속을 버리지 않고 보이는 굴곡에 앞서 조향·감속하는 구성을 검증한다. 근거는 `docs/experiments/launch-surge-20260929.md`와 `docs/experiments/anticipatory-steering-speed-20260928.md`다.

## 고정할 구성

- 원본 full-road-guard ZIP `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40`을 보존한다.
- 그 안의 base driver 위에 `ObstacleFullRoadGuardAgent → LaunchSurgeAgent → AnticipatoryBendAgent → SpeedCoupledPreviewAgent` 순서로 기존 고정 코드를 결합한다. pixel observation과 기본 행동만 추론에 사용한다. simulator 위치·속도·지도는 평가 기록에만 사용한다.
- 현재 wrapper들이 장애물 진단을 읽을 수 있도록 `LaunchSurgeAgent`의 corridor를 외부 wrapper에 명시적으로 노출한다. 조향·gas·brake의 임계값은 재조정하지 않는다. ZIP을 바이트 해시로 고정하고 원본 launch ZIP과 비교한다.
- 관측 가설: 도로 중심의 먼 행이 현재·이전 프레임에서 일관되게 휘면 선행 조향 또는 제동이 실제로 발생해 출발 가속 이후의 이탈을 줄인다. 반증은 preview 동작 0회, 후보만 미완주, 완료율 열세, 무효 행동 또는 실행 제한 위반이다.

## 평가 설계

- 비교 control은 원본 출발 가속 ZIP. 선택된 full-road-guard ZIP 수치는 보조 참조다. 판정은 각 트랙 1·2·3 × 동일 seed 4개, 팔당 12셀, 결승선 통과율 우선, 동률일 때 완주 중앙시간과 등록된 후순위 지표다.
- 먼저 이미 소비된 `2900–2903`에서 활성화만 진단한다. 이 블록은 완주 일반화나 SOTA 근거로 사용하지 않는다.
- tune 후보 seed `7000–7003`, held-out `7004–7007`, confirmation `7008–7011`을 예약한다. 실행 전에 전체 보이는 V2 run manifest와 계획 예약을 다시 조회해 중복·충돌이 있으면 새 계획으로 바꾼다. tune에서 완료율 열세면 held-out을 열지 않는다. held-out 통과 시에만 변경 없는 ZIP을 confirmation에 보낸다.
- 각 episode 최대 2,000 decision, 총 24 episode/블록, 로컬 Linux Python 3.11 CPU, 2 CPU/2 GiB, 실행 한도 1,800초. 정식 환경 코드는 바꾸지 않는다. 원본 trace, 행동 변경 횟수, 출발 10% 시간, 도로 이탈·충돌·손상, import/reset/act 지연, RSS, 무효 행동을 보존한다.
- 정확한 계획 해시를 CLI에 등록하고 설계·실행을 별도 이벤트로 기록한다. 2026-09-29 사용자 상시 로컬 승인 지시를 두 이벤트의 source reference로 사용한다. 대회 사이트 업로드·공식 제출·모델 확정은 포함되지 않는다.
