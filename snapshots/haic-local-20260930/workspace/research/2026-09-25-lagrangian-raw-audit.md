# Lagrangian OFF vs Adaptive 원시 결과 감사

- 감사일: 2026-09-25
- 범위: `artifacts/haic/lagrangian-off-vs-adaptive-20260925/`의 모든 로그와 지표 JSON을 직접 재집계. `research/2026-09-23-lap13-strategy-lab.md`는 참고용으로만 읽음.
- 변경/실행: 이 문서 외 파일은 수정하지 않음. 코드, 테스트, trainer, 시뮬레이터, rollout, 학습, TUNE/공식 평가는 실행하지 않음.

## 결론

계획은 PPO seed 8104/8105 × Lagrangian off/adaptive의 4개 arm이지만, 유효하게 8개 update를 마친 arm은 3개뿐이다. seed 8104 off는 첫 시도가 actor 초기화 해시 불일치로 학습 전에 종료(exit 1)했고, 재시도만 유효 결과로 센다. `seed8105-adaptive/` 결과는 없다. 따라서 두 seed를 짝지은 완전한 2×2 비교가 아니다.

유효한 세 실행은 각각 8,192 decision, 8/8 update(회당 1,024 decision)를 채웠다. 세 실행 모두 lap 완주 0회다. 더 중요한 제한은 입력 계약이다. preflight의 `actor_input`은 `pixel_observation_only`라고 쓰여 있지만, 실제 preflight 및 실행 로그는 `use_hud=true`, `use_visual_features=false`를 기록한다. 따라서 이를 현재의 pixel-vision-only runtime 결과로 해석하거나 SOTA와 직접 비교하면 안 된다. 이 불일치가 해소되기 전까지 이 실험은 HUD 입력 설정의 TRAIN-only 진단으로 한정한다.

같은 seed 8104에서 fresh actor hash와 나머지 설정을 맞춘 내부 비교는 가능하다. 이 한 쌍에서는 adaptive가 충돌 onset을 6→1로 줄였지만, 최대 damage는 양쪽 모두 1.0이고, terminal episode 평균 progress는 0.15885→0.13653, episode 평균 속도는 38.441→37.677 m/s로 낮아졌다. 완주가 하나도 없고 두 번째 seed의 adaptive arm도 없으므로 안전성·주행성 개선의 일반적 증거나 우승 전략으로 볼 수 없다. 다만 해당 HUD 설정에서 관찰된 단일 seed의 안전/진행 trade-off 신호다.

## 원시 로그 재집계

완료된 episode는 각 update의 `train_episode_outcomes`에서 세고, 그 update에서 끝나지 않은 주행 조각은 `incomplete_train_episode_outcomes`로 따로 다뤘다. 완주율·평균/max progress·episode 평균 속도는 종료 episode 행에서 계산했다. decision 가중 속도와 gas/brake는 종료 행과 조각의 decision 수를 가중치로 계산했다. 충돌 onset/decision 총계는 모든 행을 포함해야 JSON 지표와 일치한다. seed8104-off에서 충돌 1회/decision 1회가 update 경계의 조각에만 기록되어, 종료 episode만 합치면 요약보다 각각 1씩 적게 나온다.

| 실행 (PPO seed / arm) | 종료 episode / lap 완주 | 종료 경계 (off-track / damage cap) | 평균 progress / 최고 progress | 속도 평균 (episode / decision 가중 / 최고) m/s | 충돌 onset (충돌 episode / collision decision) | 최대 episode damage | 최종 λ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8104 / off | 22 / 0 | 21 / 1 | 0.158851 / 0.321138 | 38.4412 / 38.8210 / 61.9871 | 6 (3 / 8) | 1.0 | off |
| 8104 / adaptive | 23 / 0 | 22 / 1 | 0.136528 / 0.329268 | 37.6769 / 39.6385 / 61.9871 | 1 (1 / 5) | 1.0 | 31.8333 |
| 8105 / off | 23 / 0 | 23 / 0 | 0.139711 / 0.266932 | 38.6268 / 39.5267 / 64.6932 | 1 (1 / 2) | 0.4 | off |

표에서 `episode / decision 가중` 속도는 다른 분모다. 전자는 종료 episode별 속도 평균의 평균이고, 후자는 모든 8,192 decision을 episode/조각의 decision 수로 가중한 평균이다. 둘 다 원시 행에서 재현되어 지표 JSON과 일치한다.

지표 JSON의 `damage` 값(1.0, 1.0, 0.4)은 raw `damage_max`의 최댓값과 일치한다. 이를 평균 피해량으로 읽으면 안 된다. 종료 episode의 `damage_final` 산술평균은 각각 0.06364, 0.04348, 0.01739였다. 마찬가지로 `episodes`는 종료된 rollout 수이고, `completed`는 lap 완주 수라서 모두 0이다.

### 학습 예산·초기화·맵

- 실제 완료 arm: `seed8104-off`, `seed8104-adaptive`, `seed8105-off`. 각 로그에서 `step=8192`, `training_steps=8192`, `updates_completed=8`; update의 step 합과 종료+조각 행의 decision 합도 각각 8,192.
- 계획된 PPO seed는 8104와 8105이며 rollout seed도 각 PPO seed와 같다. 두 seed 모두 off/adaptive 초기화 쌍은 preflight상 같은 hash로 계획되었다.
  - seed 8104 두 arm의 fresh actor SHA-256: `cfa1dd799ef87eb0c1c454088686f798530738e5bab76d2ccc7acb30ce5aa04a`
  - seed 8105 계획 actor SHA-256: `c2b37f16bd925158c2cd406ba0651a9b4d0d9ff9a75f8e199a5c9570e4209b65` (실행된 off arm만 확인됨)
- 결과 JSON은 모두 `fresh_random_actor`, `checkpoint=null`, optimizer state 0, teacher warm-up 0이다. 기존 SOTA/교사 checkpoint에서 이어 학습한 결과가 아니다.
- 네 실행 명령의 공통 예산/주요 설정은 8,192 steps, 8 updates, max decisions 2,000, learning rate `2e-6`, gamma `.99`, GAE `.95`, throttle expansion `3.5`, brake expansion `1.0`, speed target `70`, speed shortfall `.6`이다. preflight는 `--lagrangian-mode`만 arm 간 변경으로 기록한다.
- 실제 로드 목록은 TRAIN 그룹의 custom map 2개, 총 네 map/seed cell이다. official/non-TRAIN 파일은 열지 않았고 TUNE은 deferred다.

| TRAIN map ID | map seed |
|---|---|
| `custom-track-haic-obstacles-20260920` | 20260920, 20260924 |
| `custom-track-haic-train-20260921` | 20260921, 20260925 |

각 유효 arm은 위 네 cell만 포함한다. 종료 episode 수는 각각 off8104 `(8, 6, 2, 6)`, adaptive8104 `(8, 6, 3, 6)`, off8105 `(7, 6, 6, 4)` 순이다. 이는 같은 두 map geometry를 반복한 TRAIN 진단이지, 서로 독립적인 지도 네 개의 일반화 평가가 아니다.

### λ 및 cost 원시 기록

adaptive 설정은 초기 λ 30, cost budget `.05`, dual step 5, window 8 completed TRAIN episodes, `cost_gamma=1`, λ 범위 `[0,60]`이다. 완료 episode cost를 8개씩 모은 첫 두 window에서 평균 cost가 모두 `.233333`으로 budget을 넘었다. 따라서 λ가 `30 → 30.916667 → 31.833333`으로 두 번 갱신되었다. 총 종료 episode는 23개여서 세 번째 완전한 8-episode window는 없었다. update별 조각은 λ window에 포함되지 않는다. 이 λ 동작은 adaptive 로그의 window 기록으로 확인했다.

## 불완전/실패 산출물과 집계 차이

- `seed8104-off/console.log`는 첫 시도의 traceback이며 `initialized actor does not match the paired preflight hash`로 학습 전에 실패(exit code 1). `console-attempt-2.log`는 재시도이며 8 updates와 exit code 0을 확인해 표에 사용했다. 두 시도를 독립 arm처럼 세지 않았다.
- seed8105 adaptive의 preflight 명령/hash는 존재하지만 해당 디렉터리·console·metrics 결과는 없다. 계획된 값은 실제 학습 결과가 아니다.
- 충돌 합계는 완료 episode만 합산하지 않고 update 중간 조각까지 포함해야 한다. 이 원칙으로 raw의 collision onset/decision 합계가 metrics JSON과 모두 일치했다.
- raw episode 행에서 종료 episode의 평균 progress/속도, 최고 progress, decision 가중 속도를 다시 계산한 값은 metrics JSON과 일치했다. `damage`는 평균이 아니라 raw 최대 `damage_max`와 일치했다.
- 세 지표 JSON 모두 `status=diagnostic-only-incomplete-pair`, `restriction_status=pass`, `tune_run=false`, `held_out_or_official_maps_loaded=false`, `transfer_to_sota_eligible=false`다. preflight의 `execution_started=false`는 실행 전 snapshot이고, 이후 실제 실행 여부는 각 console의 8 updates와 exit code 0으로 확인했다.

## 입력 계약과 해석의 한계

preflight의 문자열 필드 `actor_input=pixel_observation_only`와 실행 설정은 일치하지 않는다. 실행 console 세 개 모두 HUD를 켜고 visual features를 끈 것으로 기록한다(`use_hud=true`, `use_visual_features=false`); 결과 JSON도 같은 architecture다. temporal features는 꺼져 있다. 이 감사에서는 tensor 구성 코드를 열어 실제 actor 입력의 내부 의미를 임의로 추정하지 않았다. 따라서 문자열 metadata만 믿고 이 결과를 pixel-only vision 실험으로 분류할 수 없다. 다음 pixel runtime 비교 전에 preflight, trainer, 평가 runner의 입력 계약이 모두 같은 값을 선언하도록 확인해야 한다.

해석 가능한 범위는 좁다. seed8104의 같은 fresh actor에서 Lagrangian 모드만 바뀐 HUD 설정의 한 쌍은 내부 방향성 비교로 남길 수 있다. adaptive는 그 한 쌍에서 충돌이 적었지만, lap 완주가 없고 평균 progress와 episode 평균 속도는 낮았으며 최대 damage는 같았다. seed8105-off는 두 seed의 off 기준선 일부일 뿐이고 누락된 adaptive arm을 대신할 수 없다. current pixel-only runtime, SOTA, 미공개 맵 또는 대회 성능에 대한 주장은 불가하다.

## 후속 가설 및 중단/진행 기준 (제안만, 실행 안 함)

1. 먼저 입력 계약을 고정한다: pixel-only actor라면 preflight와 실행에서 `use_hud=false`, `use_visual_features=true`, `use_temporal_features=false`가 일치해야 하고, `actor_input` 설명도 실제 설정과 일치해야 한다. seed별 fresh paired actor hash, 0 optimizer state, TRAIN-only map gate를 실행 전에 확인한다. 이 문서의 HUD 결과를 새 baseline에 섞지 않는다.
2. B의 현재 trainer/preflight 작업과 공유 학습 슬롯이 닫힌 뒤, B의 Lagrangian 변수나 C의 action-repeat/recurrent 변수를 섞지 않는 판별 후보는 strategy lab에 이미 등록된 pixel hazard-potential 비교다: Lagrangian off를 양쪽 고정하고 `β=0` 대 `β=0.10`만 바꾼다. `Φ(risk)=-risk`, `F=β(γΦ(next)-Φ(now))`, `γ=.99`; 동일 fresh actor/optimizer, PPO seed 8104/8105, 8,192 decisions/8 updates, 같은 TRAIN map cells를 짝지어 사용한다. 이는 현재 감사에서 실행되거나 검증된 결과가 아니라 후속 가설이다.
3. **현재 중단:** 이 Lagrangian 결과는 SOTA/제출 승격이나 held-out 평가에 사용하지 않는다. 완주 0, seed8105 adaptive 누락, 입력 modality 불일치가 해소되지 않았다.
4. **진행 게이트:** B가 슬롯을 반환하고 pixel 입력 계약/hash/map preflight가 일치할 때만 별도 승인된 후속 실험을 시작한다. shaping 후보는 양쪽 seed에서 위험 감소 전이와 collision/damage가 개선되고, 그 안전 이득이 completion/progress/pace 또는 under-13 성능 악화만으로 얻어진 것이 아닐 때에만 한 번의 locked evaluation 후보로 남긴다. 방향이 seed 간 재현되지 않거나 주행 성능을 악화시키면 예산을 확대하지 않고 기각한다. 어떤 경우에도 2-seed TRAIN screen만으로 SOTA를 승격하지 않는다.

## 확인한 원시 파일 및 SHA-256

아래 hash는 이번 감사에서 직접 계산했다. `seed8104-off/console.log`는 실패 시도이고, `console-attempt-2.log`가 그 재시도 성공 로그다. `seed8105-off/console.log`는 별도의 정상 완료 arm이다.

| 파일 | SHA-256 |
|---|---|
| `preflight.json` | `7d43e29ccf80d22ce61ee48e4c8a7ec72e7d21d4fc39e01fa0fdaa7de0237426` |
| `seed8104-off/console.log` (failed attempt) | `7fe96cf5286a4bc44813bc5acd44984b677bdb15b2e84a3b56ca389cb38822e6` |
| `seed8104-off/console-attempt-2.log` | `969d50b707a786e323af9e5c6254758641d07041e87f75e674da98085bd33a65` |
| `seed8104-off/training-result.json` | `3c4e1e0ddf610e26ec5811f1dae5d33e227cde7b4c528132619e2282b4803eb9` |
| `seed8104-adaptive/console.log` | `51d83b8d84f1621b4bfc2fe4b6b245038a4a1d806fec6959cd23708ccf1779c8` |
| `seed8104-adaptive/training-result.json` | `60cbea309fbf0abe9aafb4a5e31e6fa8abdfaf8310eb79ff273e0c92f746ac59` |
| `seed8105-off/console.log` | `7d68eb4a168da497e0c762bfb55a62d46e2032aa7af15ed0e709bcf8c46e2473` |
| `seed8105-off/training-result.json` | `a7e906320a3a8b8f88116739f9322d1fe24f8dceeb9b05c842343de11f33ce0f` |
