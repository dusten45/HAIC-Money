# Strategy history

Append decisions below; do not rewrite earlier entries. Each entry must include date, hypothesis, candidate/control, registered comparison, gate outcomes, decision, and evidence paths. A rejection is retained with its raw evidence.

The committed pre-v2 local README with older implementation and experiment notes remains in Git commit `0bcd59b:README.md`; the then-uncommitted corridor benchmark is preserved below. The official participant template is pinned separately under `docs/sources/official-participants/`.

| Date | Strategy and hypothesis | Comparison | Gate outcomes | Decision | Evidence paths |
|---|---|---|---|---|---|

## Pre-v2 provisional strategy lanes

The following S1/S2/S3 lane choices were recorded in the former `RULES.md` and `RESTRICTIONS.md`. They are historical implementation choices, not binding competition restrictions or v2 promotion decisions. The new workflow must validate any candidate under the current rules and registered gates.

| ID | 전략 | 당시 제출 runtime 분류 | 당시 용도 |
|---|---|---|---|
| S1 `ppo_actor_only` | 시각 PPO actor-only | 허용·기본 | 제출 주체. train-only obstacle-risk reward와 learner-state DAgGER는 이 lane의 학습 변형으로만 기록한다. |
| S2 `ppo_cem` | PPO actor + latent dynamics CEM | 비교 전용 | 짧은 후보 행동을 dynamics로 평가한다. actor-only보다 좋아질 때만 후보로 유지한다. |
| S3 `vision_corridor_teacher` | pixel corridor 교사 | 제출 금지 | train-only teacher·진단 기준선. 제출 ZIP과 SOTA runtime에는 넣지 않는다. |

Historical restrictions text also treated teacher/corridor use as train-only diagnostic or warm-up work and excluded it from the then-current ZIP. That describes the earlier strategy selection, not a current official rule.

The earlier selection path sent a tune-selected candidate to held-out evaluation, official Track 1 seed 42, and an `official_plus_custom` obstacle map as separate diagnostics. This was a historical local protocol choice; v2 comparisons require their own registered split and control.

## Pre-v2 corridor benchmark narrative (unregistered)

This is the complete user-authored section moved from the pre-v2 README on 2026-09-26. It is a historical local benchmark, unregistered in the v2 protocol, and ineligible as a submission result or v2 SOTA evidence. Its source is the README working-tree section 15 preserved below without changing its claims.

## 15. 학습 교사의 속도·장애물 벤치마크

픽셀 corridor 후보는 도로 중심선과 HUD 속도를 읽어 가감속하고, 밝은 장애물 픽셀을 보고 회피합니다.
`act()`는 프레임만 사용하며, 별도 벤치마크가 완주·속도·가속도·충돌을 기록합니다. 공식 트랙의 내장 장애물,
사용자 전용 맵의 장애물, 공식 트랙에 추가 장애물을 얹은 경우를 따로 비교합니다. `fast_plus`, `sprint_guarded` 등은
`--profile`을 반복 지정해 한 번에 비교할 수 있습니다.

```powershell
python -m training.benchmark_corridor `
  --profile fast_plus --profile sprint_guarded `
  --track 1:43 --track 2:102 `
  --site-map-split training/maps/site/site_map_split.json `
  --site-group held_out --site-limit 1 `
  --site-map training/maps/site/official-track1-seed42-plus-obstacle.json `
  --max-decisions 800 `
  --output artifacts/haic/corridor-profile-sweep-obstacle-fix/profile-comparison.json
```

속도와 가속도는 CarRacing 물리 단위의 decision 간 측정값이며, 각 간격은 기본 0.08초입니다. 평균 절대
가속도와 충돌이 없던 프레임의 피크 가속·감속을 봅니다. 충돌 프레임의 속도 급락과 종료 프레임 변화량은
별도 지표라서 브레이크 성능으로 오인하지 않습니다.
DNF 행의 평균속도·가속도는 중단 시점까지의 부분 주행 기록이므로 완주 기록과 직접 비교하지 않습니다.

| 프로필 | 조건 (모드 / 장애물) | 결과 | 평균 / 최고 속도 | 충돌 제외 평균 절대 가속도 | 피크 가속 / 일반 감속 | 충돌·손상 |
|---|---|---:|---:|---:|---:|---:|
| fast_plus | 공식 track 1 / seed 43 (official / 6) | 22.60초 | 47.35 / 68.24 | 13.61 | 65.67 / -45.98 | 0회 / 0% |
| fast_plus | 공식 track 2 / seed 102 (official / 6) | 20.98초 | 46.58 / 63.86 | 14.53 | 65.67 / -53.67 | 0회 / 0% |
| fast_plus | held-out 사용자 맵 / seed 20260923 (custom_only / 5) | 17.56초 | 46.32 / 63.15 | 13.74 | 65.60 / -44.29 | 0회 / 0% |
| fast_plus | 공식 track 1 / seed 42 + 추가 장애물 (official_plus_custom / 7) | DNF, 진행 14.8% | 11.89 / 55.01 | 6.18 | 65.67 / -38.77 | 1회 / 20% |
| sprint_guarded | 공식 track 1 / seed 43 (official / 6) | 22.40초 | 47.85 / 73.21 | 16.46 | 65.67 / -64.57 | 0회 / 0% |
| sprint_guarded | 공식 track 2 / seed 102 (official / 6) | 20.92초 | 47.19 / 67.06 | 17.19 | 65.67 / -75.79 | 0회 / 0% |
| sprint_guarded | held-out 사용자 맵 / seed 20260923 (custom_only / 5) | 17.32초 | 46.76 / 65.46 | 16.51 | 65.60 / -54.71 | 0회 / 0% |
| sprint_guarded | 공식 track 1 / seed 42 + 추가 장애물 (official_plus_custom / 7) | DNF, 이탈 14.5% | 46.62 / 68.81 | 17.28 | 65.67 / -50.52 | 0회 / 0% |

선두 기록(track 1 17초, track 2 20초)과 비교하면 `sprint_guarded`도 각각 5.40초, 0.92초 뒤입니다.
`sprint_guarded`는 공식 트랙에서 조금 빨라졌지만, 추가 장애물 시험에서는 충돌 전 도로를 벗어나 DNF입니다.
`fast_plus`는 같은 시험에서 충돌 1회와 손상 20%를 기록했습니다. 따라서 장애물 옵션도 평가에 포함했으며,
두 프로필 모두 추가 장애물 상황은 아직 통과하지 못했습니다. 반복 실행 시 추가하는 공식 트랙 맵은
진행도 14.5%, 차선 위치 0.6, 반경 1.2의 사용자 장애물을 정의합니다. `custom_only`의 여러 시드는 같은 도로
설계의 반복이므로 독립 지도 수로 세지 않습니다. `fast_plus` 시험의 충돌 속도 급락 -513.67은 브레이크가
아니며, 일반 주행의 피크 감속은 -38.77입니다.

이 수치는 다른 알고리즘 경로인 픽셀 corridor 후보의 결과이며, PPO 제출 ZIP의 기록이 아닙니다. 확인한
PPO ZIP은 공식 track 1/seed 42에서 진행 13.8%, track 2/seed 101에서 6.9%, held-out custom-only 맵에서
3.8%에 그쳤고, 세 번 모두 완주하지 못해 랩타임을 기록하지 못했습니다.
따라서 corridor 후보는 더 나은 비교안이지만, official_plus_custom 회피까지 통과하기 전에는 제출 후보로
확정하지 않습니다.
