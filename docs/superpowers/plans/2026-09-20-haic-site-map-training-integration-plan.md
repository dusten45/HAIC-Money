# HAIC Track Lab 맵 학습·평가 통합 계획

> 상위 설계: `docs/superpowers/specs/2026-09-20-haic-visual-model-based-rl-design.md`

**목표:** Track Lab에서 만든 schema v2 사용자 맵을 PPO와 잠재 전이 모델 학습, 그리고 폐쇄루프 평가에 직접 사용할 수 있게 한다.

**구조:** 사이트 JSON을 독립적인 학습 맵 모델로 검증하고, 저장된 중심선을 따라 공식 CarRacing 물리를 구성한다. 장애물은 사이트와 같은 progress/lateral/radius 좌표에서 생성하며 워밍업 뒤 실제 물리 월드에 붙인다. JSON manifest가 맵 ID 단위로 train/tune/holdout을 분리한다.

**기술:** Python 3.11, Gymnasium, NumPy, PyTorch CPU, 기존 vendored Box2D CarRacing. 추가 런타임 패키지는 없다.

## 전역 제약

- 추론 계약은 4×84×84 grayscale frame stack 입력과 [steer, gas, brake] 출력으로 유지한다.
- 학습 decision은 기본 raw tick 4개이며 reset 전 camera warmup은 50 raw tick이다.
- 맵·시뮬레이터 레이블은 학습/평가 환경에만 전달하고 제출 Agent의 관측 계약은 바꾸지 않는다.
- 개발 Python은 3.11 venv를 사용하고 공식 `requirements.txt`는 변경하지 않는다.
- 맵의 장애물 충돌, 손상, 종료 판정은 기존 HAIC CarRacing/CarEnvironment를 사용한다.

## 파일 구조

- `training/site_maps.py`: 사이트 맵 JSON 검증, 맵 스펙, episode split manifest 로드.
- `training/site_environment.py`: 저장 중심선 CarRacing과 사이트 좌표 장애물 부착.
- `training/env_factory.py`: 일반 track/seed 환경과 site map 환경의 공통 수집 wrapper.
- `training/train_policy.py`, `training/train_dynamics.py`: manifest를 받아 맵별 학습 transition 수집.
- `training/evaluate_closed_loop.py`: 동일한 사이트 맵 episode에서 PPO-only/PPO+CEM 폐쇄루프 결과 기록.
- `training/maps/site/`: 사이트 UI가 생성한 맵 JSON과 기본 split manifest.
- `tests/test_site_map_training.py`: 스키마, split 격리, 실제 커스텀 트랙/장애물 물리와 학습 transition 검증.
- `README.md`: 사이트 맵 학습·평가 명령과 결과 읽는 법.

## 검토 집중 항목

1. 0/1 경계의 progress, lateral ±1, 반경 0.2/4.0, 비유한 숫자와 잘못된 schema 입력을 안전하게 거부한다.
2. 커스텀 트랙이 저장 중심선·도로 폭·출발 방향으로 재현되고 finish line 폭도 일치한다.
3. 사용자 장애물이 warmup 도중 만들어지지 않고 실제 episode 시작 전에 한 번씩 부착되어 충돌 손상으로 연결된다.
4. 같은 map ID가 train/tune/holdout에 중복되지 않으며 한 split 안에서 여러 환경 seed를 허용한다.
5. 일반 track/seed 학습과 submission Agent 입력·출력은 기존 계약을 유지한다.

## Task 1: 사이트 맵 데이터와 split manifest

**파일**

- 생성: `training/site_maps.py`
- 생성: `training/maps/site/*.json`
- 생성: `training/maps/site/site_map_split.json`
- 생성: `tests/test_site_map_training.py`

- [x] Track Lab UI로 생성·저장한 네 개 기술형 맵을 `training/maps/site/`에 보관한다. canonical obstacle map과 train/tune/heldout 맵은 각각 5개 사용자 장애물을 가져야 한다.
- [x] schema v2 custom map과 schema v1/v2 official map의 필수 필드, 숫자 범위, 최대 장애물 수를 검증하는 loader를 테스트 우선으로 구현한다.
- [x] manifest의 `train`, `tune`, `held_out` 항목은 `{ "map": "파일명.json", "seeds": [정수, ...] }` 형식으로 읽는다. 상대 파일은 manifest 디렉터리 아래에만 허용한다.
- [x] 맵 ID가 split 사이에서 재사용되면 거부하고, 같은 split에서 다른 episode seed로 반복하는 것은 허용한다.

**통과 기준:** 사이트에서 저장된 맵 4개가 로드되고, 네 맵 모두 technical 템플릿·폭 8·장애물 5개를 보고한다. 잘못된 schema, 경로 이탈, 맵 누수 테스트는 각 오류의 원인을 설명하며 실패한다.

## Task 2: 중심선 트랙과 장애물 환경

**파일**

- 생성: `training/site_environment.py`
- 수정: `training/env_factory.py`
- 테스트: `tests/test_site_map_training.py`

- [x] loader에서 받은 중심선을 따라 도로 tile을 구성하는 `SiteCustomCarRacing`을 테스트 우선으로 구현하고, map width를 finish-line tracker에도 반영한다.
- [x] reset seed와 사이트의 `start_index`/`direction`을 트랙 시작점에 반영한다.
- [x] 장애물 `progress`를 트랙 인덱스로, `lateral`을 도로 법선 offset으로 바꾸고 Box2D 정적 원형 장애물로 붙인다.
- [x] `CollectingEnvironment.reset()`의 50-tick warmup이 끝난 뒤 사용자 장애물을 추가한다. 일반 트랙 환경과 기존 기본값은 바꾸지 않는다.
- [x] 실제 `.venv`에서 커스텀 맵을 reset/step하고 관측·label·collision·damage 필드를 확인한다.

**통과 기준:** canonical 사이트 맵에서 실제 road tile 점이 JSON 중심선과 일치하고 장애물 body가 정확히 5개 생성된다. reset 두 번 후 중복 obstacle이 없고, 정상 84×84×4 observation 및 transition이 반환된다.

## Task 3: PPO/전이 모델/폐쇄루프 split 연결

**파일**

- 수정: `training/train_policy.py`
- 수정: `training/train_dynamics.py`
- 수정: `training/evaluate_closed_loop.py`
- 수정: `tests/test_training_data.py`, `tests/test_latent_dynamics.py`, `tests/test_site_map_training.py`

- [x] 수집 episode가 `(track_id, seed)` 또는 `SiteMapEpisode`를 받을 수 있게 공통 episode resolver를 추가한다.
- [x] `train_policy.py --site-map-split training/maps/site/site_map_split.json`은 train maps에서 rollout을 수집하고 tune map에서 checkpoint 선택 및 HUD ablation을 기록한다.
- [x] `train_dynamics.py --site-map-split ...`은 같은 train map set의 정책 transition으로 dynamics를 학습하고 holdout map에서 예측 지표를 계산한다.
- [x] `evaluate_closed_loop.py --site-map-split ...`은 tune map에서 planner를 선택하고, 분리된 heldout map에서 PPO-only/PPO+CEM을 같은 episode로 비교한다. JSONL에는 map ID, map kind, 장애물 수, seed, collision, damage, progress, 종료 사유를 기록한다.
- [x] 기존 기본 split 테스트와 표준 맵 평가 경로는 유지한다.

**통과 기준:** train/tune/heldout map ID가 겹치지 않는다. quick run의 JSONL과 summary에 사이트 map ID·장애물 수가 나타나고, PPO-only와 PPO+CEM이 동일한 map/seed 쌍으로 평가된다.

## Task 4: 학습 수준과 장애물 성능 확인

**파일**

- 수정: `README.md`
- 생성: `artifacts/haic/site-map-*` 결과물

- [ ] 256-step smoke checkpoint를 성능 후보로 사용하지 않는다. 더 충분한 rollout step으로 PPO를 학습하고 step 수·벽시계 시간·checkpoint를 기록한다.
- [ ] canonical 학습 맵, 별도 tune 맵, 완전히 분리된 heldout 맵에서 completion rate, progress, collisions, damage를 비교한다.
- [ ] PPO-only와 PPO+CEM을 같은 heldout episode에서 비교하고, 성능이 낮으면 수치를 그대로 보고한다.
- [ ] `python -m unittest discover -s tests` 전체를 실행하고 결과를 README의 사이트 맵 실험 섹션에 요약한다.

**통과 기준:** 결과는 heldout map에서 확인한 완주율과 진행률로 판정한다. 형식/로딩 검증 성공만으로 주행 성능이 좋다고 주장하지 않는다.
