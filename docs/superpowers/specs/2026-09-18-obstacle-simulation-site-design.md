# 장애물 배치 및 주행 시뮬레이션 사이트 설계

## 1. 목표

대회 공식 물리 환경을 기준으로 다음을 확인할 수 있는 로컬 웹 애플리케이션을 만든다.

- seed와 track ID로 절차적 트랙을 자동 생성한다.
- 공식 장애물 또는 사용자가 지정한 추가 장애물을 배치한다.
- 브라우저에서 차량을 직접 조작하고 주행을 일시정지·재개·한 스텝씩 진행한다.
- 진행률, 완주 여부, 랩타임, 손상도, 충돌, 종료 사유를 표시한다.
- 반복 실행 결과를 기록해 이후 에이전트 제작 단계의 평가 기준으로 사용한다.

첫 단계의 산출물은 시뮬레이션 사이트와 평가 기록 저장 기능이다. 에이전트 학습·모델 비교·자동 개선 루프는 시뮬레이터가 검증된 뒤 별도 단계로 추가한다.

## 2. 범위와 제약

### 포함

- 공식 `CarRacing` 및 `CarEnvironment`를 사용하는 정확 모드
- 공식 장애물 6개를 유지하는 모드
- 공식 장애물을 제거하고 사용자 장애물만 사용하는 모드
- 공식 장애물에 사용자 장애물을 추가하는 모드
- 트랙 진행률 기반 장애물 배치
- 키보드 수동 제어
- 자동 seed/장애물 조합 생성
- 세션별 주행 기록 및 요약 기록 저장
- 브라우저용 정적 파일과 로컬 HTTP API

### 제외

- `agent.py` 최종 구현과 학습 파이프라인
- 공식 평가용 `core/`, `env_wrapper.py`, `damage.py`의 수정
- 브라우저에서 공식 Box2D 물리를 다시 구현하는 작업
- 인증, 사용자 계정, 다중 사용자 협업
- 외부 공개 호스팅을 전제로 한 영구 데이터베이스

공식 물리 파일은 그대로 유지한다. 사용자 장애물 기능은 별도 모듈에서 공식 환경의 장애물 생성 인터페이스를 호출해 추가한다.

## 3. 사용자 경험

첫 화면은 다음 영역으로 구성한다.

1. **트랙 설정**
   - seed
   - track ID
   - 최대 시뮬레이션 스텝
   - 자동 트랙 생성 버튼

2. **장애물 편집**
   - 모드: 공식만 / 사용자만 / 공식+사용자
   - 장애물의 트랙 진행률(0~1)
   - 도로 중심 기준 좌우 오프셋
   - 반지름
   - 추가·삭제·랜덤 생성 버튼

3. **시뮬레이션 화면**
   - 실제 환경의 RGB 렌더링
   - 시작·일시정지·한 스텝·리셋 버튼
   - 방향키 또는 WASD 수동 제어
   - 현재 행동값 표시

4. **상태 및 기록**
   - progress
   - FINISHED/DNF
   - lapTimeMs
   - damage
   - collision count
   - retire reason
   - seed, track ID, obstacle configuration

작은 트랙 미니맵에는 현재 트랙과 장애물의 상대 위치를 표시한다. 실제 주행 화면은 서버가 공식 환경에서 렌더링한 프레임을 사용해 브라우저 시각화와 평가 물리의 차이를 줄인다.

## 4. 시스템 구조

```text
Browser
  ├─ static/index.html, app.js, styles.css
  └─ JSON HTTP API
       ↓
web_simulator/server.py
  ├─ session registry
  ├─ track/obstacle validation
  ├─ run history writer
  └─ SimulationSession
       ↓
web_simulator/simulation.py
  ├─ CarRacing + TimeLimit
  ├─ CarEnvironment
  ├─ custom obstacle placement
  └─ metrics extraction
       ↓
official core/ and env_wrapper.py (read-only)
```

표준 라이브러리 기반 HTTP 서버를 사용하고, 이미 공식 환경에 포함된 NumPy·OpenCV를 활용한다. 별도 웹 프레임워크나 브라우저용 대형 물리 엔진은 첫 단계에 추가하지 않는다.

## 5. 트랙 및 장애물 모델

### 트랙

- `seed`는 공식 환경의 트랙 형상 생성에 사용한다.
- `track_id`는 공식 장애물 배치의 결정성에 사용한다.
- 동일한 `(track_id, seed)`는 동일한 공식 트랙과 장애물을 재현해야 한다.
- 자동 생성기는 seed를 무작위로 선택하되, 학습·검증용 seed 목록을 분리할 수 있도록 입력 seed를 저장한다.

### 사용자 장애물

브라우저에서는 월드 좌표 대신 다음 구조로 지정한다.

```json
{
  "progress": 0.42,
  "lateral": -0.25,
  "radius": 1.2
}
```

`progress`로 track point를 선택하고 `lateral`을 도로의 접선 방향에 직교한 오프셋으로 변환한다. 반지름과 개수에는 서버 상한을 둔다. 사용자 장애물은 공식 모드와 분리하며, 공식 장애물과 겹치는 경우 경고하거나 해당 배치를 거부한다.

사용자 장애물은 트랙 생성과 워밍업이 끝난 직후 추가한다. 그러면 공식 50 raw-frame 워밍업, 프레임 스킵, 손상 판정, 접촉 리스너는 그대로 유지하면서 사용자 장애물만 추가할 수 있다.

## 6. API 계약

### `GET /api/health`

서버와 시뮬레이터 의존성 상태를 반환한다.

### `POST /api/tracks/generate`

트랙 설정과 장애물 생성 옵션을 받아 하나 이상의 재현 가능한 트랙 설정을 반환한다. 이 단계에서는 큰 렌더링 파일을 저장하지 않고 JSON 설정만 반환한다.

### `POST /api/sessions`

트랙 설정으로 시뮬레이션 세션을 생성하고 초기 프레임, 미니맵 데이터, 상태 요약을 반환한다.

### `POST /api/sessions/{id}/step`

`[steer, gas, brake]`를 받아 공식 wrapper의 한 제어 스텝을 진행한다. 반환값에는 PNG 프레임, 상태 요약, 누적 기록이 포함된다.

### `POST /api/sessions/{id}/reset`

같은 설정 또는 새 설정으로 세션을 재시작한다.

### `DELETE /api/sessions/{id}`

세션을 닫고 Box2D 리소스를 해제한다.

에이전트 실행 endpoint는 첫 단계에서 추가하지 않는다. 시뮬레이터와 기록이 검증된 뒤 동일한 `step` 인터페이스를 `Agent.act()`에 연결한다.

## 7. 기록 저장

각 종료된 실행은 다음 요약을 JSON Lines 형식으로 저장한다.

```json
{
  "run_id": "...",
  "created_at": "...",
  "track_id": 1,
  "seed": 42,
  "obstacle_mode": "official_plus_custom",
  "obstacles": [],
  "finished": false,
  "lap_time_ms": null,
  "progress": 0.137809,
  "damage": 0.0,
  "collision_count": 0,
  "retire_reason": "off_track"
}
```

대용량 프레임·체크포인트·실험 산출물은 `D:\HAIC` 아래에 저장하고, 소스 저장소에는 요약 데이터와 작은 테스트 fixture만 둔다. 저장 위치는 환경 변수로 바꿀 수 있으며, 기본값이 불가능한 환경에서는 프로젝트의 `data/`로 fallback한다.

## 8. 오류 및 안전 처리

- seed와 track ID 범위를 서버에서 검증한다.
- progress, lateral, radius, 장애물 개수에 상한을 둔다.
- 잘못된 JSON이나 존재하지 않는 세션은 HTTP 400/404로 반환한다.
- 세션별로 환경을 분리하고, 종료·타임아웃 세션은 정리한다.
- 매 스텝 입력은 공식 runner와 동일하게 유한성·shape·범위를 검증하고 클리핑한다.
- 사용자 장애물이 트랙 밖 또는 시작·결승 안전 영역에 있으면 경고하거나 거부한다.
- 공식 모드와 사용자 지정 모드를 화면에 명시해 기록을 혼동하지 않도록 한다.

## 9. 테스트 전략

### 단위 테스트

- progress/lateral을 월드 좌표로 변환하는 결과
- 동일한 seed의 트랙 재현성
- 공식·사용자 장애물 모드별 장애물 개수
- 장애물 반지름·개수·범위 검증
- 사용자 장애물 충돌 시 damage와 collision count 반영
- 세션 종료 시 기록 저장

### API 테스트

- health, track generation, session create/step/reset/delete
- 잘못된 입력에 대한 400 응답
- 세션 격리

### 수동 브라우저 검증

- 페이지 로드
- 자동 트랙 생성
- 장애물 추가 및 미니맵 표시
- 키보드 주행
- 일시정지 및 한 스텝 진행
- 종료 결과와 기록표 표시

기존 `python -m unittest discover -s tests -v`도 매 변경 후 실행한다.

## 10. 배포 경계

첫 구현은 정확한 Python/Box2D 로컬 서버를 기준으로 한다. ChatGPT Sites 배포를 시도할 수 있도록 프론트엔드와 API 경계를 분리하고, Sites 런타임에서 Python 백엔드가 지원되지 않으면 다음 중 하나로 fallback한다.

1. 로컬 정확 모드와 Sites용 브라우저 데모 모드를 함께 제공한다.
2. 별도 호스팅된 Python API URL을 설정해 Sites 프론트엔드가 연결하도록 한다.

외부 공개 배포와 인증은 별도 승인 및 호스팅 설정이 필요하므로 첫 구현의 완료 조건에 포함하지 않는다.

## 11. 완료 조건

- 로컬 서버 실행 후 브라우저 페이지가 열린다.
- seed/track ID로 재현 가능한 트랙을 생성할 수 있다.
- 사용자 장애물을 추가·삭제·랜덤 생성할 수 있다.
- 차량이 키보드 입력에 따라 공식 환경에서 움직인다.
- 사용자 장애물과 충돌하면 공식 손상 규칙에 따라 상태가 변한다.
- 진행률·종료 사유·랩타임·손상·충돌 기록이 표시되고 저장된다.
- 기존 계약 테스트와 새 시뮬레이터/API 테스트가 통과한다.
- 제출용 `agent.py` 및 공식 물리 파일은 변경되지 않는다.
