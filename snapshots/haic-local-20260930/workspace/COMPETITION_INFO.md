# HAIC Competition 정보

이 문서는 과거 로컬 대회 요약이다. 규칙·제출·일정 사실은 현재 [대회 사이트](https://ships-duo-ethical-saver.trycloudflare.com/)와 [공식 Participants 저장소](https://github.com/2026-HAIC/Participants)를 먼저 확인한다. 공식 Participants README와 LICENSE의 고정 로컬 원문은 [docs/sources/official-participants/](docs/sources/official-participants/README.md)에 있다. 출처 우선순위는 [AGENTS.md](AGENTS.md), 프로젝트 목적은 [PROJECT_INFO.md](PROJECT_INFO.md), 실행·누수·제출 제한은 [RESTRICTIONS.md](RESTRICTIONS.md)에 있다. 아래 `variables-6` 수치는 로컬 스냅샷이며 외부 조치 전 재확인이 필요하다.

## 관측과 행동

- 관측은 `float32`, shape `(4, 84, 84)`, 범위 `[0, 1]`인 최근 흑백 프레임 4장이다.
- `act()`는 `[steer, gas, brake]` shape `(3,)`의 유한값을 반환한다.
- steer는 `[-1, 1]`, gas와 brake는 `[0, 1]`이다.
- 현재 제출 주체는 strict checkpoint loading을 사용하는 PPO actor-only다.

## 시간·메모리 제한

| 항목 | 제한 |
|---|---:|
| import + 생성 | 10초 |
| reset 1회 | 5초 |
| act 1회 | 5초 |
| 프로세스 메모리 | 1,024 MB |
| 연속 invalid action | 10회 |

공식 환경은 Python 3.11 Linux CPU이며 고정 의존성은 `gymnasium[box2d]==0.29.1`, `torch==2.1.0`, `numpy==1.26.0`, `opencv-python==4.8.1.78`이다.

## 순위와 종료

- 95% 이상의 고유 타일을 방문한 뒤 정상 방향으로 결승선을 통과해야 완주다.
- 손상 100%, 101회 연속 음수 보상, 영역 이탈, max-steps 소진, invalid action 10회 연속이면 미완주다.
- 완주자는 실제 시뮬레이션 랩타임이 짧을수록 높고, 미완주자는 진행도가 높을수록 높다.
- 초기 50 raw frame 뒤 50 FPS 물리 tick을 기준으로 랩타임을 잰다.

## 로컬 실행과 패키지

연구 실행과 승인은 [README의 지원 CLI](README.md#supported-local-cli), [PROJECT_INFO.md](PROJECT_INFO.md), [AGENTS.md](AGENTS.md)의 v2 계약을 따른다. 아래 직접 runner 명령은 과거 로컬 사용 예이며 새 하네스의 승인 흐름을 대신하지 않는다. 패키지 생성은 등록된 로컬 작업이고 공식 업로드·모델 확인은 실행 직전 별도 사용자 승인이 필요하다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python local_runner.py --track-id 1 --seed 42 --no-render
python -m unittest discover -s tests -v
```

제출 ZIP 최상위에는 `agent.py`가 있어야 한다. 모델과 필요한 모듈만 넣고 `.venv`, `__pycache__`, 학습 데이터와 백업 checkpoint는 제외한다. ZIP 500 MB, 파일 수 1,000개, 압축 해제 후 2 GB 제한을 지킨다.
