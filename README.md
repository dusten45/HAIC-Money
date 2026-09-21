# 2026 HAIC CarRacing AI Challenge

이 디렉터리는 2026 HAIC CarRacing AI Challenge 참가자를 위한 공식 로컬
학습·테스트 템플릿입니다.

참가자는 기본적으로 `agent.py`의 `Agent` 클래스를 구현합니다. 제공된 로컬 실행기로
트랙, 장애물, 프레임 전처리 및 차량 손상 규칙을 적용해 에이전트를 테스트할 수 있습니다.

현재 환경 버전은 `variables-6`입니다.

## 1. 권장 환경

- Python 3.10 또는 3.11
- 공식 평가 서버: Python 3.11, CPU 환경
- Windows, macOS 또는 Linux

먼저 참가자 저장소를 clone하고 저장소 루트로 이동합니다.

```bash
git clone https://github.com/2026-HAIC/Participants.git
cd Participants
```

Python 3.12 이상에서는 고정된 PyTorch 및 Box2D 버전이 설치되지 않을 수 있습니다.
가상 환경을 만든 뒤 아래 명령을 실행하십시오.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import gymnasium, Box2D, torch, numpy, cv2; print('설치 완료')"
```

## 2. 첫 실행

```bash
python local_runner.py --track-id 1 --seed 42
```

기본 `agent.py`는 조향이나 제동 없이 가속하는 예시입니다. 정상적으로 창이 열리고
차량이 움직이면 환경 설치가 완료된 것입니다.

주요 옵션:

| 옵션 | 설명 | 기본값 |
|---|---|---:|
| `--track-id` | 트랙 식별자(1 이상의 정수) | `1` |
| `--seed` | 트랙 생성 시드 | `42` |
| `--max-steps` | 에이전트 행동 횟수 상한 | `2000` |
| `--frame-skip` | 한 행동을 유지할 raw frame 수 | `4` |
| `--no-render` | GUI 창 없이 실행 | 사용 안 함 |

`track-id`와 `seed`가 같으면 트랙과 장애물 배치가 동일합니다. `42`는 설치 확인을
위한 예시이며 공식 평가 시드가 아닙니다. 공개 트랙 시드가 제공되면 해당 값을
사용하십시오.

## 3. Agent 구현

수정 대상은 [agent.py](agent.py)입니다.

```python
import numpy as np


class Agent:
    def __init__(self):
        # 모델 생성 및 가중치 로드
        pass

    def reset(self, observation):
        # 선택 구현: 트랙별 RNN 상태 등을 초기화
        pass

    def act(self, observation) -> np.ndarray:
        # 반드시 shape (3,)의 유한한 수를 반환
        return np.array([steer, gas, brake], dtype=np.float32)
```

### 관측값

| 항목 | 값 |
|---|---|
| shape | `(4, 84, 84)` |
| 의미 | 최근 4개의 흑백 프레임 |
| 값 범위 | `0.0`~`1.0` |
| dtype | `float32` |

관측값은 PyTorch 모델에 바로 전달할 수 있는 `float32` 배열입니다.

```python
state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
```

### 행동값

`act()`는 `[steer, gas, brake]` 순서의 길이 3 배열을 반환해야 합니다.

| 인덱스 | 의미 | 허용 범위 |
|---:|---|---:|
| `0` | 조향: 음수는 좌회전, 양수는 우회전 | `-1.0`~`1.0` |
| `1` | 가속 | `0.0`~`1.0` |
| `2` | 제동 | `0.0`~`1.0` |

NaN, 무한대, 잘못된 shape 또는 예외는 유효하지 않은 행동으로 처리됩니다. 공식
서버는 해당 행동을 무동작 `[0, 0, 0]`으로 대체하며, 유효하지 않은 행동이 10회
연속 발생하면 해당 트랙에서 리타이어시킵니다. 범위를 벗어난 유한한 행동값은 허용
범위로 잘립니다.

### 모델 가중치 로드

공식 평가는 CPU 환경에서 실행됩니다. GPU에서 저장한 PyTorch 가중치는 CPU로
명시하여 로드하는 것을 권장합니다.

```python
self.model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)
self.model.eval()
```

공식 서버는 제출 ZIP의 루트에서 참가자 코드를 실행하므로 같은 디렉터리의 모델 파일을
위와 같은 상대 경로로 불러올 수 있습니다. 모델 추론 중에는 일반적으로
`torch.no_grad()`를 사용하십시오.

## 4. 실행 제한

공식 평가의 기본 제한은 다음과 같습니다.

| 항목 | 제한 |
|---|---:|
| `Agent` import 및 생성 | 10초 |
| `reset()` 한 번 | 5초 |
| `act()` 한 번 | 5초 |
| 참가자 프로세스 메모리 | 1,024 MB |
| 유효하지 않은 행동 | 10회 연속 시 리타이어 |

`reset()`은 공식 서버에서 선택 메서드입니다. 다만 이 템플릿에는 기본 메서드가 이미
포함되어 있으므로, 사용하지 않는 경우 빈 메서드로 유지해도 됩니다.

### 추가 라이브러리

학습 환경은 자유롭게 구성할 수 있지만, 제출용 `agent.py`와 모델은 다음 고정
라이브러리 버전 및 CPU 환경과 호환되어야 합니다.

- `gymnasium[box2d]==0.29.1`
- `torch==2.1.0` (CPU)
- `numpy==1.26.0`
- `opencv-python==4.8.1.78`

위 패키지는 평가 이미지에 설치되어 있으므로 제출용 `requirements.txt`에서 생략할 수
있습니다. 추론에 필요한 추가 패키지는 ZIP 최상위의 `requirements.txt`에
`패키지==버전` 형식으로 정확히 고정하십시오.

```text
stable-baselines3==2.2.1
safetensors==0.4.3
```

추가 패키지는 공식 PyPI의 Python 3.11/Linux용 바이너리 wheel이어야 합니다. Git·URL·
로컬 경로, 별도 패키지 인덱스, 중첩 requirements 파일과 소스 빌드는 지원하지 않습니다.
고정 라이브러리와 직접·간접 의존성이 충돌하거나 설치에 실패하면 평가도 실패합니다.

설치 제한은 최대 50개·32 KiB·180초·512 MiB입니다. 평가 중 인터넷 다운로드는 허용되지 않으므로 필요한 모델과 가중치는 ZIP에 포함하십시오. 추가 패키지가 없다면 제출용 `requirements.txt`는 생략할 수 있습니다.

## 5. 트랙과 장애물

환경은 Gymnasium `CarRacing-v2`를 기반으로 하며 도로 위에 물리 장애물 6개가
배치됩니다.

- 트랙과 장애물은 `(track_id, seed)`에 따라 결정됩니다.
- 같은 입력은 항상 같은 결과를 생성합니다.
- 장애물은 시작 및 결승 부근을 제외하고 서로 일정 거리 이상 떨어져 배치됩니다.
- 장애물 하나와 계속 접촉하는 동안에는 한 번만 손상으로 집계됩니다.
- 장애물에서 완전히 떨어졌다가 다시 충돌하면 새로운 충돌로 집계됩니다.
- 다른 장애물과 충돌하면 별도의 손상으로 집계됩니다.
- 프레임 스킵 도중 발생한 충돌도 누락하지 않습니다.

새 트랙이 시작되면 장애물의 충돌 기록과 차량 손상이 모두 초기화됩니다.

## 6. 차량 손상

장애물 충돌 1회당 손상이 20% 증가합니다.

| 누적 충돌 | 접지력 | 엔진 출력 | 조향 반응 | 결과 |
|---:|---:|---:|---:|---|
| 0회 | 100% | 100% | 100% | 정상 |
| 1회 | 90% | 95% | 95% | 주행 계속 |
| 2회 | 80% | 90% | 90% | 주행 계속 |
| 3회 | 70% | 85% | 85% | 주행 계속 |
| 4회 | 60% | 80% | 80% | 주행 계속 |
| 5회 | - | - | - | 충돌 리타이어 |

잔디 마찰 계수는 표준 환경과 같은 `0.6`입니다.

## 7. 종료와 점수

주행은 다음 조건 중 하나에서 종료됩니다.

- 고유 도로 타일을 95% 이상 방문한 뒤 정상 방향으로 결승선 재통과
- 충돌 손상 100% 도달
- 음수 보상이 101 행동 스텝 연속 발생하여 오프트랙 판정
- 차량이 플레이 영역 밖으로 이탈
- `max-steps` 소진
- 유효하지 않은 행동 10회 연속 발생

오프트랙은 차량 좌표만으로 판단하지 않습니다. 한 행동의 프레임 스킵 구간에서 합산한
원본 CarRacing 보상이 음수이면 카운터가 증가하고, 0 이상이면 즉시 초기화됩니다.
따라서 정지하거나 역주행하여 새 도로 타일을 방문하지 않는 경우도 포함될 수 있습니다.

고유 도로 타일 95% 방문은 완주 자격일 뿐이며 그 자체로 완주 처리되지 않습니다.
차량이 시작 영역을 벗어난 뒤 트랙의 정상 주행 방향으로 결승선을 다시 통과하고,
결승선 앞쪽 영역을 완전히 빠져나와야 완주로 확정됩니다. 역방향 통과, 옆으로 스침,
결승선 위 정지와 시작 직후의 접촉은 인정되지 않습니다. 자격을 얻기 전에 결승선을
통과했거나 95% 이상 방문한 채 행동 스텝 제한에 도달한 경우에도 미완주입니다.

참가자는 제출 전에 반드시 현재 배포된 `variables-6` 환경에서 모델의 완주 여부와
종료 시점을 다시 검증해야 합니다.

공식 평가 및 순위 기준:

- 완주: 실제 시뮬레이션 랩타임으로 비교하며 시간이 짧을수록 높은 순위
- 미완주: 완주 기록보다 뒤에 배치하고 진행률이 높을수록 높은 순위
- 실행 실패: 순위 집계에서 제외

공식 랩타임은 초기 50 raw frame의 카메라 준비 구간이 끝난 시점부터 실제 결승선
중심을 통과한 50 FPS 물리 tick까지 측정하며, 환경 내부 시뮬레이션 시간을 밀리초
단위로 기록합니다.

`local_runner.py`도 완주 시 같은 환경 내부 시뮬레이션 랩타임을 출력하고, 미완주
시 진행률을 출력합니다. 함께 표시되는 `누적 보상`은 공식 순위 점수가 아니며 원본
CarRacing의 학습 보상입니다. raw frame마다 `-0.1`, 처음 방문한 타일마다
`+1000/N`이 적용됩니다.

## 8. 제출 ZIP

제출물은 `.zip` 형식이어야 하며 ZIP의 최상위에 `agent.py`가 있어야 합니다.

올바른 구조:

```text
submission.zip
├─ agent.py
├─ model.pth
├─ requirements.txt  # 추가 패키지가 있을 때만
└─ 필요한 사용자 모듈 및 데이터 파일
```

잘못된 구조:

```text
submission.zip
└─ Participants/
   └─ agent.py
```

제출물 제한:

| 항목 | 제한 |
|---|---:|
| ZIP 압축 파일 크기 | 최대 500 MB |
| ZIP 내부 파일 수 | 최대 1,000개 |
| 압축 해제 후 전체 크기 | 최대 2 GB |
| 개별 파일 크기 | 최대 500 MB |
| 개별 파일 압축률 | 최대 100배 |

`model.pth`를 포함한 모든 개별 파일은 500 MB를 넘지 않아야 합니다. 모델을 여러
파일로 나누더라도 ZIP과 압축 해제 후 전체 크기, 실행 시 메모리 제한을 모두
준수해야 합니다.

제출물의 모든 `.py` 파일은 정적 검사를 받습니다. 다음 import는 허용되지 않습니다.

```text
ctypes, importlib, multiprocessing, os, pathlib, resource,
shutil, signal, socket, subprocess, sys
```

다음 동적 코드 실행 함수도 허용되지 않습니다.

```text
compile, eval, exec, __import__
```

실행 파일 및 네이티브 라이브러리 형식인 `.com`, `.dll`, `.dylib`, `.exe`, `.msi`,
`.scr`, `.so`도 ZIP에 포함할 수 없습니다.

### 제출 전 체크리스트

- ZIP 최상위에 `agent.py`가 있는가?
- `Agent` 클래스와 `act()` 메서드가 있는가?
- 모든 행동이 shape `(3,)`의 유한한 숫자인가?
- CPU 환경에서 모델을 로드할 수 있는가?
- 추가 패키지를 `requirements.txt`에 정확한 버전으로 명시했는가?
- 추가 패키지가 고정 라이브러리 버전과 호환되는가?
- ZIP과 각 모델 파일이 500 MB 이하인가?
- 금지된 import와 함수가 없는가?
- `.venv`, `__pycache__`, 학습 데이터 및 체크포인트 백업본을 제외했는가?

## 9. 파일 구성

| 경로 | 역할 | 참가자 수정 여부 |
|---|---|---|
| `agent.py` | 에이전트 및 모델 추론 구현 | 수정 |
| `requirements.txt` | 로컬 환경 및 제출용 추가 패키지 버전 | 필요시 수정 |
| `local_runner.py` | 로컬 주행 실행 및 점수 출력 | 제출 불필요 |
| `env_wrapper.py` | 전처리, 프레임 스킵·스택, 로컬 규칙 적용 | 수정하지 않음 |
| `damage.py` | 로컬 충돌 손상 계산 | 수정하지 않음 |
| `core/track_variables.py` | 결정적 장애물 배치 | 수정하지 않음 |
| `core/obstacle_contacts.py` | 장애물 접촉 판정 | 수정하지 않음 |
| `core/vendor/` | 공식 CarRacing 및 Box2D 차량 물리 | 수정하지 않음 |
| `tests/` | 참가자판 계약 및 서버 공통 파일 동기화 검사 | 제출 불필요 |
| `LICENSE` | 배포 및 vendored 코드 라이선스 | 수정하지 않음 |

`core/`, `env_wrapper.py`, `damage.py`를 변경하면 로컬 결과가 공식 평가와 달라질 수
있습니다. 이 파일들은 로컬 학습·테스트용이며 제출 ZIP에는 포함할 필요가 없습니다.

## 10. 자동 테스트

의존성을 설치한 환경에서 다음 명령으로 참가자판의 주요 계약을 확인할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```

테스트는 관측 shape·dtype, 행동 검증과 클리핑, 선택적 `reset()`, 충돌 손상 계산을
확인합니다. 이 저장소와 `2026-HAIC` 서버 저장소가 같은 상위 디렉터리에 있으면 공통
물리 파일이 서버와 동일한지도 함께 확인합니다.

일반 참가자 환경에는 서버 저장소가 없으므로 서버 일치 테스트 1개가 `skipped`로
표시되는 것이 정상입니다. 이 검사는 운영진이 두 저장소를 같은 상위 디렉터리에 둔
환경에서 실행합니다. 나머지 테스트가 모두 `OK`인지 확인하십시오.

## 11. 문제 확인

환경 설치가 실패하면 다음 정보를 함께 확인하십시오.

```bash
python --version
python -m pip --version
python -m pip list
```

특히 Python 버전이 3.10 또는 3.11인지, 현재 가상 환경의 Python과 `pip`가 같은
경로를 사용하는지 먼저 확인하십시오.

## 12. 맵 생성·실행 로그 시뮬레이터

`local_simulator`는 공식 `CarRacing`/`CarEnvironment`를 그대로 사용해 재현 가능한
맵을 만들고, 로컬에서 주행한 결과를 `run.json`으로 저장합니다. 브라우저 사이트는
이 로그를 읽어 궤적을 재생할 뿐이며, 브라우저 안에서 Box2D 물리를 실행하거나
매 스텝마다 서버를 호출하지 않습니다.

큰 맵·로그·프레임 파일은 기본적으로 `D:\HAIC`에 보관합니다. 이 저장소의 정적
사이트는 별도 설치 없이 열 수 있습니다.

### Windows 권장 실행

```powershell
# 공식 환경과 호환되는 Python 3.11 가상 환경
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.map `
  --track-id 1 --seed 42 `
  --obstacle-mode official_plus_custom `
  --auto-obstacles 4 `
  --output D:\HAIC\maps\map.json

D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.run `
  --map D:\HAIC\maps\map.json `
  --output D:\HAIC\runs\run.json

D:\HAIC\haic-env\Scripts\python.exe -m http.server 8000 `
  --directory web_simulator
```

그 다음 브라우저에서 `http://127.0.0.1:8000/`을 열고 `map.json`과 `run.json`을
각각 불러옵니다. 사이트의 재생 버튼은 로그를 자동으로 끝까지 재생하며, 일시정지,
앞·뒤 한 스텝, 처음으로, 재생 속도와 타임라인 이동을 지원합니다. 여러 `run.json`을
차례로 추가하면 완주 여부, 랩타임, 진행률, 손상, 충돌 횟수를 비교합니다.

실제 로컬 시뮬레이터를 브라우저 버튼으로 실행하려면 정적 서버 대신 통합 런처를
사용합니다.

```powershell
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.web `
  --host 127.0.0.1 --port 8765 `
  --project-root C:\Users\koi\Coding\HAIC\.worktrees\obstacle-sim-site
```

통합 런처에서는 자체 맵 생성, `agent.py` 주행, 수동 키보드 주행, `run.json` 저장을
웹에서 바로 실행할 수 있습니다. 정적 `http.server`는 파일을 불러와 로그를 보는
전용 모드로 계속 사용할 수 있습니다.

### 자체 트랙 템플릿

커스텀 트랙 생성기는 코너 배치와 곡률이 서로 다른 여섯 가지 템플릿을 제공합니다.

| 템플릿 ID | 화면 이름 | 주행 특성 |
| --- | --- | --- |
| `oval` | 타원형 | 넓은 코너 네 개로 이루어진 기준 코스 |
| `s_curve` | S 커브 | 좌우 방향이 바뀌는 연속 코너 |
| `hairpin` | 헤어핀 | 급격한 코너와 제동·탈출 구간 |
| `chicane` | 시케인 | 빠른 좌우 전환이 반복되는 코스 |
| `technical` | 복합 기술형 | 서로 다른 반경과 코너 순서가 섞인 코스 |
| `extreme_technical` | 극한 복합 기술형 | 주요 코너 12·14·16개, S자 구간 최소 3개, 90도에 가까운 급코너 최소 3개 |

`--design-seed`는 코너 조합과 중심선을 결정합니다. 같은 생성기 버전, 템플릿,
seed를 사용하면 같은 트랙을 다시 만들 수 있습니다. 생성된 맵은 기존 schema 2를
그대로 사용하며, 생성기 버전 5는 코너 수·순서·실제 회전 방향·반경을
`generator` 메타데이터에 기록합니다. 기존 맵의 중심선은 버전이 바뀌어도 자동으로
재생성되지 않습니다.
새 트랙 생성기의 도로 반폭은 0.5–9입니다. 더 큰 폭은 고정된 게임 플레이 영역에서
안전한 다중 코너를 보장하지 못해 생성을 거부합니다. 기존에 저장한 폭이 큰 맵은
불러올 수 있지만, 실행 전 형상 안전성 검사를 통과해야 합니다.

```powershell
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.map `
  --kind custom --template technical --design-seed 42 `
  --output D:\HAIC\maps\technical-42.json
```

장애물은 트랙 기하와 별도인 선택 레이어이므로, 필요한 경우 웹 편집기나 기존
`--obstacle` 옵션으로 추가합니다. 자체 트랙은 에이전트의 일반화와 과적합을 점검하기
위한 로컬 테스트용이며, 공식 평가의 비공개 트랙을 대신하거나 복제하지 않습니다.

### 사용자 장애물

맵 파일은 `track_id`, `seed`, `obstacle_mode`, `max_steps`, `frame_skip`과 함께
정규화된 장애물을 저장합니다. 장애물은 `progress,lateral,radius` 형식으로 반복해
지정할 수 있습니다.

```powershell
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.map `
  --track-id 1 --seed 99 `
  --obstacle-mode official_plus_custom `
  --obstacle 0.42,-0.25,1.2 `
  --obstacle 0.70,0.30,1.0 `
  --output D:\HAIC\maps\custom-map.json
```

`--auto-obstacles N`은 같은 seed에서 같은 위치를 다시 만드는 검증·과적합 점검용
장애물 N개를 생성합니다. `official` 모드는 공식 장애물만 사용하므로 사용자
장애물을 함께 지정할 때는 `custom_only` 또는 `official_plus_custom`을 선택합니다.

프레임까지 로그에 넣어 카메라 화면을 재생하려면 실행 명령에
`--record-frames`를 추가합니다. 프레임 로그는 크기가 빠르게 커지므로 기본값은
궤적·상태만 저장하는 방식입니다.

### 이후 에이전트 연결

현재 CLI의 기본 정책은 파이프라인 확인용 `BaselinePolicy`입니다. 다음 단계에서는
`agent.py`의 `Agent.reset()`/`Agent.act()`를 같은 policy adapter에 연결해, 동일한
맵·로그·웹사이트 형식으로 여러 에이전트 정책을 비교할 수 있습니다. 공식 파일인
`core/`, `env_wrapper.py`, `damage.py`는 계속 수정하지 않아야 합니다.
