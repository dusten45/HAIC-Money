# 2026 HAIC CarRacing AI Challenge

이 디렉터리는 2026 HAIC CarRacing AI Challenge 참가자를 위한 공식 로컬
학습·테스트 템플릿입니다.

참가자는 기본적으로 `agent.py`의 `Agent` 클래스를 구현합니다. 제공된 로컬 실행기로
트랙, 장애물, 프레임 전처리 및 차량 손상 규칙을 적용해 에이전트를 테스트할 수 있습니다.

현재 환경 버전은 `variables-6`입니다.

## 1. 권장 환경

- Python 3.10 또는 3.11
- 공식 평가 서버: Python 3.11, Linux Docker, CPU 환경
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

`Agent`는 기존 baseline `model.pt`와 HAIC visual-policy `policy.pt` 실행 형식을 모두
지원합니다. `model.pt`가 있으면 baseline을 사용하고, HAIC checkpoint나 설정을 지정했거나
baseline checkpoint가 없으면 `policy.pt`/선택적 `dynamics.pt` 경로를 사용합니다. HAIC 개발 중
checkpoint가 없으면 인터페이스 smoke 용도의 초기 actor를 만들 수 있지만, 제출 ZIP은 strict
loading으로 유효한 학습 checkpoint가 없으면 실행을 거부합니다. Corridor 교사는 train split의
시연 수집과 학습 전용 벤치마크에서만 쓰며 별도 teacher 모듈로는 `Agent`에 포함하지 않습니다.
대신 bare `model.pt`를 사용하는 legacy baseline 실행에서는 제출 루트에 포함되는
forward-corridor safety controller가 도로 중심선·속도·밝은 장애물 신호를 사용해 급격한
역주행성 조향을 제한합니다. 직선 구간은 중심선 deadband 안에서 조향을 0으로 고정하고,
곡선 구간도 최대 조향량과 프레임별 조향 변화량을 제한합니다. 가속과 제동은 상호 배타적으로
선택하며, 명시적 action-contract payload와 DrQ actor는 기록된 모델 계약을 그대로 유지합니다. CEM은
별도 학습 모델과 평가 결과가 선택한 경우에만 PPO 행동 제안을 보정합니다.

`local_runner.py --track-id N --seed S`에서 고정 공식 트랙을 만들면 해당 트랙에 장애물 6개가
항상 배치됩니다. Track Lab 맵은 `official`, `custom_only`, `official_plus_custom` 모드로 공식
장애물과 사용자 장애물을 구분해 평가할 수 있습니다.

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

### DrQ-v2 시연 replay

픽셀 corridor 교사의 명시적인 학습용 episode를 DrQ transition artifact로 수집할 수
있습니다. 교사의 official `[steer, gas, brake]` 행동은 DrQ의 3차원 symmetric native
행동으로 한 번만 변환되며, 원본 reward와 terminal/truncation 의미는 유지됩니다.

```powershell
python -m training.drq_demonstrations `
  --episodes 1:1001,2:1002 `
  --max-decisions 2000 `
  --output artifacts/haic/drq-demonstrations.pt
```

학습 시 artifact와 고정 demonstration batch 크기를 함께 지정합니다. 시연 replay는
online replay와 분리되어 online 경험에 의해 덮어써지지 않으며, checkpoint에 sampling
RNG와 함께 저장됩니다. 제출용 actor에는 교사나 replay가 포함되지 않습니다.

```powershell
python train_drqv2.py `
  --name drq-with-demonstrations `
  --total-steps 131072 `
  --batch-size 64 `
  --replay-capacity 100000 `
  --warmup-steps 10000 `
  --demonstrations artifacts/haic/drq-demonstrations.pt `
  --demonstration-batch-size 2
```

## 11. 문제 확인

환경 설치가 실패하면 다음 정보를 함께 확인하십시오.

```bash
python --version
python -m pip --version
python -m pip list
```

특히 Python 버전이 3.10 또는 3.11인지, 현재 가상 환경의 Python과 `pip`가 같은
경로를 사용하는지 먼저 확인하십시오.

## 12. PPO Baseline 1.1

`train.py`의 기본값은 Baseline 1.1 안정화 설정입니다.

```bash
python train.py --name ppo-cnn-baseline1-1
```

- learning rate `1e-4`, `n_epochs=5`, `clip_range=0.2`, `target_kl=0.03`
- 65,536 step마다 PPO update 이후의 model checkpoint와 VecNormalize 통계를 함께 저장
- 학습 seed `42,1337,2024,777`로 seen 평가, holdout seed `10001`~`10008`로 best model 선택
- holdout의 완주율, 평균 진행률, 평균 랩타임 순서로 `best_model.zip`을 선택
- 마지막 rollout도 학습 종료 직후 평가하므로 최종 모델이 미평가 상태로 남지 않음
- model 선택 평가는 저장된 checkpoint를 CPU에서 새로 load해 수행하고, best model은
  다시 load해 per-episode 결과가 같은지 기록한다. 이는 GPU 학습 경로와 공식 CPU
  제출 경로 사이의 수치적 궤적 차이를 조기에 드러내기 위한 검증이다.

학습 worker는 기본적으로 매 episode마다 새 `(track_id, seed)`를 결정적으로
sample한다. 따라서 `--track-ids 1,2,3`은 obstacle layout도 다양화하고, 도로
geometry는 unsigned 32-bit seed 전체에서 새로 생성한다. sampler는 평가에 쓰는
seed를 피하며, `--track-sampler-seed`로 재현할 수 있다. 예전의 고정 worker layout은
비교 실험에서만 `--training-track-mode fixed`로 사용한다.

완주를 아직 학습하지 못한 정책에는 training-only obstacle curriculum을 사용할 수
있다. `--training-obstacles none`은 road geometry와 차량 물리는 유지하되 obstacle을
만들지 않는다. 이 phase의 checkpoint를 저장한 뒤, paired VecNormalize state와 함께
`--training-obstacles official`인 새 run으로 resume한다. `evaluate()`와 제출 검증은
항상 official six-obstacle environment를 사용한다.

충돌 때문에 실패하는 정책에는 `--collision-penalty 5.0` training-only ablation을
사용할 수 있다. frame skip 동안 하나 이상 충돌한 agent action마다 native reward에서
한 번만 차감하며, official evaluation과 submission에는 적용되지 않는다.

bang-bang 조향을 비교할 때는 `--action-smoothing steering-ema`를 사용한다.
EMA는 고수준 agent action마다 한 번 적용되고 episode reset에서 `[0, 0, 0]`으로
초기화된다. 기본 alpha `0.35`는 조향에만 적용하며 gas/brake는 alpha `1.0`으로
유지한다. 동일한 canonical config와 fingerprint가 training, CPU evaluation,
export payload, submission Agent에 기록된다. `none`은 같은 stateful 경로의 alpha
`1.0` control이다.

resume 시에는 같은 checkpoint의 VecNormalize 통계가 자동으로 탐색됩니다.

```bash
python train.py \
  --resume runs/<run>/best_model.zip \
  --name ppo-cnn-baseline1-1-resume
```

### Throughput 비교

rollout 크기를 16,384 sample로 고정한 짧은 비교 실행입니다. 두 명령 모두 3 rollout만
실행하며, 평가와 checkpoint I/O를 제외해 로그의 FPS를 비교할 수 있습니다.

```bash
python train.py --name throughput-8x2048 --n-envs 8 --n-steps 2048 \
  --total-timesteps 49152 --save-freq 0 --eval-freq 0 --skip-final-eval

python train.py --name throughput-16x1024 --n-envs 16 --n-steps 1024 \
  --total-timesteps 49152 --save-freq 0 --eval-freq 0 --skip-final-eval
```

### 제출용 Torch 모델 export

학습 모델은 SB3 zip 대신 순수 Torch state dict로 export합니다. export 과정은 무작위
관측값에서 SB3 deterministic action과 export 모델의 action이 일치하는지 검증하고,
warm-up 후 CPU action latency를 측정합니다.

```bash
python export_policy.py \
  --input runs/<run>/best_model.zip \
  --output model.pt \
  --verify-samples 256 \
  --benchmark-calls 500
```

제출 ZIP에는 export된 `model.pt`, `agent.py`, 그리고 stateful smoothing을 공유하는
`action_smoothing.py`가 포함됩니다. `agent.py`는 Stable-Baselines3를 import하지
않습니다.

HAIC PPO/CEM 형식의 inference-only ZIP은 별도 `training.package_submission` 경로로 생성하며,
`policy.pt`, `dynamics.pt`, `agent.py`, 필요한 `haic_agent` 모듈을 포함합니다.

### 독립 checkpoint 평가

`evaluate_policy.py`는 학습 run을 만들거나 `runs/_latest`를 바꾸지 않고, CPU
single-thread deterministic action으로 checkpoint를 비교합니다. `checkpoint-v1`
protocol은 `policy.pth` hash로 alias checkpoint를 제거하고, environment/source hash,
per-episode action trace, damage, termination class, runtime을 immutable artifact로
기록합니다.

전체 checkpoint screen은 다음처럼 실행합니다. 이 protocol은 track ID `1`~`4`와
unseen seed `20001`~`20004`의 고정 matrix를 사용한다.

```bash
python evaluate_policy.py \
  --protocol checkpoint-v1-screen \
  --run-dir runs/<run>
```

결과는 `evaluations/<UTC>_checkpoint-v1-screen/`에 원자적으로 생성된다. `summary.json`
은 finish, 진행률, track별 결과를 정렬해 기록하고, `episodes.jsonl`은 모든 cell의
원시 결과를 기록한다. `checkpoint-v1-confirmation`은 별도 seed 집합에서 각 cell을 두
번 실행해 action trace와 종료 결과의 결정성을 검사한다. Screen은 model promotion을
위한 결정성 audit을 수행하지 않으므로, screen의 상위 후보를 바로 선택하면 안 된다.

confirmation에는 screen 상위 후보만 `--model`로 명시한다.

```bash
python evaluate_policy.py \
  --protocol checkpoint-v1-confirmation \
  --run-dir runs/<run> \
  --model runs/<run>/checkpoints/ppo_baseline_<step>.zip
```

과거 정책을 비교만 하려면 `--legacy-model`을 명시한다. comparator는 결과에 포함되지만
promotion 대상은 아니다.

한 후보를 선택한 뒤에는 `checkpoint-v1-blind`를 한 번만 사용한다. 이 protocol은
track ID `7,8,9`, seed `20201`~`20208`을 각 두 번 실행해 final generalization과
결정성을 함께 확인한다.

작은 임시 비교에는 ad-hoc mode를 사용할 수 있지만, 이 결과는 protocol 결과와 직접
비교하거나 model selection에 사용하지 않는다.

```bash
python evaluate_policy.py \
  --model runs/<run>/best_model.zip \
  --model runs/<run>/model.zip \
  --track-ids 1,2,3 \
  --seeds 20001,20002,20003,20004,20005,20006,20007,20008 \
  --output runs/<run>/broad-evaluation.json
```

새 정책의 model selection은 finish, 진행률, 랩타임 순으로 수행한다. screen/selection에
사용한 seed와 track ID는 최종 보고용 grid에서 재사용하지 않는다.

### 제출 ZIP 검증

공식 평가와 같은 CPU 경로로 agent import, 모델 로드, 단일 action을 확인하고 ZIP 구조와
정적 금지 import를 검사합니다.

```bash
python package_submission.py \
  --label baseline1-final \
  --source-model runs/<run>/best_model.zip \
  --smoke-test
```

각 실행은 `submissions/<UTC timestamp>_<label>/submission.zip`과 `manifest.json`을
생성합니다. manifest에는 생성 시각, source model, code commit, agent/model/ZIP SHA-256,
그리고 smoke 결과가 기록됩니다. 기록된 submission archive는 수정하거나 덮어쓰지 않습니다.

현재 작업 공간에는 Python 3.11 Docker가 없으므로, 업로드 전 Python 3.11 Linux 환경에서는
같은 명령에 `--python python3.11`을 지정해 다시 확인해야 합니다.

## 13. 시각 PPO 및 학습된 CEM 계획기 실험

아래 경로는 이 저장소의 `variables-6` 환경에서만 학습용 상태 레이블을 읽습니다. 제출
에이전트는 84×84 흑백 프레임 네 장과 ZIP 안의 가중치만 사용합니다.

### 개발 가상환경

공식 `requirements.txt`는 평가 Linux 환경을 위한 고정 파일이므로 수정하지 않습니다. 현재
Windows 개발 환경에서 `box2d-py` 빌드가 SWIG 단계에서 실패하면, 공식 설치 뒤 개발용 wheel만
추가로 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install Box2D==2.3.10
```

### 학습과 예측 모델 재학습

먼저 실제 PPO rollout/update를 실행하고, 같은 정책 checkpoint에서 latent dynamics를 다시
학습합니다. 아래 256 decision update는 실험 기준선일 뿐 대회 성능을 뜻하지 않습니다.

```powershell
python -m training.train_policy --output artifacts/haic/ppo --total-steps 256 --max-decisions 64
python -m training.train_dynamics --output artifacts/haic/dynamics --steps 128 --max-decisions 64 `
  --policy-checkpoint artifacts/haic/ppo/policy.pt
```

### 보류 시드 평가

평가기는 train/tune/held-out tuple이 겹치면 실패합니다. tune split에서 horizon, population,
uncertainty cost 후보를 먼저 비교하고, 그 뒤 같은 10개 held-out tuple에서 PPO-only와 PPO+CEM을
비교합니다. `episodes.jsonl`에는 완료·DNF·오류를 모두 남기고 `summary.json`에는 latency와
집계를 남깁니다.

```powershell
python -m training.evaluate_closed_loop `
  --output artifacts/haic/evaluation `
  --policy-checkpoint artifacts/haic/ppo/policy.pt `
  --dynamics-checkpoint artifacts/haic/dynamics/dynamics.pt
```

기본값은 행동 호출당 4.5초, episode cap 2,000 decision입니다. 빠른 반복은 작은 budget을
명시해 별도 보고서에 저장합니다. 2026-09-20 검증에서 0.1초 budget, 2,000 decision cap으로
10개 held-out tuple을 비교했습니다. 두 모드 모두 10개 전부 자연 off-track DNF였으며 평균
진행률은 PPO-only `0.077411`, PPO+CEM `0.074530`이었습니다. 4.5초 budget의 PPO-only 전체
에피소드도 별도로 측정했고 268 decision 뒤 off-track으로 종료했습니다. 빠른 2,000-cap 결과는
CEM의 대회 성능을 보장하지 않으므로 제출 기본값은 PPO-only입니다.

### 제출 ZIP

아래 명령은 ZIP 루트에 `agent.py`, 필요한 `haic_agent` 추론 모듈, `policy.pt`, `dynamics.pt`만
넣습니다. 깨끗한 임시 디렉터리에서 CPU import, 두 번의 reset/act, 유한 행동, 5초 호출 제한과
프로세스 RSS 1 GiB를 확인합니다. `--disable-planner`는 tune 결과가 충분히 입증되지 않은 현재
기준선의 보수적 PPO-only 제출 설정입니다.

```powershell
python -m training.package_submission `
  --policy-checkpoint artifacts/haic/ppo/policy.pt `
  --dynamics-checkpoint artifacts/haic/dynamics/dynamics.pt `
  --evaluation-summary artifacts/haic/task5-eval-fast-fullcap/summary.json `
  --output artifacts/haic/submission/submission.zip `
  --smoke-test
```

현재 256-decision PPO와 128-transition dynamics는 학습·추론 경로의 pilot입니다. 완주율과
대회 성능은 아직 입증되지 않았습니다. 전체 이미지 CNN과 HUD ROI branch를 같은 96개 고유
decision frame에서 비교한 오차 기록은 `artifacts/haic/task5-hud-validation-20.json`에 있습니다.
HUD branch의 보조 오차 개선은 미미했고 steering/yaw는 약간 악화되어, branch를 끌 수 있는 구조를
유지했습니다. 이 HUD 결과는 한 track/seed만 사용했으므로 일반화 근거로 보지 않습니다.

`training/vision_teacher.py`는 train split에서 PPO 시연을 수집하는 교사입니다. 교사는 아스팔트
중심선, HUD 속도 막대, 밝은 장애물 픽셀을 사용하며, 교사 페달은 PPO actor의 보정된 액션 범위로
비율 변환해 actor 상한의 75%로 제한하고, 같은 액션을 시연 라벨과 수집 환경에 적용합니다. 이 여유는
`tanh` action 평균이 포화되는 것을 줄여 PPO가 더 낮은 throttle이나 brake도 탐색하게 합니다.
behavior-cloning 워밍업 후 PPO 학습이
시작되면 교사 손실은 끕니다. corridor 제어기는 제출 코드에서 불러오지 않으며, corridor 단독 주행
기록은 학습 정책의 SOTA나 제출 성능으로 계산하지 않습니다.

비교 episode별 기록과 요약은 `artifacts/haic/task5-eval-fast-fullcap/episodes.jsonl` 및
`summary.json`에 있습니다. 보류 시드 비교에서 CEM이 개선되지 않아 생성된 제출 ZIP은 계획기를
끄며, 나중에 더 오래 학습한 checkpoint에서 개선이 확인되면 요약 JSON을 전달해 같은 패키저로
재생성할 수 있습니다.

## 14. Track Lab 사이트 맵으로 학습·평가

`training/maps/site/`에는 Track Lab 웹 UI에서 저장한 technical 맵 4개와 split manifest가
있습니다. 네 맵 모두 중심선·도로 폭 8·사용자 장애물 5개를 그대로 보존합니다. canonical
`custom-track-haic-obstacles-20260920`과 별도 train 맵은 PPO transition 수집에 쓰고, 다른
맵 하나는 tune, 네 번째 맵은 최종 held-out 비교에만 씁니다. held-out의 다섯 항목은 **한 개
맵 설계에서 환경 seed만 바꾼 반복**이므로, 다섯 개의 독립적인 도로 모양으로 일반화했다고
해석하면 안 됩니다.

커스텀 중심선으로 CarRacing 도로 타일과 finish line을 구성합니다. 장애물 progress/lateral/
radius는 사이트의 정의대로 Box2D 원형 장애물 위치로 옮기고, 수집 환경 warmup 뒤 붙입니다.
물리 접촉은 기존 충돌 손상 시스템으로 전달됩니다. 폐쇄루프 JSONL에는 map ID, map 종류,
장애물 수, seed, 충돌, 손상, 진행률, 종료 사유와 act latency가 남습니다.

현재 페달 수정 후보는 정책 안에서 `steer`와 signed longitudinal 두 값을 학습하고, 실행할 때
기존 `[steer, gas, brake]` 형식으로 변환합니다. 양수 longitudinal은 최대 가속 `0.02`, 음수는
최대 제동 `0.03`으로 바뀌므로 한 행동에서 가속과 제동이 겹치지 않습니다. 사이트 튜닝 지도에서
입력축별 고정 행동을 비교해 외부 steer `-0.12`가 첫 굽이에 맞는 방향임을 확인했고, 초기 행동은
조향 `-0.12`, 가속 `0.01` 근처로 둡니다. 같은 출발 상태에서 제동 `0.01`은 속도를 `12.14`에서
`7.26`으로 낮췄고, `0.03`은 20결정 내 정지시켜 제동 상한을 낮췄습니다. 측정 원본은
`artifacts/haic/action-axis-calibration-v1.json`에 있습니다. 이는 초기 정책값일 뿐이며,
이후 steering과 pedal 선택은 화면을 입력받는 PPO가 학습합니다. 이 제한은
성능이 확인된 제출 설정이 아니므로 시드별 tune 및 held-out 결과와 함께 판단해야 합니다. 초기
탐색 표준편차는 steer `0.35`, signed pedal `0.4`로 두어 가속뿐 아니라 제동 행동도 PPO rollout에
나오게 합니다. 각 update의 실제 steer/gas/brake 비율을 학습 결과에 기록합니다. 추가로 훈련 보상에는
시뮬레이터에서만 계산하는 차선 중심·방향 오차를 넣어, PPO가 같은 조향을 고정 출력하지 않고
화면 상태에 맞춰 조정하도록 돕습니다. 이 기하 정답은 제출 추론에 전달하지 않습니다.

후속 trace에서 v6/v7의 deterministic actor가 steer 약 `-0.13`, gas 약 `0.0115`를 거의 고정
출력했고, route reward만으로는 화면 상태에 따른 조향 변화가 생기지 않았습니다. 그래서 policy
입력과 같은 시점에 수집한 lateral offset과 heading error의 sine/cosine도 시각 보조 목표로
추가했습니다. 이 라벨은 현재 관측 프레임과 같은 상태에서 읽으며, 이전처럼 행동 한 번 뒤의
상태를 현재 화면의 정답으로 쓰지 않습니다. PPO는 계속 최종 행동을 학습하고 이 보조 헤드는
시각 표현을 학습시키는 용도입니다. 보조 목표를 늘린 모델은 가중치 차원이 달라지므로 이전 ZIP의
체크포인트와 섞지 말고 새로 학습한 후 tune/held-out 결과를 확인합니다.

현재 learner는 train 맵에서만 교사 픽셀·행동 쌍을 수집하고, 기본 3 epoch 동안 actor 평균에
MSE를 적용한 뒤 같은 actor를 PPO로 fine-tune합니다. 시연은 episode당 최대 800 decision에서
자르며 픽셀은 uint8로 저장했다가 batch 단위로 `[0, 1]` 정규화합니다. tune/held-out은 교사 수집과
워밍업에 전달되지 않습니다. 워밍업 손실, 시연 수, 메모리 크기, 수집·학습 시간은 checkpoint
metadata와 학습 JSON에 남습니다. `--teacher-warmup-epochs 0`은 BC 없이 비교하는 옵션입니다.

```powershell
python -m training.train_policy `
  --site-map-split training/maps/site/site_map_split.json `
  --output artifacts/haic/site-map-ppo `
  --total-steps 8192 --updates 4 --max-decisions 2000 `
  --evaluation-max-decisions 800 `
  --teacher-warmup-epochs 3 --teacher-max-decisions 800

python -m training.train_dynamics `
  --site-map-split training/maps/site/site_map_split.json `
  --output artifacts/haic/site-map-dynamics `
  --policy-checkpoint artifacts/haic/site-map-ppo/policy.pt `
  --steps 2000 --max-decisions 500

python -m training.evaluate_closed_loop `
  --site-map-split training/maps/site/site_map_split.json `
  --output artifacts/haic/site-map-evaluation `
  --policy-checkpoint artifacts/haic/site-map-ppo/policy.pt `
  --dynamics-checkpoint artifacts/haic/site-map-dynamics/dynamics.pt `
  --plan-budget 4.5 --max-decisions 2000
```

8,192 transition은 네 번의 on-policy PPO update로 나눠 사용합니다. 각 update 뒤 새 정책으로
다음 rollout을 수집하고, tune 성능이 가장 높은 체크포인트를 보관합니다. 수집기 decision cap에서
episode를 끊을 때는 GAE에 truncation 경계를 전달해 다음 reset episode의 보상이 이전 주행에
섞이지 않게 합니다. critic이 큰 점수 합계에 끌려가지 않도록 학습 보상은 `0.1`배로 두고,
속도·바퀴 회전·조향·yaw 보조 정답도 물리 단위 범위로 정규화합니다. 기존 256-step ZIP보다
transition을 32배 늘린 비교 후보입니다. update 사이의 빠른 tune 선택은 800 decision cap으로
실행하고, 최종 tune은 전체 2,000 decision cap으로 다시 평가합니다. 실행 결과에는 rollout,
PPO update, tune 평가, HUD ablation의 경과 시간이 각각 기록됩니다. ZIP 제출
전에는 tune 성능으로 정책을 고르고, held-out에서 PPO-only와 PPO+CEM을 같은 map/seed에 대해
비교해야 합니다. smoke 실행은 연결 상태만 확인하며 성능 근거로 쓰지 않습니다.

## 15. 학습 교사의 속도·장애물 벤치마크

주행기는 도로가 곧으면 가속하고, 화면에서 읽은 속도가 도로 굴곡에 맞춘 목표를 넘으면 제동합니다.
장애물이 보이면 회피 방향을 유지하면서 목표 속도를 낮춥니다. `act()`는 프레임만 사용하고, 별도
벤치마크가 평가용 시뮬레이터 계측값으로 속도·가속도·충돌을 기록합니다.

```powershell
python -m training.benchmark_corridor `
  --site-map-split training/maps/site/site_map_split.json `
  --site-group held_out --site-limit 1 `
  --max-decisions 800 `
  --output artifacts/haic/corridor-controller-candidate-v3/benchmark.json
```

기본 벤치마크는 공식 track 1/seed 42, track 2/seed 101을 실행합니다. 위 명령은 held-out 사이트
맵 1개도 더합니다. 속도와 가속도는 CarRacing 물리 단위의 decision 간 측정값이며, 가속도 피크는
관측 간 속도 차이를 0.08초로 나눈 값입니다.

| 조건 | 장애물 | 완주 시간 | 평균 속도 | 최고 속도 | 피크 가속 | 충돌·손상 |
|---|---:|---:|---:|---:|---:|---:|
| 공식 track 1 / seed 42 | 6 | 23.98초 | 41.28 | 56.04 | 58.72 | 0회 / 0% |
| 공식 track 2 / seed 101 | 6 | 25.28초 | 41.13 | 57.30 | 58.72 | 0회 / 0% |
| held-out 사용자 맵 / seed 20260923 | 5 | 20.14초 | 39.99 | 53.11 | 58.66 | 0회 / 0% |

세 조건 모두 완주했고 행동 호출 p95는 Windows 로컬에서 약 16ms였습니다. 같은 track 1/seed 42의
이전 `.005` 고정 가속 후보 기록 47.34초와 비교하면 이번 결과는 23.98초입니다. 사용자가 알려준
선두 기록(track 1 17초, track 2 20초)에는 아직 6.98초와 5.28초 뒤처지므로, 이 후보를 1등 수준이라고
판단하지 않습니다. 사용자 맵 항목은 하나의 고정 도로 설계에서 실행한 시드 하나입니다.

이 수치는 픽셀 교사만 단독 실행한 벤치마크 기록이며 PPO actor의 주행 결과나 제출 후보가 아닙니다.
교사 모듈을 제외한 제출 ZIP은 학습된 PPO actor checkpoint를 strict-load하며, 이전 corridor 모드
패키징 옵션은 제거했습니다. ZIP smoke test는 제출 Agent의 import/reset/act와 메모리만 검증합니다.

## 16. 사용자 트랙과 로컬 시뮬레이터

`local_simulator`는 공식 로컬 물리 환경을 이용해 재현 가능한 맵을 만들고, baseline 또는
`agent.py`로 실행한 주행을 `run.json`으로 저장합니다. `web_simulator`는 맵과 로그를 브라우저에서
미리 보고 재생하며, 브라우저에서 물리 시뮬레이션을 직접 실행하지 않습니다. 큰 맵·로그는 기본적으로
`D:/HAIC`에 저장하고, 해당 드라이브가 없으면 저장소 아래 `.haic-artifacts/`를 사용합니다.

### 맵 생성과 실행

고유한 기술 트랙은 같은 generator version, 템플릿, seed, 폭으로 다시 만들 수 있습니다.
`extreme_technical`은 12·14·16개 주요 코너 중 하나를 선택하고, 코너를 공유하지 않는 S자 구간을
최소 3개, 75–105도 범위의 급코너를 최소 3개 포함하도록 검증합니다.

```powershell
python -m local_simulator.map --kind custom --template extreme_technical `
  --design-seed 73 --width 8 `
  --output .haic-artifacts/maps/extreme-73.json

python -m local_simulator.run --map .haic-artifacts/maps/extreme-73.json `
  --policy agent --project-root . `
  --output .haic-artifacts/runs/extreme-73.json
```

일반 템플릿은 `oval`, `s_curve`, `hairpin`, `chicane`, `technical`을 사용할 수 있습니다.
커스텀 장애물은 공식 장애물과 별도로 추가하며, `official` 모드에서는 사용자 장애물을 함께 지정할 수
없습니다.

```powershell
python -m local_simulator.map --track-id 1 --seed 99 `
  --obstacle-mode official_plus_custom `
  --obstacle 0.42,-0.25,1.2 --auto-obstacles 4 `
  --output .haic-artifacts/maps/obstacles-99.json
```

`local_simulator.run`은 기본 `baseline` 정책 외에 `--policy agent`를 지원합니다. 긴 재생 화면을
JSON에 포함하려면 `--record-frames`를 추가하세요. 프레임 로그는 용량이 커질 수 있습니다.

### Agent 진단 리포트와 MP4

현재 `agent.py`를 로컬 맵에서 실행하면서 카메라 프레임, 궤적, 행동, 진행률,
손상도, 충돌을 함께 기록하려면 다음 명령을 사용합니다. `--project-root`에는
`agent.py`와 해당 `model.pt`가 있는 폴더를 지정합니다.

```powershell
python -m local_simulator.diagnostic `
  --map .haic-artifacts\maps\track-1-seed-42.json `
  --project-root .haic-artifacts\drq-agent `
  --agent-path .haic-artifacts\drq-agent\agent.py `
  --output-dir .haic-artifacts\diagnostics\track-1-seed-42 `
  --video
```

이 명령은 `run.json`, 조작·궤적·카메라를 동기화한 `report.html`, 그리고 선택적인
`replay.mp4`를 생성합니다. HTML에서는 스텝을 이동하거나 재생하면서 현재 카메라,
조향·가속·제동, 진행률·손상도, 충돌 위치를 함께 확인할 수 있습니다. `--run-log`를
사용하면 이미 저장된 `run.json`에서 HTML/MP4만 다시 만들 수 있습니다.

```powershell
python -m local_simulator.diagnostic `
  --run-log .haic-artifacts\runs\agent-run.json `
  --output-dir .haic-artifacts\diagnostics\agent-run `
  --video
```

리포트의 경고 신호는 성능 점수가 아니라 관찰을 돕기 위한 규칙 기반 표시입니다.
조향 포화, 가속·제동 동시 출력, 충돌 스텝을 찾아 해당 구간을 먼저 확인하십시오.

### 브라우저에서 보기

저장된 맵·로그 파일만 열어 재생하려면 정적 서버를 사용할 수 있습니다.

```powershell
python -m http.server 8000 --directory web_simulator
```

통합 Track Lab 서버는 브라우저 UI와 로컬 API를 함께 제공해 웹에서 맵 생성, 시뮬레이션, 수동 주행과
로그 저장을 할 수 있습니다.

여러 Agent를 웹에서 선택하려면 각 Agent를 별도 폴더에 넣습니다. 폴더에는 최소한
`agent.py`와 추론용 `model.pt` 또는 `policy.pt`가 있어야 합니다.

```powershell
$agentRoot = ".haic-artifacts\agents\drq-demo"
New-Item -ItemType Directory -Force -Path $agentRoot | Out-Null
Copy-Item .\agent.py, .\action_smoothing.py, .\action_representation.py -Destination $agentRoot
Copy-Item ".\runs\<실험 폴더>\actor.pt" (Join-Path $agentRoot "model.pt")
```

```powershell
python -m local_simulator.web --host 127.0.0.1 --port 8765 `
  --project-root . `
  --artifact-root .haic-artifacts `
  --agents-root .haic-artifacts\agents
```

브라우저의 `실행할 Agent` 목록에서 Agent를 고르고, 공식 track/seed 또는 자체 생성
트랙을 설정한 뒤 `카메라 프레임 기록`을 켜고 `자동 주행`을 누르면 됩니다. 실행이
끝나면 같은 화면에서 카메라, 궤적, 행동, 속도, 진행률, damage, 충돌을 재생할 수
있습니다.

이 로컬 API에는 인증이 없으므로 `--host`는 `127.0.0.1`, `localhost`, `::1` 같은
loopback 주소만 허용합니다. 변경 요청은 JSON 콘텐츠 타입과 같은 출처도 확인합니다.
다른 기기에서 사용하려면 이 서버를 직접 노출하지 말고 인증을 갖춘 원격 서비스를 별도로 사용하세요.

자체 트랙은 에이전트의 일반화와 과적합을 점검하는 로컬 도구이며, 공식 평가 트랙을 복제하거나
대체하지 않습니다. 맵 생성 및 트랙 안전 검증이 실패하면 다른 레이아웃으로 조용히 대체하지 않고
오류를 반환합니다.
