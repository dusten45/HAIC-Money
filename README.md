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
