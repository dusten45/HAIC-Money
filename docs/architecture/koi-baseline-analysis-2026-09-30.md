# KOI Baseline Analysis (2026-09-30)

## 1. Baseline 선정과 분석 범위

사용자는 최근 `koi312500` 커밋에 포함된 팀 최고 모델을 앞으로의 개선
baseline으로 지정했다. 이번 작업은 **분석과 문서화만** 수행한다.
모델 코드 수정, 파라미터 튜닝, 학습, 새로운 주행 평가, 공식 제출 또는
모델 확정은 수행하지 않았다.

위 범위와 1~11절은 07:58 UTC에 종료한 baseline 분석 기록이다.
09:56 UTC의 별도 사용자 지시로 만든 동적 회피 속도 후보는 12절에 구분한다.

**앞으로의 baseline은 아카이브의 최신 검증 후보인
`ContactContinuityAgent('crossing_projection')`로 기록한다.**
이는 `FarHazardAgent('arrival_speed')`를 상속하는 결정론적 화면 기반
규칙 제어기이며, 신경망 체크포인트가 아니다. 다운로드 가능한 arrival
릴리스만 보고 그것을 최신 최고 모델로 해석하면 한 세대 이전을 선택하게 된다.

선정의 근거는 사용자 지시와 고정 아카이브의
`docs/experiments/current-best-validated.json:2-5,10-40`이다. 해당 기록은
crossing을 최신 후보로, arrival을 `previous_current`로 명시한다.
이것은 **내부 개선 기준의 지정**이지, 공식 대회 최고의 단일 모델이나
미관측 트랙 일반화가 입증되었다는 선언이 아니다.

| 역할 | 정확한 모델/모드 | 고정 ZIP SHA-256 | 현재 분석 가능한 범위 |
|---|---|---|---|
| 개선 baseline | `ContactContinuityAgent('crossing_projection')` | `4447c8dfad1392ddbbb83f0b3d23b8d5237af2b9be785cdfd08edaf6c4623f64` | 런타임 소스, 패키지 생성 코드, 계획/요약. 원본 ZIP과 상세 주행 report는 별도 자산이라 현재 없음 |
| 이전 후보 | `FarHazardAgent('arrival_speed')` | `b168a17dac5fe5d6b28a265fb4adeeba6b3346391b35acd4a5818e7dc0b9693b` | 원본 ZIP, 정확히 같은 9개 소스, 공개된 평가/게이트 보고서 |
| 역사적 안정화 대조군 | `submission-fast-completion.zip` | `7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835` | 비교 기록. 고정 ZIP 자체는 별도 자산 |

crossing 클래스의 기본 생성자 인자는 `control`이다. 따라서 클래스를
인자 없이 만드는 것은 위 baseline과 다르다. 아카이브 루트 `agent.py`의
학습된 visual actor/CEM planner도 이 baseline이 아니다.

## 2. Git 상태와 고정 출처

| 커밋 | 역할 |
|---|---|
| `1e3dd6d25a2dbd77b964be902009e9926a6d5de2` | arrival 릴리스: 소스 9개, ZIP, source manifest, 평가 증거 추가 |
| `7dfb327d38fbe42eb48d69ffdcfc436d0e732bc0` | 위 릴리스의 PR #1 병합 |
| `c4e224d465eea12acc49d2700ba1e1e5b6146976` | 연구 작업 전체 아카이브, 최신 crossing 소스/선정 기록 포함 |
| `283cc59c426203908207c1054c4814136ae3d09f` | 위 아카이브의 PR #2 병합 |

`git pull --ff-only --no-autostash origin main`은 원격 fetch까지 성공했지만
기존 로컬 변경을 덮어쓸 수 있어 병합을 거부했다. 충돌 경로는 `README.md`,
`docs/context/current-state.md`, `docs/experiments/INDEX.md`,
`docs/plans/rlpd-completion-first-research-2026-09-26.md`다.
사용자 승인에 따라 `c4e224d`를 `/tmp/kilo/koi-baseline-c4e224d`로
`git archive` 추출하여 분석했다. 기존 staged/unstaged/untracked 작업은
보존했으며, 현재 작업 트리의 pull 병합은 완료되지 않았다.

이 문서의 소스 참조는 모두 **`c4e224d`의 내용**을 뜻한다.
나중에 루트 구현이나 다른 연구 lane이 변경되어도 이 분석의 출처는 바뀌지 않는다.

- `R/`: `releases/arrival-speed-20260930/`.
- `A/`: `snapshots/haic-local-20260930/workspace/`.
- `S/`: `snapshots/haic-local-20260930/`의 아카이브 메타데이터.
- `R/source/haic_agent/`의 주요 런타임 모듈 7개는 `A/haic_agent/` 복사본과 byte-identical이다. `__init__.py`와 루트 `agent.py`는 다르므로 서로 대체하면 안 된다.

고정 출처:

- [최신 모델 선정 기록](https://github.com/dusten45/HAIC-Money/blob/c4e224d465eea12acc49d2700ba1e1e5b6146976/snapshots/haic-local-20260930/workspace/docs/experiments/current-best-validated.json)
- [Crossing 런타임](https://github.com/dusten45/HAIC-Money/blob/c4e224d465eea12acc49d2700ba1e1e5b6146976/snapshots/haic-local-20260930/workspace/haic_agent/contact_continuity_runtime.py)
- [Crossing 패키지 생성/평가 코드](https://github.com/dusten45/HAIC-Money/blob/c4e224d465eea12acc49d2700ba1e1e5b6146976/snapshots/haic-local-20260930/workspace/training/confirm_contact_continuity.py)
- [Arrival 원본 릴리스](https://github.com/dusten45/HAIC-Money/tree/c4e224d465eea12acc49d2700ba1e1e5b6146976/releases/arrival-speed-20260930)
- [Arrival 상세 보고서](https://github.com/dusten45/HAIC-Money/blob/c4e224d465eea12acc49d2700ba1e1e5b6146976/releases/arrival-speed-20260930/evidence/report.json)

## 3. 배포 코드와 연구 아카이브의 경계

arrival ZIP은 13,536 bytes이며, 압축 해제 소스는 정확히 9개,
41,617 bytes, 928 lines다. manifest의 모든 파일 크기와 SHA-256이 일치하고,
ZIP 내부와 `R/source/` 파일이 모두 byte-identical이며 추가 파일이 없다.
`1e3dd6d`와 `c4e224d` 사이에 `R/` 변경도 없다.

crossing 패키지 생성기는 arrival ZIP의 9개 파일을 복사하고 `agent.py`를
교체한 뒤 `contact_continuity_runtime.py`를 추가한다.
따라서 의도한 최종 구성은 **10개 파일**이다
(`A/training/confirm_contact_continuity.py:16-26,42-53`).
이는 생성 코드로 확인한 구성이지, 현재 없는 crossing ZIP을 열어서 확인한 결과는 아니다.

crossing 모듈 자체의 SHA-256은
`bd0518542d0c9165c2d5c87be81d4c77bb82183c4d0334269389154143d34e80`이다.
이 값은 **모듈 해시**이며 ZIP 해시 `4447c8...`와 혼동하면 안 된다.

`S/manifest.json`의 인벤토리 재계산 결과, 전체 보관 대상은 26,511 files,
원래 크기 3,706,092,653 bytes다. 그중 Git workspace는 1,485 files,
14,538,494 bytes이고, 나머지 25,026 files는 외부 ZIP 12개에 들어 있다.
외부 ZIP의 합계 압축 크기는 1,319,234,434 bytes다. 제외 항목은 431개다.
따라서 clone/git archive로 전체 연구 evidence를 복원한 것이 아니다.
아카이브 자체도 non-atomic project capture이며 Git metadata, 설치 환경,
cache를 포함하는 disk image가 아니다 (`S/README.md:12,22-32`).

현재 없는 핵심 자산의 식별자는 다음과 같다.

| 자산 | 원래 크기 | SHA-256 | 보관 위치 |
|---|---:|---|---|
| crossing package primary `report.json` | 39,204 bytes | `73478a42b31fde8ef6274c6946b4812807bea2b7fdaad8acdb13d52d299b094f` | `haic-artifacts-01.zip` |
| crossing 원본 ZIP | 15,057 bytes | `4447c8dfad1392ddbbb83f0b3d23b8d5237af2b9be785cdfd08edaf6c4623f64` | `haic-artifacts-01.zip` |

위 크기/해시는 `S/manifest.json:13782-13793`의 보관 기록이며 payload 검증은
아직 아니다. 저장소와 `/tmp/kilo`의 관련 경로에 원본 ZIP/chunk가 없음을
확인했고, 이번 분석에서 외부 1.3 GB 자산을 다운로드하거나 복원하지 않았다.

배포 런타임은 NumPy와 OpenCV(`cv2`)에 의존한다. Torch, 학습 optimizer,
policy/dynamics 체크포인트, 환경 import, 트랙 파일, pose/geometry API,
Oracle 호출은 배포 closure에 없다. 허용된 화면, HUD, 자신의 과거 상태만 사용한다.

`A/agent.py`, `A/train.py`, `A/training/train_policy.py`, PPO/BC/CEM 모듈과
다른 수십 개 runtime은 역사적 연구 자산이다. 존재한다는 이유만으로
crossing에 연결되었다고 해석하면 안 된다. 전체 아카이브를 루트 submission으로
압축하거나 연구용 `agent.py`를 실행하면 이 모델을 평가하는 것이 아니다.

## 4. 실제 제어 계층과 우선순위

생성자 선택과 `act()`의 내부 실행 순서는 다음과 같다.

```text
Agent wrapper (crossing 패키지 생성 코드의 진입점)
  ContactContinuityAgent('crossing_projection')
    FarHazardAgent('arrival_speed')
      AccelerationEnvelopeAgent('swept_sprint')
        FastCompletionCoordination('preview_row_repair')
          FixedHighSpeedRecoveryV2('impact_clear')
            FixedHighSpeedAgent('damping')
              composed VisionCorridorAgent
```

상속 체인을 따라 가장 안쪽 action을 먼저 계산하고 바깥 계층이 덮어쓴다.
따라서 최종 action을 base steering/gas/brake 규칙만으로 설명할 수 없다.

| 계층 | 최종 선택에서 실제 역할 | 소스 |
|---|---|---|
| Vision corridor | 도로/장애물 인식, 피할 측면 선택과 latch, 원래 road/avoid steering | `R/source/haic_agent/corridor_agent.py` |
| Fixed high speed | 모든 decisions에 60 기준 fixed pedals, 처음 10개 이후 과거 middle road 변화로 damping | `fixed_high_speed_runtime.py:13-15,33-70` |
| Recovery v2 | 급격한 HUD 속도 하락을 impact proxy로 판단하고 잠시 obstacle steering을 제거 | `fixed_high_speed_recovery_v2.py:26-61` |
| Fast coordination | 의도한 braking을 충돌로 착각하는 proxy veto, row 42 복원, 도로 굽음/near obstacle에 따른 38~60 preview budget | `fast_completion_coordination.py:33-98` |
| Acceleration envelope | 예측 경로의 asphalt footprint가 충분하면 gas 1/brake 0으로 sprint override | `acceleration_envelope_runtime.py:7-23,35-74` |
| Far hazard | far/near obstacle까지 영상 거리로 arrival cap 계산, 초과할 때 gas 0과 최대 brake 0.6 | `far_hazard_runtime.py:49-113` |
| Contact continuity | 예상 충돌선에 들어오는 near object에만 avoidance steering을 강화, pedals는 부모 결과 그대로 | `A/haic_agent/contact_continuity_runtime.py:24-86` |

유효하고 유한한 입력에서 최종 steering은 `[-0.7,0.7]`, gas는 `[0,1]`,
brake는 `[0,0.6]` 범위다. 처음 10 decisions에는 바깥 안전 intervention들이
적용되지 않는다. base가 speed/obstacle을 계산해도 launch pedals에 덮어쓰인다.
정확히는 처음부터 `fixed_pedals(speed)`의 target 60 제어가 적용되며,
저속 출발에서 gas 0.6/brake 0이 된다. 처음 10개가 입력과 무관한 고정 gas
0.6이라는 뜻은 아니다. base obstacle steering은 prefix에도 남아 있다.
fixed pedals의 식은 gas `clip(0.12+0.04*(60-speed),0,0.6)`,
brake `clip(0.02*(speed-62),0,0.15)`다. 기본 corridor의 curve/obstacle
target와 exclusive pedals는 이 단계에서 버려진다.

## 5. 화면 인식

### 5.1 관측과 HUD 속도

`current_frame()`은 `(4,84,84)` 또는 `(84,84)`를 받으며 float32로
변환하고 최신 프레임만 finite, `[0,1]` 여부를 검사한다
(`pixel_features.py:32-46`). archive wrapper는 RGB 영상을 84x84로 resize,
grayscale, `/255` 정규화한 뒤 4-frame stack을 만든다
(`A/env_wrapper.py:7-10,13-28,54-56,85-86`).

속도 추정은 다음과 같다 (`pixel_features.py:128-139`).

```text
mass = sum(frame[77:83, 10:13])
speed = clip((mass - 0.27) / 0.085, 0, 80)
stack 입력이면 최신 두 HUD speed의 평균 사용
```

실제 renderer가 hull linear velocity 크기를 speed bar로 그리는 것은
`A/core/vendor/car_racing.py:741-750`에서 확인된다. 따라서 임의의 신경망
latent가 아니라 HUD의 물리 속도에 대응시키려는 값이다. 다만 rasterization,
resize, HUD 오염과 cap 때문에 정확한 simulator speed와 동일하다는 보장은 없다.
clip 전 HUD decoder 추정값이 80 이상이면 해당 frame은 80으로 읽힌다.
실제 물리 속도 80에서 정확히 포화한다는 보장은 없으며, 현재 frame이 포화해도
이전 frame이 낮으면 평균 speed는 80 미만일 수 있다. 두 프레임 평균에는 관측 지연이 있다.

### 5.2 도로 중심

asphalt는 grayscale `[0.24,0.52]`로 분류한다. row
`54,50,46,42,38,34,30`을 가까운 곳부터 따라가며 이전 center의
`+/-17 pixels` 안에 asphalt가 4개 이상이면 그 x 평균을 center로 저장한다
(`pixel_features.py:49-61`). 초기 x는 42다.

단일 연결된 road strip을 선택하는 방식이 아니라 선택된 pixel의 평균이다.
분리된 asphalt 조각이나 교차 도로를 섞을 수 있고, 화면 전체가 asphalt 색인
합성 입력도 정상적인 7개 centered rows로 해석한다. road가 없으면 빈 dict를
반환한다. `center_at()`은 빈 입력에서 42를 반환하고, 보이는 rows 밖에서는
`np.interp`의 endpoint 값을 사용한다 (`pixel_features.py:64-75`).

`road_sweep()`에는 없는 row를 center 42로 대체하는 항이 있으므로 그 값을
물리 curvature로 읽으면 안 된다 (`pixel_features.py:142-150`). 최종 preview는
주로 `max(centers)-min(centers)`를 사용한다. 어느 것도 도로 폭이나 heading의
정확한 측정치는 아니다.

### 5.3 Near/far 장애물

near detector는 road rows가 3개 이상일 때 작동한다. brightness `>=0.54`,
rows `[22,62)`, 8-neighbor component를 검사하며 area `4~80`, width `2~9`,
height `2~10`, interpolated road center에서 lateral `<=12 pixels`를 통과해야 한다.
가장 큰 y의 component 하나를 선택한다 (`pixel_features.py:78-125`).

far detector는 row 30부터 `26,22,18,14,10,6`까지 같은 `17 pixels` window로
도로를 연장한다. 한 row에서 asphalt가 부족하면 연장을 중단한다.
OpenCV components는 rows `[8,62)`에서 찾지만 far 후보는
`min(extended rows) <= y <22`이고 같은 component 크기/lateral 조건을 만족해야 한다
(`far_hazard_runtime.py:10-32`).

이 검출기는 장애물의 정답 identity/종류를 모른다. 밝은 road marking도 후보가
될 수 있고, 큰 component 또는 화면 가장자리 장애물을 무시할 수 있다.
near 객체가 있으면 arrival 제어는 far보다 near를 우선한다.

## 6. Parent Arrival 제어의 핵심

### 6.1 Steering, preview, impact

기본 road steering은 다음 형태다.

```text
road = 0.022 * (middle_x - 42) + 0.018 * (middle_x - near_x)
avoid = selected_side * 0.34 * clip((obstacle_y - 22) / 18, 0, 1)
damping correction = 0.025 * (current_middle_x - previous_middle_x)
```

clipping의 순서는 계층마다 다르다. damping은 현재 row 30/42/54와 이전
row 42가 존재하는 경로에서만 계산된다. row 42가 없고 54 및 두 개 이상 rows가
있으면 `observed_center()`로 middle을 선형 복원한다. 이 repaired 경로의
correction은 이미 0이므로 새 damping을 계산하는 것이 아니다
(`fixed_high_speed_runtime.py:39-50`, `fast_completion_coordination.py:10-15,48-53`).

impact proxy는 과거 속도보다 크게 낮아진 현재 속도로 trigger되는 자기 관측이다.
진짜 collision telemetry가 아니다. 선택 모드의 obstacle suppression은
방향 재획득 의도이며, 최근 최대 4 decisions의 executed brake가 `>0.01`이면
새 impact trigger를 braking-induced로 veto한다. veto는 base의 obstacle-side
latch까지 초기화하지는 않는다. 각 바깥 계층이 `brake_history[-1]`을 갱신해
다음 decision에는 최종 brake가 반영된다.
정확한 trigger는 launch 이후, 남은 impact timer가 0, 현재 speed `<50`,
이전 최대 4 readings의 최대에서 `>10` 속도 손실이다. timer를 12로 세팅한
현재 decision부터 최대 12개 steering suppression을 의도한다. 현재 row42/54가
없으면 timer가 남아도 road steering replacement는 적용되지 않는다
(`fixed_high_speed_recovery_v2.py:37-56`).

preview target은 `max(38, 60 - 0.8*sweep)`이고 near obstacle이면 최대 44다.
target이 60보다 낮으면 gas `clip(0.12+0.04*(target-speed),0,0.6)`,
brake `clip(0.02*(speed-target-2),0,0.28)`를 쓴다.
`target+2 < speed < target+3`에서는 gas와 brake가 동시에 양수일 수 있다.
예를 들어 target 44, speed 46.5면 gas 0.02/brake 0.01이다.
두 pedals가 독립적이므로 기본적으로 exclusive-pedal 모델은 아니다.

### 6.2 Swept sprint

예측은 24 samples, 각 모델 내부 timestep `0.04`, wheelbase 3.24, steering `[-0.4,0.4]`에 대한
단순 bicycle path다. 예상 화면 위치는 `(42+1.3608*x, 63-1.701*y)`다.
`py>55`면 sample을 건너뛰고, `py<6`, `px<4` 또는 `px>79`면 중단한다.
row6와 columns4/79는 포함된다. 3x7 footprint의 asphalt 비율 `>=0.85`를
만족한 마지막 거리를 사용한다. 이 모델의 최대 horizon은 `24*0.04=0.96` 모델
seconds이며, 실제 환경 decision의 nominal `4/50=0.08` seconds와 같은 timestep은 아니다.

계수 1.3608/1.701과 원점 42/63은 고정 renderer의 zoom과 resize에 대응한다:
`2.7*6=16.2`, `16.2*84/1000=1.3608`, `16.2*84/800=1.701`
(`A/core/vendor/car_racing.py:48-60,624-645`). 실제 차량 역학/타이어/손상 상태를
추정해 rollout하는 학습된 world model은 아니다.

현재 near obstacle 없음, row 42/54 관측, near center error `<3`,
`abs(parent steer)<0.18`, speed `<72`, clear samples `>=4`,
clear distance `>=max(18, proposal*0.4)`이면 gas 1/brake 0으로 override한다.
proposal은 `max(speed,72)`이므로 활성 구간에서는 사실상 72다.
그 후 arrival cap이 최종적으로 braking을 다시 적용할 수 있다.

### 6.3 Arrival cap

```text
obj = current near object, 아니면 current far object
d = max((63 - obj.y) / 1.701 - 7, 0)
cap = min(72, sqrt(44^2 + 110*d))
excess = pixel_speed - cap
excess > 0일 때 gas = 0
brake = max(parent_brake, clip(0.04*excess, 0, 0.6))
```

`110*d`는 `2*a*d` 형식의 거리-감속 budget이며 `a=55`를 가정한 것과 같다.
실제 손상/노면/제동 상태에서 이 감속이 가능함을 측정했다는 증거는 아니다.
물체가 있어도 speed가 cap 이하면 sprint gas를 미리 줄이지 않는다.

검출 가능한 최상단 y가 8이므로 유효 검출 객체에서 cap 최대는 약 68.72다.
y=22에서는 약 61.79, y=44에서는 약 48.94이며 y>=약 51.09에서 44가 된다.
따라서 식의 `min(72,...)` 상한은 유효 객체 좌표에서는 실제로 bind하지 않는다.

far tracking 상태는 계속 계산하지만 최종 `arrival_speed` 모드는 predicted track을
pedal 제어에 쓰지 않는다. detection dropout에 temporal hold가 적용된다고
해석하면 안 된다 (`far_hazard_runtime.py:56-74,95-103`).

## 7. 최신 Crossing Projection의 차이

`ContactContinuityAgent`는 arrival의 완성 action을 받은 뒤 **steering만**
변경한다. 새로운 speed target, reward, 학습된 모델이나 pedal policy가 아니다.
해당 클래스의 `impact_override`, `contact_coast`, `side_hysteresis`는 대안
모드이며 최종 `crossing_projection`에 동시에 활성화되지 않는다.

contact는 `observed_center(54)`와 `observed_center(42)`를 **각각** 호출한다.
따라서 projection이 활성화되는 경우에는 부모가 하지 않는 missing row54의
extrapolation도 사용해 road term을 다시 계산할 수 있다. 완전한 geometry에서는
기존 road term과 같지만, sparse geometry에서는 단순히 부모 steering에 avoid
계수만 더하는 것과 다르다. 이 복원은 projection 조건이 실패한 경우에는
action에 반영되지 않는다 (`contact_continuity_runtime.py:30-36,66-74`).

활성 조건은 다음과 같다 (`contact_continuity_runtime.py:24-74`).

1. Launch 10 decisions 이후다.
2. 현재 near object와 직전 decision의 near object가 모두 있다.
3. 현재 visible/extrapolated near/middle center로 road steering을 만들 수 있다.
4. 진짜로 해석된 impact-clear 경로가 아니다. 조건은 `(impact_before>0 or impact_proxy_trigger) and not braking_proxy_veto`다.
5. 현재 object y는 `[30,55]`, `dy=y-previous_y >1`, 화면 이동거리 `hypot(dy,dx)<=15`다.
6. 화면 y=60까지 projected object x가 차량 x=42의 6 pixels 이내다.

```text
projected_x = x + (x - previous_x) * (60 - y) / (y - previous_y)
if abs(projected_x - 42) < 6:
    avoid = base_selected_side * 0.55 * clip((y - 22) / 18, 0, 1)
    steering = observed/extrapolated road + strengthened avoid + correction
               (아래의 정확한 nested clipping 순서 적용)
```

기존 avoidance 크기 0.34를 0.55로 강화하지만, 피할 side를 새로 고르는 것이
아니라 `base._obstacle_side`를 그대로 쓴다. 선택 방향이 잘못되면 그 잘못된
방향을 강화할 가능성이 있다. 새 coefficient가 모든 frames에 적용되는 것도 아니다.

contact가 계산하는 `road_base`를
`0.022*(observed_middle-42)+0.018*(observed_middle-observed_near)`로 정의하면,
부모가 `geometry_repaired=True`로 표시한 경우에는
`clip(road_base+correction+avoid)`를 사용한다. 그렇지 않은 경우에는
`clip(clip(road_base+avoid)+correction)`으로 부모의 ordinary clipping 순서를 유지한다
(`contact_continuity_runtime.py:31-36`).

`previous_object=obj`를 매 decision 갱신하므로 객체가 한 번 없어지면 다음
decision의 projection에도 직전 객체가 없다. 물체 ID matching, optical-flow
ego-motion 분리, 여러 물체의 일관된 association은 없다. 가장 가까운 객체가
바뀌어도 proximity 조건만 통과하면 서로 다른 객체를 연결할 수 있다.

`contact_speeds`, `contact_left`, `pass_side`도 갱신되지만 crossing 모드의
steering 조건은 이 contact timer/hysteresis를 사용하지 않는다. 최종 모드에
8-decision coast나 12-decision steering hold가 있다고 읽으면 안 된다.
`last.contact_changed`는 실제 contact 전후 action 비교이며,
`mechanism_changed/completion_changed`는 더 안쪽 preview-parent 대비 비교다.

**기전 해석은 가설:** 연속 near 영상에서 물체가 앞으로 차량 중심선에 진입할
것으로 예상되면 짧게 lateral avoidance를 강화하여 arrival의 미완주를
구제하려는 설계다. 반복 충돌이 실제로 줄어 개선됐다는 인과 결론에는 raw
trajectory/검출/trigger 시점 분석이 필요하며 이번에는 해당 raw 자산이 없다.

## 8. 성능 증거의 수준과 해석

### 8.1 Arrival: 포함된 report로 확인 가능한 수치

`R/evidence/report.json`은 candidate TRAIN 재현 12건과 held-out 대조 24건,
총 36 episode records를 담는다. 아래 arrival held-out 값은 이 primary
report를 기준으로 확인한 값이다. 새 주행 평가에서 얻은 결과가 아니다.

| 코호트 | Historical stable | Arrival | 검증 수준 |
|---|---|---|---|
| 소비된 TRAIN, tracks 1~3 x seeds 38300~38303 | 12/12, 완주 중앙값 20.52 s | 12/12, 20.13 s | arrival 12건은 report에 포함. stable TRAIN 값과 최초 72-episode screen 전체는 아카이브 요약이며 원래 primary screen report는 별도 자산 |
| 당시 별도 held-out, tracks 1~3 x seeds 49300~49303 | 10/12, 19.34 s | 10/12, 18.46 s | 두 모델의 24 episode records 포함 |

held-out에서는 동일한 10개 conditions를 모두 완주했고, 모두 arrival이
0.66~1.16 s 빨랐다. 완주 중앙값 감소는 약 4.55%다.
두 모델 모두 `track2/seed49303`, `track3/seed49300`에서 off-track 미완주다.
19.34/18.46 s는 완주한 episodes만의 중앙값이며 전체 expected completion time이 아니다.

primary held-out의 완주 쌍과 미완주도 별도로 보존한다.

| Track:Seed | Historical seconds | Arrival seconds | Arrival - Historical |
|---|---:|---:|---:|
| 1:49300 | 19.98 | 19.10 | -0.88 |
| 1:49301 | 18.34 | 17.68 | -0.66 |
| 1:49302 | 19.28 | 18.44 | -0.84 |
| 1:49303 | 21.72 | 20.56 | -1.16 |
| 2:49300 | 19.78 | 19.02 | -0.76 |
| 2:49301 | 18.72 | 17.94 | -0.78 |
| 2:49302 | 19.24 | 18.46 | -0.78 |
| 2:49303 | DNF | DNF | 해당 없음 |
| 3:49300 | DNF | DNF | 해당 없음 |
| 3:49301 | 18.92 | 18.18 | -0.74 |
| 3:49302 | 19.40 | 18.46 | -0.94 |
| 3:49303 | 21.88 | 21.06 | -0.82 |

TRAIN arrival 12건의 collision-positive decisions 합계는 3, terminal damage 합계는
0.6이며 모두 `1:38303`에서 발생했다. held-out arrival은 collision/damage
합계 0/0, historical은 1/0.2다. 이 collision 수는 distinct obstacle 수나
raw physics contact 수가 아니라 wrapper collision flag가 true인 decisions 수다.

`2:49303`에서는 arrival progress 약0.347826, historical 약0.354037이므로
damage 감소에도 불구하고 진행도는 조금 낮다. `3:49300`에서는 arrival 약0.168350,
historical 약0.138047이다. 둘 다 미완주인 점은 변하지 않는다
(`R/evidence/report.json:601-691`).

resource audit 36건 전체의 cold import/create 최대는 약0.268228 s,
reset 최대 약0.000103321 s, act 최대 약12.574105 ms, RSS 최대284,217,344 bytes,
전체 duration 약243.246720 s다 (`R/evidence/batch_audit.json:166-178`).
arrival의 계산 지연은 historical보다 늘었다. held-out episode별 p50의 산술평균은
4.67226 ms 대3.88556 ms, p95의 산술평균은 5.82980 ms 대4.71232 ms다.
이는 **episode percentiles의 평균**이지 모든 decisions를 합친 percentile은 아니다.
공식 server의 단독 Agent latency/RSS로 재명명하지 않는다.

### 8.2 Crossing: 현재는 요약으로만 확인 가능한 수치

다음 값의 근거는 `A/docs/experiments/current-best-validated.json:42-65`와
`A/docs/experiments/contact-continuity-20260930.md`다. primary
`contact-package-20260930/report.json`은 아카이브 manifest에 있지만 현재
로컬에 없으므로 **독립적인 raw 재검증 완료 수치로 표현하지 않는다.**

| 코호트 | Historical stable | Arrival/current parent | Crossing baseline |
|---|---|---|---|
| 당시 별도 held-out, tracks 1~3 x seeds 51300~51303 | 12/12, 19.67 s | 11/12, 19.06 s | 12/12, 18.96 s |

archived report 요약에 따르면 package 검증은 TRAIN candidate replay 12건과
held-out 3모델 비교 36건, 총 48건이다. arrival의 미완주 condition은
`track3/seed51302`이며 crossing이 이를 완주했다. 두 모델이 함께 완주한
11 conditions에서는 crossing이 7건 느리고 3건 빠르고 1건 동일하며,
paired time delta 중앙값은 **+40 ms**다.

그 이전 consumed TRAIN screen의 documented crossing 결과는 tracks1~3 x
seeds50300~50303에서 11/12, arrival은 10/12다. crossing도 그 TRAIN 전체를
완주하지 못했다. 대안 `side_hysteresis` 역시 11/12였지만 다른 failure를 구제했다.
이 대안을 crossing과 조합한 정책이 최종 ZIP이라고 해석하면 안 된다
(`A/docs/experiments/contact-continuity-20260930.md:13-24`).

따라서 crossing의 채택 근거는 우선 **완주 보존/추가**다. 18.96 s가 19.06 s보다
작다는 aggregate만으로 동일 roads에서 전반적으로 더 빠르다고 주장하면 안 된다.
완주 집합과 denominator가 다르다. 이전 arrival 코호트의 18.46 s와 crossing의
18.96 s를 비교하는 것도 unmatched 비교다. crossing을 seeds49300~49303에
재평가한 결과는 이 증거에 없다.

두 held-out 코호트 모두 이미 소비되었다. 재튜닝하거나 fresh blind로 재명명하면
안 된다. 각각 4 seeds x 3 tracks의 좁은 로컬 비교이며, private tracks,
독립 training seeds, 공식 leaderboard나 서버 model confirmation의 증거가 아니다.
저장 기록상 공식 제출과 모델 확정은 모두 false다.

### 8.3 도로 수, 종료, 시간 및 parity의 정확한 의미

동일 seed의 geometry 생성 뒤 track ID에 따라 obstacle 배치를 달리한다
(`A/core/vendor/car_racing.py:276-283,465,504-522`,
`A/core/track_variables.py:45-48,60-71`). 따라서 각 코호트의
`3 tracks x 4 seeds =12 conditions`는 **4 geometry seeds의 장애물 조건**이지
12개의 독립 도로 geometry가 아니다. arrival TRAIN+held-out 24 conditions도
8 geometry seeds다.

당시 held-out freshness 근거는 기존 v2 manifests/plans에 대한 기록 검색이다
(`A/docs/plans/active/far-hazard-package-20260930-design.md:7`). 모든 lane의
전체 과거 exposure를 독립적으로 재검증한 결과와는 구분한다. confirmation/blind
미접촉도 저장 문서의 진술이며, 없는 전체 ledger까지 이번에 감사한 것은 아니다.

evaluator의 시간/종료 의미:

- 50 raw no-op warmup ticks 뒤 관측을 넘기고, decision당 raw ticks 4개, frame stack 4개를 사용한다. 로컬 decision cap은 1,200이다 (`A/training/env_factory.py:17-19,79-84,174-203`).
- `lapTimeMs`는 reset/warmup 이후 simulation 시작시각에서 `finish_time_s`까지의 차이다. wall-clock inference 시간이 아니다 (`A/training/evaluate_closed_loop.py:314-324,450-455`).
- 95% tile 방문은 finish qualification이며 그 뒤 올바른 forward finish-line crossing과 영역 이탈이 필요하다. 일부 완주 progress가 1 미만인 것은 모순이 아니다 (`A/core/finish_line.py:7,41-43,59-95`).
- wrapper의 off-track retirement는 geometric edge 판단이 아니라 100개 초과 consecutive negative-reward decisions다. evaluator는 별도 이유 없는 simulator termination에도 `off_track` label을 줄 수 있으므로 compact label만으로 기전을 확정하지 않는다 (`A/env_wrapper.py:64-83`, evaluator `:457-460`).
- compact package report에는 공식 scalar ranking score 또는 accumulated reward가 없다. lap time, progress, damage를 임의의 공식 점수로 바꾸지 않는다.

ZIP 검증 helper는 먼저 extracted ZIP의 agent를 import/create하고, 이미 import한
`haic_agent.*`가 extracted 경로에서 왔는지 검사한 뒤 evaluator paths를 추가한다.
각 condition은 fresh subprocess이며 전체 process timeout은 90 s다.
단, per-call timeout은 return 뒤 latency check이고 evaluator는
`per_call_timeout_enforced:false`로 기록한다. 공식 server의 hard per-call
isolation과 동일하다고 주장하면 안 된다.

`same_full_trace`는 steer/gas/brake, action 이후 car x/y/yaw, progress의
7 fields를 정확히 비교하고, completed/lapTimeMs/collisions/damage/retire_reason의
5 terminal outcomes를 추가로 비교한다. 모든 JSON field, 픽셀, wheel 상태,
Box2D 내부 solver state를 비교하는 것은 아니다. first10 prefix equality도
treatment/control의 전체 궤적 동일성을 뜻하지 않는다
(`A/training/confirm_contact_continuity.py:27-35,72-102`).

RSS에는 evaluator가 가져오는 Torch/network/planner 등도 포함된다. 이는 배포
런타임의 dependency와 별개다. 이 helper에는 statistical bootstrap,
confidence interval 또는 독립 반복 held-out cohort가 없다.

## 9. 코드 리스크와 증거 한계

| 항목 | 확인 수준 | 의미 |
|---|---|---|
| 최신 baseline ZIP/primary report 부재 | 확인된 아카이브 경계 | 코드는 분석했으나 최종 packaged crossing bytes/48 episodes를 재검증하지 못함 |
| 고정 grayscale/HUD/카메라 계수 의존 | 코드 사실 | renderer, resize, 조명/road palette 변화에 취약할 수 있음. 실제 일반화 실패로 단정하지 않음 |
| 장애물 identity와 ego-motion 분리 없음 | 코드 사실 | nearest-object 전환 또는 카메라 회전으로 잘못된 crossing prediction 가능 |
| 잘못 선택한 side를 강하게 유지 | 코드 사실 + 위험 가설 | crossing은 새 방향을 찾지 않고 기존 side coefficient를 강화 |
| Launch 첫 10 decisions의 추가 안전 개입 미적용 | 코드 사실 | fixed60 pedals/base obstacle steering은 있으나 preview/arrival/crossing refinement는 launch 후에 적용 |
| Damping의 안정화 효과 미입증 | 코드 사실 + 위험 가설 | center motion과 같은 부호의 보정이며 차량 lateral 안정성을 이름만으로 보증하지 않음 |
| Active steering slew limit 없음 | 코드 사실 | 기본/강화 avoidance가 한 decision에 크게 전환될 수 있고 actuator_rate 모드는 비활성 |
| Impact는 속도 손실 proxy | 코드 사실 | 의도한 braking/손상/도로 이탈을 충돌과 혼동할 수 있으며 veto도 완전 분류기는 아님 |
| 순간 cap 이하에서 gas 1 가능 | 코드 사실 | arrival은 anticipatory throttle taper가 아니라 cap 초과 후 개입. 가정한 감속 55의 실현 가능성은 미측정 |
| near frame dropout에 projection 중단 | 코드 사실 | far의 tracking 상태를 arrival/crossing의 실제 hold로 해석하면 안 됨 |
| diagnostics가 최종 action과 완전히 일치하지 않음 | 코드 사실 | base gas/brake/counters와 preview target은 바깥 override 이전 값. 분석에는 returned action과 final_gas/final_brake 사용 |
| penultimate HUD NaN이 final action으로 전파 | 합성 입력에서 재현 | 최신 frame만 검증하므로 `[0,NaN,NaN]` 가능. 정상 공식 입력에서 발생했다는 주장은 아님 |
| 잘못된 shape/최신 NaN은 exception | 합성 입력에서 재현 | final wrapper에 안전 fallback 없음. 비정상 관측 계약에 대한 한계 |
| final contact/far mode의 전용 unit tests 부족 | 아카이브 test 경로 확인 | 이번 합성 branch 검사와 과거 패키지 검증을 전체 분기/전체 episode 테스트로 확대 해석하지 않음 |

`pixel_features.py`의 module docstring은 learned policy가 driving decisions를
담당한다고 적지만, 실제 이 배포에는 learned policy가 없다. 주석이나
historical docs보다 실행 가능한 최종 dependency closure를 우선해야 한다.

## 10. 이번 작업의 검증 기록

검증은 source/hash/저장 JSON과 **환경 없는 합성 입력**에 한정했다.

- `git diff --exit-code 1e3dd6d c4e224d -- releases/arrival-speed-20260930`: 차이 없음.
- arrival ZIP SHA-256, manifest 9개 크기/해시, ZIP과 extracted source 동일성: 모두 통과.
- 포함된 arrival primary report/batch audit/official source audit의 해시는 아래와 같고, archive manifest의 원래 artifact 해시와 일치한다.
- 배포 9개 모듈 AST parse/import 분석: 통과. learned/environment dependency 없음.
- NumPy 1.26.0에서 archive의 기본 corridor/perception tests 11개를 release 소스 대상으로 실행: 11/11 PASS.
- 최신 crossing source와 동일한 release parent closure를 직접 import하여 합성 straight 관측 12 decisions 실행: 모두 finite, reset 후 동일 sequence.
- crossing의 mocked-parent branch 검사 7개: 중앙 crossing 활성, projection 바깥/작은 dy/row 범위 밖/impact-clear/launch/객체 없음 경로 확인. 7개 모두 parent pedals 불변.
- speed 20/40/60 합성 HUD는 설정값에 근접, 80/100은 모두 80, 이전20/현재60 stack 평균은 약40으로 확인.
- bright area 3/88은 무시, area4/80은 검출; 원래 near detector는 row10 객체를 검출하지 않음.
- 최신 frame finite이고 penultimate HUD에 NaN이 있으면 `[0,NaN,NaN]` 재현. wrong shape, newest NaN, 비정규화 uint8은 `ValueError`.

11 tests의 범위는 `A/tests/test_corridor_agent.py`의 기본 controller tests 8개와
`A/tests/test_visual_features.py`의 pixel/temporal extractor tests 3개다.
Torch actor/training/env를 사용하는 tests는 실행하지 않았다. crossing의
mock 검사는 부모 action을 대체하므로 실제 주행 개선이나 ZIP parity의 증거가 아니다.

| 포함된 evidence | SHA-256 |
|---|---|
| `R/evidence/report.json` | `6100506488b6b2bc14a5266a98150eea75600241a0251e2a1804680d07ed475b` |
| `R/evidence/batch_audit.json` | `0fcad2ff7bb745869cdbd8c2103bdd9d1d5aad5afd0ac653ad0bc6a26966060f` |
| `R/evidence/official_source_audit.json` | `7fea6a27eacdb66bd80f272d4f23577c1981348d150ee3ed438692daad964bdc` |

official source audit는 당시의 보관된 근거다. 이번에 공식 규정을 새로 확인하거나
submission validator를 실행한 것이 아니므로 현재 공식 acceptance를 보증하지 않는다.

## 11. 앞으로 유지할 Baseline 계약

이번 문서가 지정하는 개선 기준은 **crossing mode + 부모 arrival closure +
원래 ZIP 식별자 + 원래 평가 코호트/검증 수준**이다. 파일명만 같거나 새로운
패키지로 재압축한 결과를 같은 frozen baseline이라고 부르면 안 된다.

- 최신 모드 선택은 반드시 `crossing_projection`으로 명시한다.
- arrival은 이전 후보/부모 reference이며 현재 baseline과 성능 기록을 혼합하지 않는다.
- crossing 원본 ZIP과 primary/raw 자산을 확보하여 hash를 확인하기 전에는 최종 패키지를 완전히 복원했다고 주장하지 않는다.
- 비교는 완주 우선, preserved/lost/gained와 같은 조건의 paired 완주 시간으로 해석한다. unmatched 중앙값은 개선 근거로 쓰지 않는다.
- 소비된 TRAIN/held-out의 exposure를 유지하고 새로운 검증과 구분한다.
- 연구용 root agent, PPO/CEM, RLPD/DrQ/TD-MPC2 후보를 이 baseline으로 자동 교체하거나 그 결과를 승계하지 않는다.
- 원본/negative evidence를 보존하고 공식 제출/확정은 별도 사용자 승인 없이 수행하지 않는다.

**07:58 UTC baseline 분석 종료 시점에는 개선 작업을 시작하지 않았다.**

## 12. 별도 개선 후보: Adaptive Avoidance V1

09:56 UTC의 사용자 요청은 과도한 고정 속도 감속을 바꾼 별도 개선 버전 구현이다.
원래 crossing/arrival 소스와 선택된 baseline은 그대로 보존했다. 새 후보는
`haic/algorithms/koi/adaptive_avoidance.py`의 composition governor이며,
원래 crossing action을 받은 뒤 확실한 near passage에서만 pedals를 바꾼다.
구현 종료 시점에는 실제 완주/충돌/주행시간 개선을 검증하지 않았다.
이후 사용자 지정 unchanged A/B 결과는 14절에 기록한다.

수정 대상은 active preview의 `min(target,44)`와 arrival의
`sqrt(44^2+110*distance)` 두 제한이다. base의 43/36은 이미 fixed layer가
덮어쓰므로 그 상수만 바꾸는 것은 이 문제를 해결하지 않는다.

통과 속도는 다음 image-space budget으로 계산한다.

```text
pass_speed = 44 + 16 * clearance_quality * steering_quality * alignment_quality
road_target = max(38, 60 - 0.8 * visible_center_span)
arrival_cap = min(72, sqrt(pass_speed^2 + 110 * distance_to_arrival))
adaptive_target = min(road_target, arrival_cap)
```

- baseline이 현재 선택한 회피 side만 사용한다. 더 넓은 다른 쪽으로 steering을 바꾸지 않는다.
- near object의 실제 component bbox 바깥부터 같은 side의 연결된 asphalt 폭을 센다. bbox 전체 높이와 위/아래 인접 rows의 최솟값을 사용한다.
- 이미지/search window에 잘린 경계와 떨어진 asphalt island를 passage로 합치지 않는다. 기본 footprint 7 pixels와 reserve 2 pixels를 제외한 폭이 커질수록 속도 budget이 높아진다.
- 현재 steering의 bicycle arc가 visible approach rows에서 전체 3x7 asphalt footprint를 유지하고 선택된 obstacle bbox를 margin과 함께 지나갈 수 있어야 한다. 공간만 존재하지만 현재 명령이 중앙 장애물을 향하는 cancellation case에서는 완화하지 않는다. bbox top이 55보다 가까워 확인할 수 없는 경우도 fallback한다.
- 최종 crossing steering demand가 클수록 budget이 작아진다. 현재 steering이 선택된 회피 side와 반대이거나 절댓값 0.5 이상이면 기존 action으로 fallback한다.
- near center alignment가 나쁘거나 원래 curve budget이 44 이하이면 완화하지 않는다. road target보다 높은 주행 target을 새로 부여하지 않는다.
- pass budget과 최종 target의 상승은 모두 decision당 최대 1이며 위험 증가에 따른 하락은 즉시 반영한다. 첫 진입/재진입은 원래 44 budget부터 점진적으로 완화하고, fallback은 보수적인 baseline 상태로 되돌린다. curve budget 회복만으로 target이 갑자기 뛰는 것도 제한한다.
- 기존 preview/arrival brake를 `max(parent_brake,new_brake)`로 남기지 않고 두 constraint를 함께 재계산한다. 새 governor가 braking할 때는 gas를 0으로 두며 brake 상한 0.6을 유지한다.
- 실제 반환한 brake를 inner `brake_history[-1]`에 다시 기록해 다음 impact-veto가 실행되지 않은 원래 brake를 보지 않도록 한다.

Launch 1~10 decisions, near object 없음, far-only, missing road rows, 확인되지 않은
passing boundary, 좁은 통과 공간, impact hold/new trigger/contact proxy/급격한 속도
손실은 기존 action 그대로다. current-call steering은 바꾸지 않는다. 다만 이후
실행 brake history가 달라지므로 미래 impact veto 및 전체 steering sequence까지
원본과 항상 같다는 보장은 아니다.

44~60 endpoint, 폭 reserve, steering/alignment score와 기존 감속 계수 110은
**검증 전 설계 가정**이다. 도로/검출 신뢰성 gate가 있다고 실제 collision-free
속도가 증명되는 것은 아니다. 이후 실주행 검증은 frozen candidate를 같은
TRAIN 조건의 baseline과 비교하여 preserved/lost/gained finishes, collision/damage,
공통 완주 조건의 paired time을 평가해야 한다. 소비된 held-out에서는 튜닝하지 않는다.

재현 가능한 별도 패키지 생성 명령은 다음과 같다.

```bash
python -m scripts.package_koi_adaptive_avoidance --output submissions/koi-adaptive-avoidance-v1.zip
```

packager는 `c4e224d`의 Git blobs에서 arrival ZIP/manifest와 contact source를
검증하고, 원래 crossing entrypoint를 읽어 dependency closure를 복원한다.
원본 모듈은 수정하지 않고 adaptive runtime/새 entrypoint만 추가한다. 생성된
ZIP은 11 files이며 고정 metadata를 사용해 deterministic하다. 별도 JSON manifest는
원본 source hashes, candidate ZIP hash, synthetic smoke와 미검증 상태를 기록한다.
존재하는 후보/manifest는 덮어쓰지 않는다. 원래 crossing ZIP 자체를 복원한
것이나 공식 제출한 것으로 표시하지 않는다.

구현 검증 기록:

- `python -m pytest tests/test_koi_adaptive_avoidance.py tests/test_package_koi_adaptive_avoidance.py -q`: **153 tests + 2 subtests PASS**.
- steering cancellation, component의 row55 이하/이상 전체 footprint, cold prefix, 재진입, curve/거리 변화 시 target 상승률과 즉시 risk drop 회귀를 포함한다.
- 패키지 smoke는 prefix/no-obstacle/far-only equality와 실제 moving-object crossing activation의 fallback parity를 확인한다. Agent reset만 있으며 environment reset은 0이다.
- ZIP/receipt 생성 오류는 해당 실행이 만든 파일만 rollback한다. 기존/동시 생성 receipt를 보존하는 회귀 tests도 통과했다.
- 생성 후보: `submissions/koi-adaptive-avoidance-v1.zip`, SHA-256 `5f7057a432074d2e7215f5d2aeb863aafc4d557d9ba93c9fb3d34ee2b46e679c`.
- 생성 receipt: `submissions/koi-adaptive-avoidance-v1.manifest.json`.
- 고정 합성 wide-passage smoke의 12번째 call에서는 같은 steering 0.34를 유지하면서 brake가 약0.59378에서0.51574로 낮아지고 adaptive target은46이다. 이 수치는 제어 분기 재현이며 실주행 속도/완주 개선 증거가 아니다.

## 13. 사용자 지정 Consumed-TRAIN A/B 검증

11:15 UTC의 사용자 지시는 추가 모델 구현보다 unchanged `crossing_projection`과
`koi-adaptive-avoidance-v1`의 실제 matched 주행 검증이 먼저다. 중단 세션은 평가
operator를 작성했지만 환경 reset/주행은 시작하지 않았다. 재개 세션은 두 모델의
코드/ZIP을 그대로 유지하고 평가 도구의 오류 기록과 측정 기준만 보완한다.

고정 조건은 tracks 1/2/3 x seeds 38300~38303, 50300~50303의 24쌍/48 episodes다.
독립 도로 geometry는 8개이며 track ID는 장애물 배치를 바꾼다. 38300 cohort의
기존 TRAIN 소비는 포함된 primary report로, 50300 cohort는 frozen TRAIN 선언과
integration result로 확인했다. 원본 contact primary report 부재는 그대로
공개한다. historical held-out 49300~49303, 51300~51303 및 confirmation/blind/private
조건은 사용하지 않는다. fresh generalization이나 공식 성능 검증이 아니다.

평가 전 고정 기준:

- 48 scheduled slots, 모델/package/member/source/runtime hashes, reset intents와 오류/미실행 slots를 보존한다. CPU21 Python/Torch/Pygame runtime에서 양쪽에 동일한 50-tick warmup/4-tick action repeat/max1200 decisions를 적용한다.
- Agent에는 원래 pixels만 전달한다. 평가 전용 observer가 매 raw 50Hz tick의 actual speed, 전체 hull/wheel fixture clearance 및 obstacle contacts를 기록한다.
- 검출은 pre-action pose에서 world obstacle center를 화면으로 투영하여 4px 이내의 유일한 ID에만 연결한다. unmatched/ambiguous 검출 및 미통과 장애물을 분모에서 삭제하지 않는다.
- 검출부터 전체 차체의 rear clearance까지의 시간과 같은 obstacle anchor 앞 25 simulator units의 공통 진입부터 통과까지를 분리한다. 무한 tangent plane의 먼 구간 교차, seam/nonlocal/reverse station ambiguity는 유효 passage로 세지 않는다.
- clean passage는 sampled positive clearance, no contact/raw collision/overlapping collision decision을 모두 요구한다. 44는 target이며 실제 속도 ceiling이 아니다. adaptive activation과 실제 min/mean/max 속도 및 baseline의 44 초과 통과도 별도로 보고한다.
- 채택 후보 gate는 lost finishes=0, 모든 paired cell의 damage/collision decisions 비증가, 기존 clean obstacle의 새 hit 없음, 유효 공통 진입 segment의 per-cell mean time 감소, 서로 다른 최소 2개 geometry seed에서 adaptive 변경이 연결된 clean 전체 passage의 raw minimum speed>44다. raw tick 시간 해상도는 20ms이며 작은 차이는 quantization 한계를 공개한다.
- DNF는 terminal/last-window contact, lateral/heading, wheel-road contacts, wrapper/simulator 원인으로 분류하되 단순 시간상 연관을 causal mechanism으로 단정하지 않는다. 명백한 trace-supported defect가 있을 때만 한 번의 모델 수정/전체 재평가가 허용된다.

실행 도구는 `scripts/evaluate_koi_adaptive_ab.py`, 수동 trace 분석 도구는
`scripts/analyze_koi_adaptive_ab.py`다. Protocol과 primary episode/raw artifacts는
새 run 경로에 보존했다. 완료된 실제 결과와 채택 판정은 아래에 기록한다.

## 14. 첫 Unchanged A/B 결과: 채택하지 않음

실행 명령:

```bash
python -m scripts.evaluate_koi_adaptive_ab --output runs/koi-adaptive-ab-20260930-v1
python -m scripts.analyze_koi_adaptive_ab --run runs/koi-adaptive-ab-20260930-v1 --output experiments/koi-adaptive-ab-v1-result.json
```

Primary 결과는 [result JSON](../../experiments/koi-adaptive-ab-v1-result.json),
[protocol](../../runs/koi-adaptive-ab-20260930-v1/protocol.json),
[episode report](../../runs/koi-adaptive-ab-20260930-v1/episode-report.json)와
각 episode의 decision JSON/50Hz raw JSONL이다. 48 slots와 reset intents 모두
보존했으며 source/model/member/raw hashes, initial observations/state,
first10 action/world parity 검증이 통과했다. runtime은 Python3.11.14,
Torch2.1.0+cpu, NumPy1.26.0, OpenCV4.8.1, Gymnasium0.29.1, Box2D2.3.5다.
Pygame/Python을 포함한 정확한 버전은 protocol과 각 episode에 고정했다.
유효하지 않은 action, operational errors, planned censor는 모두 0이다.
별도 [독립 primary 감사](../../experiments/koi-adaptive-ab-v1-audit.json)는
48 episode/48 raw hashes, 45,581 raw ticks와142 environment Git blob pins를
재확인하고, 완료 passages만의 속도 및 실제44 초과 사례를 원시 trace에서 검증했다.

| 지표 | crossing_projection | adaptive_v1 |
|---|---:|---:|
| 완주 | 21/24 | 21/24 |
| terminal damage 합계 | 1.4 | 1.4 |
| collision-positive decisions 합계 | 7 | 7 |
| 전체 24 episodes 시뮬레이션 주행시간 합계 | 455.58 s | 456.04 s |
| 공통 완주 21조건 lap time 평균 | 19.054286 s | 19.079048 s |
| 공통 완주 lap time 중앙값 | 19.040 s | 19.040 s |
| near 검출→통과 평균, 138 passages | 0.616377 s | 0.618261 s |
| far 검출→통과 평균, 130 passages | 0.761538 s | 0.762462 s |
| 같은 anchor 앞 25 units→통과, 유효 120 pairs | 0.649009 s | 0.649890 s |
| 동일 완료 138 passages의 구간별 평균 실속도 평균 | 42.943732 m/s | 43.002545 m/s |
| 전체 catalogue obstacles / 실제 완료 passages | 144 / 138 | 144 / 138 |
| clean physical passages | 138 | 138 |
| 전체 physical 구간의 min actual speed>44인 clean passages | 0 | 1 |

완주 kept21/lost0/gained0/neither3이며 모든 cell의 damage/collision 수가 동일하다.
새로운 baseline-clean obstacle hit도 없다. 공통 완주 lap의 paired 평균 변화는
**+24.762 ms**다. 유효 segment의 per-cell mean delta를 평균하면
**+0.839 ms**이며 단축된 cell mean은 없다. weighted segment 평균 변화는
+0.881 ms다. raw crossing 시간은 20ms 단위로 bracket되므로 이 작은 segment
차이를 정밀한 slowdown 증명으로 해석하지 않지만, **시간 단축은 관찰되지 않았다.**
서로 다른 검출 시점이 있어 near/far duration과 공통 진입 duration은 분리한다.
공통 완주21 cells와 유효 segment21 cells는 동일 집합이 아니며 overlap은18이다.
속도 표는 같은 완료138 passages만을 사용한다. Primary result의 n140 speed
summary에는 미완료 조우2개가 들어 있으므로 이를 완료 구간 평균으로 표시하지 않는다.
공통 완주 lap은 adaptive가1개에서20ms 빨랐고5개에서 느렸으며15개는 같았다.

### 실속도 44 초과와 개입 빈도

adaptive는 5,710 decisions 중 9회 active, 7회만 실제 pedals를 바꿨다
(약0.123%). 실제 바뀐 호출의 pass/target budget은 45였으며 높은 budget을
연속 유지하는 회피가 널리 발생하지 않았다. near object 없는4,521 calls 외에도
impact guard502, passage unobserved106, steering guard105, narrow passage90,
swept-path guard88 등에서 기존 제어로 되돌아갔다. 이는 측정된 낮은 개입률이며,
어느 guard를 완화하면 안전하게 빨라지는지는 아직 검증되지 않은 가설이다.

`3:38300`, obstacle2의 실제 clean passage에서 sampled speed는
**min44.161963 / mean44.868360 / max46.344395**, 최소 fixture clearance는
2.572856 units였다. step143, t12.38s에서 adaptive가 target45로 gas를
0.085951에서0.125951로 높인 기록이 있으며 physical overlap은12.48s,
전체 rear clear는12.66s다. peak만44를 넘은 것이 아니라 이 한 passage의
모든 sampled 속도가44보다 높았다. 다만 한 geometry의 한 사례로 안전한
고속 회피가 일반화된다고 판단하지 않는다.

같은 baseline passage는 min43.979798 / mean44.687551 / max46.169116이었으며,
두 arm의 overlap/rear-clear tick과 near 검출→통과0.68s는 동일했다. adaptive
step143의 near detection은 obstacle2에 유일하게 연결되고 projection residual은
0.591308px였다. 첫 state/action divergence 직전 pre-state는 정확히 같았다.
따라서 실제 개입과 이 한 clean44 초과 사례의 연결은 확인되지만, **그 passage가
빨라지지는 않았고 해당 lap은60ms 느렸다.** Baseline도 완료138 passages 중
114개에서 peak actual speed가44를 넘었으므로 target44를 hard speed cap으로
취급하거나 peak 초과 자체를 adaptive 개선으로 세지 않는다.

38300의 세 track/layout 조건의 초기 nearest station이 reset seam에서
이미 entry station 뒤로 분류되어 `initially_beyond_entry`로 공통 진입 통계에서
제외됐다. 그래서 frozen analyzer의 `qualifying_geometry_seeds`는 빈 배열이다.
**그 값은 실제44 초과 사례가 전혀 없다는 뜻이 아니다.** 120 eligible pairs와
138 complete passages를 구분하고 이 측정 한계를 공개한다. 결과를 보고 seam
eligibility를 바꾸거나 이 사례를 gate 통과로 재분류하지 않았다. 최소 두 geometry의
반복된 clean 고속 회피 및 장애물 구간 단축 gate는 통과하지 못한다.

### 판정

완주/안전성 보존 gate는 통과했지만 시간 단축과 반복된 고속 회피 gate는
실패했다. 따라서 adaptive-v1을 채택 후보로 올리지 않고 crossing baseline을
유지한다. 실패 원인은 별도 trace 검토로 분류하며, 낮은 개입률이나 작은 시간
차이만으로 안전 guard의 구현 오류를 단정하지 않는다. 명백한 adaptive defect가
확인되지 않는 한 모델을 수정하거나 두 번째 평가를 실행하지 않는다.
TRAIN 소비 조건의 내부 proxy 결과이며 공식 score/새 도로 일반화 결과가 아니다.

### 실패 Trace 분류

| 조건 | 실제 분류 | 주요 trace 근거 |
|---|---|---|
| `1:38302` | obstacle5 접촉 후 반복 damage로 crash | step234~238의 collision-positive decisions 5회가 damage를1.0까지 올렸다. raw contact는19.70~20.06s에 지속됐으며 초기 접촉에서 실제 속도가 약41.63→13.90으로 떨어졌다. terminal은 wrapper_crashed=true이며 wheels에는 road contacts가 남아 있다. |
| `2:38302` | 비접촉 도로 이탈과 재획득 실패 | episode 전체 collision/damage/contact=0. step112 sparse-row geometry repair가 steering을-0.7로 바꾼 뒤10.18s부터 모든 wheels가 road를 잃었다. steps115~214는 road rows가 비어 있고 steering=0이며 progress도 더 늘지 않았다. wrapper_off_track 종료다. 단순 overspeed나 crossing override의 인과적 결함이라고 단정하지 않는다. |
| `2:50302` | obstacle4에 걸린 접촉/stall 후 no-progress 종료 | 두 collision-positive decisions로 damage0.4. obstacle4 contact는11.88~19.98s에 지속됐으며 terminal 속도는 거의0, wheel contacts는[1,1,1,1], progress0.545763이다. 여기의 off_track은 실제 잔디 이탈이 아니라100초과 negative-reward decisions에 따른 wrapper 종료다. |

위 시간은 raw simulator clock이며 warmup 이후 주행시간은 각 clock에서1.02s를
뺀다. 세 실패쌍 모두 전체 action/position/yaw/progress trajectory와 raw trace가
동일하고 adaptive active/changed decisions가0이다. 따라서 **adaptive 때문에
새로 생긴 실패가 아니라 두 모델이 공유하는 baseline 실패**다. 마지막 case에서
episode 전체에 individual wheel contact 손실이 전혀 없었다고 주장하지는 않는다.

Inherited `impact_steps_remaining`은 중간 stage의 stale diagnostic일 수 있다.
예를 들어 첫 crash에서는 이 값이11이어도 최종 braking veto가 지운 실제
`actual_impact_left`는0이다. 회복 hold 판단은 passive observer가 기록한 actual
state를 사용했다. 세 번째 stall에서는 실제 hold가141~151 decisions에 있었다.
명백한 adaptive 구현 결함이 확인되지 않아 governor/ZIP을 수정하지 않았고,
사용자가 조건부로 허용한 두 번째 전체 평가는 실행하지 않았다.

## 15. Minimum-Clearance 가설과 Adaptive-v1 종료

2026-09-30 14:06 UTC 사용자 지시로 adaptive-v1은 **실패 종료 / 비채택**으로
고정한다. 12~14절의 frozen source/ZIP/negative evidence는 보존하고 속도 target
조정 방향을 더 추적하지 않는다. 기준선은 계속
`ContactContinuityAgent('crossing_projection')`이며 새로운 후보로 교체하지 않는다.

별도 가설은 현재 회피가 필요한 것보다 큰 안전 margin을 사용해 과도한 횡이동,
긴 경로와 도로 이탈을 만든다는 것이다. 이 주장은 아직 검증 전이다. 후보는
steering 전체에 상수를 곱하는 방식이 아니라 관측된 obstacle 경계, 실제 hull/wheel
footprint에 대응하는 차량 폭, 연결된 asphalt 경계로 통과 가능한 최소 lateral
target을 계산한다. Projection 소멸 후에도 inherited side/avoidance가 남는지
확인하고, 여전히 차체와 obstacle이 겹칠 위험을 버리지 않는 범위에서 복귀한다.
속도 target과 frozen baseline은 변경하지 않으며 runtime은 pixels/HUD/history만
사용한다. 정답 obstacle/road/pose는 평가 observer 전용이다.

평가는 13절과 같은 consumed TRAIN 24 matched cells / 8 geometry seeds만 재사용한다.
완주 kept/lost/gained, 조건별 damage/collision, 새로운 baseline-clean obstacle hit,
전체 fixture 최소 clearance, 최대 lateral deviation, 동일 road-station 회피 구간
path length, steering 크기/적분, rear-clear 이후 sustained centerline 복귀와 lap time을
함께 측정한다. Seam·미통과·미복귀·미완주·operational failure는 삭제하지 않는다.
안전성과 기존 완주를 보존하고 유의미하게 짧은 회피 경로를 보일 때만 내부 채택
후보로 판단한다. 명백한 원인이 trace로 확인되면 별도 source/ZIP/run으로 직접
수정·재평가하며 최초 시도는 보존한다. 신규/protected/공식 평가나 자동 제출은 없다.

첫 reset 전 기록: 당시 별도 후보 구현과 zero-reset 검증만 완료했고 실제 주행
결과는 없었다. Measurement, gate와 source/runtime hashes를 첫 reset 전에
protocol로 고정했다. 이후 완료 결과와 직접 수정·재평가는16~18절을 따른다.

### 고정 소스에서 확인한 Margin과 회피 종료

기존 crossing은 실제 clearance 목표나 차량 폭을 계산하지 않는다. Near component의
중심을 두 프레임으로 extrapolate하고 `abs(projected_x-42)<6`이면 회피 항을
`side*0.55*urgency`로 바꾼다. 이 **6px는 collision-projection trigger band**이며
차체와 obstacle 표면 사이의 보장된 margin이 아니다. Trigger가 꺼져도 내부
corridor가 현재 near object에 대해 `side*0.34*urgency`를 더하므로, projection
비활성화와 회피 비활성화는 같지 않다. Side latch는 검출 누락 두 번째 call에
지워지지만 검출이 사라진 call에는 이 회피 steering 항 자체를 더하지 않는다.
근거는 고정 `A/haic_agent/contact_continuity_runtime.py:66-74`와
`A/haic_agent/corridor_agent.py:160-204`다. Projection 소멸만으로 rear-clear를
단정할 수 없으므로 조기 복귀에도 재진입 collision 검사가 필요하다.

기존 swept-sprint의 3x7 asphalt sample은 가속 허용 검사이며 obstacle minimum
clearance planner가 아니다. 물리 footprint는 vendor `SIZE=0.02`, hull 최대
반폭 `60*SIZE=1.2` units, wheel anchor 반폭 `55*SIZE=1.1`, wheel 반폭
`14*SIZE=0.28`, 반길이 `27*SIZE=0.54`와 joint limit `+/-0.4rad`에서 계산한다.
Steered wheel의 nominal 최대 반폭은
`1.1+0.28*cos(0.4)+0.54*sin(0.4)`로 약1.568183 units이며 hull보다 크다.
Polygon fixture skin0.01을 포함한 반폭은 약1.578183, 전체 폭은3.156366 units다.
Hull nominal 전방/후방 extent는 `+130*SIZE=2.6` / `-120*SIZE=-2.4`이며
skin 포함 extent는 `+2.61/-2.41`다. Raster/resize 불확실성은 별도다.
근거는 `A/core/vendor/car_dynamics.py:17-38,84-128`다. 새 후보의 물리 여유는
평가 시 동일 full hull/wheel fixture 거리로 검증하며, pixel 목표만으로 안전을
입증했다고 표시하지 않는다.

### 별도 후보와 평가 전 고정 범위

구현은 `haic/algorithms/koi/minimum_clearance.py`이며
`MinimumClearanceAgent`가 immutable crossing driver를 감싼다. 기존 선택 측면에서
`0.05px` 간격으로 가장 작은 지원 가능한 lateral displacement를 찾고, smooth
lane reference의 전체 yaw-expanded hull/wheel footprint, obstacle component box,
연결 asphalt 경계를 검사한다. Steering은 원래 corridor의 `0.022/0.018` gains로
그 target lane에서 계산하며 steering 전체 상수 배율이 아니다. 속도 target/pedal
계산은 그대로다. 동일 관측에서는 pedals가 정확히 같지만 주행 경로가 달라지면
후속 관측과 그에 따른 pedals는 달라질 수 있다.

Skin 포함 차량 반폭은2.147591403px, enclosing 반길이는2.61units다. 추가 raster
allowance는 obstacle edge1px + footprint/sample0.5px다. 따라서 직선 기준
관측 obstacle edge부터 차량 중심까지의 최소 분리는3.647591403px이며 yaw
확장과 `<0.05px` grid rounding은 별도다. 이는 raster/model 가정의 최소이며
최적 물리 경로나 안전 보증이 아니다. 지원 asphalt rows는8~69이고, 현재 hull와
wheel 각도 범위의 명시적 silhouette만 occlusion mask로 사용한다. 보이는 row56
삭제나 mask 밖 wheel-support hole은 proposal을 거부한다.

실제 최종 steering과 damping correction의 full-command kinematic sweep도
별도로 검사한다. 미래까지 현재 steering이 계속된다고 보는 보수적 rear-clear
guard라서 countersteering이 필요한 centered 통과는 기존 action으로 돌아갈 수
있다. Reference 추종/실제 차량 역학은 미검증 가설이다. Supported return은
회피 projection이 사라졌다는 이유만으로 활성화하지 않고 reference와 applied
command 양쪽의 clearance가 필요하다. Near crop/dropout 시 기존 action parity를
보존하며 이를 physical rear-clear로 표시하지 않는다.

첫 별도 package는
`submissions/koi-minimum-clearance-v1.zip`, SHA-256
`ddadbc512b02469f96d04785f27c7a74045ad0f26582c779780a750aac17a73e`다.
11개 members 중 unchanged baseline dependencies는 고정 source와 같다. Manifest의
zero-reset package smoke를 실제 CPU21 interpreter에서도 재현했으며 같은 ZIP
hash를 얻었다. 독립 검토가 찾은 applied-command 검사 누락과 visible row56
support 누락은 첫 실주행 전에 수정하고 regression으로 보존했다.

평가 도구는 `scripts/evaluate_koi_minimum_clearance_ab.py`와
`scripts/analyze_koi_minimum_clearance_ab.py`다. Matched window는 동일 obstacle
anchor의 station-25부터 station+25까지다. Candidate-invalid window를 삭제하고
남은 평균만 채택하지 않도록 baseline-eligible window 보존을 필수 gate로 한다.
Baseline whole-episode에서 hit가 없던 obstacle은 미통과 여부와 무관하게 새 hit를
허용하지 않는다. 실제 첫10개 action/world parity와 full-episode censor 없음도
요구한다. Measured projection-off는 유한한 projected x가 `abs(x-42)>=6`일 때만
세고 unavailable projection과 단순 crossing-branch inactivity를 분리한다.

미리 정한 내부 의미 있는 단축 기준은 cell-weighted fractional path mean이
`<=-2%`이고 최소 두 geometry seed의 mean에서도`<=-2%`다. 기존 완주 손실0,
조건별 damage/collision 비증가, 새 no-hit-reference obstacle hit 없음, baseline
window 보존과 공통 완주 mean lap time 비증가를 모두 요구한다. 이는 유의성 검정
또는 fresh generalization 기준이 아니라 consumed TRAIN 개발 판정이다. Rear-clear
후 `abs(lateral)<=1unit`가0.24s 이상 유지되는 복귀를 측정하며 censored/unpassed
분모를 숨긴 returned-only 전체 평균은 만들지 않는다.

## 16. Minimum-Clearance 첫 결과와 Hold-Horizon 수정

첫48 episodes는 완료했고
[원래 result](../../experiments/koi-minimum-clearance-ab-v1-result.json)와
[run protocol](../../runs/koi-minimum-clearance-ab-20260930-v1/protocol.json)를
그대로 보존한다. 양쪽 완주21/24, kept21/lost0/gained0/neither3, damage 합1.4,
collision-positive decisions7이다. 새로운 baseline-no-hit obstacle hit, censor,
operational error는 없으며119 baseline-eligible paired windows가 보존됐다.

| 첫 비교 지표 | crossing_projection | minimum_clearance_v1 |
|---|---:|---:|
| 같은119 회피 window path 평균 | 50.483046 units | 50.504859 units |
| 같은119 window 최대 abs lateral의 평균 | 6.207286 units | 6.201403 units |
| Episode 전체 wheel-road contact-loss ticks | 1703 | 1678 |
| 유효 projection-off + 회피 steering decisions | 423 | 423 |
| 실제 steering 변경 decisions | 0 | 5 |

Cell-weighted fractional path 평균 변화는+0.038408%이며 공통 완주21 laps 평균
시간 변화는+8.571429ms다. 유의미한 경로 단축이나 빠른 lap으로 판단하지 않는다.
이 첫 후보는 **비채택**이다. Wheel contact-loss는 단일 wheel만 떨어져도 포함하는
proxy이며 실제 전차량 도로 이탈이나 official score와 같지 않다.

명확히 확인한 correction 대상은 applied-command guard의 horizon이다.
실제 action은4 raw ticks/0.08s마다 갱신되지만 첫 guard는 동일 steering을 full
rear-clear까지 유지한다고 extrapolate했다. 충분한 desired-lane reference도
그 미래 constant-turn 가정으로 막힐 수 있다. 이것이 수정 후 안전하거나 짧은
경로를 만든다는 인과적 증거는 아직 아니다.

별도v2는 full desired-lane footprint/obstacle/road clearance와 기존 margin을
그대로 유지하고, 적용 command만 실제0.08s hold distance에서 검사한다. Current와
최근4 평균 readings 및 observation stack의 개별 HUD readings를 모두 검사한 뒤
최대값으로 거리를 잡는다. 평균에 가려진 신규/이전 frame의80 포화나 임의 입력의
비유한 값은 fallback하며 최대 속도가 비양수여도 fallback한다.
속도 target/pedals 계산은 바꾸지 않는다. Default full-horizon mode와 첫 ZIP 및
run source copies는 보존하고, 명시적 `command_hold_seconds=.08` 후보를 새 ZIP과
동일48-slot r2 protocol/run으로 평가한다. HUD lag·미래 feedback 추종·타이어 응답은
여전히 모델 불확실성이므로 실제 완주/안전/경로 gate를 다시 통과해야 한다.

[독립 첫-result audit](../../experiments/koi-minimum-clearance-ab-v1-audit.json)은
48 episode/raw/process와150 frozen source copies를 검증했다. 새 baseline24개는
이전 adaptive A/B baseline의5703 decisions/22779 raw ticks에서 action, pose,
scalar speed와 contact/progress가 정확히 반복됐다. Raw에 velocity-vector나
angular velocity가 없으므로 hidden-state 전체 일치 주장은 하지 않는다. 평가 후
의도적으로 변경한 live packager 대신 첫 run의 copied helper/ZIP을 검증했으며,
첫 result는 수정하지 않았다. Source-drift gate 때문에 현재 live packager를
사용한 첫 protocol 재분석은 거부되는 것이 정상이다.

같은138 complete physical passages의 최소-clearance 평균은4.667251->4.667235
units, 그 최소값은1.371783->1.444876이다. 최소값 identity는 다르며 전체138에서
각 passage minimum의 평균/최솟값이지 DNF collision을 숨긴 안전 gate가 아니다.
119 matched windows의 abs steer time-mean 평균은0.141869->0.142469, max 평균은
0.421853->0.426704, integral 평균은0.148389->0.149157 command-s다.
144 return rows는 양쪽 모두 returned1/window-exit censored118/unpassed6/invalid19;
both-returned는 단1 pair의0.24->0.24s다. 전체 복귀시간 평균이나 복귀 개선을
주장하지 않는다. 38300 seam18 windows와 기타7 exclusions는 삭제하지 않았다.

## 17. Actual-Hold R2의 안전 실패와 Projection 수정

[r2 result](../../experiments/koi-minimum-clearance-ab-r2-result.json)와
[독립 audit](../../experiments/koi-minimum-clearance-ab-r2-audit.json)는48 episodes를
검증한다. 완주21->20, kept20/lost1/gained0, damage1.4->1.6, collision decisions
7->8이다. 손실 조건은3/38301이며 obstacle3에 baseline0/candidate404 contact ticks,
최소 clearance1.942127->-0.012023이 기록됐다. Baseline119 window 중 object3/4/5
세 개가 사라져116 survivors뿐이다. Survivor window path와 conditional kept20
lap-2ms가 줄어도 안전/완주/분모 손실을 이익으로 바꿀 수 없다. **v2 비채택.**

[단일 조건 diagnosis](../../experiments/koi-minimum-clearance-r2-lost-finish-diagnosis.json)는
처음30 actions/120 ticks가 같고 후보 override가 단1회임을 확인한다. Step31에서
current-motion projected x38.633803의 차체 중심42까지 분리는3.366197px이지만,
full wheel halfwidth2.147591 + measured box extent1 + 기존 raster1.5로 계산한
요구 분리는4.647591px이다. Desired reference의 positive margin과 actual.08s
command feasibility만 확인하고 이 현재 움직임과 목표 경로의 차이를 놓쳤다.

그렇다고 최초 object0을 충돌한 것은 아니다. Object0은 실제 최소2.661434로
안전 통과했다. 실제 collision은 약7.6s 뒤 obstacle3, 전부 inherited impact-guard
actions 중 발생했다. Frozen corridor의 `abs(offset)<=1` motion branch에서
offset-0.944318/motion+3.555682로 flank가+/-/+ 전환했고 한 call은 steer-0.7이다.
Baseline offset-1.018941은 그 branch에 들어가지 않고+flank를 유지했다.
Retirement label은 off_track이지만404 contact ticks 모두 wheel-on-road이며,
실제 관측은 장애물에 걸려 거의 정지한 상태다. Premature return, 잘못된 steer
방향, 최초 장애물 margin 부족 또는 bbox 오류를 확인했다고 하지 않는다.

직접 수정한 별도v3는 minimum target을 적용하기 전에 **현재 motion projection도
footprint가 통과하는지** 검사한다. Threshold는 물리 차량 반폭, 현재 component
centroid부터 bbox 양 끝까지 최대 extent와 기존1.5px allowance로 계산하며 고정6px
재사용이나 steering 상수 배율이 아니다. 유한한 projection이 선택한 통과 측면의
반대쪽에서 그 threshold보다 충분히 떨어져야 한다. Unknown/nonfinite/wrong-flank/
touch면 baseline action 전체를 보존한다. Full desired-lane 및.08s command 검사는
그대로다. `require_projection_clearance=True`와.08 config를 v3 entry에서 명시한다.
기존 v1 default와 strict-off v2 동작은 frozen ZIP 대비200-input 검증에서 같고,
v3도 동일 관측 pedals/target이 같다. 이 수정은 미래 추종이나 baseline side latch의
전면 재설계가 아니며 실제 안전/단축은 r3에서 다시 평가한다.

## 18. 최종 R3: 안전 보존, 의미 있는 단축 없음

v3 ZIP `594fca15005ec38cc67801824e9725fc84da440c2309083f02a24955b0ab8fca`를
별도 [r3 protocol](../../runs/koi-minimum-clearance-ab-20260930-r3/protocol.json)에
고정했고, [r3 result](../../experiments/koi-minimum-clearance-ab-r3-result.json)는
같은24 consumed TRAIN layouts/48 episodes 전부 valid다. 완주21/24, kept21/lost0/
gained0/neither3, damage1.4/collision7, 새 hit 없음, baseline119 windows 모두 보존한다.

| 같은119 matched window 지표 | crossing_projection | minimum-clearance v3 |
|---|---:|---:|
| Path 평균, units | 50.483046 | 50.412679 |
| Window별 max abs lateral 평균, units | 6.207286 | 6.151952 |
| 전체 max abs lateral, units | 9.334206 | 9.608379 |
| Window별 time-mean abs steering 평균 | 0.141869 | 0.142343 |
| Window별 max abs steering 평균 | 0.421853 | 0.427649 |
| 전체 max abs steering | 0.700000 | 0.700000 |
| Abs steering integral 평균, command-s | 0.148389 | 0.148885 |
| 전체 wheel-road contact-loss ticks | 1703 | 1632 |

같은138 complete physical passages의 per-passage minimum clearance 평균은
4.667251->4.485511 units이고 그 최소는1.371783->0.662154다. Clearance 여유는
줄었지만 의미 있는 경로 단축은 입증하지 못했다. 전체144-object inventory의
기존 hit2개와 signed minimum-0.013049는 양쪽 그대로이며, completed-passage
양의 minimum만 보고 충돌이 없다고 표시하지 않는다.

채택 기준인21-cell equal-weight fractional path 평균은-0.138079%로 사전2%에
못 미치고 qualifying geometry seed도0이다. Window 평균과 cell 평균은 다른
집계이므로 혼용하지 않는다. 공통 완주21-lap 평균은+2.857143ms로 감소하지 않았다.
단축/두 geometry/lap 세 improvement gate 모두 실패한다. Steering 변경22회는
실제로 발생했지만 target의 기하학적 최소성이 최소 실제 경로나 빠른 복귀를
입증하지 않는다. 최대 deviation도 커졌으며 전체 contact-loss proxy 감소만으로
채택하지 않는다. **v3도 비채택, crossing_projection 기준선 유지.**

Return 분모144는 양쪽 returned1/censored-window-exit118/unpassed6/invalid19다.
118 censors를 버린 전체 return mean은 정의하지 않는다. 최종335 tests+28 subtests,
독립 source review와 CPU21 package smoke를 통과했고 실제 CPU21 deterministic
rebuild도 같은 v3 ZIP SHA를 재현했다.
[최종 독립 primary audit](../../experiments/koi-minimum-clearance-ab-r3-audit.json)은
48 episode/raw/process,150 frozen copies,148 consumed pins와 모델 members를
검증하고 모든 gate와 분모를 독립 계산해 같은 결론을 얻었다. R2에서 손실했던
3/38301은221 recorded decisions 및881 raw records 전체가 baseline과 같아졌고
17.6s/0damage/0collision으로 완주했다. 전역 model은22/5704 calls,14 cells에서만
steering을 바꿨고 return_centerline call은0이다. 기존6px diagnostic 기준으로
projection-off 후 avoidance calls는423->424라서 조기복귀 감소도 입증되지 않았다.
Both-returned n1 pair는0.24->0.28s이며 cohort return mean으로 해석하지 않는다.
세 comparisons의144 episodes는 같은 consumed
TRAIN 재사용이며 신규·protected·공식·generalization 평가가 아니다. 속도 target
방향은 계속 종료 상태이고 Root Agent/다른 연구 lane/공식 model status는 바꾸지 않는다.

## 19. Steering 생성/유지/해제 가설과 Minimum-Clearance 종료

**Historical scope / first-reset 전 계획:** 아래 가설·분석·freeze 순서는 당시 계획이다.
분해와 두 full A/B는 이후 완료했으며 현재 판정과 다음 gate는22절을 따른다.

2026-10-01 사용자 지시로 minimum-clearance 계열 전체를 **실패 종료 / 비채택**으로
고정한다. 속도 target과 safety-margin 축소 방향 모두 더 추적하지 않는다. 15~18절의
모든 frozen code/ZIP/성공·실패 trace/protocol/result/audit는 보존하며 기준선은 계속
`ContactContinuityAgent('crossing_projection')`이다. 원본 missing ZIP 복원이나
root Agent/다른 lane/공식 model 교체를 주장하지 않는다.

새 독립 가설은 큰 회피 궤적이 clearance 자체보다 avoidance steering의 생성,
지속, 해제 과정에서 발생한다는 것이다. 새 reset 전에 기존 consumed TRAIN baseline
trace를 decision별 기본 road-follow / collision-projection 회피 / near·urgency
회피 / damping·far·impact·prefix·clipping 등 기타 항으로 가능한 한 분해한다.
Replacement override는 additive 항과 구분하고 clip 전후 및 unidentified residual을
명시한다. 실제 steering과 복원 합의 오차를 확인하며 미기록 정보를0으로 꾸미지 않는다.

동일 장애물의 physical full-fixture lateral separation과 rear-clear는 별개로
기록한다. 전방에 멀리 있어 positive 거리라는 것만으로 이미 비켜갔다고 하지 않는다.
유한한 valid collision projection의 active->off 시점, unavailable/dropout, 실제
avoidance 약화·반전, outward lateral 성장과 road correction의 관계를 시간순으로
분석한다. 대표 episode는 큰 deviation을 기준으로 선정하고 DNF·seam·미통과를
숨기지 않는다. 이 분석만으로 counterfactual steering의 안전·개선은 증명하지 않는다.

명백한 lifecycle mechanism이 확인될 때 별도 후보에 최소 수정하고 같은24 consumed
TRAIN conditions/48-slot A/B로 완주·조건별 damage/collision·새 hit 보존, max lateral,
avoidance 지속시간/적분, steer 적분, 회피 path, centerline sustained return와 lap을
검증한다. 기존 ±25 path window는 동일하게 사용하고 복귀는 별도로 다음 장애물의
entry 또는 고정 follow-up/episode end에서 censor를 보존해 비교한다. 정확한 정의와
source/runtime/gate는 첫 reset 전에 새 protocol에서 고정한다. 명백한 원인 수정은
새 ZIP/source/run으로 계속하되 개선이 없으면 이 가설을 종료한다. Margin과 속도
target을 재조정하는 우회는 없다. 당시 source/trace 분석 중이었고 실제 결과는 없었다.

## 20. Source-Exact Steering 분해와 해제 지연 관측

**완료된 baseline 진단과 historical v1 preflight:** 아래 관측은 보존하며 마지막
첫-run 계획 문단은 현재 진행 상태가 아니다. 실제 v1 실패는21절, 최종r2는22절이다.

[Baseline diagnosis](../../experiments/koi-steering-release-baseline-diagnosis-v1.json)는
24 baseline episodes/5703 decisions/22779 raw ticks를 전부 기록하고 issued와 sum
residual이 정확히0이다. [Decision JSONL](../../runs/koi-steering-release-diagnosis-20261001-v1/decisions.jsonl)는
raw road-position/lookahead, near urgency, collision replacement, correction과
각 clip/float32, ordered effective contributions 및 unidentified residual을 보존한다.
`.55`는 `.34`에 더해지는 것이 아니라 replacement다. Inherited/damping5640,
crossing48, repair3, impact12 paths를 구분한다. Impact expiry의 포화로 latent branch가
동일한 output이면 pre-act counter 없이는 ambiguous로 표시하고 replay를 금지한다.
기존 trace는 동일-cell 직전 postcounter와 source-proven constructor0을 사용한다.
19 saturation records에는 veto로 되돌아간 intermediate impact clip4개가 포함돼
최종 issued saturation 수와 같다고 하지 않는다. Source 리뷰와71 테스트를 통과했다.

144 objects 중42 same-object active-to-finite-off transitions가 있고,36 object에서
off epoch와 보수적 lateral separation, 지속 outward avoidance가 겹친다. 이 core
중첩은4.22s/1.410474 command-s다. 현재 foot이 lateral 방향으로 분리됐다는 것과
미래 조향 후 통과가 안전하다는 것은 별개이며 global positive distance만으로
separation을 판단하지 않는다. 실제 transverse fixture extents는 오래된 raw에
없으므로 yaw-enclosing full-footprint bound와 exact fixture distance를 함께 쓴다.
새 observer는 실제 transverse extent를 기록한다. [Events JSONL](../../runs/koi-steering-release-diagnosis-20261001-v1/events.jsonl)는
approach-window-safe / 실제 longitudinal-overlap-safe / full rear-clear를 분리하고
left-censored entry, 누락, DNF, reset seam와 모든 epoch를 보존한다.

대표3/38301 object4의 `raw.t`(warm-up 포함 simulation clock) 순서는 다음과 같다.

| Decision / t | Projection | Road + lookahead + correction | 적용 회피 항 | 최종 steer |
|---|---|---:|---:|---:|
| 152 / 13.10 | active, x47.7625 | -0.029000 | crossing replacement -0.290278 | -0.319278 |
| 153 / 13.18 | valid off, x56.512 | +0.106500 | -0.329697 | -0.223197 |
| 154 / 13.26 | valid off, x56.077519 | +0.024250 | -0.340000 | -0.315750 |
| 155 / 13.34 | unavailable, y57.8 범위 밖 | -0.005250 | -0.340000 | -0.345250 |
| 156 / 13.42 | dropout, same-object identity unavailable | -0.018500 | 0 global avoidance | -0.018500 |

Approach-window-safe 최초 bracket12.96~12.98은 eligibility 진입 전에 이미 분리된
left censor이지 새로 비켜간 transition이 아니다. Overlap-safe bracket은13.34~13.36,
rear-clear는13.60~13.62다. Projection은152->153에서 꺼졌으나 near 항이 복귀방향
road/correction을 압도한다. Detector dropout156/t13.42에서 global avoidance0을
관측하지만 same-object release라고 식별할 수 없고 그때 rear-4.166<radius로 아직
rear-clear도 아니다. Old2/50302 on-road contact stall과2/38302의 뒤늦은 무검출
offroad runaway는 별도 실패로 유지하며 모든 큰 deviation을 near 항 탓으로 돌리지 않는다.

별도 `SteeringReleaseAgent`는 원래 actor를 건드리지 않고 정확한 clipping replay로
near 항만 alpha0/.25/.5/.75 중 가장 크게 해제 가능한 값으로 바꾼다. Same-component
finite opposite-flank projection-off에서만 lifecycle을 arm하고, 이후 unavailable은
새 clear observation이 아니다. Current bbox/full body envelope 또는 원래6px band에
재진입하면 즉시 rearm하며 unknown bounds, identity/flank 변화와 reacquisition도
이전 proof를 버린다. Original `.54`/8-connected component를 정확히 재사용하고
public skin-inclusive full rectangle의 proposed constant-command rear passage와
proposed/previous.08s hold를 검사한다. Previous-command hold는 actuator lag proxy,
component continuity/bbox-circle enclosure도 가정이어서 physics certificate가 아니다.
속도 target, margins, 원래6px band, crossing-active/impact/first10 controls는 그대로다.
Minimum lateral target/grid나 전체 steering scaling은 구현하지 않는다.

521 tests+28subtests와 actual CPU21 package smoke를 통과했다. Package3566a5c2...
12 members 중9 baseline dependencies는 고정 source와 같다. 최초 v1 publisher
manifest contract 오류는 [별도 zero-reset receipt](../../experiments/koi-steering-release-preflight-package-interface-v1.json)에
보존했고 code/ZIP bytes를 바꾸지 않은 v1a manifest로 교정했다. 첫 실제48-slot
[protocol](../../runs/koi-steering-release-ab-20261001-v1/protocol.json)은 동일24 consumed
conditions만 실행하도록 당시 고정했다. 이 첫 비교는 이후 완료되어21절의 safety
실패로 비채택됐다. 당시 gate는 Safety/119 windows/actual10 prefix를 보존한
뒤5% cell-weighted max-lateral과5% duration 또는 avoidance-integral 개선을 각각
두 geometry에서 요구하고 path/실제steer integral/keptlap/common-followup censors는
나빠지지 않아야 한다. 복귀 censor onset bound는max(0,H-.24)이며 confirmation과
onset/pending run을 구분한다. Returned-only 평균이나 bounds를 KM/RMST로 표시하지 않는다.

## 21. 첫 Lifecycle A/B의 Generation 실패와 직접 수정

**Preserved v1 negative proof / historical r2 prefreeze:** 첫 실패는 그대로 유효하다.
아래 수정의 사전 검증을 최종 driving 증명으로 취급하지 않으며 실제 재평가는22절이다.

첫48 episodes와 [result](../../experiments/koi-steering-release-ab-v1-result.json),
[independent audit](../../experiments/koi-steering-release-ab-v1-audit.json)는 고정한다.
Baseline21/C23 완주지만 kept20/lost1/gained3이며 baseline에서 clean이던
1:38301 object1과2:50301 object4에 새 contact가 생겼다. Damage 합1.4->0.4,
collision-positive decisions7->2로 총량은 감소했어도 조건별 safety와 baseline-no-hit
객체 보존은 실패한다.119 baseline windows 중 lost2:50301 object4/5를 빼고 남은
117의 cell-weighted max-lateral-11.203357%/avoidance-integral-17.883506% 감소는
survivor 비교일 뿐 채택 근거로 쓰지 않는다. Duration 감소-3.841825%는5%에 못 미친다.
20 kept laps의-25ms는 손실된 baseline17.7s lap을 뺀 조건부 값이다. **v1 비채택.**

[첫 hit 진단](../../experiments/koi-steering-release-v1-new-hit-diagnosis.json)은
1:38301 first37의 release가 object0에만 적용되고 그 object가 안전 통과한 뒤,
이후 object1의 inherited commands에서 contact가 발생함을 확인한다. Step75 offset
-4.5/side+에서76 offset-0.675/motion+3.825로 같은 object의 선택 flank가-로 바뀌고,
79에+로 복구되지만 늦었다. Baseline은 analogous motion+1.182479로 flip하지 않는다.
Opposite issued command 후에도 body yaw가 반대 방향으로 움직인 관측은 있지만,
wheel angles/velocity vector/angular velocity가 없어서 lag/slip/inertia를 분리하지 않는다.

[손실 조건 진단](../../experiments/koi-steering-release-v1-lost-finish-diagnosis.json)은
2:50301 last release148/object3가 끝난12.86s 뒤14.62s에 다른 object4와 contact한다.
169 current offset-0.667105, previous-3.238887, motion+2.571782가 frozen motion-only
branch에서+->-를 유발한다. Current offset은 여전히 기존+pass 쪽을 지지하는데
crossing 항까지-.55로 반전되어 steer-0.3285, 다음 near-.34/steer-0.16이 이어진다.
차량은 object4에 걸려 끝내 off_track으로 retire했다. 초기166:-에서167:+로 바뀌는
unambiguous choice까지 막는 blanket side lock은 이 근거로 정당화되지 않는다.

별도v2 `stabilize_ambiguous_flank=True`는 original actor 호출 전에 원래 동일한
pixel decoder로 같은 component를 확인하고, 기존 y<52/abs(offset)<=1/motion>2
branch가 **현재 offset sign이 기존 pass 쪽을 여전히 지지하는데도** 반전하려는
경우 그 derivative만 한 call 무시한다. Actor-instance의 previous-offset cache를
None으로 두고, 원래 actor가 true current offset을 다시 저장하게 한다. 센서 값을
꾸미거나 전역 flank를 고정하지 않고 real 이전/현재 offset·motion·suppression을 기록한다.
Original abs(offset)>1 및 sign-crossing choices와 .34/.55/6px/targets/margins는 유지한다.

v2 ZIPb1911d7d.../source52f54099...를 별도
[r2 protocol](../../runs/koi-steering-release-ab-20261001-r2/protocol.json)에 고정하고
동일48-slot 전체를 재평가하도록 당시 고정했다. 이 재평가는 이후 완료됐다.
Default `stabilize_ambiguous_flank=False`는 oldv1 동작을 보존한다.
Source review/positive-negative mirror/sign-cross/
identity/reset tests와525 tests+28subtests, CPU21 real-source generation smoke를 통과했다.
이것은 원인이 관측된 source branch 수정이지 미래 안전/개선의 사전 증명은 아니다.

## 22. Final r2 Internal Adoption Candidate

**INTERNAL ADOPTION CANDIDATE:** [final r2 result](../../experiments/koi-steering-release-ab-r2-result.json)
와 [frozen r2 protocol](../../runs/koi-steering-release-ab-20261001-r2/protocol.json)은
동일24 consumed TRAIN layouts의48 slots를 모두 완료했고17 gates가 모두 PASS다.
Baseline은 계속 source-reconstructed `crossing_projection`으로 FIXED이며 root Agent와
공식 model은 바꾸지 않는다. Missing original crossing ZIP 복원도 주장하지 않는다.
첫v1 실패/ZIP/source/result/audit와 두 failure diagnoses는 그대로 보존한다.

Candidate ZIP `b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce`,
runtime source `52f540993eecc4453d36900b726f8fb54bda189e58b0c3e5eb6126ac3fcdd54b`,
term helper `cb1149b03e489c81babda7171349483961b82d367748e717a21b3a5dfe82bbe7`를
고정한다. [v2 manifest](../../submissions/koi-steering-release-v2.manifest.json)의
UNEVALUATED 표시는 frozen zero-reset preflight 시점 기록이다. 현재 verdict는 최종
result가 정한다. FlagTrue의 최소 generation 수정만 oldv1과 다르며 defaultFalse는
oldv1을 보존한다. 12 members 중9 baseline dependencies는 byte-identical이고,
original `.34`/`.55` replacement/6px band/skin/raster margins/targets는 그대로다.
기록된 최종 preflight525 tests+28subtests와 CPU21 real-source smoke는 통과했다.
후보는 pixel/HUD/history만 사용하며 world geometry는 passive observer 측정용이다.
Candidate의 `baseline_steer`/`baseline_steering_terms`는 local generation 수정 후,
near release 전 bookkeeping이다. 수정 없는 frozen shadow counterfactual로 부르지
않는다. 독립적으로 실행한 full B arm이 고정 baseline comparator다.

### Final Paired Result

| Metric / denominator | Baseline B | Candidate C | Interpretation |
|---|---:|---:|---|
| Finishes /24 cells | 21 | 24 | kept21/lost0/gained3 |
| Total damage /24 cells | 1.4 | 0 | every cell nonincreasing |
| Collision-positive decisions /24 episodes | 7 | 0 | every cell nonincreasing |
| Whole-episode hit objects /144 | 2 | 0 | no new baseline-no-hit object hit |
| Baseline fixed windows retained /119 | 119 | 119 | no survivor selection;21 cells |
| Mean window max absolute lateral /119, m | 6.207286383 | 5.366848501 | arithmetic window means |
| Maximum across matched119 windows, m | 9.334206342 | 8.360373696 | not whole-episode maximum |
| Mean actual steering integral /119, command-s | 0.148388997 | 0.139754300 | executed hold intervals |
| Mean avoidance duration /119, s | 0.468570717 | 0.448241029 | not the equal-cell relative mean |
| Mean avoidance integral /119, command-s | 0.129380880 | 0.105767507 | exact afterclip component |
| Mean path /119, m | 50.483046064 | 50.358155611 | fixed station-25 to station+25 |
| Matched lap mean /21 kept cells, s | 19.054285714 | 19.016190476 | delta-38.095238ms; includes38300 |

Gained layouts are1:38302,2:38302,2:50302. Whole-episode max lateral is separately
263.990841186->9.550317192m; B includes a DNF runaway, so this is NOT the matched-window
max reduction. Candidate own24 finished-lap mean is not a matched21 comparison.

Primary reductions average each object's relative change within its cell, then
equally weight21 eligible cells; they are not ratios of the arithmetic means above.
38300 reset-seam cells remain in safety/finish/return/lap inventories but are excluded
from these fixed-window means. There are seven eligible geometry seed IDs, not21
independent road geometries.

| Primary equal-cell relative metric /21 cells | Change | Gate reading |
|---|---:|---|
| Max lateral | -12.544445530% | >=5% and seven qualifying geometry seeds |
| Avoidance duration | -3.312396214% | does NOT meet5%, despite two qualifying seeds |
| Avoidance integral | -17.805736658% | >=5% and seven qualifying seeds; OR branch passes |
| Fixed-window path | -0.169630703% | nonincrease, not a2% shorter-route claim |
| Actual steering integral | -5.367766574% | nonincrease |

Lateral/integral qualifying IDs are38301,38302,38303,50300,50301,50302,50303.
Five percent is a predeclared internal meaningful threshold, not significance or
fresh generalization. The actual full r2, rather than a source-only prediction,
establishes the retained consumed-TRAIN gain after the two later-generation failures.

### Return And Temporal Limits

Prospective return begins at measured full-fixture rear-clear and requires
abs lateral<=1m sustained>=.24s. Cutoff is min(clear+2s,next obstacle station-25
entry,episode end); no future confirmation is peeked past cutoff.

| Full prospective status /144 objects per arm | B | C |
|---|---:|---:|
| Confirmed returned | 113 | 125 |
| Next-obstacle-entry censored | 23 | 17 |
| Fixed-time censored | 0 | 1 |
| Unpassed | 6 | 0 |
| Invalid followup / next-entry continuity | 2 | 1 |

All136 baseline-comparable objects retain common followup, truncated per object to
the shorter arm horizon. On that SAME136 cohort B has113 returned/23 censors
(14 next-entry,9 common-followup); C has117 returned/19 censors (9 next-entry,
9 common-followup,1 fixed-time). Full-arm125 returns must not replace common117.
Common response means0.726180453->0.666090505s mix observed confirmed **onset**
delays and conservative censored onset bounds `max(0,H-.24)`. Confirmation delay
and full confirmation horizon H remain separate; pending runs are not confirmed.
This is a descriptive mixture, NOT an unbiased144-object mean, returned-only
population estimate, Kaplan-Meier or properly estimated RMST. The legacy ±25-window
return statuses are a different endpoint and do not replace prospective followup.
The109 pairs confirmed returned in BOTH common followups have actual onset-delay
means0.724220183->0.646972477s, conditional on both returning; this is not the
136-object or144-object population mean and does not discard the23/19 censors.

The20절3:38301/object4 timeline remains diagnostic evidence: projection turns finite
off at153/t13.18 while near urgency persists through154 and unavailable155/t13.34;
dropout156/t13.42 produces global avoidance0 before rear-clear13.60~13.62. It is not
an identified same-object safe release. Old exact transverse vertices were not
archived;12.96~12.98 approach-entry left censor and13.34~13.36 overlap-safe brackets
use a conservative yaw-enclosing footprint plus observed exact fixture distance,
not invented measured transverse extents. The new observer records skin-inclusive
actual fixture transverse intervals. Its separately measured same-object off-epoch
avoidance is101->77 decisions/8.08->6.16s/2.724028122->1.868508104 command-s;
subsequent unavailable-projection avoidance is169->123 decisions/13.52->9.84s/
4.552000014->3.218360002 command-s. Finite-off-and-laterally-separated residual
decisions238->260 span19.04->20.80s but integral5.938313794->4.915003977 command-s;
these visited-state counts are not the fixed119-window primary metric or causal
proof that every residual command is excess. Pixel identity/bbox enclosure,
instantaneous-command arcs, previous-command lag proxy and unlogged wheel angles/
velocity vectors still limit a continuous physical or unique lag/slip explanation.

The [actual r2 candidate episode](../../runs/koi-steering-release-ab-20261001-r2/3-38301-steering_release_v1.json)
shows object4 at154/pre13.26 with finite projection54.477807 and measured transverse
gap4.924359m: guarded alpha.25 changes near term-.34->-.085 and final steer
-.302111119->-.047111113. At155/pre13.34 the impact guard restores unchanged
near-.34, so this is not permanent suppression. Both arms pass object4 clean;
rear-clear13.62->13.66 is later, while that matched window max lateral
9.334206342->7.082517398m improves but path35.688296554->39.047798600m and actual
steer integral0.135506136->0.173876519 command-s increase. Aggregate gate success
does not mean every object window or every temporal endpoint improves.

### Verdict And Next Gate

The steering hypothesis is not closed under the user's no-improvement rule: r2
shows meaningful lateral/integral gains,24 finishes and all preservation gates.
This is an internal adoption candidate on repeatedly consumed TRAIN evidence only,
not automatic official adoption, a fresh/generalized model or permission to reopen
speed-target/margin/minimum-clearance directions. All minimum-clearance variants
remain FAILED / NOT ADOPTED by explicit user direction. Crossing_projection remains
FIXED; root Agent and other lanes' designations are unchanged.

[Final independent primary audit](../../experiments/koi-steering-release-ab-r2-audit.json)
SHA-256 `b09872a6efa33596745fa12e6d54ce880e603731ab994a90ca6d793c7afad199`
rehashes48 episode/raw/bound-original-process sets,158 frozen copies/1046 evidence
pins,10/12 model members and96 ledger rows. Actual and post-generation/pre-release
22900 formula rows have exactly0 residual/unidentified rows; all5747 candidate
pedal checks are exact in that post-generation context. All17 gates independently
PASS with zero derived mismatches; baseline24 repeats5703/22779 old recorded core.
Five generator suppressions and true next-call offset caches are verified, not
fabricated observations. Candidate all144 objects have no touch/collision and
minimum observed clearance0.614938m; wheel-contact-loss701 ticks remain, so no
claim of universally zero offroad. Actual CPU21 rebuild reproduces the same ZIP.
Only separately user-authorized future confirmation can open the
next gate, after protocol/source/provenance/exposure checks and applicable official
source rechecks. No new evaluation is queued automatically, no consumed cell is
relabeled fresh, and no fresh-cell availability is promised.

## 23. Frozen v2 TRAIN Generalization Gate

2026-10-01 user direction freezes the currentv2 ZIP/source/parameters and both
crossing_projection/root Agent unchanged. No further interaction is allowed on
the old24 consumed TRAIN conditions. The next authorized action is a separately
audited, nonprotected TRAIN/dev paired generalization study, never protected,
blind or official evaluation. Candidate corrections from its outcome are forbidden.

Proposed24 uint32 road seeds3184000001-3184000024 each have tracks1/2/3 obstacle
layouts:72 matched cells/144 episodes/432 prospective objects per arm. Road-only
catalog digests distinguish24 geometries from72 layouts; initial states, pixels,
full layouts and actual first10 decisions must match between arms. The audit must
check cross-lane prior interaction, allocations, active claims, exposure and retired
protected exclusions under the shared claim lock, including partial evidence.

The new analyzer derives every baseline-eligible fixed window and baseline finish
from this cohort, rather than hardcoding historical119 windows or21 kept laps.
It retains the same passive physical/steering/return measurement, safety and5%
cell-weighted lateral/avoidance gates. Predeclared generalization consistency also
requires5% equal-geometry lateral and avoidance duration OR integral reduction,
plus joint lateral/integral decrease on at least half eligible geometries.
Coverage must include at least18 of24 geometry seeds and12 seeds per track;
road-only signatures must also differ from the eight historicalr2 roads. All
baseline windows/finishes/comparable returns must remain covered, with no worse
per-cell damage/collision, new object hit, matched path/steer integral/kept-lap mean
or common return censor count. Empty or lost denominators cannot pass. Report
per-track/per-geometry coverage and chronological failure traces without choosing
or replacing roads based on performance. No causal mediation, KM/RMST, protected
confirmation or official ranking claim follows from these internal proxies.

The [protocol](../../experiments/koi-steering-generalization-v1.json) is frozen at
SHA`ccc6720f676814eb705e853792eceb31c6c5fb2b2aa695b1144641c2befcd7dd`;
actual CPU21 source/runtime/claims/resource preflight passed with zero resets.
Independent review found that the frozen raw loader does not check the producer's
failure-time partial artifact pins. Preserve the protocol/operator/analyzer
byte-identical and use the separate
[evidence guard](../../experiments/koi-steering-generalization-v1-evidence-guard.json)
and `scripts.finalize_koi_steering_generalization` as the only authoritative
finalization route. This verifies exact partial inventory/hash/bytes before the
frozen loader runs; its source and guard hashes must be bound in independent
review and the result. Direct raw-loader analysis retains that documented risk.
No model, metric, threshold, cohort or frozen source is changed by this stronger
forensic check; no reset occurred during discovery or correction.

**Final status: frozenv2 is not a generalized adoption candidate.** Original144-slot
execution is operationally incomplete; rejection follows independent, valid matched
safety counterexamples, not the timeout itself. Neitherv2, baseline nor root Agent
was modified, and no protected/blind/official evaluation was used.

### Execution and Matched Evidence

- [Guarded result](../../experiments/koi-steering-generalization-v1-result.json)
  SHA`2572e09dfc8dd4a1399a1168ea15692b6be946cbb20a72b18d0532ea00d5843d`:
  fixed4584.848s budget ends95 valid saved episodes, one bootstrap timeout and48
  unrun slots. Final3/3184000016/v2 timeout2.123655s has process/log pins but no
  episode/raw/decision files;96 reset intents and95 completions remain preserved.
  It is not a policy loss, source-proven fresh cell or successful full-matrix run.
- The [reviewed boundary manager](../../experiments/koi-steering-generalization-pause-review.json)
  refused before lock/signal because the parent had already finalized. No boundary
  receipt/continuation/composite protocol or extra rollout was created. Preparatory
  operator/tests remain unexecuted, not a second candidate or an approved retry.
- [Independent postrun audit](../../experiments/koi-steering-generalization-v1-postrun-audit.json)
  SHA`c4bd5b112cc9ea414a0e1dd0b22626324257e0607e863d0cbe4c323132875b78`
  exactly reproduces the frozen summary and rehashes sources/models/copies/raw/
  streams/process/ledger. All47 complete pairs have matching full geometry/catalog,
  initial state/pixels and first10 actual actions/states. Matched census is16 roads,
  tracks16/16/15 cells,282 prospective objects EACH, not48B versus47C episodes.
- B41/47 versusC40/47 finishes: kept36/lost5/gained4/neither2. Five valid natural
  losses (four `off_track`, one `crash`) span four road seeds. Two baseline-clean
  new hits also violate per-cell damage/collision and no-new-hit preservation.
  More unexecuted cells cannot remove those prospectively fixed counterexamples;
  no additional evaluation or partial retry is needed to decide non-adoption.
- Outcome-audit publication is deliberately not hidden: the frozen broad metadata
  scanner now returns `HOLD` for its explicit consumed outcome identities. Before
  publication the authenticated self-audit was clear; no new exposure/source change
  caused this conservative classification. Do not add an exception or recycle cells
  merely to run again. This does not invalidate already source-bound failure evidence.

| Matched Metric / Denominator | Crossing | Frozen v2 | Interpretation |
|---|---:|---:|---|
| Finishes /47 cells | 41 | 40 | Five baseline finishes lost, four gained; gains do not offset strict loss gate |
| Damage total /47 cells | 4.4 | 3.4 | Better total hides two worsening cells |
| Collision-positive decisions /47 cells | 22 | 17 | Repeated-contact decision metric, not unique incident count |
| Whole-episode hit objects /282 objects | 5 | 4 | Two NEW baseline-clean hits still reject adoption |
| Whole-episode max centerline lateral proxy (m) | 29.9527 | 260.0025 | Severe later lane/heading failure, not captured by retained-window means |
| Matched max lateral mean /173 retained windows (m) | 6.17242 | 5.50618 | Conditional only;17 of190 baseline windows missing |
| Avoidance integral mean /173 windows (abs-steer*s) | 0.127446 | 0.106307 | Equal-cell-15.9312%, equal-geometry-15.8473% |
| Avoidance duration mean /173 windows (s) | 0.468858 | 0.449742 | Equal-cell-3.5129%, below5% duration threshold |
| Avoidance-window path mean /173 windows (m) | 49.95762 | 49.86369 | Equal-cell relative+0.0378%, mixed weighting/coverage; not full-path improvement |
| Common return censors /235 followups | 37 | 43 |20 of255 baseline followups lost; no returned-only/KM/RMST claim |
| Observed returned statuses /282 objects | 216 | 206 | Full matched census retained, including all nonreturn/invalid/unpassed statuses |
| Kept-lap mean /36 cells (s) | 19.091111 | 19.037222 | -53.889ms survivor-only timing omits five baseline finishes |

Conditional equal-cell lateral-10.4390% and integral-15.9312% remain coherent
on all12 eligible measured geometries, including all four failure clusters. The
same reduction is also present in retained windows of each lost episode. All17
missing baseline windows occur in these five failures; four have longer retained
path and three have larger actual-steer integral. Thus the old24
`less excessive avoidance -> less lateral -> better collision/completion` chain
**does not transfer as a sufficient whole-episode improvement**. No population
effect across all24 planned geometries or official/private-track claim is made.

### Failure Chronology

[Matched diagnosis](../../experiments/koi-steering-generalization-v1-diagnosis-result.json)
SHA`72c3bb0f2f24ccd6405f479e87bd474889d5ea8224b41dbf74aeb5ca1d551245`
and [independent five-failure audit](../../experiments/koi-steering-generalization-v1-failure-audit-result.json)
SHA`55897acfaf394f2ff932b44517b4bb7bc2b9f5c9ef8c504a6c1e339cdab49ed8`
preserve exact raw/episode/stream/process and mechanism source pins.

| Road Seed / Track | First Divergence | Later Observed Failure |
|---|---|---|
| 3184000004 /3 | near_release step43, t4.38 | Last release6.22 on object1; different unreleased object2 contact10.86; low-motion negative-reward retirement labelled off_track18.94 |
| 3184000005 /1 | near_release step41, t4.22 | Last release13.58 on object3; different unreleased object4 contact17.02; crash17.58 |
| 3184000013 /1 | near_release step70, t6.54 | Object3 rear-clear14.84; heading>1rad14.90; empty road features15.26 and all wheels off-road15.32; terminal lateral260.0025 |
| 3184000013 /3 | near_release step93, t8.38 | Only release; object0 rear-clear8.80; empty road features10.38 and all wheels off-road10.40; terminal lateral182.2713 |
| 3184000014 /2 | near_release step33, t3.58 | Object1 rear-clear7.52; all wheels off-road7.60, heading>1rad7.68; spin/lost progress; terminal wheel contacts restored but negative-reward retirement17.18 |

**Verified:** all prior actions/physical states/raw prefixes and the divergent PRE
state are exactly equal. At that first state, candidate shadow steering equals the
actual frozen baseline steering; pedals stay equal, release changes steering. No
generation suppression occurs in any of these five failures. This distinguishes
the first divergence from the old ambiguity-generation correction without making
post-divergence shadow actions counterfactual baselines.

All17 changed release decisions have positive sampled all-object clearance during
their issued holds (minimum4.05605m). Selected released objects then reach physical
rear-clear without observed hit (minimum selected-passage clearance2.78041m).
The two contacts are later objects never released; the other three losses have no
obstacle hit. Do **not** claim that the release-held action immediately collided,
that generation suppression caused these losses, or that negative transverse
interval overlap before longitudinal passage is itself a collision.

**Source boundary:** `release_safe` is a selected pixel-component/constant-command
lag proxy, not dynamics certification. It has no supported-road/curvature/heading
recovery or next-object safety condition. `off_track` is a negative-reward streak
or simulator-termination fallback in the frozen wrappers, not a wheel-contact
predicate. Raw reward/official score is not reconstructed.

**Untested hypothesis, separate follow-up only:** release changes subsequent
heading/arrival states and inherited nonlinear perception/control feedback; locally
clear object passage need not preserve lane-return viability or later flank choice.
**Current user direction:** ordinary obstacles still cause visually excessive
lateral avoidance in v2, sometimes enough to leave the road; it is less than before
but remains excessive. Margin reduction and steering-release have failed to resolve
this issue and are on hold, including the former release road-recovery H1.
Reconsider only in future at the **avoidance trajectory / side-selection** stage.
All further implementation, tuning, verification and evaluation are stopped;
only this documentation note is requested. Exact mediator/correction is unproved.

## 24. Separate Crossing Collision/Heading-Recovery Test

2026-10-01 user closure: this long-lived recovery-state-machine direction is
FAILED / NOT ADOPTED and will not be extended. Preserve its code/results as
reference only. The separately authorized next hypothesis is a minimum-intervention
collision shield directly on unchanged crossing_projection, not recovery v2.
Only predicted short-horizon footprint collision triggers replacement; collision-
free steering choices minimize baseline deviation with a soft road-departure cost.
Immediate handback and a six-action encounter budget prohibit persistent control.
Rearm requires three observed threat-associated clear baseline decisions;
unresolved threat disappearance disables rearming until reset without retaining
control. No heading recovery
or pedal change is part of this experiment. Separate matched A/B will reuse only
the six consumed TRAIN cases below, including corner conflicts and ordinary
obstacle controls. No Track4 geometry, fresh/protected cells or official action.
Preservation of all baseline finishes, per-cell safety, no new clean-object hits,
bounded intervention and measurable safety/completion benefit are required;
ordinary-obstacle lateral/path metrics must not be hidden by survivor selection.
The safe-baseline no-op contract does not itself reduce already collision-free
overavoidance. No clear measured improvement means nonadoption.

This section records the separate 2026-10-01 user-directed implementation and
consumed-TRAIN test, not a modification of the frozen v2 study or its near_release.
The user restores crossing_projection as comparator. Original crossing/v2 runtime,
both v2 ZIPs and root Agent remain unchanged; no official action or Track4 geometry.

The [archived diagnosis](../../experiments/koi-collision-priority-diagnosis-v1.json)
finds source-exact cancellation before later collisions: v2 3/3184000004 d118
road+.4485 versus avoid-.448148 produces+.000352, contact0.48s later; v2
1/3184000005 d197 road-.477971 versus avoid+.34 still steers-.137971, contact
0.32s later. Clean corner controls also show cancellation and nominal footprint
overlap, so those predicates alone do not establish a needed intervention.

The [post-passage diagnosis](../../experiments/koi-collision-recovery-diagnosis-v1.json)
shows crossing2/3184000006 rear-clears object1 at7.20s with heading-.147rad and
lateral6.069m;1s later heading1.163rad/lateral-20.627m. It regains road contact
after1.14s fully off-road but eventually travels nearly backward on asphalt
(terminal heading-2.966rad). Crossing1/3184000002 instead stalls after collision
ON road. These must not be conflated through the evaluator's off_track label.

The initial v2-based priority-only candidate is rejected and preserved at
`runs/koi-collision-priority-v1/`: four versus three finishes in six pairs.
Its stale nominal-road-danger latch could retain avoidance despite a clear actual
base command. The new standalone `CollisionRecoveryAgent` does not import v2 or
near_release. It directly wraps crossing, changes only armed corner/obstacle or
post-passage states, permits emergency trajectories without a hard asphalt veto,
then aligns the near observed road tangent before targeting the nearest safe road
edge. Recovery uses the inherited38/44 speed levels with braking, retains an
obstacle re-entry guard and waits for three supported/aligned decisions before
handback. Missing road invokes a bounded-heading search with braking. Pixel tangent,
support and short kinematic sweep are estimates, not physical safety certification.

Official [Participants README](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/README.md#L227-L240)
and [wrapper](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/env_wrapper.py#L56-L100)
confirm retirement iff the negative summed-raw-reward decision counter exceeds100:
the101st consecutive negative decision, nominal8.08s at4/50s. Physical return to
asphalt does not reset it without a nonnegative decision reward. The website was
checked but had no numerical predicate. Agent receives pixels only; its25-decision
avoidance-phase cap reserves most of the verified100-decision limit for recovery
but cannot know pre-existing streak or guarantee survival. Passive telemetry logs
the exact counter after the full action block, separate from50Hz wheel contacts.

### Measured Outcome

The [frozen protocol](../../runs/koi-collision-recovery-v1/protocol.json) and
[result](../../experiments/koi-collision-recovery-v1-result.json) cover12 valid
episodes on six outcome-selected consumed TRAIN cells, all matched geometry,
initial pixels/state and action prefixes. Three rescue cells are1/3184000002,
2/3184000006 and2/3184000001; three preservation controls are3/3184000002,
1/3184000013 and1/3184000015. They are five road seeds, not six independent roads.

| Metric | Crossing | Recovery Candidate |
|---|---:|---:|
| Finished | 3/6 | 2/6 |
| Total damage | 1.4 | 1.0 |
| Collision-positive decisions | 7 | 5 |
| Physically contacted objects | 2 | 1 |
| Maximum exact negative-reward streak | 101 | 101 |
| Longest all-wheels-offroad duration | 1.14s | 7.96s |
| Minimum all-object fixture clearance | -0.012577m | -0.004932m |
| Departures without any wheel-contact return | 0/3 | 3/11 |
| Sole mutually finished lap | 19.96s | 23.84s |

Candidate gains2/3184000001 but loses3/3184000002 and1/3184000015; the latter hits
a baseline-clean object4. Its gained finish itself reaches reward streak97 and
2.08s continuous full departure. Recovery changes816 decisions across the six
episodes, so the tested intervention is not sufficiently narrow in practice.
First wheel contact is reacquisition, not stable all-wheel/heading recovery;
the run additionally records all-four-contact timing and censored post-passage
returns. Conditional recovered-event means are not used to claim improvement.

**NOT ADOPTED.** Lower aggregate damage/contact count does not outweigh lost
finishes, new collision, prolonged departure and slower retained lap. No positive
result exists to justify the requested unchanged-policy repeat. Keep crossing
as baseline and the candidate as negative development evidence only. The general
heading-safe recovery hypothesis is not disproved by this unsuccessful heuristic.
Five runtime and seven evaluator tests pass; no broad audit or Git write performed.

## 25. Minimum-Intervention Collision Shield

The user-authorized independent hypothesis is implemented in
`haic/algorithms/koi/collision_shield.py`, not derived from collision_recovery.
Crossing_projection remains unchanged. Only a predicted collision of its actual
returned action can cause steering replacement; there is no separate heading,
road-search, pedal, or estimated reward-budget controller.

### Frozen Method

Full-image bright components supply all plausible above-HUD obstacle boxes,
including near/rear objects, independently of the baseline's selected obstacle or
road-following/avoidance opposition. A skin-inclusive hull/wheel rectangle uses
four-axis SAT along a nominal rate-limited bicycle projection, positive-right
steering, anisotropic raster scale and sub-half-pixel corner sweep spacing.
The .32s horizon applies each candidate for the actual .08s action, then assumes
the current baseline command for the remaining .24s. Candidate commands are the
baseline and33 values in[-.4,.4] at.025 spacing. Only collision-free choices are
eligible; cost is absolute action deviation plus .01 times observed mean road
departure in metres. This is a finite-grid weighted minimum, not a continuous
optimality or physical collision-free guarantee. Road departure is never a hard
veto. Both pedals and baseline internal state remain unmodified.

Safe/unknown/infeasible projections return the baseline immediately. A threat
encounter permits at most6 changed actions;3 genuinely observed threat-associated
clear decisions rearm. Detection memory lasts at most2 misses, uses translated/
rotated bounds and one-to-one, size-consistent matching. Unresolved threatening
dropout disables rearming until episode reset; this latch never retains driving
control. Three source-review issues (far-object omission, unrelated-object rearm,
and shrunken-fragment replacement) were fixed and tested BEFORE freezing or resets.
Unobserved slip, damage-dependent actuation, uncertain detection, raster geometry,
HUD magnitude/saturation and assumed future baseline actions remain limitations.

The [protocol](../../runs/koi-collision-shield-v1/protocol.json) SHA
`afeeb356ff1fae1e74b6606ab81073c1e10ebebefddf90ae945596e2c922ea8b`
pins the source`ad772bde...`, baseline ZIP`a4b35c56...`, runtime, environment,
observer, analyzer, preserved root/v2/recovery files, six consumed cells and gates.
Separate evaluate/analyze scripts leave all old frozen operators unchanged.
The first four selected layouts are prior challenge roles; the final two are
ordinary control roles, not assertions that their roads contain no corners.
Twelve serial CPU21 episodes completed, without operational errors or missing
mates. Every pair has identical initial geometry/pixels/full logged state and
exact raw/state/pixel prefix up to first changed action, including its PRE-state.
This is six outcome-selected layouts on five roads, not fresh generalization.

### Measured Outcome

[Primary result](../../experiments/koi-collision-shield-v1-result.json):

| Metric | Crossing | Shield |
|---|---:|---:|
| Finished | 3/6 | 5/6 |
| Kept / lost / gained | reference | 3 / 0 / 2 |
| Damage | 1.4 | 0 |
| Collision-positive decisions | 7 | 0 |
| Hit objects, all36 objects per arm | 2 | 0 |
| Minimum sampled physical fixture clearance | -0.012577m | 0.990591m |
| Longest full-offroad run | 1.14s | 1.14s |
| Total full-offroad sampled occupancy | 1.20s | 1.28s |
| Changed actions | 0 | 17/1639 candidate decisions |
| Intervention bursts / longest burst | 0 | 16 / 2 actions (0.16s) |
| Largest encounter intervention count | 0 | 5 (cap6) |
| Kept3 lap mean change | reference | +40ms |
| Kept3 path mean change | reference | +0.714544m |

The previous recovery's816 was the TOTAL changed decisions over six episodes,
not evidence for a single816-action burst. The new consecutive/encounter metrics
are separately measured, not inferred from its17 total interventions.
The administrative encounter can outlive an intervention: on2/3184000006 its five
changed decisions span22.24s first-to-last, but total override time is only.40s
and each burst is one.08s action. This is NOT22.24s of separate-controller use.
All six episodes end with conservative unresolved-threat rearming blocked;
general coverage/rearming across independent obstacles is not established by
this cohort. Neither the accounting latch nor blocked rearm extends an action.

| Cell | Finish B/C | Changed Actions | Max Lateral B/C (m) | Max Heading Error B/C (rad) | Matched Lap Delta |
|---|---|---:|---|---|---|
| 1/3184000002 | no / yes | 3 | 6.291832 / 7.011504 | .543258 / .557207 | not comparable |
| 2/3184000006 | no / no | 5 | 29.952721 / 29.952721 | 3.141575 / 3.141204 | not comparable |
| 2/3184000001 | no / yes | 3 | 8.660912 / 9.342283 | 2.194339 / 1.038794 | not comparable |
| 3/3184000002 | yes / yes | 2 | 8.107503 / 8.541311 | .563452 / .643817 | +180ms |
| 1/3184000013 | yes / yes | 0 | 9.297824 / 9.297824 | 1.079318 / 1.079318 | 0ms |
| 1/3184000015 | yes / yes | 4 | 6.691664 / 6.901651 | .806041 / .806041 | -60ms |

The two collision-associated failures become clean finishes. No old finish is lost,
no cell increases damage/collision, and no new clean-object hit occurs. The remaining
2/3184000006 no-hit failure is not rescued: its full-offroad run occurs within the
identical115-decision prefix, before first shield action116. This shield does not
solve inherited noncollision departure/backward travel, and the observation alone
does not authorize adding a heading-recovery controller.

Ordinary controls retain2/2 clean finishes, with one entire episode exactly equal
and four changed actions in the other. Their combined path2131.748912->2132.331110m
(+0.582198m) and lateral integral64.843274->65.955569m*s do NOT demonstrate less
overavoidance. Combined total absolute heading change43.234031->43.000137rad is
slightly lower, but road-relative heading-error integral4.554203->4.593250rad*s
is higher; neither is a blanket heading improvement. Minimum physical clearance
2.917713->1.853884m stays positive. Mean matched lap decreases30ms, which does not
erase the lateral/path counterevidence. Whole-six path/heading integrals are also
reported, but longer newly completed episodes make their totals unsuitable for
an efficiency claim.

**Frozen verdict: NOT ADOPTED.** Eight of nine gates pass. The sole failed gate is
mutually-finished efficiency:3/3184000002 increases17.68->17.86s, violating the
predeclared per-kept-lap+20ms limit. Its path+1.561435m remains within the separate
1% path limit. No limit is relaxed after seeing results. The lossless safety gain
is materially more promising than the rejected long recovery, but the joint goal
of collision prevention AND less ordinary overavoidance is not established.
Keep crossing as baseline and retain this candidate as frozen development evidence,
without tuning, repeat, new cells, promotion, or official action.

For the slower clean control, the actual first changed decision82 replaces
steer-.036 with+.025 (delta+.061), then decision83 replaces-.029222 with0. Its
baseline episode was contact-free with minimum fixture gap2.537091m. At decision82
the projection uses recent-maximum HUD speed75.070359m/s while observer PRE speed
is66.822621m/s and unchanged brake is.241623; at83 they are67.504036 versus
60.948659m/s with brake.28. The constant-speed/assumed-command continuation is
therefore a plausible source of overly early threat prediction while braking,
not an established unique cause of the180ms whole-lap loss. No projection,
speed target, pedal, threshold or gate was changed after this observation.

Validation:16 runtime tests; integrated new runtime/evaluator/analyzer plus old
recovery/operator regression51 tests and31 subtests PASS. Actual frozen crossing
versus wrapped driver has32 bit-exact synthetic no-op actions, and both frozen
arms pass isolated CPU21 import/construct preflight with zero environment resets.

The [independent primary audit](../../experiments/koi-collision-shield-v1-audit.json)
SHA`24bbe0608092c1034306e2ac55ed6470d3250c1637fbf8c9d29f5fe57c4bde3e`
verifies48 episode/raw/decision/process hashes and205 frozen inventory pins,
independently replays the pure finish tracker on archived geometry/positions
(no environment), and reproduces all9 gate outcomes and per-cell metrics. The
entire frozen analyzer output equals primary result SHA`25819a46...`. All3,201
decisions/12,790 raw ticks remain accounted for, including all four natural DNFs
(baseline3, candidate1). No operational censor, missing pair or artifact discrepancy
was found. No additional simulator interaction or policy/result change occurred.

## 26. Separate Nominal Trajectory Selection (2026-10-02)

The current user-designated submission baseline is the exact frozen
crossing_projection + collision-shield v1 ZIP `c9e376a0...`, not standalone
crossing. Preserve all11 bundle members and the shield source `ad772bde...`.
The user authorizes a separate consumed-TRAIN development candidate at nominal
trajectory generation/side selection, explicitly excluding another margin
reduction or steering-release variant. No root Agent, frozen policy, Track4
geometry, protected/fresh cells or official external action may change.

### Hypothesis And Gates

Inspect how nominal avoidance chooses lateral targets and flanks before adding
a candidate. The falsifiable hypothesis is that feasible flanks/trajectories are
not ranked by required lateral travel, path excess and steering variation, so
a geometry-aware choice can reduce ordinary-obstacle detours without changing
the existing clearance constants or the final safety layer. This is initially a
hypothesis, not a demonstrated causal mechanism.

Both contemporary A/B arms must use the same exact v1 shield. Freeze a separate
candidate, source/runtime/environment pins, completed consumed TRAIN cells and
metric/gate definitions before the first reset. Prefer the completed 318400000x
safety cases plus ordinary 3184000013/0015 controls; do not reopen the historical
38300/50300 cohort. Reuse is outcome-selected development, never fresh validation.

Reject on any lost baseline finish, per-cell increase in damage or collision
decisions, new hit object, lost baseline eligible obstacle window or return
coverage, or operationally invalid/missing pair. Report maximum centerline
lateral distance, fixed baseline avoidance-window path length, actual absolute
steering integral and total steering variation, prospective centerline return
with censoring, and mutually finished lap time. Do not use surviving windows or
newly completed laps to conceal safety/coverage regressions. Efficiency requires
measurable ordinary-control improvements across more than one road, with no
kept-lap worsening beyond one20ms physics tick; numerical noise is not a gain.

Only clear safe efficiency gains warrant another separately frozen minimal
iteration on these consumed cells. Preserve and discard regressing candidates
without editing the frozen reference or relaxing post-outcome gates.

### Source Diagnosis And First Candidate

Frozen `corridor_agent.py:168-190` selects the flank from obstacle centroid
minus perceived road center, optionally reversing it from offset motion, then
issues the same `.34` avoidance term once image y reaches40. The term has no
lateral target/clearance feedback. `contact_continuity_runtime.py:66-74` replaces
that term with `.55` on the inherited flank for its crossing trigger; it is not
an additional `.55`. No bilateral path cost or centerline rejoin is constructed.
The existing v1 shield intentionally leaves predicted-clear nominal commands
unchanged. These are source-backed observations; their quantitative contribution
to ordinary overavoidance remains a driving hypothesis.

Read-only synthetic frozen-source checks reproduce both the offset-independent
`.34` demand and a motion-only flank reversal while both centroid offsets remain
negative. A supplied shifted-road fixture also shows why minimum road-relative
lane offset is not necessarily minimum actual car movement/path. These checks
are not driven trajectories or competition results.

New `nominal_trajectory.py` generates bilateral smooth approach/pass/rejoin
references; it does not call the rejected minimum-clearance or release agents.
Only their pure component/road/footprint primitives are reused, with unchanged
physical envelope, skin, one-pixel object pad and half-pixel sampling allowances.
The new `trajectory_rollout.py` checks the complete planned nominal actuator
tracking, including countersteer/rejoin, before selection. Both references and
actual model rollouts must clear all visible objects and supported road; unknown
or infeasible geometry falls back exactly. Original crossing runs once and keeps
its pedals; v1 remains the final unchanged safety layer. Neither geometry nor
actuator projection is a physical safety guarantee.

Generated references are retained only as a nominal trajectory, transformed by
the final post-shield modeled ego motion and revalidated each decision. A complete
rejoin must fit the remaining12-action plan budget, finish forward passage and
match its terminal lateral position within1m/heading within0.2rad. The full first
0.08s command is independently road/object checked even when a short terminal
truncates the complete rollout. Geometry/impact guards do not zero a valid HUD
speed for executed-action actuator feedback. These are separate candidate
contracts, not modifications to any v1 shield rule or safety guarantee.

The first fixed cohort has8 layouts/5 roads: the six original shield cells plus
completed TRAIN2/3184000015 and3/3184000015. There are16 contemporary episodes,
48 objects per arm, and ordinary4 layouts/2 roads/24 objects per arm. Additional
0015 layouts have prior crossing/v2 evidence but no prior shield run; their new
control finish counts must be measured, not assumed.

Predeclared ordinary equal-cell efficiency thresholds are max lateral at least3%
lower, fixed-window path at least0.1% lower and actual steering integral at least3%
lower, with improvement direction on both ordinary road geometries. Steering
variation and common-horizon return bound must not increase; retained lap mean
must decrease by at least20ms and every kept cell may worsen by at most20ms.
No safety/coverage exception follows an aggregate efficiency gain.

### Completed A/B And Rejection

[Frozen protocol](../../runs/koi-nominal-trajectory-v1/protocol.json) SHA
`77bda836df249b08d0a0863a82ffbf7f0016123e76fa97046a2393515ad38078`
completed all16 contemporary episodes/eight matched pairs/five consumed roads.
Both arms retain exact submitted ZIP`c9e376a0...` and shield`ad772bde...`.
The [result](../../experiments/koi-nominal-trajectory-v1-result.json) SHA
`b020c3cbf45d212ac10d95ef17543d4ab45d5052c6a422c3e1b2627cd5c9c2b0`
is **REJECTED / NOT ADOPTED**. No source, metric or gate changed after freeze.

| Metric | Result |
|---|---:|
| Finishes / kept / lost / gained | 7/8 -> 7/8; kept7/lost0/gained0 |
| Damage / collision-positive decisions / hit objects | 0 -> 0 each |
| Safety objects | 48 per arm, including unpassed objects |
| Baseline fixed-window coverage | 38/38 retained |
| Common prospective-return coverage | 43/43 retained |
| Ordinary equal-cell fixed-window max lateral | +0.068713% |
| Ordinary equal-cell fixed-window path | -0.014782% |
| Ordinary equal-cell issued steering integral | +4.193204% |
| Ordinary equal-cell issued steering variation | +4.125609% |
| Seven mutually finished lap mean change | +20ms |
| Largest retained-lap increase | 3/3184000015: +140ms |

All eight pairs preserve geometry/initial full state/pixels and exact raw/decision
prefixes up to the first changed action. Six pairs are complete exact no-ops;
1/3184000002 first diverges at decision200 and3/3184000015 at133. Ordinary road0013
has no action change. Thus the small aggregate path decrease is not improvement
across two ordinary roads, and it does not compensate steering/return/lap regressions.
No missing windows or lost finishes are hidden by conditional efficiency means.

All finish/per-cell damage/collision/no-new-hit/shield contract/window+return
coverage and return-censor-count gates pass. Common-return bound, steering
variation, per-kept-lap ceiling, mean lap reduction and all three ordinary
reduction gates fail. The reference generator's predicted cost/complete-model
feasibility did not establish improved real closed-loop efficiency. Reject this
candidate without tuning a repeat, relaxing the gate or editing the baseline.

Integrated zero-reset validation:75 tests+30 subtests PASS, both isolated CPU21
imports/constructors pass, and focused independent pre-run review resolves
guarded-frame feedback-speed drift and unchecked terminal-hold remainder before
freeze. The baseline bundle's11 members and ZIP, shield and root Agent rehash
unchanged after execution. No fresh/protected/Track4 or official action occurred.

### Observed Planning/Execution Gap

The slower3/3184000015 primary decision diagnostics show two generated plans,
not an executed complete approach/pass/rejoin. At133 the selected+2px flank
changes nominal steer+.204979792 to-.005195152 with shield no-op; the following
decision returns to the original nominal under the geometry/impact eligibility
guard. At200 the selected-2px flank changes nominal-.130888894 to+.111954167.
At201 its one continuation+.158199161 is limited to+.125 by unchanged v1. At202
remaining-trajectory revalidation fails and the candidate returns immediately to
original crossing-.436500013, including the inherited-.55 crossing replacement.
The primary episode retains these proposals, reasons and independent action taps.

The first abort is not missing road geometry: at134 all seven road rows remain,
`geometry_repaired=False` and `contact_proxy=False`, but both
`impact_proxy_trigger=True` and `braking_proxy_veto=True`. The new candidate's
conservative eligibility guard aborts on the trigger without honoring the veto;
the frozen controller itself retains its original veto behavior. This explains
that immediate plan interruption, not the entire later lap regression. The exact
failed remaining-path predicate at202 is not logged, so attributing it uniquely
to the preceding shield intervention or perception/slip would be speculation.

Thus minimizing a full reference-tracking rollout cost did not minimize the cost
of the actually executed mixture of short nominal proposals, safety interventions
and original-controller fallback. This planning/execution gap is observed; it
does not establish a unique causal explanation for every later steering or lap
delta. No guard weakening, forced plan completion, margin/release retuning or
extra driving follows this failed candidate.

The same slower cell has fixed-station evidence of local gains followed by costs:

| Obstacle | Path B/C (m) | Issued Steering Integral B/C (command-s) | Steering Variation B/C |
|---|---|---|---|
| 3 | 51.603337 / 51.234594 | .119778 / .106256 | 1.472460 / 1.500668 |
| 4 | 51.330143 / 52.185580 | .140819 / .221686 | 1.500250 / 2.324714 |
| 5 | 51.586939 / 50.912830 | .128655 / .198773 | 1.610537 / 2.289278 |

Obstacle4 has no nominal intervention, yet its inherited closed-loop demands on
the divergent images grow; its window max lateral is6.132022->7.746545m. This
is observed downstream counterevidence, not proof of a unique perception or slip
mechanism. The selected cost also omits absolute steering integral and a scored
baseline-path option, and its second selection has only one eligible candidate.
Thus neither local shorter path nor a minimum within this bank demonstrates
efficient full-episode behavior against the actual frozen control.

The [independent primary audit](../../experiments/koi-nominal-trajectory-v1-audit.json)
SHA`9ab104941db7a2e01c984594665dde30392a7fc9e85da828fc24de90a4593027`
verifies all64 episode/raw/decision/process bindings and every frozen inventory,
exactly reproduces the full passive result and all15 gate outcomes, and independently
reconstructs window/passage/return and action-contract metrics. Only4/2132 candidate
decisions change the pre-shield nominal action (one in1/0002 and three in3/0015).
This is distinct from23 candidate shield overrides and26 final-vs-original crossing
changes, with one nominal/shield overlap. Six whole pairs are exact action/raw no-ops.

Ordinary efficiency covers24 windows/four cells/two roads. Common43 return rows
retain33 returned and10 next-entry censors in both arms; their descriptive bound
sum rises31.555812->31.675812s. This mixes observed onset delay and conservative
censored bounds, not an unbiased population return-time mean. The returned
3/0015/object5 example instead directly slows.74->.82s after rear-clear. Total
child-process wall is403.011148s, excluding parent/preflight/audit work. The audit
has zero new policy executions/environment resets and no artifact discrepancy.

## 27. Stateless Bounded Avoidance Magnitude (2026-10-02)

The user authorizes one separate reaction-based hypothesis: fixed nominal `.34`
near / `.55` crossing avoidance demand causes excessive lateral motion on ordinary
obstacles. Preserve the frozen submission ZIP`c9e376a0...`, all11 members, root
Agent and byte-identical v1 shield`ad772bde...`. Do not repeat margin reduction,
steering-release, persistent trajectory planning or long recovery/release state.
This section is the narrow active plan; section26 remains frozen negative evidence.

### Minimal Treatment

`haic/algorithms/koi/avoidance_magnitude.py` calls the original crossing exactly
once, preserves its selected flank and pedals, and changes only its identified
avoidance term through the original nested clipping/float32 pipeline. Current
full connected-component bounds reuse a pure geometry primitive, not a previous
minimum-clearance/release controller. The one-pixel object-edge pad, skin-inclusive
full wheel/hull envelope and half-pixel footprint allowances are identical to v1.
Unidentified steering, genuine impact suppression, launch prefix, unmatched full
components or invalid HUD/geometry retain the exact original nominal action.

Every decision recomputes the selected-flank remaining lateral displacement `d`
from the current padded obstacle and full footprint. Available approach `L` ends
when the front reaches its nearest longitudinal edge. Geometric demand is
`atan2(2*wheelbase*d, max(L, speed*.08)^2)`. The existing observed crossing-centroid
projection, when available, encloses the transverse interval swept from the current
box to its projection for a lateral-overlap/approach TTC risk proxy. Endpoint-only
risk was corrected before freeze: both endpoints can be clear while the interval
crosses the full footprint. No projection is treated as a collision-clear proof:
without it, risk uses current box geometry. Imminent deep overlap within the actual
first hold gives risk1 and keeps the original strong term. Magnitude is bounded
by `min(legacy_term, max(geometric_demand, legacy_term*risk))`; the old y-based
urgency remains in that ceiling. Reaching the ceiling returns the original command
byte-for-byte rather than introducing rounding-only differences.

The adapter owns only `nominal` and last diagnostics. It has no plan, timer,
actuator feedback, object memory, release latch or action-observation callback.
It hides `.driver` so the unchanged submitted v1 shield remains the final safety
layer and assesses the reduced nominal command, not the unmodified crossing tap.
These geometric/risk calculations and the shield projection are pixel proxies,
not guarantees of physical safety or future reaction-controller trajectories.

### Frozen Comparison And Stop Criteria

Predeclare exactly four consumed ordinary TRAIN layouts/eight contemporary
episodes/two road geometries:1/3184000013 and1,2,3/3184000015. Completed prior
episodes/protocols establish consumption; this is outcome-selected development,
not fresh generalization. No38300/50300, protected, fresh or Track4 geometry and
no official upload/confirmation. Both arms use the exact submitted baseline and
shield. Freeze candidate/helper/model/environment/runtime pins, resource forecast,
schedule and metric/gate definitions before the first reset; no automatic retry.

Reject on any lost baseline finish, per-cell damage/collision increase, new
baseline-clean hit (all24 objects/arm, including unpassed), missing baseline fixed
station+-25 window/common-return followup, invalid/missing matched evidence or
changed shield/pedal/state contract. Ordinary equal-cell per-object reductions
must reach3% max lateral,0.1% path and3% actual issued-steering integral, with
negative directions on both road geometries. Issued-steering jump variation,
common-horizon return-censor count and descriptive onset/censor bound must not
increase. Kept-lap mean must improve at least20ms; no kept lap may slow more than
20ms. Return is abs lateral<=1m sustained .24s after valid physical rear-clear,
with next-obstacle/fixed-followup/episode censors retained, not a survivor mean.

If this single frozen candidate regresses or lacks meaningful efficiency, preserve
its negative evidence, discard it and close overavoidance optimization. No tuning
repeat, persistent-controller extension or post-outcome gate relaxation follows.

### Completed A/B And Closure

[Frozen protocol](../../runs/koi-avoidance-magnitude-v1/protocol.json) SHA
`1456126e31bdbcfce1e3d158bf032990820a4b46fa85dba49dd401031769b890`
completed exactly8 valid contemporary episodes/four matched ordinary cells/two
actual road geometries. The [primary result](../../experiments/koi-avoidance-magnitude-v1-result.json)
SHA`55d489f9418d2aceb786ceaede343a9b4a07b7ea2a23ea503e880d701dad2d01`
is **REJECTED / NOT ADOPTED**. All four pairs match geometry, initial logged full
state/pixels and raw/decision prefixes up to their first changed action. No
operational error, missing mate or repeat masks a natural failure.

| Metric | Frozen Champion | Bounded Magnitude |
|---|---:|---:|
| Finishes /4 | 4 | 3 |
| Damage | 0 | 1.4 |
| Collision-positive decisions | 0 | 7 |
| Physical contact events | 0 | 4 |
| Hit objects /24, including unpassed objects | 0 | 3 |
| Baseline fixed windows retained /24 | 24 | 19 |
| Required prospective return followups retained /24 | 24 | 19 |
| Shield intervention decisions | 9 | 25 |

New baseline-clean hits are1/3184000013 object4,1/3184000015 object1 and
3/3184000015 object1. The1/0015 candidate loses its baseline finish; it ends after
84 decisions versus245 in the control. Its short failed route is not path or lap
efficiency credit. Safety still includes all six objects, not only the passed one.
Exactly98/820 candidate nominal actions change before the same v1;20 overlap
shield interventions, with103 final-vs-local-crossing changes in total. Existing
shield intervention therefore does not imply preserved physical safety.

The three retained cells have all six baseline windows each. Equal-object changes
within each cell, not ratios of whole-episode means, are:

| Cell | Window Max Lateral | Window Path | Issued Steering Integral | Issued Steering Variation | Kept Lap |
|---|---:|---:|---:|---:|---:|
| 1/0013 | -30.487369% | +2.813968% | +14.947015% | +15.514245% | +80ms |
| 2/0015 | -54.292908% | -2.109875% | -31.005219% | -8.939204% | -340ms |
| 3/0015 | -59.786614% | -2.283220% | -20.633667% | +1.338262% | -100ms |

The aggregate `ordinary_equal_cell_relative` values (max lateral-48.188963%,
path-0.526375%, integral-12.230624%, variation+2.637768%) condition on those three
complete-window cells. The failed cell has only1/6 usable windows, so every
`metric_denominators_valid` flag isFalse; these are NOT valid four-cell gains.
Even conditionally, road0013 worsens path and integral, so improvements in both
road geometries are absent. Variation has only19/24 comparable windows and fails
the frozen completeness/nonincrease gate despite its conditional negative sum.

Kept3 mean lap change is-120ms, but excludes the lost baseline finish and includes
1/0013+80ms, above the20ms ceiling. For return, common19 rows have5->1 censors and
descriptive onset/censor-bound sum13.131535->9.258961s;5 required followups are
missing. Full-arm statuses are baseline18 returned/6 next-entry censors versus
candidate18 returned/1 next-entry censor/5 unpassed. Thus neither the common-row
bound nor equal full-arm returned counts establish preserved24-object return
coverage or an unbiased population return-time improvement.

Seven of17 gates pass; finish, per-cell safety, new-hit, window/return coverage,
variation, individual-lap ceiling and all three complete-cohort efficiency gates
fail. The source-level fixed-demand deficiency remains an observation, but this
single stateless clearance/risk treatment does not deliver safe efficient driving.
Per user instruction, discard this candidate and **close overavoidance optimization**.
No tuning, repeat, margin/release/planning detour or new controller follows.

Pre-freeze integrated277 tests+133 subtests PASS; both isolated CPU21 arms import
with zero resets. Independent review found and resolved a dual-end-clear crossing
risk omission and an analyzer raw-tick/outer-wrapper counter/damage phase mismatch
before freeze. The [runtime review](../../experiments/koi-avoidance-magnitude-v1-runtime-review.json)
and [preflight](../../runs/koi-avoidance-magnitude-v1/preflight.json) retain source
binding and limits. Final root/champion ZIP/shield/prior candidate hashes remain
unchanged. No protected/fresh/Track4 geometry or official external action occurred.

### Observed Shield Limit In The Lost Cell

The primary1/0015 decision77 reports selected-component required clearance0,
risk0 and magnitude0 instead of the inherited near.34; issued steer changes
`.042234614 -> -.297765374`. The final unchanged shield already predicts threat
but logs `encounter_budget`, actions6, `rearm_blocked=True` and candidate_count0.
At decision80, the first collision-positive decision, nominal risk is1 and the
full crossing.55 is restored byte-for-byte; issued steer is`.481999993` and
damage rises to.2. Shield again logs `encounter_budget`, negative projected gap
`-1.133664456m`, actions6, blocked rearm and no safe grid candidate. These are
observations in the episode's decision trace, not a counterfactual replay.

Thus retaining v1 as the final layer did not certify the changed nominal closed
loop, and the collision was not simply logged as a projection-clear handback.
The trace identifies its non-intervention branch; it does not prove an unexecuted
grid choice would rescue the lap or identify a unique causal action/mechanism.
Do not change the frozen shield's budget/rearm rules or tune another nominal
variant in response. The prescribed result remains rejection and closure.

### Independent Primary Verification

[Audit receipt](../../experiments/koi-avoidance-magnitude-v1-audit.json)
SHA`7bc1297f416c40366c1a4e2507aca86879ba182923fa08f3e55e16113b19a5db`
rehashes all63 study-root files (60 primary artifacts plus protocol/report/ZIP)
and frozen source/environment/model/runtime/consumed/preservation inventories.
Pure frozen-source re-analysis equals the entire result. Independent raw/decision
reductions reproduce safety, all24 objects/arm including5 candidate unpassed,
logged prefix parity,19/24 windows/common-return rows, held steering/jumps and
all17 gates (7 pass/10 fail). Kept-lap and conditional-three-cell limitations
remain explicit. All24 recorded resource assessments reproduce from receipts.

There is no artifact discrepancy and no additional Agent, policy, simulator,
environment reset or driving replay. Finish flags are reduced from hash-bound
episode records, not a separate tracker/physics replay. The audit adds evidence
only; rejection, no-repeat/no-tune flags, frozen champion and direction closure
remain unchanged.

## 28. Corner-Exit Throttle Diagnosis (2026-10-04)

The user authorized one separate hypothesis: unnecessary retained deceleration
after corner exit delays throttle recovery. Preserve frozen crossing_projection
plus collision-shield v1, entry/mid-corner control and all avoidance logic; the
overavoidance direction remains closed. Implement a minimal separate candidate
and consumed TRAIN A/B only if the existing logs first establish this mechanism.
Otherwise close the direction without extra driving, generalization or official
evaluation. This section is the narrow plan and completed outcome.

### Passive Evidence

[Diagnosis](../../experiments/koi-corner-exit-diagnosis-v1.json) reads only the eight
archived frozen_shield episodes in `runs/koi-nominal-trajectory-v1`: Track1 seeds
3184000002/0013/0015, Track2 seeds3184000001/0006/0015, and Track3 seeds3184000002/0015
(suffixes expand within3184000000). These are eight consumed layouts/five road
geometries,2,130 decisions; archived finishes7/8 with damage0/collision decisions0.

An intentionally broad putative exit requires prior obstacle-free |steer|>=.25
and |physical road-heading error|>=.15rad, then two consecutive decisions with
|steer|<=.12, |heading error|<=.10rad and all wheels onroad. Obstacle/recovery
intervention invalidates corner history. These36 events are alignment candidates,
not36 established straight-road corner exits. At30/36 first aligned decisions,
gas is already>=.5. Four of the six low-gas onsets exceed the current target.

| Track / Seed | Decision | Gas | Target / Pixel Speed | Next28.8m Road Heading Change |
|---|---:|---:|---:|---:|
| 1 / 3184000002 | 160 | .064 | 56.0 / 57.40 | 19.62deg |
| 1 / 3184000002 | 189 | 0 | 54.0 / 57.40 | 54.31deg |
| 1 / 3184000013 | 113 | .192 | 59.2 / 57.40 | 70.38deg |
| 1 / 3184000015 | 94 | 0 | 57.2 / 61.21 | 50.27deg |
| 3 / 3184000002 | 135 | .282 | 55.2 / 51.15 | 29.64deg |
| 3 / 3184000015 | 93 | 0 | 58.0 / 61.21 | 40.08deg |

All six fail the existing projected free-space sprint condition. The physical
heading changes are computed from archived road geometry, not a simulator reset
or a new controller input. Current alignment alone does not establish that those
preview limits are unnecessary. In particular,1/0013 already has a recovered
target of59.2 and approaches the next sharp turn;3/0002 still follows a bending road.

The actual target is `max(38,60-.8*current_road_center_spread)`, with no corner
holding timer. It matches all1,601 eligible logged targets exactly. The832
eligible non-sprint/no-arrival-cap pedal pairs match current-speed proportional
control within2.8e-8. The underlying corridor target EMA is not an issued-pedal
lag: its pedals are overwritten. The active two-HUD speed average exists but
does not establish an unnecessary exit delay in these records.

Descriptive effective-target recovery to>=58 (within2 of base straight target60,
using72 when sprint overrides it) is immediate at31/36 onsets,80ms at3/36 and
censored at2/36 when alignment ends. Gas>=.5 recovery is immediate at30/36,
80ms at1/36,160ms at1/36 and censored at4/36. Censors are retained, not zeroed;
alignment-run duration is not a fixed-station exit-time or matched A/B metric.

### Decision

**HYPOTHESIS NOT CONFIRMED; close this direction without implementing a candidate.**
No clearly unnecessary retained post-exit deceleration was isolated. This is not
proof that every possible corner-exit optimization is ineffective. Do not loosen
entry/preview limits or reopen avoidance/shield work to manufacture a candidate.
No A/B safety, fixed-station exit-time or lap delta is measured or claimed.

Reproduction: `python -m scripts.diagnose_koi_corner_exit --output experiments/koi-corner-exit-diagnosis-v1.json`.
The passive script verifies the exact ZIP/all11 member hashes and unchanged
worktree shield, checks the logged target/pedal formulas and preserves input
hashes/event contexts. Zero new policy executions, simulator resets, driving
repeats, generalization runs, submissions or confirmations.

## 29. Corner Target-Speed Headroom Diagnosis (2026-10-04)

### Scope And Decision

The user authorized a single hypothesis: road-shape target speed is unnecessarily
conservative before corners. Start with existing consumed TRAIN logs; only clear,
repeated headroom permits a separate minimally raised mapping and consumed A/B.
Do not change the frozen champion, lookahead, steering, avoidance or shield, and
do not reopen exit-throttle or overavoidance. Work is on
`research/koi-corner-target-speed`, based on preservation commit `2408bed`.
Existing staged and unrelated work is retained, not folded into this diagnosis.

**NO CLEAR HEADROOM; close this direction without candidate implementation or A/B.**
One promising passage is not replicated cross-road support for changing the
mapping. This is a bounded evidence decision, not proof that all higher corner
speeds are unsafe or that the current controller is optimal.

### Active Mechanism

Frozen `fast_completion_coordination.py` computes after decision10:
`T=max(38,60-.8*S)`, where `S` is the max-minus-min horizontal center position in
available image rows54/50/46/42/38/34/30. Fewer than two rows uses `S=28`; a near
obstacle separately capsT at44. **S is a pixel spread, not calibrated curvature.**
Camera/vehicle heading, lateral position and missing rows also affect it.

ForT<60, gas is `clip(.12+.04*(T-v),0,.6)` and brake is
`clip(.02*(v-T-2),0,.28)`. AtT60 the inherited brake cap is.15. Herev is the
two-frame HUD estimate, not exact physical hull speed. Sprint and arrival braking
can replace these pedals; crossing and shield then affect steering. The discarded
corridor EMA is not an active target lag. The separate source inspection and
832 eligible logged pedal pairs agree (maximum residual2.80e-8). Across these
832 observations HUD minus physical speed ranges-1.44 to+5.66, mean+1.14, so
above-target tests use HUD values rather than subtracting unlike measurements.

### Passive Windows And Limits

[Result](../../experiments/koi-corner-target-diagnosis-v1.json) reads the same eight
`frozen_shield` episodes in `runs/koi-nominal-trajectory-v1` as section28: five road
geometries and2,130 decisions, no new interaction. Geometry-only corner grouping
uses same-sign curvature>=.01rad/distance-unit, bridges<=10.5 units without a
meaningful opposite turn, and requires total heading change>=15deg. Observe through
corner end+10 units; exclude the track seam. Keep68 complete, one censored/nonforward
and nine unreached windows. These78 episode/corner observations overlap across
layouts and sometimes in followup; they are not78 independent trials.

Entry must have an active nominal target, and the passage must have no detected
obstacle/recovery/shield intervention to enter the unconfounded descriptive subset:
23 complete passages. Raw20ms wheel contacts and obstacle collision/contact records
plus cumulative damage determine observed safety. The local margin proxy is
`40/6-|lateral|-1.6*|cos(heading_error)|-2.61*|sin(heading_error)|`.
Road half-width40/6 is from the protocol-pinned simulator; the footprint envelope
is conservative locally, not exact clearance to the curved road boundary. A
negative proxy is NOT itself proof of road departure. Actual contact losses are
reported separately; metric uncertainty cannot be treated as positive headroom.

The working diagnostic screen requires at least three complete clean passages
across two geometries in one pixel-spread band, entry HUD>=T+2, and the same excess
in>=80% of at least three eligible same-band decisions before corner end. Minimum
margin proxy must be>=1, with no unconfounded low-margin/contact-loss counterexample
in that band. Post-corner target recovery is NOT counted against speed support.
These are explicit observational screening definitions, not a physical safety
theorem, an independently preregistered trial, or a causal speed-limit estimate.

### Results

The following table covers only the23 unconfounded passages. Target and entry
speed are separate native-scale means; do not read their difference as calibrated
speed headroom. Minimum margin is the worst local proxy over the full passage and
followup. Heading-change strata and all exclusions are retained in the JSON.

| Entry Pixel Spread | Passages | Mean Target (HUD) | Mean Physical Entry Speed | Minimum Margin Proxy | Any-Wheel-Offroad Passages | Damage-Positive |
|---|---:|---:|---:|---:|---:|---:|
| 0-5 | 6 | 57.40 | 53.57 | -3.85 | 1 | 0 |
| 5-10 | 5 | 54.34 | 52.66 | -0.79 | 0 | 0 |
| 10-15 | 4 | 51.13 | 50.24 | -1.40 | 0 | 0 |
| 15-20 | 5 | 46.64 | 50.67 | -0.17 | 0 | 0 |
| 20-27.5 | 3 | 40.83 | 45.60 | -0.73 | 0 | 0 |

Braking occurs in72/91 nominal decisions with spread15-20 and31/37 with
spread20-27.5; this establishes that the target can bind, not that braking is
unnecessary. In16 unconfounded passage approach windows, the first observed
eligible brake lies from17.95 units after to39.28 units before geometric entry;
signed time lead ranges-.24 to+.72s. The means are13.84 units/.266s before entry.
This is a bounded40-unit lookback's first brake observation, not a guarantee of
the original braking onset if braking was already active at the window boundary.

There are12 unconfounded, clean above-target entries. Only
`2/3184000001/corner240` satisfies sustained same-band support: target54.8,
HUD62.45, physical entry59.31 and minimum margin proxy3.26. All four relevant
same-band commands exceed target+2, but the full passage still applies six brake
commands and physical speed falls to49.48. First observed approach braking is
21.42 units/.34s before entry. It is one geometry, not repeated mapping evidence.

Nine descriptive comparisons find an equally/more sharply curved clean passage
entered at least two physical speed units faster. They are unmatched and not
independent: for example, the same road/corner189 on tracks2 and3 of seed0015
differs by4.68 at entry. Different preceding states/obstacles and later braking
prevent interpreting this as a causal safe target increase.

Counter-evidence: `3/3184000002/corner222` enters at physical44.76/HUD44.85 with
target57.2, yet later has17 partial-wheel-offroad raw ticks, including two
all-wheel-offroad ticks, with zero damage/collisions. This is an observed current
safety limitation, not proof that entry speed caused it. Other low-margin proxy
examples include `3/3184000002/corner200` and `3/3184000015/corner214`; neither
has observed wheel-contact loss, so do not relabel them as actual departures.

No pixel-spread band has repeated headroom across distinct geometries. The
required basis for even a narrow mapping increase is absent in this bounded
sample. Candidate/A-B are skipped; no lap improvement or safety preservation of
a changed policy is claimed. Frozen ZIP, all11 source members and worktree shield
rehash unchanged; root Agent and all policy code have no changes.

Reproduce with `python -m scripts.diagnose_koi_corner_target --output experiments/koi-corner-target-diagnosis-v1.json`.
Only four focused synthetic checks were run via
`python -m unittest tests.test_diagnose_koi_corner_target`; all pass. No full-suite
audit, replay, simulator reset, new geometry, blind or official evaluation.

## 30. Soft-Boundary Corner Cutting Diagnosis (2026-10-04)

### Scope And Decision

The user authorized a new single hypothesis: the champion follows unnecessarily
long corner paths and a minimal inward apex shift, with brief offroad allowed,
could shorten laps. Preserve exact crossing_projection plus collision-shield v1,
speed targets, obstacle avoidance and straight control. Exit-throttle, target-speed
increases and overavoidance stay closed. Analyze consumed logs first; only repeated
meaningful viable-looking shortcuts authorize a separate minimal candidate/A-B.
The working branch is `research/koi-corner-cutting`; prior work and staging remain.

**CLOSED AT THE PASSIVE GATE, no runtime implementation or A/B.** Geometric savings
exist on two roads, but the tested small-offset, fixed-speed family supplies no
route passing the combined reentry/steering/demand screen. This is a bounded
engineering decision, NOT a physical proof that corner cutting cannot work. Do
not hide positive geometric savings or present model-based rejection as a driving
failure. Do not widen to a full planner or change speed to rescue this direction.

### What Was Measured

[Primary result](../../experiments/koi-corner-cutting-diagnosis-v1.json) uses only
the same eight archived `frozen_shield` episodes/five geometries as sections28/29.
Use the previous geometry-only corner grouping, now requiring >=45deg total turn.
Observe from10 distance units before corner start to10 after end, retaining raw
endpoints. Of62 episode/corner observations,55 are complete. The other seven
censored/nonforward/unreached windows stay in the artifact, not fresh or successful.
Only15 complete windows lack recorded obstacle/recovery interference. Observations
across layouts and overlapping corner windows are not independent samples.

Actual distance sums raw XY increments; centerline arc uses the same endpoints'
projected stations. The straight endpoint chord is an unconstrained lower bound,
not the minimum drivable route. **44/55 actual paths are already shorter than
their centerline arcs**: the literal premise of simply driving the full centerline
arc is not supported for most observed corners, even though nominal targets may
still be road-center based.

For each complete window, resample the actual path and blend toward its chord with
a `sin(pi*u)^4` envelope. Four bounded peak displacements, .75/1.5/2.25/3 units,
give220 geometric alternatives. Endpoint offset and its first two derivatives
vanish analytically; this does not eliminate noise in the sampled baseline path.
The family is a passive feasibility probe, not a runtime trajectory planner or
proof that a pixel aimpoint shift would track those exact paths.

### Soft Edges And Feasibility Limits

Archived track beta/XY construct the actual road quadrilateral union, half-width
40/6. Project four wheel centers against it; **nonzero offroad is allowed**.
Report total/longest any-wheel and longest all-wheel offroad at archived speed,
plus a half-speed sensitivity. Wheel-center membership is not exact wheel-fixture
contact. Reentry means predicted all-center road support resumes; failure to rejoin
within the fixed observation window is unresolved, not proof of permanent loss.

The operational screen asks for >=2 distance units AND3% shortening in at least
three corners across two geometries; longest predicted offroad<=.5s, reentry/exit
heading<=15deg, >=.5 rectangle/obstacle clearance and no recorded obstacle/recovery
interference. These are diagnostic definitions, not official rules. Geometry uses
archived world positions only offline; none can be supplied to a runtime Agent.

Curvature is averaged over approximately one80ms controller hold to avoid treating
piecewise-linear interpolation vertices as real steering impulses. Wheel demand
is calibrated locally as recorded front-wheel angle plus the change in
`atan(3.24*curvature)`. Screen against the larger of the nominal.4rad stop and
observed maximum, plus.005rad, and110% of the larger of3rad/s and observed rate.
P95 lateral demand must not exceed110% of baseline. This optimistic local slip
approximation is NOT a validated dynamics model; it cannot prove infeasibility.
Measured joint overshoot is retained rather than rejecting unchanged behavior
merely because the motor target is smaller. The initial uncalibrated derivative
estimate was replaced during diagnostic development, not treated as a real run.

Independent source inspection confirms that grass uses60% of per-wheel road force
capacity, NOT60% speed or an immediate retirement predicate. Longitudinal and
lateral demands share that budget. Thus logged-speed timing is optimistic and a
shorter path can still be slower. Break-even fractional mean-speed loss equals
fractional distance saving. Finishing requires>=95% unique-tile coverage plus
qualified forward start crossing, not merely reaching a centerline station.
Predicted newly skipped wheel-center tiles are reported as proxies; they are not
exact contact/whole-lap coverage. The101-negative-reward-decision retirement rule
is not an8.08s guaranteed grass allowance or a wheel-offroad timer.

### Observations

All figures below concern the best geometric member of the bounded family, not
executed candidate driving. Distance is in simulator world units. A dash for
reentry means no predicted road departure; it is not an unmeasured return set to0.

| Track / Seed / Corner | Actual | Centerline Arc | Chord Lower Bound | Bounded Cut | Longest Predicted Offroad | Reentry Heading | Estimated Wheel Demand / Screen Limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 / 3184000013 / 145 | 48.82 | 51.10 | 30.86 | 45.54 | 0s | - | .532 / .464rad |
| 1 / 3184000015 / 236 | 42.48 | 44.12 | 32.40 | 40.18 | 0s | - | .580 / .423rad |
| 1 / 3184000015 / 284 | 38.83 | 44.52 | 28.31 | 36.30 | .234s | 31.44deg | .479 / .418rad |
| 2 / 3184000015 / 92 | 53.80 | 55.75 | 41.83 | 51.80 | 0s | - | .528 / .439rad |

These four unconfounded windows on two geometries pass the geometric-only gain
criterion (2.00-3.27 units,3.72-6.71%). None passes all feasibility screens.
The predicted.234s grass excursion is below the permitted diagnostic duration;
its rejection is reentry/demand, NOT a hard road-boundary constraint. Rectangle
obstacle margins in these four are respectively39.67/25.53/7.41/21.21 units, so
predicted obstacle intersection is not their limiting factor. The other11
unconfounded best alternatives fail the minimum gain and/or return/demand tests.

Across all220 alternatives, reason counts overlap:185 small gain,202 steering
demand,161 increased lateral demand,160 baseline obstacle/recovery interference,
53 reentry heading,32 exit not aligned/onroad and only4 prolonged offroad
predictions. Do not interpret these as independent failures or observed collisions.
No complete alternative passed the combined screen, so no repeatable feasible
gain is established for the minimal fixed-speed proposal. The unconstrained chord
alone cannot justify new driving. A/B path, offroad, heading and lap deltas are
unmeasured, not zero.

### Preservation And Reproduction

Separate source inspection identified a potential post-crossing/pre-shield
stateless nominal boundary, which would preserve same-observation pedals and the
shield. It was NOT implemented because this diagnostic gate did not pass. Frozen
ZIP/all11 members verify unchanged; root Agent, shield and all policy code have
no changes. No candidate, simulator reset, new cells, blind or official action.

Reproduce with `python -m scripts.diagnose_koi_corner_cutting --output experiments/koi-corner-cutting-diagnosis-v1.json`.
Four focused geometry/timing/zero-shift calibration tests pass via
`python -m unittest tests.test_diagnose_koi_corner_cutting`. No broad audit.

## 31. Outside-Entry Pre-Positioning Diagnosis (2026-10-04)

### Scope And Decision

The user explicitly narrowed section30's closure to directly pulling the apex
from the recorded entry state. The new single hypothesis is that an earlier
outside entry position/heading enables a gentler and shorter line. Prepare10/20/30
decisions ahead; preserve the frozen champion, speed targets, obstacle avoidance
and v1 shield. Brief offroad is allowed, but not prolonged or unresolved departure.
The branch is `research/koi-corner-preposition`, preserving all earlier work.

**CLOSED AT DIAGNOSIS; no candidate or A/B.** Natural comparisons do not supply
multi-geometry isolated support. A bounded, genuine outside-entry path family does
not retain meaningful repeated savings after setup/return cost. This is not proof
that every racing line or preposition controller is physically impossible. The
limited family, model uncertainty and small positive observations remain explicit.

### Natural Entry Evidence

[Natural artifact](../../experiments/koi-corner-entry-natural-v1.json) verifies the
24 archived input files against section30's hashes: eight consumed champion
episodes, five roads,2,130 decisions and8,511 raw ticks. All62 observations remain:
55 complete, one censored/nonforward and six unreached. The complete-window
geometric-entry split is6 outside/15 center/34 inside; the censored passage is also
inside and is not pooled as a complete outcome.

Outside-positive lateral is `turn_direction*raw_lateral`, with outside>1,
center[-1,1], inside<-1. Entry is the first observed crossing of geometric corner
start, preserving its bracket/overshoot. Histories use exact logged controller
decisions, not assumed raw-step offsets. Heading is actual BODY road-heading
error, not the tangent of the measured XY path. Original complete-window endpoints
remain unchanged for the natural outcome comparisons.

| Classification Time | Outside-vs-Other Same-Geometry Contrasts | Outside Path Shorter | Shorter And Lower Max Front-Wheel Angle | Both Prior Corner Windows Without Recorded Interference |
|---|---:|---:|---:|---:|
| Geometric entry | 3 | 0 | 0 | 0 |
| 10 decisions earlier | 7 | 4 | 3 | 1 |
| 20 decisions earlier | 8 | 4 | 1 | 0 |
| 30 decisions earlier | 10 | 6 | 4 | 0 |

These are contrasts among29 same-seed/corner layout pairs, not independent trials
or matched controller A/B. All geometric-entry contrasts are on seed3184000002;
outside paths are1.289/10.863/11.521 units longer, but preceding state/obstacles
confound that association. Every earlier-outside contrast with BOTH shorter path
and lower maximum front-wheel angle occurs only on seed3184000015. No contrast
has two complete30-decision approaches plus windows free of recorded interference.
The sole pair with both old corner windows unflagged still has approach interference.

The artifact compares actual path/arc, maximum issued steering and INDIVIDUAL front
wheel angles, minimum physical speed, total/longest partial/full offroad, damage,
collision and actual BODY-heading at contact reentry. Across55 complete windows,
14 have some wheel contact loss, three have all-wheel loss, ten observed reentries,
and four remain offroad at the window boundary. All have zero damage/collision.
These are descriptive outcomes, not benefits caused by outside entry. Missing
returns are null/censored, not zero heading. Pooled different-corner averages must
not be used as causal evidence.

In the prior four rejected opportunities, three are ALREADY outside10 decisions
earlier:1/0013/145 lateral+4.749;1/0015/236 +5.596;2/0015/92 +4.759. None remains
outside at geometric entry. The fourth,1/0015/284, changes from-.963 at-10 to
-5.651 at entry. All four have obstacle/impact-proxy/recovery flags somewhere in
the30-decision approach; flags are not physical collisions. Recorded pixel centers
and their second differences are included, but do not establish future-corner
observability or justify an oracle lookup in a runtime controller.

### Counterfactual Method

[Recalculation](../../experiments/koi-corner-preposition-diagnosis-v1.json) uses
only those four prior opportunities on two geometries. For each10/20/30-decision
start, compare common full start-to-return endpoints, ending20 station units after
the geometric corner. Include setup, corner and return cost; use identical
resampling for reference and alternatives and retain raw baseline length too.

The bounded grid is outside offset2/4, inside apex2/4/6, apex at50/65% of corner
station span, and outside turn-in5/10 units before entry:72 paths/case,288 total.
The outside waypoint is also explicitly held at original entry. Cubic C2 waypoint
interpolation is corrected with endpoint quintics to preserve locally fitted
position/tangent/curvature jets. Every8-unit setup waypoint gradually blends the
original approach toward the outside using quintic smoothstep. This prevents a
long endpoint spline from taking credit for cutting an unrelated earlier corner.
The entry classification uses the proposed path's ACTUAL projected station
crossing, not merely the spline parameter called entry. All final paths reach
outside, inward-directed entry; geometric endpoint curvature debt is also closed.

This construction is deliberately a small offline family, not a full runtime
planner. During diagnostic development, unconstrained splines that crossed entry
on the inside or shortened unrelated approach bends were corrected, not counted
as valid preposition evidence. The original closed diagnostics were not rewritten.

Use the same archived physical-speed schedule as a function of projected road
station for all routes. Report analytic, unfiltered curvature/bicycle angle/rate
and lateral demand separately for setup/corner/return, plus the previous filtered
kinematic measure for same-model comparison to direct pulling. Do not dilute a
corner peak with a longer setup's whole-window P95, or carry baseline steering
residuals to a different entry state. Physical speed is prescribed here, not a
measured consequence of unchanged pedal code. Minimum-speed improvement cannot
be inferred from these paths.

Road edges remain soft. The screen allows up to.5s predicted continuous wheel-center
offroad, requires observed-window return and <=15deg reentry/exit heading,
rectangle-obstacle clearance>=.5, forward progress and small endpoint debt. The
nominal bicycle screen uses angle<=.405rad/rate<=3.1rad/s and a5% same-model peak
steering improvement over direct pull. These are screening proxies, not physical
limits/proofs for a slipping car. Footprints follow velocity tangent, not known
counterfactual hull yaw, and wheel centers are not exact fixture contacts. The
source-checked grass force and95% unique-tile finish caveats from section30 remain.

### Net Results And Sensitivity

Each cell below is the BEST net distance saving over its24 configurations, with
the full setup and return included. Positive means shorter; negative means longer.
These optimistically selected geometric quantities are not measured lap gains.

| Prior Case: Track / Seed / Corner | Start10 Decisions Early | Start20 Early | Start30 Early |
|---|---:|---:|---:|
| 1 / 3184000013 / 145 | +0.608 | +1.114 | +1.459 |
| 1 / 3184000015 / 236 | -0.781 | -0.133 | -0.391 |
| 1 / 3184000015 / 284 | -5.101 | -4.855 | -4.509 |
| 2 / 3184000015 / 92 | -0.941 | +0.419 | +0.616 |

Nineteen of288 paths have positive net savings, none reaches the operational
2-unit meaningful-gain threshold, and none passes the combined feasibility screen.
This conclusion does NOT depend only on the uncertain bicycle model: dropping
its veto still leaves zero accepted paths. Dropping demand/steering-improvement/
overlap vetoes AND the2-unit threshold leaves only two positive geometry-only
paths, both1/0013/145 at30 decisions, saving.186725/.228157 units with no predicted
departure. They are retained as small positive signals, not repeated evidence.

No final path was rejected for excessive offroad duration or endpoint curvature
debt. Thus this is not a disguised hard-road-boundary rule or an endpoint-splice
artifact. Other overlapping flags are128 reentry-heading,90 obstacle-clearance,
240 existing avoidance/recovery overlap,195 no filtered peak-steering improvement,
and288 analytic bicycle-demand failures. Those last failures are model sensitivities,
not288 observed driving failures. Same-code pedals, terminal slip/yaw/steering state
and lap-wide coverage would still require real A/B; no such result is fabricated.

### Preservation And Reproduction

No runtime candidate, A/B, simulator reset, fresh road, blind/private or official
evaluation. Frozen ZIP/all11 sources verify unchanged; root Agent, speed targets,
avoidance and shield have no edits. The previous direct-apex pull, exit-throttle,
target-speed increase and overavoidance closures remain separate and unchanged.

Commands: `python -m scripts.diagnose_koi_corner_entry_natural --output experiments/koi-corner-entry-natural-v1.json`
and `python -m scripts.diagnose_koi_corner_preposition --output experiments/koi-corner-preposition-diagnosis-v1.json`.
Eleven focused tests pass via
`python -m unittest tests.test_diagnose_koi_corner_preposition tests.test_diagnose_koi_corner_entry_natural`.
No broad audit. Code, evidence and shared summaries remain uncommitted local work;
existing unrelated staging was not modified.

## 32. Fixed Sprint72 Handback Relief (2026-10-04)

### Authorized Scope And Frozen Design

The user authorized ONE separate candidate and a small matched consumed TRAIN A/B
after the passive diagnosis gate. Work is on `research/koi-sprint72-relief`.
The previous diagnosis was reported in conversation, not saved as a standalone
sprint protocol/result; the canonical primary evidence is the eight champion
episodes in `runs/koi-nominal-trajectory-v1`. Do not invent a prior frozen sprint
artifact or reuse that study's avoidance-efficiency gates as this study's gates.

Keep the exact submitted champion ZIP `c9e376a0...` and all its eleven source
members immutable. The candidate package alone inserts the helper from
`haic/algorithms/koi/sprint72_relief.py` into its copy of `FarHazardAgent.act`, after
current far detection/tracker update/near selection, before the unchanged arrival
block. All other original package members, including shield `ad772bde...`, remain
byte-identical. No root Agent, previous candidate, or frozen evidence is edited.

The branch requires finite original two-frame HUD `v>=72`, exact parent `T==60`,
`steps>10`, and the EXACT original sprint spatial predicate: no near object;
original road-center rows42/54 present; `abs(C[54]-42)<3`; samples>=4;
`abs(parent_steer)<.18`; `D>=max(18,max(v,72)*.4)`. Reuse logged/runtime D and N;
do not recalculate distance, use extended far-road centers, round or repair FP.
Current far and unresolved track must be absent. Genuine impact clear is
`(impact_before>0 or impact_proxy_trigger) and not braking_proxy_veto`; it must
be false. Require post-parent `recovery_left==0` and no geometry repair, then
verify current pre-arrival gas0, brake action-dtype `.15`, and parent steering.

Only there, issue `gas=0` and `brake=clip(.02*(v-72),0,.15)`. Do not write steering.
Original arrival/contact code writes the actual issued brake to `brake_history`.
The existing preview target and sprint-activity diagnostics retain their meanings;
dedicated candidate fields distinguish eligibility from actual action change.
Later impact veto/steering may change through real history and physical feedback.
No new timer, latch, deadband, hysteresis, cap/target increase or hidden-state veto.

### Prerun Evaluation Contract

Use ALL eight original consumed cells, not a success-selected subset: track1
3184000002/0013/0015, track2 3184000001/0006/0015, track3 3184000002/0015.
This is eight layouts/five geometries and sixteen contemporary natural episodes,
not fresh generalization. Include the reverse-driving cell and the diagnostic
steps194/195/447/448 in its historical applicability account; privileged heading
is evaluation-only and never filters policy eligibility. No repeat, replacement,
fresh/protected/private/official evaluation or submission is authorized.

Before reset, freeze independent arm packages, source/environment/CPU runtime,
consumed evidence, metric definitions, resource forecasts and preservation hashes.
Require one intent and one natural episode per slot, complete raw/decision coverage,
same geometry/initial state/pixels and exact prefix before the first changed action.
Operational tails and unmatched pairs remain visible and cannot pass adoption.

Predeclared safety gates retain every baseline finish, per-cell damage/collision/
physical-contact counts, all objects/no new hits, and nonincreasing any-wheel and
all-wheel offroad occupancy and longest spells. Efficiency requires full baseline
cap-window coverage, lower station-matched physical undershoot and fewer repeated
issued brake-to-gas cycles. Report HUD and physical speed separately, all incomplete
or reverse windows, own-policy and matched station trajectories, and actual arrival
onsets. Distinguish eligibility-end (such as T<60) from the later real restrictive
command; source/FP-boundary failure alone is not proof of a physical restriction.
Include downstream steering, impact triggers/vetoes and shield interventions.

Common-finish lap mean must improve, with negative means on at least two road
geometries and no common cell slower by more than one20ms raw tick. Lost finishes
cannot be offset by survivor means. Exact numerical metrics/gates are bound in
the new protocol; no gate is relaxed after seeing outcomes. Failure closes THIS
specification unchanged, with no `.02` tuning, FP repair or maintain-state extension.

### Frozen Execution And Verdict

**REJECTED / NOT ADOPTED. Close this exact specification without further tuning.**
The [protocol](../../runs/koi-sprint72-relief-v1/protocol.json)
SHA`7fca6b24509812aadde8f6dd2b2dd3a0a57bcb217a796ed9f35e5c6cb73d771d`
completed all16 natural episodes, eight matched pairs, with16 reset intents/ends,
zero operator errors/unmatched episodes/retries. The
[result](../../experiments/koi-sprint72-relief-v1-result.json)
SHA`b20da6b392f63448871828369693a2bfc638dcc4f638d34ea39297cca76e783d`
passes source, law/history, prefix, geometry and all-object integrity checks.
All8 initial states/pixels/geometry and logged pre-divergence raw/action prefixes
match. Two zero-applicability controls are complete action/raw no-ops.

Candidate ZIP is `e7062c66aaf9407d14e4f153b0e1473e13672194e5ecc2b73d4ddbf5e7d3e5be`;
helper source is `ba053977c26743bd7d03fb1e7c550780cf282a5ad0c7d843b31371f2242914cd`.
Candidate changes39 nominal decisions. Final source/environment/model/preservation
validation succeeds after execution; champion/all11 original source members,
unchanged v1 shield, root Agent and all previous candidates are preserved.

| Track / Seed | Champion | Candidate | Common Lap Delta |
|---|---:|---:|---:|
| 1 / 3184000002 | 17.560s | 17.460s | -100ms |
| 2 / 3184000006 | off_track | 21.800s | gained, not a paired lap |
| 2 / 3184000001 | 19.420s | 19.420s | 0ms |
| 3 / 3184000002 | 17.860s | 17.860s | 0ms |
| 1 / 3184000013 | 19.960s | off_track | LOST, not omitted |
| 1 / 3184000015 | 19.560s | 19.740s | +180ms |
| 2 / 3184000015 | 19.780s | 19.860s | +80ms |
| 3 / 3184000015 | 19.400s | 19.400s | 0ms |

Finishes7/8->7/8 are NOT preservation: kept6, lost1, gained1, neither0. The common6
mean is+26.666667ms. Their road means are0002:-50ms,0001:0ms,0015:+86.666667ms;
only one improving geometry, not the required two. Both+180/+80ms cells violate
the20ms ceiling. Gained21.8s and the lost19.96s cannot be combined into a paired
lap-time estimate.

The metric is the environment's native `lapTimeMs`, not raw episode duration:
3/0015 reports19400ms in both arms despite971/970 raw ticks. Do not relabel its
native tie as a20ms lap improvement by substituting a different clock.

| Full-Cohort Safety | Champion | Candidate |
|---|---:|---:|
| Damage / collision decisions / physical contact events / hit objects | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Objects retained | 48 | 48 |
| All-wheel-offroad ticks / total duration | 64 / 1.28s | 366 / 7.32s |
| Longest all-wheel-offroad spell | 1.14s | 6.88s |
| Any-wheel-offroad ticks / total duration | 576 / 11.52s | 767 / 15.34s |
| Longest any-wheel-offroad spell | 1.36s | 7.00s |

The lost1/0013 cell has all-wheel-offroad2->345 ticks and longest.04->6.88s;
any-wheel-offroad62->364 ticks and longest.36->7.00s. Its collision/damage stays0.
The gained2/0006 cell improves57->16 all-wheel-offroad ticks and1.14->.32s longest
spell. Preserve both positive and negative cases. `off_track` is the natural
termination label, not a synonym for the separately measured wheel-contact state.

### Cap Windows And Oscillation

The frozen evaluation definition starts at an eligible cap hold and follows to
the next actual restriction, recording earlier source-eligibility exit separately.
It gives13 baseline windows, all reaching a restriction, versus11 candidate windows
(9 reaching a restriction,2 terminal-censored). The prior diagnosis's14 windows
also included an arrival-only boundary with no eligible relief hold; that control
episode remains fully present here. Neither different window count nor terminal
censoring is treated as zero delay or an improvement.

There is a real local amplitude signal. On2/0006's first cap window, both arms
start at physical72.735977 and run decisions40-46 before arrival47; minimum speed
is65.175136->69.782224, entry-to-trough drop7.560842->2.953753, and reference72
undershoot6.824864->2.217776. Issued brake-to-gas transitions are1->0 in this
window. These are physical-speed units;72 remains an imperfect reference because
runtime eligibility uses the original averaged HUD, not physical velocity.

The same window's raw physical velocity at each80ms decision boundary is:

| Seconds After First Cap Hold | Champion | Candidate |
|---|---:|---:|
| .00 | 72.735977 | 72.735977 |
| .08 | 68.822291 | 72.325917 |
| .16 | 65.175136 | 71.390241 |
| .24 | 66.177082 | 71.087788 |
| .32 | 68.243974 | 70.761424 |
| .40 | 71.159425 | 70.435020 |
| .48 | 74.000039 | 70.108621 |
| .56,arrival47 begins | 70.088113 | 69.782224 |

Candidate removes this particular down/up excursion, but ends at a slightly
lower speed when arrival starts. Do not treat only its higher trough as a lap gain.

However only4/13 baseline station windows are fully covered. Six candidate spans
meet a real restriction earlier, two baseline reverse starts are not reached in
the same direction, and one end is unreached. The complete-cohort matched deficit
and repeated-cycle means remain NULL, not zero and not a survivor estimate.
The four valid individual station-mean physical deficits are:

| Cell / Baseline Start | Champion | Candidate |
|---|---:|---:|
| 1/0002,30 | .479594 | 0 |
| 1/0002,56 | .894996 | 7.700446 |
| 2/0006,40 | 2.857699 | .992571 |
| 2/0015,221 | .232088 | 0 |

Thus even the covered subset is not uniformly better. The later1/0002 span is
slower despite its earlier intervention's smaller undershoot. Do not infer a
full six-cell/four-geometry improvement from three favorable rows.

Own-window repeated brake-to-gas cycles are3->0, but windows differ and two
candidate tails are terminal-censored. Whole-episode transitions215->198 also
mix changed duration, the rescued reverse lap and the newly failed lap. Among the
six common finishes they are135->136. These observations do not pass the frozen
full-coverage matched oscillation/undershoot gates.

All raw velocity samples through the next restrictive decision are retained.
For1/0013, first-cap minimum65.773->69.752 is a descriptive local improvement,
but the actual restriction moves61->59. Baseline eligibility already ended at58
while its original sprint still ran; this is not mislabeled as braking onset.
The candidate's later failure and missing second baseline span are not hidden.

### Reverse Travel And Downstream Effects

The baseline keeps all four eligible reverse decisions194/195/447/448 in2/0006,
two reverse cap windows and1,431 reverse-tagged raw ticks. No heading gate is added.
The candidate changes the earlier forward cap sequence, finishes this cell in
21.8s and has zero reverse-tagged raw ticks. It therefore does NOT execute those
identical four reverse states in the real A/B. Their baseline windows remain as
explicit unmatched same-direction comparisons, and source-bound archived-input
tests separately verify that all four pass the unchanged runtime applicability.
This rescue is positive evidence for this cell, not permission to discard its
old reverse records or ignore the different lost finish.

Arrival code remains byte-identical, but its inputs and entry speeds change:

| Cell / First Post-Cap Object | Arrival Decision B/C | Cap B/C | Physical Entry Speed B/C |
|---|---|---|---|
| 1/0002,object0 | 76/77 (+80ms) | 68.309/65.729 | 70.296/72.212 |
| 2/0006,object0 | 47/47 | 67.722/67.033 | 70.088/69.782 |
| 1/0013,object0 | 67/67 | 67.948/66.571 | 64.274/75.034 |
| 1/0015,object0 | 57/57 | 65.181/64.849 | 63.939/66.825 |
| 3/0015,object0 | 49/49 | 65.937/64.949 | 72.872/65.353 |

The late2/0015 cap has no further arrival onset before finish; the two zero-
applicability controls retain identical arrival histories.

For1/0013 and1/0015, the first cap-presence onset does not yet brake in the
champion but does in the candidate. Entries differ in road station as well as
speed. Object attribution is an offline nearest-projection proxy, not runtime
identity proof. These are downstream perception/trajectory effects, not changes
to the cap function, and not proof of a unique causal action for the later DNF.

First same-ordinal steering differences occur at33 (1/0002),49 (2/0006),61
(1/0013),51 (1/0015),225 (2/0015),50 (3/0015); the two controls remain exact no-ops.
Those comparisons are no longer same-state counterfactuals after pedal divergence.
Impact triggers/vetoes total232/232->204/204: every observed trigger remains
vetoed, and genuine impact-clear is0 in both arms. Do not attribute this failure
to an unobserved impact-veto release merely because history coupling is possible.
Shield interventions22->23 (lost cell0->1) confirm downstream behavior can differ
under byte-identical shield code; they are not a guarantee of road safety.

### Validation And Stop

Final pre-freeze integrated159 tests+132 subtests PASS, including43 candidate
runtime tests and actual candidate-package Far/Contact/shield action checks.
Both exact CPU21 isolated model imports pass with zero environment resets.
Execution then performs exactly16 authorized resets and natural episodes; the
result's `environment_resets:0` refers ONLY to its passive analysis invocation.
The independent runtime review's single steering-threshold test-isolation issue
was corrected before freeze; the helper/formula itself did not change.

The [independent passive postrun audit](../../experiments/koi-sprint72-relief-v1-audit.json),
SHA`cfe02a625639c5a2509cfbd46e58f2ea205cd3d532b70d1d820f458016081d85`,
finds no artifact discrepancy. It verifies116 run files, all16 unique processes/
reset intents/ends, all96 object records, all4094 actual-brake/history/law decisions
and all8 exact prefixes. The frozen analyzer reproduces the saved2,502,700-byte
result byte-for-byte. Source, environment and preservation inventories match.
Native finish-line arithmetic over archived states independently reproduces all
finish flags/laps and explains the center-crossing versus terminal-tick timing
distinction above. This audit adds no policy call, physics step, reset or driving
repeat and does not alter a gate or the rejection.

Reproduce passive analysis only:
`/tmp/kilo/haic-cpu21/bin/python -B -m scripts.analyze_koi_sprint72_relief --run-dir runs/koi-sprint72-relief-v1 --output <new-result.json>`.
Do not rerun the operator or overwrite primary frozen evidence. The operator
refuses a second run. All raw trajectories, per-decision actual history, source
copies, source pins and CPU/resource/process receipts remain in the run directory.

The decisive lost finish, offroad regression and inconsistent/slower retained
laps reject the candidate independently of the incomplete cap-window comparison.
Local speed smoothing and one rescued lap do not erase those failures. Preserve
this negative result; no coefficient adjustment, deadband/hysteresis, timer, FP
boundary repair, expanded A/B, large generalization or official action follows.
