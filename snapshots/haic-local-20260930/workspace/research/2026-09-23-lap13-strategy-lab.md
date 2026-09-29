# 13초 기록 단축: 아이디어·경쟁 실험 운영

2026-09-23. 총괄 대화: 01a0bdf1-f514-71c2-9bbd-afb3eb98f44d.
사용자 요청: 실제 대화끼리 협업, 실행 담당 Luna max, 총괄은 아이디어 담당. 리뷰 요청 없이 진행한다.

## 확인된 증거와 범위

- fetch한 origin/main cd49862는 DrQ-v2/Dreamer 연구와 docs/ 중심 문서 구조를 사용한다. 로컬 미커밋 PPO 연구와 동일 버전으로 취급하지 않는다.
- 원격 current-state에는 DrQ-v2 control의 확인 평가 완주가 4~7/32, Dreamer pilot은 조향 포화 실패로 기록되어 있다. 원시 결과 재현 전에는 문서에 기록된 결과라고만 표현한다.
- 로컬 scratch_official_summary2.json은 Track1 seed42 및 추가 장애물 조건에서 미완주를 기록한다.
- tmp/lap-time-ceiling.json은 15초 목표의 직진 진단이다. 곡선 주행이나 최적 경로의 성능 상한 증명이 아니다.
- 중앙선 길이를 단순히 13초로 나누면 T1 seed42 76.46, T2 seed101 81.85, T2 seed107 92.08 simulator-unit/s이다. 타이머 시작, 출발 가속, 실제 경로 길이와 코너 감속을 확인해야 한다. 13초 달성은 아직 미입증이다.
- 5초 act 계산 제한과 시뮬레이션 랩타임은 서로 다른 지표다. actor-only 계약에서 계산 시간을 억지로 채울 이유는 없다.

## 경쟁 전략과 소유권

| 전략 | 실제 대화 ID | 가설 | 첫 판별 실험 |
|---|---|---|---|
| A: 속도와 학습량 | 01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1 | 행동 변환과 학습 예산이 빠른 정책을 제한 | 출력·가속 포화 및 변환 일치 측정, 같은 조건에서 학습량별 곡선 |
| B: 충돌 전 판단 | 01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de | 충돌 이후 정지 자료보다 회피 직전 자료가 필요 | 충돌 전후 데이터 비율 측정, balanced pre-impact BC+PPO pilot |
| C: 빠른 교사 전이 | 01a0cec2-4799-7303-aba2-baa80f015189 | 느린 교사가 학생에게 느린 운전만 가르침 | train에서 더 빠른 유효 trajectory 확보, actor 증류 후 PPO |

각 작업은 독립 worktree 및 고유 결과 경로를 사용한다. 공유 RULES/RESULTS/SOTA/report를 동시에 덮어쓰지 않는다. A가 공통 평가 계약을 제안하고 총괄이 결과를 모은다. 상대 전략의 유용한 발견은 전달하되 첫 인과 실험에 여러 변경을 합치지 않는다. 제출은 PPO actor-only이며 교사·privileged state·경로 최적화는 훈련 전용이다.

## 가장 먼저 검증할 새로운 가설

1. **허용 가속과 실제 탐색은 다르다.** 현재 확장식 `gas = MAX_GAS*k*tanh(u/k)`는 작은 u에서 `MAX_GAS*u`이다. k를 키워도 기존 작은 출력은 거의 유지된다. 최대 행동값만 넓힌 실험은 빠른 행동을 탐색한 실험이 아닐 수 있다. 행동 분포와 gradient, rollout/eval 일치부터 측정한다.
2. **회피를 시작해야 하는 시점에 학습 신호를 준다.** train의 충돌 전 0.5~2초 구간을 정상 구간과 균형 있게 수집한다. 정지 프레임 반복이 회피 사례 수로 부풀려지지 않게 episode 단위로 센다. 기존 DAgger 데이터의 실제 분포를 확인한 뒤 적용한다.
3. **한 구간의 빠른 행동은 다음 구간까지 평가한다.** 코너 진입 직전 속도만 보상하면 출구에서 손해를 볼 수 있다. 섹터 진입부터 다음 출구까지 시간과 실패율을 함께 측정한다. 구간 타이밍은 훈련·진단용이며 비공개 평가 경로 정답을 actor에 넣지 않는다.
4. **critic만 더 잘 보게 하는 후속 후보.** train simulator state로 critic의 가치 추정을 돕고 actor는 허용 이미지 입력을 유지한다. B의 데이터 구성 실험과 별도 ablation으로 다룬다.

## 공통 실험 계약

### A 공통 screen v1 수신

**A continuation 완료 회신:** base/+4096/+8192 공통screen완주2/4,2/4,3/4; 완주중median24.720,25.030,25.860초. 신규완주T2/132는26.360초collision1,기존완주T1/74 24.260→25.200초/T2/133 25.180→25.860초. tune완주4/6→1/6→1/6. 따라서screen완주증가만으로승격하지않고현재reward continuation 확대보류. 보상변경confound/1trainingseed경계유지. +4096hash1d423ea4e07d1832a60c4c047bf73c9eee07b36bf4d30266ea52164e672a8791, +8192hash380cf8bfad8a70afb4171c552b29860de1cb770c82695ef6bb3307965d9611d3.

**후속가설: 업데이트 일부만 보존.** A에base↔last8192 actor가중치alpha0.25/0.5/0.75 세개만보간하는bounded평가배정. 동일architecture/관측/행동변환/버퍼검증,optimizer평균금지,최종단일actor. tune6에서먼저판정하고유망1개만공통screen4. 양끝대비개선없으면촘촘한추가탐색없이기각. [Model soups, Wortsman et al.2022](https://arxiv.org/abs/2203.05482)의fine-tuned 가중치평균아이디어를차용하나논문은HAIC/RL성공근거가아니며본실험가설이다. 학습run 추가는하지않는다.

**학습량실험 해석변경(A provenance회신):** Adam state는저장/복원가능하나 RNG state미저장, seed8101로새설정. selected metadata는 safe_speed_target50/weight0.1이며현재trainer target70/shortfall0.6/curve0.4/brake6/recovery5+4와달라 당시reward동일성을보장못함. 따라서 pure-dose가아닌 current-reward continuation pilot. baseline0→추가학습은보상변경과학습량혼합,4096→8192는같은현재reward/동일경로의추세로만본다. 당시reward추정복원을선행조건으로두지않고현snapshot고정후실행. D의selected평가trace bonus계산도현재reward를적용한반사실이며checkpoint의과거학습보상증거가아니다.

A baseline4cell 완료 회신: T1/73 DNF progress0.49064 collision1; T1/74 clean24.260초; T2/132 DNF progress0.63141 collision5/damage1; T2/133 clean25.180초. 완주2/4, 완주중median24.720초/p9025.088초, 전체progress평균0.77447, 13초미만0/4. 총1078 decisions gas최대0.09233, 95%cap도달0. 단일screen이며 확인평가 아님. 각원본은 A diagnostics/t1-seed73,t1-seed74,t2-seed132,t2-seed133에 있음(본 집계는 A회신 기준).

학습량pilot 재개 계약: Adam/RNG 미저장이면 exact resume 복구 구현을 선행조건으로 두지 않고 selected weights+fresh optimizer/고정RNG에서8192회 단일실행,4096/8192 last snapshot을 비교한다. optimizer초기화 효과와 순수학습량을 완전분리하지 못한다는 제한을 기록한다. 기존변수 고정,13초미만 비율도 포함. 실행 준비가 계속 확장되지 않도록 A에 전달했다.

A 같은trace 후처리 회신: 학습된 Normal sigma steer0.350/longitudinal0.400은 상태별 mu의std와 별개다. 검사용shadow samples는 환경에 미적용. collision step153 직전10결정 speed평균46.42, progress0.4494→0.4869, gas평균0.0555; 직후10 speed평균1.07, progress0.49064고정, gas0.0857/brake0. 따라서 전체평균22.30을 정상주행의 cruise속도로 해석하면 안 된다. provenance있는 현재actor의 충돌후정체 증거이나 보상누수로 학습했다는 인과증거는 아니다. D에 전달하되 screen seed73을 train으로 옮기지 않고 train내 관련 상태 노출을 다루도록 했다.

첫 계측 결과 원본 `C:/Users/koi/.codex/worktrees/strategy-a-ppo/HAIC/artifacts/haic/strategy-a/diagnostics/t1-seed73/summary.json` 직접 확인: selected hash `3ceb5ebbe2a4bd96944649a693b8ef924c0252fa620b0fd547c7fb494afff199`, k1.0. T1seed73 254decisions DNF/off_track, progress0.4906367, collision1/damage0.2. gas평균0.07685/max0.09232, 95%cap 도달0/254. eval/rollout 행동변환 오차0, inverse logprob약1.43e-6. 이 사례에서 상한포화가 병목이라는 증거는 없다. 시간축 raw u std와 학습된 탐색 Normal std는 구분한다.

A 후속 실행: 나머지3개 screen baseline 후 같은 selected checkpoint/고정설정에서 추가4096/8192 decisions의 연속 학습량 pilot으로 전환. 0/+4096/+8192의 last checkpoint를 고정비교하며 best-selection과 혼동하지 않는다. optimizer resume/reset 및 누적예산을 명시. 첫1개 training seed는 탐색이고 유망시 반복seed 확대. C/D 수정은 섞지 않는다. 충돌전후 구간분리는 기존 trace 후처리로 수행하여 계측 추가가 학습 실행을 지연시키지 않는다.

원본 `C:/Users/koi/.codex/worktrees/strategy-a-ppo/HAIC/artifacts/haic/strategy-a/contract-v1.json` 직접 확인. T1 seeds73/74, T2 seeds132/133, 각600 decisions, actor-only, 공식 장애물 유지 및 추가 장애물 없음. 내부 screen/tune이지 blind/confirmation이 아니다. 서로 다른 훈련조건에서도 동일 환경·허용 관측·runtime으로 평가한 actor들의 최종 시스템 성능은 비교할 수 있다. 단일 알고리즘 효과나 동일 예산 학습효율을 주장하려면 lane 내 matched 대조가 별도로 필요하다. teacher는 actor 후보 순위에 포함하지 않는다. 이 해석은 현재 개별 pilot 조건을 변경하지 않는다.

체크포인트 provenance 정정(A 조사 회신): selected checkpoint saved global step19456, source15360에서+4096 decisions이며 실행 전체는8192 decisions/8updates 완료했다. 따라서 '실행에서 학습한 총량'과 '선택된 파일에 반영된 학습량'을 구분한다. 기존 step23552 언급은 이 selected 파일의 provenance로 사용하지 않는다. 전체 hash는 A의 원본 산출물을 따라 기록한다.

- 같은 환경 revision, map/seed/장애물, 완주 판정, timer, frame skip, action 계약을 freeze한다.
- 이미 반복 사용한 holdout은 진단/tune로 명확히 표시한다. 신규 최종 확인은 선택 후 한 번만 수행한다.
- 학습 충분성은 step 숫자 하나가 아니라 checkpoint별 완주율·시간·충돌·정체·학습 seed 편차로 판단한다. actor/critic loss만 내려가는 것은 충분한 학습의 증거가 아니다.
- 최초 pilot 이후 유망한 후보에 같은 추가 env-decision 예산을 부여한다. 1x/2x/4x 비교의 기준 checkpoint와 누적량을 명시하고 실제 walltime도 보고한다. 가능하면 최소 3개 학습 seed로 재현하되 예산과 실행 여부를 별도 표시한다.
- 미완주를 누락하지 않는다. completed, lapTimeMs, progress, collisions, damage, retire_reason, act latency, source/model/package hash, train decisions를 원본 episode별 저장한다.
- 핵심 목표 지표는 전체 평가 중 `completed && lapTimeMs < 13000` 비율, 전체 완주율, 완주 시간 분포다. 공식 점수와 자체 평균지표는 구분한다.
- 빠른 후보와 안정 후보를 둘 다 유지하여 Pareto 비교한다. 쉬운 맵 중앙값으로 공식 장애물 맵 개선을 주장하지 않는다.
- 장기 학습 전에 작은 판별실험을 한다. 같은 실패를 반복하면 예산 확대 대신 가설을 바꾼다. 실패·불확실 결과도 남긴다.

## 문헌과 적용의 경계

- [Pinto et al., Asymmetric Actor Critic (2017)](https://arxiv.org/abs/1710.06542): critic에 full state, actor에 이미지 사용. HAIC에서 효과가 있을지는 실험할 가설이다.
- [Ross et al., DAgger (2011)](https://proceedings.mlr.press/v15/ross11a.html): policy가 방문하는 상태 분포에서 모방 자료를 보강하는 근거. 충돌 전 구간 균형화는 이 프로젝트의 추가 가설이다.
- [Kaufmann et al., Swift (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4): simulation RL과 지각·동역학 보정으로 고속 racing을 달성한 사례. 자동차 13초 달성을 보증하지 않는다.

## 진행 상태

A/B에 Luna max 설정으로 지시 전달 및 active 상태 확인. C는 별도 worktree 대화 생성 후 시작 메시지 수신. 본 문서는 실험 설계와 조율 기록이며 새 성능 개선 측정 결과가 아니다.

## 추가 작업 생성 권한과 반복 운영

사용자는 2026-09-23 후속 지시에서 아이디어를 계속 발급하고 기존 대화 수가 부족하면 Luna max 실제 작업을 추가 생성해 반복하도록 허용했다. 담당이 없는 독립 가설이 있고 기존 담당들이 수행 중이면 새 작업을 만든다. 완료된 담당의 재사용을 우선하고, 새 작업은 독립 worktree·고유 결과 경로·실험 예산·중단조건·공통 평가 계약을 받는다. 생성 후 ID와 담당 가설을 이 문서에 기록하고 결과를 총괄로 회신하게 한다. 같은 가설의 중복 실행과 훈련 자원 경합을 피한다. 30분 주기 조율 자동화 `haic-13`에 이 권한을 반영했다.

## 2026-09-24: 실제 진행과 추가 가설

- A/B는 초기 조회에서 active지만 최근 turn items가 비어 있어 산출물 및 blocker 회신을 요청했다. 이후 A는 독립 worktree `C:/Users/koi/.codex/worktrees/strategy-a-ppo/HAIC` 생성과 한 에피소드 계측 준비를 회신했다. 새 학습은 아직 시작하지 않았다고 명시했다. B의 독립 worktree도 git 목록에서 확인됐으나 결과는 미수신이다. active 표시는 학습 진행의 증거가 아니다.
- C는 목표속도 62→80만 바꾸는 train-only teacher pilot을 수행한다고 회신했다. 기존 safe25.32s/fast23.12s는 C가 과거 결과에서 읽어 보고한 값이며 단일변수 비교가 아니다. 새 pilot의 완주·속도·가속포화 결과를 기다린다.
- D 생성 요청: Luna max, `client-new-thread:c68f571a-3f5c-4e46-ada2-a4cb9c08f518`. 실제 threadId 수신 후 등록한다. 보상 분해와 정체 행동의 반복 수익 가능성을 담당한다. 생성 접수와 실행 성공은 구분한다.

### D: 행동 자체의 보상이 목표를 방해하는가?

#### 01:16 train 노출 확인 및 재개

train_exposure_diagnostic.json 직접확인: 원래train manifest의중복geometry제거최초4조건,총1044decisions. custom-track-aug0-v0-obs에서충돌후101연속zero-progress recovery구간검출,나머지3조건gate0. 일부는같은원본map family변형이므로독립map family4개라고부르지않는다. D에 selected checkpoint의관측/초기weights/Adam/RNG를공통으로유지하고 노출map+정상train map 혼합의action vs potential 각4096decision/첫seed1개 bounded pilot을배정. deterministicprobe노출과stochastic학습노출을별도계수로검증. 기존무노출pilot과다른protocol로기록하며 실패map선택편향을명시하고일반성은별도tune에서판정. 화면평가조건을train으로전환하지않는다.

#### 00:56 무노출 pilot 종료와 후속 조건

D result.json 직접 확인:4run×2048=총8192 decisions, recovery조건/zero-progress gate 모두0. 결과분류 unexposed_inconclusive이며 보상수정의 학습효과를 판정하지 않는다. selected actor T1seed73의 충돌직후10transition에서는 실제gate10/10, 진행0, 기존actionbonus합4.01230/할인합3.83288로 계산됨. 이것은 해당평가궤적의 보상진단이지 학습원인의 증명은 아니다.

후속은 학습량 확대가 아니라 train-only 노출진단: selected checkpoint 원래train partition을 manifest에서복원해 최대4조건×800decisions 또는 기존동일출처trace로 gate노출을 확인. 평가/tune/heldout을 train에 옮기지 않는다. 노출있는train조건을확보한후 양팔동일교과과정/초기값/예산으로 실험가능. 4조건에서도노출0이면 해당학습효과실험은 보류하고 재개조건만 남긴다. 현재 C/A 학습은 변경하지 않는다.

#### 00:25 실측 근거와 후속

D 실제 대화 ID `01a0ceca-0f09-7571-a83e-6d1f80ffe655` 확인. 격리 경로 `C:/Users/koi/.codex/worktrees/strategy-d-reward/HAIC`. 총괄은 `experiments/strategy-d-reward-potential/analysis.md` 및 protocol.json을 직접 읽었다. D 보고에 따르면 plus-obstacle trace164개 transition이 시뮬레이터 재현과 일치한다. step64~164의101회 무진행·비충돌이벤트 구간에서 recovery 지급101회, 할인 bonus+25.160 및 모든 비용 포함 total+15.590. 잠재함수 대조는 동일 궤적 반사실에서 total-10.423. controller provenance가 없고 진단 map이므로 현재 actor가 이를 학습한 증거는 아니다.

D는 default train(1,101)/(2,102), tune(3,201), visual_features=false, policy seed2개의 각2048 decisions/4updates pilot을 수행한다. 이 레인은 현재 PPO 후보나 C teacher와 직접 순위 비교하지 않는다. train에서 recovery/무진행 보상 노출을 세어, 노출0이면 무차이를 가설 기각으로 오해하지 않도록 요청했다. 정상 구간과 정체 구간의 시작상태가 달라 return 차이를 최적정책 우열 증명으로 취급하지 않는다. 다음 유망한 단계에서는 recovery bonus=0 대조로 기존 bonus 제거효과와 potential shaping 효과를 분리한다. 현재 pilot 팔은 추가하지 않는다.

C 최신 command marker 실패 및 gas 결과 파일 미생성 확인으로 최초 오류 회신을 요청했다. A/B에는 같은 진행 요청을 재발송하지 않았다.

C 후속 정정 회신: gas pilot은 실행 실패가 아니라 아직 실행 전 runner 수정 단계였다. 최근 apply_patch는 성공했으며 실행 프로세스는 없다고 보고했다. 따라서 이전 '실험 진행 중' 표현을 실행 준비 중으로 정정한다. 총괄이 추가한 분석 요구가 선행 작업을 늘리지 않도록, frozen protocol에 필요한 최소 수정 후 먼저 주행하고 paired/transfer 분석은 원본 episode JSON 후처리로 분리하도록 전달했다. 실행 시작·첫 episode·완료 산출물로 진척을 구분한다.

C 실행 시작 회신 수신: session81310, actuator-gas-v1-run.py, maxgas0.12 vs0.24, 기존4조건씩 총8회, max800 decisions. runner hash `421D4FF025840CCAF15306284672BAD64837791DEEF28E57732A022F847C9BA6`. benchmark는 전체 episode 완료 후 JSON 저장하므로 현재 결과 파일이 없는 것만으로 정체/실패로 판단하지 않는다. 세션 polling은 C가 소유하며 총괄이 중복 실행하거나 설정을 추가 변경하지 않는다. 완료 여부는 아직 미확인이다.

조율 자동화는 10분 간격으로 변경했다. 상태 조회로만 종료하지 않고 증거 기반 가설·반증조건·후속 실험을 진전시키도록 지침을 추가했다. 진행 중 작업에는 같은 지시를 반복 발송하지 않는다.

로컬 train_policy.py의 shaping은 장애물 위험×제동 비율에 +6, 손상 후 저속·비충돌일 때 가속 비율에 +5 및 조향 크기×가속에 +4를 준다. 이 항목 자체에는 실제 전진 증가 조건이 없다. 이론적으로 행동 반복이 완주보다 유리해질 여지가 있으므로 실제 trace로 각 항목과 할인 누적값을 분해한다. 모든 조건의 동시 도달 가능성과 다른 패널티를 확인하기 전에는 exploit 확정으로 쓰지 않는다.

후보는 단순 행동 보상을 제거하는 단일변수 실험, 이후 `gamma*Phi(next)-Phi(current)` 형태의 진행/위험개선 shaping이다. terminal/truncation 처리 및 원래 목표 정의를 확인해야 잠재함수 방식의 정책 보존 논리를 적용할 수 있다. 근거: [Ng, Harada, Russell 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

### B 후속: 좌우 정답을 평균 내어 직진을 배우는가?

#### 00:46 총괄의 optimizer membership 직접 계측

`research/2026-09-24-bc-optimizer-coverage.json` 생성. 현재 imitation source hash4f299fa5...32c64에서 actor_parameters AST 표현을 실제 fresh model 객체로 평가해 membership 확인. 두feature 활성시 visual weight224/bias32, temporal weight128/bias128 모두 requires_grad=true이나 BC optimizer 미포함. temporal모듈은 전부0 초기화. 훈련이나 성능평가가 아닌 객체 membership 계측이며, BC에서 해당 모듈이 업데이트되지 않는 구조는 확인했다. B에 optimizer 포함만 바꾸는 paired BC를 배정; 선택checkpoint temporal 비활성이면 visual-only를 primary로 한정한다. C의 gas label 변경은 섞지 않는다.

같은 주기 D의 첫control 학습 산출물이 존재하고 회신상 recovery gate 노출0이다. 이는 보상누수 가설 기각 근거가 아니며 전체 pilot의 노출 계수를 확인한 뒤 후속 결정한다. 진행 중인 D/C 실행은 변경하지 않았다.

중앙 장애물에서 좌우 회피가 모두 좋은데 BC가 단일 평균 행동을 회귀하면 직진을 낼 수 있다는 가설. 실제 유사 관측군의 teacher 조향 분포와 학습 손실을 확인하고, 좌우 선택을 일관되게 한 자료와 비교한다. 현재 정책에서 이 현상이 실제 원인이라는 증거는 아직 없다.

추가 코드 증거: imitation.py의 BC 손실은 pretransform action에 대한 MSE이다. 전체 조향 histogram만으로는 좌우 커브를 혼동하므로 유사 관측과 도로 곡률을 통제해야 한다. 또한 BC optimizer의 명시적 actor_parameters에 visual_feature_encoder와 temporal_feature_encoder가 포함되지 않는다. 활성 모델에서 의도적 freeze인지, feature 사용 여부 및 실제 업데이트를 확인하도록 B에 전달했다. temporal encoder는 networks.py에서 zero 초기화되므로 해당 모듈을 활성화한 실험의 BC 학습 여부가 특히 중요하다. 아직 성능 저하의 원인으로 확정하지 않는다.

### 추가 관측: 제동 후 위험구간에서 다시 가속

scratch_trace_official-track1-seed42.json에서 step57/58은 brake0.056/0.040, step59~61은 brake0이며 gas가0.0034→0.0388로 증가한다. 첫 collision은 step62다. 조향은 step55~61에서 약-0.09~-0.37로 존재하므로 해당 trace를 '각도를 전혀 안 꺾음'으로 설명할 수 없다. 아직 모델 hash와 이미지 시점이 연결되지 않았으므로 현재 SOTA 인과 증거로 쓰지 않는다. 고속 학습 전 제동 유지·관측 타이밍·좌우 선택을 분리해야 한다.

### 01:47 결과 회수 및 다음 실행 연결

C `physical-longitudinal-loss-bc-v1/result.json`과 `tune-diagnostic-correction-v1.json` 직접 확인. 같은472 자료 hash 재현, BC3epochs/동일초기화에서 physical pretransform control은 progress0.62963,미완주,collision5,최종damage1. normalized longitudinal loss는20.380초완주,progress0.98354,collision3,최종damage0.6. 양쪽13초미만0/1,clean완주0/1. 최초 damage 집계는 누적값을 매step 합산한 오류였으며15.6/109.8을 최종값1.0/0.6으로 정정했다. 모델 재학습은 이 정정에 포함되지 않는다. treatment checkpoint41a38b70bd7e66fe6e527c407314bd2cd9d7e9ddb343473eb179f262b1c16df8. 실패에서 완주로의 신호가 있으나 기존.75 BC18.88초clean보다 우월하지 않다.

C에 동일기존계약 PPO8192 상한/8updates/lr2e-6/seed8101 후속을 배정. normalized BC checkpoint에서 출발하고 기존 Adam 복원방식·split을 유지하며4096/8192 last를 평가한다. B optimizer나D reward 변경은 혼합하지 않는다. 기존physical/.75 PPO와출처가일치하는범위만비교하며추가배증하지않는다.

D `matched_resume_pilot_result.json` 직접확인. 두팔각4096decisions/4updates,episode order matched. action gate330/4episodes, potential286/4episodes로노출확인. 단일tune은양팔1/1완주,23.320→23.120초,progress동일0.98551,13초미만0/1. 단일seed·실패조건을선택한train분포이므로일반성능확정불가. 현재tune집계에충돌/손상이없어총괄은성능승격하지않는다.

D후속은새학습없이두final모델을같은tune+train2에서팔당최대3episode/600decisions로진단하고기존동일결과는재사용. 충돌·최종누적damage·완주·시간·정체종료원인을회수한다. gate총량감소만으로회복향상이라판정하지않는다. A보간은진행유지,B는idle/산출물미확인으로중복요청없음.

### 07:38 runner 작성 진척과 reference 오차 좌표 해석

C의 `physical-longitudinal-loss-bc-mean-anchor-ppo-v1-run.py`가07:38생성돼구현진척을확인했다. 파일에는지정teacher-only복구checkpoint,4096decision/팔,계수.1,별도torch.Generator reference인덱스,초기mean/decoded drift0검사가들어있다. 이름의physical-longitudinal 접두어는기존연구계열명이며초기모델이과거physical-label/normalized-loss후보라는뜻은아니다. 최종provenance는지정a9cfe42파일hash로구분한다. 아직학습결과파일은없고진행중작업을재시작하지않았다.

보존손실은tanh이전2좌표MSE이므로같은손실값이같은물리행동변화를뜻하지않는다. steering=tanh(mu_s)의국소민감도는1-tanh²(mu_s),gas/brake도branch별scale과포화정도에따라달라진다. 따라서기록예정인pretransform drift와decoded행동오차를함께읽어야한다. 특히포화영역mean변화는커도실제행동변화는작을수있고,제동/가속경계근처에서는작은변화로branch가달라질수있다. 이번에는좌표/손실계수를추가변경하지않고현재대조의메커니즘해석에만적용한다.

### 07:28 reference 관측 정규화 경계 확인

C는reference NPZ uint8와ROOT model normalized float 입력차이를확인중이며학습미시작이라고보고했다. 총괄이ROOT training/imitation.py333~336의기존변환을확인해uint8→float/255,normalized float→그대로라는계약을전달했다. 실제C cwd는4fd3로확인됐고새runner산출물은아직없다. D는종료·대기,A/B변화없음.

동일state_dict에서reference drift가발생하면정책학습효과가아니라입력경계문제일수있다. 특히한쪽uint8을그대로float캐스팅하고다른쪽만255로나누면이미지값이255배달라져보존손실이정책을잘못된출력에붙잡을수있다. 양쪽이같이잘못정규화되면초기drift0검사만으로는잡히지않으므로dtype/값범위와기존BC변환경로도함께확인해야한다. normalized float의이중255나눗셈도피한다. 이는계약확인이지새성능개선가설이아니며관측표현변경을실험변수로추가하지않는다. 입력경계해결후기존계수/예산의학습을진행하고재수집하지않는다.

### 07:18 행동보존 손실의 작동 여부와 성능을 분리

C는새대조active이나기존4fd3/experiments/strategy-c조회에는새protocol/학습산출물이아직없다. 배정초기단계로실행성공을단정하지않고중복지시하지않는다. D는AR1 paired4행/changed_inputs=[]/same_noise_schedule_prefix=true/visited_state_distribution_matches=false를다시직접확인후기각·대기회신했다. A/B새결과없음.

현재reference-mean MSE는초기정책과reference가동일하므로초기값과gradient가0이어야하며업데이트가진행돼평균행동이달라질때만복원방향gradient를만든다. 또한이손실은policy_log_std를직접벌하지않으므로평균행동오차감소가탐색분포전체보존을뜻하지않는다. 이를KL제약이나안전장치로표현하지않는다. 원teacher472관측밖의방문상태에도같은보존효과가있다는보장은없다.

판별표: (1) reference drift감소+주행개선이면동결행동보존가설을지지하는pilot; (2) drift감소+주행퇴행/동일이면보존기능은작동했으나성능효용미확인; (3) drift차이없으면이번계수/예산의개입이충분히작동했는지불명확하며성능차이만으로메커니즘확정금지. 초기drift가0이아니면학습설정·관측정규화·CPU thread·reference계산오류를먼저구분한다. scalar손실값만으로gradient지배를판단하지않고,한seed/tune결과를일반화하지않는다. 현재실험조건은바꾸지않았다.

### AR1 완료 기각 및 C PPO 행동보존 대조 배정

AR1 최종 result.json 직접확인:4/4실행1007decisions,전부off_track/완주0/13초미만0/collision0/damage0,changed_inputs=[]. paired진행은map1/8101 .4634→.0447,8102 .3130→.4268;map2/8101 .5378→.2948,8102 .5857→.4821. 충돌없이일찍이탈한결과를안전개선으로보지않고현rho/진폭후보기각. 분산축소·시간상관계수추가탐색없이D대기.

다음C는PPO업데이트중기존BC행동보존가설을담당한다. [Rajeswaran et al., RSS2018](https://arxiv.org/abs/1709.10087)의시범자료활용은동기이며본고정reference-mean MSE대조는DAPG재현이아니다. 기존보간은업데이트후가중치를섞었으나이번에는업데이트중동일train관측에서동결BC의출력을보존하는추가손실만비교한다. base와같은결함을고착할수있으므로실험전효과를가정하지않는다.

초기정책teacher-onlyBC복구a9cfe42…(원9289b1),양팔freshAdam/lr2e-6/seed8101/ROOT현재보상·표준PPO/원sigma. 원train2맵같은일정,각4096decisions(1024×4updates),총8192. 고정reference는같은초기BC,원teacher472관측에서2차원pretransform mean을no_grad로미리저장. treatment loss=L_PPO+.1*mean((mu_current-mu_reference)^2),control추가계수0. minibatch마다별도고정seed의32reference행을같은순서로조회하여PPO/RNG소비를섞지않는다. 새로운teacher/자료수집/가속상한/분산/value detach변경없음. 전체PPOoptimizer membership/gradient clip등양팔동일,추가reference연산시간별도기록.

각 update의 동일 472관측mean drift/decoded행동오차를기록해추가손실이실제로행동변화를제한하는지판별한다. 마지막4096checkpoint만동일tune400결정각1회평가,총새환경8992이하/45분,중간best선택없음. 완주/13초미만/충돌/최종damage/시간/종료원인/hash/설정/provenance저장. 양팔 실패나 동일 퇴행이면계수탐색없이종료,좋은단일tune도독립성능증거는아님. 제출은 단일 actor이며 reference는 훈련 전용이다. 실제source변경은고유runner범위로격리하고root파일은수정하지않는다.

### 07:08 입력검사 원인 해소 및 지연 보상 가설 범위

D회신상검사실패는기존protocol의map2 SHA hex대소문자표기차이였고정규화후같은digest다. schedule/trace/checkpoint/source/split도일치했다고보고했다. 기존protocol은보존하고새기록에정규화값을남기며주행을준비중이다. 파일내용변경증거나기존결과무효사유로취급하지않는다. 이번조회에는새episode결과가없고중복실행지시도없다.

총괄이ROOT PPOConfig 기본값gamma=.99,GAE lambda=.95와train_policy가이를반환/advantage계산에전달하는코드를확인했다. .08초결정간격에서GAE의직접TD-residual가중계수는(.99*.95)^k로,약1.30초에1/e수준이된다. 이는critic의bootstrap가먼미래가치를전달하는경로를포함한'정책의전체예측시간'이아니다. 특정과거run이기본값을썼는지는각run metadata로확인해야한다.

향후보상시점가설을선택한다면현재train trace에서회피조향시점과충돌/완주신호사이의거리,그구간value오차/advantage를먼저계측해야한다. 단순히lambda를높여오래생각하게한다는해석은부정확하며분산증가로악화될수있다. 현재관측된noise주행퇴행은학습update전에도나오므로이보상신용할당가설만으로설명할수없다. 따라서당장GAE변경을배정하지않고기존AR1진단을마친뒤의별도가설후보로남긴다.

### 06:58 시간상관 진단의 반증 조건 보강

D는 기존입력 재사용 preflight의 integrity assertion이 실패해 환경실행전에 원인을 조사중이라고 보고했다. 새4회주행결과는없고현재체크를완화하거나중복실행하지않는다. A/B/C는새결과없음. 실패assertion의정확한원인/수정은담당결과수신후기록한다.

현재rho=.8/결정간격.08초의이론적noise성질을계산했다. 단위분산stationary AR1의인접차분분산은2(1-rho)=.4로iid의2보다작지만,10결정(.8초)누적noise합분산은iid대비약5.4295배다. 상관이1/e로감소하는시간은약.3585초다. 즉순간방향전환을줄이는동시에한쪽방향편향이누적되기쉬워진다. 이값은tanh이전noise의수학적성질이며실제조향/차량궤적/충돌확률예측은아니다.

후보가실패하더라도'smooth exploration은효과없음'으로일반화하지않고,이진폭·지속시간의시간상관이현재동결policy에서불리했다고분류한다. 반대로steering increment만작아지고완주/진행이퇴행하면성공으로세지않는다. 현재rho대조가끝나기전진폭동시조정이나rho검색을추가하지않는다. noisy조향누적과정책mean의보정은서로작용하므로rawnoise통계와실행action통계를분리해읽는다.

### D 탐색노출 완료: 진폭 축소 혼합 결과와 시간상관 대조

총괄이 최종 result.json 10행 직접 확인. 2540decisions/10회완료,deterministic1/2(맵2 clean19.16초),원sigma0/4,half-steer1/4(맵1/noise8101 clean23.36초),13초미만0/10. half-steer 맵1/noise8102는progress.5/collision2/off_track,맵2/noise8101은.250996/collision5/crash,8102는.474104/collision0/off_track. 원sigma의맵2대응진행.537849/.585657보다양쪽악화됐다. 따라서조향분산축소를안정개선으로채택하지않는다. top-level episodes_started는최종10으로정리됐고중간0은최종집계에사용하지않는다.

다음독립가설은잡음진폭이아닌시간상관이다. [Raffin/Kober/Stulp, Smooth Exploration for Robotic Reinforcement Learning, PMLR2022](https://proceedings.mlr.press/v164/raffin22a.html)는상태의존탐색과noise재표본주기가부드러운탐색에기여하는방식을다룬다. HAIC효과의증거가아니며아래AR(1)진단을gSDE재현이라고부르지않는다.

D후속4회만배정:같은동결control/ROOT환경/원sigma두축/맵·noise seed/400결정상한을유지하고조향epsilon만stationary AR(1) rho=.8로변경. 저장iid epsilon에서 z0=epsilon0, zt=.8*z(t-1)+.6*epsilon_t;longitudinal은기존iid열그대로. 모든분기에서정책mean은매결정다시계산하고최종행동평활은하지않는다. 이론적무조건부noise분산1보존(유한trace실측분산동일을보장하지않음). 기존원sigma4행은대조로재사용하고half-steer/결정적은참고. 새환경최대1600decisions/4episode/20분,학습0/teacher0. rho추가탐색없음. 동일seed대응완주/시간/충돌/진행및noise lag1상관/증분을기록한다.

긴편향이탈선을늘릴수있어부드러움자체를성능개선으로판정하지않는다. 성공해도時系列noise를현재독립Normal PPO logprob에그대로끼워넣지않는다. 이후학습에사용하려면조건부분포/latent noise 처리와logprob일관성을별도설계해야한다. 현후속은실제주행노출만분리하는진단이고제출runtime수정은없다.

### 06:48 D 최초 주행 원본 확인: 결정적 기준 재현

2d04/research/strategy-d-exploration-exposure/result.json의running snapshot을직접읽었다. 완료episode 요약4개/누적969decisions이며잔여6회는진행중이라전체결론은보류한다. top-level episodes_started=0은started=true인4행과불일치하므로중간집계필드를그대로주행수증거로쓰지않고최종보고에서대조한다.

deterministic map1은223decisions/off_track/progress.5/collision1/damage.2로기존A459행수집의해당에피소드종료와일치했다. map2는240decisions/19.16초clean완주/progress.988048이다. 기존수집cap236에서의미완주는이정책의완주불가능을뜻하지않았으며이번늘어난cap에서finish를관측했다. 맵1의실패는탐색noise가없어도발생하므로현재정책실패전체를noise로설명할수없다.

원sigma map1/noise8101은279decisions/off_track/progress.463415/collision0,8102는227/off_track/.313008/collision1/damage.2다. collision0이어도완주·진행이개선된것은아니다. half-steer의대응행과map2 stochastic행은아직없어탐색축소효과를판정하지않는다. 재실행/조건수정없이남은사전일정을유지한다. 공통맵평가나13초성과로승격할근거는아직없다.

### 06:38 D 산출물 위치 정정과 노출 비교 집계 조건

D의새protocol은기존strategy-d-reward가아닌실제task cwd `C:/Users/koi/.codex/worktrees/2d04/HAIC/research/strategy-d-exploration-exposure/protocol.json`에06:36생성된것을확인했다. 총괄의기존경로조회만으로산출물없다고보았던점을정정한다. read_thread의실제fileChange와파일을대조했으며학습/환경source는여전히ROOT로명시돼있다. protocol내d_workspace가기존경로여서D에실제runner/protocol/output경로를일관되게기록하도록전달했다. 작성중작업을재시작하지않는다. 아직episode결과는없다.

진단집계조건: deterministic2회와stochastic팔당4회는분모가다르므로전체완주수만비교하지않는다. 각맵의deterministic1회결과를같은맵의두noise seed와함께보이고,원sigma대half-steer의4개paired cell을중심으로신규완주/상실/공통완주시간/충돌을보고한다. 결정적결과를두번복사해표기할수는있어도독립반복2회로세지않는다. 한맵에서만효과가나면다른맵에도일반화했다고말하지않는다. timeout/crash/off_track의조기종료가충돌누적을낮출수있으므로총충돌감소만으로개선판정하지않는다. 현재동결계약의범위는변경하지않았다.

### 06:28 탐색 분포 변경 크기와 이후 PPO 경계

D는 repaired파일 검증과10episode/noise일정 준비를 보고했고 진행중이다. 이번 파일조회에는새주행결과가아직없다. A는가중치pilot종료회신후대기,B/C변화없음. 기존D실행을재시작하거나중복배정하지않았다.

조향표준편차만절반으로바꾸는것은mean정책을보존하지만확률분포변경은작지않다. 같은mean의1차원Normal공식으로 KL(original||half)=log(.5)+1/(2*.5²)-.5=0.806853 nat,역방향KL=log(2)+.5²/2-.5=0.318147 nat이다. longitudinal분포가같아그축기여0이다. 이는tanh이전분포의해석계산이며경험적주행KL이나손실계측값이아니다. 이상적가역변환에서는KL이보존되지만실제float clipping등을무시한계산임을명시한다.

따라서향후PPO대조를택하면이변경을기존rollout에소급적용하지않는다. 변경된분포로새rollout과old_logprob를수집하고그분포를업데이트시작점으로삼아야한다. old rollout 재사용/예전logprob고정으로실험하면탐색변경과off-policy오류가혼합된다. 현재진단은optimizer0이라이문제가실행중발생하지않는다. 진단성공은좋은수집분포후보의근거일뿐mean정책기록단축이나학습충분성의결론이아니다.

### 가중치 pilot 종료 및 D 조향 탐색노출 실행 배정

총괄이 weighted screen/episodes.jsonl 직접 확인: T1/73 crash/progress.486891/collision5/damage1;T1/74 clean23.92초;T2/132 crash/.628205/collision5/damage1;T2/133 off_track/.986711/collision0/damage0. 완주1/4,13초미만0/4,총collision10/damage2,1180decisions. checkpoint SHA9a4482bac6ace99fe9a3eb43811b4770940c3498a753ea07319603389ab9d93e. teacher-only2/4 대비퇴행이며unweighted v2가잃은T1/74를회복하는대신T2/133을잃었다. 완주집합이다르므로median24.16→23.92를paired시간개선으로쓰지않는다. 계수추가탐색/이후PPO없이가중치pilot종료,A대기.

D에앞선10episode 탐색노출안을배정한다. 선택정책은mixed/weighted가아닌teacher-only BC control9289b1의가중치,검증된metadata복구파일a9cfe42…를사용한다. 해당선택은이번internal screen의기술적상대결과이며SOTA승격아님. 학습환경ROOT/source고정,train2맵각첫seed. deterministic맵당1회+원sigma/조향sigma만half각noise8101/8102×2맵,총10episode각400/전체4000decisions이하,30분. 학습/teacher/heldout없음. noise tensor는별도generator로맵/seed/index별고정하여환경RNG소비와분리한다. 평균가중치와longitudinal sigma는고정한다.

핵심판정은stochastic수집이deterministic기준보다퇴행하는지,조향noise축소가동일noise seed들에서일관되게개선하는지다. deterministic도실패하면noise만의문제라고결론내리지않는다. 원본episode/행동/epsilon/progress/종료원인/충돌/최종damage/완주/시간/13초미만및source/모델hash를저장한다. D가직접수행하고서브에이전트/리뷰없음. 완료후에만PPO대조를선택하며현재는학습량확대없다.

### 06:18 가중치 학습 산출물 확인 및 탐색노출 대조 초안

A의 `bc-state-distribution-dagger-v2-postcollision-downweight-0p1-v1/run.json` 직접 확인: status=trained_root_reference_ready_for_a_strict_parity. 06:18에 checkpoint/row-weights/root reference가 생성됐다. 정체52행/rawweight.1/나머지420행1/정규화472/425.2/minibatch재정규화false가 배정과 일치한다. 이는 학습 산출물 진척이며 strict parity/공통4cell 결과는 아직 미확인이다. A에 재지시하지 않았고 B/C/D는 새 결과 없음.

**다음 탐색노출 대조의 사전 초안(미배정):** A결과를 받은 뒤 하나의 동결 actor를 먼저 지정하고 원train2개geometry에서 deterministic/원sigma/조향sigma만0.5배를 비교한다. deterministic은맵당1회, stochastic은두팔각동일noise seed8101/8102×2맵,총10episode×최대400decisions=4000이하. 새난수seed는맵geometry증가로세지않으며map seed는기존20260920/20260921고정. raw epsilon을맵·noise seed·결정index로정해팔사이공통난수를유지하되방문상태가달라짐을명시한다. optimizer/teacher실행0,모델평균과longitudinal sigma고정. 이것은성능승격이아닌학습수집분포진단이다.

원sigma가deterministic보다퇴행하고조향축소가두noise seed에서같은방향으로회복하면동일예산PPO대조의근거가된다. 모두미완주이거나seed간상충하면이노출pilot만으로탐색축소를채택하지않는다. deterministic부터실패하는맵은탐색noise없이도정책결함이있다는대조로남긴다. 조향축소가진행을느리게해충돌을피하는지완주/시간/진행/충돌/손상을함께보며,13초미만지표도기록한다. A의현재주행자원과경합시키지않고결과후필요할때만배정한다.

### 06:08 가중치 대조의 실효 자료 구성 명시

A는 새 가중치 실험 배정 후 초기 단계이며 이번 조회에서 새 산출물은 아직 없다. D의 완료 cursor23을 수신했고 앞서 받은 결과와 같아 추가실험 지시하지 않았다. B/C 변화 없음.

배정한 가중치의 합을 계산했다. 정규화 배율은472/425.2=1.11007, 정체52행의 전체 loss weight 비율은52/472=11.017%에서5.2/425.2=1.223%가 된다. teacher 비율은50%에서55.503%, learner는44.497%로 바뀐다. map1 비율도230/472=48.729%에서183.2/425.2=43.086%,map2는56.914%가 된다. 행 수와 표본순서가 같아도 실효 자료 혼합비는 유지되지 않는다.

이는 의도된 정체 downweight의 결과이므로 현재 실험을 바꾸지는 않는다. 다만 개선하면 '정체 라벨 자체가 해로웠다'와 'teacher/map2 비중 증가가 유리했다'를 이 한 대조로 완전히 분리할 수 없다. 해당 결론이 후속 선택에 중요해질 때만 출처·맵별 총가중치를 보존하는 층별 대조를 별도 고려한다. 지금 추가 팔을 만들거나 계수를 탐색하지 않는다. 실패하면 단일계수 실패를 모든 learner-state 자료 보강의 불가능으로 일반화하지 않고 이 bounded 가설을 종료한다.

### A 공통 screen 완료: mixed v2 기각, 정체 가중치 단일 대조

총괄이 resumed/episodes.jsonl 8행을 직접 확인했다. teacher BC는2/4완주,총collision12/damage2.4,완주중median23.59초; mixed v2는1/4완주,collision11/damage2.2,median24.16초. 양쪽13초미만0/4. T1/74는23.88초clean완주에서progress.35417/collision2/off_track으로 퇴행했고, 유일한공통완주T2/133도23.30→24.16초(+.86),collision2→3으로 악화됐다. T1/73 collision5→1은 여전히DNF(crash→off_track)이며 총충돌 감소를 안전개선으로 승격하지 않는다. v2의단일tune개선은공통screen으로이어지지않아현후보를기각,추가PPO없음.

복구파일 SHA: control a9cfe42a5f2f4149b0cfc439f88dc76f79096e86d1b642b2e0f521f7dcab304b, v2 a96f27fbd2c7f07c83da6418f0be5923a91e24e51de15c7d916b4922e61c7ca1. 25개tensor원본동일/양팔472행CPU1thread동등성확인. 첫다중thread last-bit차이의실패기록보존. screen2084decisions/83.41초는 학습량이아니다.

후속A는같은mixed472행에서최초충돌후연속무진행52행의BC loss weight만1→0.1로변경하는단일pilot. 나머지420행weight1,전체472행평균weight1로한번정규화(분모425.2). 원래initial3ceb/18params/epochorder/3epochs/batch32/lr1e-3/seed8101유지. minibatch별재정규화금지,기존loss의행별값에가중후기존mean사용. row선정은저장trainprovenance만사용하며화면평가자료를훈련에추가하지않는다. 고정자료량에서정체상태대비나머지상태의상대비중을바꾸는대조이며전체평균weight를맞춰도gradient크기는같지않다.

추가수집/PPO없음. 기존unweighted v2 screen을대조로재사용,weighted후보만같은4cell×600=최대2400decisions평가,30분. 이미반복사용screen을튜닝에쓴후속임을명시하고heldout으로부르지않는다. teacher-only control도참고로두며완주/공통셀시간/충돌악화가있으면가중치추가탐색없이중단. 개선신호라도독립확인전SOTA금지. runtime metadata는검증된설정으로저장하고학습메모리모델과strict평가행동일치후주행한다. 조향sigma가설은이결과와독립이며지금추가실행하지않는다.

### D 완료 수신: 조향 축만 탐색 폭 조정하는 후속 후보

D가 source learner=teacher-only BC control9289b1임을 runner로 확인하고 summary/result 문구를 정정했다고 회신했다. 원시 shadow NPZ SHA a6f96f0f5e18238dd55bc843ac4faae84cf8d63465d0298af53fde838fbf8998, D commits c119b61/fa0915d/ecb6fe5. 총괄은 앞 주기에 수치 원본을 확인했고 이번에는 완료 및 출처 정정 회신을 받았다. D는 대기하며 추가 실행 없음.

전체 sigma 절반은 조향뿐 아니라 가속 탐색도 줄인다. 총괄이 ROOT networks._bound_actions와 sample_actions_with_pretransform을 확인한 결과 조향 좌표와 longitudinal 좌표가 분리되어 있다. 따라서 다음 후보는 sigma=(.34996387,.40023994)에서 조향 축만 (.17498194,.40023994)로 바꾸는 단일변수 대조다. 같은 관측/mean/공통난수에서는 steering 통계가 half-sigma 진단과 같고 gas/brake는 original-sigma 진단과 같다. 이는 좌표 분리에 따른 수학적 귀결이며 새 주행 실측이 아니다. 궤적이 달라진 뒤 가속 분포까지 같다는 뜻은 아니다.

반증 조건: 동결 정책의 train 주행에서 원sigma 대비 충돌/완주가 개선되지 않으면 조향 noise가 주요 단기 병목이라는 근거가 약해진다. 주행 안정성이 좋아져도 PPO가 더 빨리 학습하거나13초에 가까워진다는 증거는 아니므로 동일예산 학습 대조가 별도로 필요하다. 큰 조향 시도가 회피 탐색에 필요할 수 있어 무조건 작은 sigma를 채택하지 않는다. A 공통screen 결과 확인 후 이 대조의 필요성과 예산을 결정하며 현재 후보 가중치/제출 runtime은 변경하지 않는다.

### 05:58 metadata 동등성 확보 및 shadow 진단 결과 확인

A resumed/a-action-parity-cpu1.json 직접 확인: ROOT 기준행동 대 A strictAgent/capturingAgent가 양팔 각각472관측에서 exact=true, provenance4그룹 reset 경계도 exact=true. CPU thread1 조건의 동일성 증거이며 새 파일/manifest 진척이 확인됐다. 공통8cell 최종 결과는 아직 미수신이다. 반복 blocker 요청 없음.

D research/strategy-d-exploration-shadow/result.json 원본 complete와 집계를 확인했다. 원 sigma에서 deterministic 조향과 절대차이>.2인 shadow 비율은 control54.722%,v2 53.988%; sigma0.5배에서는23.325%,22.450%. v2의 가속↔제동 전환(near-zero .05 제외, 전체draw 분모)은4.022%→1.452%. v2 gas95%cap 비율은.933%→0으로 줄었다. 총918관측 forward,각459×128draw이며 환경주행/optimizer0이다. sigma축소가 행동변동을 줄이는 동시에 높은 가속 탐색도 줄인다는 진단이며 주행 안정성·학습향상의 증거가 아니다.

자료 방문정책은 teacher-only BC control9289b1 자체이며 mixed v2 학습 전에 수집했다. D summary의 '양쪽 BC 이전 source learner' 표현을 정정 요청했다. 양팔의 방문분포나 충돌위험을 직접 비교한 결과로 쓰지 않는다. 다음 폐쇄루프를 선택한다면 같은 동결 후보/같은train환경에서 deterministic 기준+원sigma/half-sigma stochastic를 비교하되, 속도 감소만으로 충돌이 줄었는지 랩타임·진행도도 확인해야 한다. 현재 A 공통 평가 전에는 추가주행/학습을 배정하지 않는다.

### 05:48 정체 관측의 중복 여부 계측

A는 앞서 import 원인 회신 이후 새 복구/평가 산출물이 이번 조회에서 확인되지 않았다. 이미 blocker 요청과 답변을 주고받았으므로 반복 요청하지 않는다. D는 원본459행 hash/맵 분할/uint8 정규화 확인 진척을 보고했다. D가 read-only subagent를 잠시 시작했다가 앞선 금지 명확화 후 중단했다고 회신했으며 이후 직접 수행 중이다. B/C 변화 없음.

총괄이 learner NPZ의 충돌후 관측을 byte 단위 hash로 비교했다. 전체101행은101개 모두 다르고 v2에 뽑힌52행도52개 모두 다르다. 연속 관측의 전체4×84×84 평균 절대 픽셀변화는 uint8 단위 평균.3015(전체101),.4522(선택52)였다. 원본 hash와 통계는 `research/2026-09-24-postcollision-observation-diversity.json`에 저장했다.

따라서 단순 동일 이미지 중복제거로 이 정체 구간의 비중이 줄어들지는 않는다. 픽셀 차이는 위치/차량운동/계기판 변화 등을 함께 반영하며 평균 차이가 작다는 이유만으로 회복에 불필요한 관측이라고 단정할 수 없다. 후속 정체 가중치 가설을 선택한다면 관측 hash 중복이 아닌 에피소드 내 연속 무진행 구간을 단위로 정의해야 한다. 지금은 기존 v2에 손대지 않고 공통 평가/탐색 진단 결과를 기다린다. 새로운 주행/학습은 수행하지 않았다.

### A import 캐시 원인 회신

A가 prior를 먼저 import하면 editable-install finder가 원본 checkout training을 캐시하는 것을 확인했다고 회신했다. A 경로 설정 직후 training을 먼저 import하면 training/train_policy/strategy_a_diagnostic 모두 A 경로로 고정됨을 검증했다고 보고했다. 이는 A 회신 기준이며 총괄의 앞선 find_spec 확인과 구분한다. 이 순서에서도 prior 의존성이 A로 바뀔 수 있으므로 ROOT 기준행동 생성과 A strictAgent 출력을 별도 프로세스에서 비교하는 경계를 재확인했다. 새 체크포인트/환경 실행은 아직 없으며 다음은 동등성 및 기존8cell 결과 회수다.

### A import blocker 경로 원인 확인

A 회신: stdin preflight에서 training.strategy_a_diagnostic import 오류, 새 파일/환경 실행0. 총괄 파일 확인 결과 해당 모듈은 ROOT에 없고 A worktree에만 있으며 frozen screen manifest도 A 파일 SHA a0698276c1077b6976963f80093ffacce37712ae37f8f1beb6ca1e58f205525a를 지정한다. root .venv Python을 사용하되 cwd=A인 새 프로세스의 find_spec으로 정확한 A 모듈 경로 반환을 직접 확인했다.

학습 source ROOT와 frozen 평가 source A는 구분해야 한다. 같은 프로세스에서 이미 ROOT training을 import한 뒤 sys.path만 추가하면 캐시된 패키지가 남을 가능성이 있으나 A 오류의 구체적 캐시 상태까지 측정한 것은 아니다. A에 기존 C runner처럼 새 subprocess cwd=A에서 frozen strictAgent/parity/eval을 수행하도록 경로 근거를 전달했다. metadata복구의 학습 설정 근거는 ROOT를 유지하고 두 프로세스 사이에는 명시적 checkpoint/관측/출력 배열로 비교한다. 모듈 복사/환경 재설치/소스 계약 변경 없이 재개하며, 경로 확인 자체를 주행 성공으로 기록하지 않는다.

### 05:38 복구 blocker 확인과 독립 탐색 진단 배정

A는 복구 지시 이후 약14분간 app 메시지/도구기록과 지정 experiments/artifacts의 새 파일이 확인되지 않았다. active 표시만으로 실행 진행을 단정하지 않고 현재 단계/마지막 산출물/막힌 점을 한 번 요청했다. 같은 복구 지시 재발송이나 작업 재시작은 하지 않는다. B/C는 변화 없음.

D 기존 실제 작업(01a0ceca-0f09-7571-a83e-6d1f80ffe655)을 Luna max로 재사용해 환경0/학습0의 독립 오프라인 진단 배정. A 원본459개 learner 관측에서 teacher-only BC와 mixed BC의 평균 및 원래 sigma/0.5배 sigma shadow 분포를 비교한다. 각 관측128개 공통표준정규draw(seed8101), 최대918개 관측 forward/20분, D고유 산출물. 모델구성은 base3ceb의 입증된 실제 설정+각 state_dict strict load, ROOT source만 사용. A의 metadata복구/평가 파일은 수정하지 않는다.

집계는 deterministic 조향 대비 절대변화>.1/.2 비율, longitudinal 가속/제동 부호전환 및 near-zero 별도,gas95%cap 비율. map1 최초충돌 index121 직전10/그외충돌전/충돌후와 map2로 구분한다. 이는 현재 상태에서 탐색 변동의 크기를 측정하며 sample128개를 독립주행128회로 세지 않는다. 진단임계값은 안전기준이 아니며 높은 변동 자체가 충돌 원인이라는 결론도 금지한다. 이전 A base/screen shadow 분석과 달리 이번에는 두 BC 모델/동일 train learner 관측/분산배율만의 효과를 묻는다. sigma 축소 정책은 아직 채택하지 않았으며 폐쇄루프 대조는 A결과 이후 별도로 선택한다.

### 05:28 BC 이후 탐색 분산의 보존 확인

A 복구 작업은 active이나 이번 조회에서 복구 파일/공통 주행 결과는 아직 없다. B/C/D compact cursor는 변화 없음. 중복 실행 지시는 보내지 않았다.

총괄이 base3ceb, teacher BC9289b1, mixed BCae1ba0의 model_state를 직접 읽었다. 세 모델의 policy_log_std는 정확히 [-1.0499253273,-0.9156910777], exp값은 [0.3499638736,0.4002399445]로 같다. networks.py의 sample_actions_with_pretransform은 이 표준편차를 가진 Normal에서 표본을 뽑는다. BC가 결정적 평균 정책을 바꾸어도 기존 확률적 탐색 폭은 그대로 남았다는 증거다. 값은 tanh 이전 좌표이고 물리 조향/페달 표준편차와 동일하지 않다.

**기존 탐색 가설의 다음 분기:** 공통 screen이 유망하더라도 결정적 BC 완주가 PPO 수집 시의 완주를 보장하지 않는다. 차후 PPO 확대 전에 같은 동결 BC 정책에서 deterministic 대 원래 stochastic 행동의 train-only 짧은 노출 비교가 원인 분리에 유용하다. stochastic부터 실패하면 초기 탐색 분산 조정이 후보이고, 양쪽 모두 안정적인데 update후에만 무너지면 optimizer/목표함수 쪽을 우선한다. 아직 두 경우를 측정하지 않아 탐색이 원인이라고 판정하지 않는다. 분산을 줄이는 실험을 선택한다면 초기 log_std만 변경하고 실제 새 분포로 rollout 및 old logprob를 일관되게 저장해야 한다. 환경에만 작은 noise를 적용하면서 원래 Normal logprob를 쓰는 대조는 허용하지 않는다. 현재 metadata 복구/공통 screen이 우선이며 이 진단은 아직 미배정이다.

### 공통 screen 사전 중단: 체크포인트 설정 누락 복구

A summary.json 직접 확인: 두 BC 체크포인트 strict load 실패로 episode0/환경decision0. 미실행8행을 DNF로 집계하지 않는다. root agent.py는 metadata의 use_hud/use_visual_features/use_temporal_features 및 pedal expansion으로 구조를 만든다. 총괄이 v2 runner 저장부를 읽어 연구정보만 metadata에 쓰고 학습 모델의 구조 설정을 전달하지 않는 것을 확인했다. 따라서 현재 파일은 직접 제출 가능성이 확인된 파일이 아니다. 기존 tune은 학습 중 메모리 모델 평가이므로 그 주행 결과와 파일 로딩 실패를 구분한다.

후속 A: 원본과 실패기록 보존, 초기 checkpoint 및 실제 생성 코드에서 근거가 확인되는 runtime metadata만 별도 패키징 파일에 복원한다. 두 팔 같은 방식 적용, tensor/step/기존 metadata 유지, adapter/strict 완화/가중치 수정 금지. 입력 파일 hash와 변경 key/value/근거/출력 hash를 기록한다. 실제 학습 설정의 in-memory 모델 대비 strict Agent의 472개 저장 관측 결정적 행동을 비교하여 동등성을 확인하고 통과 시 원래8cell/4800decision 이내 공통 screen을 재개한다. 불일치면 주행 전 중단한다. 이는 새 학습이나 성능 개선 실험이 아닌 실행 파일 설정 복구다.

### 05:18 동일 학습자 관측에서 교사와 실행 행동 비교

A 공통 screen은 active 표시이며 새 결과 산출물은 아직 확인되지 않았다. 배정 후 수분 경과 단계여서 실행 재촉/재시작 없이 유지한다. B/C/D는 새 결과 없음.

저장 NPZ의 map1 최초충돌 직전10개 관측에서, 실행 learner와 조회 raw teacher를 같은 상태 기준으로 비교했다. learner 평균 steer/gas/brake=(-.14058,.04819,.00516), teacher=(-.06031,.09000,.00400), 평균 절대 조향 차이 .08505. 제동 양수는 learner2/10,teacher1/10이다. 결과는 `research/2026-09-24-learner-teacher-action-diagnostic.json`에 원본 hash와 저장했다. raw teacher gas는 .75 변환 전이며 학습 target과 직접 혼동하지 않는다.

이 train 사례는 교사가 더 강하게 제동하는 정답을 제공한다는 가설을 지지하지 않는다. learner는 이미 제동을 했고 teacher는 평균 가속이 더 크다. 따라서 성공한 v2를 단순한 '브레이크 교육'으로 설명하면 안 된다. 조향·진입 자세와 이후 상태 분포 변경도 가능한 경로다. 충돌후101행에서는 teacher steer평균.35058 대 learner.18700, 양쪽brake0이나 교사의 행동을 실제 실행하지 않았으므로 회복 가능 여부는 미측정이다.

후속 공통 screen에서 같은 맵/seed의 최초충돌 위치·시점 및 조향 변화를 확인하면 다음 가설 선택에 도움이 된다. 서로 다른 정책이 방문한 상태 차이는 남으므로 동일 상태 인과효과로 표현하지 않는다. 현재 실험의 loss/teacher/예산을 수정하지 않으며 새 주행도 추가하지 않았다.

### A v2 완료: 단일 tune 무충돌 개선과 공통 screen 연결

총괄이 v2/run.json 직접 확인: BC3epochs/18parameters/seed8101, PPO 추가 없음. 동일 tune에서 기존 teacher-only BC control 18.88초/collision1/damage.2 대비 mixed v2 18.06초/collision0/damage0, 양쪽완주/progress1,13초미만은아님. 0.82초(약4.34%) 단축. 후보 checkpoint SHA256 `ae1ba00dc5215d2f7ef3e60479d7ad1a9e4421d2092c8d12a95b8b2130135ceb`, model state `f0d7a95af1d28132b327656cb43f7674bb5abf95113fc553a6248abbecc3cbe9`. 저장 수집459+tune226=685decisions, 새 수집/교사 재조회0. v1 실패원본 보존. 단일 tune 및 사후 sampling 수정의 탐색 후보이며 SOTA는 유지한다.

후속은 A가 동일 공통 screen v1에서 teacher-only BC control9289b1…와 mixed v2를 각각 평가한다. T1 seeds73/74,T2 seeds132/133,팔당4회×최대600decisions,총4800이하,actor-only 결정적추론. source/환경/관측/행동/타이머 계약 일치와 strict loading을 확인하고 기존 root base 및 C 후보는 조건 일치 시 참고값으로 재사용한다. teacher-only BC control의 공통 screen은 아직 없어 반드시 포함하며 기존 PPO base와의 비교만으로 자료 교체 인과효과를 주장하지 않는다. 학습/계수탐색 없음. episode 원시기록으로 완주/13초미만/충돌/최종damage/시간/종료원인을 보고한다. 반복 사용 screen이므로 독립 확인평가로 부르지 않는다.

정체52행이 포함되어도 단일 tune 개선은 가능했다. 따라서 정체 자료의 존재만으로 악영향을 확정했던 것은 아니며 현재 가중치 수정 후보는 대기한다. 공통 screen 결과 전 PPO나 추가 자료 가중치 변경을 섞지 않는다.

### 05:08 learner 자료의 정체 비중 실측

A v2는 active 표시만 확인됐고 이번 조회에는 새 runner/결과 파일이 아직 없다. 재지시하지 않았다. B/C/D의 compact cursor는 변화 없음.

총괄이 저장 learner-collection.npz를 읽어 v2의 실제 길이 균등추출 규칙을 오프라인 적용했다. 원본 SHA는 앞 기록과 일치하며 계산 결과는 `research/2026-09-24-learner-sampling-diagnostic.json`에 저장했다. map1 최초 충돌 transition은 0-based index121이고 이후101개 transition의 progress_after가 모두0.5다. 선택115행에는 충돌직전10결정 중5행, 충돌 transition1행, 충돌후52행이 포함된다. map2 선택121행은 충돌 노출이 없다. 따라서 혼합472행 중52행(약11.0%)이 이 하나의 충돌 뒤 무진행 구간에서 온다. 고유 transition이어도 독립적인 회피 사례52개를 뜻하지 않는다. 교사의 충돌후 brake 평균0이라는 사실만으로 label이 잘못됐다고 단정하지 않는다.

**다음 판별 가설(미배정):** v2가 개선되지 않으면 learner-state 보강 자체보다 한 번의 정체가 반복 학습되는 자료 가중치를 조사할 근거가 생겼다. 동일 저장자료/472행/총 업데이트를 유지한 채 충돌후 연속 무진행 구간의 총 loss weight만 제한하는 대조를 고려할 수 있다. 이때 weight 총합을 정규화하고 teacher/learner·map별 실효 가중치를 기록해야 하며, 실제 학습률 변화와 자료 내용 변경을 섞지 않아야 한다. 충돌 전10결정 창은 이번 진단용 정의이며 회피 가능 시간이라는 증명은 없다. 현재 v2는 조건 변경 없이 끝내고 이 가설은 결과를 받은 뒤 선택한다.

### A 자료 수집 종료: 고정 시간 인덱스 미충족과 v2 재사용 설계

A `bc-state-distribution-dagger-v1/run.json` 직접 확인: status=data_insufficient, learner 459행, teacher 조회459/실행0, BC/tune 미실행. map1은223decisions/off_track/progress.5/collision1/damage.2, map2는236decisions/cap/progress.98008/collision0. NPZ SHA256 `163b9121d3ee0ee574ba791fff9f89d1c0501d17495dc233109f4b00c8ca4367`. 기존 control 재사용 검증은 통과했다.

종료 원인은 map1에 필요한115개보다 관측이 적어서가 아니라, 236길이에 사전 고정한 시간 인덱스 중223 이후7개가 없기 때문이다. v1 중단은 계약을 따른 올바른 처리이며 원본을 보존한다. 실패한 에피소드를 제외하지 않고 활용하려는 후속 v2를 별도 탐색 실험으로 배정한다. 자료를 본 뒤 선택 규칙을 변경했다는 점을 명시하며 v1 사전등록 성공으로 표현하지 않는다.

v2는 저장된459행만 사용한다. 각 에피소드 실제 길이 n에 대해 `rint(linspace(0,n-1,k))`로 map1 k115/map2 k121을 선택하고, 기존 teacher 선택236행과 합쳐472행을 만든다. 고유 transition 인덱스만 허용하며 같은 이미지가 반복될 수 있다는 점과 중복 복제를 구분한다. 교사 재조회·재수집·추가 PPO는 없다. 원본/선택 인덱스/hash를 저장하고 initial3ceb,18parameters,3epochs,batch32,lr1e-3,seed8101 및 기존 tune max400을 유지한다. 추가 환경 예산400, 총 수집+평가 최대859decisions. 자료 접근/출처 불일치면 중단한다. 후속 성능 결과는 아직 없다.

### 04:58 자료 교체 가설의 판정 범위 고정

A의 compact 상태 메시지는 여전히 비어 있으나 `experiments/strategy_a/bc_state_distribution_dagger.py`가 04:58:53에 갱신되어 실제 구현 진척을 확인했다. 새 결과 디렉터리는 이번 조회 시점에 없었으므로 학습/완료 증거는 아직 없다. runner에는 에피소드별 teacher.reset 및 모든 learner 관측의 teacher.act 호출이 들어 있다. 이는 04:48 이력 조건이 코드에 반영된 증거이며 실제 실행 성공과 구분한다. B/C/D는 이전 cursor에서 변화 없음. blocker 재촉이나 중복 실행을 요청하지 않았다.

**고정 자료량 대조에서 알 수 있는 것:** 원래 teacher 472행 대 teacher 236+learner 236행 비교는 같은 학습량에서 자료 절반을 학습자 방문 상태로 교체하는 전략의 효과를 묻는다. 원래 교사 자료 감소와 새 상태 보강은 이 교체에 함께 포함된다. 따라서 실패를 곧바로 covariate shift 가설의 반증으로 쓰거나, 성공을 특정 충돌 전 프레임의 효과로 단정하지 않는다. 동일 472행/3epochs가 같아도 서로 다른 자료의 평균 BC loss는 같은 평가 오차가 아니다.

**결과 수신 시 분기:** 고유 자료가 부족하면 자료 수집 계약 미충족으로 종료하고 정책 실패로 세지 않는다. treatment DNF이면 현 교체 전략을 확대하지 않는다. 완주하되 충돌이 늘면 시간 단축과 안전성의 교환으로 보존한다. 기준 18.88초/collision1보다 시간 또는 충돌이 좋아지고 다른 항목이 악화되지 않으면 후속 공통 screen의 후보로만 둔다. 완주·시간·충돌이 같으면 긍정 증거 없음으로 기록한다. 어느 경우에도 이번 단일 tune 결과만으로 13초 가능성, 학습 충분성, SOTA 갱신을 주장하지 않는다. 현재 실행의 자료/예산/모델은 변경하지 않았다.

### 04:48 수집 교사의 이력과 자료 대조의 해석

A는 새 learner-state 실험 active 표시이나 이번 조회에서 새 수집 산출물은 아직 확인하지 못했다. 실행 완료로 간주하지 않고 기존 지시를 유지한다. C는 새 seed 재현 종료 기록 commit 1b591ce4로 정리 후 idle, B/D도 새 성능 원본이 없다. 추가 작업을 중복 생성하지 않았다.

총괄이 root `haic_agent/corridor_agent.py`를 읽어 교사의 상태성을 확인했다. 회피 방향은 이전 obstacle offset과 side를 사용하고 목표속도는 `0.65 * 이전값 + 0.35 * 현재값`으로 평활한다. 따라서 동일 이미지에 대한 정답도 관측 이력에 따라 달라질 수 있다. learner 에피소드마다 교사를 reset하고 모든 관측에 순서대로 조회한 뒤 사전 지정 인덱스를 추출해야 기존 교사 의미를 보존한다. 추출한 프레임에만 교사를 호출하면 상태 분포뿐 아니라 라벨 생성 방식까지 변경된다. A에 이 조건만 보충했으며, 이미 다른 방식으로 수집했다면 재실행 대신 차이를 기록하도록 했다. 예산과 teacher 실행 0% 계약은 유지한다.

**반증 조건 보강:** 현재 학생은 temporal=false이므로 이력 의존 교사를 단일 관측으로 모방하는 데 모호성이 있을 수 있다. 이는 소스에서 얻은 가능성이며 실제 동일 관측의 상충 정답을 측정한 결과가 아니다. 이번 자료 혼합이 실패하더라도 곧바로 데이터 보강 전체를 기각하거나 recurrent actor로 전환하지 않는다. 먼저 수집 이력 보존, 고유 행 수와 맵 비율, 충돌 전후 구성, 동일 평가 종료 원인을 확인한다. 성공하더라도 단일 tune/seed의 탐색 결과로만 기록하고 PPO 확대나 SOTA 승격은 별도로 판단한다.

### 04:38 새seed 재현 종료와 learner-state 자료 대조 실행 배정

총괄이8103양팔result.json직접확인:control77decisions/progress0.255144,detach77/0.259259,양쪽collision5/damage1/crash미완주.8102포함새두seed는완주회복0/2,control도0/2. 8101선택후보의개별주행기록은보존하지만detach를안정개선으로채택하지않고추가seed/학습배증없음. C는총4run16384결과정리후대기.

사전자료분포안을A에구체적배정:동결A .75 BCcontrol9289b1...로원train2맵각236decision이하,total472수집,teacher는조회만하고환경에는learner행동100%. 기존C cruise80/maxgas.12/.75gas-only label변환고정. treatment자료는원teacher236+learner236=472,원map비율230/242유지를위해각반쪽115/121행을episode내사전균등간격규칙으로선정. 조기종료로고유행부족시추가수집/복제없이중단보고. 원초기actor3ceb5e.../기존18parameter BCoptimizer/3epochs/batch32/lr1e-3/seed8101고정. 원control은A원본정합시재사용,불일치시보고. treatment최종동일tune1회max400,총환경예산872/BC3epochs/PPO없음. teacher라벨의회피성공은가정하지않고상태출처/행동유효성/충돌전후행수/환경teacher실행0기록. source ROOT수정없이A전용runner,새리뷰/인프라확장없음. 이번단일변수는고정총량자료의상태구성이다.

### 04:28 새seed 대기 중 자료분포 가설의 대조 조건

8102 양팔result.json 원본확인,8103control2048checkpoint진척회신. 현재네run예산내진행을유지하고추가seed나계수변경배정없음. A/D대기,B notLoaded상태.

향후자료분포대조안(미배정/미실행): 깨끗한teacher경로472행의MSE를낮추는것과학습자방문상태에서의회피능력은다른질문이다. A에서MSE개선과미완주가동반됐고현재PPO후작은행동변화에도tune실패가발생했으나,이것만으로covariate shift를확정하지않는다. 다음에이질문을실험한다면기존train geometry에서만학습자rollout을모으고동일teacher/동일.75label변환으로그상태를라벨링한다(기존DAgger 가설의구체화). screen/tune의충돌직전이미지를훈련으로옮기지않는다.

단일변수는새로운optimizer/보상/모델이아닌훈련상태자료구성이다. control은기존teacher방문자료,N행;처치는사전비율로teacher방문+learner방문자료,N행,동일초기actor/epochs/minibatch/총gradient업데이트로맞춰자료수증가와혼합하지않는다. 라벨링교사가그상태에서실제안전한행동을주는지미확인상태이므로단순teacher조회성공을회복정답으로간주하지않는다. 정체프레임복제와충돌전상태의비율·episode출처를보존한다. 현재8103결과와기존collector제약을확인한뒤에만예산/비율을동결해배정할수있으며지금은구현하지않는다.

### C 새 학습seed8102 쌍 중간 회신: 완주 회복 미재현

C회신기준양팔4096decisions/4updates정상완료. 동일tune에서control과value-detach모두76decisions,progress0.259259,collision5,최종damage1,crash미완주,13초미만false. 완주시간paired비교는해당없음. controlSHA6c13967fbc15f45fad5fbd075e8e27ee868c1de844b49b38b44078922709ca76,treatmentSHAf1bfd29fcbe056c6f497783a3ff64148950b582c06d381be911fbe8a8476d00e. 가중치는다르지만요약결과동일. seed8101의완주회복이8102에서는재현되지않았으므로안정된개선이라고판정하지않는다. 사전배정8103쌍은조건/예산변경없이계속,새seed추가탐색없음. 동일tune학습seed재현이며독립맵검증아님.

### 04:18 기록 단축의 기하적 규모 점검

C seed8102 control실행시작회신과04:17출력디렉터리확인,첫1024완료아직없음. 다른작업새결과없음,진행중실험조건변경없음.

총괄은기존완주4개trace의연속car_x/y샘플거리합을계산해research/2026-09-24-finished-path-diagnostic.json에저장. base→detach T1/74는거리1018.54→1014.45(약-0.4%),시간24.26→22.44초;T2/133은1043.45→1052.74(약+0.9%),25.18→23.70초. 샘플사이곡선/첫구간을생략한chord근사이므로정확한주행거리나최단경로하한증명은아니다. 현재시간개선은이근사상큰경로단축과동반되지않는다.

후보의현재경로를13초에주행한다고단순가정하면평균진행속도규모는약78.03/80.98 simulator-unit/s,시간비율22.44/13=1.726및23.70/13=1.823이다. 출발가속·곡선·장애물·타이머/샘플구간차이를무시한규모계산으로가능성/불가능성을확정하지않는다. 현상태는수퍼센트개선만누적하면곧13초가된다는단계가아니며큰속도증가와충돌회피를함께배워야한다. 이전gas cap0.24실험은충돌로실패했으므로그설정을재반복하지않는다. 당장학습상한을올리기보다현재detach효과재현여부를확인하고훈련용안전진입/회피자료가필요한지다음분기로검토한다.

### 04:08 저장 trace의 최초충돌 전 구간 후처리

총괄이base/new candidate의T1/73,T2/132 episodes.jsonl을직접읽어최초collision직전10decisions(0.8초)의행동/속도를계산,sourcehash와결과를research/2026-09-24-screen-precollision-windows.json에저장했다. 새주행없음.

T1/73 base첫충돌153/progress0.49064,후보146/0.45693;직전평균속도46.42→51.38,gas0.05547→0.08853,양쪽brake0. T2/132 base201/0.62821,후보182/0.62821;속도41.31→46.25,gas0.08302→0.10924,양쪽brake0. 서로방문위치·진입궤적이달라같은상태의인과대조가아니다. 로그nearest_obstacle_index2가같아도실제충돌대상동일성은별도확인이필요하다.

T1후보collision5는146~150연속프레임이고T2base5도201~205연속이다. '서로다른장애물5개와충돌'이아니며기존충돌지표의이벤트정의를보존한다. 진단적으로제동없는고속진입이관찰되지만,perception위험추정누락인지정책선택인지안전한조향실패인지현재필드만으로분리못한다. 이를이유로brake보상상수를즉시올리지않고새seed재현결과를먼저받는다. 이후필요한학습자료는screen맵을학습에옮기지않고원train의유사충돌전상태에서만확보해야한다.

C는seed재현runner의과거protocol필드확인오류를실행전수정중이라고보고했고04:04runner파일확인. 새학습완료없음. A/D대기,B notLoaded로표시변경(새성능결과아님),중복지시없음.

### 03:58 공통screen 원본 확인: 충돌 합계가 숨긴 조건별 변화

총괄이C strategy-a-screen-v1/value-detach-v1-run-output/summary.json 직접확인해완주T1/74=22.44초,T2/133=23.70초·양쪽무충돌을확인했다. total collision6/damage1.2는base와같지만분배는다르다. T1/73은basecollision1/damage0.2/off_track에서candidatecollision5/damage1/crash로악화;T2/132는base5/damage1/crash에서candidate1/damage0.2/off_track로변했다. 양쪽여전히미완주이고실패progress도0.49064→0.46067,0.63141→0.62821로낮아졌다. 따라서'안전성이같다'는해석은부적절하며확정표현은총합동일·조건별trade-off다.

새후보의실제screen방문중95%gas cap비율은T1/73 50%,T1/74 33.45%,T2/13329.63%로기록돼있다. 앞선D의약7%는다른freshAdam모델을BC초기train관측에서평가한값이므로두비율차이를detach인과효과로비교하지않는다. 속도개선과실패악화가공존하는후보라는전체기록을유지한다.

C app의마지막commentary는이미완료된screen사전확인으로뒤처져있지만완성원본과완료회신이우선이다. 새seed재현산출물은이번조회아직미확인,배정직후이므로중복지시없음. A/B/D새작업없음. 기존최고기록승격없음.

### C value-detach 공통screen 완료 회신 및 새seed 재현 배정

C회신:기존evaluator fingerprint/source12hash/contract/runtime일치하여base/.75원본재사용,checkpoint provenance차이는정상. 새후보4cell한번씩총1011decisions,2/4완주·13초미만0/4·collision6·최종damage합1.2. 미완주T1/73 progress0.4607 crash,T2/1320.6282 off_track. 완주subset median23.07초. base는2/4/24.72초/동일충돌손상,공통완주T1/74 -1.82초,T2/133 -1.48초. .75는2/4/23.94초,동일셀-1.62초/-0.12초이며충돌손상은이번후보가작다. 이숫자는담당완료회신기준,원본은C strategy-a-screen-v1에있다. 낮아진실패진행도도함께보존하고전지표우월이라표현하지않는다. 이미재사용한선택용screen,독립확인아님.

유용한속도절충이남아사전계획한PPO seeds8102/8103 paired재현을C에배정했다. 동일BC41a38b70...+freshAdam에서control/value-detach각4096/4updates,총4run16384decisions순차실행. source/train/하이퍼파라미터고정,seed별양팔동일초기RNG,BC재학습없음. 각1024last보존하되최종4096만기존tune1cell/max400평가,screen추가없음. 각seed성공실패를모두보고,최초쌍완료시회신후나머지고정예산진행. NaN/실제학습crash중단보고,임의확대없음. 8101은가설선택용,새seed도같은tune이므로heldout/SOTA승격없음.

### 03:47 화면 평가 준비와 재현 질문 사전 구분

C는공통screen runner/출처확인중이며새4cell결과없음. 총괄이attempt2/result.json tune_episode직접읽어19.56초/245decisions/완주true/충돌1/damage0.2/13초false확인. D후처리완료,A기존결과유지,B새결과없음. 조건변경·중복실행지시없음.

다음재현안(미실행): screen에서기존후보들보다모든핵심지표가나빠지는경우에는이candidate의광범위일반화신호가없으므로자동배증하지않는다. 유용한완주/충돌/동일셀시간절충이남으면freshAdam control 대value-detach를새PPO RNG seeds8102/8103에서각4096last로비교하는안을우선한다(총4run/16384decisions;아직배정하지않음). 양팔같은BC초기모델,맵순서규칙/학습량/하이퍼파라미터를맞추고실패seed도보존,각seed양팔결과를paired로보고한다. 같은BC모델을고정하므로BC를포함한전체파이프라인의재현성실험은아니다. 환경custom seed label변경은같은정적geometry를반복할수있으므로PPO RNG seed와구분한다.

새훈련seed에서효과가되풀이되어도이미여러번본tune/screen은선택용이다. heldout은후보선택과계약이닫힌뒤에만사용한다. 이번screen은최종성능후보비교이고BC와label이다른base/.75와의차이는value-detach인과효과측정이아니다. 해당인과대조는동일normalized BC/freshAdam 두팔에서만한다.

### C value-detach4096 완료: tune 완주 회복 및 공통screen 배정

C attempt2 result.json status/final checkpoint직접확인:4096decisions/step23552,finalSHA5f87019869b6ff2b35031fdbae9c66bd185452605656a7e7a901071799e8c114,modelstate01bb0ba4fbcb28497c7fb1dfc097d53a0254501d301ddfab50447c7c3b477cfe. 담당회신의동일tune19.56초완주/collision1/damage0.2/progress0.983539,13초미만없음. freshAdamcontrol미완주177decisions/collision4/damage0.8/progress0.255144대비완주회복신호다. 원BC20.38초/collision3과도구분해기록하며.75다른label후보18.74보다빠르다고주장하지않는다. 실패attempt1비용1024+정상4096=5120유지.

C에추가학습없이공통screen4(T1/73,74,T2/132,133),각600decisions/총2400이하평가배정. 기존C strategy-a-screen-v1동일runner/source환경/loader고정,조건동일할때만base/.75원본재사용하고불일치면보고. 전체완주·13초미만·충돌·최종damage·실패진행도와동일완주셀시간을보고한다. 이미반복사용한선택용screen이므로heldout확인아니다. SOTA승격없음,screen결과후훈련seed재현을검토하며현재학습배증없음.

### D 상한 근접·행동 차이 후처리 완료 회신

담당이기존policy-movement.npz만읽어추가추론/주행없이계산.512초기BC train방문관측의95%gas cap비율은BC42/512(8.2031%)→PPO37/512(7.2266%);steer/brake양쪽0. obstacles25/286→20/286,다른train17/226→17/226. exact-bound hits0과구분한다. PPO-BC 행동절대차이meanabs/p90/max는steer0.00844998/0.01079071/0.01998434,gas0.000892576/0.00186020/0.00494545,brake0.000054012/0/0.00732320. 자료내nonfinite없음.

이관측집합에서는PPO의상한근접가속증가가확인되지않으므로'가속포화가증가해tune실패'라는설명은지지되지않는다. PPO실제tune상태분포에서의가속/위험은이자료로평가하지못하며전체과속가능성을기각하지않는다. 평균행동차이가작아도폐쇄루프결과동등성은보장되지않는다. 추가계수/학습실험을파생시키지않고C value-detach결과와종합한다. 근거는D policy-movement-summary.md/result.json/npz,본숫자는담당후처리회신기준.

### 03:37 BC→PPO 정책 이동 결과 회수

D policy-movement-summary.md 직접확인. 저장512train관측에서KL(BC||freshPPO4096) mean0.0016518877,median0.0013368306,p900.0031056964,max0.007076021;평균축기여steer0.000353646/longitudinal0.001298242. 반대방향mean0.001652081. 두맵별집계도보존. exact physical-bound hits는두모델전채널0이나tanh의정확한상한도달과상한근접은다르므로포화가없다고단정하지않는다.

D에이미저장한행동배열만으로기존A기준95%cap도달률과행동delta meanabs/p90/max후처리를요청했다. 새추론/주행/학습없고배열없으면한계표시로종료. 평균KL만으로변화가충분히작아서안전하거나너무커서실패했다고분류하지않는다. BC초기훈련방문관측에서의정책거리이므로tune위험상태분포변화·폐쇄루프오차누적을판정하지못한다. 이자료만으로KL규제강화나learning rate변경을추가하지않고C detach대조결과를먼저받는다.

C attempt2는03:37 runner수정산출물확인,아직완료결과없음. A최종종료원인기존자료확인중,중복실행요청없음.

### A BC 종료 사유 확인 회신

A회신: treatment run.json에는finished=false/terminated=true/truncated=false만있고retire_reason/off_track원시bool미저장. 따라서명시적종료사유복원불가. 담당이읽은평가기관례로는미완주terminated를off_track으로분류하지만이것을해당주행의원시사유로대체하지않는다. collision1/damage0.2는damage>=1의파손기준에못미치므로'crash로종료'라는이전담당표현은근거없어철회. 확정결과는180decisions,진행도0.25103에서미완주종료. control18.88초는충돌1/damage0.2가있는완주로clean아님. 재평가없이자료보존·후보확대없음.

### A BC visual encoder 대조 완료: 낮은 훈련오차, 주행 퇴행

총괄이A bc-visual-encoder-optimizer-v1/run.json 직접확인. 동일472자료/.75labels/초기state/순서/3epochs에서membership18→20은visual encoder weight/bias만추가. encoder delta control0,treatment L2 0.16293/maxabs0.02403. 훈련MSE0.00829166→0.00565196이나단일tune은control18.88초완주/237decisions/collision1/damage0.2/progress1.0,treatment미완주/180decisions/collision1/damage0.2/progress0.25103. 양쪽13초미만없음. control SHA9289b1d60238bcdd70ac3e3b6dbeb847a93219e73d5fff24a85b2647ab2267a0,treatment46c9fae04beceab68b16ab70ee9aa50e08ea11614e4eedb2c71c61d679829a7b.

이번조건에서해당treatment기각/PPO확대없음. 모듈미포함을성능버그로간주해원본수정하지않는다. 기존.75 BC18.88의'clean'표현은신규원본의충돌1회로정정하며.75 PPO후18.74결과와혼동하지않는다. run.json은treatment terminated=true만저장하여collision1/damage0.2를곧바로crash라쓰지않고종료원인기존자료확인만A에요청했다. 새평가없음. 낮은train MSE가폐쇄루프개선을보장하지않는사례이며모듈학습의일반적해로움으로확대하지않는다.

### C value-detach attempt1 실행 오류와 명시적 재실행 비용

C회신: forward일치,ordinary value sharedgradient4.4076→detach0,valuehead12.2417유지로처치사전확인통과.1024rollout과첫PPO updater호출후runner의values_are_finite 잘못된모듈참조로중단. checkpoint/tune없음. 'post-checkpoint completed0'을학습0으로해석하지않고이미소비1024및첫updater실행으로기록한다. 모델NaN이나처치성능실패증거가아니다.

총괄은정의된함수의참조최소수정및wrapper속성사전확인후별도attempt2배정. 부분가중치미보존이면원BC/freshAdam/seed에서4096재실행,원실패기록보존. 이번연구누적환경비용상한은명시적으로5120(실패1024+비교4096)으로변경하며최종정책대조dose는양팔4096이다. checkpoint를updater후조기저장하여후처리실패와분리. baseline재실행/학습설정변경없음,학습NaN/crash중단유지.

A새BC는실행회신수신:두팔3epochs완료,control tune18.88초/충돌1/최종damage0.2,처리군평가중. 이전C .75 BC의clean표현과차이가있어최종원본확인전무충돌로쓰지않는다.

### 03:27 진행 및 다음 측정: BC에서 PPO까지 정책 이동

C는03:26 value-detach protocol생성,아직학습실행산출물미확인. A는새BC배정약20분후에도artifact미확인으로blocker/실행위치확인을이번실험에대해한번요청했다. 재시작이나예산확대요청은하지않았다. B상태변화없음.

D에기존fixed_batch512관측의BC→freshAdam4096 정책분포이동후처리배정. 이미C가분석한fresh-vs-restored비교와달리출발BC와실패PPO간누적변화를묻는다. 동일ROOT source/strict load에서pretransform diagonal Normal 양방향KL을행별계산,전체/맵별mean·median·p90·max와두행동축기여·포화율기록. 새학습/rollout없음,10분범위. PPOclip의존재만으로누적정책변화가작다고가정하지않고실측한다. 데이터는초기정책의훈련방문분포이므로tune실패원인이나전체상태분포거리를입증하지않으며임의KL임계값으로실패를분류하지않는다. intermediate checkpoint유무만확인하고없으면재구축하지않는다. C detach조건변경없음.

### 03:17 총괄 후처리: 합성 gradient의 방향은 반전되지 않음

D result.json의norm/cosine으로dot(g_policy,g_total)=||g_policy||²+0.5 dot(g_policy,g_value)+0.1 dot(g_policy,g_aux)를계산했다(shared entropy0). 원본hash와8행계산은research/2026-09-24-policy-total-gradient-projection.json에저장. 모든batch에서dot양수,||g_policy||²로정규화한값0.9143~1.0517,total과policy cosine0.4453~0.8870이다. pairwise음수cosine이있어도합성gradient가정책손실의국소1차하강방향을뒤집은것은아니다. 'value가정책을거꾸로학습시켰다'는강한설명은이자료에서지지되지않는다.

이는Adam/실제step/후속상태분포전의공유파라미터국소계산이며value표현변화의장기효과를기각하지않는다. 현재C detach실험에는해석정보만전달,조건/예산변경없음. A active표시는있지만이번조회산출물미확인,배정후약10분이므로중복blocker요청없음. C는전용updater준비메시지확인,새완료성능없음. B/D는기존상태유지.

### D gradient 진단 완료와 C value detach 대조 배정

D 회신및result-summary.md확인. source13hash/strict load정합후단한번512decisions수집:obstacles286행/train226행. 사전고정첫256행8minibatch는모두첫obstacles geometry에속함. policy/value cosine음수5/8,policy/aux3/8,계수적용value+aux합성gradient norm>policy2/8,entropy sharedgradient0/8. optimizer.step/학습/평가없고state불변. 이는한trajectory국소gradient기하이며독립8회재현이나미완주원인증명이아니다. batch fixed_batch.npz보존;D result commit c95895a는담당회신기준.

조건부계획에따라C에value head입력latent detach대조배정. normalized BC41a38b70...+freshAdam/seed8101에서기존freshAdam4096을control로재사용,처치는value MSE를value_head(output.latent.detach())로계산하는것하나. valuehead학습유지,policy/aux/entropy/reward/계수/source/runtime불변. 원본root수정없이C전용updater,처치전forward동일·shared valuegradient0·valueheadgradient유지확인. 단한팔4096decisions/4updates,각1024last보존하되최종4096만동일tune400decision평가,중간최고점선택없음. 학습NaN/crash중단,추가확대없음. global clipping/critic/GAE의후속변화도있으므로결과를순수gradient방향효과로과장하지않는다. A visual feature BC대조와별개초기화/산출물로유지한다.

### 03:07 미실행 가설 소유권 이관: BC visual feature optimizer

D는03:07 measure_gradients.py작성확인,수집완료아직없음. B는계속idle이고BC feature optimizer대조산출물미확인이므로새작업생성없이A에이관했다. B의같은가설은대기열에서제외하여중복실험하지않는다. A기존보상분석과분리된후속이다.

근거는총괄membership계측파일과imitation.py의명시목록누락이다. 활성visual_feature_encoder만추가하는단일변수BC: C저장472자료/기존.75gas label/선택초기actor3ceb5e.../seed8101/3epochs/batch32/lr1e-3동일,양팔control기존목록대treatment+visual encoder. temporal=false유지,normalized loss·fresh Adam PPO·D reward/gradient변경혼합없음. 데이터재수집없고원본root코드수정없이A전용runner. 양팔BC총6epochs,동일C tune각최대400decision1회,추가PPO없음. source/label/hash와feature parameter실제변화를확인한다.30분상한,리뷰없음.

판별: featureweights가실제로변해도tune완주·충돌이개선되지않으면누락이현재성능병목이라는주장은지지되지않는다. 훈련loss감소만으로승격하지않고서로동일단위loss인지확인. 무시된branch가원래정책에서유용했는지도이실험전단정하지않는다. C완료작업은대기,D기존진단은재시작하지않는다.

### 02:57 D source 해소 확인과 진단 범위 한정

D의source-resolution-correction.json/protocol.json 직접확인. 루트13개source hash 일치,visual=true메타데이터strict load성공및실제import __file__기록으로초기blocker해소. 현재protocol작성까지확인했고gradient결과/실제수집완료는아직없다. A/B변화없음,C출력비교완료. 진행중진단재시작/추가요청없음.

프로토콜은512행수집후첫256행을연속32행×8batch로측정한다. 인접batch는독립시행이아니고첫geometry/주행국면에치우칠수있으므로8개batch의음수cosine개수를8개독립맵이나8seed재현으로표현하지않는다. 실제행별map/seed와terminal/cutoff는저장되며자료가포함한구간으로해석범위를제한한다. advantage정규화모집단512행은C학습의1024행과달라진단gradient를원학습gradient의정확한재현으로쓰지않는다.

코드상action_log_std는관측과독립적인policy_log_std파라미터를expand한값이다. 현재entropy는이log_std만사용하므로shared encoder에대한entropy gradient는0인것이예상된다. 이때cosine은NA이며계측실패나탐색효과부재로해석하지않는다. entropy는policy_log_std를통한탐색변화에는영향을줄수있다. 본예상은코드구조에대한사전판별기준이며D실측결과를대체하지않는다.

### C Adam 처치 유효성 후처리 회신

C가저장된두4096checkpoint와동일472관측만비교했다(새학습/rollout없음). 담당회신상두모델step23552/동일BC출발,모델state hash는fresh a62f5290.../restored97f8b821...로상이. parameter delta L2=0.0272671,relative L2=0.00147733,maxabs0.00111340. 472행모두행동출력차이가있으며steer/gas/brake mean absolute delta는0.00723137/0.00036197/0.00017268. 따라서state초기화처치는실제다른모델과행동을만들었지만동일tune요약실패를개선하지못했다. 요약값동일은전체궤적동일이나일반적성능동등성증명이아니다.

fresh policy-step-4096.pt 파일SHA e19b2dd82e65edd7f342c15bde644dee4eb0751a50476f49bc862e590e408cbc와앞서policy.pt SHA66ade45...는파일명이달라구분한다;두경로의파일hash를같은것으로쓰지않는다. 후처리loaded model hash는결과의fresh model hash와일치한다고보고됐다. 근거파일C attempt2/checkpoint-output-delta.json,commit e98e49a는담당회신기준. 추가PPO배정없고D gradient진단대기.

### 02:47 완료 대기 중 후속 대조의 변수 정의

C는두checkpoint/472관측출력대조중,D는수정된source경로사전확인중으로새완료결과없음. A/B변화없고중복요청없음. 총괄이fresh Adam updates.jsonl 마지막행직접확인: update4 value_loss13.78018,policy_loss-0.001010,auxiliary_loss0.40769,grad_norm4.24475. PPOUpdater의grad_norm은clip_grad_norm_ 반환값의minibatch평균이며clip기준0.5이다. 일부업데이트에서clipping이작동했음을시사하지만각항기여/클리핑빈도/Adam후실제이동은이평균만으로알수없다. scalar value항이크다는이유만으로간섭원인을확정하지않는다.

조건부후속설계(미배정·미실행): D가value공유gradient상충신호를확인한다면value coefficient를0으로만드는대신value_head에입력하는latent만detach한학습대조가더직접적이다. 현재forward는latent를PolicyOutput에반환하므로value_head(output.latent.detach())로head학습은유지하면서공유encoder로향하는value gradient만차단할수있다. policy/auxiliary/reward/label/optimizer는고정,원래forward runtime동일,critic은계속훈련전용이다. aux상충만나오면value를바꾸지않는다. coefficient0은critic학습까지제거해GAE오류와혼합되므로이질문에부적절하다. detach도이후표현·critic예측·advantage를바꿀수있어전체학습동역학효과와분리되지않으며단일seed성공을일반화하지않는다. D진단후에만실행여부결정한다.

### D gradient 진단 source 경로 혼동 정정

D가C checkout HEAD의모델로strict load하다visual encoder누락/fusion128대160불일치로0decisions중단했다. 총괄이C runner 두개를직접읽어LOCAL_ROOT=C:/Users/koi/Coding/HAIC 및sys.path우선삽입확인. 실제학습source는루트작업트리, C TASK_ROOT는runner/산출물위치다. 총괄배정의경로표현이혼동을만들었으며C HEAD와manifest가다르다고곧바로manifest가stale이라고할수없다.

D에blocker원본보존·정정추가,실제루트파일hash와manifest일치및로드__file__확인후원512decision진단재개배정. 메타데이터visual=true strict load유지,부분load/모듈수정없음. 불일치시구체적파일을보고한다. 기존수집0이므로예산확대없음. 이오류는gradient간섭유무나C학습유효성에대한반증이아니다.

### C fresh Adam4096 완료: 단독 초기화로 완주 회복 안 됨

C attempt2/result.json 직접확인:4096decisions/4updates완료,checkpoint SHA66ade45aaaa0ff564deab368ba889f8daf7fd0bca9ac3da439d33fb6c204646f,model state a62f5290b852c7724e5c1a4396840d825550b90f1f0e4f7c09a6ea45e4599df4. 담당회신의동일tune 결과는restored와동일한177decisions/off_track/progress0.255144/collision4/damage0.8,13초미만없음. 따라서이한seed/조건에서Adam전체state초기화만으로BC완주가복구되지않음. .75후보보존,8192확대없음.

aggregate가정확히같아C에기존두checkpoint와저장472관측만이용해tensor L2/maxabs/행동출력차이및평가로드경로hash를10분내후처리요청했다. 기록된model state hash는restored97f8b821...와이미다르지만hash차이만으로주행차이나인과효과를판정하지않는다. 새rollout/학습없음. D에는결과만공유하고현재shared gradient 진단범위유지.

### 02:37 진행 확인 및 gradient 결과 판별 기준 보강

C attempt2는02:36 별도출력디렉터리생성,회신상TRAIN_MAX_DECISIONS600과원control동일성사전확인후실행착수. 아직완료파일없음. D는기존고정batch유무조사단계이며gradient측정완료로보고하지않는다. A/B는idle,중복지시없음.

현재rollout.py는advantage만batch단위정규화하고returns는그대로value MSE에쓴다. 따라서policy/value scalar loss의크기차이는정의·단위부터다르며정책을망치는증거가아니다. 공유gradient의계수적용norm/cosine이우선이고,전체gradient clipping은합계벡터에한번적용하므로특정항을독립적으로잘라준것으로해석하지않는다. 이후Adam이좌표별state로변환하므로raw gradient norm우세도최종parameter update우세와동일하지않다. D의현재무업데이트진단범위를확대하지않고이한계를결과해석에적용한다.

후속분기: fresh Adam만으로완주가회복되면optimizer state효과의한seed신호로보고C에서같은조건재현을우선검토; fresh도실패하면서D에서공유gradient상충이관측될때만value/aux의공유표현영향을분리하는단일변수학습을설계한다. 둘다신호가없으면계수임의탐색이나PPG전면도입으로넘어가지않는다. 작은batch의negative cosine은장기성능퇴행인과증명이아니며양쪽성공/실패모두독립재현전확정하지않는다.

### C fresh Adam 첫 실행: 학습 전 설정 참조 오류

C 회신상사전동일성검사는통과했으나wrapper module에서TRAIN_MAX_DECISIONS를찾는AttributeError로rollout전중단. steps0/updates0,평가나새checkpoint없으므로fresh Adam성능에대한증거가아니다. 총괄은이미로드한training_pilot.TRAIN_MAX_DECISIONS로참조만수정하고control과값동일성을확인한attempt2를배정했다. attempt1 result/manifest/runner를보존하고새hash별도기록,초기actor/state/seed동일. 소비학습0이므로누적예산4096유지,baseline재실행없음. 학습NaN/crash시중단조건유지. 원인확인된runner수정에한정하며추가학습배증이나무제한재시도가아니다.

### 02:27 독립 가설: 공유 특징에서 정책/가치 gradient 간섭

C fresh-Adam 프로토콜/runner(02:22/02:26)생성확인,회신상기존4096 prefix·source·초기모델·schedule·평가seed일치하여대조재실행불필요. 아직새팔완료증거없음. A/D기존분석종료,B결과없음.

총괄이현training/ppo.py와networks.py직접확인: policy/value/auxiliary head가fusion latent를공유하며loss는policy+0.5value+0.1auxiliary-0.01entropy,전체파라미터gradient clip이다. 따라서BC운전이PPO후무너지는별도가설로공유특징gradient간섭을검토한다. [Cobbe et al., Phasic Policy Gradient](https://arxiv.org/abs/2009.04416)는정책/가치학습간섭과단계분리를다루지만HAIC에서의효과를입증하지않는다. PPG전체구현이나별도네트워크도입은아직제안하지않는다.

D 기존작업01a0ceca-0f09-7571-a83e-6d1f80ffe655에별도진단배정. C normalized BC41a38b70...와동일source의고정batch에서shared parameter gradient norm(계수전후),policy대value/aux cosine을측정. optimizer.step없음. 자료없을때만같은train2geometry/seed8101에서512decisions한번수집,고정32행minibatch8개,30분상한. batch/sourcehash와terminal/truncation보존. zero gradient cosine은NA,scalar policy loss가0이어도gradient가0은아님. 이실험은회복보상D와별도소유범위이며C학습변경없다. negative cosine/큰value gradient가나와도성능인과확정은아니고,반대결과면해당batch의간섭가설을약화시키는근거로쓴다. actor-only/훈련전용labels유지.

### A 보상 분석 종료 회신: 계수 증대는 근거 부족

A는trace-run.log외per-step자료미저장,두TRACE_DONE후요약KeyError를확인했고재주행하지않았다. 같은train geometry의두정책주행이며두독립트랙이아니다. brake보상이전체aggregate를압도하지않으며회피개선은기존screen에서혼합,clean랩은더느리다. 담당제안brake6→8은제동보상부족인과근거가없고반복보상위험을키울수있어채택하지않았다. 구간별위험/제동분해는자료부족으로닫고추가계수탐색·재주행없음. A는후속대기,C fresh Adam4096 판별실험을진행한다.

### 02:17 C PPO 퇴행 확인 및 Adam state 대조 배정

C physical-longitudinal-loss-ppo-v1/result.json 직접확인. normalized BC는동일tune20.38초완주였지만PPO4096은progress0.25514/off_track/collision4/damage0.8,8192는progress0.25926/crash/collision5/damage1.0으로모두미완주. 각각last SHA92c3018b70357d185bd6a1d29fd12fa8d1ba2c1d25564380b1bfca3f9855fec7 및801d368f6edf811102048ac2f0e60f3027034c71808478de3acf6a3a52c6de39. 8updates/8192학습은완료됐고finite loss는주행개선을뜻하지않는다. normalized loss의BC개선이PPO전이에서유지되지않았으므로현재후보기각,예산배증없음. .75기존후보보존.

다음C 단일변수는BC후이전PPO Adam state 재사용대fresh state. 동일BC41a38b70.../source/train/seed8101/lr2e-6/하이퍼파라미터에서moment와step만초기화한팔4096decisions/4updates한번;기존restored4096을대조로재사용한다. 기존8192의첫4096과schedule/평가RNG일치확인,불일치면보고하고임의baseline재실행없음. last4096저장과동일tune평가,NaN/crash시중단,8192확대없음. Adam 전체state효과이며moment방향과bias-correction기여는분리하지못한다. 원인확정이아닌판별가설이다. BC/보상/label/featureoptimizer는바꾸지않는다.

A는작업완료표시이나산출물은후처리KeyError를포함한trace-run.log만확인돼한번blocker/결과경로요청했다. 두주행재실행없이기존자료후처리만허용하며자료없으면aggregate범위로닫는다. D분석commit bf234f26b345cfa902719e12e311f10566cacd8c는담당회신기준로컬보존,성능증거는원시주행별로유지한다.

### D 회복 경험 자료 분석 종료: 성공 사례 부재는 미확정

D 회신: matched PPO stdout/result는episode별decisions/gate counts/reward sums만보존하며gate exit·후속tile progress·terminated/truncated·terminal reason·transition trace가없다. 따라서stochastic학습에서회복성공이없었다고판정할수없고,update cutoff와실제종료도분리할수없다. action gates는update1 obstacles100+aug30,update2 obstacles99,update4 aug101;potential은update1 obstacles100+aug30,update2 obstacles99,update4 obstacles57. 모두gate내zero-progress라는집계이며그이후상태에대한증거가아니다.

별도deterministic평가의101-step정체는해당정책·상태의실패만보인다. 성공회복자료부재가학습병목이라는가설은미확정으로닫고,그가설을근거로demonstration학습이나기존rollout재구축을시작하지않는다. D이번bounded분석은종료,새증거가생기면기존작업을재사용한다. A보상분해/C고정예산PPO가현재실행담당이다. 다음실제로필요한새학습에서만gate진입·이탈및후속진행/종료종류와last checkpoint를함께보존하도록하며진단인프라만의추가작업은만들지않는다.

기록출처: D experiments/strategy-d-reward-potential/analysis.md, result.json의matched_training_recovery_trajectory_audit 및matched-resume-pilot/{action,potential}-seed8101/stdout.json. 본항목의자료충분성결론은담당의원본분석회신기준이다.

### D 마지막 체크포인트 미보존 확인 회신

D 담당이각run폴더에policy.pt만있고별도step23552/last파일이없다고확인했다. stdout은양쪽4096decisions/4updates/training_final_step23552를기록하지만저장action은step21504/SHA256 b0ef38eb1915d8085d6cfd39e7f90e094108c46d54c9d5f3c94fbc895b979f6c,potential은22528/SHA256 28e55cbc9d55482445d0e41ea4a0f6e1bd93d0258553fa1252bfe056620e2ffb이다. 이파일유무확인은담당회신기준. 동일dose최종모델비교는현재산출물로불가능하며재학습복원하지않는다. 기존주행결과는서로다른tune-selected step후보비교로보존. 이후새로운paired학습을실제로수행할때는선택본과별도로사전지정step의last저장을계약에포함해야한다. 현재진행중C는이미4096/8192 last계약이있어중복지시하지않는다.

### 02:07 출처 정정 및 보상 가설 좁히기

**D checkpoint 비교 정정:** analysis.md와final_train_map JSON의actor_metadata를직접읽어action step21504(+2048),potential22528(+3072)확인. 각run은4096학습했지만평가policy.pt는best-tune선택본이며마지막step23552가아니다. 이전총괄기록의 '두final정책/동일4096체크포인트'는이사실로정정한다. 서로다른시점에서선택된후보의관측주행결과는유효하나동일dose reward-only 인과차이로해석하지않는다. D에이미저장된last파일유무만확인요청,없으면재학습하지않음.

A trace-run.log에base/last각주행완료와후처리KeyError(explicit_progress)가있다. 아직완성된분석JSON은없으며로그계측만으로잠정해석. 정상train맵양쪽무충돌완주,brake보상합0.217→0.327,native보상86.258→85.671,throttle패널티-1.632→-0.509,평균gas0.06629→0.06201,평균속도40.004→39.339. 이맵에서는brake항이전체보상을압도한다는가설을지지하지않는다. hazard×gas 패널티회피와속도감소의동반가능성으로질문을좁혔고,A에기존자료의구간별분해를요청했다. 종료progress·시간차이가있어합계차이를인과효과로취급하지않는다. 두주행을중복실행하지않도록전달.

C는6144/8192학습완료를보고,최종결과아직없음. D경험자료분석진행중,B변화없음. SOTA갱신없음.

### D 실제 train 맵 진단 완료: 회복 개선 미관측

총괄이 `final_train_map_deterministic_evaluation.json` 직접 확인. aug0-v0-obs에서 action/potential 모두step168 충돌1회,169~269 연속101decisions(8.08초)무진행,269 off_track종료,최종damage0.2. progress0.60976/0.61382이나회복우열근거는없다. normal train obstacle맵은양쪽clean완주21.22→20.76초. 네주행13초미만0/4. 동일train2조건에서완주1/2씩이며훈련성능과일반화증거를구분한다. 현재4096예산에서potential변경으로회복향상이확인되지않았고보상방식의일반적불가능을증명하지않는다.

D후속은15분내기존train자료만으로성공회복경험존재를확인한다. gate진입후같은episode에서새tile진행이재개·유지됐는지를terminal/cutoff와구분한다. gate내무진행빈도만으로gate를벗어난회복성공부재를추론하지않는다. per-step자료가없으면자료부족으로닫고진단인프라나학습을확대하지않는다. 다음가설은성공회복trajectory부재이며증거가지지할때만훈련전용recovery demonstration 대조를설계한다. B충돌전회피와범위를분리한다.

### 01:57 D 진단 회수와 평가 대상 정정

`final_checkpoint_deterministic_evaluation.json` 직접 확인. tune T3/202는 action23.32초/potential23.12초,양쪽완주·collision2·최종damage0.4. 추가official T2/102는23.56→23.72초,양쪽clean완주. 네주행모두recovery gate0,13초미만없음. 따라서속도개선은일관되지않고충돌감소증거도없으며회복성능우열은이평가로판정불가.

총괄이쓴 'tune+train2'가학습맵2개가아닌official track2로해석됐다. 현재matched학습맵은custom-track-aug0-v0-obs/custom-track-haic-obstacles-20260920(각seed20260920)이다. 해당맵누락을정정하고두final정책×두custom맵각600decisions,총4회만추가허용했다. 기존결과는보존,기존tune재실행·새PPO없음. 원계약팔당최대3episode에서이미2회실행후custom2회를추가하는범위변경을명시하며기존실험과동일평가집합으로합치지않는다.

A는기존trace에visual features/native reward/route labels가없음을회신했다. 이미허용한원래train1geometry(base/last각600이하)의두진단으로현재보상분해를진행한다. C는01:53 protocol과01:55 PPO runner생성확인,아직PPO완료원본없음. B는idle/새결과없어재촉하지않는다.

### A 보간 기각 및 정상 주행 보상 진단 연결

A 완료 회신 및 `weight-interpolation-seed8101/tune/verification.json` 직접 확인. 동일 tune6에서 base4/6, alpha.25 3/6, .5/.75/last8192 각1/6완주. .25는 T2/107의29.860초완주를 잃었고, 공통완주3행은 각각+120,+120,+20ms 느려졌다. custom 두행은같은geometry의seed반복이며독립맵2개로부풀리지않는다. .25 median20.00초가base21.57초보다낮은것은느린완주탈락에따른선택효과다. 신규완주와동일셀시간개선이없으므로보간기각,추가alpha나공통screen미실행. strict runtime load3개통과는성능증거와구분한다.

후속A는현재속도·제동shaping의정상/최초충돌전구간분해를담당한다. D의손상후recovery보상과범위를분리했다. 기존trace우선,없으면기존train1geometry에서base/last각600decisions이하만추가진단,30분범위,새PPO없음. 진행·속도부족·곡률완화·장애물제동항의상대크기와반복지급을확인하고다음단일변수를선정한다. 현재수식을적용한반사실이며과거checkpoint학습보상을복원했다고표현하지않는다. 즉시보상의행동선호를장기행동가치증명으로취급하지않는다.

### A 보간 사전조건 회신 수신

A 담당 회신 기준: base step19456와 last+8192 step27648의 model_state 키/shape/dtype가 모두 일치하며 25개 tensor 전부 float32, nonfloat buffer는 없다. split, HUD=false, visual=true, temporal=false, throttle/brake expansion=1.0도 일치한다. 동일 T1/73 관측에서 메모리 내 alpha=0 행동은 base와, alpha=1 행동은 last+8192와 bitwise 일치했다고 보고했다. 총괄이 원시 진단 파일을 직접 확인한 결과와는 구분한다.

이는 보간 구현의 사전조건 확인이며 성능 결과가 아니다. alpha=.25/.5/.75 생성 및 tune6 평가가 다음 단계다. 끝점 행동 일치는 한 관측의 검사이므로 전체 궤적 일치나 중간 정책의 안전성을 뜻하지 않는다. 이미 배정한 예산과 중단조건을 유지하며 추가 학습·중복 실행 지시 없이 결과를 회수한다.

### 01:40 조율: 진행 확인과 결과 해석의 반증 조건

이번 조회에서 C는 loss 대조 runner를 작성했고 source/checkpoint/optimizer provenance 일치를 보고했다. 472개 자료의 hash 일치는 실행 중 검증 예정이므로 아직 재현 성공으로 기록하지 않는다. D는 action 팔 4096 decisions 실행 중이라고 보고했고 완성된 update 결과는 아직 없다. A는 app active 표시만 있고 이번 조회에서 interpolation 산출물을 찾지 못했다. B는 idle이며 새 결과가 없다. 기존 실행을 재시작하거나 동일 지시를 재발송하지 않았다.

**D 해석 보강 — 노출은 결과에 따라 달라진다.** 두 팔의 recovery gate 총량만 비교하면 충돌 감소와 충돌 후 탈출 향상이 섞인다. 기존 원시 기록이 제공하는 범위에서 (1) 전체 decisions당 collision 수, (2) 충돌 후 저속 진입 횟수, (3) 진입 뒤 연속 무진행 길이와 종료 원인, (4) 총 완주·시간을 분리한다. crash/episode cutoff는 탈출 성공이 아니며 중도 종료된 정체 길이를 짧은 회복으로 세지 않는다. 충돌 후 사례는 정책에 따라 선택되는 집단이므로 조건부 차이만으로 동일 상태에서의 회복 인과효과를 주장하지 않는다. 두 팔 모두 gate=0이면 가설 미노출; action 팔만 노출되고 potential 팔 충돌이 감소하면 노출 감소도 탐색적 결과다. 현재 pilot의 팔·예산은 변경하지 않는다.

**C 해석 보강 — 손실 좌표 변경의 이득과 포화 한계.** normalized longitudinal을 u=tanh(z)로 쓰는 경우 MSE의 z 미분은 2*(u-y)*(1-u²)에 비례한다. 큰 pretransform 목표의 지배를 줄일 수 있지만 이미 포화된 잘못된 출력에는 작은 gradient가 생길 수 있다. 이는 수학적 가능성이며 실제 pilot 원인으로 확정하지 않는다. 서로 다른 목적함수의 최종 scalar loss를 우열로 비교하지 않고, 같은 472행에서 decoded longitudinal 오차를 공통 단위로 비교한다. 특히 371개 cap 가속 정답이 전체 평균을 지배하므로 제동 정답 행과 비포화 가속 행에서 오류가 악화되지 않았는지 기존 출력으로 분리해 볼 가치가 있다. 이는 후처리 해석이며 sampling/loss weight 변경 요청이 아니다. 학습 오차만 감소하고 tune 완주·충돌이 개선되지 않으면 PPO 예산 확대 근거로 쓰지 않는다.

현재 새 완주 결과나 13초 달성 증거는 없다. 위 조건은 후속 결과 수신 시 적용하며 SOTA를 갱신하지 않는다.

### C 최초 결과 수신 및 총괄 확인

#### 01:26 공통 screen 결과와 다음 loss 대조

총괄이 strategy-c/strategy-a-screen-v1/summary.json 직접확인. 동일runner base재측정2/4완주 median24.720초; .75후보2/4 median23.940초. totalcollision6→10, damage1.2→2.0, meanprogress0.77447→0.77014,13초미만양쪽0/4. 빠른완주후보로보존하나전체성능단일승자로승격하지않는다. 평가맵을학습으로이동하지않는다.

C 다음실험배정: physical targets/shared472data/초기ckpt/optimizer/seed/BC3epochs고정, longitudinal loss만pretransform MSE→normalized physical signed longitudinal MSE. steer는기존pretransform손실그대로. label·sampling·모델·상한변경은섞지않음. 동일tune의BC후성능과포화분포부터보고,개선신호없으면긴PPO를확대하지않는다. 이는loss좌표/수치scale대조이며항상우월한loss라는가정은없다.

#### 00:35 가속 label 대조 완료 및 actor 단계 연결

**C actor pilot 완료/총괄 결과파일 확인:** `experiments/strategy-c/actor-gas-label-ppo-v1-result.json`, 두팔각8192PPO. 단일tune base19.88초, 기존.75label BC18.88/PPO18.74완주. physical label BC미완주progress0.6296/PPO미완주0.46502. 현physical+pretransformMSE 후보는이조건에서기각, 실제값보존접근전체의불가능은아님. .75final checkpoint hash `eae03f40a62f20c3baa45d1918d43846cdad6c262467616623eab468fc73f5f4`, physical hash `3b7fab791f87aa5712034f587c1e3ddb1cd20e04973fc26732ae1f8bee202fe3`.

다음은.75후보를A공통4cell screen에서검증하도록C에배정. 같은환경/runner manifest를확인하고불일치하면같은runner에서base도재측정. A추가학습은유지하며명령/runner만C와공유. 단일tune개선을SOTA승격으로바꾸지않는다. 후보의13초달성은아직없다.

#### 01:06 완료 대기 중 label 수치 진단

총괄이 actor-gas-label-ppo-v1-label-audit.json의472행을읽어계산: physical gas가0.119999이상인행371개(78.60%). 현재network epsilon1e-6에서정규화가속0.75의atanh는0.973,1-epsilon은약7.254이다(해석계산값; float32실제반올림값과구분). 따라서대다수full-gas 정답의pretransform 목표가크게커지며 MSE지배/가속포화가능성이있다. 아직PPO최종결과전이므로기각확정하지않음. 후속loss대조를선택하면 같은physical targets/동일data를유지하고loss좌표만바꿔야한다. bounded-action MSE도tanh포화시gradient가작아질수있으므로자동해결책이라고가정하지않는다. margin target은가속정답자체를바꾸는별도실험이다.

이주기A/B에중복지시하지않았다. B는다시idle이나산출물미확인으로처리하고진행성공으로보고하지않는다. D의첫4manifest행은custom seed반복일수있어, 실행전이라면geometry중복제거최초4개로선택규칙을고정하고이미시작했다면재시작하지않도록전달했다.

C 진행 회신: .75 arm PPO8updates 완료/최종tune평가, physical-preserving arm BC후 tune0/1완주 progress0.6296, 이후PPO진행중. BC MSE .75쪽0.07961→0.00829와physical쪽16.7449→2.8139는정답스케일이다르므로직접비교금지. 새해석가설은physical정답이cap0.12에닿을때 inverse tanh target이epsilon에의해커져 pretransform-MSE 난도가달라지는것. label audit에서cap근접비율/target분포/epsilon후처리만요청했고진행중학습은변경하지않음. 양팔최종결과이후에만 bounded-action loss 대조등을선택한다.

C actor pilot preflight 회신: train2geometry 각첫seed로 raw teacher자료1회 수집, 양팔동일dataset. checkpoint step19456 weights 및PPO Adam 동일초기화. BC3epochs/batch32/lr1e-3, PPO추가8192 decisions/8updates/lr2e-6 양팔고정. 가속label 기존.75mapping 대 physical-preserving만 변경. tune geometry custom-track-haic-tune-20260922/seed20260922,400decisions에서 base/BC후/PPO후 평가, heldout미사용. 한geometry 탐색결과이므로 유망후 A공통screen에 별도 평가한다. RNG/shuffle동일, dataset/hash 기록 및 예정last checkpoint 비교를 요청했다. BC후 이전PPO Adam moment 복원은 양팔공통이나 stale moment 제한을 명시한다. B optimizer 변경은 혼합하지 않는다. 이 실험은 실제collector의 mapped상태분포 변화를 추정하지 않는다.

**mapper 의미 정정 확정(총괄 소스 직접 확인):** C가 '인자 하나만 받음, instance cap 미사용'이라고 회신했으나 이는 잘못이다. 현재 imitation.py SHA256 `4F299FA5FA37CAEFFB36A386A716A3B092F86630C4062204ABA16D5972D32C64`의 signature는 keyword-only teacher_pedal_range/pedal_expansion/cap_fraction을 받으며 실제 BC/DAgger collector는 teacher_pedal_reference(teacher)를 명시 전달한다. 이번 진단 runner만 `_teacher_action_tensor(action)`으로 옵션을 생략해 기본0.12 범위를 사용했다. 0.12교사 대조결과는 동일하게 유효하지만, 진단 기본호출을 실제 collector 전체의 의미로 일반화하지 않는다. C에 inspect signature/file 및 호출부를 구분해 기록하도록 전달했고 재실행은 요구하지 않았다.

**현재BC collector의 추가 인과 정정:** target을 bound한 policy_action을 실제 environment.step_transition에 적용한다. 따라서 '빠른 raw 궤적을 수집하며 label만 낮춘다'는 이전 가설은 이 collector에서는 성립하지 않는다. 현재 코드의 메커니즘은 수집주행 자체도 축소된 행동을 따른다는 것이다. 동일data에서 label만 변경하는 대조와, 변경label로 새 rollout자료를 수집하는 대조는 각각 정답과 상태분포 요인을 바꾸므로 분리한다. 과거 dataset에 이 코드가 사용됐는지는 provenance 확인 전 단정하지 않는다.

총괄이 `C:/Users/koi/.codex/worktrees/4fd3/HAIC/experiments/strategy-c/raw-mapped-gas-v1-result.json` 직접 확인. 같은 초기 관측 hash의 train geometry1개에서 raw teacher19.200초, gas-only mapped20.380초, 양쪽clean완주. gas최대0.12→0.09. 1.180초 손실은 label 변환만 적용한 단일geometry 메커니즘 근거이며 학습actor 결과가 아니다.

다음 C 단계: 같은 student range0.12, 초기checkpoint, teacher자료, optimizer로 gas label만 기존 상대축소 대 실제값 보존의 BC 및 동일PPO 예산 비교. 깨끗한0.12 teacher 사용으로 range확장 가설과 섞지 않는다. 경계 atanh epsilon 처리는 수치오차로 기록한다. train결과와 tune선택을 구분한다. 원본 JSON의 mapper 설명(instance maxgas 인자 없음)이 이전 감사 설명과 달라 실제 호출인자/소스hash에 따른 차이를 C에 확인 요청했다. 현재0.12실험을0.24교사 일반론으로 과장하지 않는다.

B는 completed/idle이나 app의 turn본문에 산출물이 없었다. 기존 worktree 결과 회수 후 feature encoder의 BC optimizer 포함여부 계측을 우선하는 후속을 보냈다. 다른 B 가설은 대기열로 좁혀 실행을 집중한다.

#### gas 상한 후속: 실패 결과 확인

총괄이 `C:/Users/koi/.codex/worktrees/4fd3/HAIC/experiments/strategy-c/actuator-gas-v1-result.json` episode별 결과를 직접 읽었다. target80 고정 gas0.12→0.24: simulator완주4/4→2/4, clean4/4→0/4. 독립 geometry는2개이며 각2회 반복이다. 장애물map은 control18.680초 clean에서 treatment progress0.25203, collision5, damage1, crash로 퇴행. 다른map은19.200→18.460초이나 collision1/damage0.2로 clean 실패. 평균속도45.15→46.02는 실패로 관측구간 길이가 달라졌으므로 대응 속도 향상의 증거로 쓰지 않는다. 가속 상한 증대 후보는 기각하며0.12 control 유지.

다음 실행은 이미 clean인 cruise80/gas0.12 teacher를 primary로 raw vs mapped 폐쇄루프 대표 geometry1개 대조다. 실패한gas0.24를 raw 기준으로 삼으면 mapping 효과와 teacher 자체 실패가 혼동되므로 보조 분석으로만 남긴다. 향후 빠른 교사 후보는 속도에 따른 preview/braking 거리 부족을 검증해야 하며 현재는 대기열 가설이다.

C가 실제 다음 대조 범위를 map2의1개 seed label, raw clean cruise80/maxgas0.12 대 **gas-only label-mapped teacher**로 고정했다. steering/brake는 raw 유지하고 기존 `_teacher_action_tensor`의 decoded gas만 적용한다. actor 학습/추론은 없다. 따라서 full-action mapping의 효과로 일반화하지 않는다. subprocess 사용 시 모듈 반복 기동 대신 persistent worker를 사용해 실행 지연을 피하도록 전달했다.

#### 00:15 조율: 학습 전에 label 변환 자체를 반증하는 대조

C가 확인한 seed 의미: custom `_create_track`이 procedural geometry를 대체하고 domain_randomize=False, 장애물도 static이므로 각 맵의 두 seed는 독립 환경이 아닌 결정적 반복이다. 독립 geometry 수는2로 확정해 보고한다(C 소스 조사 회신 기준). 후속 일반화에는 실제 geometry/장애물 배치 차이를 canonical hash로 구별하고, 같은 원본의 변형은 map family로 묶어 분할 누수를 막는다. 현재 단일변수 pilot의 맵 조건은 변경하지 않는다.

C v1 provenance 회신: source manifest `fbe21331399540962316950b89a83f988a62daa4e7d596c57b61a0cc2600c254`, environment contract `35ca08788bf82330f87ddb1556176b0606a649bc503a45efb32515fe1f265ac9`, teacher config `bb293475d001cb51c3d92acd7a19327268c59304576ee27b0d452148285c516b`. learned model hash는 null이다. 이 값은 교사 실험의 출처이며 PPO 모델 증거가 아니다.

A/B compact 조회에는 새 결과가 없고, C는 gas pilot 실행 스크립트와 수정된 결과 파일의 진척이 확인됐다. 기존 실행을 재시작하지 않았다. D는 아직 실제 threadId/산출물이 확인되지 않아 생성 접수 상태를 유지한다.

전이 가설 후속을 C에 구체화했다: 대표 train geometry 하나에서 raw teacher와 `현재 관측→teacher→현행 label encode/decode→환경 행동`의 폐쇄루프를 비교한다. 후자는 학습자의 근사오차 없이 label 변환만 적용하는 진단이다. 이미 기록된 궤적을 open-loop로 재생하면 상태 분포가 달라지므로 이 대조를 대체하지 못한다. 학생 가속상한0.12, cap_fraction0.75, 교사 최대가속0.24 예시에서는 교사 full-gas0.24가 학생 target0.09가 된다. 실제 실험 설정을 확인하기 전 현재 후보가 이 예시에 해당한다고 단정하지 않는다.

- 변환 teacher부터 느려지거나 실패: label/action 의미 불일치를 우선 수정할 근거.
- 변환 teacher는 유지되지만 BC actor만 실패: 표현·optimizer·자료 분포를 우선 조사.
- raw teacher부터 실패: teacher trajectory 개선으로 되돌아감.

이 대조는 현재 gas pilot 이후 대기열이며 아직 실행 결과가 없다. 추가 학습 예산을 소모하기 전에 실패 원인을 나누려는 실험이다.

**전이 단계 후속 가설:** imitation.py의 target 변환은 `teacher_gas / teacher_max_gas * student_throttle_limit * cap_fraction`이고 기본 cap_fraction은0.75이다. 교사 최대 가속만 높여도 full-throttle 정답은 같은 학생 가속으로 변환된다. 일반적으로 절대 행동값이 보존되지 않으므로, 빠른 교사 궤적과 느리게 변환된 label 사이의 불일치 가능성을 C에 전달했다. 현재 action range와 설정을 확인하고 실제 action→target→decoded action 오차를 측정한 뒤 전이 실험을 설계한다. 학생 range 안에서 절대 행동 보존과 기존 상대 정규화는 별도 단일변수로 비교한다. 아직 학습 성능 차이는 측정하지 않았다.

C의 read-only 계측 회신: 선택 actor는 expansion1, gas상한0.12, brake상한0.28. 교사 상한0.12의 실제gas0.12→target0.09, 교사 상한0.24의 실제gas0.12→target0.045 및 실제gas0.24→target0.09. 직접 physical action encode/decode 최대오차1.19e-7, 저장target 왕복1.49e-8로 수치적 가역성은 양호하나 물리적 행동 의미는 보존되지 않는다. headroom0.75는 의도된 설정이며 그 의도와 폐쇄루프 효과는 별개다. BC/PPO 개선 실측은 아직 없음.

다음 학습 대조는 동일 student range/초기화/예산에서 gas label의 상대 정규화 대 절대값 보존만 비교한다. 현재 range 밖 teacher0.24를0.12로 clipping하면 절대값 보존 대조가 되지 않는다. 공통 확장 range를 먼저 양쪽에 동일 적용하거나 range 안 행동으로 진단 범위를 제한한다. cap_fraction만1로 바꾸어도 teacher 최대값에 따른 정규화 문제는 남는다. 현재 gas pilot 이후 mapped-teacher 폐쇄루프 결과가 먼저다.

원본: `C:/Users/koi/.codex/worktrees/4fd3/HAIC/experiments/strategy-c/fast-teacher-v1-result.json`.
목표속도만62→80: map1 20.740→18.680초, map2 20.760→19.200초. 각 map 2seed가 완전히 같은 요약을 내므로 2개 geometry/4회 실행으로 표시하며 4개 독립 복제로 취급하지 않는다. 전체4회 중앙값20.750→18.940초, simulator 완주4/4→4/4, 무충돌 완주2/4→4/4, 13초 미만0/4→0/4. 기존 clean-only 중앙값의 2vs4 비교는 matched 개선율로 사용하지 않는다. treatment throttle cap0.12 포화율78.46%. C는 target80 고정, maxgas0.12→0.24의 다음 단일변수 pilot에 착수. 모두 train-only teacher 결과이며 PPO actor나 공식 기록의 향상이 아니다.

### 2026-09-24: 연속 운영 유지와 후속 진단

- haic-13 자동화를 10분 주기 ACTIVE로 갱신했다. 1등, 13초 미만 기록, 현 SOTA 달성은 종료 조건이 아니며, 사용자의 명시적 중지 요청 전에는 자동화나 작업을 취소·아카이브하지 않는다. 매 회차 등록 작업의 실제 결과를 확인하고 다음 판별 가능한 단일변수 가설로 잇는다.
- A는 이번 조회에서 새 결과가 없는 idle 상태다. D의 AR(1) 조향 시간상관 후보는 paired 결과 4행 중 3행에서 진행도가 낮고 변경 입력도 없어서 기각됐다. 같은 sweep은 재실행하지 않는다. 두 작업은 취소·아카이브하지 않았고, 다음 유효 가설/기존 진행 결과를 기다린다.
- B의 완료 보고에 따르면 paired BC 재현은 저장된 teacher 관측·행동 시퀀스와 실행 가능한 Python 3.11/Torch 환경이 없어 막혔다. 같은 학습을 반복하지 않도록 B에 10분 이내의 비변경 진단을 배정했다: 기존 데이터/환경의 실제 존재와 해시를 확인하고, 둘 다 있으면 단일 batch에서 encoder gradient 도달과 optimizer 포함 여부를 분리한다. 설치·다운로드·공유 코드 수정은 하지 않으며, 선행조건이 없으면 최소 unblock을 보고하고 중단한다. 아직 가설 확인 결과는 없다.
- C는 control의 4,096 training decisions 및 tune 1회 평가를 마쳤다(242 decisions, collision 1, damage 0.2). 같은 frozen checkpoint/seed의 mean-anchor 처리군은 학습 중이다. 짝 결과 전에는 우열을 말하지 않는다.
- 현재 이 turn에서 13초 미만 유효 완주 또는 SOTA 갱신 증거는 없다.
### 2026-09-24: SOTA obstacle failure와 속도 병목 원인 추적

SOTA 포인터의 원본 `artifacts/haic/final-ppo-actor-selection.json`과 해당 체크포인트의 per-decision trace를 대조했다. held-out은 6/8 완주, 중앙 랩 19.32초, p90 22.42초이며 episode 중앙 mean speed 약 40.4, max speed 약 53.0이다. act p95는 16ms로 5초 제한과 큰 차이가 있으므로 계산 지연은 랩타임 병목이 아니다. 같은 경로에서 13초를 내려면 단순 시간비 환산상 현재 평균속도보다 약 49% 높은 속도가 필요하다. 속도 summary는 episode별로 랩시간과 짝지어지지 않아 직선 구간의 단독 병목은 아직 확인되지 않았다.

Track1 seed42 + obstacle 원본에서 step53은 speed 41.83, 이미지 곡률 feature 0.333, obstacle_present 0, brake 0이었다. step55에 obstacle이 처음 보였을 때 거리는 15.87m, speed 41.89, urgency 0.806, gas 0.015, brake 0이었다. step59~63에 5회 충돌했고 해당 에피소드는 progress 0.1484에서 실패했다. 별도 Track1 trace도 first-visible 약16m, 위험 구간 brake 평균 0, high-risk gas 0.0735 대 clear gas 0.0837로 남아 있다. 따라서 여기서 확인된 직접 원인은 늦은 시각 신호와 미약한 제동 행동의 결합이다. step53 곡률 feature는 obstacle보다 먼저 오지만 그 크기가 현재 reward의 local target을 충분히 낮추지 않는다.

현재 `training/train_policy.py`는 비충돌 상태에서 `max(target_speed - speed, 0)`에만 비용을 준다. 곡률이 커져 local target을 낮춰도 실제 속도가 target보다 높을 때 감속해야 할 직접 PPO 비용이 없다. 곡률 feature 0.333, target 70, relief 0.4일 때 계산된 target은 약60.7로 trace의 speed41.8보다 높아, 이 상태의 보상은 오히려 속도 부족을 계속 벌한다. obstacle brake reward는 obstacle pixel이 보인 뒤에만 활성화한다. 이는 구현된 reward 식과 raw trace에서 함께 확인한 원인 가설이며, 학습 효과는 아직 실험 전이다.

기존 실험은 두 원인을 구분한다. 구형 safe-speed reward(target 50) 8,192-step PPO 실행은 tune 4/6 및 중앙 21.73초로 baseline과 같은 결과였고, 50 target 자체는 13초 목표를 압박하지 않았다. obstacle-brake PPO 후보는 하나의 held-out custom map family에서 5/5 완주·중앙 19.26초를 보였지만 mean speed 40.56, max 52.57로 느렸고 13초 미만은 0/5였다. 따라서 제동 보상은 완주 쪽 신호가 될 수 있지만 속도 해결을 입증하지 않으며, 해당 기록의 restriction status도 unknown이다. 최신 C mean-anchor paired pilot은 한 seed/한 tune episode에서 18.7초 대 19.3초, 충돌·damage 동일이었으므로 재현 이득으로 승격하지 않는다.

다음 단일변수 PPO 가설은 bend feature가 obstacle보다 먼저 나타나는 구간에 직접 학습 신호를 주면 제동 행동이 앞당겨진다는 것이다. A(01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1)에 current source/checkpoint의 동일 seed 8101, fresh optimizer, train split, 8,192 decisions/arm, 8 updates, lr 2e-6 matched pair를 맡겼다. 기존 모든 reward와 모델/관측/action mapping을 고정하고 `CURVE_BRAKE_REWARD=0` 대 `6`, 즉 PPO reward에만 `curve_magnitude * brake_fraction` 항을 추가한다. 예산 30분, tune-only 선택, held-out/official은 train/tune에서 제외한다. 완주, 13초 미만 유효 완주, 시간, 충돌·damage, 곡률 구간 및 obstacle 가시 전 brake를 함께 본다. 이 항은 actor runtime의 규칙이 아니며 제출은 계속 PPO actor-only다. 실행 중이며 결과는 아직 없다.

B(01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de)는 encoder optimizer 진단을 위해 기존 관측/action 자료와 실행환경의 존재·해시를 확인 중이다. C(01a0cec2-4799-7303-aba2-baa80f015189)는 mean-anchor pilot을 완료했고 현재 idle이다. SOTA는 바꾸지 않았다.
#### 속도 진단의 provenance 정정

선택된 SOTA 체크포인트의 experiment_config.json은 구형 보상을 기록한다: safe_speed 목표 50/가중치 0.1, obstacle throttle penalty 8, 양의 obstacle brake 보상 없음. 현재 미커밋 학습 코드는 target 70·shortfall·obstacle brake가 포함된 다른 설계다. 따라서 새 코드의 단방향 speed-shortfall 식은 다음 실험 가설이지 저장된 SOTA 실패의 원인으로 입증된 항목이 아니다. 저장된 모델의 직접 증거는 구형 설정과 충돌 전 brake가 없던 원본 trace다. A의 paired 비교는 두 arm 모두 같은 구형 체크포인트에서 시작해 현재 코드/설정을 공통으로 사용하고 curve-brake 계수만 바꾼다고 명시해야 한다. 또 19.32초 완주 중앙값과 40.4 평균속도 집계는 episode별로 짝지어진 값이 아니므로 49% 속도 환산은 같은 경로 가정의 대략치다. 직선 구간만 분리한 속도·행동 trace는 아직 없다.
#### 실행환경 범위 정정

프로젝트 루트 `.venv\Scripts\python.exe`와 그 base CPython 3.11.15의 `--version` 실행을 다시 확인했다. C의 완료 JSON도 이 `.venv`에서 Python 3.11.15, Torch 2.1.0+cpu, NumPy 1.26.0, Gymnasium 0.29.1로 8,192 PPO training decisions가 실행됐음을 기록한다. 따라서 B의 격리 작업에서 interpreter 실행이 실패한 것은 프로젝트 전역 의존성 부재가 아니다. 옛 paired BC 대조는 historical raw sequence가 없는 것이 남은 blocker라 같은 실험을 재지시하지 않는다. A의 PPO 대조는 루트에서 확인한 3.11 환경을 쓴다.
#### 장애물 반응 시간과 제동 거리의 거친 계산

현재 map은 50 FPS, 4 raw frame을 한 decision으로 묶어 decision 간격이 0.08초다. Track1 + obstacle에서 처음 보인 step55와 첫 충돌 step59 사이 네 간격은 약0.32초다. 속도 약42이면 그동안 직선 환산으로 13.4m를 이동하며, trace의 obstacle 거리는 15.87m에서 4.03m로 줄었다. held-out policy가 관측한 non-collision peak deceleration 약34 units/s²를 일정하게 유지한다고 가정하면 정지거리는 약26m다. 이는 차의 물리적 최대 제동 한계가 아니라 해당 정책 trace의 참고치지만, step55의 첫 시각 신호에서 시작하는 제동만으로는 늦다는 설명과 맞는다. 한두 단계 앞의 곡률 신호(예: step53, obstacle 21.54m, curve feature 0.333; step50, 28.35m, curve 0.095)에 반응하는지 A가 기록해야 한다. 확정 제동거리는 아니다.

#### 실험 split의 geometry 중복 감사 및 A blocker 해소

`training/maps/site/full_site_map_split.json`의 공식 맵을 `map_id`와 `seed`를 제외한 `(map_kind, track_id, obstacle_mode, obstacles, max_steps, frame_skip)` canonical JSON으로 비교했다. Track1 seed42/43–48은 모두 geometry SHA-256 `6133ba1a68ba02d0136e7469f9c0c764a1fbdc4d18fb6c0867fae435b03d9ea7`, Track2 seed101–107은 `0f30c5a9e43f56f23b62af062b0ec9b428ca075da25b29ecfcd0e11db9ea1adb`로 각각 동일하다. 따라서 seed tuple만 나뉜 full-site manifest는 공식 track geometry family를 train/tune/held-out 사이에서 격리하지 않는다. 이 파일을 해당 paired 실험에 그대로 사용하면 `RESTRICTIONS.md`의 held-out geometry 비사용 조건을 충족한다고 볼 수 없다.

이번 A 대조는 등록된 `training/maps/site/site_map_split.json`으로 한정한다. 이 split은 train 2개 custom map family, tune 1개 별도 custom family의 seed 20260922/20260926, held-out 1개 별도 custom family로 구성된다. A에는 두 train family만 rollout/training에 사용하고, tune의 두 seed만 selection에 쓰며, held-out custom과 모든 공식 seed·obstacle 평가를 후보 선택 뒤 최종 확인에만 쓰도록 전달했다. Tune 두 seed는 같은 geometry의 반복이지 독립 map 두 개가 아니다. 양 arm은 과거 official-seed 데이터가 포함되었을 수 있는 동일한 SOTA checkpoint에서 출발하므로 그 checkpoint provenance를 명시하고, curve-brake 변경 효과는 이번 custom-only continuation pair 안에서만 해석한다. 기존 seed8101, fresh optimizer, 8192 decisions/arm, 8 updates, lr2e-6, `CURVE_BRAKE_REWARD=0` 대 `6`, 30분 cap은 그대로다. 실행 전 split/source/checkpoint/config hash를 남기고 성능과 완주·충돌·damage·속도·곡률/장애물 가시 전 제동량을 함께 보고한다.

A worktree의 custom-only split은 위 항목과 일치한다. root와 A 파일의 raw SHA-256은 각각 `3752f33a166e537aac0afa6cbb63618467ee3bb75ec7114df734e33d2bb1d5b2`와 `49e1774b2ff039fb601903070e2976951c974a02f16fc2542754260574bea209`로 다르므로, A에는 자신이 읽는 worktree 파일의 raw hash를 기록하라고 덧붙였다. A worktree에는 `full_site_map_split.json`이 없으며 root의 full-site manifest를 복사하지 않는다. A의 task는 여전히 active/in-progress로 표시되지만 이 확인 시점에는 새 curve-brake runner/result artifact가 확인되지 않았다. split 답변을 전달했으며 기존 turn을 재시작하지 않고 실행 진척을 기다린다.

A의 1회 blocker 회신: 학습은 아직 시작 전이고 PID/session/result JSON은 없다. 별도 worktree `C:/Users/koi/.codex/worktrees/strategy-a-curve-brake-reward/HAIC`에서 runner wiring/preflight를 만드는 중이라고 확인했다. train-only reward/action actor-only 변화와 테스트·계획 파일은 그 격리 경로에만 두고 root/SOTA/RESULTS는 수정하지 않는다고 보고했다. 남은 작업은 hash를 기록하는 runner wiring 및 preflight이며, 실험 예산은 30분으로 유지한다. 총괄은 고정된 pair를 실행하도록 회신했고, 이후 진행 판단은 실제 runner/artifact/PID와 결과 JSON으로 한다. thread의 active 표시는 단독으로 학습 중이라는 증거가 아니다.

현재 shaping과 제안 treatment의 크기도 trace에서 산출했다. step53의 SOTA trace는 speed41.83, curve feature0.333, obstacle 미가시 상태다. 현재 code constants `SPEED_TARGET=70`, `SPEED_SHORTFALL_PENALTY=0.6`, `CURVE_SPEED_RELIEF=0.4`를 대입하면 local target은 약60.68, one-sided shortfall cost는 약0.162/decision이다. 제안된 treatment의 curve-brake reward는 이 위치에서 `6*0.333*brake_fraction = 1.998*brake_fraction`; full brake에서 shortfall cost의 약12.4배, brake fraction 약0.081에서 shortfall cost와 비슷하다. 이는 treatment가 학습 신호를 낼 정도로 크다는 계산인 동시에 코너 전반에서 과제동해 랩타임을 악화시킬 위험을 뜻한다. 이 계산은 기존 SOTA trace에 새 reward를 대입한 반사실이며 새 정책의 행동이나 성능 증거가 아니다. A 결과는 pre-obstacle brake 증가 여부와 함께 clean tune 시간·완주/충돌을 보아 안전 개선과 과제동 속도 손실을 구분한다.

#### A runner 범위 선행 확인

A worktree의 실제 git status에는 `training/train_policy.py`와 `tests/test_train_policy.py`, 계획 외에도 `agent.py`, `haic_agent/networks.py`, `haic_agent/planner.py`, `training/evaluate_closed_loop.py`, `training/imitation.py`, `training/rollout.py` 변경과 `haic_agent/pixel_features.py` 추가가 보였다. 이는 A의 “보상 변수와 runner wiring만 변경”이라는 설명보다 범위가 넓다. 총괄은 이 파일들이 runner/model/obs/action/eval에 소비되는지 확인하고, frozen pair에서 관측·행동·평가 경로가 달라지는 변경은 사용하지 말도록 실행 전 선행 확인을 보냈다. 현 OS snapshot에서는 해당 worktree에 매칭되는 Python/uv 학습 프로세스가 없었다. 단일변수 조건이 해소될 때까지 재시작이나 별도 학습을 하지 않고 해당 worktree와 A turn을 유지한다.

추가로 SHA-256을 root와 A worktree에서 비교해 범위를 확인했다. `agent.py`, `haic_agent/networks.py`, `haic_agent/planner.py`, `haic_agent/pixel_features.py`, `training/evaluate_closed_loop.py`, `training/imitation.py`, `training/rollout.py`는 각각 byte-identical이다. 의도된 source delta는 `training/train_policy.py`(root hash prefix `6f5e0c44a393`, A `5c07674c6301`)이며, test 변경은 보상/init 검증용이다. 따라서 추가 파일들은 runtime 변경이 아니라 현재 root snapshot의 복사본으로 판명됐다. 대신 A의 계획문서 첫 Goal 문장에 tune 6 episodes가 남아 있어 custom-only contract의 2개 tune seed와 모순된다. A에 “1 geometry × 2 repeated seeds”로 계획·실행 manifest를 맞춘 뒤 시작하라고 전달했고, 그 외 source hashes를 기록한 후 기존 single-variable pair 진행은 승인했다.

A가 계획과 실행 경계를 다시 확인했다. runner는 `training.train_policy.train()` 및 `evaluate_policy(capture_trace=True)`를 사용하고, 두 팔에서 같은 `VisualActorCritic`, observation/action contract, env/rollout/PPO source를 고정한다. `agent.py`는 strict-load preflight 용도만, planner/evaluate_closed_loop는 pair 경로에서 호출하지 않는다. imitation module은 import chain에 있어도 warmup=0이라 teacher/BC 함수를 호출하지 않는다. strict actor load는 25 state entries에서 성공했다고 보고했으며 model-state SHA-256 `d1dd49dc21fc247c5bcaa8891c43d6e73336eb13a25cf766ed673b05595ac6b4`, structure SHA `3ed47f6289bf221fa482ceff6a6cd172e787b6b311090a2c2972094ca6a4f6a6`, observation/action contract SHA `5ccc986c1efbf5e48fc5b52fdb3ab6fe31fae515f60ddccc57d2dab9815f7008`을 제시했다. contract는 normalized float32 4×84×84 frames, pixel feature branch true, HUD/temporal false, deterministic 3-action output, gas max0.12/brake max0.28, expansion1, planner false다. 이는 총괄의 독립 재계산이 아닌 담당 회신이므로 다음 source manifest에서 재확인한다.

현재 A 입력 artifact를 직접 확인: SOTA policy 입력은 root checkpoint와 byte-identical이며 SHA-256 `3ceb5ebbe2a4bd96944649a693b8ef924c0252fa620b0fd547c7fb494afff199`; 두 train map 복사본은 각각 `a4b10ffd848a7b35827bb9d90c311396169ce30b1bffd6614b1cd44c80c0ad16` 및 `cff9b8ca1f758adfdc3d6c5df3f50e88ef566b51a6bbc9a7a544c3a83bc89c0f`, tune map은 `15d2459bd1f02db1be4166a06bf1770348fcb954b686fa6bce5fbf9da1fb2154`로 root와 일치한다. split 파일은 bytes hash가 root와 다르지만 항목은 같고 A의 SHA는 `49e1774b2ff039fb601903070e2976951c974a02f16fc2542754260574bea209`다. materialized input directory에는 이 세 custom map만 있고 held-out map geometry와 공식 geometry는 없다. 아직 source manifest 및 pair runner/result가 없고 학습 프로세스도 확인되지 않았다. source-manifest 생성 뒤 hash와 실제 loaded map 목록을 보고 학습 실행/결과를 판정한다.

A가 계획문서의 tune 수를 두 seed로 바로잡았고 실제 runner 구성은 별도 manifest 전에 계속 준비 중이다. B(기존 `01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de`)에는 중단된 historical DAgger 수집을 재시도하지 않도록 했다. 대신 이미 완료된 C의 compliant custom train/tune PPO 결과를 읽어 update 역학과 성공/실패를 대조하는 제한된 offline 분석을 배정했다. 공식/held-out trace 접근, 새 환경 평가, 학습, 테스트, 코드 및 DB 수정은 금지하고, 학습량 부족 대 보상/행동 불일치 중 어느 쪽인지 근거·반증·후속 actor-only 단일변수 가설 하나를 회신하도록 했다. A 학습과 자원을 공유하지 않는다.

#### A 평가 표본의 실제 해석 (2026-09-24 09:00 KST)

A의 `run_pilot.py`와 `training/train_policy.py`를 읽어 실행 프로토콜을 확인했다. 설정의 `selection_evaluations_per_arm=8` 항목은 runner에서 참조되지 않는다. 실제로는 각 팔에서 PPO update 8회마다 동일한 tune split 2개 에피소드를 최대 400 decisions로 평가해 체크포인트를 고르고, 마지막에 선택된 체크포인트를 같은 두 에피소드에서 최대 600 decisions로 trace를 포함해 다시 평가한다. 두 tune seed는 동일 geometry 반복이므로 통계상 unique tune 사례는 팔당 2개 episode / geometry family 1개다. `8`은 독립 평가 표본 수가 아니다.

이 재사용은 양팔에 공통이어서 탐색용 matched 비교는 가능하지만, 반복된 tune 평가로 checkpoint를 선택한 만큼 최종 tune 성능을 일반화 증거로 해석하지 않는다. A에 실행 중 config/소스 해시는 보존하고, 결과에는 unique n=2를 표시하며 update별 selection과 final trace를 따로 기록하라고 전달했다. 승격 여부는 기존 gate까지만 판단하고 held-out/공식 평가는 별도 승인·예산 범위에서만 다룬다. 현재 A의 실제 Python 3.11 프로세스 PID 16248은 PID 32680의 child process이며, `control/policy.pt` 파일 생성으로 control arm 실행을 확인했다. 이는 아직 성능 결과가 아니다.

#### A control 팔 부분 체크포인트 감사 (2026-09-24 09:07 KST)

A가 알린 중단 사유는 control이 첫 update도 마치지 못했고 `policy.pt`도 없다는 내용이었다. 실제 파일과 프로세스를 독립 확인한 결과는 달랐다. 09:07에 해당 실행의 Python 프로세스는 사라졌지만 `control/policy.pt`는 남아 있었고, `treatment/`, `paired-comparison.json`, control의 `run_record.json`, 최종 튠 trace는 없었다. 체크포인트를 읽기 전용으로 열어보니 8,192 중 7,168 step, PPO update 8회 중 7회가 완료됐고, 체크포인트 선택용 튠 요약은 반복된 2개 사례에서 완주 2/2, 충돌 0, 평균 진행도 1.0, 중앙 랩타임 19.92초였다. 이는 선택된 중간 체크포인트이지, 8,192-step control 완료 결과나 마지막 600-decision 튠 trace가 아니다.

초기 SOTA actor와 비교했을 때 파라미터 78,383개 중 19개 tensor가 바뀌었다. 전체 parameter delta L2는 0.05845, 초기 가중치 L2는 18.31254로 상대 delta는 0.00319다. PPO update가 actor/encoder와 critic에 도달했다는 뜻이지만, 주행 정책이 유의미하게 바뀌었다는 증거는 아니다. A에 부분 산출물을 보존하고 체크포인트와 중단 사유의 불일치를 확인하도록 요청했다. 처리군이 실행되지 않아 계획된 paired 효과는 아직 측정되지 않았다. 19.92초 튠 값은 체크포인트 선택에 반복 사용된 단일 geometry 결과이므로 독립 SOTA 개선이나 13초 미만 달성으로 해석하지 않는다.

#### 같은 tune에서의 초기 SOTA 대 control 7-update 폐루프 비교 (2026-09-24 09:15 KST)

초기 SOTA actor와 중단된 control step-7168 체크포인트를 같은 두 custom tune seed에서 결정론적 평가했다. 두 seed 결과가 완전히 같아 독립 표본은 사실상 tune geometry 한 개다. 초기 SOTA는 19.88초, 평균속도 42.005, 최대속도 52.672, 충돌 2회/반복 1회였고, PPO control은 19.92초, 평균속도 41.972, 최대속도 52.462, 충돌 0회였다. 두 정책 모두 13초 미만 완주 0이며 완주율은 2/2다. 즉 안전은 나아졌지만 속도는 약 0.04초 느려졌고 사실상 그대로다.

속도 지연의 직접 단서는 clear·저곡률 구간이다. 평균속도는 44.053→44.062로 같고 평균 gas는 0.08438→0.08533으로, 허용 상한 0.12보다 낮다. 해당 구간에서 brake는 양쪽 모두 0이었다. 따라서 이 표본에서는 제동이 직선 속도를 깎은 것이 아니며, 정책이 허용된 가속을 충분히 쓰지 않는 문제가 남는다. 전체 parameter delta가 작았던 사실과 함께 보면, 현재 reward/학습량은 안전을 유지하는 방향으로는 움직였으나 속도 목표를 향한 정책 변화를 만들지 못했다. 단일 geometry·반복 튠 사용 결과이므로 일반화 결론은 아니다.

평가 요약과 체크포인트/맵 해시는 [원시 진단 JSON](research/artifacts/ppo-control-step7168-vs-sota-same-tune-20260924.json)에 저장했다. 다음 단일변수 후보는 actor/action mapping을 건드리지 않고 PPO의 `speed_shortfall_penalty`만 0.6에서 1.2로 올리는 paired 대조다. 이는 직선에서 관측된 낮은 gas 사용과 속도 정체를 겨냥한다. 곡선·장애물 제동 보상, 모델, action limit, seed, maps, PPO budget은 양팔에서 고정한다. 완료/충돌이 악화되면 속도 이득으로 인정하지 않는다.

#### A 부분 실험 정정 및 다음 PPO 대조 배정 (2026-09-24 09:18 KST)

A가 추가한 `artifacts/haic/curve-brake-reward-ppo-pilot-v1/interruption-status.json`을 확인했다. 중단은 자연 종료나 환경 실패가 아니라, 빈 output directory를 보고 operator가 session 63426에 Ctrl+C를 보낸 결과(exit 1, traceback 없음)였다. 실제 경과 약472초, 저장된 control은 7168/8192 step·7/8 update이고, treatment는 시작하지 않았다. config/source-manifest/SOTA/RESULTS 변경은 없으며 체크포인트는 보존됐다. 따라서 curve-brake pair의 결론은 없고, 중단 사유는 진행률 확인 오류로 분류한다.

다음 A 단일변수 가설은 PPO reward 안의 `speed_shortfall_penalty`를 0.6 대 1.2로 바꾸는 것이다. 이는 tune 저곡률 구간에서 gas 평균이 0.085로 action 상한0.12를 밑돌고 속도 개선이 없었던 관측을 겨냥한다. 동일 SOTA actor, seed8102, fresh optimizer, arm당 8192 decisions/8 updates/lr2e-6, custom train 2 geometry/4 episode, tune 1 geometry/2 반복 seed로 고정했다. 30분 총 wall cap, actor-only, teacher/규칙/runtime 변경 없음. A에 기존 부분 디렉터리를 덮어쓰지 않고 새 결과 경로 및 source-manifest 해시를 고정하라고 보냈다. 상태는 아직 준비 단계이며 프로세스 시작 확인 전이다.

C에는 A와 겹치지 않는 두 번째 PPO 기제를 설계하도록 배정했다. 같은 결과에서 longitudinal action exploration 또는 update-scale 계열의 단일변수 가설 하나만 제시하고, 예측 행동/속도 변화와 완주·충돌 반증 기준, custom-only 30분 비교 설정을 보고하도록 했다. 분석만 허용했고 학습/코드/테스트는 배정하지 않았다. C worktree의 `train_policy.py`와 `networks.py`가 A worktree와 다르므로 최신 실행 소스로 오인하지 않도록 해시 차이도 명시하도록 요청했다.

#### PPO action-range 병목 재검증 및 보상 가설 우선순위 변경 (2026-09-24 09:28 KST)

A partial checkpoint에 저장된 rollout은 현재 throttle limit 0.12, 평균 gas 0.0683, 최대 gas 0.1170, saturation fraction 0.0이었다. 코드의 saturation 지표는 limit의 98% 이상만 세므로 0.1170은 경계에 가깝지만 기준 바로 아래다. 더 직접적으로 `haic_agent/networks.py`는 throttle limit을 `MAX_GAS(0.12) × throttle_expansion`으로 계산하고, expansion1은 0.12 상한이라고 명시한다. 같은 파일의 track 측정 주석은 gas0.12에서 40 decisions 후 speed42.5, 직선 최대59–71 수준이며 gas0.42에서 speed80이 가능했다고 기록해 action transform ceiling이 13초 목표에 제약이 될 수 있음을 뒷받침한다. 이 구현 주석은 근거 있는 가설이지 현재 경쟁 트랙에서 재검증된 결과는 아니다.

확인을 위해 SOTA model_state를 바꾸지 않고 throttle expansion만 1.0→2.0으로 바꾼 deterministic tune screen을 실행했다. 같은 tune geometry 반복 2 seed에서 expansion1은 median19.88초, mean speed42.005, collision2였고, expansion2는19.50초, 42.739, collision6이었다. 완주율은 둘 다2/2, under13은 둘 다0이다. clear-low-curve gas는0.08438→0.09840, speed44.053→45.081로 증가했고 brake는 양쪽 모두0이었다. action 범위를 넓히자 속도는 0.38초 빨라졌지만 충돌도 늘었다. 한 geometry의 무학습 변환 결과이므로 정책 개선·일반화로 주장하지 않는다. 상세 결과는 [expansion screen JSON](research/artifacts/sota-zero-shot-throttle-expansion-screen-20260924.json)에 저장했다.

이 근거로 아직 시작하지 않은 speed-shortfall reward .6→1.2 pair를 우선순위에서 내리고, A에 PPO throttle-expansion1 대2 paired continuation을 배정했다. 초기 SOTA actor, fresh optimizer, seed8102, arm당8192 decisions/8 updates/lr2e-6, 같은 custom-only split을 쓰고 brake expansion 및 보상은 고정한다. 현재 loader가 expansion mismatch를 거부하므로, actor weights strict-load는 유지하면서 initialization 때 throttle만 넓히고 brake는 고정하는 명시적 경로를 먼저 구현·검증한다. `resume`에서의 범위 변경은 허용하지 않는다. 이 pair는 zero-shot 충돌 증가를 PPO가 행동정책 학습으로 완화할 수 있는지 확인한다. treatment/paired output은 아직 없다.

C에는 action-range 실험을 복제하지 않도록, 속도-충돌 교환을 다루는 독립적인 PPO risk-sensitive/Lagrangian 변수 하나를 제안하도록 갱신했다. held-out/official 데이터, 코드 수정, 학습, 테스트는 배정하지 않았다.

#### 확장 action 범위의 제출 계약 확인 (2026-09-24 09:31 KST)

A worktree의 `agent.py`는 checkpoint metadata에서 `throttle_expansion`/`brake_expansion`을 읽어 동일한 `VisualActorCritic` action transform을 구성한 뒤 strict state load를 한다. 복구용 규칙 action은 쓰지 않는다. 실험 제한 문서는 제출 action이 유한한 shape `(3,)`이면 된다고 규정하며 gas 0.12 상한을 제출 계약으로 요구하지 않는다. 따라서 PPO가 학습한 확장 gas action을 actor-only checkpoint metadata와 함께 제출하는 방식은 현재 확인된 계약과 맞는다. 학습 초기화 쪽 mismatch 검사는 별도로 좁혀야 하며, runtime transformer가 확장값을 보존하는지 추후 패키지 preflight에서 확인한다.

#### 장애물 실패 trace 재감사와 A 실행 상태 (2026-09-24 09:50 KST)

현재 로컬 scratch 기록을 다시 읽었다. `scratch_official_summary2.json`은 Track1 seed42에서 완주 실패·progress 0.16254·collision 2·damage 0.4, 추가 장애물 사례에서 progress 0.15901·collision 3·damage 0.6을 기록한다. 그런데 별도 `scratch_trace_official-track1-seed42-plus-obstacle.json`은 step63의 collision 1회와 max progress 0.16254를 기록한다. 두 파일은 생성 시각도 약 30분 차이고 연결하는 run manifest, checkpoint hash, 생성 명령을 저장소 검색에서 찾지 못했다. 따라서 이 trace를 summary나 현재 SOTA actor에 귀속시키지 않는다.

그 trace만 독립적으로 보면 step50→55 동안 speed는 약41.3→41.8, gas는 0.088→0.012로 줄고 brake는 0이다. step55의 obstacle distance 15.84에서 step58의 6.41까지 접근한 뒤 brake 0.071이 처음 나오며, step63에서 speed 7.58·progress 0.16254로 collision이 난다. 이 정황은 가속을 줄인 것과 실제 제동·회피를 시작한 시점 사이의 간격을 의심하게 하지만, trace에 per-step visual feature와 checkpoint provenance가 없어 obstacle을 인식하고도 늦게 반응했다고 확정할 수 없다. 다음 원인 판별은 checkpoint·map·seed·feature·action을 한 manifest로 묶은 새 trace가 필요하다. 공식/held-out map은 후보 선택 데이터로 쓰지 않는다.

A의 현재 고정 pair는 `throttle-expansion-ppo-pilot-v1/run_config.json`에서 확인했다. 같은 SOTA 초기 actor·fresh Adam·seed8102·8192 decisions/arm·custom train split을 유지하고 gas 상한만 0.12 대 0.24로 바꾼다. 09:50 KST에는 `run_pilot.py`와 config가 생성됐지만 control/treatment 출력 디렉터리와 학습 프로세스는 아직 확인되지 않았다. 이 실행의 완료 결과를 먼저 받는다. 결과가 action 확장이 속도를 올리면서 충돌도 높이는 것으로 나오면, C가 제안한 후속 단일변수 가설은 expansion 2.0 양팔 고정 후 PPO collision penalty 60 대 120이다. 이는 후보일 뿐이며 A pair가 끝나기 전에 시작하지 않는다.
#### A PPO action-range pair 실제 시작 및 다음 대조 고정 (2026-09-24 10:00 KST)

A의 `throttle-expansion-ppo-pilot-v1`가 09:58:52 KST 시작한 것을 프로세스와 `run_status.json`으로 확인했다. 현재 phase는 `running`, arm은 control 1.0x이며 treatment 2.0x는 control 뒤 순차 실행한다. PID 25164→32832→47940 중 47940이 control worker이고, 세 프로세스가 실제 조회 시점에 살아 있었다. `control/worker-status.json`도 phase `training`과 worker PID 47940을 기록한다. 초기 두 arm의 state SHA-256은 동일 `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`, strict load 통과, fresh optimizer entry 0이다. run config hash `1A7426F743C9B7C754D0896912FDFCA3C57B8A38C10E16881AB2CD44620789ED`, source manifest hash `A75CA99FCD393C7F58B112C7E5F8CD201C621CD26E1F527378E9E1A10DF2E7F7`, 초기 SOTA checkpoint hash `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`를 파일에서 독립 재계산해 일치시켰다. custom-only train/tune이며 held-out 로드는 0이다. 이 시점은 실행 시작 직후여서 update 결과는 아직 없다. session id는 다른 실행기라 root에서 직접 polling되지 않았지만 Windows process 조회와 worker-status로 실행을 검증했다.

C가 제안 분석을 완료했다. A pair 결과 뒤 별도 seed 8103에서 두 arm 모두 throttle expansion 2.0/brake 1.0으로 고정하고 PPO collision penalty만 60 대 120으로 바꾼다. 양팔 fresh Adam, 각 8,192 decisions·8 updates·LR 2e-6, A와 같은 custom split, 30분 상한이다. 제로샷 확장 screen에서 속도 향상과 충돌 증가가 함께 나온 trade-off를 겨냥한다. 이 제안은 단일 geometry tune에 의존하는 탐색 가설이며, 13초 달성 예측이 아니다. 후보 통과는 완주 유지와 충돌·damage 비악화에 더해 under-13 비율 또는 랩타임 개선이 있어야 한다. A의 실제 결과 및 최신 source manifest를 확인하기 전에는 실행하지 않는다. C 결과는 [proposal task 01a0cec2](codex://threads/01a0cec2-4799-7303-aba2-baa80f015189)에서 확인 가능하다.
A progress heartbeat 10:03 KST: 담당은 control arm 2/8 updates, 2,048/8,192 decisions라고 보고했다. 총괄은 PID 25164, launcher 32832, worker 47940를 다시 조회해 모두 살아 있음을 확인했고, worker CPU 누적은 10:00의 86.41초에서 10:03의 247.86초로 증가했다. worker-status는 `training`; `control/policy.pt`는 10:01:14에 저장돼 있고 treatment와 결과 JSON은 아직 없다. 이는 실제 학습 진행 증거이지 성능 향상 결과는 아니다. 기존 arm 순서와 30분 cap을 유지한다.
#### A 실행 진행과 B 장애물 노출 proposal 검증 (2026-09-24 10:08 KST)

A control worker PID 47940와 orchestrator/launcher 25164·32832가 살아 있는 것을 다시 확인했다. worker CPU 누적은 약544초였고 `control/policy.pt` 저장 시각은 10:07:13으로, 10:01 저장본 뒤 새 checkpoint가 기록됐다. treatment는 아직 시작 전이다. 학습된 best checkpoint 시각은 전체 update count가 아니므로 step 진행은 담당의 run record/update log와 최종 산출물로 확정한다.

B의 Strategy E 초안 수치를 root의 `full_site_map_split.json` 해시 `331C9A3F303094840A5FF556241065135BECB7DA8766D11EC8518A8E2BEEB827`로 재계산했다. train group 중 custom map은 12개, seed-expanded episode는 14개이며 각 map에 obstacle 5개가 있다. 따라서 map-slot 기준 obstacle 포함률은 이미 100%다. 이 수치만으로 실제 rollout의 visual `obstacle_present`/positive-urgency 전이율은 알 수 없다. 또한 manifest는 Strategy E worktree에 없어 현재 proposal의 실행 근거가 worktree 자체에 고정되지 않았다. B에 custom-only inventory 경로/해시를 명시하고, 25% 대 100%를 노출을 늘리는 실험이 아니라 학습 mix 효과로 다시 서술하며 속도·valid-under-13 지표와 geometry signature를 포함하도록 요청했다. 해당 실험은 A가 끝나기 전에는 실행하지 않는다.
#### A action-range v1 기록기 오류와 v2 복구 (2026-09-24 10:11 KST)

첫 pair는 09:58:52에 control 학습을 시작했고 584.9초 뒤 control 팔 결과 기록에서 실패했다. `run_status.json`과 `worker-status.json`은 phase `failed`, `KeyError: 'brake_limit'`, child exit 1을 기록한다. `control/policy.pt`는 남았지만 treatment, `run_record.json`, paired 결과는 없다. 소스를 대조해 원인을 확인했다. `training/train_policy.py`의 `train()` 반환에는 `throttle_limit`과 `brake_expansion`이 있지만 `brake_limit`은 없고, `run_pilot.py`가 반환값에 없는 `result["brake_limit"]`를 run_record에 기록하려 한다. 따라서 이것은 주행 정책 성능 실패가 아니라 사후 계측 schema 오류이며, v1 control 체크포인트를 matched 결과 또는 다음 treatment의 출발점으로 사용하지 않는다.

A는 v1 산출물을 보존하고 `throttle-expansion-ppo-pilot-v2` 새 폴더에서 같은 pair를 재시작하도록 수정 중이다. brake limit은 arm의 고정 `PedalScale` 설정에서 계산하며, 재훈련 전 기록 반환 schema를 좁게 확인하도록 요청했다. 10:11 KST의 마지막 독립 확인에서는 v2 폴더만 생성됐고 학습 프로세스는 아직 없었다. v2의 초기 SOTA hash, run-config/source-manifest hash, seed8102, 양팔8192-step 예산, custom-only split이 고정됐는지 확인한 뒤에만 재개 결과를 인정한다. v1 손실 시간은 기록하되 성능 수치로 취급하지 않는다.
#### A action-range pair v2 정상 재시작 및 B 데이터 제안 보정 (2026-09-24 10:13 KST)

v1의 사후 기록 오류를 고친 뒤 A가 새 경로 `throttle-expansion-ppo-pilot-v2`에서 paired run을 재시작했다. `run_status.json`의 `runner_record_contract.verified=true`, 양팔 brake limit 0.28·throttle limit 0.12/0.24, 동일 초기 model-state SHA `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`, fresh optimizer entry 0, held-out 0을 확인했다. run config SHA `3775B0776B6EE3DCD666480644795F30AFDC29871694AEC25C24B82C183D5557`, source manifest SHA `9C765CE8DA05F168FFA87F8A94B7C8CF0DBFDE996529AAE994D57E2EE4A1117B`. 10:12:53 KST에 control 1.0x가 시작했고, PID 10628/46352와 `control/worker-status.json` phase `training`을 실제 조회했다. 아직 완주/시간 결과는 없다.

B Strategy E proposal은 root의 split-manifest 경로 및 해시를 분리 표기하고, custom-only map inventory JSON SHA `75A3FB411D9F0846F6C8CE7870C588A366B9A226E077FCB3DFD9071C234B170F`를 E worktree에 자체 보관하도록 보정됐다. 제안은 이미 100%인 obstacle-bearing custom train mix의 25% 대 100% 비교로 한정되고, 실제 pre-warning 전이수·진행도·완주·충돌·damage·랩타임·13초 미만 비율을 계측한다. 이 데이터 대조는 분석/제안 상태이며 실행하지 않았다. 남은 선행조건은 official/held-out 이력이 없는 공통 actor checkpoint의 존재 확인과 A/C config 선택이다.
#### v2 control 팔 원시 결과 분석 및 treatment 진행 (2026-09-24 10:23 KST)

A의 `throttle-expansion-ppo-pilot-v2` control 팔을 원본 `control/run_record.json`, `update-selection-metrics.jsonl`, `weight-delta.json`, `tune-traces.jsonl`에서 확인했다. 설정 해시는 `3775B0776B6EE3DCD666480644795F30AFDC29871694AEC25C24B82C183D5557`, source manifest는 `9C765CE8DA05F168FFA87F8A94B7C8CF0DBFDE996529AAE994D57E2EE4A1117B`이며, 양팔 동일 초기 actor hash·fresh Adam과 custom-only train/tune(held-out 0)을 유지했다. control 학습은 549.6초에 8/8 updates, 8192 decisions로 완료했다.

체크포인트 선택 기준상 best는 update7/7168이다. tune finish 2/2, median 19.82초, mean speed 42.07, collision 0, damage 0이며 valid-under-13은 0/2다. 두 tune seed가 동일 procedural geometry를 재사용해 독립 지오메트리는 1개다. update별 tune 시간은 20.96, 19.94, 20.26, 20.88, 20.00, 20.12, 19.82, 20.28초이고 집계 collision은 6,0,8,8,2,6,0,6으로 요동했다. 모두 완주는 했지만 checkpoint 품질이 학습 중 안정적으로 개선되지는 않았다. 비교적으로 SOTA zero-shot의 같은 tune screen은 19.88초였으므로 control의 0.06초 차이는 의미 있는 속도 개선으로 보기 어렵다. 안전 결과는 0 collision이지만 한 geometry에 한정된다.

best control의 tune trace는 248 decision에서 평균속도 42.07, 최고 52.85, 평균 gas 0.0644였다. obstacle 미검출·곡률 feature <0.2인 161 decision에서도 평균 gas 0.0785, 평균속도 43.75, brake 0이었다. training rollout은 gas 평균 0.05696, saturation fraction 0.00098, brake fraction 0.1416이다. 현재 evidence는 주행 actor가 제한 gas를 대부분 쓰지 않는다는 가설을 지지하며, 동시에 곡률 구간에서 gas를 낮추고 제동도 쓴다. 따라서 lap-time 병목을 액추에이터 상한 하나로 단정할 수 없다. PPO 8192-step continuation은 gas 사용이나 lap time을 안정적으로 끌어올리지 못했고, 다음 확장 처리군이 실제로 gas/속도를 높이되 충돌을 억제하는지가 판별점이다.

10:22:05 KST부터 treatment(expansion 2.0, brake expansion 1.0)가 순차 시작했고 worker PID 8124가 살아 있는 것을 확인했다. 총 paired cap은 1800초이며 당시 1248초가 남았다. treatment 완료 전 새 학습을 중복 시작하지 않는다. 원시 control 자료는 A worktree의 `artifacts/haic/throttle-expansion-ppo-pilot-v2/control/`에 있다.

#### v2 PPO throttle-range matched pair 결과 및 다음 가설 (2026-09-24 10:34 KST)

`paired-comparison.json`과 두 `run_record.json`의 파일 SHA를 독립 대조했다. 실제 checkpoint, run-record, run-config, source-manifest 해시가 paired record와 모두 일치하고 `run_status.phase=complete`다. 각 팔은 fresh Adam으로 8/8 update·8192 training decisions를 완료했으며 합계 1103.2초로 1800초 제한 안에 끝났다. 양팔의 초기 actor hash는 동일 `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`였다.

- Control, throttle expansion 1.0 / limit .12: tune 완주 2/2, median 19.82초, mean speed 42.07, collision 0, damage 0, valid-under-13 0/2.
- Treatment, throttle expansion 2.0 / limit .24: 완주 2/2, median 19.42초(0.40초·2.02% 개선), mean speed 42.93, collision 0, damage 0, valid-under-13 0/2.
- 저곡률·장애물 미검출 구간은 평균 gas .07847→.09021, 평균속도 43.75→44.65였다. 관측 최대 gas .0900→.1081이고 어느 팔도 자기 throttle ceiling에 도달하지 않았다. 절대 gas .1081은 기존 .12 ceiling 아래이며 treatment .24 ceiling의 절반에도 못 미친다.

paired gate는 완주 비열화 없음, 더 빠른 tune lap, 충돌 증가 없음으로 통과했고 자동 판정은 `candidate_for_separate_confirmation`이다. 하지만 두 tune seed 결과가 완전히 동일한 하나의 custom geometry 반복이므로 독립 맵 일반화나 SOTA 향상으로 취급하지 않는다. 13초 목표에는 여전히 valid completion이 없다. SOTA.md와 RESULTS.md는 수정하지 않았다.

선택된 best checkpoint는 control update7/7168, treatment update3/3072이며 후속 update는 tune time/충돌을 악화시켰다. 한 geometry에서 PPO checkpoint 품질이 update마다 흔들린다. treatment `policy_mean.weight` 변화 norm은 초기 대비 약0.053%였고, LR은 `2e-6`이었다. 동시에 action head는 기존 throttle limit보다 낮은 gas를 출력했다. 따라서 더 넓은 action ceiling만 반복 확대하기보다 PPO가 가속 행동을 실제로 학습하도록 하는 변수가 다음 판별점이다. C에는 기존 collision-penalty 제안(선택 checkpoint 충돌 0)을 재평가하고, `learning_rate 2e-6→1e-5`와 `speed_shortfall_penalty .6→1.2` 중 더 직접적인 단일변수 하나를 제안하도록 보냈다. 다음 fit은 A/C action config와 source manifest를 다시 고정하고 custom-only budget·동일 초기 weights·held-out 0으로 진행한다.

원본: `artifacts/haic/throttle-expansion-ppo-pilot-v2/paired-comparison.json` 및 각 팔의 `run_record.json`, `update-selection-metrics.jsonl`, `weight-delta.json`. 이 결과는 tune 후보이며 공식/held-out 평가는 수행하지 않았다.

#### Strategy E 시작 checkpoint provenance 감사 (2026-09-24 10:31 KST)

B가 read-only로 custom-only 후보들을 조사했지만 Strategy E의 공통 시작 weight로 승인할 수 있는 checkpoint는 찾지 못했다. 가장 가까운 후보 `artifacts/haic/site-map-lateral-randomized-ppo8192-seed8101/policy.pt`는 파일 hash `c84bdaf0302e95599e5ecd69f62dbd626f2fabf08078f2cc3246ae58cdaf919d`, canonical model-state hash `5c401c6201e3de9db07e4a2c6b80138ad092082dcd45a6a4a96d9dd661bf241b`다. 자체 metadata의 14개 train episode는 모두 custom이고 held-out 미사용으로 표시되지만, parent initialization/config가 저장되지 않아 전체 lineage를 입증할 수 없다. 더구나 파일 metadata는 step2048/4 updates 중1회, 인접 `training-result.json`은 같은 checkpoint 경로에 8192 steps/4 updates 완료를 주장한다. action adapter scale도 저장되지 않았다. 이 후보는 증거 부족으로 제외했다. 다른 pedal/temporal 후보는 non-custom 또는 untyped train 항목을 포함한다.

따라서 E 노출률 실험은 미착수 상태다. B에는 현재 root/A 코드의 fresh PPO actor 초기화 경로, 결정적 seed, 동일 초기 state 저장/복제, action config와 custom-only split 해시를 고정하는 방법을 read-only로 확인하도록 새 과제를 보냈다. A 실행이 끝난 뒤에만 설정 변경 또는 CPU 학습을 고려한다. 상세 근거는 B worktree의 `experiments/strategy-e-obstacle-exposure/CHECKPOINT_AUDIT.md`에 있다.

#### LR 5배 실험의 중단 기준 재고정 (2026-09-24 10:53 KST)

A가 LR pair를 시작하기 전 첫 두 PPO update 뒤에 조기 종료를 보장하는 observer를 검토하고 있었다. 현재 합의된 프로토콜에는 두 update 조기 중단 조건이 없다. 대조군과 처리군 모두 8×1,024=8,192 decisions를 수행하며, 실제 비유한 학습 실패 또는 pair 전체 30분 제한에서만 중단한다. tune이 느리다는 이유로 중간 종료하지 않는다. 불필요한 observer 추가를 멈추고, source/config와 두 팔의 learning rate 외 모든 조건을 사전 확인한 뒤 run하도록 A에 정정했다. 정정 시점에는 LR 산출물 폴더와 PPO 학습 프로세스가 없었고 A 작업만 활성 상태였다. 아직 성능 결과는 없다.

#### 다음 독립 PPO 탐색 가설 담당 (2026-09-24 10:58 KST)

A는 2-update 조기중단 observer를 추가하지 않고, 8×1,024 paired LR run을 위한 설정·preflight 후 실행하겠다고 확인했다. 10:58 KST 재조회에서도 LR 출력 폴더와 `train_policy`/PPO 프로세스는 아직 확인되지 않았다. 중복 실행을 시작하지 않고 A의 작업을 유지한다. B(`01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de`)에는 다음 판별 단계용 read-only 분석을 새로 맡겼다. A v2 custom rollout과 update metrics에서 throttle action distribution/entropy가 낮은 gas의 원인인지 확인하고, LR·speed-shortfall reward·action ceiling과 겹치지 않는 entropy coefficient 또는 initial policy std 중 한 변수만 제안한다. 근거가 부족하면 가설을 기각하며, 실행이나 코드 변경은 하지 않는다.

#### SOTA Track1 진단에서 확인한 정지 후 회복 실패 (2026-09-24 11:15 KST)

현재 SOTA `policy.pt` SHA-256은 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`이며 A v2 및 LR pair 입력 사본과 일치한다. 저장된 공식 Track1 seed42 진단 trace에서 progress .162544에 step63·65 충돌 후 damage .4가 되었고, step70–163의 94 decisions 모두 speed≤1, progress는 그대로였다. 그 구간 gas 평균은 .0754, brake는 0, steer 평균은 -.2427이었다. 추가 obstacle 진단은 progress .148410에서 step59–63 연속 5회 충돌하고 damage 1.0으로 종료했다. 이 trace는 official 진단이므로 실패 모드 분류에만 쓴다. map/seed/obstacle 정보를 학습·튜닝 또는 후보 선택에 가져오지 않는다.

원인 가설은 obstacle 접촉 후 회복 상태에서 전진을 다시 만들지 못하는 것과 접근 구간에서 제동 반응이 일정하지 않은 것이다. 단, trace에는 actor가 실제 본 visual feature가 기록되지 않아 perception miss와 policy action 선택을 분리하지 못한다. 현재 `training/train_policy.py`는 `damage≥0.15`, 직전 속도≤8, 비충돌 상태에서 gas 크기와 `abs(steer)×gas`에 보너스를 주지만, 그 상태에서 실제 전진 progress가 회복됐는지를 직접 보상하지 않는다. C(`01a0cec2-4799-7303-aba2-baa80f015189`)는 이 회복 조건의 custom TRAIN rollout 노출도/회복률을 확인하고, 증거가 있을 때만 custom-only RL 변수 하나를 제안하도록 배정했다.

A의 LR 실험은 별도 새 경로에서 11:13 KST control arm 학습을 실제 시작했다. preflight에서 양팔 initial model-state SHA가 `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`로 같고 fresh optimizer, LR 2e-6 대 1e-5, action limits .24/.28, custom train 4·tune 2·held-out 0을 확인했다. run-config SHA `D9A0E229303F2A96DCCDBD75050F8B56EA81440B43CC86286BC7B8300C261310`, source-manifest SHA `B968D854E8AADDCBE9CB6936C8880EAF80D32E9E76C300459321AAF02CD0BC8A`는 파일 해시와 일치한다. 11:15 KST에 control worker PID 19208이 `training` 상태이고 CPU time이 150초로 증가했으며 체크포인트가 저장됐다. 처리군 결과는 아직 없다.

#### PPO 탐색 분산 정체 감사와 다음 독립 변수 (2026-09-24 11:21 KST)

B가 `strategy-a-ppo` worktree의 실제 `current-reward-continuation-seed8101-attempt2/training.log` 및 checkpoint를 읽기 전용으로 확인했다. rollout-update 8회의 entropy는 약 0.8723으로 사실상 평탄했고, throttle limit .12에서 gas mean은 .0564–.0618, gas max .1151–.1184, ceiling saturation은 거의 0이었다. source/8192-step actor의 `policy_log_std`는 [-1.049925, -0.915691]에서 [-1.049974, -0.915415]로 바뀌었고 상대 변화 0.000201, policy-mean weight 상대 변화 0.002395, 전체 model 상대 변화 0.005236였다. 이는 entropy collapse가 병목이라는 가설을 약화시키고, 낮은 longitudinal mean action 및 작은 PPO 정책 이동 가설을 남긴다. 다만 그 run은 `reward_source_mismatch=acknowledged`, train22/tune6/held-out8 구성의 별도 continuation이므로 현재 LR pair나 SOTA tune 성능의 직접 대조 증거가 아니다.

B가 확인한 `weight-interpolation-seed8101/tune/tune-results.json` 전체는 custom 두 반복과 Track1/2 4개가 합쳐진 6-episode 결과다. 전체 selection score로 승격하면 안 된다. custom 행만 보면 base 2/2, 19.88초이고 alpha .25가 2/2, 20.00초라 interpolation 속도 개선은 없다. 후속 아이디어는 LR pair 뒤, actor의 longitudinal `policy_log_std`만 -0.9157→-0.7157(σ 약0.400→0.489)로 바꾸는 RL 탐색 변수다. 현재 train/tune trace에는 학습 당시 state별 gas 분포가 완전히 보존되지 않아 인과 근거는 약하다. 그러므로 이것은 C의 custom recovery 노출 분석과 A의 LR pair가 끝나기 전 실행하지 않는 미확정 가설이다. 다음 평가에서는 한 geometry의 반복 seed 두 개만으로 일반화 결론을 내지 않고 완주·13초 유효 완주·충돌·damage·랩타임과 state-conditioned gas를 함께 기록한다.

#### LR pair control 원시 결과 및 처리군 진행 (2026-09-24 11:27 KST)

A의 `control/run_record.json`, `tune_summary.json`, `update-selection-metrics.jsonl`, `weight-delta.json`을 실제 파일에서 확인했다. control은 LR 2e-6, 8/8 updates, 8,192 decisions, 541.26초에 끝났다. custom tune map `custom-track-haic-tune-20260922`의 두 seed는 동일 geometry 반복이며 둘 다 완주했다: 19.36초, mean speed 43.178, max speed 53.948, collisions 0, damage 0, under-13 0/2. 선택된 best update는 6/6144였다. update별 collision은 4,6,0,0,0,0,4,4로 요동해 tune checkpoint 선택이 여전히 불안정하다.

선택된 tune 행동에서 mean gas는 .07149, 저곡률·장애물 미검출 구간은 .08997, max gas .10831이었다. 액션 한계 .24에서 포화는 0이고 절대 최대 .10831은 기존 .12 한계보다도 낮다. 따라서 이 pair control에서도 병목은 gas ceiling 자체가 아니라 actor가 gas mean을 충분히 올리지 않는 쪽이다. policy-head weight 상대 delta .000497, policy-logstd .0000398, 전체 model .002852로 정책 변화는 작았다. entropy는 update 내내 약 .87이었다. 19.36초는 v2의 19.42초보다 0.06초 짧지만 동일 맵 반복 seed만으로는 의미 있는 개선으로 판정하지 않는다. 13초까지는 아직 6.36초 차이가 남는다.

11:22 KST에 treatment LR 1e-5 worker PID 26008이 시작했고, 11:23 KST 조회에서 실제 process와 `worker-status.phase=training`을 확인했다. 양팔 run-config/source-manifest SHA와 초기 state hash는 일치했다. runner의 `run_record.selected_checkpoint_sha256`은 입력 SOTA 파일의 SHA를 담고 실제 학습 선택 모델 SHA는 worker-status 및 pair의 `checkpoint_sha256`에 따로 기록된다. 해석 시 두 해시의 역할을 구분한다. treatment 결과가 나오기 전에는 LR 효과를 결론내지 않는다.

#### PPO LR 5배 pair 원시 결과 및 회복 계측 설계 (2026-09-24 11:31 KST)

A의 `ppo-lr-5x-at-expansion2-seed8103-v1` 산출물을 직접 확인했다. pair는 8/8 update·8192 decisions씩 끝났고 전체 1059.05초였다. 양팔의 초기 actor state는 SHA `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`로 동일하고, run config `D9A0E229303F2A96DCCDBD75050F8B56EA81440B43CC86286BC7B8300C261310`, source manifest `B968D854E8AADDCBE9CB6936C8880EAF80D32E9E76C300459321AAF02CD0BC8A`가 양팔과 paired comparison에 맞았다. custom train 4개, tune 2 seed 반복, held-out 0의 조건이다.

| 선택된 tune checkpoint | LR | 완주 | lap 중앙값 | 평균속도 | 충돌 / damage | 13초 미만 | 평균 gas |
|---|---:|---:|---:|---:|---:|---:|---:|
| control update 6/6144 | 2e-6 | 2/2 | 19.36s | 43.178 | 0 / 0 | 0/2 | .07149 |
| treatment update 4/4096 | 1e-5 | 2/2 | 18.90s | 44.038 | 0 / 0 | 0/2 | .07631 |

처리군은 control보다 0.46초(2.38%) 빨랐고 평균속도는 약 1.99% 높았다. low-curvature·obstacle 미검출 구간의 평균 gas는 .08997→.09458, 평균속도는 45.03→45.90이었다. 양쪽 모두 .24 상한 포화는 0회였으며 최대 gas는 .11028 이하라 actuator ceiling에 막힌 증거는 없다. 하지만 두 tune seed는 동일한 custom geometry 반복이므로 독립 맵 확인은 아니며, 13초 미만 유효 완주는 여전히 0이다. paired file의 판정도 `candidate_for_separate_confirmation`이고 SOTA/RESULTS 승격 조건은 아니다.

checkpoint별 변동은 핵심 위험이다. LR 2e-6 control은 8개 checkpoint 모두 2/2 완주했지만 update 1,2,7,8은 충돌이 발생했다. LR 1e-5 treatment는 update 1과 4만 2/2 완주했고, 나머지 6개 checkpoint는 0/2였다. 가장 빠른 update4는 충돌·damage 0이지만, 5배 LR을 고정한 PPO가 안정적으로 개선한다고 볼 수 없다. 강한 LR이 더 빠른 일시 후보를 만든다는 가설은 지지됐고, 안정적인 정책 개선이라는 가설은 지지되지 않았다.

C의 recovery telemetry 최소 설계도 완료됐다. A v2 pinned source 기준 `collect_rollout()`에서 `environment.step_transition(action)` 직후, `RolloutStorage.add()` 전에 custom TRAIN transition만 sidecar JSONL에 기록한다. actor가 실제 받은 visual features는 action 직후 복사하고 PPO batch/actor/reward는 수정하지 않는다. 발생률은 코드 조건 `damage >= .15 && previous_speed <= 8 && !collision`을 그대로 계산하고, 연속 eligible 상태 시작 후 1/5/10 policy decision의 progress·speed 변화를 집계한다. tune/held-out/official에는 계측을 연결하지 않는다. 제안 산출물은 `artifacts/haic/recovery-rollout-diagnostic-v1/train/transitions.jsonl` 및 `summary.json`이다. 코드는 아직 바뀌지 않았고 계측도 실행되지 않았다.

다음 속도 가설은 reward 변수 하나로 분리한다: 처리군에서만 `SPEED_SHORTFALL_PENALTY .6→1.2`, control은 `.6` 유지. 두 팔은 같은 선택 checkpoint(18.90s), fresh Adam, 같은 LR 및 seed, custom train/tune, 8×1024 decisions를 써야 한다. 판정은 완주·충돌/damage·13초 유효 완주·median lap·mean speed를 함께 보고, 13초 전환이나 뚜렷한 시간 개선이 완주/충돌을 악화시키면 거부한다. 단일 geometry 반복 tune은 후보 선별만 허용한다. 이 pair는 아직 시작하지 않았다. LR 5배의 high-checkpoint variance 때문에, 먼저 A의 원시 paired audit를 완료한 뒤 제한된 단일변수 실험으로 진행한다.

원시 기록: `artifacts/haic/ppo-lr-5x-at-expansion2-seed8103-v1/paired-comparison.json`, 두 arm의 `run_record.json`, `update-selection-metrics.jsonl`, `worker-status.json`. `run_status.phase=complete`, paired `tune_gate_passed=true`, `valid_under_13_improved=false`이며 공식/held-out 데이터는 사용하지 않았다.

#### LR pair 초기 checkpoint lineage 적격성 보류 (2026-09-24 11:32 KST)

원시 결과 보존과 별개로 후보 적격성에 새 제한이 있다. A v2 `source-manifest.json`의 `selected_checkpoint.historical_provenance_caveat`는 입력 actor가 teacher warmup 30 epochs/3,625 demonstration steps를 거쳤고, 역사적 학습 split에 official Track1/2 seeds가 포함됐다고 기록한다. 이번 paired LR run 자체의 추가 rollout은 custom TRAIN 4 map/seed만 사용했고 tune은 custom geometry 한 개 반복, held-out/official 평가는 0이다. 그러나 초기 가중치가 공식 seed에 노출되었으므로, 이 run을 완전히 clean-lineage 학습 증거로 볼 수 있는지는 현재 `RESTRICTIONS.md`만으로 단정하지 않는다.

따라서 18.90초 best checkpoint는 현재 성능 후보 실험 결과로만 보존하고, 이어 학습·제출 후보·SOTA 승격에 사용하지 않는다. 직전 섹션의 `.6→1.2` speed-shortfall pair 제안은 적격 clean starting checkpoint 확인 전 보류한다. 다음 조율은 artifact lineage를 찾고 규칙 문구를 대조한다. 적격 시작점이 없으면 허용된 custom TRAIN만으로 actor를 처음부터 만들 수 있는 clean-start 실험을 설계한다. 이 상태에서 `SOTA.md`와 `RESULTS.md`는 그대로 둔다.

A의 provenance addendum을 원본 paired JSON에서 다시 확인했다. paired SHA `98D5E42D6898C2AEB78E77CAFD34D527EE8C97453B00951D93509F9FAF741A64`, audit SHA `ECE3FE65BC8279941092F5612962893E6E9F790F9D95D287B40857482F6CA20C`다. JSON은 측정된 `tune_gate_passed=true`와 `promotion_status=on_hold`, `decision=hold_pending_checkpoint_provenance_review`를 구분한다. arm run-record와 checkpoint 해시도 JSON에 들어 있다. SOTA/RESULTS는 변경되지 않았고, 이전 `.6→1.2` 보상 실험은 clean initializer 확인 전 실행하지 않는다.

#### 다음 clean-lineage 학습용 PPO 안정화 가설: 선형 learning-rate anneal (2026-09-24 11:35 KST)

A v2의 실제 `training/ppo.py`는 모든 PPO minibatch update에 같은 `PPOConfig.learning_rate`를 쓰고, `train_policy.py` 초기화 시 optimizer 그룹에도 같은 고정값을 넣는다. LR 5배 pair에서 treatment의 최고 checkpoint는 update4였고 이후 update5–8이 모두 튠 미완주/충돌 상태로 무너졌다. 따라서 가장 직접적인 새 안정성 변수는 reward나 action range가 아니라 **update 간 learning-rate schedule**이다.

판별할 단일변수 pair는 깨끗한 train-only 시작 actor를 양팔에 동일하게 strict-load하고, constant `1e-5` control과 linear decay `1e-5 → 2e-6` treatment만 비교한다. 8×1024 decisions, 동일 custom TRAIN 4 episode split/seed, custom TUNE의 두 반복 seed, 같은 optimizer·PPO 설정·초기 RNG를 고정한다. 양쪽 fresh Adam. 각 update의 finish/progress/collision/damage/median lap과 LR, grad norm, policy delta를 저장한다. Treatment의 late update 완주율/충돌이 개선되고 최종 선택 lap이 control보다 나빠지지 않아야 진행 후보로 남긴다. 13초 유효 완주율을 우선 지표로 계속 보고한다. 이 pair는 clean initializer를 확보한 뒤에만 실행한다.

방법 근거는 과장하지 않는다. 원 PPO 논문은 PPO가 수집된 데이터에 대해 minibatch 여러 epoch를 최적화한다고 설명하지만 linear annealing이 이 대회에서 더 빠를 것이라고 증명하지 않는다([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). CleanRL의 연속 행동 RPO/PPO 기준 구현은 선택 가능한 anneal을 두고 update마다 `frac = 1 - (update - 1) / num_updates`로 Adam 학습률을 낮춘다([reference implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py#L1191-L1199)). 이는 실험 가설을 정하는 참고 사례일 뿐 HAIC에서의 성능 증거는 raw paired run으로 확인해야 한다.

#### 실패 update trace 한계 및 clean checkpoint 조사 (2026-09-24 11:38 KST)

A가 LR 처리군의 원시 `tune-traces.jsonl`과 `update-selection-metrics.jsonl`을 다시 읽었다. 처리군 trace 파일은 **2개 episode만** 담고 있는데 둘 다 선택된 update4/U4의 기록(각 237 decisions)이다. U2/U3/U5–U8의 tune 자료는 episode 요약만 있으며 per-step image feature, action, 충돌 시각, retire reason이 없다. 따라서 0/2 finish인 checkpoint의 실패를 obstacle/curve/off-track/stall 중 하나로 구분할 수 없다. U4 두 반복에서는 collision/off-track/damage가 없었다. U4의 obstacle-present/high-urgency 상태에서 gas는 낮고 brake는 올라갔지만, 동일 progress .74–.76 근처에서 여러 실패 update가 끝났다는 사실만으로 인과 원인을 주장하지 않는다.

update 집계는 불안정을 확인한다. U2/U3/U5/U6/U7/U8의 finish는 모두 0/2, collisions 6/8/10/8/10/10, mean final damage .6/.8/1/.8/1/1, mean progress .753/.753/.757/.753/.374/.255였다. 이들 action aggregate는 U4와 겹치는 구간이 있어 어떤 행동이 충돌을 만든지 설명하지 못한다. 따라서 차기 paired screening은 매 update의 tune decision trace를 보존해야 한다. 이 데이터는 Tune 선택/실패 진단에만 쓰고 TRAIN에 넣지 않는다.

A의 독립 가설은 clean-lineage actor를 공통 초기값으로 한 constant LR `5e-6` 대 `1e-5` 비교다. 앞선 `1e-5` 결과에서 완주한 update를 최소 5/8로 늘리고, `1e-5` 대비 적어도 4개 더 안전한 완주 checkpoint를 얻는 것이 안정성 목표다. 이는 아직 설계 제안이며 실행하지 않았다. 연구 기록에는 별도 가설인 linear schedule `1e-5→2e-6`도 남겨 두고, D가 두 방향을 같은 조건에서 비교 설계한다.

B의 read-only inventory에서는 현재 SOTA 입력 actor SHA `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199` / model-state SHA `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`가 공식 Track1/2 seed training 및 teacher warmup lineage를 가진 것으로 확인됐다. `site-map-ppo-8192`, `site-map-pedal-ppo-*` 등 일부 metadata는 custom train/tune/held-out split과 Tune 선택 metric을 기록하지만, parent/초기화 lineage가 남지 않아 inherited clean model임을 입증하지 못한다. `site-map-actuator-range-v2-ppo2048`는 teacher warmup 30 epochs/1,066 demos를 기록한다. 현 시점에 clean source로 승인된 파일은 없다. 이 분류는 “오염 확정”과 “clean 증명 실패”를 구별하며, 재현 audit note가 별도 작성 중이다.

clean-start 시 입력 `.pt`는 사용하지 않는다. 추가로 현재 루트 `training/site_maps.py`의 `load_site_map_split_payload()`는 `train`, `tune`, `held_out` 세 그룹 모두 `_episodes_for_group()`에 전달해 각 map JSON을 읽는다. 이는 A v2 custom-only runner의 manifest 방식과 다르다. 새 clean-runner는 split manifest에서 오직 허용 TRAIN (그리고 Tune은 성능선택/진단에만 필요한 별도 평가 경로)만 materialize하고, held-out/official map과 seed를 학습 프로세스에 올리지 않는 조건을 source manifest로 증명해야 한다. 초기 model-state와 optimizer SHA, `initialize_from:null`, teacher warmup 사용 여부/훈련 episode 목록을 step 0에 기록한다.

#### Fresh actor gas policy-mean 초기 bias 가설 배정 (2026-09-24 12:02 KST)

A의 LR pair에서 더 빠른 선택 checkpoint도 평균 gas `.07631`, 최대 `.11028`로 action limit `.24`에 도달하지 않았다. LR를 5배로 높이면 median lap이 19.36초에서 18.90초로 0.46초 줄었지만, 처리군 8개 update 중 6개가 tune에서 완주하지 못했다. 따라서 단순히 LR을 더 올리는 대신, 새 actor의 종방향 gas policy-mean 초기 bias를 적당히 높이면 PPO가 학습 초반부터 더 빠른 주행 영역을 탐색할 수 있다는 별도 가설을 검토한다. 다만 이 원시 pair는 오염 가능성이 있는 시작 checkpoint에서 나왔으므로 clean 후보 성능 증거로 취급하지 않는다.

A(`01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1`)에 네트워크의 mean-head 출력, action squashing 및 bounded-gas 변환을 읽고, zero-step custom TRAIN action 통계로부터 작은 bias 크기를 정하는 **설계 전용·읽기 전용 분석**을 배정했다. B의 entropy/log-std 검토와 C의 clean-start·warmup 비교와 변수를 겹치지 않도록, 처리 변수는 gas mean 초기 bias 하나로 한정한다. 후속 paired run은 C가 clean-start 경로를 확정한 뒤 fresh actor로만 설계하고 기존 `.pt`는 불러오지 않는다. 두 팔의 seed, 나머지 actor 초기값, optimizer, PPO 설정, custom TRAIN episodes, 선택된 warmup 경로, 8192-decision 예산을 동일하게 둔다. custom TUNE은 별도 checkpoint 선택에만 쓰고, held-out/official geometry와 seed는 materialize하지 않는다.

모든 update의 Tune finish/progress/collision/damage, median lap, gas/steer 분포를 남긴다. 진행 후보 기준은 동일한 조건에서 완주를 악화시키지 않으면서 반복된 Tune geometry의 lap을 낮추고, 충돌·damage가 늘지 않는 것이다. 13초 미만 유효 완주율은 독립 지표로 반드시 보고하며 Tune geometry 하나의 반복만으로 일반화나 SOTA 승격을 주장하지 않는다. actor mean에 bias를 줘도 bounded gas가 의미 있게 바뀌지 않거나, 가스 증가가 충돌·미완주를 늘리거나, PPO 업데이트가 해당 차이를 빠르게 지우면 가설을 기각한다. clean-start source와 split 격리가 증명되기 전에는 코드 수정·테스트·훈련을 시작하지 않는다.

#### Clean-start PPO paired 설계 및 구현 단계 (2026-09-24 12:05 KST)

C가 루트 규칙과 실제 trainer를 대조해 설계를 마쳤다. 기존 `.pt`는 모두 배제하고, paired arm은 `fresh random actor → PPO` 대 `fresh random actor → custom TRAIN-only teacher BC → PPO`로 정한다. 별도의 BC-only 진단은 PPO 제출 성능으로 합산하지 않는다. BC는 custom TRAIN 네 episode에서 최대 2,400 transitions, 30 epochs, batch 32, LR `1e-3`를 쓰는 초기 비교안이며 최적값이라는 주장은 아니다. BC 뒤 RNG를 phase boundary에서 재설정하고 PPO는 양쪽 모두 fresh Adam으로 시작한다.

paired PPO seed는 8104/8105, arm별 budget은 8,192 decisions(8×1,024)다. 동일한 PPO 설정을 사용하며 현재 코드 기본 LR `3e-4`, gamma `.99`, GAE `.95`, clip `.2`, entropy `.01`, minibatch 32를 초기 비교값으로 고정한다. TRAIN episode는 custom obstacle map `custom-track-haic-obstacles-20260920` seeds 20260920/20260924와 custom train map `custom-track-haic-train-20260921` seeds 20260921/20260925뿐이다. 학습 중 Tune 평가를 끄고 이 TRAIN 네 episode만 materialize한다. 8번째 update의 최종 actor를 사전 지정한 뒤 별도 프로세스에서 custom TUNE만 읽어 평가·선택한다. Held-out/official geometry와 seed는 학습이나 후보 선택에 넣지 않는다.

현재 기본 loader와 PPO loop는 모든 split map을 읽거나 매 update Tune을 평가하므로 위 분리를 위한 별도 clean-start 경로가 필요하다. C(`01a0cec2-4799-7303-aba2-baa80f015189`)의 다음 단계는 이 경로를 격리된 작업 트리에 구현하고, 먼저 split 격리·`initialize_from:null`·step-0 hash·BC→PPO RNG reset·학습 후 별도 Tune 평가를 겨냥한 테스트를 RED→GREEN으로 작성·실행하는 것이다. 사용자가 테스트와 학습 평가를 요청했으므로 software verification 이후 위 2×2 paired PPO run을 실행하고, raw run/seed/map/config/model hashes 및 finish, under-13, collision/damage, median/P90 lap, progress/speed/action 통계를 기록한다. 외부 코드 리뷰는 요청하지 않는다. 안전 조건은 paired control보다 완주율이 낮아지지 않고 충돌·damage가 늘지 않는 것이며, 이 조건에서 lap 개선 또는 under-13 유효 완주가 생길 때만 내부 후보로 둔다. 단일 Tune geometry 반복으로 일반화를 주장하거나 SOTA를 올리지 않는다.

D(`01a0ceca-0f09-7571-a83e-6d1f80ffe655`)는 별도 설계 제안으로 clean initializer에서 constant LR `2e-6` 대 `5e-6`의 단일변수 비교를 권했다. 근거는 이전 오염 가능 continuation에서 LR `1e-5`가 불안정했던 점이며, 새 random-start 성능 근거는 아니다. 이는 C의 warmup 비교에 섞지 않고, clean paired baseline이 나오면 다음 독립 실험으로 보존한다. 아직 어느 설계도 새 학습 성능을 측정하지 않았다.

#### B checkpoint provenance audit 완료 및 변수 분리 (2026-09-24 12:08 KST)

B의 재현 가능한 read-only 감사가 `research/artifacts/checkpoint-provenance-audit-20260924.md`와 companion JSON으로 끝났다. JSON 요약은 대표 artifact 6개 중 `clean_proven=0`, `contamination_confirmed=6`이다. SOTA actor의 file SHA-256은 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`, model-state SHA-256은 `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`다. 그 metadata에는 teacher warmup 30 epochs/3,625 demo steps, official Track1/2 train entries, tune-based selection이 기록돼 있다. 다른 대표 checkpoint도 tune selection 기록으로 contamination-confirmed 분류이며, parent/warmup 기록 누락은 clean 근거로 취급하지 않았다. 감사를 legality 판정이나 실행 데이터 평가로 확대하지 않았다.

감사 문서 SHA-256 `65A805A8B8AFF9AA2BF6B83FFF10B287CAD790FCB55E3CEAB78F04722D831059`, JSON SHA-256 `034483A9F57B8C4FB257BCB837BC4CDF67E2C55266094272CCA0BEE4D4B35D40`. Child config이 training 당시 parent hash를 보존하지 않았으므로 parent hash는 현재 관측값이라는 한계도 audit에 적혀 있다. 이후 새 PPO 실험은 기존 `.pt`를 쓰지 않는 fresh actor로 고정한다. SOTA/RESULTS와 체크포인트는 수정하지 않았다.

B(`01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de`)는 다음 읽기 전용 가설로 longitudinal gas policy-mean bias와 분리된 초기 gas `policy_log_std` 변수를 분석 중이다. 과거 continuation의 entropy 평탄성과 미세한 log-std 변화는 lineage가 clean하지 않아 원인 증명이 아니라 후보 가설의 동기만 된다. 후속 paired run은 C의 clean-start warmup 비교가 끝나기 전에는 실행하지 않는다.

#### Tune checkpoint 선택 계측 경계 명시 (2026-09-24 12:10 KST)

C의 초기 설계는 update 8 checkpoint만 Tune으로 확인하는 안이었다. 후속 paired run의 변동을 진단할 수 있도록 각 update checkpoint를 저장하고, PPO 학습 프로세스가 완전히 종료된 뒤 별도 evaluator에서만 같은 custom TUNE episodes로 전체 checkpoint trace를 수집한다. 사전등록 주 endpoint는 U8이며 Tune best-of-eight은 보조 checkpoint 선택 결과로 분리한다. 이 방법은 TUNE을 gradient/training process에 노출하지 않으며, 단일 geometry 반복을 독립적인 map 표본으로 세지 않는다.

#### TRAIN 초기 상태/seed 다양성 가설 배정 (2026-09-24 12:14 KST)

D(`01a0ceca-0f09-7571-a83e-6d1f80ffe655`)에 clean-start 이후의 독립 후보로 지정된 custom TRAIN 네 episode의 실제 초기 위치·관측·장애물 배치가 서로 다른지, 그리고 규칙상 추가 TRAIN seed를 만들거나 사용할 수 있는지 read-only 확인을 배정했다. 기존 research 기록에서 동일한 seed-diversity/curriculum 실험을 찾으면 중복으로 표시하고, 허용되지 않거나 상태 다양성이 거의 없으면 가설을 기각한다. 차이가 증명될 때만 동일 fresh actor/PPO/warmup/LR/8192 budget으로 episode 다양성 하나를 비교하는 paired 설계를 제안한다. 이 검토는 Tune/held-out/official 입력을 열지 않으며 코드·테스트·훈련을 건드리지 않는다.

#### 과거 Strategy C의 종방향 loss 실패와 gas-brake 결합 원시자료 재확인 (2026-09-24 12:19 KST)

현재 C worktree의 raw JSON을 확인했다. `physical-longitudinal-loss-ppo-v1/result.json`은 SOTA actor 초기 SHA `3ceb5eb...`를 사용한 historical continuation이며, train/tune map hash가 모두 포함되고 tune은 같은 custom geometry의 1 episode뿐이다. 이 오염 가능 run에서 4,096 PPO decisions 뒤 Tune은 progress `.2551`에서 off-track 종료, 충돌 4회, damage `.8`; 8,192 뒤에는 progress `.2593`에서 crash 종료, 충돌 5회, damage `1.0`이었다. 같은 시작점의 gas-only comparator는 서로 다른 action label을 사용했고 Tune 1/1, `18.74s`였으므로 matched causal comparison이 아니다. 별도 `physical-longitudinal-loss-value-detach-seed-replication-v1`은 두 seed의 control/treatment 네 rollout 모두 한 geometry에서 약 `.258` progress, crash 4/4, 충돌 20회, damage 합계 4.0, 완주 0이었다. 이 자료는 value-detach가 해당 Tune failure를 고치지 못했다는 진단일 뿐, clean-source나 일반화 증거가 아니다. saved artifact에는 per-decision tune traces가 없어 충돌 직전 관측/행동을 복원할 수 없다.

`physical-longitudinal-loss-bc-mean-anchor-ppo-v1/summary.json`에서는 4,096 decisions 후 control `19.3s`, mean-anchor treatment `18.7s`였으나 둘 다 같은 한 Tune episode에서 충돌 1회/damage `.2`, under-13 0이었다. 초기 checkpoint는 teacher-only 이전 actor에서 왔으며 승격 근거가 아니다.

A의 현재 network/action transform 읽기 결과는 throttle 가설을 더 좁힌다. actor mean은 steer와 **signed longitudinal 하나**를 출력하며, signed 값의 양수부가 gas, 음수부가 brake로 변환된다. longitudinal bias를 올리면 gas가 늘 수 있지만 brake 출력 확률은 함께 줄어든다. 따라서 ‘gas만 올리기’는 독립 action 변화가 아니다. A는 실제 변환식이 `gas=.12*k*max(0,tanh(u/k))`, brake는 반대 부호의 대응값이며 zero-step bounded action은 관측 latent에도 좌우된다고 확인했다. 근거 없이 고정 offset 수치를 정하지 않는다.

이 두 증거는 지금의 clean-start PPO 문제 원인을 확정하지 않는다. 다만 기존 실험의 결정적인 한계는 단일 Tune episode와 terminal aggregate뿐이라 off-track/crash의 선행 시각·actor visual feature·signed longitudinal/gas/brake/steer를 연결할 수 없었다는 점이다. C의 clean runner는 TRAIN transition마다 이 분리된 signed/decoded action들과 visual feature, progress delta, collision/off-track/termination을 남기고, 학습 종료 후 별도 Tune evaluator도 step trace를 기록한다. 신규 clean run은 fresh actor만 사용하며 이 contaminated `.pt` 및 결과를 학습 초기값으로 재사용하지 않는다.

#### 현재 actor 소스와 C 격리 worktree 불일치 차단 (2026-09-24 12:24 KST)

학습 전에 source 계약 대조가 필요하다는 점을 확인했다. 사용자 작업 트리 `haic_agent/networks.py` SHA-256은 `478B231A97E88B63D8198A602489EE28BF2AD021B7BDA7652559F42EFA18441D`, `training/train_policy.py`는 `6F5E0C44A3932DC51F14F70E99EC55CB7F0E09AEAD57AEE8D6C88831E038E169`다. 이 네트워크는 `MAX_GAS=.12`, `MAX_BRAKE=.28`, branch별 PedalScale/throttle expansion을 지원하며 actor는 steer와 signed longitudinal 두 Normal 좌표를 샘플한다. 반면 C의 4fd3 격리 worktree에는 legacy `MAX_GAS=.02`, `MAX_BRAKE=.03` 및 고정 tanh transform이 남아 있고 network SHA는 `9A1D932E0ACD34B41AC5FEB51C53E4FA2449E4AB95848CC68540324FBEBAB265`, trainer SHA는 `BB13CD62291807EA6F41FBE933A127355D1C7AB7E2A4E43390AC6095CC2E497D`다. 이 worktree에서 legacy action scale로 학습하면 현재 actor 의도에 답하지 않으므로, C는 학습 전 root의 정확한 모델/action/trainer 계약을 hash 고정해 동기화하고 그 일치 여부를 RED→GREEN test와 manifest로 증명해야 한다.

따라서 warmup 대 no-warmup paired run은 양팔 모두 `PedalScale(throttle_expansion=2.0, brake_expansion=1.0)`을 고정해 limits gas `.24`, brake `.28`을 사용한다. 이는 직전 PPO 진단에 사용된 action contract를 맞춰 비교 가능성을 확보하려는 공통 설정이며, 이번 pair의 처리 변수는 여전히 TRAIN-only teacher BC warmup 유무 하나다. root 기본 scale 1.0이나 4fd3의 legacy `.02/.03`로 실행하지 않는다. source-sync가 끝나기 전에는 전체 학습 금지.

B의 longitudinal `log_std` sanity calculation은 (sigma `.40→.50`)을 선택했으나 초기 수치는 expansion `k=1`일 때만 제시됐다. `k_throttle=2`, `k_brake=1`, 현 actor의 초기 absolute gas command `.55` 조건으로 mean/variance 및 gas-zero/brake 확률을 다시 계산하도록 했다. scale을 높여도 bounded gas mean 증가가 보장되지 않을 수 있고 signed axis 때문에 brake 확률도 동반되므로, 후속 action-variance 가설은 이 분석과 C 실험 결과 전까지 대기한다.

#### 비대칭 pedal scale에서 초기 longitudinal variance 수치 재계산 (2026-09-24 12:26 KST)

B가 학습·평가·코드 변경 없이 Gaussian 적분으로 재계산했다. C paired 실험의 고정 scale은 `k_gas=2`, `k_brake=1`, `MAX_GAS=.12`, `MAX_BRAKE=.28`이고 nominal initial longitudinal mean은 `mu=2*atanh(.55/2)=.5645298`이다. At this nominal mean, deterministic action gas is `.066`, brake zero. Sigma `.40` gives expected gas mean `.0654152`, variance `.00158197`, brake mean `.00384006`, variance `.000298705`, and no-gas/brake-positive probability `.079072`. Sigma `.50` gives gas mean `.0663640`, variance `.00214887`, brake mean `.00843342`, variance `.000811924`, and probability `.129434`. Treatment therefore changes expected gas mean only +1.45% while gas variance rises 35.8%; expected brake rises 119.6%, its variance 171.8%, and brake/no-gas probability rises by 5.04 percentage points. The simple gas-minus-brake proxy drops about 5.9%; it is not a vehicle dynamics model. The estimate isolates nominal mu; random actor's low-gain observation-dependent network shifts exact rollout statistics. This makes log_std a plausible exploration-width change but provides no speed-up prediction; safety and lap evidence must decide. Keep this experiment queued until C's clean-start pair finishes.

#### Clean experiment source parity gate (2026-09-24 12:26 KST)

현재 root model/trainer와 C의 4fd3 baseline source가 실제로 달랐다. Root networks SHA `478B231A97E88B63D8198A602489EE28BF2AD021B7BDA7652559F42EFA18441D`, trainer SHA `6F5E0C44A3932DC51F14F70E99EC55CB7F0E09AEAD57AEE8D6C88831E038E169`; C branch는 각각 `9A1D932E0ACD34B41AC5FEB51C53E4FA2449E4AB95848CC68540324FBEBAB265`, `BB13CD62291807EA6F41FBE933A127355D1C7AB7E2A4E43390AC6095CC2E497D`다. C branch의 old `.02/.03` limits는 현재 `.12/.28` + asymmetric PedalScale 계약을 대체하지 못한다. C에 이를 최신 계약으로 고정하고 parity tests/manifest를 만든 다음에만 run을 허용했다. `PedalScale(throttle_expansion=2.0, brake_expansion=1.0)`을 양팔 공통으로 고정한다.

#### GAE horizon follow-up queued after clean baseline (2026-09-24 12:28 KST)

B's asymmetric-action variance analysis concluded with a strict limit: sigma `.40→.50` gives only +1.45% nominal expected gas mean but increases brake mean about 2.20× under the paired `k_throttle=2`, `k_brake=1` transform. It is an exploration-width candidate, not a speed recommendation, and remains unrun.

A separate future RL hypothesis is GAE credit horizon. Existing record already derives gamma `.99` × lambda `.95` TD-residual weight at `.08s` per policy decision, with about 1.30s e-folding; it also cautions that bootstrapped value propagation makes this different from total policy horizon and that pre-update noisy driving cannot be explained by lambda alone. The previous AR(1) action-noise experiment was rejected and will not be repeated. B(`01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de`) is now asked to read the exact current reward/truncation/source paths and design a clean-start paired test changing only lambda `.95→.98` while gamma and action scale stay fixed. It is queued until C's fresh PPO/warmup baseline yields clean measurements; no code, tune-based training, or checkpoint reuse is allowed in the design phase.
#### GAE lambda contrast: read-only source audit (2026-09-24)

B's follow-up read the current reward and rollout paths without running training or evaluating maps. Most route, speed-shortfall, and hazard terms are dense per decision; finish and terminal failure remain sparse (+100 and -60). With gamma=.99, lambda=.95 gives gamma*lambda=.9405 and a direct TD-residual e-fold of 16.30 decisions / 1.304 seconds at 0.08 seconds per decision. Lambda=.98 gives .9702 and 33.05 decisions / 2.644 seconds. At three seconds the residual weights are .1002 vs .3216; at a 13-second lap they are .0000469 vs .00733. These weights describe the GAE residual sum, not the policy's total planning horizon. At the lap endpoint the longer tail is still small in absolute terms, while dense per-step rewards already provide nearer learning signals.

The source audit confirmed that true termination bootstraps zero and ends the trace, while truncation bootstraps the next value once and stops recursion; each 1,024-decision collector boundary is marked truncated. PPO's critic learns from the GAE return, so changing lambda affects both actor advantages and critic targets. Without clean-run advantage/value-error statistics, the variance cost cannot be quantified. A higher lambda could reduce critic-bootstrap bias but may increase variance under stochastic actions and large collision/terminal penalties; it cannot explain noisy behavior before an update.

After C's clean-start baseline, B proposes a paired test changing only lambda .95 to .98: preserve the selected fresh-start/warmup path, gamma .99, LR 3e-4, throttle/brake expansion 2.0/1.0, custom TRAIN split, matched seeds 8104/8105, and 8,192 decisions per arm. Keep Tune out of the trainer; evaluate saved checkpoints afterward in a separate process, with U8 primary and best-of-eight secondary. Record completion, valid under-13, collision/damage, lap distribution, progress/speed, policy/value loss, advantage/target variance, terminal/truncation fractions, and finish/failure contribution versus distance. Reject the useful-speed hypothesis if U8 does not improve lap or under-13 consistently across both seeds, or if safety worsens without a clear speed gain; a Tune-only best-of-eight win is selection noise. This is a proposed design, not measured performance evidence.
#### 지속 개선 loop: 신규 가설 담당 등록 (2026-09-24)

사용자가 1등/13초 달성 이후에도 개선을 계속하라고 요청해 heartbeat `haic-13`을 10분 주기로 ACTIVE 상태로 갱신했다. 목표 점수는 중단 조건이 아니다. 완료된 기존 작업은 재사용하고, 다음 가설은 중복을 확인한 뒤 독립 변수를 하나씩 검증한다. 현재 C clean-start pair가 기준선 담당이므로 새 compute-heavy run은 겹치지 않게 대기한다.

- A (`01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1`): signed longitudinal Normal 좌표를 유지하는 현재 actor와 gas/brake 상호 배제 계약을 읽기 전용으로 감사한다. 후속 가설은 categorical brake/coast/throttle mode와 조건부 pedal 크기 분포로 mode와 크기를 분리하는 정책이다. 상호 배제와 actor-only 제출을 유지하는 단일 변수 paired 실험만 C 기준선 후 제안한다.
- D (`01a0ceca-0f09-7571-a83e-6d1f80ffe655`): temporal pixel-feature 경로가 과거 관측에서 실제 어떤 값을 계산하며 제출 actor에서 pixels만으로 재현되는지 읽기 전용 확인한다. 후속 가설은 visual feature 설정을 양팔에 고정하고 temporal feature만 켜고 끄는 ablation이다. custom TRAIN 사용과 actor-only 계약을 확인하고 C 기준선 후 설계한다.

A/D는 코드·테스트·학습이나 Tune/held-out/official 입력을 사용하지 않는다. 이 두 가설은 아직 설계 중이며 성능 증거가 아니다. 완료 후에도 1등을 이유로 중단하지 않고, 실험 결과를 받아 안전성과 랩 성능을 함께 판정한 뒤 다음 독립 가설로 이어간다.
#### PPO visual encoder 보존 후속 가설 등록 (2026-09-24)

B의 GAE 분석은 설계 기록으로 저장했다. 별도 후속 분석으로 B에 PPO representation-retention 가설을 배정했다. C의 기준선 비교와 겹치지 않게, 같은 fresh actor와 TRAIN-only BC warmup에서 PPO 첫 구간 동안 shared visual encoder를 고정하는 arm과 모든 파라미터를 계속 학습하는 대조 arm을 설계할 수 있는지 읽기 전용으로 조사한다. 분석은 실제 BC/PPO optimizer 경로, encoder drift/gradient 측정, freeze window를 임의 튜닝 없이 정하는 방법, 성능·안전 반증 조건을 포함한다. C 기준선 이전에는 코드·테스트·학습·Tune 입력을 사용하지 않는다. 이 가설은 아직 검증되지 않았다.
#### Clean-start 경로 TDD 진행 상태 (2026-09-24)

C의 최신 실제 작업 기록에서 `test_ppo_update_reports_finite_approximate_kl`는 통과했고, 새 `test_step_zero_manifest_proves_fresh_paired_contract_and_hashes`는 `training.clean_start_paired` 경로가 아직 없어서 실패했다. 이는 계획된 RED 단계로, 구현 완료나 parity 통과로 간주하지 않는다. 현재 C의 `haic_agent/networks.py`와 `training/train_policy.py` SHA는 root의 각 파일과 일치한다. 나머지 실행 의존 파일의 전체 해시/parity gate는 계속 확인해야 한다. 4fd3 경로에서 학습 프로세스는 확인되지 않았다. 따라서 아직 성능 결과가 없으며, C는 manifest·split 격리·RNG 경계 구현과 테스트를 계속한다.
#### C clean-start implementation gate: live diff audit (2026-09-24)

A fresh worktree inspection found the finite approximate-KL test passed after an earlier failure (`approx_kl` was absent before C's change). The step-0 manifest test is still RED because `training.clean_start_paired` has not yet been implemented; this is not a training result. The current SDD `progress.md` still says no tests or production changes exist, contradicting the live diff, so C was asked to reconcile it.

Current root/worktree SHA comparison: `haic_agent/networks.py`, `training/imitation.py`, and `haic_agent/agent.py` match. `training/train_policy.py`, `training/rollout.py`, `training/site_maps.py`, `training/env_factory.py`, `haic_agent/planner.py`, and `training/ppo.py` differ; some may be intended clean-runner changes, so C must classify each, pin the final source manifest, and verify the action/environment contract. No Python training process was present. The paired run remains blocked on implementation, explicit source classification, split isolation, and relevant tests; do not infer performance from this stage.
#### Runner scaffold arrival; ledger still stale (2026-09-24)

A later C update reports the TRAIN-only loader and per-decision trace tests green and a paired-runner scaffold added. A fresh filesystem check confirms `training/clean_start_paired.py` (29,299 bytes) and `tests/test_clean_start_action_contract.py` exist. `progress.md` still says no tests/source edits exist, so this evidence is not yet recorded in the SDD ledger. No Python process using the 4fd3 worktree was present. Full geometry/action/source gates and the runner's manifest test remain unverified; training is not underway.
#### Runner test failure diagnosis and focused rerun pending (2026-09-24)

C's latest focused suite had one pass and one failure. The failure was not in map isolation: `test_train_arm_saves_every_update_without_tune_evaluation` read checkpoint files after its `TemporaryDirectory` had exited. A fresh source check shows the test now captures checkpoint existence before leaving the temporary-directory context, matching runner behavior that saves `policy-step-{total_steps}.pt`. C was asked once to rerun that focused test and record the result. The step-0 manifest test passed afterward (1 passed); the overall implementation suite remains unverified. No training process was running.
#### Teacher warmup coverage and current temporal feature contract (2026-09-24)

B's representation-retention audit found a more immediate clean-start validity issue than its later freeze hypothesis. In current `training/imitation.py`, `behavioral_cloning_warmup()` updates `full_frame_encoder`, `hud_encoder`, `fusion`, and `policy_mean`; it does not optimize `visual_feature_encoder` or `temporal_feature_encoder`. C's current runner protocol enables `use_visual_features=true`, so an action-affecting feature encoder remains at its random initialization during BC while its output feeds the updated fusion layer. This makes the treatment only a partial actor warmup. C was asked to include every enabled policy-mean-path encoder in the BC optimizer with a test that it changes, or disable optional branches in both arms before training; warmup presence must remain the single paired variable. This is not an observed performance cause yet.

B's separate freeze proposal is conditional on C selecting the BC-warmup path: compare all-parameter PPO against freezing only the BC-updated shared visual trunk during U1, then unfreezing at U2. Before any such experiment, measure whether the control actually drifts in U1; if drift is negligible, skip. If implemented later, preserve PPO global gradient clipping by masking trunk gradients immediately before optimizer step, and log head/trunk gradient norms and encoder/latent drift. This remains a design, not measured improvement.

D's report initially said the current constructor lacked temporal features. Current root source disproves that: `VisualActorCritic` supports an optional temporal encoder, and `pixel_features.extract_temporal_features()` derives a one-dimensional previous-frame obstacle-side cue when the current view is centered. It is not a general raw-frame-difference tensor. Root `agent.py` and `training/package_submission.py` rebuild optional feature branches from checkpoint metadata, so the actor-only loader supports the flags. D was asked to reconcile its source/cwd and scope its proposed ablation to this existing cue. `haic_agent/runtime_config.py` defaults the planner on; any future packaged actor-only evaluation must explicitly verify it is off. No feature ablation or package evaluation has run.
#### Pedal-mode 가설의 source-backed 조건 및 temporal cue 정정 (2026-09-24)

A의 읽기 전용 조사는 규칙과 구현을 구분했다. `RESTRICTIONS.md`와 `COMPETITION_INFO.md`는 동시 gas/brake 금지를 명시하지 않으며 제출 contract는 유한한 3-vector와 페달 범위다. 현재 actor는 2D Normal의 signed longitudinal 부호로 두 페달을 구조적으로 상호 배제하지만, `agent.py`는 각 페달을 독립적으로 clip하고 simulator도 별도 값을 받는다. 그러므로 A의 새 가설은 규칙 강제가 아니라 표현력 가설이다: steer Normal + brake/coast/throttle categorical mode + 선택된 페달의 조건부 Beta magnitude. Coast=0, 선택 mode 하나만 페달을 활성화한다. 단, categorical/Beta joint log-probability·entropy·deterministic inference 안정성이 필요해 헤드만 바꾸는 작은 수정은 아니다. Source-backed data와 paired test는 아직 없다. A의 원래 seed 초안은 B의 8104/8105를 제외하자는 것이었으나 공통 비교 목적상 이를 자동 제외하지 않는다. 같은 seed block을 paired arms와 후속 전략에서 공유해도 되며, 실제 예약/사용 여부와 각 실험의 paired 동일성을 ledger에 기록한다.

D의 후속 확인으로 현재 temporal 경로를 바로잡았다. Root `haic_agent/networks.py`는 `use_visual_features=true`일 때 `use_temporal_features` 경로를 지원하고, `pixel_features.extract_temporal_features()`는 현재 obstacle가 중앙에 있고 이전 frame obstacle이 옆에 있으면 그 이전 lateral side 하나를 반환한다. 따라서 제안 실험은 일반 optical-flow/frame-difference 이득이 아니라 기존 한-값 이전 obstacle-side cue의 on/off 비교다. `agent.py`와 package validator는 checkpoint metadata로 feature encoder를 복원한다. 제출용 actor-only 확인에는 planner disabled가 필요하다. D가 처음 읽은 경로는 stale worktree `...\\worktrees\\2d04\\HAIC`였고, current root와 달라졌음을 정정했다.
#### PPO encoder freeze 가설의 상세 설계 (2026-09-24)

B가 원본 optimizer 경로를 추가 조사했다. 현재 BC MSE는 actor mean을 목표로 하며 `full_frame_encoder`, `hud_encoder`, `fusion`, `policy_mean`만 optimizer에 포함한다. Value/aux heads와 `policy_log_std`는 BC에서 학습되지 않고, 옵션 visual/temporal feature encoders도 빠져 있다. PPO에서는 모든 model parameters를 새 Adam으로 학습하며 loss는 policy + 0.5 value MSE + 0.1 auxiliary MSE - 0.01 entropy다. 공유 trunk는 policy/value/aux gradient를 받지만 state-independent entropy는 encoder gradient를 주지 않는다. 따라서 BC가 선택됐을 경우, BC 직후 PPO U1이 공유 trunk 표현을 바꾸는 효과는 가능하지만 아직 실제 drift/performance 원인은 측정되지 않았다.

기존 `bc-visual-encoder-optimizer-v1`은 선택 visual encoder를 BC optimizer에 넣을지 비교한 것으로, BC 이후 PPO에서 trunk를 잠그는 가설과 다르다. 제안된 처리군은 첫 1,024-decision PPO update U1 동안 BC로 실제 학습한 visual trunk (`full_frame_encoder`, `hud_encoder`, `fusion`)만 optimizer step에서 고정하고 U2부터 푼다. Control은 모든 파라미터를 계속 업데이트한다. 그 외 초기화, 같은 TRAIN-only BC, fresh PPO Adam, LR/config, pedal scale, two seed blocks, 8,192 budget은 동일하다. C가 warmup을 선택하지 않으면 해당 가설은 적용하지 않는다.

기전 gate는 Tune을 보기 전에 고정한다: BC→PPO control에서 U1 trunk relative-L2 drift/latent RMS 또는 cosine drift가 사실상 0이면 freeze 실험을 생략한다. 측정값이 있으면 module별 policy/value/aux gradient norm과 cosine, U1/U2 trunk delta를 기록한다. U1에서도 전체 gradient clipping을 그대로 계산하고 optimizer.step 직전에 treatment trunk gradient만 마스킹해야 head clipping scale이 control과 달라지지 않는다. Paired U8에서 완주/유효 under-13/lap이 일관되게 나아지지 않거나 안전성이 악화되면 기각한다. 이는 아직 제안 단계이며 실행 결과가 아니다.
#### C 구현 ledger 갱신 및 다음 학습 gate (2026-09-24)

현재 4fd3의 SDD ledger가 실제 작업을 반영하도록 갱신됐다. 액션 계약, TRAIN-only group loader, decision trace, approximate-KL, step-0 manifest 각각의 RED/GREEN test는 완료로 기록됐고, clean-start paired runner와 분리된 evaluator scaffolds가 존재한다. 정확 source-parity 분류와 evaluator coverage 및 전체 관련 suite는 아직 남았다. Ledger는 현재 `training/labels.py`, `training/site_environment.py`, `training/vision_teacher.py`, `haic_agent/observation.py`, `haic_agent/planner.py`, `agent.py`의 root와 worktree 차이를 확인해야 한다고 구체화했다. Task 4에는 artifacts/실제 run status까지 수집하는 부분 코드가 있으나, 훈련/평가 산출물은 아직 없다. 최신 프로세스 조회에도 4fd3 Python training process가 없었다.

이전 paired-run smoke test의 checkpoint assertion은 TemporaryDirectory 종료 뒤 경로를 확인해 실패했다. 현재 테스트 원본은 context 안에서 존재 여부를 먼저 저장하도록 고쳤지만, 이 수정 이후 재실행 결과는 아직 확인되지 않았다. 또한 C의 protocol은 `use_visual_features=true`이고 현 BC optimizer는 선택 visual encoder를 제외하므로, 이 문제도 수정 또는 양 arm 공통 feature-branch 비활성화와 test로 해결하기 전에는 training gate가 열리지 않는다. SOTA/RESULTS는 그대로다.
#### BC feature-encoder warmup gate fixed and focused tests green (2026-09-24)

C acted on the warmup coverage issue. The paired protocol still uses `use_visual_features=true`, and `training/imitation.py` now conditionally appends enabled `visual_feature_encoder` and `temporal_feature_encoder` parameters to the BC actor optimizer; warmup metadata reports which optional encoders were trained. A new action-contract test confirms the enabled visual encoder weights change during custom TRAIN-only BC. The focused `tests/test_clean_start_paired.py` rerun passed 4/4, and `tests/test_clean_start_action_contract.py` passed 7/7. The earlier temporary-directory checkpoint assertion failure is therefore resolved in the focused suite.

This closes the BC feature-path mismatch for the current visual-on/temporal-off protocol, but it does not prove the full suite, source parity for all dependencies, end-to-end geometry isolation, or driving performance. The SDD ledger still marks Task 2 in progress and Tasks 3/5 pending; no training process or lap result exists. The next action remains to classify the remaining runtime/teacher/environment source differences, complete the relevant suite/evaluator coverage, then launch the predeclared clean paired PPO run and compare its raw Tune results.

#### D next-pixel obstacle predictor proposal and sequential gate (2026-09-24)

D's read-only source audit (root HEAD `572670c5a139e09737cc6aa143aaa19542f612ee`, dirty tree; no code/tests/training) found a feasible TRAIN-only future-obstacle auxiliary target. `CollectedTransition` already carries current and next observations; the pixel extractor deterministically returns seven values. A training-only predictor can use current latent plus executed action to predict next-observation obstacle presence/lateral offset/urgency, with targets `[4:7]` extracted only from `next_observation` pixels. The target path must not read simulator labels, `info`, map identity/geometry, or obstacle truth. Store only the 3-value target in rollout storage, not full next frames. Keep current reward shaping and physical auxiliary loss unchanged. Exclude this predictor from exported actor weights and inference; actor-only policy remains unchanged.

D's candidate paired test is control coefficient 0 versus one preregistered nonzero predictor-loss coefficient, with fresh matched actor/predictor/optimizer state, the same C baseline flags/reward/BC/PPO settings, exact TRAIN map×seed cells and equal decisions. Score the same custom evaluation cells and report completion, valid under-13, collision/damage, lap-time, progress, and obstacle response lead. Prediction loss must beat a persistence diagnostic and correspond to an earlier obstacle response or repeatable driving benefit; lower auxiliary loss alone does not qualify. Risks recorded: heuristic misses/false detections, sparse positives, persistence, conflict with physical auxiliary loss, and only two geometries. This is a design proposal, not measured improvement, and must wait until C's clean baseline is completed.

Sequential work queue update: B's gamma contrast proposal is complete; a follow-up now audits longitudinal policy mean versus state-independent exploration spread and will choose one single-variable post-baseline test. D's predictor proposal is complete; its next read-only task checks whether train-only photometric input augmentation is both source-feasible and non-duplicative, and will design at most one paired intervention after C. A and C retain their existing assignments. No training is to overlap C's current suite/baseline work. Keep `haic-13` ACTIVE and continue after any milestone.

#### C wider-suite interim failures (2026-09-24 13:05 KST)

C reported that the wider project suite is still executing and had shown two failures at the interim progress update. C is waiting for the full run to finish so the failures can be classified against the clean paired trainer before training. Treat these as unresolved test failures, not as model performance evidence. Do not launch a duplicate suite or training while this run is active; record the final failure names/output and gate training on whether the actor/data/evaluator path is affected.

#### Entropy-pressure 단일 변수 후속 가설 — C 기준선 뒤에만 실행 (2026-09-24)

B의 source·action 계산은 현재 원인을 확정하지 않는다. 초기 mean bias는 nominal gas `.066`을 정하고 deterministic action에서는 `log_std`가 직접 작동하지 않는다. Normal sampling에서는 sigma `.40`의 nominal 기대 gas `.065415`, brake `.003840`, no-gas/brake-positive 확률 `.0791`이고 sigma `.50`은 gas `.066364`(+1.45%)에 그치지만 brake `.008433`(+119.6%), branch 확률 `.1294`로 증가한다. 해당 결과는 양의 탐색 폭을 단순히 키우는 것이 저속 문제를 해결한다는 근거가 아니다. 기록된 custom TUNE의 저곡률·장애물 미검출 구간에서도 gas는 scale 1 대비 scale 2에서 `.08438→.09840`였지만, 이는 오염 가능 SOTA actor의 pedal-range 비교이며 clean-start mean/std 인과 비교가 아니다.

A에는 mean bias/action 표현 실험이, B의 기존 대기열에는 `log_std .40→.50` 직접 변경이 이미 있으므로 새 처리 변수는 PPO `entropy_coefficient` 하나로 한정한다. C가 clean-start baseline과 필수 검증을 끝낸 뒤 같은 fresh-start/warmup 경로, matched seed 8104/8105, 8,192 custom TRAIN decisions/arm, gamma `.99`, lambda `.95`, LR `3e-4`, PedalScale `(2,1)`을 고정한다. Control은 `.01`, treatment는 `.005`; 체크포인트·Adam·RNG 초기 조건은 arm별 paired 계약에 맞춘다. 기존 checkpoint를 불러오지 않고 TRAIN에는 허용된 custom train만 materialize한다. 학습 후 별도 custom TUNE 평가에서 U8을 primary로 보고, best-of-eight은 secondary로만 남긴다. held-out/official 입력은 사용하지 않는다.

각 update와 tune trace에서 steer/longitudinal `mu`, `sigma`, sampled/decoded gas·brake, no-gas/brake-positive 비율을 기록한다. 운전 지표는 양 seed별 완주율, 13초 미만 완주, 중앙 lap, progress, collision, damage 및 steer 변동을 함께 본다. treatment에서 종방향 sigma나 brake/no-gas 비율이 변하지 않으면 entropy 가설은 기전 단계에서 기각한다. 분포가 예상 방향으로 변해도 두 seed에서 lap 또는 유효 완주가 일관되게 개선되지 않거나 완주·충돌·damage/steer 안전성이 악화되면 성능 가설을 기각한다. `entropy_coefficient`는 steer와 longitudinal 두 분포 모두에 작용하므로 gas만 좋아졌다는 해석은 trace로 확인할 때만 허용한다. 이는 새로 사전등록한 설계이며 실행·성능 결과가 아니다. 현재 C의 wider-suite 최종 보고와 clean baseline을 대기한다.

#### Repeated DNF localization and source-only reward diagnosis (2026-09-24 13:08 KST)

Fresh reads of Strategy-C raw artifacts in the 4fd3 worktree confirm an early repeatable failure on the single reused custom Tune geometry. The physical-longitudinal PPO run's U4 checkpoint retired off-track after 177 decisions at progress 0.2551 with 4 collisions and damage 0.8; U8 retired by crash after 77 decisions at progress 0.2593 with 5 collisions and damage 1.0. In the value-detach seed replication, all four arms crashed on that same Tune cell around progress 0.255–0.259, with 20 total collisions and final damage 1.0 per arm. This is stronger evidence for a repeatable local policy/map failure than for simply needing more PPO steps. It is not yet an obstacle-specific diagnosis: old result files do not retain per-decision visual/action traces at the failure point. C's new isolated evaluator is intended to save these traces.

The same old physical-longitudinal run's PPO updates show mean sampled gas around 0.077–0.080 and maximum around 0.119; its frozen manifest sets the training throttle limit to 0.12, below the clean runner's 0.24 action cap. Policy loss remains roughly -0.0003 to -0.0018 while value loss is roughly 9–21 in the displayed updates. This supports an under-throttle/weak policy-update hypothesis, but the old checkpoint lineage, single Tune map, and cap mismatch prevent attributing the DNF solely to throttle or transferring the numeric action distribution directly to a new actor. The 8101 mean-anchor screen did finish the same Tune episode in 18.7–19.3s, with one collision and damage 0.2, but neither arm was under 13s; the later 8102/8103 replications all crashed. Thus the selected actor's apparent finish is seed/checkpoint sensitive, not a stable fix.

A's source-only reward audit proposes one later PPO variable: scaled finish bonus +10→+20 while holding the 0.1 scale and all other reward terms fixed. The known source-only terms make a 13s finish's tile reward about +100, progress +1, time cost -7.335, and finish +10 before unknown curve/hazard terms. At gamma .99/lambda .95 the direct GAE terminal residual decays to about 5e-5 over a lap, so the larger finish bonus may still have weak direct influence on early actions; critic bootstrapping is uncertain. Keep this as a conditional screening hypothesis after clean baseline, not as a proven speed fix.

The test checkout audit found that 65 focused tests passed in the root checkout, not C's 4fd3 implementation checkout. The root wide run failed collection on Windows because `tests/test_drqv2_matched.py` imports POSIX-only `fcntl`; excluding that file, it completed with a failure in unrelated root `tests/test_agent_inference.py`. Both are root/platform evidence only. C has been told to finish preserving that run, then verify imports and run the relevant suite from `C:/Users/koi/.codex/worktrees/4fd3/HAIC` before its clean-start PPO. No 4fd3 worktree test result or clean-baseline lap result is established yet.

#### Train-only photometric path audit and bounded visual robustness follow-up (2026-09-24)

Read-only audit against root HEAD `572670c5a139e09737cc6aa143aaa19542f612ee` (working tree already dirty). The active preprocessing path resizes RGB to 84×84, converts it to grayscale, and scales it to float32 `[0,1]`; reset tiles one processed frame into four stack channels and subsequent actions append frames (`env_wrapper.py:7-10,41-54,76-77`). BC collection and PPO rollout/update pass these processed pixels through without brightness, contrast, gamma, or other image augmentation (`training/imitation.py:127-155,333-405`; `training/train_policy.py:324-363`; `training/ppo.py:35-67`). No matching photometric study or prior proposal was found in the inspected research log/plans. The nearby completed design already uses a grayscale frame stack with a HUD branch (`docs/superpowers/plans/2026-09-20-haic-visual-ppo-mpc-plan.md`); the pedal plan covers pixel-derived road/obstacle cues and TRAIN-only obstacle-position randomization (`docs/superpowers/plans/2026-09-21-haic-pedal-policy-plan.md:95-98`), which is a different intervention.

Reject global brightness/contrast/gamma on the shared observation tensor. `VisualActorCritic.encode_observation()` currently sends the same pixels to the full-frame CNN, HUD crops, visual features, and optional temporal features (`haic_agent/networks.py:231-261`). The lower HUD strip is rows 72–83 (`haic_agent/observation.py:13-23`); the pixel feature extractor uses calibrated speed intensity and hard road/obstacle thresholds (`haic_agent/pixel_features.py:12-17,49-52`), and PPO reward shaping consumes the extracted visual features (`training/train_policy.py:329-362`). A shared photometric transform would therefore change speed interpretation, threshold detections, and possibly the reward signal.

One bounded replacement to queue after C: scene-branch-only multiplicative gain. Define the sole augmentation variable `g ~ Uniform[0.90, 1.10]`; multiply rows `0:72` of all four frames by the same per-episode `g`, then clip to `[0,1]`. Feed those transformed scene pixels only to `full_frame_encoder`. Keep the original unmodified stack for the HUD encoder, visual and temporal feature extraction, and reward shaping. In the control, use `g=1`. Use the same transform and range in any selected BC warmup and PPO path; sample one gain per demonstration/environment episode and hold it fixed for the episode. PPO must store and replay that gain (or the exact transformed CNN input) during optimization so each recomputed likelihood sees the same input that produced the rollout action. Do not change any other branch, loss, reward, or training setting. This scene-only variant is a new, narrower intervention; it is not an existing result.

Run only after C’s clean-start baseline and required source/evaluator gates finish. Use fresh matched actor/optimizer state, C’s frozen initialization, BC choice, PPO recipe, paired seeds, four custom TRAIN map×seed cells, and equal decision budget (current contract: 8,192 decisions per arm, subject to C’s frozen protocol). Primary evaluation remains clean, deterministic, actor-only, with `g=1` and the planner disabled; use only the same custom TRAIN cells and label results as internal TRAIN proxies. Report finish rate, valid finishes under 13 seconds, collision count/incidence and damage, lap time among finishers, progress at cap, steer/gas/brake distributions, and pixel-derived obstacle-response lead. Summarize by map because the four cells repeat only two geometries.

Falsify the gain hypothesis if it does not produce a repeatable benefit across both map groups in completion, valid-under-13, lap time, collision/damage, or progress; if a gain on one measure requires a material loss in another; or if the clean actor shows worse obstacle-response/action behavior. One actor seed and two geometries remain a screening result only. No code changed, training or evaluation ran, and no held-out, Tune, or official inputs were opened for this audit.

For the queued paired protocol, inherit the exact TRAIN cell roster: obstacles-map seeds 20260920/20260924 and train-map seeds 20260921/20260925; pair model-initialization seeds 8104/8105 across arms. At the current 8,192-decision cap that is 2,048 decisions per map×seed cell; if C freezes a lower budget, use that exact budget instead.
For BC batches, retain episode association and store each transition's sampled gain at collection so the same gain is replayed during warmup; do not resample it by image or minibatch.

#### Correction: C test commands were run from 4fd3 (2026-09-24 13:10 KST)

Supersedes the checkout inference in the previous 13:08 note. The command records show `cwd=C:/Users/koi/.codex/worktrees/4fd3/HAIC` for the focused and broad pytest runs; the root `.venv` executable supplied Python/dependencies only. Independent import-path verification from that cwd resolved `training.clean_start_paired.py` and `training/site_maps.py` under 4fd3. The final relevant suite from 4fd3 passed **65 tests**, with 4 warnings and 4 subtests passed. A preceding run found one stale gas-limit assertion, which passed after synchronization. The broad suite completed with four failures in legacy Agent/planner action-limit assertions and submission smoke packaging, and the unrestricted collection also hits the Windows-only `fcntl` import. These broad-suite failures are recorded but do not exercise the paired PPO/data-split path. C was told to finish source/map/hash and fresh-state preflights and then launch the authorized PPO pair without waiting on unrelated packaging tests.

#### C clean-start preflight passed; paired PPO launch is the next action (2026-09-24 13:19 KST)

C reports the preflight now passes: runner/imports resolve under 4fd3, its source/evaluator files are included in the hash record, exactly the four custom TRAIN map×seed episodes loaded, Tune and held-out counts are zero, and the action contract is gas cap `.24` / brake cap `.28`. I independently verified the working directory and module paths. At the latest filesystem/process check, the paired-run output directory did not yet exist and no `clean_start_paired`, evaluator, or training process was present. The 65-test relevant worktree suite is green; broad legacy/runtime failures are separately recorded above. C has explicit instruction to complete final source/map/fresh-actor hashes and launch the authorized paired PPO now, without waiting on unrelated platform/package failures. No new clean-lap result exists yet.

Current sequential queue: A is checking training-to-actor-only runtime action-scale parity; B is assessing a per-action-dimension entropy intervention that preserves steering exploration; D is checking custom TRAIN episode diversity and legal sampling options. Their prior finish-bonus, global entropy, scene-gain, and next-obstacle auxiliary proposals remain separate unrun candidates. C's clean PPO pair remains the sole active compute run; every later learned change must be a fresh PPO actor comparison, not a rule-based replacement.

#### Clean-start PPO live snapshot: seed8104 control U5 (2026-09-24 13:25 KST)

The actual run manifest/status confirms only four custom TRAIN episodes are loaded and the paired process is live. Seed8104 `fresh_adam_control` has completed 5/8 updates (5,120/8,192 decisions); seeds/arms after it are still pending. U5 reports mean gas `.0713` against the `.24` cap with zero gas saturation, mean speed `33.47`, max speed `49.85`, mean progress statistic `.0783`, zero collision decisions, and 4 off-track decisions among 1,024 transitions. U4 was `.0631` gas mean and `32.43` mean speed; U5 rebounded, so the per-update speed is not monotonically degrading. Training progress/action aggregates are not lap evaluation and cannot establish completion or 13-second performance. This strengthens the observation that the early actor is not using the available gas ceiling, but action/speed benefit and crash cause remain untested until paired Tune traces after all four arms. Do not alter or duplicate this live run.

#### Per-dimension PPO entropy audit: longitudinal-only pressure candidate (2026-09-24)

Read-only source audit: `VisualActorCritic.action_parameters()` returns two independent Normal coordinates in order `[steer, signed_longitudinal]`; `policy_log_std` is a learned, state-independent length-2 parameter. `sample_actions_with_pretransform()` stores the exact 2-D Normal draw. PPO recomputes one joint log probability by summing both Normal coordinate log-probabilities and subtracting the combined transform Jacobian; the ratio and clipped policy loss therefore remain joint. `VisualActorCritic.entropy()` currently computes per-coordinate Normal entropy then sums over dimension 1, and `PPOUpdater` multiplies that sum by a single coefficient. Since diagonal Normal entropy is additive, the entropy regularizer can validly be split into `H_steer` and `H_long` while leaving rollout log probabilities, the joint PPO ratio, and inference action contract untouched. This is valid under the existing **pre-transform Normal entropy** surrogate; it is not exact entropy of the bounded environment action distribution. Do not split or otherwise change the joint likelihood ratio.

Queue exactly one PPO-only treatment after C finishes its sole clean-start baseline and source/data gates: control `(alpha_steer, alpha_long)=(.01,.01)`; treatment `(.01,.005)`. This halves only the direct entropy-gradient coefficient on `log_std_long`; the steering coefficient remains `.01`. It does not promise unchanged empirical steering variance: the joint clipped PPO ratio, shared gradient clipping, and shared policy/value representation can couple the learned updates. The hypothesis is narrowly that less longitudinal entropy pressure can reduce random braking/no-gas exploration without reducing steering exploration enough to harm obstacle response, and that any useful speed gain must appear in the learned mean/policy, since deterministic actor-only inference uses `mu` and ignores `sigma`.

Mapping check at the existing nominal fresh-actor mean `mu_long=2*atanh(.55/2)=.5645298`, PedalScale `k_gas=2`, `k_brake=1` (`gas=.24*max(0,tanh(u/2))`, `brake=.28*max(0,-tanh(u))`): with `sigma_long=.40`, deterministic gas is `.066`, expected sampled gas `.065415`, expected brake `.003840`, and `P(u<=0)` (no gas / positive-brake branch) `.0791`. If training happened to reduce sigma to `.35`, these become gas `.065166` (-0.38%), brake `.002144` (-44.2%), and branch probability `.0534` (-2.57 pp). At sigma `.30`, they are gas `.065094` (-0.49%), brake `.000955` (-75.1%), and `.0299` (-4.91 pp). These are conditional Gaussian integrals at the nominal mean, not a prediction that changing alpha will produce either sigma; under obstacle states with negative `mu_long`, branch effects can differ. They show that lowering spread alone is not a throttle increase and can slightly lower expected gas. The post-training `mu`, `sigma`, and state-stratified decoded actions must be measured.

Paired protocol, only after C releases the baseline gate: use its finalized clean-start initialization/warmup contract, identical fresh actor state reconstructed per matched seed (no checkpoint reuse), fresh Adam per arm, and the same two paired model seeds (provisionally 8104/8105 if C's frozen manifest retains them), custom TRAIN map×seed cells and equal decision budget. Keep gamma, lambda, LR, reward, pedal scale `(2,1)`, steering entropy coefficient, minibatches, and all other settings fixed. Both arms use PPO actor-only policy/inference; do not run in parallel with C. Train and report only on already-authorized custom TRAIN cells; do not materialize held-out or official data.

Record each update's per-axis entropy, `mu`, `log_std`, joint KL/approximate KL, and decoded steer/gas/brake distributions, including clear-road versus obstacle-present/urgency strata. The mechanism is falsified if the treatment does not lower longitudinal entropy or change the relevant longitudinal action distribution versus its paired control. Axis-isolation is falsified if empirical steering `log_std`/steer distribution materially shifts; such a run cannot be described as steering-preserving. Accept a speed candidate only if both matched seeds improve lap or valid-under-13 outcome without lowering completion or increasing collisions/damage, and steering/obstacle response remains safe. Any seed with a completion loss or collision/damage increase rejects the safety claim; lower entropy by itself is not a win. This is a design only; no code, tests, training, or evaluation ran.

#### Clean-start PPO live snapshot: seed8104 control U7 (2026-09-24 13:27 KST)

A verified wait on the same PID 47164 advanced the control from U5 to U7/8 (7,168/8,192 decisions), with the process still live and no training failure. U7 shows gas mean `.0634`, gas max `.1761` under `.24` limit, saturation 0, mean speed `35.03`, max speed `46.73`, progress statistic `.0792`, collision decisions 0, and 3 off-track decisions/1,024. Across U1–U7 the recorded gas mean stays near `.06–.07` and never saturates the allowed pedal; sampled speed has varied rather than rising monotonically. This is direct evidence that the fresh PPO control currently chooses modest throttle, not evidence that the `.24` actuator cap is binding. It still has no Tune evaluation; wait for U8 and all paired arms before concluding whether teacher warmup changes speed or finish behavior.

#### Seed8104 PPO-only control arm completed (2026-09-24 13:28 KST)

The first 8,192-decision control arm completed from a fresh actor with fresh Adam. Across its eight TRAIN updates, the raw per-update aggregates average gas `.0658` (maximum sampled gas `.1864`) under the `.24` cap, with zero gas saturation; mean speed averaged `35.62` and the highest update max speed was `59.17`. There were zero collision decisions and 25 off-track decisions in 8,192 TRAIN transitions; average per-update progress statistic was `.0903`. The actor never reached Tune and no lap/finish time was measured. This confirms the random-PPO control's early behavior is conservative on throttle and makes little route progress, while it remains collision-safe on these training rollouts. The next arm, seed8104 TRAIN-only teacher-BC treatment, is now running; its result must be compared before attributing the behavior to insufficient BC or changing PPO settings.

#### Clean-start teacher arm serialization failure and preserved control (2026-09-24 13:33 KST)

The live C run is no longer running. `training-status.json` shows status `failed`, completed_runs 1/4. Seed8104 `fresh_adam_control` completed all 8,192 TRAIN decisions and remains preserved. Seed8104 `train_only_teacher_bc` failed before its first PPO decision (`last_completed_step=0`) while writing `ppo-start.json`: strict JSON serialization rejected a NaN (`allow_nan=False`). This is a warmup/step-0 metadata-path failure, not a PPO rollout or driving outcome. No Tune evaluation ran. C is currently reproducing the failure using the four frozen TRAIN episodes only; it writes no PPO checkpoint and does not access Tune/held-out. The failed artifact is retained; any retry must use a new artifact directory and matched source/config across arms.

A's source-level actor/package parity audit found a separate submission-path defect in the C worktree. C training and its clean evaluator construct `VisualActorCritic` with visual features enabled and pedal expansion 2.0/1.0 (gas ceiling .24, brake ceiling .28). Its checkpoint metadata currently omits `use_hud`, `use_visual_features`, and `use_temporal_features`, and stores pedal expansion nested. The isolated C Agent/package validator creates a default non-visual model and strict-loads only the state dict, so a C-produced actor can fail before inference due to architecture mismatch. The current root loader recognizes architecture flags and action-scale fields, but those fields are absent from C's checkpoint. If the model were otherwise loadable, the default gas scale 1.0 would cap gas at .12, half the .24 training/evaluator ceiling; brake scale remains .28. Planner-disabled mode does not remove these loading/scale mismatches. This is A's source audit; it did not modify or execute code.

A is now implementing a canonical inference metadata contract and actor/package golden-action tests in its isolated worktree. C will first finish NaN root-cause isolation, then incorporate architecture/action metadata and ensure package/evaluator parity before interpreting custom Tune as submission-representative. B is analyzing the preserved seed8104 control's per-update policy/KL/action changes using TRAIN artifacts only; no new training or Tune access is authorized for that diagnostic. D's custom TRAIN episode diversity/legal-sampling audit remains active. The modest control throttle measurements are training diagnostics only; no clean lap-time, finish-rate, or under-13 evidence exists yet.

#### Custom TRAIN state-diversity audit and bounded post-C data-selection test (2026-09-24)

Read-only static audit of only the two custom TRAIN map files behind the four allowed cells: obstacles map seeds 20260920/20260924, and train map seeds 20260921/20260925. The root checkout has no `docs/experiments/INDEX.md`; prior-history check used the current research log and the site-map/pedal design plans. No Tune, held-out, or official map/episode files were opened. The user-supplied repeated Tune DNF progress around 0.255–0.259 is retained only as a failure symptom; it was not used to infer a cause or set the TRAIN curriculum.

| TRAIN layout | Reset spawn | Centerline summary | Obstacle lateral offsets by progress |
|---|---|---|---|
| `custom-track-haic-obstacles-20260920` (2 episode seeds) | `(76.96241, 156.02147)`, `start_index=0`, `direction=1` | 246 points; 841.4 m loop; 10 authored corners; mean absolute discrete curvature 0.845°/m, p90 3.209°/m, max 5.742°/m | `p=[.2467,.3733,.5000,.6267,.7533]`; `+0.11,-2.31,-1.22,-1.77,-2.48 m` |
| `custom-track-haic-train-20260921` (2 episode seeds) | `(-70.06816, -162.33915)`, `start_index=0`, `direction=1` | 251 points; 865.3 m loop; 10 authored corners; mean absolute discrete curvature 0.824°/m, p90 3.196°/m, max 4.935°/m | Same five progress positions; `+0.11,-1.83,+2.40,-1.57,+1.04 m` |

Curvature is a discrete centerline estimate: absolute tangent change divided by mean adjacent segment length. The two layouts are distinct coordinate maps. Both use the technical generator template with 10 authored corners, but their corner sequences differ; their aggregate bend statistics are nevertheless very similar. The environment seed is passed to reset, while custom track order comes from the fixed saved centerline, `start_index`, and `direction`; the prior source audit also records `domain_randomize=False` and fixed obstacles. Thus the four cells give two unique spawn/layout families, each repeated twice. Code predicts the same reset scene within each pair, but there are no pixel-hash receipts, so exact pixel identity is not claimed.

Both maps author five obstacles of radius 1.2 m at the same five progress fractions, from 24.67% through 75.33% of the loop, at roughly 12.66–12.67% spacing. `site_environment.py` scales lateral values by 4.8 m for these maps; positive offsets are to the right of travel and negative to the left. The obstacles map has one near-center obstacle and four left-side obstacles; the train map has one near-center, two left-side, and two right-side obstacles. The obstacle-side pattern therefore differs meaningfully, while longitudinal placement and count do not. `env_factory.reset()` attaches obstacles only after `CarEnvironment.reset()` has returned its 50-no-op, four-tile observation stack, so the first returned observation contains no obstacle bodies. Static map files do not establish when later obstacles enter the 84×84 view or their per-decision urgency; no rendered trace was run. Each map defines a full loop, but without policy trajectories the fraction of that route actually visited by an agent is unknown. In particular, four seed labels do not imply four route-coverage samples.

The earlier research note at line 645 already records the duplicate-seed finding: custom geometry replaces procedural track generation and obstacles are static, leaving two independent geometries. The site-map integration plan permits distinct episode seeds within a map split but does not make them new geometry. No completed matched clean-start curriculum comparison on these four cells was found. An older train-exposure mixture note is adjacent history, but it concerns a different data set/protocol and is not this map-share treatment. Local `RESTRICTIONS.md` bars non-TRAIN map/seed/obstacle truth from training or augmentation and bars map/state inputs at inference; it contains no local numeric cap on TRAIN-only episodes. Additional seed IDs are unnecessary here because the saved custom maps make them deterministic repeats.

If C's baseline and source/evaluator gates finish cleanly, queue one data-selection variable using the same four cells: TRAIN-map transition share `q`. Control uses `q=0.50` from `custom-track-haic-train-20260921`; treatment uses `q=0.75`, with the remaining `0.25` from `custom-track-haic-obstacles-20260920`. Split each map's quota evenly across its two existing episode seeds. At 8,192 PPO decisions per arm, control receives 2,048 decisions per cell; treatment receives 3,072 per train-map seed and 1,024 per obstacles-map seed. This modestly raises exposure to the only map with both left- and right-side obstacle placements; it does not create new geometry or claim that map is harder. If BC is selected, keep its data and warmup identical in both arms and apply `q` only to PPO collection. Hold actor initialization, optimizer, PPO hyperparameters, rewards, actions, total decisions, and all other settings at C's frozen contract; use paired model seeds 8104/8105 and fresh actor/optimizer state.

Evaluate both arms with the clean deterministic actor-only path, planner disabled, and the canonical `start_index=0` on all four custom TRAIN cells with equal evaluation weight. Report finish rate, valid finishes under 13 seconds, collision count/incidence, damage, lap time among finishers, and progress at cap. Advance only if valid-under-13 count or median finisher lap time improves in both map groups, finish rate does not fall, and neither map group has worse collision/damage or lower capped progress. If gains occur on one map only, or no primary gain appears, reject the sampling-weight hypothesis; two geometries remain a screening result. No code changed, tests/simulation/training/evaluation ran, and no prohibited input was opened.

#### NaN reproduction isolated to teacher-metric sentinel (2026-09-24 13:35 KST)

C's focused reproduction reached the same strict-JSON failure before any PPO update and localized the non-finite value to an asymmetric-pedal-scale BC warmup metric sentinel in warmup_metrics. The failure is therefore in the teacher warmup reporting/serialization boundary; the run had already passed its post-warmup model-finiteness guard. This is not evidence of a PPO rollout divergence or a bad driving trajectory. The cause that creates the sentinel is still under investigation; do not replace it with a JSON null/finite default until its semantics are understood. The failed teacher arm remains at zero PPO decisions, and the only completed 8,192-step data are the seed8104 TRAIN control aggregates already recorded above. No Tune evaluation or new training is running.

C is also adding explicit architecture/action metadata and evaluator checks so a future checkpoint cannot be scored under a different policy shape or pedal scale. A is implementing isolated package actor parity tests against that canonical contract; B is analyzing only the saved control update traces; D's TRAIN-only state-diversity audit is still active. The four task threads are active; new training remains gated on the NaN root cause, metadata parity, and fresh paired artifacts.
#### Paired retry attempt 2: control complete, teacher arm warming up (2026-09-24 13:57 KST)

C's fresh attempt-2 artifact directory is `experiments/strategy-c/clean-start-train-only-teacher-bc-ppo-seeds-8104-8105-v1-attempt2`. The raw `training-status.json` reports `running`, 1 of 4 arms completed, four loaded episodes from split group `train`, and last update 2026-09-24 04:54:04 UTC. C's task update identifies the completed arm as seed8104 `fresh_adam_control` with all 8,192 TRAIN decisions finished; the matched seed8104 teacher-BC arm is in TRAIN-only warmup and has not begun PPO. No Tune evaluation has run. The recorded Python PIDs 47960 and 19804 are parent and child for the same paired-runner command and output directory, not two concurrent experiments.

Attempt 1's strict-JSON sentinel fix is in C commit `7a7a657`; C reports its focused 67-test suite passed. Attempt 2 uses a fresh artifact directory. The already-recorded control aggregates remain TRAIN-only diagnostics: gas mean .06583/max .1864 under the .24 cap with no saturation, mean speed 35.62/max 59.17, zero collision decisions, 25 off-track decisions, and mean progress statistic .0903. These figures measure neither lap completion nor lap time. There is still no clean lap, finish rate, lap-level collision/damage, or under-13 result from this retry.

The coordinator's compact result shows A still working on a read-only cross-track official DNF audit, D still working on TRAIN-only decision-trace timing, and B's latest analysis turn completed without a surfaced report. B was asked once to return only that completed report or its artifact path, with no rerun. Treat these as task statuses, not as new performance evidence. Keep the first-place/13-second objective open, but do not update SOTA.md or RESULTS.md before a restriction-cleared raw evaluation exists.
#### Attempt 2 teacher-BC progress and next PPO integrity diagnostic (2026-09-24 14:02 KST)

C's latest live update reports the seed8104 teacher-BC treatment at PPO update 6/8 (6,144 decisions). The control and treatment began PPO from matching step-0 actor hashes and fresh Adam optimizers; the TRAIN-only teacher warmup changed the treatment actor before PPO as intended. This confirms the intervention reached the policy parameters, not that it improved driving. The outer status file still says one of four arms completed because the treatment arm has not completed; its last write was 04:54:04 UTC. The runner remains live, and no Tune/lap evaluation exists.

To avoid overlapping C's run and the other diagnostics, B received a separate read-only source audit: trace the sampled latent action, stored rollout value, saved and recomputed log-probability, pedal transform, clipping, and environment action. The audit must distinguish a true stored-variable mismatch from a mathematically valid deterministic action transform; it may use only C's exact source and already-produced seed8104 TRAIN artifacts. It will not add tests, edit code, train, or evaluate. This is a diagnostic assignment, not a performance result.
#### Attempt 2: matched TRAIN comparison exposes progress-safety tradeoff (2026-09-24 14:19 KST)

The raw runner status now says all four arms completed, each with 8,192 PPO decisions. The trainer loaded exactly four TRAIN episodes and records no TUNE, held-out, or official geometry. C reports it is checking checkpoint/source parity before starting the separate custom-TUNE evaluator. The local split manifest has a distinct custom TUNE map, `custom-track-haic-tune-20260922.json`, with seeds 20260922 and 20260926; its five held-out seeds remain outside this stage. Do not start a competing evaluator or read held-out/official data.

| Seed | Arm | Mean speed | Mean progress statistic | Mean progress delta | Mean gas | Max progress | Collision decisions | Off-track decisions | Terminations | Finished |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8104 | fresh PPO control | 35.62 | .0903 | .000439 | .0658 | .2276 | 0 | 25 | 25 | 0 |
| 8104 | TRAIN-only teacher BC + PPO | 37.58 | .2466 | .001479 | .1056 | .8821 | 29 | 16 | 30 | 0 |
| 8105 | fresh PPO control | 36.82 | .1010 | .000534 | .0657 | .3171 | 5 | 23 | 30 | 0 |
| 8105 | TRAIN-only teacher BC + PPO | 31.91 | .1773 | .001078 | .0959 | .4512 | 18 | 29 | 33 | 0 |

These are per-decision TRAIN rollouts, not lap completions. Across seeds, BC raises mean gas about 53% and the progress statistic about 2.2x, but average speed falls from 36.22 to 34.75; collision decisions rise from 5 to 47, off-track decisions stay similar (48 to 45), and none of the four policies finishes a training episode. Thus BC helps the learner reach more of the course but does not yet produce safer completion or reliably higher speed. Max gas stays below the .24 limit and saturation is reported as zero, so the actuator ceiling alone does not explain the slow behavior.

In seed8104's treatment trace, the pixel-derived reward hazard-risk calculation is 1.0 on 28 of 29 collision decisions. Among all 554 high-risk decisions, 28 collide; brake is active on 8.84%, compared with 9.96% across 7,638 zero-risk decisions, while mean gas is .100 versus .106. Most observed impacts therefore happen with the visual risk cue present and without an increased braking rate. This narrows the cause to late or ineffective action response as a live hypothesis; it does not prove the obstacle was visible with enough reaction distance. The matched control's 36 high-risk decisions have no collision, but its max progress is only .2276, so it mostly failed before reaching comparable hazards.

The teacher treatment also has much larger PPO update movement on both seeds: approximate KL ranges .0569–.4125 versus .0078–.0273 in controls; value loss reaches 18.58 versus 8.15, and reported pre-clipping gradient norm reaches 70 versus 51.9. These update summaries make post-BC PPO instability a competing hypothesis, not a proven cause. The C protocol holds LR at 3e-4 and the same reward config, so a future LR contrast can isolate this mechanism if the TUNE result supports it.

The current C source uses `COLLISION_PENALTY=60`, `TERMINAL_FAILURE_PENALTY=60`, `OBSTACLE_THROTTLE_PENALTY=8`, `OBSTACLE_BRAKE_REWARD=6`, reward scale .1, and `SPEED_TARGET=70`. Raising collision cost blindly is not the first response: it is already present, and the treatment still collides at visible-risk states. A is refining a single-variable `OBSTACLE_BRAKE_REWARD=0` versus 6 follow-up using this new evidence; it must keep the selected warmup path fixed, use matched seeds/config/budget, and evaluate only on TUNE after training. B is auditing action/log-probability consistency; D is checking terminal/reward semantics. No new training has started.

A speed-target check is also recorded as a later question, not a current conclusion: C's source comments put official map lengths at 962–1,197 m, which implies 74–92 m/s average for a 13-second lap (`distance / 13`). The implemented 70 m/s target was chosen from a 15-second calculation, but current policies average only about 32–38 m/s in TRAIN. The target may be too low for the user's 13-second goal, yet collisions and failures must be corrected before increasing speed pressure. 13 seconds is the user's target; the local competition document ranks lap time and does not list 13 seconds as a pass gate.

A's existing official-candidate audit remains diagnostic only: all three old candidates finish 2/7 scenarios, and their apparent aggregate differences disappear when the single near-finish obstacle row is removed. Training budgets, reward configurations, output rows, and checkpoint provenance are not matched; the ranking cannot support a reward-causality claim or SOTA promotion. Keep SOTA.md and RESULTS.md unchanged. This is old official failure evidence, separate from the new TRAIN-only C runs.
#### Attempt 2 TUNE result: no U8 completion, early BC checkpoint is only a candidate (2026-09-24 14:34 KST)

Raw evaluation status reports 32/32 checkpoints complete, split group `tune`, two loaded episodes, no TRAIN/held-out/official geometry, no errors, and `source_hashes_match_training=true`. The only geometry was `custom-track-haic-tune-20260922.json` (SHA256 `15d2459bd1f02db1be4166a06bf1770348fcb954b686fa6bce5fbf9da1fb2154`) with seeds 20260922 and 20260926. Those two resets produced identical outcomes for each checkpoint, so this is one tune geometry repeated, not two independent maps.

Primary endpoint U8 failed for all four trained actors: 0/2 finishes and 0 valid-under-13 in every run. Controls retired off-track at progress .0617 (seed8104) and .2675 (8105), with zero collision/damage. BC+PPO seed8104 crashed at progress .3786 with 5 collision decisions per episode and damage 1.0. BC+PPO seed8105 retired off-track at .2510 with one collision decision and damage .2 per episode. The predeclared overall U8 gate is false; collision/damage non-worsening and lap-time/under-13 improvement both fail. Thus the BC path does not provide safe completion at U8 despite higher TRAIN progress.

The auxiliary best-of-eight endpoint selects different update checkpoints per trained seed: control 8104 U2 reaches .1317 without finishing; BC 8104 U1 finishes both tune resets at 19.86s, with one collision and .2 damage each; control 8105 U8 reaches .2675 without finishing; BC 8105 U5 reaches .4940 without collision/damage but does not finish. None is under 13 seconds. The 8104 BC U1 checkpoint SHA is `ae9ae58984628046355e2c52e2f222746aa2750c68199fee5dec2c5c83c90f34`. It is the top Tune candidate by finish count, but remains a diagnostic candidate: one geometry, slow relative to the 13-second goal, and collision/damage present. Do not call it a win or promote SOTA.

The U1-to-U8 collapse on seed8104 aligns with the TRAIN update data: BC arms have much larger approximate KL than controls and high value loss/gradient norms. This supports post-BC PPO policy drift as the next testable cause, but TUNE has no per-decision action traces, so it cannot attribute the crashes to a specific action. Seed8105 is less monotonic, which keeps the LR hypothesis falsifiable rather than established.

A recommends a matched LR-only test, replacing the current warmup-presence comparison: both arms perform identical TRAIN-only BC once per seed and clone the same post-BC actor; then use fresh Adam and compare PPO LR 3e-4 versus 1e-4, holding all other source/config/reward/action/map/budget settings fixed. Pair seeds 8104/8105, four TRAIN cells, 8,192 PPO decisions per arm. Evaluate only after training on the same custom TUNE map/seeds; U8 primary, best-of-eight auxiliary. Reject the high-LR harm hypothesis if KL does not fall or if safety/progress/finish outcomes do not improve; also reject gains that come only from >5% speed reduction. B is independently analyzing the same TUNE checkpoint/update pattern without access to held-out/official data. A's separate obstacle-brake-reward proposal is deprioritized unless the new LR experiment fails to explain the outcome.

Under local `RULES.md`, the seed8104 BC U1 checkpoint is the Tune-selected candidate for a separate post-selection held-out/official diagnostic. It must remain actor-only and cannot affect this experiment's training or Tune choice. The held-out evaluation will be recorded as a one-time diagnostic, not a basis for tuning this same held-out split. No held-out/official file has been opened yet, and no upload/submission occurred.

#### Persistent improvement loop reaffirmed; next owners (2026-09-24 14:44 KST)

The existing Codex Goal and `haic-13` heartbeat remain ACTIVE. First place and a sub-13-second lap are milestones, not stop conditions. Current compact task state: A (`01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1`) and B (`01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de`) are active on their separate speed and PPO checkpoint-drift analyses; C (`01a0cec2-4799-7303-aba2-baa80f015189`) is active but has not started the post-selection diagnostic, because it is checking whether the current workflow can safely compare the Tune-selected actor and SOTA actor with verified provenance; D (`01a0ceca-0f09-7571-a83e-6d1f80ffe655`) was given a new read-only TRAIN-only audit of teacher BC action-label timing and obstacle-state coverage, distinct from B's post-BC PPO drift analysis. No new training/evaluation was launched by the coordinator and no task was interrupted.

The D audit is limited to attempt-2 TRAIN artifacts. It must stop if teacher observations/actions were not retained, and must not inspect TUNE, held-out, or official data. Its purpose is to distinguish weak/late teacher targets from post-BC PPO drift before selecting the next single-variable PPO experiment. Continue with existing task threads; if a task completes, give it a distinct bounded follow-up and register the evidence and owner here. Preserve actor-only PPO inference, TRAIN-only teacher/privileged inputs, split/hash checks, and raw-evidence requirements. No external upload or submission is authorized in this coordination loop.

#### A speed-target proposal and mechanism gate (2026-09-24 14:47 KST)

A's source-backed design keeps the only proposed speed intervention at `SPEED_TARGET=70` versus `84`, with all other settings fixed. The source's 962–1,197 m track-length comments imply 74–92 m/s average for the user's 13-second target, but the current target contrast is not justified as a live training run because no policy meets a safe U8 Tune baseline: all four U8 actors failed; the only selected early BC checkpoint finished at 19.86 s with collision/damage. A therefore requires a safe U8 readiness gate first: warmup path fixed and both model seeds finishing without collisions or damage. The target test, if that gate is later met, uses matched TRAIN cells and paired 8,192-decision PPO budgets, with progress/completion and collision/damage non-regression required; target uplift alone is not a win. These are design gates, not results. A has a separate read-only follow-up to calculate whether the reward term can affect policy at observed TRAIN speeds. B's U1–U8 report did not surface despite its latest turn completing; one status/blocker request was sent, with no rerun requested. C remains active checking the safe diagnostic workflow, and D's separate TRAIN-only teacher-target audit is queued. No SOTA/RESULTS update, training, or new evaluation was made here.

#### B identifies missing parameter-drift evidence; bounded comparison queued (2026-09-24 14:49 KST)

B clarified that the prior U1–U8 drift task did not compute checkpoint weight deltas: the available notes only show logged approximate-KL spikes (up to about .375 on seed8104 and .412 on 8105) and gradient norms (about 67/70). There is no weight-level drift result yet, so KL/gradient logs alone do not establish that actor parameters moved excessively or caused the TUNE crashes. B is now comparing existing attempt-2 U1/U8 state dicts read-only, per module and seed, after verifying hashes/architecture metadata. If comparable tensors are missing it must stop; no retraining/evaluation is allowed. This measurement is distinct from the previous incomplete turn and should settle whether the proposed LR response has direct parameter-change support.

#### U1–U8 parameter comparison weakens the actor-drift explanation; TUNE traces queued (2026-09-24 14:57 KST)

B completed the existing-checkpoint comparison. All four attempt-2 arms have matching U1/U8 and intervening update checkpoint hashes; all 32 hashes matched `updates.jsonl`. The embedded checkpoint metadata and run manifests agree on `VisualActorCritic`, 78,383 parameters, HUD+visual features enabled, temporal features disabled, and asymmetric pedal scales (throttle 2.0, brake 1.0). Within each arm the actor trunk moved substantially from U1 to U8, but BC arms moved less than controls: trunk relative-L2/cosine was .301/.9605 and .241/.9732 in BC, versus .376/.9382 and .429/.9213 in controls. Actor-head relative-L2 was similarly small across arms (.046–.054, cosine .9987–.9992). The seed8104 BC value head moved more (.832 relative-L2, cosine .9095). The large BC KL spikes (up to .3746/.4125) therefore do not correspond to larger measured actor-trunk drift than control. KL and weight-space movement are different diagnostics; this weakens, but does not disprove, the hypothesis that excess actor drift explains the TUNE failures. It does not establish a causal effect or rule out value-function instability.

The same raw TUNE artifact directory contains `evaluation/decision-traces.jsonl`, despite an earlier note saying no per-decision TUNE traces were retained. Its actual schema/content still needs verification. B has a bounded read-only follow-up to establish whether it records enough signals to localize the U8 retirements and U1 finish by obstacle response versus corner/off-track behavior. Treat the two TUNE seed labels as repeated resets on one geometry. Separately, C reported that a post-selection loader preflight rejected an actor for a strict architecture mismatch before any episode ran. I loaded the exact selected seed8104 BC U1 `policy-step-1024.pt` with `weights_only=True`: its embedded metadata is present and says HUD=true, visual=true, temporal=false, throttle expansion=2, brake expansion=1; the run manifest agrees. This narrows the mismatch to the exact actor/path/import used by preflight (possibly the SOTA actor or a stale loader), but does not identify it. C is tracing which actor and module fail; do not start a driving episode or change code until that is known. No held-out/official file was opened by this coordinator.

#### TUNE traces separate two failure modes; speed penalty has measurable leverage (2026-09-24 15:09 KST)

B verified the attempt-2 `evaluation/decision-traces.jsonl` provenance: 15,022 per-decision rows, SHA-256 `44fd7caee5f139f15753d5334ef4fe79210391b478fc8e4e6d35bbe2609cdcdf`; schema includes action, speed/progress, collision/damage/off-track and seven actor visual features. The TUNE map hash is `15d2459bd1f02db1be4166a06bf1770348fcb954b686fa6bce5fbf9da1fb2154`. The paired Tune reset sequences are exactly identical within each actor, so they are repeated runs on one geometry, not independent samples.

Failure signatures differ by arm. Both fresh-PPO U8 controls retired off-track without collision/damage at progress .0617 and .2675. Their traces show prior high curve magnitude (.786/.929) and far-road-center offset magnitude up to .869/.881; one reached speed 71.87 m/s at retirement with brake zero. Their obstacle-urgency cue was present earlier without a collision. This fits lateral road/corner tracking failure better than obstacle contact, but raw frames and offset direction are absent, so the steering diagnosis is not causal proof. Both BC U8 episodes instead encountered obstacle-urgency=1 during collision/damage: seed8104 accumulated five collision decisions and damage 1.0 before crash at progress .3786; seed8105 had a collision/damage .2 at progress .251, then stalled and retired off-track. The BC U1 finisher also had one obstacle-associated collision/damage .2 and completed in 19.86 s. There is no single failure cause: the fresh control needs stable curve/road tracking, while BC additionally has a hazard-response problem. Two tune resets still represent only one map.

A's read-only reward replay covered 32,768 TRAIN rows. At current target 70, changing `SPEED_TARGET` to 84 would add about .005149 scaled penalty per eligible decision in fresh control and .004962 in BC; more than 99.9% of eligible rows had a nonzero delta, so curve/hazard relief does not mask the signal on most TRAIN states. In low-speed, strong-relief BC rows, the increment falls to about .000925. The overall increment is similar to the existing per-step time penalty (~.005), but collision/terminal-failure penalties (~6 scaled) are roughly 1,200 times larger. This shows reward leverage, not that the actor will become faster or safer. A's earlier zero-collision, zero-damage U8 readiness gate remains a proposed promotion/safety condition; it may be too strict as a *run* gate while no U8 actor completes, so the next paired-screen design should separate permission to learn from criteria to promote. No 70-vs-84 training has run.

The first strict architecture preflight issue is not yet attributed. I directly loaded both the selected candidate checkpoint and the SOTA `policy.pt` with the root checkout's strict `Agent._load_policy_with_status`: both succeeded. Candidate metadata resolves to HUD/visual on, temporal off, throttle/brake expansion 2/1; SOTA resolves to HUD off, visual on, scale 1/1. The packaged SOTA ZIP itself records `PLANNER_ENABLED=False` and `STRICT_CHECKPOINT_LOADING=True` (its archive passes ZIP integrity), while source `runtime_config.py` has development-only defaults True/False and the package builder rewrites them. Do not change those source defaults based on the apparent mismatch. The root and C worktree `haic_agent/networks.py` hashes match the attempt-2 manifest; their `agent.py` files differ, so C is verifying the exact imported loader/runtime for the diagnostic. No driving episode, source edit, or package was made. B is now measuring whether TUNE failure states fall outside the TRAIN feature support; D is independently checking teacher target timing/coverage.

#### 2026-09-24 coordinator update: paired speed screen un-gated for learning, candidate diagnostic resumed

A resolved the circular gate question for the `SPEED_TARGET=70` versus `84` S1 screen: safe zero-collision/zero-damage U8 completion is not required to *start* a paired screening experiment, because every current U8 actor fails and that gate would prevent learning. Keep the same requirement as a strict promotion gate. Source replay already shows the target delta reaches nearly every eligible TRAIN decision (mean scaled reward delta about .00496–.00515), so a separate reward-leverage prerequisite is unnecessary. A's design holds teacher-only BC warmup, initialization, optimizer, all PPO settings except the target, actions, rewards, TRAIN-only cells, and paired 8,192-decision budgets fixed; Tune is used once after training for the U8 screen, and cannot tune training. It explicitly separates informative screening from promotion. This is design evidence only; the run has not started.

A is now doing a source-only feasibility preflight for cloned post-BC initialization, exact four-cell round-robin quotas, target-only override, and raw artifact hashes. It must not train or evaluate while C owns the one-time held-out/official-local diagnostic. B remains active on TRAIN-to-TUNE feature-support coverage. C has accepted `ae9ae5…c90f34` as the authoritative selected checkpoint and is rechecking restrictions and evaluator provenance before its registered seven-cell comparison against the existing SOTA actor; no result from that diagnostic is available yet. D completed a static audit of the two saved TRAIN geometries: they have similar aggregate bend statistics but materially different obstacle left/right patterns; the four episode seeds repeat two geometries. D's already logged `q=0.50` versus `q=0.75` TRAIN-map sampling contrast is only a screening proposal, not causal evidence. D is now preparing its bounded preregistration against local result and paper records; it must not run training or any non-TRAIN evaluation.

Current allocation therefore has no overlapping run: C owns the only active evaluation; A and D are design/source audits, and B is a read-only support audit. The clean-start BC comparison remains negative on safety: attempt-2 U8 had 0/8 finishes and 0 under-13; BC collisions occurred on all four TUNE cells while control had none, despite higher treatment progress. Its selected U1 candidate is 19.86 s with collision/damage, so it remains diagnostic only. No SOTA/RESULTS pointer changes. Continue collecting A-D results, then queue one distinct experiment at a time; maintain the RULES.md requirement to compare S1 PPO actor-only, S2 PPO+CEM (comparison-only), and S3 vision-corridor teacher (diagnostic-only) on matched splits, and keep teacher/privileged inputs TRAIN-only.

#### 2026-09-24 diagnostic preflight: false failure from Windows path separator (new blocker classified)

C's current read-only preflight reached the final SOTA path-text assertion and raised `AssertionError`. In the script, checkpoint/archive/split/map hashes, the candidate-selection record, strict actor loads, package runtime flags, and held-out cell list were checked earlier; the failing assertion searches for a slash-separated SOTA source path, while `SOTA.md` stores the same relative path with Windows backslashes. This is a preflight string-normalization defect, not a checkpoint mismatch. The output directory was absent and the failing preflight did not run episodes. C has been asked once to normalize separators, repeat only the preflight, and proceed with the seven registered diagnostic cells only after all gates pass. No checkpoint promotion or SOTA/RESULTS change.

#### 2026-09-24 B feature-support audit: no broad covariate gap; obstacle-state occupancy remains a candidate

B completed a read-only analysis of the same attempt-2 TRAIN and TUNE decision traces. The evidence uses the two TRAIN map geometries (TRAIN map file hashes `a4b10ffd848a7b35827bb9d90c311396169ce30b1bffd6614b1cd44c80c0ad16` and `cff9b8ca1f758adfdc3d6c5df3f50e88ef566b51a6bbc9a7a544c3a83bc89c0f`) and the existing TUNE trace SHA `44fd7caee5f139f15753d5334ef4fe79210391b478fc8e4e6d35bbe2609cdcdf`. The four TRAIN traces contain 32,768 decisions; the two TUNE reset labels collapse to 7,511 unique rows because the same saved TUNE geometry generated exactly repeated trajectories.

Every TUNE feature stayed inside pooled TRAIN min/max. Only 6.52% of TUNE points exceeded the TRAIN leave-one-out 95th-percentile nearest-neighbor distance and 1.41% exceeded its 99th percentile. The off-track window for control seed8104 was inside common joint support; some high-curve and obstacle-contact windows were sparse, but most BC U8 collision windows were also inside common support. TUNE's obstacle-active share was 10.0% versus 5.0% pooled TRAIN, with per-cell TRAIN rates 3.4–7.1%. This points away from a general visual-feature distribution failure and toward a possible state-occupancy imbalance, but policy occupancy and repeated obstacle/stall states can inflate the TUNE rate; this is not causal evidence.

B proposes one unrun S1 candidate: increase only the TRAIN-only share of urgent-obstacle states (`obstacle_present=1` and `urgency>0`) from the observed ~5% toward a fixed 8% per rollout/update, sampling only existing TRAIN transitions. Falsify it if the same one-time TUNE screen does not reduce urgent-window collisions/damage without reducing finish/progress. Before any run, verify unique eligible transition counts and per-cell feasibility; do not use TUNE/held-out/official data for sampling. Keep this distinct from D's geometry-level `q=0.50` versus `0.75` proposal and A's reward target contrast; no combination treatment is registered.

#### 2026-09-24 C preflight correction: strict loaders and provenance pass; diagnostic still not launched

C's first read-only preflight failed on a text assertion that searched for a slash-separated SOTA checkpoint path although `SOTA.md` stores Windows backslashes. It was a path-normalization false failure, not a checkpoint mismatch. C reran the preflight successfully (exit 0): candidate `ae9ae5…c90f34`, SOTA checkpoint `3ceb5e…afff199`, package `451831…54b14`, and split manifest `331c9a…eb827` match the frozen expectations; the candidate and SOTA both strict-load through the root evaluator's `Agent`, and the evaluator resolves that same root `Agent`. All three registered diagnostic map hashes match. The output path remained absent, so no map or episode had been constructed at that receipt. C is cleared to execute only the already-registered seven held-out/official-local diagnostic cells; no results or SOTA/RESULTS change yet.

C then corrected the path assertion and completed the preflight without changing code. Its frozen protocol is at `experiments/strategy-c/clean-start-train-only-teacher-bc-ppo-seeds-8104-8105-v1-attempt2/post-selection-diagnostic-v1/protocol.json` (SHA-256 `45668de36ea0666674d502c0a70e8c163036b40163357c3233cbaae979e83767`), recorded before the first episode. The diagnostic output directory now exists, but the latest receipt is still the pre-episode manifest freeze; no episode result has been reported yet. This supersedes the prior note that the output directory was absent at the earlier preflight.

#### 2026-09-24 SOTA 제출 가능성 및 최신 스크리닝 증거 (15:xx KST)

현재 SOTA.md가 가리키는 제한 준수 PPO actor-only 기록은 held-out completion 0.75, median lap 19.32초, p90 22.42초이며 제출 archive는 artifacts/haic/submission/haic-obstacle-risk-ppo-actor.zip이다. 이는 현재 확보된 제출 후보지만 사용자의 13초 목표에는 미달한다. 현재 새 paired speed-target screen은 구현·검증 단계이며 학습/평가를 실행하지 않았다. 따라서 기존 SOTA보다 나은 새 제출 후보는 아직 없다.

A의 TRAIN-only counterfactual reward replay: 32,768 decision 중 32,623 eligible. 목표 속도 70→84 변경의 누적 speed reward delta는 -164.928이고, 기록된 충돌 52건 및 terminal failure 118건의 페널티 합은 -1,020(사건 겹침 가능)이며 off-track 항은 -93이다. 네 TRAIN cell의 실제 노출은 7,353–9,387 decision으로 계획 균등량 8,192에서 벗어났다. 이 replay는 기존 궤적의 reward 재산정일 뿐 재학습 결과나 예상 PPO return이 아니다. raw trace에는 reward/advantage/return이 없어 전체 보상 변동성도 산출할 수 없다. 목표 속도 효과를 판별하는 paired 재학습은 여전히 필요하며, exact cell quota 구현의 동기를 제공한다.

C의 최신 provenance 재확인은 앞서 기록된 “7개 진단 셀 실행 허가”를 supersede한다. 선택 경로 checkpoint의 실제 SHA-256은 ae9ae58984628046355e2c52e2f222746aa2750c68199fee5dec2c5c83c90f34였고, 사전 승인 hash ae9ae58984628046355e2c52e2f222746aa2750c68199fee5dec2c5c83bcf7e1와 일치하는 파일은 저장소/worktree의 859개 checkpoint 파일 및 검사된 ZIP 내부 checkpoint 142개에서 발견되지 않았다. 현재 파일 기준 기존 14회 평가는 승인 후보의 증거로 쓸 수 없다. 이번 재확인에서는 새 평가나 파일 변경은 없었다. 기존 SOTA 파일은 별도 held-out 근거의 actor이며 이 후보 해시 혼선과 구별한다.

결론: 지금 제출 후보로 손에 있는 것은 기존 SOTA archive 하나다. 13초 미만 완주와 기존 SOTA 대비 개선은 입증되지 않았다. 새 paired speed screen은 소스/프로토콜 검증을 마친 뒤 RULES 게이트를 통해서만 실행한다. SOTA.md와 RESULTS.md는 수정하지 않는다.

#### 2026-09-24 긴급 장애물 상태의 재진입과 dwell 분석 (16:xx KST)

B의 TRAIN-only 재구성 결과: attempt-2의 32,768 decision 중 urgent 조건(obstacle_present>0, urgency>0)은 1,637건(5.00%)이고, 종료/reset 경계에서 나눈 event-run은 220개다. 같은 물리 장애물인지 식별하는 obstacle_id와 원시 observation/reward는 로그에 없어 event-run을 서로 독립 표본으로 간주할 수 없다.

fresh-PPO control은 urgent decision 169건/80 event-run, BC+PPO는 1,468건/140 event-run이다. control의 dwell 중앙/90백분위/최대 길이는 seed8104에서 1/1/2, seed8105에서 2/7/9 decision이었다. BC+PPO는 각각 8/15/114 및 7/12/110 decision으로 길어졌다. 가장 긴 dwell 114 decision은 학습자가 같은 urgent 상태를 탈출하지 못하는 현상과 일치하지만, 로그는 종료 완료가 모두 false이고 원시 화면·보상·물리 장애물 ID도 없으므로 원인이나 인과효과로 단정하지 않는다. 이 결과는 “위험 상태 표본이 부족하다”는 설명을 보완한다. 전체 urgent 행 비율만 보면 실제로는 같은 event/episode 내부에서 반복된 프레임이 표본 수를 부풀릴 수 있다.

B가 먼저 제안했던 transition 고정 8% oversampling은 새 dwell 결과와 중복 가중 위험을 반영해 다음 후보 설계에서 우선하지 않는다. 별도 미실행 가설은 각 TRAIN map/reset cell의 현재 PPO 정책으로 수집한 고유 on-policy episode segment를 단위로 노출을 늘리고, segment/event/transition ID를 기록하는 것이다. 동일 segment를 중복 재생하지 않고 4개 TRAIN cell의 할당은 고정한다. 현재 trace에는 원시 observation/next observation/reward와 고유 episode ID가 없어 이 기록만으로 PPO minibatch를 만들 수 없다. 완주·충돌 비교는 이후 별도 평가가 필요하며 현재 학습 로그는 완주 증거가 아니다.

결과 해석 경계: trace 탐색 도중 TUNE 경로의 첫 행 하나를 잘못 읽었지만 map ID 확인 직후 제외했다. 계산은 아래 네 개의 명시된 TRAIN trace만 사용했다. 이는 새 평가나 훈련 결과가 아니다. 추후 sampler 실험을 실행한다면 transition 재생이 아닌 on-policy 수집/episode 분포 변경인지 명확히 분리하고, 이전 70↔84 속도 보상 화면과 함께 변경하지 않는다.

#### SOTA 속도/완주 실패의 현재 원인 구분 (2026-09-24 16:xx KST)

현재 로컬 PPO actor의 원본 feature trace를 기준으로, 속도 저하와 완주 실패는 서로 다른 현상이 섞여 있다.

- Held-out의 8회 중 6회가 완주했고 중앙 랩타임은 19.32초다. 그 묶음의 median mean-speed는 약 40.4, median max-speed는 약 53.0이다. 13초 목표까지는 랩시간을 약 33% 줄여야 한다.
- 공식 Track2 seed101은 25.52초/평균속도 40.62로 완주했다. 반면 공식 Track1 seed42는 진행도 0.1625에서 163 decisions 후 off-track으로 끝났고 충돌 2회, 손상 0.4를 기록했다. Track1+장애물은 진행도 0.1484의 63 decisions에서 crash, 충돌 5회, 손상 1.0으로 끝났다. 이 실패들은 전체 랩속도만 조정해서 해결할 수 없다.
- Track1+장애물 trace의 55–63 구간은 속도 약 42에서 진행도 .14–.15로 진입한 고곡률 구간(곡률 .32–.85)이며 urgency는 .81–1.0으로 높아진다. 해당 구간의 gas는 0–.06, brake는 한 결정에서 .08이고 나머지는 거의 0, steering은 약 -.09에서 -.58로 움직인다. 이어 5회 충돌과 damage 1.0이 발생한다. 시각 feature가 위험 구간에서 반응을 유발했지만 회피에 성공하지 못했다는 관측이다. 행동만으로 조향 지연 또는 급선회가 원인이라고 확정할 수는 없다.
- 같은 checkpoint lineage의 PPO update-8 기록은 8,192-decision continuation 이후 gas mean .066/max .117, brake mean .00565, policy loss 약 0.000055, value loss 약 40.43이다. 당시 설정은 speed reward target 50/weight .1, LR 2e-6이었다. 이는 보상 설정과 학습 신호가 현재 13초 속도 목표를 강하게 밀지 않았다는 가설을 지지하지만 loss 수치만으로 actor 업데이트 부족을 인과 증명하지는 않는다.
- 이미 보유한 zero-shot throttle expansion screen은 같은 TUNE geometry를 두 번 반복했다. gas ceiling .12→.24로 바꾸자 lap 19.88→19.50초, 평균속도 42.00→42.74로 소폭 변했지만 충돌 2→6, 최종 damage .2→.6이고 under-13은 양쪽 0/2였다. 따라서 액추에이터 상한을 키우는 단독 조치는 채택하지 않는다. 반복 seed는 독립 geometry 복제가 아니다.


#### 2026-09-24 재색인 및 TRAIN-only 교사 표적 감사

RULES 게이트의 실제 재색인에서 이전 문서에 반영되지 않은 PPO held-out 요약이 발견됐다. `eval-brake-reward-heldout/summary.json`은 5/5 완주, 중앙 랩 19.26초, 충돌 0, 손상 0으로 기록하지만 다섯 seed 모두 같은 단일 held-out map ID를 사용하고 관측 지표가 완전히 동일하다. 이 요약에는 checkpoint 경로/해시, 학습 protocol 및 제한 검사 상태가 없다. 그래서 기록 DB는 `restriction_status=unknown`으로 분류했고 SOTA 승격을 거부했다. 이 값을 독립 맵 5회 재현이나 확인된 제출 개선으로 해석하지 않는다.

가까운 이름의 공식 진단 산출물 `site-map-official-domain-brake-reward-fullsplit-ppo8192-u8-lr2e6-seed8101/policy.pt`는 다른 확인 문제를 드러낸다. checkpoint metadata는 요청량 8,192 decision 중 이번 run 2,048 decision, PPO update 2/8에서 멈췄다고 기록한다. 해당 후보의 official 로컬 비교는 7회 중 2회 완주했고, 공식 Track1 seed42는 progress .163에서, Track1+custom obstacle은 .989에서 damage 1.0으로 끝났다. held-out 8회는 6회 완주, median 19.42초다. 따라서 이 파일명만으로 완전 학습된 U8 후보라고 볼 수 없고, 공식 완주 실패가 개선되지 않았다. 두 held-out 요약의 checkpoint 연결도 raw metadata만으로 입증되지 않아 서로 합치지 않는다.

속도 원인은 행동 통계와 보상 설정이 함께 가리킨다. 기존 SOTA의 held-out 평균 속도 중앙값은 약 40.4 m/s이고 최고 속도 중앙값은 약 53 m/s다. 학습 기록은 gas 평균 .060, 최대 .117, brake 평균 .0092이며 이전 reward target 50/weight .1을 썼다. 이는 속도 목표가 랩 요구보다 낮고 actor가 허용된 가속 상한도 충분히 쓰지 않았다는 가설과 맞는다. 단, 더 높은 pedal 상한 단독 실험은 충돌을 늘렸으므로 throttle만 더 여는 처방은 기각한다.

누락된 D 응답을 기다리는 대신 정확한 v2 protocol의 네 TRAIN map/seed cell만 읽는 교사 행동 감사를 직접 실행했다. 분할 loader는 `train` 그룹만 물질화했다. 총 990 decision에서 교사는 네 cell 모두 완주했고 충돌/손상은 0이었다. 긴급 장애물 122 decision에서 평균 gas .041, brake .048이었고, high-urgency 84 decision에서는 평균 gas .030, brake .060이었다. 고곡률 58 decision에서는 평균 gas .010, brake .075였다. 수집 순서도 현재 observation → 교사 행동 → 같은 transition 적용으로 일치한다. 따라서 교사 표적의 시점 오류나 TRAIN에서의 기본 장애물 회피 실패는 현재 원인 후보가 아니다. PPO가 교사 방문 분포에서 벗어난 뒤 긴급 상태에 오래 머무는 분포 이동은 여전히 유력하지만, 별도 on-policy 측정이 필요하다.

운영환경도 바로잡았다. 시스템 기본 Python 3.13/Torch 2.10은 v2 freeze와 달랐으나 기존 `.venv` Python 3.11.15/Torch 2.1/NumPy 1.26/Box2D가 발견됐고 frozen runtime과 정확히 일치한다. paired PPO source는 해당 환경에서 import/CLI preflight를 통과했다. 다음 단일변수 학습은 동결된 v2의 70 대 84 m/s target contrast다. 84 arm이 속도를 높이지 않거나 완주/손상을 악화시키면 이 가설을 버리고, actor의 저속 행동 대신 곡률 구간의 PPO 방문 상태·방향제어 학습을 먼저 바꾼다. 기존 RULES 실행 게이트로 기록하며 SOTA 포인터는 독립 평가를 통과하기 전까지 유지한다.
#### PPO 속도 목표 대조 결과와 업데이트 붕괴 진단 (2026-09-24 18:01 KST)

속도 목표 70/84 paired 학습을 모두 8,192 decision까지 완료하고, 동결된 TUNE 두 reset seed에서 고정 U8 actor를 평가했다. 네 actor(2 PPO seed × 2 Tune seed label) 모두 완주하지 못했다. 70 목표는 평균 진행도 .658, 평균 속도 31.60 m/s, 충돌 decision 8, 평균 최종 손상 .40으로 off-track 종료했다. 84 목표는 평균 진행도 .315, 평균 속도 27.19 m/s, 충돌 decision 12, 평균 최종 손상 .60이었고 crash/off-track으로 종료했다. 84 목표는 안전 비열화 및 진행 악화로 기각한다. Tune 두 seed label은 같은 고정 geometry의 반복 reset이고, 같은 policy/훈련 seed의 기록은 행동 trace까지 같으므로 네 row를 네 독립 맵 표본으로 세지 않는다.

동일한 두 PPO seed의 PPO 전 shared-post-BC actor를 별도 Tune 진단했다. seed8104 BC actor는 두 반복 reset 모두 progress .621에서 다섯 충돌 후 crash; seed8105 BC actor는 두 반복 reset 모두 완주(18.96초), 충돌 1회, 손상 .2, 평균속도 44.56 m/s였다. seed8105의 같은 초기 BC actor에서 출발한 두 U8 PPO actor는 Tune에서 모두 off-track으로 끝났고, 각각 progress .584/.374에서 멈춘 뒤 속도 0에서 가속만 반복했다. 따라서 PPO fine-tuning이 이 seed의 성공 가능한 warm-start 궤적을 보존하지 못한 직접 증거가 있다. 다만 Tune은 한 geometry뿐이고 두 training seed이므로 일반화 원인 판정은 아직 제한적이다.

PPO 학습 기록은 매 update 고정 4 epoch, minibatch 32를 사용하며 KL 조기 중단이 없다. 네 팔의 minibatch 근사 KL은 약 .048–.312, U8 gradient norm 로그는 약 52.6–82.7까지 상승했다. 이는 큰 policy update 가설과 맞지만, 단독 인과 증명은 아니다. 다음 단일변수 실험은 PPO learning rate를 3e-4에서 1e-4로 낮추는 것이다. target 70/84 비교, 2 seed(8104/8105), 네 TRAIN cell, 8,192 decision 예산과 모든 나머지 항목은 그대로다.

새 frozen protocol experiments/speed-target-70-vs-84-lr1e4-v1/protocol-v1.json의 SHA-256은 f01c055f8302b50202bcc06fed122171959b722a229cddf27b4e9943fc2ed188이다. 이전 프로토콜과 JSON 비교에서 ppo_learning_rate만 바뀌었고 TRAIN/TUNE map hash, split, runtime, seed, target 및 예산은 동일하다. RULES 실행 게이트로 새 학습을 시작했으며 현재 4회 중 0회 완료, split loader는 TRAIN만 열었다. 이번 비교도 TUNE 진단용이며 SOTA 승격은 held-out/official 평가와 restriction pass 없이는 하지 않는다. 결과 산출물: artifacts/haic/ppo-speed-target-70-vs-84-u8-20260924/evaluation-u8/summary.json, artifacts/haic/ppo-speed-target-70-vs-84-u8-20260924/evaluation-bc-tune-retry4/summary.json, 새 LR 실험 artifacts/haic/ppo-speed-target-lr1e4-70-vs-84-u8-20260924/.


#### LR 1e-4 실험 중간 로그와 속도 보상 점검 (2026-09-24)

seed8104의 target70/84 두 팔은 모두 U8 8,192 decision 학습을 끝냈고, seed8105 두 팔은 아직 완료되지 않았다. 아직 4개 팔 전체가 끝나지 않았고 held-out/Tune/official map은 열지 않았다. 따라서 아래는 학습 중간 진단이며 후보 성능 판정으로 쓰지 않는다. target70의 첫 팔 update1–8에서 minibatch 평균 근사 KL은 0.120에서 대체로 0.016–0.034로 내려와 이전 LR 3e-4 실험의 약 0.048–0.312보다 작았다. 그러나 배치 평균 속도는 40.5 m/s에서 대체로 25–33 m/s에 머물렀고, 배치당 gas 평균은 0.10–0.12로 action ceiling 0.24의 절반 안팎이며 gas saturation은 0이었다. update3/7/8의 collision decision은 각각 9/10/7이고 max damage는 1.0이었다. update별 평균 진행도는 .15–.23 범위여서 작은 KL만으로 주행 품질이 회복됐다고 볼 수 없다.

보상식은 안전과 속도 압력의 크기 차이를 확인할 수 있다. 현재 `SPEED_SHORTFALL_PENALTY=0.6`에 `REWARD_SCALE=0.1`을 적용하면 직선·무장애 상태의 속도 부족 손실은 decision당 최대 약 0.06이다. 반면 시각 hazard가 차선을 가로막을 때 `OBSTACLE_THROTTLE_PENALTY=8`은 gas가 ceiling에 가까우면 최대 약 0.8까지 내려간다. 충돌 벌점은 scaling 후 -6이다. 이는 hazard 구간에서 속도를 포기하는 쪽이 유리해질 수 있는 설계 신호지만, 현재 로그의 평균 gas도 ceiling에 닿지 않으며 collision decision이 많아서 단일 원인으로 확정하지 않는다. 특히 hazard-present만으로 risk가 높다고 계산하지는 않고, urgency와 차선 겹침을 함께 곱한다.

중간 판별: LR 감소는 policy update 크기를 줄였으나, 첫 팔의 낮은 속도와 잦은 충돌을 고치지 못했다. 이는 `LR만 낮추면 BC warm-start를 보존한다` 가설을 약화하지만 아직 폐기하지 않는다. 4팔 학습을 끝내 고정 U8 actor를 같은 Tune reset으로 평가한다. LR 감소 actor가 Tune 완주/진행도를 회복하지 못하면 다음 단일변수 실험은 frozen BC actor 분포에 대한 KL anchor를 PPO loss에 추가하는 것이다. 이 방식은 픽셀 actor가 PPO를 계속 수행하면서 초기 성공 궤적에서 지나치게 멀어지는지 직접 판별하며 teacher/privileged state를 runtime에 넣지 않는다.


#### BC 성공 궤적과 PPO 실패의 행동 차이 (2026-09-24)

고정 Tune 기록의 seed8105 두 reset은 같은 한 geometry의 반복이므로 독립 표본으로 세지 않는다. 그래도 원인 후보를 좁히는 짝지은 진단이다. PPO 이전 BC actor는 두 reset 모두 완주했고 평균 속도 44.6 m/s, 평균 gas 0.10, 평균 절대 steering 0.10이었다. LR 3e-4 PPO U8 target70은 두 reset 모두 DNF, 평균 속도 24.5 m/s, gas 0.10, 절대 steering 0.09였고 둘 다 off-track으로 끝났다. target84는 평균 속도 20.0 m/s, gas 0.15, 절대 steering 0.07이며 역시 둘 다 off-track이었다. 그러므로 이 seed에서 느림은 낮은 평균 gas만으로 설명되지 않는다. PPO 후 steering 명령 크기가 작아지고 차를 도로 위에 유지하지 못해 랩 진행 자체를 잃은 패턴이 더 잘 맞는다. 이 비교는 PPO가 BC보다 느린 속도로 주행했다는 효과와 완료 여부를 분리하지 못하고, 단일 geometry라 일반 원인 증명은 아니다.

코드도 후보를 준다. policy mean, critic value head, auxiliary head가 같은 128차원 fusion latent를 공유한다. 현재 update loss는 `policy_loss + 0.5 * value_loss + 0.1 * auxiliary_loss - 0.01 * entropy`이고 로그의 `value_loss`는 약 9–18, `policy_loss`는 약 0.008–0.066이다. loss 원값 비교만으로 gradient 간섭을 입증할 수는 없지만, critic/auxiliary gradient가 시각 encoder를 크게 움직여 BC steering feature를 흔들 가능성이 있다. 이는 이전의 큰 PPO KL보다 한 단계 구체적인 후보이며 원인으로 확정하지 않는다.

다음 판별 후보는 frozen BC actor의 action 분포에 KL anchor를 두는 PPO와 anchor 없는 paired control이다. 두 arm 모두 RL reward로 업데이트하며 BC actor는 학습 때 기준 분포로만 읽고 제출 actor에는 포함하지 않는다. 기준 actor와의 KL은 각 rollout 관측에서 계산한다. 현재 PPO의 approx-KL은 직전 rollout policy 대비 값이라 여러 update를 거친 BC와의 누적 거리 자체를 제한하지 않는다. OpenAI Spinning Up PPO 문서도 clipping만으로 지나치게 멀어진 update가 생길 수 있다고 설명하며 mean KL 임계치 조기 종료를 사용한다: https://spinningup.openai.com/en/latest/algorithms/ppo.html. 이번 LR 1e-4 Tune 결과에서 정책 보존 실패가 재현되면, reference-KL 계수만 바꾼 두 arm을 같은 TRAIN maps/seeds/decision budget으로 비교한다. 보조 원인인 shared critic gradient는 별도 후속 실험에서 critic loss의 encoder 전달만 차단해 분리한다.

중간 추가 진단: frozen BC checkpoint와 PPO checkpoint의 파라미터별 상대 L2 차이를 읽기 전용으로 계산했다. seed8104 target70 U8에서 full-frame encoder 19%, HUD encoder 20%, visual-feature encoder 7%, fusion 24%, policy-mean head 7%, log-std 0.4%, value head 45%, auxiliary head 29%였다. seed8105 target70 update5에서는 각각 17%, 8%, 3%, 17%, 7%, 0.5%, 36%, 25%였다. 낮은 LR이어도 shared visual/fusion 표현과 critic이 꽤 움직이는 점은 BC 보존 실패 가설과 맞지만, 파라미터 거리만으로 어떤 loss가 이를 만들었는지는 알 수 없다. Tune은 아직 열지 않았다. 모든 변경은 현행 프로세스가 소스 fingerprint 검증을 끝낸 뒤 진행한다.

#### 충돌 직전 시야와 회피 방향 진단 (2026-09-24)

LR 1e-4 학습 중 확보된 TRAIN decision trace(완료 팔 3개와 마지막 팔의 일부)를 읽기 전용으로 묶어 봤다. 충돌 시작점 중 대부분에서 픽셀 feature는 장애물을 이미 표시했다. 현재 관측된 팔별 충돌 시작의 직전 5 decision 구간에서 평균 obstacle urgency는 0.82–0.86이었고, 해당 구간의 89–100%에서 장애물 present가 켜져 있었다. 마지막 팔의 trace는 미완료라 이 비율은 중간값이다. 즉 이 표본의 충돌은 단순히 시각 detector가 전혀 보지 못해서 난 것만은 아니다. teacher의 pixel-side convention(장애물이 왼쪽이면 오른쪽으로 조향, 오른쪽이면 왼쪽으로 조향)으로 collision precursor action을 검사하면 원하는 방향 조향 비율은 팔별 0.41–0.60에 그쳤다. 충돌 순간은 0.50–0.72로 조금 올라가지만 이미 늦을 수 있다. 이 분석은 proxy feature 부호와 5-step window에 의존하므로 정식 인과 결론은 아니다.

장애물 측면을 연속 frame에서 읽는 것도 후속 후보로 남긴다. `extract_temporal_features`에는 장애물이 현재 화면 중앙에 들어올 때 이전 frame의 측면 위치를 주는 feature가 이미 있지만, 현재 PPO manifest는 `use_temporal_features=false`라 actor가 이 보조 입력을 받지 않는다. 다만 더 직접적인 근거는 성공했던 BC actor와 PPO 후 actor의 조향·완주 차이이므로 LR 실험의 고정 Tune 결과가 실패하면 frozen BC reference-KL PPO를 먼저 판별한다. 그 방법이 기존 경로 완주를 보존하지만 장애물 회피가 계속 실패할 때만 temporal feature를 하나의 단일변수 PPO 실험으로 켠다. 검증에서 쓸 수 있는 것은 끝까지 Tune뿐이며 held-out/official map은 후보를 고정한 뒤 RULES gate를 통과할 때만 연다.

#### LR 1e-4 PPO 속도 목표 실험 최종 Tune 결과 (2026-09-24)

네 arm 모두 8,192 TRAIN decision까지 끝난 뒤 고정 Tune evaluator를 실행했다. target70과 target84는 각각 0/4 완주, 0/4 13초 미만 유효 완주였고 완료 랩타임은 없었다. Tune의 두 seed label은 장애물 5개인 같은 geometry의 반복 reset이며, 한 PPO seed의 두 trace가 동일하므로 이를 독립 맵 표본 네 개로 해석하지 않는다.

target70의 평균 DNF 진행도는 .5658, 평균 속도 31.62 m/s, 충돌 decision 16회(4 episode), 평균 최종 손상 .8, 최대 손상 1.0이었다. target84는 평균 진행도 .4115, 평균 속도 30.81 m/s, 충돌 decision 2회(2 episode), 평균 최종 손상 .1, 최대 손상 .2였다. 84 m/s 목표는 같은 화면에서 충돌·손상을 줄였지만 완주와 진행도를 개선하지 못했다. 이전 LR 3e-4 비교와 합치면 학습률을 1e-4로 낮추는 것만으로 성공한 BC warm-start를 보존한다는 가설은 지지되지 않는다. 특히 seed8105의 BC actor는 이 Tune geometry를 18.96초에 완주했지만, PPO 후 actor는 target70/84 모두 off-track으로 끝났다.

이 결과는 Tune-only 탐색 증거다. held-out/official 지도는 열지 않았고 `SOTA.md` 포인터는 그대로다. 별도 후보 검증은 승인된 checkpoint SHA-256을 찾지 못해 중단되었으며, 다른 현재 checkpoint로 먼저 수행한 14회 평가는 승인된 후보의 성능 증거로 세지 않는다. 다음 PPO 실험은 frozen BC actor의 행동분포 KL 항을 PPO loss에 하나만 추가해 BC가 가진 조향 궤적을 보존하는지 확인한다. 기준 actor와 teacher는 학습 시점에만 읽고 제출 actor에는 넣지 않는다.

같은 실행에서 RULES 결과 수집의 결함도 수정했다. `target_summaries` 형식 Tune 요약을 speed target별 기록으로 나누고 split, episode 수, finish 비율, 평균 진행도, 충돌 episode, 평균 손상, 추론 지연, 완료 랩타임을 정규화한다. `RESULTS.md` 재생성 결과 target70/84 각각 Tune 4회·완주 0으로 원본과 일치한다. `RULES` plan에 비 actor 전략별 승격 판단을 추가해 결과 DB의 비교가 실행 가능한 판정으로 이어지게 했다. 관련 연구 운영 및 CLI 테스트 11개 통과.



#### PPO BC reference-KL paired screen 중간 결과 — seed8104 (2026-09-24 19:24 KST)

고정 프로토콜 `ppo-bc-reference-kl-speed70-lr1e4-20260924`에서 seed8104의 두 arm이 각각 8,192 TRAIN decision / 8 updates를 완료했다. 업데이트 로그의 8개 equal-sized rollout 평균을 비교하면 KL=0 control 대비 reference-KL=0.5에서 평균속도는 32.19→34.41 m/s, 평균진행도는 .1780→.1874, collision decision은 39→23으로 변했다. 평균 gas는 .1097→.1062로 비슷하고 두 arm 모두 gas saturation은 0이었다. 평균 절대 조향은 .3382→.3136, 평균 brake는 .00974→.00804, brake action 비율은 10.01%→9.52%였다. 최대 damage는 둘 다 1.0이다. 따라서 관측된 속도 차이는 gas 증가보다 낮은 조향 크기·제동량과 함께 나타났다. 다만 8,192 TRAIN decision을 update×cell 32개 segment 안에서 연속 비교하면 인접 steering 절대 변화량은 .381 대 .379로 거의 같고, |steer|≥.1 인접쌍의 부호 반전은 1,000쌍당 365 대 395로 KL arm에서 오히려 높다. 그러므로 KL이 조향 oscillation을 줄여 속도를 올렸다는 기전은 이 seed 로그에서 지지되지 않는다. 같은 TRAIN rollout 지표의 상관관계이므로 원인은 미결이다. KL arm의 update-8 rollout은 control보다 평균속도가 30.27→37.71 m/s로 높았지만 진행도는 .2289→.1957로 낮고 collision decision은 7→5였다. reference KL은 실제 적용됐으며 arm의 평균 기록값은 .1484이다.

이는 같은 고정 TRAIN cell에서 관측한 학습 rollout 통계일 뿐, 독립 평가 episode, 완주율, Tune 결과가 아니다. 높은 평균속도나 적은 충돌 decision만으로 개선을 판정하지 않는다. 양 arm 모두 완주 decision이 없고 최대 손상 1.0이므로 현 시점 우열은 미결이다. 두 번째 seed 쌍은 시작됐고 control arm이 현재 update 0이다(4개 실행 중 2개 완료). Tune/held-out/official은 모두 닫혀 있다. 프로세스 PID 30480은 응답 중이며 CPU time 증가로 같은 실행의 진행을 확인했다. 전체 네 실행이 고정 U8까지 끝난 뒤에만 protocol의 Tune screen을 수행한다. SOTA/RESULTS는 아직 변경하지 않는다.

#### PPO BC reference-KL paired screen 최종 Tune 결과 및 제출 판단 (2026-09-24)

고정 U8 Tune 요약 `artifacts/haic/ppo-bc-reference-kl-speed70-lr1e4-20260924/run/evaluation-u8/summary.json`을 기준으로 비교했다. KL=0 control은 0/4 완주, KL=0.5 arm은 2/4 완주했고 두 완주는 모두 training seed8104에서 나왔다. 두 Tune seed label은 장애물 5개인 동일 map geometry의 반복 reset이므로 독립 지도 네 개로 세지 않는다. KL arm도 4/4 episode에서 충돌을 기록했고, 완주 두 건은 각각 19.6초·최종 damage .6이었다. 나머지 seed8105 반복 reset은 progress .259에서 damage 1.0으로 crash했다. 두 arm 모두 13초 미만 유효 완주는 0/4이며, 비교기는 seed 간 안전성 악화를 확인해 `rejects_on_safety`로 판정했다.

따라서 KL arm의 2/4 완주는 후속 실험을 정당화하는 약한 신호일 뿐 제출 개선 근거가 아니다. held-out/official 평가와 승격 조건을 통과하지 않았고 `SOTA.md` 포인터는 갱신하지 않았다. `RESULTS.md` 색인기는 이 실험의 `arm_summaries`를 arm별 Tune 행 두 개로 기록하도록 보완했고, 재색인 뒤 전체 142개 결과 및 KL arm 두 행(ac5fd03fb34f, b820eb170aab)이 확인됐다. 두 기록의 restriction 상태는 unknown이며 Tune-only screen으로 남긴다.

현재 제출 archive는 `artifacts/haic/submission/haic-obstacle-risk-ppo-actor.zip`이다. 기존 선택 기록은 actor-only 패키지 smoke와 당시 전체 테스트 통과를 보유하지만, 해당 정책은 held-out 6/8 완주·중앙 19.32초이고 공식 Track1 seed42 및 추가 장애물 지도에서는 완주하지 못했다. 그러므로 archive는 현재의 포맷 기준선/비상 제출물로 보관할 수 있으나, 13초 목표나 좋은 순위를 기대할 제출 후보로 판정하지 않는다. 외부 업로드는 수행하지 않았다.

같은 시점에 A/B/C/D 실제 작업을 compact 조회했다. C는 승인된 checkpoint SHA-256과 일치하는 파일을 찾지 못했고, 검색 과정에서 새 평가나 파일 변경을 하지 않았다고 보고했다. A/B/D의 최신 turn은 완료 상태였으나 compact thread 응답에 정량 보고나 새 산출물이 노출되지 않아 성능 증거로 세지 않는다. 이 조회에서 KL screen보다 나은 새 결과는 확인되지 않았다.

#### 2026-09-24 조율 재개 및 KL 보존 강도 후속 실험 배정 (20:21 KST)

반복 goal과 `haic-13` 10분 heartbeat가 모두 ACTIVE임을 확인하고, 최신 A/B/C/D task 상태를 새로 compact 조회했다. 직전 작업들은 완료 또는 idle/notLoaded였고 새 성능 원자료를 내지 않았다. C의 후보 checkpoint hash 검색은 불일치를 확인해 종료했으며, 그 파일을 대상으로 승인된 성능 증거는 없다. 현재 돌아가는 학습 프로세스도 확인되지 않았다.

현재 제출 기준은 여전히 `SOTA.md` actor-only ZIP 하나다: held-out 6/8, median 19.32초, p90 22.42초. 공식 Track1 seed42와 추가 장애물 실행은 완주하지 못했고, under-13 완주 증거는 없다. Teacher-only 대 DAgger는 2/4 대 1/4 완주(23.59초 대 24.16초), post-collision downweight는 1/4(23.92초), reference-KL 0 대 0.5는 0/4 대 2/4였지만 treatment가 4/4 episode에서 충돌했고 19.6초에 그쳤다. 속도 목표 70/84 paired screen도 완주 0/4로 끝났다. 따라서 새 제출 후보는 확인되지 않았다.

KL=0.5의 2/4 완주 신호와 4/4 충돌은 한 가지 미검증 질문을 남긴다. 기준 BC actor에서 멀어지는 업데이트를 더 강하게 억제하면 충돌이 줄면서 완주를 유지할 수 있는가? B(`01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de`)에 후속 paired 실험을 보냈다: KL coefficient 0.5 대 2.0만 변경하고, seed 8104/8105, 동일 TRAIN 네 cell, 8,192 decision/arm, lr 1e-4, speed target 70, 고정 U8를 유지한다. 새 protocol과 artifact 경로 및 source/split/map/runtime hash를 먼저 고정하고, 모든 run 뒤 custom TUNE만 평가한다. held-out/official 접근·SOTA 갱신·제출은 이 화면에서 금지하며 RULES 실행 게이트가 거부하면 우회하지 않는다. 판정은 완주율, under-13 비율, collision/damage, 진행도, finisher lap-time 중앙값/p90을 함께 본다. treatment가 충돌/damage를 줄이지 못하거나 완주·진행도를 잃으면 가설을 기각하고 같은 계수 탐색을 반복하지 않는다.

#### 2026-09-24 진행 확인 및 독립 행동표현 가설 준비 (20:34 KST)

현재 B task는 inProgress지만 새 protocol, artifact, 학습 프로세스는 아직 확인되지 않았다. 11:28 UTC 프로세스 검색에서는 KL/paired training 명령이 없었고 기존 KL 실행만 completed 4/4로 남아 있었다. B에는 한 번만 현재 단계·구체적 blocker를 물었으며, 같은 실험 지시를 반복하지 않고 기다린다. 이 상태만으로 training 중이라고 기록하지 않는다.

제출 ZIP을 다시 읽기 전용 점검했다. `haic-obstacle-risk-ppo-actor.zip`의 SHA-256은 `451831350208c328734e69d9c87c86f7aef7cb4c6b5a00e484c2eca260854b14`이고 ZIP CRC 검사는 통과했다. manifest의 runtime policy는 `trained_visual_actor`, `planner_enabled=false`, strict loading이며, 여기에는 현재 held-out/official 원자료 성능을 바꾸는 새 근거가 없다. `final-ppo-actor-selection.json`은 held-out 6/8, median 19.32초, p90 22.42초, 공식 Track1 seed42 progress .163 및 plus-obstacle progress .148에서 미완주라고 다시 확인했다.

B의 KL 실험과 자원 충돌이 없는 후속 아이디어로, A(`01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1`)에는 사전 등록된 categorical brake/coast/throttle 표현을 읽기 전용으로 감사하도록 요청했다. 기존 signed-longitudinal 연속 분포와 teacher/learner label의 gas·brake·near-zero 비율 및 aliasing을 TRAIN 원자료로 측정하고, 데이터 근거가 있을 때만 PPO actor의 종방향 head 하나를 3-mode+조건부 크기로 바꾸는 paired 시험을 설계한다. TUNE/held-out/official 입력, 학습, 평가, 코드 변경은 이번 분석에서 금지했다. 이 아이디어는 성능 개선 결과가 아니라 다음 독립 가설이며, 근거가 부족하면 폐기한다.



#### Frozen BC-reference KL 0.5 vs 2.0 follow-up (2026-09-24 20:37 KST)

- Unique artifact root created before execution: `C:\Users\koi\Coding\HAIC\artifacts\haic\ppo-bc-reference-kl-0p5-vs-2p0-speed70-lr1e4-20260924T113750Z`.
- Status at creation: setup only; training and evaluation have not started. RULES control-plane audit returned `ready=true`; source, split, TRAIN/TUNE map and runtime hashes match the prior frozen KL screen. Pre-existing KL/PPO jobs were not found.
- Run directories will be created beneath this root only after the new two-arm protocol is frozen. No SOTA or RESULTS change has been made.
- Isolated one-off runner created: `C:\Users\koi\Coding\HAIC\artifacts\haic\ppo-bc-reference-kl-0p5-vs-2p0-speed70-lr1e4-20260924T113750Z\paired_bc_reference_kl_0p5_vs_2p0.py`. It overrides only the registered arm coefficients to 0.5 and 2.0; the original worktree runner remains untouched. Its hash will be included in the frozen protocol before training.

#### 2026-09-24 20:50 KST C의 최신 비승격 진단: 속도와 완주 실패를 분리

`strategy-c/.../post-selection-diagnostic-v1/summary.json`과 `episodes.jsonl`을 직접 재분석했다. 진단은 7개 로컬 셀을 두 actor에 각각 실행한 재사용 평가이며, 공식 점수나 신규 일반화 검증으로 보지 않는다. 원시 episode 파일 SHA-256은 `55c81a8720e0f2c9b9d4d2593d7402a035a40ad05b9abff950c1c7adfa7ec5ea`다.

- 기존 PPO SOTA는 7회 중 5회 완주, 유효 13초 미만은 0회, 완주 중앙값 19.32초, 충돌 총 7회, 평균 손상 0.20이었다. 다섯 완주는 모두 동일한 사용자 제작 geometry의 seed 반복에서 나왔다. 공식 Track1 seed42와 추가 장애물 셀은 완주하지 못했다.
- 비교 후보는 7회 모두 미완주, 유효 13초 미만 0회, 평균 진행도 0.291, 충돌 총 7회, 평균 손상 0.20이었다. 평균 최고속도는 62.36 m/s로 SOTA의 49.84 m/s보다 높았지만 완주율이 더 나빴다. 따라서 속도 목표나 throttle만 올리는 처방으로는 부족하다.
- SOTA Track1 로그에서는 진행도 0.163에서 약 39.3 m/s로 첫 접촉 후 7.6 m/s로 떨어졌고, 두 번째 접촉 뒤 진행이 멎었다. 직전·직후 gas는 0.05–0.09, brake는 0이었다. SOTA의 추가 장애물 셀은 진행도 0.148에서 다섯 번 연속 접촉 후 종료됐다.
- 후보의 추가 장애물 셀은 첫 접촉 후 속도 1 m/s 부근에서 진행이 정체됐는데 gas 0.10–0.13과 강한 steering이 계속됐고 다시 접촉했다. 후보의 장애물 없는 사용자 제작 셀도 최고속도 약 65.3 m/s까지 올라간 뒤 진행도 0.237에서 off-track으로 끝났다.

후보 run manifest의 `training/train_policy.py` 해시는 `3785830c…114a56`이며 해당 worktree 소스와 일치한다. 이 학습 소스의 회복 보상은 damaged·저속 상태에서 gas와 steering 크기를 보상하지만 실제 progress를 조건으로 삼지 않는다. 위 정체 행동은 이 보상이 행동량은 주되 탈출 결과를 보장하지 않는다는 기존 D 회복 보상 가설을 재확인한다. 새 가설이나 새 실험으로 세지 않는다. raw diagnostic에는 actor가 실제 입력으로 받은 visual feature가 없어 인식 실패와 policy action 선택은 아직 분리하지 못한다. `nearest_obstacle_distance`는 simulator 진단값이므로 actor 관측으로 취급하지 않는다.

결정: `SOTA.md`와 `RESULTS.md`는 갱신하지 않는다. 재사용한 진단 셀을 학습 자료로 쓰지 않는다. B의 사전등록된 KL 보존 강도 비교는 변경 없이 이어간다. KL 결과가 나온 뒤에는 이미 기록된 temporal obstacle cue PPO ablation을 다음 분기 후보로 삼고, 회복 reward는 기존 TRAIN-only 노출·potential 비교 결과와 중복 여부를 먼저 대조한다.

#### 2026-09-24 21:05 KST 동일 맵 첫 120 decision 행동 대조와 warmup provenance

C의 `episodes.jsonl`에서 같은 custom geometry/seed 20260923의 첫 120 decision을 맞춰 계산했다. Candidate는 평균속도 46.4 m/s, gas 0.118, 평균 |steer| 0.055로 진행도 0.237에서 off-track됐다. SOTA는 같은 120 decision 동안 평균속도 36.7 m/s, gas 0.070, 평균 |steer| 0.093으로 진행도 0.458까지 갔고 전체 episode를 19.32초에 완주했다. 다섯 seed 이름의 custom 셀들은 같은 geometry의 반복이며 같은 요약을 보여 독립 맵 표본으로 세지 않는다.

이는 낮은 throttle이 느린 완주의 유일 원인이라는 설명보다, candidate가 더 빠른 종방향 주행을 하면서 곡률을 따라갈 조향을 충분히 내지 못해 route progress를 잃는 설명과 더 맞는다. 정확한 steering 오류를 확정하려면 시점별 road/heading feature와 실제 입력을 같은 로그에 보존해야 한다.

후보의 run manifest `training/imitation.py` SHA와 해당 worktree 파일 SHA는 모두 `0f7773b999e253696813bbc8a8736fe1e7b9ebfe411fa8f83f162db60cf422e5`로 일치한다. 그 source는 `visual_feature_encoder`를 BC optimizer parameter에 포함하고 candidate protocol은 `use_visual_features=true`, `teacher_warmup_enabled=true`였다. 그러므로 이 run의 feature encoder가 BC에서 빠졌다는 이전 우려는 source-level로 반증된다. 단, manifest에는 전후 encoder weight delta가 없어 실제 변화량까지 증명하지는 않는다.

다음 판별은 진행 중인 reference-KL pair의 U8에서 mean absolute steer와 route progress를 함께 본다. KL treatment가 이를 보존하고 collision/damage를 악화시키지 않으면 PPO catastrophic drift 가설이 강해진다. 그렇지 않으면 다음 단일 변수는 기존에 등록된 temporal obstacle cue PPO ablation 또는 action-distribution 구조 실험으로 진행한다. 현재 프로토콜, SOTA 및 RESULTS는 바꾸지 않았다.

#### 2026-09-24 21:06 KST KL 보존 강도 pair 실제 실행 시작

앞의 setup-only 상태에서 변경됨. frozen protocol SHA-256 `7c594bc9ed00f014cd2b07e1d52c15b42b9c89fa54fdc1b53f6d4013e9725b51`로 B가 실행을 시작했다. `run/training-status.json`은 status `running`, `completed_runs=0/4`, `loaded_split_group=train`, `loaded_episode_count=4`, `shared_bc_states_prepared=0`, `tune_loaded=false`, `held_out_loaded=false`, `official_maps_loaded=false`를 보고한다. driver/runner PID 49632/13460가 살아 있고 응답 중임을 확인했다. 이것이 결과가 아닌 실제 실행 증거다. 학습이 끝날 때까지 동일 계산을 새로 시작하지 않는다.

#### 2026-09-24 21:11 KST KL pair 첫 PPO update 원자료 확인

같은 frozen run에서 첫 arm만 update 1을 완료했다. seed8104/KL=0.5, 총 1,024 TRAIN decisions(네 cell 각각 256), map/source fingerprint는 protocol과 일치했다. 해당 rollout에서 mean progress 0.1548, mean speed 40.50 m/s, mean |steer| 0.2966, collision decision 0, off-track decision 2, reference KL 0.0460이었다. Tune/held-out/official은 불러오지 않았다. 이는 한 arm의 초기 TRAIN rollout일 뿐 비교결과나 완주 증거가 아니다. 전체 실행은 여전히 0/4 run 완료이며 다른 arm/seed의 U8과 고정 Tune screen을 기다린다.

#### 2026-09-24 21:13 KST 첫 arm update 2 중간값 — 승격 판정 금지

같은 seed8104/KL=0.5 TRAIN run의 update 2에서 mean progress 0.1616, mean speed 37.9 m/s, mean |steer| 0.323, gas 0.107, collision decisions 5, off-track decisions 1, reference KL 0.1986이었다. 해당 한 rollout은 collision 신호가 있어 관찰 대상이지만 대조 arm/다른 seed/U8이 끝나기 전에는 효능 판정에 사용하지 않는다. driver PID 49632는 계속 응답하고 CPU time이 증가했다. 전역 `completed_runs`는 0/4, Tune/held-out/official은 미로딩이다.

#### 2026-09-24 21:20 KST D의 TRAIN-only BC 라벨 시점·장애물 관측 감사

D의 읽기 전용 감사를 반영했다. attempt-2 teacher BC collector는 `observation_t`를 교사에 넣고 그 observation과 teacher target을 묶은 뒤 환경을 step하므로, 저장된 BC target 자체에는 한 decision 지연이 확인되지 않았다. transition trace에서 actor 입력/행동은 decision t이고 collision·damage·progress는 해당 행동 이후 결과다. 따라서 collision 결과와 같은 행의 hazard feature를 단순 대조하면 시점이 엇갈려 보일 수 있다.

원시 teacher demonstration은 요약과 digest만 남기고 삭제되어, 교사 action target을 map·hazard별로 직접 비교할 수 없다. 두 PPO seed는 각각 네 TRAIN episode에서 964개 demonstration을 사용했고 digest는 동일하다(`d6a27fc0e546312691d4f2517f3829368aec7ebb0a14763c39235131759d69ed`). BC action MSE는 약 0.503에서 0.0246/0.0234로 낮아졌지만, 이는 target 적합만 보여주며 target 품질이나 다양성을 입증하지 않는다.

사후 actor trace proxy에서는 collision-marked 47행 중 46행에서 decision t의 pixel detector가 urgent obstacle을 표시했다. 다음 decision이 존재한 41행 중 37행에서도 urgent 표시가 유지됐고, 6행은 다음 decision이 없었다. 이는 일반적인 장애물 미검출이나 BC 라벨 한 칸 지연 설명을 약화시키지만, detector 정합성이나 충돌 원인을 증명하지 않는다. 남은 학습 가설은 감지된 hazard를 회피 행동으로 연결하는 actor 학습의 약함이며, 교사 label의 다중모드 평균화도 기존 미검증 가설로 남는다.

근거는 C attempt-2의 TRAIN-only `decision-traces.jsonl`, `ppo-start.json`, run manifest이며 두 map/source hash는 기존 manifest와 일치한다. 원시 경로는 `C:\Users\koi\.codex\worktrees\4fd3\HAIC\experiments\strategy-c\clean-start-train-only-teacher-bc-ppo-seeds-8104-8105-v1-attempt2\`다. 이번 감사는 TUNE/held-out/official에 접근하지 않았고 코드·학습·평가를 수행하지 않았다. 다음 BC label-quality 실험을 정당화할 원시 teacher target은 보존되지 않았다.

#### 2026-09-24 21:25 KST reference-KL 실험 첫 arm U8 완료, paired arm 진행 중

동일 frozen protocol의 `ppo_bc_reference_kl_0p5`, seed 8104가 TRAIN 8,192 decisions/U8까지 완료됐다. U8의 1,024-decision rollout 요약은 mean progress 0.196, mean speed 37.71 m/s, mean |steer| 0.295, gas mean 0.106/max 0.220, gas saturation 0, collision decisions 5, off-track decisions 0, finished decisions 0, reference KL 0.175이다. 이는 네 TRAIN cell rollout 통계이며 완주·랩타임 평가가 아니다.

B는 이어 `ppo_bc_reference_kl_2p0`, seed 8104를 시작했다. 첫 update 1,024 decisions는 시작 actor와 동일한 요약(진행도 0.155, 속도 40.5, 충돌 0, 이탈 2)을 보였고 reference KL은 0.021이었다. 이는 초기화 상태 확인에 가깝고 arm 비교 결과로 해석하지 않는다. 현재 global status는 `completed_runs=1/4`, `running`; process 49632 CPU time 증가와 KL 2.0의 갱신 파일을 확인했다. Tune/held-out/official은 아직 미로딩이다.

현재 TRAIN 수치는 gas 상한을 올린 뒤에도 실제 rollout에서 gas가 포화되지 않고 충돌이 남을 수 있음을 보여준다. 따라서 페달 상한 증대만으로 회피/완주 문제가 풀린다고 보기는 어렵다. 다만 U8의 진행도·속도는 고정 TRAIN rollout 평균이라 원인 확정이나 성능 승격 근거가 아니다. 같은 프로토콜의 네 arm과 사전 고정된 Tune 비교가 끝나기 전까지 체크포인트를 선택하지 않는다.

#### 2026-09-24 21:27 KST 13초 목표와 PPO 속도 보상 목표 불일치 진단

현재 root `training/train_policy.py` source를 읽어 속도 shaping을 대조했다. 공식 트랙 길이 source 주석은 962–1,197 simulator units이며 13초 완주에는 단순 평균으로 약 74–92 units/s가 필요하다. 반면 현재 기본 `SPEED_TARGET=70`은 70을 초과해도 추가 속도 보상이 없고, 실제 shortfall target은 `70 × (1 − 0.5×hazard_risk) × (1 − 0.4×curve_magnitude)`로 장애물·곡선 상태에서 더 낮아진다. 상수 70만 평균속도로 유지해도 코스 길이 범위상 약 13.7–17.1초 규모이며, 가속·코너·장애물은 이 단순 계산에 포함되지 않는다.

따라서 현재 속도 reward는 13초 목표를 직접 표현하지 않는 구조적 불일치가 있고, 느린 랩의 한 원인 후보로 기록한다. 다만 이것이 현재 미완주·충돌의 단독 원인이라는 증거는 아니다. 기존 고속 candidate는 최고속도만 올랐을 때 완주가 악화됐고, 안전한 U8 기준선도 아직 확보되지 않았다. 그러므로 속도 목표를 지금 올리는 학습은 보류한다. 먼저 이미 등록된 temporal obstacle cue PPO on/off 정보 ablation으로 감지된 위험을 회피 행동으로 연결하는지 확인하고, 안전한 완주 기준을 통과한 뒤에만 기존 사전등록 70 대 84 target 비교를 검토한다. 이번 내용은 source-backed 가설이며 성능 실험 결과가 아니다.

#### 2026-09-24 21:31 KST KL contrast early signal and temporal-branch history audit

B’s matched seed8104 has a directional TRAIN-only signal at U3. KL=0.5: mean progress 0.175, mean speed 24.24 m/s, 5 collision decisions, 4 off-track decisions, gas mean 0.120, reference KL 0.315. KL=2.0: mean progress 0.246, mean speed 34.84 m/s, 0 collision decisions, 1 off-track decision, gas mean 0.097, reference KL 0.048. Both use the same seed and four TRAIN cells; the registered variable is only the frozen BC-reference KL coefficient. This makes KL=2.0 promising for safety and useful progress, but U3 rollouts are not full-lap evaluation and policy trajectories already differ. Global status at the snapshot is 1/4 runs complete; KL=2.0 seed8104 is training. No arm selection or SOTA change.

I rechecked the prior `site-map-temporal-branch-ppo2048-u2-seed8101` before repeating its idea. The artifact has `use_temporal_features=true`, `use_hud=false`, two PPO updates, and `checkpoint_updated=false` on both. `save_best_checkpoint()` retains the source if Tune rank does not improve; the exported `policy.pt` SHA is exactly the same as its warm-start SHA (`4CEA8328960CFC91489D9F1DC75986F33AC43DF7B2CA94E2075C5BA4DA45F7F6`). Thus PPO updates ran, but no updated temporal actor beat the warm-start under that selection gate. Its reported ablation toggled HUD and produced identical metrics; it never compared temporal cue on versus off. The old run is not causal evidence for the cue and does not close this hypothesis.

For a valid follow-up, retain the same actor architecture/initialization in both PPO arms, feed a zero cue in control and the actual previous-obstacle-side cue in treatment, keep the selected KL coefficient and every other PPO setting fixed, and compare paired seeds at fixed U8 before separate Tune evaluation. First verify the input mode can be packaged faithfully with planner disabled. This is design only until B finishes and A confirms source/runtime parity.

#### 2026-09-24 21:37 KST seed8104 paired U8: higher reference KL slows speed without collision reduction

The frozen B run completed both seed8104 arms (2/4 total arms). At fixed U8, KL=0.5 recorded TRAIN rollout mean progress 0.196, mean speed 37.71 m/s, gas mean 0.106, 5 collision decisions, 0 off-track decisions, and reference KL 0.175. KL=2.0 recorded progress 0.241, speed 33.70 m/s, gas mean 0.098, 6 collision decisions, 1 off-track decision, and reference KL 0.051. Neither arm logged a finish in its 1,024-decision U8 rollout. These are same-seed/four-cell TRAIN summaries, not lap evaluation.

Higher KL substantially reduced the measured reference divergence, but it did not reduce collision counts and was about 4.0 m/s slower at U8; its mean progress was higher. This is a mixed tradeoff, not a winner. The second seed pair and protocol-defined evaluation remain necessary; no Tune, held-out, or official data were loaded at this snapshot. Keep both checkpoints diagnostic and do not update SOTA/RESULTS.

#### 2026-09-24 21:44 KST correction: 70/84 speed target is already tested; do not rerun

The existing artifact and `RESULTS.md` entries show that both 70-vs-84 PPO speed-target contrasts are completed; the 21:27 note above mistakenly described the 70/84 comparison as a future test. The LR=3e-4 paired U8 screen at `artifacts/haic/ppo-speed-target-70-vs-84-u8-20260924/evaluation-u8/summary.json` had 0/4 finishes for both targets. On the repeated single TUNE geometry, target84 was worse than target70: mean DNF progress 0.315 vs 0.658, mean speed 27.19 vs 31.60 m/s, 12 vs 8 collision decisions, and mean final damage .60 vs .40. This is not four independent map samples.

The separate LR=1e-4 contrast is already recorded in `RESULTS.md` and in the earlier section above: it also had 0/4 finishes for both targets; target84 lowered collisions/damage on this one geometry but did not improve progress or speed. Together the two studies do not support repeating a speed-target-only PPO change. The stronger root cause to pursue is PPO forgetting a BC actor that could finish on seed8105, followed by weak or late steering into detected hazards. Continue the active BC-reference-KL pair and the source/runtime preflight for a cue-only policy-information experiment; keep all checkpoints diagnostic until matched U8/Tune and later held-out evidence exists.

#### 2026-09-24 21:54 KST PPO rollout progress-collapse audit

The frozen reference-KL comparison is still running at `completed_runs=3/4`; only TRAIN has been loaded, so no checkpoint has passed its protocol-defined TUNE evaluation. The live process completed seed8105/KL=0.5 at U8. Its 1,024-decision summary is mean progress 0.1533, mean speed 33.89 m/s, 6 collision decisions, 3 off-track decisions, maximum damage 1.0, and zero finishes. This is training evidence, not a lap-time result.

In the same seed8105 `custom-track-haic-train-20260921:20260925` TRAIN cell, the KL=0.5 rollout trace changed sharply by PPO update: at U1 the two rollout trajectories reached about progress 0.58, while at U7 their final positions were about 0.064 and 0.080. U7 had mean speed 42.9 m/s, 0 collisions, 1 off-track decision, and 223/256 trace rows with no positive progress delta. U1 had mean speed 29.8 m/s and 4 collision decisions. These are sampled TRAIN rollouts, not deterministic submission behavior.

The map stores its first obstacle at progress 0.2467. The U7 trajectories stopped before progress 0.08, and all 256 U7 rows had `obstacle_present=0`. That zero is expected because the policy did not reach the obstacle; it does not show that the pixel detector missed an obstacle. At least in this cell, the regression starts before the obstacle field and points to route-progress/control learning rather than obstacle perception as the first failure.

The reward source suggests a specific, testable mismatch. `shape_transition_reward()` gives `10 × max(0, tile_progress_delta)` and subtracts a speed-shortfall cost of `0.6 × shortfall` on every non-collision/non-off-track step, whether or not the car advanced. With reward scale 0.1, a clear straight at 43 m/s costs about 0.023 per step for the speed shortfall; one progress increment (1/251 of the route) earns about 0.004; the time penalty is 0.005 per step. The speed term can therefore favor accelerating even when a transition makes no route progress. This is a source-backed explanation consistent with the trace, not yet a causal result. Runtime `Agent.act()` uses the bounded PPO mean deterministically, so sampled TRAIN steering fluctuations are not evidence of submission-time action jitter.

Next single-variable RL test after the frozen pair finishes: hold reference KL at the existing control value 0.5, retain speed target 70 and progress weight 10, and change only `SPEED_SHORTFALL_PENALTY` from 0.6 to 0.0. Reuse the same TRAIN map/seed cells, shared post-BC initialization, and 8,192-decision budget, then run the registered TUNE evaluation. The prediction is more route progress and fewer stagnant fast rollouts while the per-decision time penalty and finish reward still favor quick completion. Reject this treatment if progress/finish rate does not improve or collisions/off-track increase; never promote it without a valid finish under 13 seconds and held-out evidence.

#### 2026-09-24 22:04 KST matched seed8105 KL U6 snapshot

The fourth arm is now actively training at U6/8; the global experiment remains 3/4 complete and still has not loaded TUNE. At matched seed8105/U6 across the same four TRAIN cells, KL=0.5 has mean progress 0.137, mean speed 39.86 m/s, 0 collision decisions, 1 off-track decision, mean gas 0.104, and reference KL 0.164. KL=2.0 has mean progress 0.166, mean speed 38.19 m/s, 0 collision decisions, 1 off-track decision, mean gas 0.096, and reference KL 0.063. Neither logged a finish. The higher anchor retains a small progress advantage while running slower; collision/off-track counts are tied. This is still an interim TRAIN comparison, not a winner or submission signal.

#### 2026-09-24 22:13 KST frozen KL runner stopped before its last two updates

Revalidation found no live process for the recorded KL runner (`49632`) or parent (`13460`), and no Python command line for this experiment. The artifacts still say global `running`, `completed_runs=3/4`; the seed8105/KL=2.0 arm is frozen at U6/8, 6,144 decisions, last updated 13:04:15Z. No TUNE, held-out, or official maps were loaded. The previous process check was an actual live-process poll, not just a stale task status.

Source inspection found no exact recovery path in the paired runner: it rejects a nonempty output directory and has no option to skip completed arms or resume the interrupted one. Individual policy checkpoints save model and optimizer state but no Python/NumPy/PyTorch RNG state. Therefore continuing from U6 would change the stochastic trajectory, while restarting the pair would repeat completed training and break the frozen-run provenance. The KL comparison is incomplete and must not be called a result or used to select a checkpoint. B has one blocker request to inspect the runner exit and report whether any valid recovery evidence exists; do not restart it. Continue with the separate speed-shortfall reward hypothesis only after this run is accounted for.

#### 2026-09-24 frozen KL 0.5 vs 2.0: partial TRAIN record and resume audit

The paired run was launched through `research_ops.cli` with the frozen protocol unchanged. Protocol file SHA-256 is `b5eef87df950340d00cbf71e86af8e2468ce2af6ad554b8202e19a44c2cb7c04`; embedded protocol SHA-256 is `7c594bc9ed00f014cd2b07e1d52c15b42b9c89fa54fdc1b53f6d4013e9725b51`; source fingerprint is `e34d16ef039b23fb6090e58a79ea0ce2f946fd838b3d4728e78dd3bab23778b6`. The fixed settings were seeds 8104/8105, four TRAIN cells, 8,192 decisions per arm, U8, LR 1e-4, speed target 70 m/s, and KL coefficient 0.5 vs 2.0. Runtime remained Python 3.11.15, Torch 2.1.0+cpu, NumPy 1.26.0. Training status records TRAIN only; TUNE, held-out and official access flags are false.

The RULES CLI returned `subprocess.TimeoutExpired` at its 3,600-second command limit. The active trainer PID 49632 was absent afterward. Three arms completed U8; seed8105/KL2.0 stopped at U6/6,144 decisions. The raw `training-status.json` still says `running` and the last run status still says `running`, because the process was terminated before it could write a failure status. No Tune evaluation ran. The experiment was not restarted.

A one-time resume audit found that `policy-step-6144.pt` contains model weights, Adam optimizer state and step 6144 (checkpoint SHA-256 `24a7845de9c7d67a7bdc53e6b21f7defce136749143aedc3a6d4cd71b5d1ce03`; optimizer state has 25 parameter entries at optimizer step 768). It does not contain Python/NumPy/Torch RNG state or environment state. `ppo-start.json` records the seed, not the RNG state. The frozen runner has no resume argument and rejects a non-empty output directory. Restoring the weights and optimizer alone would not reproduce the next two updates exactly. Decision: preserve the partial failure; no restart, continuation, or TUNE evaluation.

The U8 TRAIN rollout summaries below are per-update aggregates across the four TRAIN cells. “Collision cells” counts how many of those four 256-decision cell rollouts had at least one collision; it is not a count of independent maps. `|steer|` is mean absolute steering. None of these values are evaluation finish-rate or lap-time evidence.

| Seed | KL | Last update / decisions | Mean progress | Mean |steer| | Mean speed | Max speed | Collision decisions | Collision cells / 4 | Max damage | Measured reference KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8104 | 0.5 | U8 / 8,192 | .1957 | .2954 | 37.71 | 58.74 | 5 | 1 | 1.0 | .1755 |
| 8104 | 2.0 | U8 / 8,192 | .2413 | .3007 | 33.70 | 56.16 | 6 | 2 | .8 | .0510 |
| 8105 | 0.5 | U8 / 8,192 | .1533 | .2883 | 33.89 | 53.15 | 6 | 2 | 1.0 | .1280 |
| 8105 | 2.0 | U6 / 6,144 (partial) | .1656 | .3135 | 38.19 | 53.89 | 0 | 0 | 0.0 | .0627 |

For the only complete paired U8 seed, stronger KL reduced measured reference KL and increased mean progress by .0456, but mean speed fell by 4.02 m/s, mean absolute steering rose slightly, collision decisions rose from 5 to 6 and collision cells from 1/4 to 2/4; maximum damage fell from 1.0 to .8. This is mixed one-seed TRAIN evidence, not a stable improvement. At matched U6 for seed8105, KL2.0 had lower mean progress (.1656 vs .2093), higher speed (38.19 vs 32.33), similar mean absolute steering (.3135 vs .3154), zero vs three collision decisions and zero vs .4 maximum damage. The second seed has no U8 pair. No training cell finished; Tune finish rate, under-13 rate, lap median and P90 were not measured.

A read-only paired trace analysis covered seed8105’s `custom-track-haic-train-20260921:20260925` cell for U1–U6 in both arms, plus KL0.5 through U8. U1 progress matched at .582 before the PPO updates diverged. The per-cell U5 rollouts show faster progress mismatch: KL0.5 ended at progress .434 at 35.49 m/s, while KL2.0 ended at .347 at 38.01 m/s. At U6, KL0.5 ended at .247 at 40.47 m/s, while KL2.0 ended at .462 at 37.78 m/s. Thus higher speed did not consistently mean more route progress. At KL0.5 U7, this cell ended at .0637 after 223/256 decisions with zero progress delta; mean speed was 42.91 m/s, with one off-track decision, zero collisions, zero damage, and zero positive obstacle-present/urgency features. The map definition contains five obstacles, with its first obstacle at progress .2467, so this rollout stopped before reaching it. The zero feature count is not evidence of a missed obstacle detection in this segment.

The contemporaneous C diagnostic adds interpretation only: on the same repeated custom geometry, candidate’s first 120 decisions had mean speed 46.4 m/s, gas .118, mean |steer| .055 and progress .237 before off-track; SOTA had 36.7 m/s, gas .070, mean |steer| .093 and progress .458. These seed labels repeat one geometry, not independent maps. Candidate source provenance shows `visual_feature_encoder` was included in that run’s BC optimizer; weights deltas were not saved. The current runtime uses deterministic `policy_mean`; sampled TRAIN steering variation is not submission-time jitter evidence. Together these traces support separating route-control/policy preservation from obstacle response, but do not prove a cause. A deterministic evaluation trace would be needed to establish runtime behavior.

Reproducible partial record: `artifacts/haic/ppo-bc-reference-kl-0p5-vs-2p0-speed70-lr1e4-20260924T113750Z/train-partial-analysis-v1.json` (SHA-256 `8d425c22c6c92e47d5cf9fc427d37266cbdaa4e8a2ffd217efc24719548b1aee`). Raw update and decision-trace JSONL files remain in the same artifact’s `run/` tree. The RULES CLI synchronized `RESULTS.md` before execution; it still has 142 allowlisted records and no entry for this incomplete screen. `SOTA.md` remains on record `8b33233e373d`. Neither this partial run nor the analysis JSON was added as performance evidence.

#### 2026-09-24 22:38 KST preregistration: speed-shortfall reward ablation

The eligible SOTA submission package remains the actor-only record `8b33233e373d`, but its held-out median finished lap is 19.32 s and its P90 is 22.42 s; its official Track1 seed42 trace does not finish. It is a format/runtime fallback, not a competitive 13 s candidate. The RULES index also surfaced held-out record `45ed03140ff1` at 5/5 finishes and 19.26 s median, but collision/damage and valid-under-13 fields are absent and its restriction status is `unknown`; it is not eligible for promotion or submission as a verified replacement. The reference-KL PPO pair was killed by the control-plane's 3,600 s timeout and has no valid Tune result. Do not resume or compare its partial checkpoints.

Next test changes exactly one PPO reward coefficient: `SPEED_SHORTFALL_PENALTY` 0.6 (control) versus 0.0 (treatment). Hypothesis: charging a speed deficit on every safe decision, even with zero route progress, encourages fast stagnation; removing that term while retaining route-progress reward, time cost, collision cost, finish bonus, and target speed may improve route progress and safe completion. This is not yet established by a controlled evaluation.

Frozen controls: PPO actor-only runtime; same TRAIN-only teacher-BC initialization within each seed; seeds 8104 and 8105; same four registered TRAIN map/seed cells; 8,192 PPO decisions per arm (eight 1,024-decision updates); LR 1e-4; speed target 70 m/s; reference-KL coefficient 0.5 in both arms; same architecture, action scaling, optimizer reset, and map split. Only the speed-shortfall coefficient differs. Teacher targets and any privileged labels remain TRAIN-only. TUNE is loaded only after all four TRAIN arms and checkpoints pass integrity checks. TUNE labels repeat one geometry, so this is a screening test, not held-out evidence.

Report paired completion rate, valid finish under 13 s, finished-lap median/P90, DNF progress, collision decisions, off-track decisions, and damage for each matched cell. Reject if treatment worsens completion, per-cell collision/off-track or damage gates, or fails to improve valid-under-13 finishes / finished-lap median. A progress-only gain without a valid under-13 finish is diagnostic and cannot update SOTA. No held-out or official evaluation is permitted from a mere positive Tune screen; first inspect its raw traces and confirm a reproducible candidate. If training fails an integrity gate or times out, preserve partial artifacts and do not treat them as a policy result.

Control-plane subprocess timeout is now configurable while retaining the 3,600 s default. New CLI tests first failed on the missing option, then passed after implementation; the repo-local test suite passes 175 tests (1 skipped, 13 subtests, 5 existing warnings). Unrestricted `pytest -q` collection also encountered nested worktree tests under `artifacts/` and a Linux-only `fcntl` import; rerunning only the intended root `tests/` folder passed. The planned run timeout is 14,400 s to prevent the previous premature kill. The experiment output target is `artifacts/haic/ppo-speed-shortfall-0p6-vs-0p0-kl0p5-u8-20260924T133743Z/`.

The frozen protocol was generated before training: protocol SHA-256 `d68238afe3758bb8666ad3d6e39b32479b5fedb64a971e834f0b5183fb610a19`; source fingerprint SHA-256 `802842c09f75f996319597ac4c0343565c868315477653685681075d72480349`; TRAIN split manifest SHA-256 `49e1774b2ff039fb601903070e2976951c974a02f16fc2542754260574bea209`; wrapper source SHA-256 `9b4c142a789956befd0d8e6c53248af565e3ba5264c2d389664d0bc626ad06e9`. Source root is B's frozen experiment worktree at `7bdfcbdf17017d957413a9b159be3fbd3afc19a2`; its trainer hash is recorded separately from the main checkout. Both arms share a 0.5 reference-KL coefficient. No training had started at the time this protocol was recorded.

#### 2026-09-24 22:42 KST speed-shortfall pair launch

After the RULES preflight reported all required control-plane documents present and the existing 3.85 MB actor-only archive passed package restrictions, execution began through `research_ops.cli improve "성능 개선해 줘" --execute --timeout-seconds 14400 --command ...`. The command session is 38003; CLI parent PID 55324 and trainer PID 51124 were observed alive. Initial `training-status.json` is `running`, `completed_runs=0/4`, `shared_bc_states_prepared=0`, four TRAIN episodes loaded, and `tune_loaded=false`, `held_out_loaded=false`, `official_maps_loaded=false`. The process is preparing its shared TRAIN-only BC state; no PPO arm result or Tune result exists yet.

At 22:48 KST the same process completed the first seed's shared TRAIN-only BC initialization and entered PPO control `penalty_0p6` for seed8104. `training-status.json` now reports `shared_bc_states_prepared=1`, while the arm is at update 0; total completed arms remain 0/4. No Tune, held-out, or official maps have been loaded.

The control arm's first two 1,024-decision TRAIN rollouts are now available. U1: mean speed 40.50 m/s, gas mean .1026 with only 3/1,024 actions at or above 90% of the .24 limit, progress reward mean .00123, speed-shortfall cost mean .02128, zero collisions, and 2 off-track decisions. Among 535 clear-straight rows above 20 m/s, only 4.49% had positive progress; none approached the gas cap. U2: mean speed 37.94, gas .1071, progress reward .00171, shortfall cost .02211, 5 collision and 1 off-track decisions. On 355 clear-straight rows, positive progress rose to 11.83%, while no action approached the gas cap. These are intermediate same-arm training samples, not a between-arm or lap-evaluation result. They show a substantial mismatch between the per-decision speed-shortfall cost and mean progress reward in these rollouts; the registered coefficient-zero treatment directly tests whether removing that mismatch helps, while the U2 collisions warn that safety may worsen. Do not select an arm before the paired U8/TUNE comparison.

The five U2 collision decisions are one contiguous impact cluster in `custom-track-haic-obstacles-20260920:20260924`, not five independent obstacle encounters: damage rises from 0.2 to 1.0 over decisions 118–122 at nearly unchanged progress .374. The obstacle is already present with urgency 1 at decisions 116–117, when the actor steers right but applies gas (.012 then .125) and no brake; the first collision follows at decision 118. This is TRAIN rollout evidence that visible hazard did not reliably trigger timely braking/avoidance, with post-impact control continuing to accumulate damage. It motivates counting collision onsets separately from collision-marked frames in future diagnostics. It is not a lap-evaluation result and does not establish the treatment outcome.

Seed8104/control U3 reproduces the previous KL=0.5 U3 TRAIN summary to rounding: progress .1754 (~.175 before), speed 24.24, gas .1201, 5 collision-marked frames, 4 off-track frames, reference KL .3147 (~.315). The 5 collision frames are 3 onset clusters across 3 TRAIN cells; each onset has obstacle-present=1 and urgency=1 in its current pixel features. In one cluster the actor braked briefly, then reapplied gas before impact; in another it steered sharply while largely coasting; in the third it alternated steering and throttle. This corroborates a repeated, source-visible hazard-response instability rather than a general perception absence, but these are sampled training rollouts, not final policy behavior. The reward ablation remains in progress and must finish before selecting the next intervention.

#### 2026-09-24 23:02 KST speed-shortfall control U7 snapshot

The same live PPO run advanced seed8104/control (`SPEED_SHORTFALL_PENALTY=0.6`) to U7/8, 7,168 of 8,192 decisions. Its 1,024-decision TRAIN rollout summary is mean progress .2208, mean speed 33.77 m/s against a 70 m/s target, 2 collision-marked decisions, 3 off-track decisions, max damage .2, and zero finishes. Mean gas is .1029, max gas .2205 under the .24 cap, gas saturation 0%; brake is nonzero in 9.3% of decisions and mean brake is .0075. Thus actuator clipping is still not the explanation for low speed, and this checkpoint has not demonstrated completion. U6 immediately before it had progress .2093, speed 32.33, 3 collision decisions, 1 off-track decision, and zero finishes; these are separate sampled rollouts, so the small differences are not a causal trend.

The paired coefficient-zero arm has not started and global status remains `completed_runs=0/4`; no Tune, held-out, or official evaluation has run. This snapshot therefore cannot answer whether removing the speed-shortfall term helps. It does reinforce the diagnostic split: (1) the current control actor remains slow despite an uncapped gas action, consistent with the already measured dominance of the speed-shortfall cost over sparse progress reward; (2) collision and off-track behavior persist in TRAIN rollouts, consistent with inconsistent response to visible hazards and weak route-progress learning. Keep this as an interim observation only; wait for all four arms and the registered Tune screen before choosing the next RL change.

#### 2026-09-24 23:04 KST speed-shortfall control U8 complete; paired treatment begins

Seed8104/control (`SPEED_SHORTFALL_PENALTY=0.6`) completed its frozen 8,192-decision PPO budget. The final 1,024-decision TRAIN rollout has mean progress .1957, mean speed 37.71 m/s, 5 collision-marked decisions, 0 off-track decisions, and 0 finishes; mean gas is .1058 with zero saturation against the .24 cap. This nearly reproduces the earlier matched KL=0.5 U8 control summary (progress .196, speed 37.71, 5 collision decisions, 0 off-track), strengthening the repeatability of this slow, non-finishing control rollout on the shared seed/cells. It does not establish the reward-term cause because the learned rollout is stochastic and this is still one seed’s TRAIN slice.

The trainer has now started seed8104/treatment `SPEED_SHORTFALL_PENALTY=0.0` at U0; global `completed_runs=1/4`. No treatment updates, TUNE, held-out, or official evaluation are available yet. Continue the frozen run and compare the paired U8 outcomes before any follow-up intervention or SOTA change.

#### 2026-09-24 23:05 KST paired treatment U1 baseline alignment

The zero-shortfall-penalty treatment has completed its first 1,024-decision rollout for seed8104. Before its first PPO optimization, U1 matches the shared-control U1 to reported precision: mean progress .1548, mean speed 40.50 m/s, zero collision-marked decisions, 2 off-track decisions, no finish, and mean gas about .103 with no saturation. This is the expected shared-BC starting behavior; it is a pairing/protocol check, not evidence that the coefficient change failed. The coefficient can affect the policy only after reward/advantage calculation and PPO updates. Treatment is at U1/8; no TUNE evaluation is loaded. Judge the intervention from later updates and the paired U8/TUNE results, not this initial rollout.

#### 2026-09-24 23:08 KST speed-shortfall treatment U2: faster clear-road samples, but progress loss and out-of-bounds termination

The seed8104 coefficient-zero arm has now completed one PPO update and its U2 TRAIN rollout. Across the same four cells, U2 reports progress .1428, speed 40.71 m/s, 0 collision-marked decisions, 1 off-track decision, no finishes, and mean gas .107; the matched coefficient-.6 control reports progress .1616, speed 37.94, 5 collision-marked decisions, 1 off-track, no finishes, and gas .107. On clear-straight rows (no obstacle feature, curve feature <.1, speed >20), the treatment is 1.64 m/s faster (50.04 vs 48.40) but has fewer positive-progress transitions (7.0% vs 11.8%) and lower mean progress delta (.00029 vs .00048). Row counts differ (498 vs 355) because trajectories differ. This is one seed and one post-update rollout, so it is an interim tradeoff, not a selected result.

Episode-aware parsing of the obstacle-map cell corrected a misleading aggregate: the control's first episode crashes at decision 122, progress .374, speed 9.5, damage 1.0; the treatment's first episode ends at decision 244, progress .289, speed 56.8, with no collision, off-track flag, or damage. Its reason is recorded only as generic `terminated`. Source inspection resolves that terminal class: the training `SiteCustomCarRacing` inherits the vendored `CarRacing`; its base environment sets `terminated=true` when the car's absolute x or y exceeds `PLAYFIELD`, while `env_wrapper.py` assigns `retire_reason` only to crash/off-track. Therefore the treatment episode is an out-of-bounds failure, not a clean finish and not an obstacle collision. It exchanged the control's obstacle crash for a boundary exit before completing the same map segment.

In the treatment's last 10 frames before termination, route progress remains .2886, speed is 56.5–58.0 m/s, and actions continue with small alternating steering and intermittent gas; collision and off-track remain false. Road-center, curve, and obstacle engineered features are zero in these frames while the speed feature remains about .72. This suggests a possible loss of useful route cues, but does not prove perception failure: `pixel_features.road_centers()` falls back to center 42 when it finds no asphalt, making absent-road and perfectly centered observations share zero offsets/curve, and the raw frames/vehicle coordinates were not saved. Record raw x/y, explicit road-center detection confidence, and terminal cause in a future RL diagnostic so a bounds exit can be separated from a perception miss. Do not modify or restart this frozen pair.

Interim decision: the zero penalty is not a solution so far. It gives a small clear-road speed gain and more braking/fewer obstacle collision frames in the hazard window, but route progress falls and this episode exits the playfield at high speed. Keep the treatment unselected until all four arms and registered TUNE evaluation finish. If that tradeoff persists at U8/TUNE, follow with a smaller penalty coefficient (single variable) while preserving the same PPO, maps, seeds, and evaluation gates; raw out-of-bounds evidence should guide the next reward/observation change.

#### 2026-09-24 23:12 KST paired U8 seed8104: shortfall penalty trades progress/safety for speed

Both coefficient arms have now completed U8 for seed8104 (8,192 PPO decisions per arm over the same four TRAIN cells). The fixed-budget final TRAIN rollouts report:

| `SPEED_SHORTFALL_PENALTY` | Mean progress | Mean speed | Collision decisions | Off-track decisions | Max damage | Finishes | Gas saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | .1957 | 37.71 m/s | 5 | 0 | 1.0 | 0 | 0% |
| 0.0 | .2846 | 33.38 m/s | 2 | 1 | .4 | 0 | 0% |

Thus, on this matched seed, removing the penalty improved the TRAIN progress statistic by .0889 and reduced collision-marked decisions/max damage, while mean speed fell 4.33 m/s and one off-track termination appeared. Neither arm finished. This is a real tradeoff in the preregistered variable, but still one-seed TRAIN evidence, not Tune or submission evidence. It suggests the 0.6 speed term buys speed at the cost of risky route behavior, whereas 0.0 makes the policy safer/slower but still does not teach full completion.

Terminal traces show distinct failure modes. In the obstacle-map cell, control has a generic bounds termination at progress .232 and speed 57.3, while treatment reaches .358 before an `off_track` termination at 52.4. On train cell 20260921, control records a crash at .255 (damage 1.0) and a separate generic bounds termination at .239/speed54.4; treatment reaches .637 at collector cutoff with damage .4 but no terminal failure. On the other train cell, control exits generically at .287/speed58.7 while treatment reaches .426 at collector cutoff. These distinct cell outcomes explain why the aggregate treatment is safer/more progressive but slower; no single episode demonstrates a completed lap.

The intervention is a diagnostic partial improvement, not a fix. Wait for seed8105's matched pair and the frozen TUNE screen. If the TUNE tradeoff is reproduced without any valid under-13 finish, test a midpoint coefficient (0.3 vs 0.6, PPO only) as the next single-variable reward balance, with explicit out-of-bounds/off-track/collision gates. Do not promote the current checkpoint.

#### 2026-09-24 23:18 KST seed8105 control learning-curve audit through U3

The second seed's `penalty_0p6` control is at U3/8. Across its four fixed TRAIN cells, per-update summaries are non-monotonic:

| Update | Mean progress | Mean speed | Collision-marked decisions | Off-track decisions | Max damage | Reference KL | Value loss | Grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U1 | .2205 | 36.38 | 4 | 3 | .8 | .0178 | 18.24 | 7.37 |
| U2 | .1320 | 38.82 | 5 | 3 | 1.0 | .1037 | 18.13 | 14.70 |
| U3 | .1759 | 36.73 | 5 | 2 | 1.0 | .1288 | 14.61 | 27.93 |

No update has a finish; gas saturation remains zero. Cell-level U1 showed three off-track terminations on three cells and two collision onsets (four marked frames, max damage .8) on the fourth, obstacle-rich cell. U2 then regressed on route progress and added damage; U3 recovers some progress and reduces one off-track decision, but collision-marked decisions and max damage remain high. The rise and fall is consistent with the previously observed PPO rollout instability / policy forgetting, but sampled stochastic rollouts are not fixed-policy evaluation, so it cannot yet establish an actor regression by itself. The rising reference KL and logged gradient norm make update stability worth checking after this preregistered pair; do not change KL mid-run.

This makes the current diagnosis more specific: the actor's main obstacle/track failures are not only low motor speed. A small number of TRAIN starts produce out-of-bounds terminations on obstacle-free cells, while obstacle-rich cells produce collision clusters; PPO updates can increase measured speed as route progress worsens. The next comparison should continue to judge progress, speed, collision, damage, off-track and true finishes together, not optimize speed alone.

#### 2026-09-24 23:31 KST seed8105 control U4: apparent collision reduction is partly low hazard exposure

Seed8105/control U4 (4,096/8,192 decisions) reports mean progress .1656, mean speed 38.86 m/s, 2 collision-marked decisions, 1 off-track decision, max damage .2, and zero finishes. Cell trace analysis shows the lower collision count cannot be read as a safety gain by itself: on custom obstacle seed20260924, maximum progress is only .195, below that map's first obstacle at .2467, and no obstacle-present pixels occur; on train seed20260921, the rollout ends generically at progress .211. Only one of the two custom-obstacle cells reaches an obstacle cue at all (18 cue rows, one collision mark), while train seed20260925 has 25 cue rows, one collision mark, and an off-track termination. This pattern says limited route progress hides hazard exposure. Include exposed-cell count, obstacle cue/onset count, and collision onsets in follow-up summaries so an early departure is not mistaken for safe driving.

Across seed8105 control U1–U4, progress remains .2205 → .1320 → .1759 → .1656; speed 36.38 → 38.82 → 36.73 → 38.86 m/s; no finishes. Collision/off-track variation is confounded by how far each sampled rollout reaches. This reinforces that the policy's first bottleneck is reliable route tracking/coverage; the obstacle-response score is meaningful only after the car reaches the hazard. The frozen coefficient comparison remains unchanged and active.

#### 2026-09-24 23:34 KST TRAIN teacher/action timing audit: no one-step BC label shift

The completed read-only D audit checked the attempt-2 TRAIN-only teacher-BC collector and its traces. The collector calls `teacher.act(observation_t)`, stores that same observation with the resulting teacher action, then steps the environment; no one-decision shift was found in BC target pairing. In PPO traces, action and pixel features are from decision t, while collision/progress/damage fields are post-action outcomes; matching a collision to same-row hazard features can therefore be off by one.

Across the two PPO seeds' attempt-2 TRAIN traces, 46/47 collision-marked rows had an urgent detected obstacle at decision t. Of the 41 rows with a following decision, 37 still showed an urgent obstacle at t+1. Actor-action deltas near collision rows varied mainly in steering; mean brake deltas were 0.00–0.09 by cell. This weakens a broad detector-missed-obstacle or BC one-step-lag explanation and favors a learned response/track-control problem: the visual cue is usually present, but the actor often does not produce timely, effective braking/avoidance. Counts are marked decision rows rather than unique impacts, teacher demonstrations were deleted after warmup, and this audit does not prove a causal response failure or cover the current run's exact source/checkpoint. No code, training, Tune, or held-out evaluation was run.

This sharpens the current failure diagnosis into two coupled bottlenecks: before the obstacle field, PPO frequently departs the road or exits the playfield at roughly 48–55 m/s; when a visible urgent obstacle is reached, policy braking/avoidance is often too small or late. Therefore, a lower collision count is not a safety improvement unless progress and obstacle exposure are also reported. Keep the current frozen reward pair intact; use its eventual Tune result to decide whether the next single-variable RL test should prioritize the progress/track-control path or hazard response.

#### 2026-09-24 23:38 KST seed8105 control U8: failures cluster at the first hazard/turn sector

Seed8105/control (`SPEED_SHORTFALL_PENALTY=.6`) finished its 8,192-decision U8 budget. Final TRAIN summary: mean progress .1533, speed 33.89 m/s, 6 collision-marked decisions, 3 off-track decisions, max damage 1.0, and zero finishes. The per-cell terminal traces localize the repeated failure: obstacle-map cell 20260920 off-tracks at progress .289/speed49.2; obstacle-map cell 20260924 off-tracks at .256/speed52.6; train cell 20260921 off-tracks at .251 with speed0/damage.2 while obstacle and urgency features are both 1; train cell 20260925 crashes at .259/speed7.2 with damage1 and obstacle/urgency both 1. The obstacles map's first obstacle is at progress .2467, so three of four endings lie just after the first hazard sector, with the fourth at .289. This is stronger evidence for a repeated local hazard/curve bottleneck than a generic lap-end failure.

The two obstacle-map exits occurred at high speed without collision; the two train-map failures were collision/low-speed states. Combined with the earlier TRAIN audit where almost every collision row already had an urgent obstacle cue, the likely issue is not mainly that the detector fails to show the obstacle. The learned actor either leaves the road while approaching the hazard or does not convert an urgent cue into enough early braking/avoidance, then receives too little later-course experience. The current training still cannot prove whether the first obstacle itself or the adjacent bend causes each departure; raw vehicle coordinates/images and the cue/action sequence should be preserved in the next RL diagnostic.

Seed8105 control U8 is the third completed arm; the zero-penalty seed8105 arm has started (currently U1). The pair's Tune evaluation has not run. Do not select either checkpoint until that paired arm and the protocol-defined TUNE screen complete.

#### 2026-09-24 23:43 KST seed8105 coefficient-zero U2: slower contact becomes a prolonged stall

The seed8105 zero-penalty arm has reached U2. Its four-cell summary is progress .1485, speed 35.01 m/s, 6 collision-marked decisions, 2 off-track decisions, max damage1.0, no finishes, and gas saturation0%. Matched seed8105/control U2 was progress .1320, speed38.82, 5 collision decisions, 3 off-track decisions, damage1.0, no finishes. This first post-update comparison is mixed: treatment is slower and slightly more progressive with one fewer off-track decision, but has one more collision-marked frame; it does not yet reproduce seed8104's U8 progress/safety signal.

The matched obstacle-map cell shows two different failure trajectories. In control, urgent obstacle cues appear while the actor is still moving about 33–37 m/s; at progress .175 the actor applies gas .14–.18 with no brake for five consecutive collision-marked decisions, damage rises .2→1.0, and the episode crashes. In the zero-penalty treatment, urgency reaches 1 before contact; the actor briefly brakes (.14/.13) and slows from 42.9 to 39.4 m/s, but then reapplies gas with no brake under urgency1. A low-speed contact follows at progress .098/speed .8/damage .2. It then gives alternating steering and gas (.11–.17) while speed stays near zero and progress remains .098; after 100+ repeated obstacle-cue rows, it retires off-track. This is direct TRAIN evidence of both unsafe hazard approach and ineffective post-contact recovery despite visible cues.

The coefficient-zero change appears to trade some approach speed for a lower-speed stall in this seed, not a completed obstacle pass. The active PPO pair still has only one update in this treatment seed; these sampled TRAIN traces are diagnostic, not final checkpoint evaluation. The next learned-policy experiment should judge obstacle clearance plus continued route progress/recovery, not collision count alone. Keep the current pair frozen until its U8 and TUNE results finish.

#### 2026-09-24 23:41 KST paired U8/Tune: removing speed pressure is rejected

The frozen run `ppo-speed-shortfall-0p6-vs-0p0-kl0p5-u8-v1` completed all four U8 arms (seeds 8104/8105, 8,192 decisions per arm) and its registered Tune-only screen. Protocol SHA-256 is `d68238afe3758bb8666ad3d6e39b32479b5fedb64a9719e834f0b5183fb610a19`; source fingerprint is `802842c09f75f996319597ac4c0343565c868315477653685681075d72480349`. The exact checkpoint comparison and per-decision records are in `artifacts/haic/ppo-speed-shortfall-0p6-vs-0p0-kl0p5-u8-20260924T133743Z/run/evaluation-u8/summary.json` and `decision-traces.jsonl`.

U8 TRAIN rollouts were mixed by seed and no arm finished. At seed 8104, coefficient 0.0 increased mean progress `.1957→.2846`, reduced collision-marked decisions `5→2` and max damage `1.0→.4`, but lowered mean speed `37.71→33.38 m/s` and added one off-track decision. At seed 8105, 0.0 improved progress `.1533→.2562`, speed `33.89→35.12 m/s`, collision-marked decisions `6→2`, off-track decisions `3→0`, and max damage `1.0→.2`. These sampled TRAIN updates are not fixed-policy lap evaluations.

The Tune result rejects coefficient 0.0. The `.6` control finished 2/4 episodes at median `19.60 s`, mean speed `39.56 m/s`, with 2 DNF episodes, 4 collision episodes, and mean final damage `.8`. The `0.0` treatment finished 0/4, mean speed `38.84 m/s`, had 4 DNF episodes, 4 collision episodes, and mean final damage `1.0`. Neither arm had a valid finish under 13 seconds. Both `.6` finishes are repeats from training seed 8104; both seed-8105 policies crashed around progress `.259`. The two Tune seed labels replay the same geometry and identical trajectories per policy, so this screen contains two trained policies on one map, not four independent map trials. The evaluator marks the treatment `rejects_on_safety`, and Tune explicitly cannot promote a checkpoint. RESULTS DB rows are `b90a3b33e050` (`.6`) and `16f1d3830cbe` (`0.0`); `SOTA.md` stays unchanged.

The slow-lap cause is not a hard submission gas limit: competition input permits gas through `1.0`, while the current training action ceiling is `.24`; Tune clear-road, low-curve samples used mean gas `.128–.135`, max `.193–.200`, with no action saturation. The actor is choosing moderate throttle. The `.6` finished lap averaged `42.44 m/s`; the same route in 13 seconds would require roughly `64 m/s` mean speed, so the gap is large. Simply deleting speed-shortfall pressure did not increase clear-road gas or speed. The finish failure is also policy-seed sensitive: both coefficient arms trained with seed 8105 crash at the first hazard/turn sector near progress `.25`, while the zero arm at seed 8104 reaches `.757` before crashing. Earlier TRAIN traces show urgent obstacle cues before most collision rows but only small brake responses; this favors a learned control/recovery and route-coverage problem over a missing hazard detector. The Tune trace and earlier timing audit are observational evidence, not proof of the exact collision cause.

Next single-variable test: PPO `SPEED_SHORTFALL_PENALTY=.6` versus `1.2`, with the actor/action scale, reward terms, KL, maps, seeds, initialization and U8 budget held fixed. Higher clear-road pressure may increase throttle while the existing hazard-specific terms remain active; reject it if it gains speed while reducing finish/safety. Task A `01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1` is active on this registered follow-up; no training start is claimed yet. Only TRAIN and Tune may be used for this screen; held-out, official maps, packaging, and SOTA promotion remain closed until a Tune candidate demonstrates a safe gain.

#### 2026-09-25 00:02 KST follow-up execution audit

Task A (`01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1`) still reports `inProgress`, but the same turn has no emitted message/tool marker, no new `.6 vs 1.2` result directory or status file was found in the main workspace or registered HAIC worktrees, and no matching PPO process is running. Its worktree has no new files for this comparison. I sent one concise status/blocker request; I have not launched a duplicate paired run while Task A remains active. The coefficient comparison remains the next authorized RL step, and no performance claim is made from its active flag.

#### 2026-09-25 00:21 KST 후속 진단: 빠른 actor는 먼저 차선 추종에서 이탈

전략 C의 비승격 local post-selection 진단(`experiments/strategy-c/clean-start-train-only-teacher-bc-ppo-seeds-8104-8105-v1-attempt2/post-selection-diagnostic-v1`) 원시 결과를 요약했다. 이 평가는 held-out/official 점수나 새 SOTA 증거가 아니라 기존 사용 이력이 있는 7개 셀의 진단이다. Tune 선택 actor는 0/7 완주, 기존 SOTA는 5/7 완주였고 양쪽 모두 13초 미만 완주는 0회였다. 같은 custom geometry를 5개 환경 seed로 반복했을 때 SOTA는 5/5 완주, 각 19.32초, 충돌/손상 0이었다. 비교 actor는 같은 geometry에서 매회 progress .237, 최고속도 65.34 m/s에 도달한 뒤 충돌·손상 없이 `off_track`으로 종료했다. 종료 직전 obstacle 거리는 228 m 이상이어서 이 실패는 해당 장애물 회피로 설명되지 않는다. 더 높은 최고속도가 차선 유지와 완주를 보장하지 않는 사례다.

공식 Track 1 seed 42에서 SOTA는 progress .163에서 off-track, 손상 .4, 충돌 2회로 끝났고 비교 actor는 progress .700까지 갔지만 손상 1.0·충돌 5회의 crash로 끝났다. `official_plus_custom` 장애물 셀에서 SOTA는 progress .148, 손상 1.0·충돌 5회, 비교 actor는 progress .152, 손상 .4·충돌 2회로 둘 다 완주하지 못했다. 장애물 접근 trace에서 SOTA는 약 13 m 거리에서 속도 약 42 m/s였고 brake 0을 유지하다가 4 m 부근에서야 .08을 줬다. 비교 actor는 제동을 먼저 시작했지만 속도와 조향을 충분히 바꾸지 못하고 충돌/이탈했다. 따라서 느린 SOTA의 위험 구간 반응과 빠른 actor의 곡선 추종 실패를 별도 문제로 기록한다.

현재 증거가 지지하는 진단은 두 가지다. (1) 느린 정책은 gas가 학습 가능한 범위에 도달하지 않아 최대속도 제한 때문이라고 보기 어렵고, reward pressure `.6→0`도 Tune 속도를 올리지 못했다. (2) 더 빠른 비교 actor는 장애물 없는 구간에서 반복적으로 같은 곡선/경계 지점에서 이탈했으므로, 속도만 더 밀면 완주 문제가 악화될 수 있다. 비교 actor는 선택 후보 진단이며 clean SOTA 비교나 원인 단독 검증이 아니므로 이 결과만으로 일반화하지 않는다.

다음 discriminating step은 현재 실행 중인 `.6 vs 1.2` PPO speed-pressure pair를 끝까지 유지하는 것이다. 2026-09-25 00:21 KST에 root에서 PID 22804와 protocol hash `b035915ca6be04a7fcf8cb96c22785e5fce65abeb187baa573ba13bef846e501`을 확인했다. 진행 상태는 0/4 arms, TRAIN 4개 episode만 로드, Tune/held-out/official maps 미로딩이다. 중복 학습을 시작하지 않는다.

후속 RL 가설(아직 실행 안 함): speed-pressure 비교가 속도만 올리고 off-track/collision을 악화시키거나 완주를 늘리지 못하면, `ROAD_TRACKING_LATERAL_WEIGHT`만 .5에서 1.0으로 바꾸고 heading weight .5, speed reward, actor 초기값, PPO, maps/seeds, U8 budget을 고정한 paired PPO를 검토한다. TRAIN에서만 simulator lateral label로 dense reward를 강화하고 제출 actor는 pixels-only로 유지한다. 동일 곡선에서 차선 이탈/충돌이 줄고 completion이 나빠지지 않는지가 판정 기준이며, lap time·평균속도·valid-under-13도 함께 기록한다. 먼저 기존 기록에서 중복을 확인하고 `.6 vs 1.2`의 결과 후에만 시작한다.

#### 2026-09-25 00:25 KST `.6 vs 1.2` paired run: control U6 snapshot

The process-backed run remains live (PID 22804; same protocol/source hashes). Only the `.6` control arm, seed 8104, has started; it reached U6/8 (6,144 decisions) across the four registered TRAIN cells. The `1.2` arm and second seed have not started, global completed arms remains 0/4, and no Tune/held-out/official maps are loaded. This is interim control-only evidence, not a treatment comparison.

Across control U1–U6, no rollout finished. Mean speed varied 40.50, 37.94, 24.24, 32.41, 36.37, 32.33 m/s; mean progress ranged .155–.210. Collision-marked decisions were 0, 5, 5, 3, 0, 3; off-track decisions 2, 1, 4, 1, 1, 1; max damage reached 1.0 at U2. Mean gas stayed .096–.120 and maximum gas .219–.227 under the .24 ceiling, with zero saturation in every update. U5's zero-collision sample also had only .210 mean progress, illustrating that low collision counts can mean the policy did not reach later hazards. Treat update-to-update changes as stochastic rollout variation; they do not establish a training trend. Wait for the matched 1.2 arm, all U8 checkpoints, and the registered Tune screen before deciding.

#### 2026-09-25 00:31 KST `.6 vs 1.2` 첫 seed paired U4 interim

The `.6` arm (seed 8104) is complete at U8; the matched `1.2` arm is live at U4/8. The exact shared post-BC actor SHA and U1 rollout match, confirming paired initialization. Through U4 both arms have zero finishes. Treatment mean progress/speed by U1–U4 is `.155/.191/.162/.218` and `40.50/38.63/32.40/36.94 m/s`; control is `.155/.162/.175/.172` and `40.50/37.94/24.24/32.41 m/s`. Treatment has 10 collision-marked and 8 off-track decisions to date; control has 13 and 8. Max damage reaches 1.0 in both. This is mixed interim evidence: treatment progress/speed are better at U2/U4, but treatment U3 damage rises to 1.0 and collision count exceeds control. At U4 mean gas is .110 for treatment vs .109 control, and neither arm saturates the .24 ceiling. Do not infer a speed-reward effect from one seed/four updates; wait for U8 and seed 8105, then compare the fixed Tune screen.

#### 2026-09-25 00:33 KST `.6 vs 1.2` seed 8104 U5: throttle rises while rollout quality falls

The live `penalty_1p2` arm reached U5/8 for seed 8104 (PID 22804 remains active; 1/4 arms complete, no Tune). Compared with the matched `.6` control's U5, treatment gas mean rose `.096→.118` (gas max `.221` vs `.222`) but mean speed fell `36.37→28.34 m/s` and mean progress fell `.210→.161`; collision-marked decisions were `0→2`, off-track `1→2`, and max damage `0→.2`. Neither rollout finished. This single update is noisy and cannot decide the pair, but it shows the stronger speed penalty can increase throttle without increasing realized speed when the sampled trajectory leaves the safe route or hits obstacles. Continue to U8 and the second seed; do not compensate by raising gas again before testing route-control learning.

#### 2026-09-25 00:35 KST seed8104 U8 and a reward-conflict hypothesis

The live `.6 vs 1.2` run has completed both seed-8104 arms (2/4); the process remains active, and Tune has not loaded. U8 control `.6`: mean progress `.195684`, mean speed `37.7109 m/s`, mean gas `.10575`, 5 collision-marked decisions, 0 off-track, max damage 1.0, no finishes. Treatment `1.2`: progress `.220308`, speed `32.3274 m/s`, gas `.10159`, 8 collision-marked decisions, 1 off-track, max damage 1.0, no finishes. On this seed the stronger speed penalty raises progress slightly but reduces realized speed and safety; it does not improve completion. Seed 8105 and the registered Tune screen remain necessary before rejecting/promoting the coefficient.

A read-only replay of U5 treatment traces exposes a more specific training-reward conflict. In two TRAIN cells, after contact the car stays at fixed progress (`.301` or `.251`) for many decisions with urgency 1.0, damage `.2`, no new collision flag, brake 0, and gas roughly `.14–.22`; steering changes sign repeatedly. The current PPO reward gives an unconditional low-speed recovery bonus when damage is at least `.15`: `5 * gas_fraction + 4 * |steer| * gas_fraction`. It does not check whether the visible hazard remains in the lane or whether progress resumes. At the same time, hazard gas costs `8 * risk * gas_fraction`. Replaying the saved pixel features through `visible_hazard_risk()` gives risk `.93–.98`; at decision 121, gas `.19` and `|steer|=.68` produce about `6.03` hazard-gas cost versus `6.06` recovery bonus. This nearly cancels the explicit gas suppression exactly while the actor is stalled; the absolute-steer bonus also does not distinguish a useful escape direction from oscillation. This is a plausible reward conflict backed by source arithmetic and TRAIN traces, not yet a causal result.

Next PPO-only experiment after the active coefficient pair finishes: compare the current recovery reward with a treatment that applies both recovery bonuses only when `visible_hazard_risk < 0.5`. Keep speed penalty fixed to the better-supported coefficient, same clean initializer/optimizer, seeds, four TRAIN cells, U8 budget, and all other reward terms. Log the high-risk damaged-stall state counts, gas/brake/steer, progress resumption and time-to-escape, plus finish/collision/damage/lap metrics. Tune-only selection follows all four arms. The gate affects TRAIN reward only; the submitted actor remains pixel-only PPO with no action-time rule. Reject the hypothesis if stalls persist, recovery elsewhere regresses, or completion/safety fails to improve. This is now the next hazard-focused candidate; the lateral-weight experiment remains a later alternative for clear-road high-speed off-track failures.

#### 2026-09-25 00:41 KST second-seed control reproduces exactly

Seed 8105 control `.6` U1 and U2 checkpoint SHA-256 values match the earlier `.6 vs 0.0` run byte-for-byte, as do shared post-BC actor SHA, mean progress, speed, collision/off-track counts, damage, finish count, and mean gas. Current `.6 vs 1.2` seed 8105 is at U2; the treatment has not started for this seed. This confirms deterministic replay of both control seeds under the frozen source/protocol and makes later paired deltas easier to interpret. It is not evidence that either control completes: U1/U2 had no finishes, and U2 recorded five collision-marked decisions, three off-track decisions, and max damage 1.0.

#### 2026-09-25 00:44 KST seed 8105 replay parity through U4

Extended the deterministic comparison: current `.6` seed-8105 control matches the prior `.6` arm exactly through U4—each actor checkpoint SHA and full progress/speed/collision/off-track/damage/finish metrics match. U4 remains non-finishing (progress `.1656`, speed `38.86 m/s`, 2 collision-marked decisions, 1 off-track, max damage `.2`). The active run has completed 2/4 arms; seed-8105 control is at U4 and has no paired 1.2 treatment yet.

#### 2026-09-25 00:46 KST second `.6` control U8 complete

Seed 8105 control `.6` completed U8. The full sequence of eight actor checkpoint hashes and reported rollout/action metrics matches the earlier `.6 vs 0.0` run exactly, extending the deterministic replay parity from U4 through U8. Final control U8: mean progress `.153340`, mean speed `33.8865 m/s`, mean gas `.10674` (max `.22104`, zero throttle saturation), 6 collision-marked decisions, 3 off-track decisions, max damage 1.0, and no finishes. Together with seed 8104 control, both registered control seeds reproduce but neither produces a finish. Global status is 3/4 arms; only `penalty_1p2`, seed 8105 remains, and no Tune/held-out/official evaluation has loaded.

#### 2026-09-25 00:49 KST seed 8105 `1.2` U3 interim: mixed updates, no finishes

In the final paired seed, U2 treatment (`1.2`) was safer than control (`.6`): progress `.166` vs `.132`, speed `37.87` vs `38.82 m/s`, 0 vs 5 collision-marked decisions, 1 vs 3 off-track, and max damage 0 vs 1.0. At U3 it reversed: treatment gas rose `.100→.117`, but progress fell `.176→.158`, speed `36.73→28.87 m/s`, collision-marked decisions rose 5→8, off-track 2→4, with max damage 1.0 in both. No finishes occurred. The U3 gradient norms were high in both arms (27.93 control, 29.03 treatment), while approximate KLs were .0226 and .0291; this does not isolate a treatment-specific optimizer instability. This is still one sampled TRAIN update per checkpoint, so keep the pair intact until U8. Global run remains 3/4, Tune not loaded.

#### 2026-09-25 00:53 KST `.6 vs 1.2` second-seed treatment U6: still no finish

The active paired PPO run remains at 3/4 arms: the seed-8105 `penalty_1p2` treatment has completed U6/8 (6,144/8,192 decisions); the other three arms are complete. Its U6 TRAIN rollout over the same four registered cells has mean progress `.1163`, mean speed `34.97 m/s`, 1 collision-marked decision, 3 off-track decisions, max damage `.2`, and zero finishes. The preceding U5 rollout had progress `.1752`, speed `30.60 m/s`, 8 collision-marked decisions, 1 off-track decision, max damage `1.0`, also with no finish. These two variable rollouts are noisy and non-monotonic; neither establishes a treatment benefit. PID 22804 is still live, source/protocol fingerprints remain frozen, and no Tune, held-out, or official map has been loaded. Continue the registered U7–U8 arm and Tune screen; do not promote this checkpoint.

#### 2026-09-25 01:16 KST: completed `.6 vs 1.2` Tune screen; recovery-stall evidence

The frozen run `ppo-speed-shortfall-0p6-vs-1p2-kl0p5-u8-v1` completed all four PPO arms at U8/8,192 decisions and the registered Tune screen (4 checkpoints, 8 episode rows, 1,170 decision rows). The two Tune seed labels replay one geometry, so this is one-map selection evidence only. At penalty `.6`, two repeated Tune episodes finish at 19.60 s (training seed 8104); the other two seed-8105 episodes crash at progress `.259`. Summary: 2/4 finishes, 0/4 under 13 s, mean speed 39.56 m/s, 4/4 collision episodes, mean final damage .8. At penalty `1.2`, 0/4 finish and 0/4 are under 13 s; mean speed is 27.89 m/s, all four episodes have collision marks, two terminate off-track, and mean final damage is .6. The comparator rejects `1.2` on safety. Keep the `.6` coefficient; do not promote either arm. The raw summary, episodes, and traces are under `artifacts/haic/ppo-speed-shortfall-0p6-vs-1p2-kl0p5-u8-20260924T151348Z/run/evaluation-u8/`.

A failure trace isolates why the policy gets slow after the first collision. In the seed-8105/penalty-1.2 Tune episode, the actor sees a centered obstacle at decision 62 (progress `.222`, speed 51.1 m/s, urgency .1), brakes through decisions 62–66, and still collides at decision 67 (progress `.251`, speed .5 m/s, damage .2). Over the next 100 decisions, all rows remain at progress `.251029`, visible hazard urgency stays 1, mean speed is `.0296 m/s`, mean gas `.1385`, mean brake 0, and mean steering `-.1379`; it then exits off-track at decision 168. This supports two distinct failure modes: obstacle avoidance remains inadequate despite braking, and the post-impact policy feeds throttle while pinned. The latter explains a large part of the low episode mean speed; the throttle actuator itself is not saturated (TRAIN gas max about `.22` under `.24`, saturation 0%). A 19.60 s Tune finish at 42.45 m/s would need roughly 64 m/s mean on that same route to reach 13 s.

The PPO collector quota is 256 decisions per TRAIN cell/update; all final rows have `rollout_truncated=true`. Therefore training never observes a complete lap or its terminal finish bonus and repeatedly samples the start sectors. This is an additional training-coverage limitation, not a proven sole cause of the Tune failures.

A single-variable TRAIN-only fix is now in `training/train_policy.py`: retain the damaged-low-speed recovery bonuses only when pixel-derived `visible_hazard_risk < 0.5`. The actor observation, PPO runtime, action transform, speed penalty `.6`, and submitted actor-only design are unchanged. Added `test_recovery_bonus_is_disabled_while_visible_hazard_risk_is_high`; it failed before the fix on the expected excess bonus and passes after it. The full suite passes 161 tests (1 skipped because the optional server checkout is absent). Task A `01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1` has the next paired PPO protocol: recovery-risk threshold `2.0` (old always-on behavior) versus `.5`, same seeds/maps/U8 budget, then Tune only. It must report high-risk stall count and time-to-escape as well as completion, under-13, collisions, damage, speed, and lap time. The reward fix remains unproven until those raw results arrive.

#### 2026-09-25 01:46 KST paired PPO recovery-risk gate launch

The previously assigned task ended without a trainer process or output, so the registered experiment was resumed inline after confirming no duplicate run was active. The isolated `paired-speed-screen` source worktree supplies the exact PPO/BC/KL runner used for the preceding speed ablation; its clean-start, rollout, imitation, PPO, environment, map, and actor modules differ from the current root checkout, so the experiment uses that matched source snapshot. The only policy-reward variable is `RECOVERY_MAX_HAZARD_RISK`: control threshold 2.0 (equivalent to always-on because risk is bounded by 1) versus treatment threshold 0.5. Speed-shortfall penalty remains 0.6. Seeds 8104/8105, the same four TRAIN cells, shared TRAIN-only BC initialization per seed, fresh Adam, reference KL 0.5, and U8/8,192 decisions per arm are frozen.

Protocol SHA-256: `4f5e6e92f36ae1e7eba11f79dcfc297dd6241978006be3bff844eb1ea3f18da2`; source fingerprint: `5fa375fb4a5c452c26f3a877e794c7cf3ed2505bb7489b1d0f4b0db70c2a3623`. Output: `artifacts/haic/ppo-recovery-gate-risk2p0-vs0p5-kl0p5-u8-20260925T014411KST/`. At launch, PID 26056 reported status `running`, 0/4 arms and 0 shared BC states prepared; only the four TRAIN episodes were loaded. TUNE, held-out, and official map flags were false. Evaluation is configured to load TUNE only after all arms pass integrity checks; there is no held-out/official evaluation or SOTA/submission update in this run.

The first `risk_threshold_2p0`, seed-8104 control arm has reached U2/8 (2,048 decisions). Its U1 rollout matches the previously completed speed-shortfall control to reported precision: progress .15485, speed 40.50 m/s, gas mean .10258/max .22050 under the .24 limit, and zero gas saturation. U2 also matches that prior control: progress .16157, speed 37.94 m/s, 5 collision-marked decisions, 1 off-track decision, max damage 1.0, and gas saturation 0%. This replay parity validates the paired start and confirms the risk threshold 2.0 reproduces the old reward behavior. It is control-only TRAIN evidence; the `.5` arm has not started, no finish-rate or lap-time comparison exists, and no TUNE/held-out/official map is loaded.

The recovery-stall postprocessor was checked against the completed `.6 vs 1.2` raw Tune traces before relying on it. It counts distinct high-risk damaged low-speed segments and collision onsets separately from collision-marked frames. On those traces it found 4/4 high-risk stall segments and 208 qualifying decisions for the `1.2` arm versus 2 segments/4 decisions for `.6`; none had a full-tile progress escape while still in the qualifying state. This confirms the raw trace pattern, though a post-hazard progress resumption that occurs after risk drops is conservatively censored; final reporting should pair this metric with the raw progress trace rather than call every censored segment a permanent stall.

#### 2026-09-25 02:08 KST seed8104 recovery-gate interim: confirmed stall state and a second speed bottleneck

The matched `risk_threshold_2p0` control finished its 8-update/8,192-decision TRAIN budget. Across the 8,192 sampled decisions it averaged 34.41 m/s and gas 0.1062 under the 0.24 actuator limit (44% of the limit; zero saturation). The trace contains 23 collision-marked rows and 13 off-track rows, but no finish rows; these are 256-decision TRAIN fragments, so that count is not a full-lap score. The control arm remains replay-consistent with the earlier `.6` speed-shortfall run through U2; its final Tune comparison has not run yet.

Pixel-feature replay exposes two distinct causes of low speed. First, 606 decisions satisfy damage >= .15, no current collision, speed <= 8 m/s, and lane hazard risk >= .5 (current speed is a proxy for the reward's previous-speed check because the trace omits previous speed). They form six 100–103-decision contiguous stall segments; summed progress across those segments is only .0082, while mean gas is .149/0.24. This is direct evidence that the always-on damaged recovery bonus repeatedly rewards gas/absolute steering during a pinned, visible-hazard state. Second, on 1,300 clean, obstacle-free, non-damaged high-curve rows (`road_curve_magnitude >= .4`), mean speed is 32.48 m/s and mean gas only .0368; on 3,454 clear low-curve rows, speed is 43.85 m/s and gas .1268. The policy therefore under-throttles even before contact on clear road, especially in curves. This is not explained by hitting the gas cap. It suggests the recovery reward explains prolonged near-zero speed after impact, while the learned speed/curve preference is a separate cause of slow clean driving.

The matched `risk_threshold_0p5` treatment is at U4/8; the control is complete. Actor checkpoints are identical through U2 and first differ after U3, when high-risk recovery transitions enter the PPO update. In the first post-divergence rollout (U4), treatment mean speed is 29.62 m/s vs control 32.41; it still has 101 high-risk damaged low-speed rows vs 100 in control, with mean gas .16 in both. One update is too little to reject the gate, but there is no immediate stall escape signal. Continue the frozen pair through all U8 arms and registered Tune-only screen; do not change coefficients, evaluate held-out/official maps, package, or promote during this run.

#### 2026-09-25 02:15 KST first matched seed: recovery-risk gate does not reduce pinned-state behavior

Seed 8104 now has both matched U8 arms complete. The control (`risk_threshold_2p0`) averaged 34.41 m/s, gas .1062, 23 collision-marked decisions, and 13 off-track decisions; the `.5` gate treatment averaged 32.43 m/s, gas .1072, 22 collision-marked decisions, and 17 off-track decisions. Both have zero `finished` trace rows, but all rows are 256-decision TRAIN fragments, not full-episode evaluations. Using the same pixel-risk reconstruction/current-speed proxy, high-risk damaged near-stall samples increased from 606 control rows in six long segments to 906 treatment rows in nine; mean gas in those samples remained .149 vs .154. The treatment also retains repeated steering-sign reversals inside zero-progress segments. Thus the risk-only gate has not produced an escape signal in this seed and presently looks worse on speed, off-track marks, and pinned-state exposure. One matched seed is insufficient for rejection; seed 8105 and the registered Tune screen remain required.

This falsifies the simple version of the idea that removing all high-risk recovery bonuses will itself teach the actor to escape. It does not yet establish whether the control's positive absolute-steering bonus is helpful or whether either version can recover. If both seeds/Tune reject the gate, the next isolated RL reward test should compare the existing recovery bonus with the same bonus conditioned on positive `new_road` progress. That preserves reinforcement when an escape advances the car while removing payment for long stationary gas/steer actions; it changes reward eligibility only and keeps actor-only PPO, initialization, KL, maps, seeds, and budget fixed. The separate clear-road curve-speed deficit remains for a later single-variable test; scalar speed-penalty values 0.0 and 1.2 are already rejected, so do not repeat them.

#### 2026-09-25 02:23 KST seed8105 control replay parity through U3

The second-seed `risk_threshold_2p0` control has reached U3/8 and remains byte-for-byte replay-identical to the prior `.6` speed-shortfall control on all three completed updates: actor hashes, shared post-BC hash, mean speed, and mean progress match. This confirms the active recovery-gate pair's control path is a valid continuation of the previously measured baseline for seed 8105. The `.5` treatment for seed 8105 has not started; global status remains 2/4 arms complete, with TUNE/held-out/official still unloaded. No candidate decision follows from this checkpoint.

#### 2026-09-25 02:28 KST seed8105 control reproduces the obstacle-pinned termination

The active `risk_threshold_2p0` control reached U7/8 and remains exactly replay-identical to the prior `.6` speed-shortfall control through U6 (actor hashes and reported rollout metrics match update by update). In U5, the obstacle-map cell `custom-track-haic-obstacles-20260920:20260920` produces a concrete failure sequence: 108 obstacle-visible rows; it reaches progress `.247967`, then spends decisions 82–182 in a 101-decision damaged, low-speed, high-risk stall with mean gas `.167` and zero progress gain. Decision 182 terminates as `off_track` at speed below .001 m/s. The collector then resets that same map/seed cell to fill its 256-decision quota, explaining the later progress reset in the trace. This confirms the repeated failure mechanism at the first obstacle sector across both training seeds; it is not a lap-end timeout. This remains TRAIN evidence from a sampled rollout, not an independent full-lap score. The `.5` seed-8105 treatment has not started, and TUNE remains unloaded.

#### 2026-09-25 02:29 KST new seed8105 trace refines the first-obstacle cause and reprioritizes the next PPO test

Decision-level TRAIN trace for the seed-8105 control shows the approach failure before the post-impact stall. At decision 75 the engineered obstacle cue first appears in the saved feature stream near its top detection row (urgency `.06`, speed `38.9 m/s`, progress `.224`). Lane risk rises to `.15` at d76 and `.32` at d77; the actor applies no brake at d76–79 while steering changes sign. At d80 risk reaches `.54`, speed is `38.3 m/s`, and brake is only `.037`; collision follows at d81 at progress `.248`/speed `4.1 m/s`, with damage `.2`. This first cell completes the same 101-decision high-risk stall and off-track termination described above. The trace records features, not raw pixels, so it does not prove the full CNN lacked an earlier visual cue; it does show that the explicit risk reward becomes substantial only one decision before contact under the current `urgency^2 * lane_overlap` formula.

This evidence prioritizes the next hazard-approach PPO hypothesis after the frozen recovery-gate pair: vary only the urgency exponent in `visible_hazard_risk`, squared (`u^2`) versus linear (`u`), while keeping the actor inputs, recovery threshold, speed reward, initialization, KL, maps, seeds, and decision budget fixed. Linear risk increases early braking/gas-shaping signal without adding an action-time rule. Judge first-brake decision and collision speed alongside exposed-cell count, safe progress past the obstacle, completions, under-13 finishes, collisions, damage, and lap time. The prior planned `new_road`-conditioned recovery reward remains a fallback if the risk-gate pair shows post-impact stall is still the dominant remaining failure. No new training run was started; finish the frozen four-arm run and its Tune screen first.

#### 2026-09-25 02:34 KST seed8105 gate treatment is mixed and trades collisions for a longer stall

The `.5` seed-8105 treatment has reached U3/8. U1 TRAIN traces were identical between arms, including the same 28 high-risk damaged near-stall decisions, but the PPO checkpoints diverged after those reward differences were consumed; both arms share the same post-BC actor and RNG reset seed. At U2, treatment recorded zero collision rows versus five in control, but its obstacle-map seed-20260924 cell never showed an obstacle cue (max progress `.1585`), so that apparent collision reduction is partly exposure-limited; the other cells still reached visible hazards. At U3 the treatment's four cells included one 101-row damaged high-risk stall on train seed-20260925 (progress `.239`, 109 visible-hazard rows, one collision, one off-track, mean speed 19.0 m/s). The matched control's same cell reached `.474` progress with five collision rows and mean speed 32.8 m/s, but had no long qualifying stall. This is a mixed response, not a treatment win: suppressing recovery reward can reduce collision-marked frames while leaving the car pinned and sharply reducing route coverage. Preserve exposed-cell counts when interpreting all future collision comparisons.

#### 2026-09-25 02:37 KST seed8105 treatment U2–U5: apparent gains fail under exposure-matched reading

The seed-8105 `.5` arm advanced to U5/8. U2 looked safer (0 collision-marked rows vs 5 control), but had fewer obstacle cues overall (10 vs 59) and the custom-obstacle seed-20260924 cell had no cue at all, so it was partly an exposure difference. In U3 the treatment encountered more hazard cues (143 vs 74) and progressed farther on average (`.196` vs `.176`), yet one train cell entered a 101-row pinned stall; mean speed was 33.90 vs 36.73 m/s. U4 had matched aggregate hazard-cue counts (45 each), while treatment had 5 collision rows/1.0 max damage vs 2/.2 in control and lower mean progress (.143 vs .166). U5 treatment again had more exposure (218 vs169 hazard rows), higher mean progress (.209 vs .191), but 6 vs1 collision rows, max damage .8 vs .2, and 153 vs101 high-risk stalled rows; its mean speed was 29.98 vs32.71 m/s. These sampled rollout comparisons are noisy and use one seed per paired update; they do not replace the fixed Tune evaluation. So far the gate has not improved obstacle-crossing behavior and may be suppressing useful escape learning. Finish U6–U8 and the registered Tune screen before rejecting it.
#### 2026-09-25 03:10 KST recovery-risk gate final paired screen

The registered U8 PPO pair completed on the same frozen source, TRAIN data, seeds 8104/8105, four TRAIN cells, and shared initialization protocol. Fixed TUNE-only screen: both arms finished 2/4 episodes and had 0/4 valid finishes under 13 seconds. The recovery gate (risk threshold 0.5) produced a 19.46 s median finished lap versus 19.60 s for the always-on control; collision-marked decisions fell from 16 to 8, mean final damage from 0.8 to 0.4, and mean DNF progress rose from 0.259 to 0.374. Collision episodes remained 4/4 in both arms. Postprocessor counted 200 high-risk stall decisions in 2 segments for the gate versus 4 decisions in 2 segments for control, with no recovered segments in either arm. The gate is a useful safety/progress signal, but it did not increase completion and is not a submission candidate. The two TUNE seed labels replay one geometry, so these are two trained policies, not four independent maps. SOTA.md remains unchanged at the validated 19.32 s held-out record.

The next single-variable PPO screen changes only the pixel-derived visible-hazard urgency exponent from squared (u^2) to linear (u), keeping the recovery threshold at 0.5 in both arms, reference KL 0.5, shared TRAIN-only BC initialization, seeds/maps, and U8 budget fixed. The paired TRAIN trace showed that the saved feature stream first marks the obstacle at progress about .224 with urgency .06 and no brake; risk reaches .54 only near progress .248, where contact follows. Linear scaling preserves obstacle presence and lane-overlap factors while making this early urgency cue shape risk sooner. Evaluate the first brake response and collision speed as diagnostics, then TUNE completion, under-13, collisions, damage, speed, and lap time. A TUNE-only result cannot promote SOTA. No held-out/official data or submission archive is part of this screen.

#### 2026-09-25 03:14 KST urgency-exponent PPO run resumed after launcher fixes

The first launch attempt used script-file invocation, which puts the research subdirectory rather than the repository root at the front of Python's import path; it stopped before creating training state. Running as a package then exposed a protocol-integrity mismatch: the runner passed the hash of the serialized JSON file, while the frozen PPO runner checks the canonical digest stored in the protocol record. The two digests were compared directly and the stored canonical digest matched the runner's canonical recomputation. research/paired_hazard_urgency.py now passes frozen["protocol_sha256"]; the failed attempt is retained for audit and was not reused.

A fresh unique run is active at artifacts/haic/ppo-hazard-urgency-square-vs-linear-kl0p5-u8-20260925T031409KST/. Its status file reports experiment ppo-hazard-urgency-square-vs-linear-recovery-gate0p5-kl0p5-u8-v1, protocol SHA-256 c37b2fab4ed7f22d740748a707d4e9407cc2c512dfc82cebb4879de19f5c396d, 0/4 completed arms, four TRAIN cells loaded, and TUNE/held-out/official maps not loaded. Windows launcher PID was 15704; training-status reports Python PID 41712. Continue monitoring this registered run without starting a duplicate. The fixed TUNE screen remains gated on all four arms and manifest integrity.

#### 2026-09-25 03:17 KST live-run heartbeat

Verified the registered urgency-exponent process is still alive (Python PID 41712). CPU time increased by 14.92 seconds during a 15-second wall-clock sample, so it is actively computing rather than stalled. It is still in shared TRAIN-only initialization: 0/4 PPO arms complete, zero shared BC states finalized, four TRAIN cells loaded, and no TUNE/held-out/official maps loaded. This is a verified wait; no duplicate run was started.

#### 2026-09-25 03:19 KST urgency-exponent run made initialization progress

A fresh poll confirms the same Python training PID 41712 is live. The run advanced from 0 to 1 of 2 shared TRAIN-only BC states prepared; PPO arms remain 0/4. Four TRAIN cells remain the only loaded episodes, with TUNE/held-out/official flags false. No outcome is available yet, so no candidate decision is made and no duplicate was started.

#### 2026-09-25 03:21 KST squared-risk control parity through U2

The current urgency-squared control for seed 8104 has completed U2/8. Its actor checkpoint hashes and rollout metrics match the previous recovery-risk-gate 0.5 arm exactly at U1 and U2: mean speed 40.50 then 37.94 m/s, mean progress 0.15485 then 0.16157, with 0 then 5 collision-marked decisions. This confirms the current control reproduces the earlier PPO path; the linear-urgency treatment has not started, so there is still no treatment comparison or candidate result.

#### 2026-09-25 03:23 KST seed8104 control replay confirmed through U3

The same live PPO process has advanced to update 3/8 for the urgency-squared control on seed 8104. Actor checkpoint hashes and mean progress, mean speed, collision-frame count, and maximum damage match the prior recovery-gate 0.5 arm exactly through U3. This strengthens the paired-control reproducibility check; the linear treatment has not begun and no treatment conclusion is available.

#### 2026-09-25 03:24 KST seed8104 control U4 reproduced prior low-speed hazard segment

The urgency-squared control reached U4/8 and still exactly matches the earlier recovery-gate 0.5 actor hashes and rollout metrics. Across each 1,024-decision update over four TRAIN cells, mean speed/progress were U1 40.50 m/s/.150, U2 37.94/.162, U3 24.24/.175, and U4 29.62/.250. U3 recorded 5 collision frames and 4 off-track frames; U4 recorded 1 collision and 2 off-track frames. These are short TRAIN rollout aggregates, not lap results, but they reproduce the low-speed first-obstacle sector already seen in the prior trace. The 70 m/s reward target is not translating into realized speed here; treatment comparison remains pending.

#### 2026-09-25 03:32 KST first linear-risk update and separate speed diagnosis

The seed-8104 urgency-linear arm has started. Its first 1,024-decision rollout exactly matches the squared control in speed (40.499 m/s), gas mean (0.10258), progress (0.15485), and collision count (0), while the post-update actor hashes differ. This is the expected paired pattern: actions are shared before the first reward update, then the changed urgency shaping alters PPO's update. It verifies the intervention is active; it is not a performance result.

Source inspection confirms risk exponent changes three TRAIN reward pathways together: hazard-adjusted target speed, obstacle throttle cost, and obstacle brake reward. At first cue urgency 0.06, linear urgency contributes 16.7 times the squared urgency factor when obstacle presence/lane overlap are held fixed. It cannot affect obstacle-free curve sections, so hazard response and clear-road speed remain separate diagnoses.

The squared control's U5 TRAIN aggregate was mean speed 35.375 m/s, progress 0.193, gas mean 0.0948, gas max 0.2166 under the 0.24 limit, and zero gas saturation; it still had two off-track frames despite zero collision frames. These short rollouts show low throttle choice and weak route coverage even without a collision in that sample, not a full-lap outcome. The older curve-brake PPO attempt has a saved 7/8 control update but no treatment; it is incomplete, not rejected. After this urgency pair, inspect whether that exact saved comparison can be resumed safely before designing another speed experiment.

#### 2026-09-25 03:34 KST urgency-linear U2 is promising but mixed

The matched seed-8104 urgency-linear arm reached U2/8. Relative to urgency-squared at U2, its four-cell TRAIN aggregate has mean progress .2065 vs .1616, mean speed 39.29 vs 37.94 m/s, collision-marked frames 2 vs 5, max damage .4 vs 1.0, and gas mean .1044 vs .1071; neither saturates the .24 gas limit. First-update rollouts were identical before actor hashes diverged, confirming the reward change drives the difference. This is still a small TRAIN sample, not a finish or TUNE result.

The U2 per-cell trace refines the action hypothesis. On the obstacle map seed 20260924, squared risk first cues at decision 78/progress .220/urgency .07/speed 40.3, then brakes at d82 with .09 and collides at d118/progress .374/speed 11.8. Linear risk cues one decision later at d79/progress .224/urgency .08/speed 40.1, brakes at d84 with .14, reaches progress .484, and has no collision in this 256-decision fragment. On train map seed 20260921, linear risk still collides at d120/progress .211/speed 31.6 after a weak .05 brake; the squared control has no collision in that fragment. Thus linear risk increased brake magnitude and helped one exposed obstacle cell, but it did not make first braking earlier and has a counterexample. Urgency exponent changes reward strength after a cue, not when pixels first expose the obstacle. Continue the complete four-arm run and fixed TUNE screen; do not promote or alter the live comparison based on U2.

#### 2026-09-25 03:38 KST urgency-linear U4–U5 reverses the early gain

Seed 8104 linear-risk updates U4 and U5 do not sustain the U2 improvement. At U4, linear versus squared measured 25.7 vs 29.6 m/s, progress .234 vs .246, collision frames 3 vs 1, off-track frames 3 vs 2, and max damage .4 vs .2. At U5 it was 26.1 vs 35.4 m/s, progress .176 vs .193, collision frames 4 vs 0, and max damage .6 vs 0. Gas mean at U5 was higher for linear (.109 vs .095) with zero saturation in both arms. This points to speed loss after unsafe trajectories/damage, not an actuator ceiling. It remains one trained seed and short TRAIN fragments; await seed 8105 and TUNE before a decision.

U5 trace on TRAIN map seed 20260925 shows a concrete stall: control first cues at d90/progress .215/urgency .07 and brakes .14 at d91, then reaches .586 progress with no collision. Linear cues at d92/progress .219/urgency .18 but brakes only .05; at d103 urgency is 1.0 while brake is still 0 and gas .16. It collides at d106/speed 9.5, then again at d109/speed 3.1, and sits at progress .255 by d113 with near-zero speed, gas .17, and steering changing sign. The recovery reward is disabled in this high-risk damaged state by the shared .5 gate. This is direct evidence that the current learned actor lacks an effective post-impact escape behavior in that cell; it does not yet prove the gate alone caused it. The next recovery hypothesis already in the queue is to make recovery feedback depend on actual positive progress, rather than rewarding stationary gas/absolute steering. Do not launch that pair until the current urgency comparison finishes.

#### 2026-09-25 03:42 KST seed8104 U8 closes with a modest aggregate gain and cell-level failures

The first paired policy seed completed U8 for both arms. Linear versus squared TRAIN aggregate: mean speed 30.95 vs 28.84 m/s, mean progress .251 vs .229, collision frames 3 vs 5, off-track frames 1 vs 2, max damage .4 vs .6; neither arm finished a training fragment and neither saturated gas. The trajectory is non-monotonic: linear was worse at U4–U5, then better on this U8 aggregate.

Per-cell evidence remains mixed. On obstacle map seed 20260924, linear reached .488 without collision while squared collided at .240. On train map seed 20260921, linear stopped at progress .359 with a zero-speed collision; on seed 20260925 it reached .625 but collided at 20.8 m/s. The U8 aggregate is not a robust safety win from one training seed. Await matched seed 8105 and the registered TUNE-only evaluation; no held-out/official map or SOTA promotion has occurred.

#### 2026-09-25 03:45 KST seed8105 paired control has begun

The second TRAIN-only shared BC initialization completed. The run now has both shared states prepared, and seed 8105 urgency-squared control has created a live run-status at update 0. Seed 8104 control/treatment remain complete at U8; total arms completed is 2/4. The same Python worker is live with CPU time increasing during initialization. No TUNE, held-out, or official map was loaded. Continue this existing run; do not duplicate its second seed.

#### 2026-09-25 03:47 KST seed8105 control U1 replay parity

The second policy seed has started its urgency-squared control and completed U1/8 (1,024 decisions). Its actor hash, progress, speed, gas, and collision count match the prior recovery-gate 0.5 arm byte-for-byte at U1: mean speed 36.376 m/s, mean progress .22049, and 4 collision-marked frames. This verifies the second-seed control replay as well. The seed-8105 linear treatment has not started; the overall run remains 2/4 arms complete and TUNE is still closed.

#### 2026-09-25 03:52 KST seed8105 control U4 reproduces a late-brake collision

The same U4 control checkpoint and aggregate replay exactly match the earlier recovery-gate run: actor SHA-256, mean speed 37.471 m/s, mean progress .14337, 5 collision decisions, and maximum damage 1.0 are identical. Its decision trace on TRAIN cell `custom-track-haic-train-20260921:20260921` shows obstacle-present first at decision 75 (progress .21, speed 41.4 m/s, urgency .10); urgency rises to .97 by d78 while brake stays at zero. The first brake is only .12 at d79 near 39 m/s. Contact starts at d81 at 13.1 m/s; d81–85 records five collision decisions and damage rising .2 to 1.0, with brake at zero and positive gas. This gives a concrete approach-and-contact mechanism: the policy reaches high speed, then reacts late and keeps applying gas through sustained contact. It is one 256-decision TRAIN fragment, not a full-lap causal result. Its high mean speed alongside low progress also shows that a high speed target alone does not ensure route progress; the short fragments mix four cells and cannot assign the overall lap-time gap to this collision alone. The registered linear-urgency PPO arm remains the next discriminating test; do not start another run before its paired seed and TUNE screen finish.

#### 2026-09-25 03:55 KST seed8105 control U7 shows noisy, non-monotonic PPO rollouts

The same live control reached U7/8. Across U1–U7, its per-update TRAIN mean speed ranged 29.98–38.23 m/s and mean progress .143–.220; collision decisions ranged 0–6, with no training fragment finish. U2 had zero collisions but the lowest obstacle exposure, while U4–U5 had five and six collisions and max damage 1.0/.8. U6–U7 improved those collision/damage counts but speed and progress did not rise monotonically. Because each update is a short rollout over four fixed TRAIN cells, this is evidence of unstable sampled behavior, not a complete-lap learning curve. It reinforces the need to judge the paired linear-risk change on the registered TUNE episodes after all arms finish. At this snapshot the run still reports 2/4 completed arms, the current control is at U7/8, and TUNE/held-out/official data remain unloaded.

#### 2026-09-25 03:59 KST seed8105 control completes; matched linear arm starts

Seed8105 squared-risk control completed U8 with 33.29 m/s mean speed, .188 mean progress, 4 collision decisions, 2 off-track decisions, maximum damage .6, and no finished TRAIN fragments. Per-cell U8 exposure varied: one obstacle cell reached .50 progress with no collision; the other obstacle cell had one collision; a train cell had three collisions and reached .25; the second train cell had none and reached .23. The four-cell aggregate is not a lap score. Seed8105 linear-risk treatment then began U1 from the same shared BC actor; its first rollout matches control U1 on mean speed 36.376 m/s, mean progress .22049, and 4 collision decisions, before the changed reward updates the actor. The worker is still active at 3/4 completed arms and the fixed TUNE screen remains closed.

#### 2026-09-25 04:03 KST seed8105 linear-risk U4 is not yet an exposure-matched win

At U4, linear versus squared had 0 versus 5 collision decisions and max damage 0 versus 1.0, but mean hazard-visible rows were also lower (33 versus 45) and off-track rows were 3 versus 2. Mean speed was 38.98 versus 37.47 m/s, while mean progress was slightly lower (.136 versus .143). Cell traces show why the aggregate cannot establish safer behavior: on train-map seed 20260921, both arms' first obstacle cue arrived around d75–76 and both first issued a small brake at d79; the squared arm then had five collision rows, while the linear arm had none in this 256-decision fragment. The linear arm did not brake earlier in that exposed cell. Other cells had different cue timing/exposure, including one with no cue in either arm. Continue U5–U8 and the fixed TUNE screen; keep collision counts paired with cue exposure, off-track events, damage, route progress, and completion.

Segmenting the reset-separated traces in that same train cell refines the comparison: squared-risk starts a collision segment with cue d75 at 41.4 m/s and urgency .10; linear-risk cues at d76 at 41.0 m/s and urgency .02. Both first brake at d79, but linear-risk commands .19 at 36.6 m/s versus .12 at 39.0 m/s for squared-risk. The squared segment then collides d81–85, accumulating damage .2→1.0 and falling to 3.6 m/s; the linear segment has no collision through d85 and continues to .235 progress. This is a promising within-cell signal that linear shaping strengthens braking enough to avert this particular contact, although it did not make the first brake earlier and remains one trained-seed trajectory. The aggregate and full TUNE evaluation still decide whether it generalizes.

#### 2026-09-25 04:08 KST seed8105 linear U5–U7 trades collision counts for route-control errors

The live linear-risk arm reached U7/8. It had six collision decisions and max damage 1.0 at U5, then zero collision decisions at U6 and U7; however off-track decisions were 3 and 4 on those updates, versus 3 and 2 for squared-risk. Mean progress remained lower (.16, .16 versus .18, .19), while mean speed was similar (37.48/35.77 versus 38.23/31.77 m/s). Gas averages stayed near .10 with no saturation. Across these short, noisy TRAIN rollouts, linear urgency has not produced a consistent safety-and-progress gain: U4's exposed cell was promising, U5 reversed, and U6–U7's collision-free aggregates had more off-track rows and lower progress. Finish U8 and the registered TUNE comparison before deciding whether this reward change helps lap completion; no candidate or SOTA promotion is justified yet.

#### 2026-09-25 04:10 KST linear urgency is rejected by the completed fixed TUNE screen

All four PPO arms completed U8 and the registered evaluation completed on the same four policy/map-seed cells. Linear-risk finished 0/4, had 16 collision decisions across four episodes, mean final damage .8, mean speed 31.20 m/s, and mean DNF progress .440. Squared-risk finished 2/4, had 8 collision decisions, mean final damage .4, mean speed 31.35 m/s, and median finished lap 19.46 s. Both had 0/4 finishes under 13 s. The two TUNE seed labels reproduce one unique map geometry and exact per-policy outcomes, so the result tests two trained policies on one TUNE layout rather than generalization across independent maps. The paired result file explicitly says `rejects_on_safety`, `candidate_is_not_sota=true`, and Tune cannot promote to SOTA; `SOTA.md` remains unchanged.


#### 2026-09-25 04:36 KST curve-brake pilot integrity audit and next PPO screen

The host process check found only the Hermes service and browser harness; no PPO training process is live. The old `curve-brake-reward-ppo-pilot-v1` was interrupted by the operator after its control had persisted U7/8 (7,168/8,192 decisions). Its treatment never started, it has no paired result or final Tune traces, and the saved 2/2 Tune completion at 19.92 s is only update-7 selection data from one repeated geometry. It is incomplete, not evidence for or against curve-brake reward.

The old pilot cannot pass its own source-integrity preflight unchanged. Its manifest expects `training/train_policy.py` SHA-256 `5C07674C630166A14315B302BB2488294B1F59BA776BB05632FA1B663463A11D`, while the registered worktree now contains `C22EF81F37ABDCB170CA194DE205316BA57017459F0C75A35FDAD7FA1AA73AFF`. The runner, run config, selected checkpoint, split, and all train/tune/held-out map hashes still match; among runtime source files only `training/train_policy.py` differs. A fresh run therefore needs a new protocol/source manifest and new output directory. Do not overwrite the partial U7 control or call it a paired experiment.

The speed gap and finish failures now separate into three mechanisms. The validated actor has held-out completion 6/8, median 19.32 s and p90 22.42 s; its aggregate mean speed is 40.41 m/s and max 52.96 m/s, but those motion values combine finishers and DNFs and are not per-lap causal estimates. Its throttle range is 0.12, while the recent PPO comparison widened it to 0.24 and still showed no saturation, so a higher hard cap alone is not supported. In the latest urgency pair, Tune speed was effectively tied (31.20 vs 31.35 m/s), linear risk finished 0/4, squared risk 2/4, and neither produced a sub-13-second finish. Separately, a faster comparison actor reached 65.34 m/s on a clear custom segment, then repeatedly left the road at progress .237 with no nearby obstacle. Thus higher speed alone harms route completion, and visible-hazard urgency alone does not teach generalizable braking.

Current reward inspection shows the speed term charges only when speed is below a curve-adjusted target; it does not directly penalize carrying excess speed into a bend. The validated SOTA trace also shows the critical obstacle on official Track 1 is inside a bend and its pixel cue arrives late. The next discriminating PPO screen is the already-registered curve-brake hypothesis: train-only `curve_magnitude × normalized_brake` reward coefficient 0 vs 6, same actor initialization, fresh optimizer, matched training seeds, same TRAIN/TUNE split and decision budget, with no runtime rule or teacher state in the submitted actor. Keep the previous pilot untouched; pin the current trainer source and use a new experiment ID/output directory. Primary gates are Tune completion and damage; report lap-time and under-13 fraction, plus brake before the obstacle becomes visible and high-curvature speed. If braking rises without improved completion or with slower laps, reject the reward. Read the updated trace before choosing the next single variable: test stronger brake reward if urgent visible cues still receive no brake, a progress/clearance recovery reward if contact avoidance improves but stalls remain, and more TRAIN approach coverage only if the relevant states were rarely sampled.

#### 2026-09-25 04:46 KST SOTA official trace: cue is visible, but braking is not sustained

Read the raw 163-decision `official-track1-seed42-feature-trace.json` for the validated SOTA actor. At d57, before the obstacle pixel cue, curvature reaches .583 at 39.65 m/s and the actor briefly brakes .053. The obstacle first becomes visible at d58 / 16.02 m / 38.74 m/s with urgency .917 and brake .045. Despite urgency .963 at d59 and 1.0 by d60, brake returns to zero; gas then rises from .046 at d61 to .082 at d63. First contact is at d63 / 3.84 m / 7.63 m/s with gas .082 and brake 0; a second contact occurs at d65 / 3.81 m / 4.27 m/s with gas .079 and brake 0. Progress stays .163 while speed falls to .04 m/s by d70, with gas .068 and no brake. The episode later retires off-track. This trace shows that pixels do expose the obstacle before contact; the failure is the actor's weak, non-persistent brake response and failure to escape after low-speed contact, not complete detector blindness.

The completed recovery-gate Tune screen provides a related but distinct signal: the 0.5 risk gate and 2.0 gate both finish 2/4, have 0/4 under-13 finishes, and recover 0 of 2 stalled segments. The 0.5 gate has 200 high-risk stall decisions versus 4 for 2.0, while the 2.0 arm has more collision-marked decisions (16 vs 8), more final damage (.8 vs .4), and slightly slower finished median (19.60 vs 19.46 s). Therefore changing the gate reduces high-risk stalling only by accepting greater contact/damage; neither gate teaches escape. Keep the curve-brake pair focused on persistent pre-contact braking. If it fails while visible urgent rows still receive no brake, the next paired RL change should increase `OBSTACLE_BRAKE_REWARD` from 6 to 12 while holding hazard risk, gas penalty, policy, maps, and budget fixed; the trace already has at least three urgent visible decisions before contact. If stronger braking prevents contact but pinned stalls remain, test a train-only recovery objective measured by actual progress or obstacle clearance, rather than widening the action-time gate again.

#### 2026-09-25 04:52 KST PPO rollout exposure audit rejects a sampling-shortage explanation

Aggregated the saved training `decision-traces.jsonl` across both policy seeds, both urgency arms, and all 8 updates. These are decision rows, not independent episodes. In U1–U2, 8,192 PPO decisions included 272 obstacle-visible rows with urgency ≥.8; 44 had brake >.02 (16.18%), mean brake was .01330, and mean gas was .08646. In U3–U5, 12,288 decisions included 1,866 urgent-visible rows; only 53 braked (2.84%), mean brake .00259, mean gas .13852. In U6–U8, another 12,288 decisions included 1,350 urgent-visible rows; 46 braked (3.41%), mean brake .00308, mean gas .13479. Thus the actor had thousands of high-risk cue opportunities; braking weakened as training continued while gas rose. This weakens the hypothesis that the main fix is simply to collect more hazard approaches. After the curve-brake screen, if TUNE traces still show visible urgent cues followed by zero brake, prioritize a single-variable increase in learned obstacle-brake reward (6→12) and inspect policy drift across PPO updates. Keep post-impact progress/clearance reward as the subsequent branch if persistent braking prevents contact but the car still gets pinned.


#### 2026-09-25 05:05 KST speed-target and recovery screens identify the control bottleneck

Read the raw U8 TUNE summaries. In the paired speed-target-70-vs-84 screen, both actors finished 0/4 and had 0/4 under-13 results. Raising target speed to 84 reduced actual mean speed from 31.60 to 27.19 m/s and mean DNF progress from .658 to .315, while collision decisions rose 8→12 and mean final damage .4→.6. Both seed labels are repeated resets of one TUNE geometry. This rejects the idea that the actor is simply slow because its target is too low: the larger target made the policy fail earlier, which lowered realized speed.

On the same registered TUNE geometry, the best recent completed PPO screen finished in 244 decisions at 19.46 s; reaching 13 s at the same decision interval requires about 163 decisions, roughly one-third fewer decisions and about 50% higher average route speed. This is a scale estimate from the TUNE track, not an official-track guarantee. The 84 m/s target test shows that pursuing that speed before learning curve control is unsafe. The root reward pays for speed shortfall but does not directly charge excess speed in a bend, so the next PPO screen remains the preregistered curve-magnitude × brake-reward coefficient 0 vs 6, with completion and damage as primary gates.

The recovery-gate artifact was found already complete in the current root checkout. Risk threshold 0.5 versus 2.0 produced 2/4 finishes on both arms, no under-13 finish, and zero recovery in either of two stalled segments. Opening the gate reduced high-risk stall decisions 200→4 but raised mean final damage .4→.8; it did not teach escape. This is evidence that rewarding action magnitude while pinned does not ensure movement. If the curve-brake screen prevents contact but leaves stalls, the next recovery PPO variable should reward observed progress or clearance during recovery rather than more gas or steering magnitude. No SOTA change is justified by these TUNE screens.


#### 2026-09-25 05:24 KST stale pilot v2 rejected; KL audit keeps curve-brake test selected

A's v2 worktree was checked against the live root before any training. The selected SOTA checkpoint is byte-identical in both locations (SHA-256 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`), but the root and worktree have different `training/train_policy.py` hashes (`54689FAB...` vs `2D9D7D7B...`), different `training/ppo.py` hashes (`34113158...` vs `10B04D73...`), and different site-split hashes (`3752F33A...` vs `49E1774B...`). The current root split has 4 TRAIN, 2 TUNE, and 5 held-out episodes; the v2 worktree snapshot is stale and must not supply training/evaluation maps. No PPO process or v2 source manifest exists, so that worktree experiment did not start.

Task B completed a read-only KL audit. The paired KL=0 vs .5 U8 runs have only a weak, seed-reversing cue-conditioned brake difference; KL=.5 vs 2.0 is incomplete; the latest urgency comparison held KL fixed while braking still faded during PPO. The evidence does not identify KL anchoring as the cause of brake collapse. KL=.5 did finish 2/4 on one repeated TUNE geometry versus 0/4 for KL=0, but both had 0 under-13, 4/4 collision episodes and mean final damage .8, so this is not a safety or generalization win.

Do not spend another run on KL or on the already-completed recovery threshold. The discriminating next RL change remains curve-magnitude × normalized-brake reward coefficient 0 vs 6, using the current root checkpoint and root split, fresh Adam and matched seeds 8104/8105, 8,192 decisions/U8 per arm. TUNE selection must be disabled during updates and both actors evaluated only after all four TRAIN arms finish. Task A has been redirected to implement this in the current root checkout; no run is claimed until a matching PID/status and artifacts are present.

#### 2026-09-25 fixed-image action diagnostic corrects the rollout-exposure interpretation

A read-only deterministic action audit evaluated the urgency-square and urgency-linear PPO checkpoints at U2, U5, and U8 on the same 312-observation bank captured from four current TRAIN episodes. Of these, 76 observations had a visible urgent obstacle cue, 50 had high curvature, and 18 had both. On urgent-visible observations, the fraction with brake above .02 increased from .053 to .184 for urgency-square and from .053 to .105 for urgency-linear between U2 and U8. Mean brake also increased in both arms. The high-curvature bank showed the same direction of change. This does not support the earlier causal phrasing that PPO updates weakened the actor's brake response for a fixed visual state. The declining brake fraction in rollout aggregates can instead reflect a change in which states the policies visited; this audit is suggestive because its image bank comes from the SOTA path rather than the exact PPO rollout observations, and it contains no held-out or official-map states. Keep the preregistered curve-brake PPO comparison as the next performance test, and have it report both fixed-bank actions and actual completion, damage, and lap time. No new SOTA or submission candidate is established by this diagnostic.

#### 2026-09-25 reward-balance calculation supports the blind-curve PPO comparison

The current root reward and official Track 1 trace quantify why the registered curve-brake coefficient 0-vs-6 test is discriminating. Before the hazard pixel cue at d57, curvature is .583 and speed is 39.65 m/s. With `SPEED_TARGET=70`, `CURVE_SPEED_RELIEF=.4`, `SPEED_SHORTFALL_PENALTY=.6`, and `REWARD_SCALE=.1`, the curve-adjusted target is 53.68 m/s and the current one-sided speed shortfall costs about .012 reward per decision. The same trace's brief .053 brake command is .189 of `MAX_BRAKE=.28`; adding coefficient 6 would yield about +.066 reward for that action at this curvature (`6 × .583 × .189 × .1`), roughly 5.5 times the immediate shortfall cost. Thus a curvature-conditioned signal can favor slowing before the obstacle is visible, when the existing hazard-risk term cannot. This calculation predicts an incentive change, not a driving improvement. The official trace still shows braking fading after the d58 cue and contact by d63, so action persistence and post-impact recovery remain separate failure modes. Judge the pair by Tune completion, damage, collision, and lap time; use pre-cue speed/brake and fixed-image actions only to explain the result.

#### 2026-09-25 curve-brake runner implementation status (not a training result)

The active root-checkout task has extended `training/train_policy.py` with the optional curve-brake reward parameter, actor-only checkpoint initialization, and deferred TUNE selection. The source now carries a new hash compared with the frozen starting version. At the latest workspace check, `training/curve_brake_screen.py`, its focused test file, the registered experiment output directory, and any matching PPO process were still absent. Therefore no arm has trained, no Tune comparison exists, and these code changes are not performance evidence. Continue the same registered experiment in the root checkout and record a result only after the four matched arms and post-training Tune evaluation produce raw artifacts.

#### 2026-09-25 recomputed hazard risk rules out a persistent risk-gate miss in the official failure

Recomputed the trainer's `visible_hazard_risk` from the stored official Track 1 controller features at the collision sequence. At the first pixel cue (d58), risk is about .372 because the lateral feature places the obstacle near the edge of the inferred lane. On d59–d63, the same formula gives approximately .849, .874, .888, .980, and .947. Despite those high values, brake is zero from d59 through contact; gas rises to .046, .071, and .082 on d61–d63. With `MAX_GAS=.12`, the existing scaled obstacle-throttle term alone charges about -.272, -.464, and -.518 reward on those three decisions. Therefore the late sequence is not explained by a missing pixel cue or a persistently low lane-risk estimate: the trained actor still selects accelerating actions in states where its own training reward strongly discourages them. This points toward weak policy generalization or credit assignment/action persistence, beyond reward magnitude alone. The curve-reward pair remains useful for the blind pre-cue approach, but if it changes pre-cue speed and leaves this high-risk gas/brake pattern intact, the next PPO hypothesis should target exposure/generalization or temporal persistence rather than simply increasing the same hazard coefficient. Do not train on held-out or official evaluation maps to test that hypothesis.

#### 2026-09-25 focused screen checks found harness blockers before PPO

Ran `.venv\\Scripts\\python.exe -m pytest -q tests/test_curve_brake_screen.py` after the screen helper appeared. Result: 3 failed, 4 warnings. The failures are in experiment plumbing, not policy performance: the split loader produced one episode per map instead of the four TRAIN/two TUNE seed episodes; fixed-image action summaries required speed/gas/brake fields absent from the bank metadata; and one test compared a float32 feature to .7 exactly. Task A received the failing assertions and was asked to fix and rerun the focused checks before any PPO arm starts. There is still no training process or experiment output directory, so these failures do not alter SOTA.

#### Correction to the focused-test failure note

The first split-loader test had already passed its 4 TRAIN/2 TUNE episode-count assertions. Its failure was only that the test expected each geometry file to be loaded once per seed; loading each geometry once and then instantiating its registered seeds is the intended behavior. The test expectation was corrected. Re-running `.venv\\Scripts\\python.exe -m pytest -q tests/test_curve_brake_screen.py` now passes: 3 passed, 4 deprecation warnings. The earlier statement that the loader produced only 1 TRAIN/1 TUNE episode was inaccurate. This clears the focused harness check; preflight and the four-arm PPO run are still pending.

#### 2026-09-25 frozen-split preflight completed read-only

Loaded the current root split through the screen loader. It materialized exactly four TRAIN episodes (two registered seeds for each of two custom map families) and two TUNE episodes (two seeds for the registered TUNE map), with zero held-out episodes loaded. The starting actor checkpoint matches the frozen SHA-256 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`; the split hash is `3752F33A166E537AAC0AFA6CBB63618467EE3BB75EC7114DF734E33D2BB1D5B2`, and PPO core hash is `34113158AA0F221ADEF0AAC1BBC53895F5B4CFE37C7BFCFF88A773A11E729D64`. During implementation, `train_policy.py` changed to hash `ABAAC4179C6DEB4E97A377FF68F42B6CC69DA5217797E4D4269A205515C9E0BB`; the run manifest must pin this final code hash only after implementation is frozen. This was a read-only preflight, not a training result.

#### 2026-09-25 trainer regression checks pass before the PPO screen

Using the project `.venv`, `python -m pytest -q tests/test_train_policy.py` completed with 22 passed, 4 deprecation warnings, and 2 subtests passed. The separate `tests/test_curve_brake_screen.py` run passed 3 tests. These checks validate trainer and screen plumbing only; they are not driving-performance evidence. The root split preflight also confirmed the exact four TRAIN/two TUNE episodes and loaded no held-out geometry. The registered four-arm learning run remains the required next evidence.

#### 2026-09-25 held-out breakdown separates custom-map completion from official performance

Expanded the selected actor's eight held-out records from `actual-agent-heldout-episode-results.json`. Five records are different seed labels for the same custom map `custom-track-haic-heldout-20260923`; all five finish at exactly 19.32 s with no collision. The three official records are materially different: Track 1 seed 42 retires off-track at progress .163 after two collisions; Track 1 seed 42 plus an added obstacle crashes at progress .148 with five collisions and damage 1.0; Track 2 seed 101 finishes but takes 25.52 s at mean speed 40.6 m/s and max 52.9 m/s, with no collision. Thus the headline 6/8 completion is dominated by five repetitions of one easy custom geometry. On these three existing official evaluations, completion is 1/3; the only finisher is still almost twice the 13 s target. The slow clean Track 2 lap shows that collision stalls are not the only speed bottleneck: the actor's learned pace remains low even without a collision. Track 1 failures remain a separate obstacle-in-bend and recovery problem. The current four-arm custom-only curve-brake screen can isolate a reward mechanism, but cannot establish official-domain success; do not promote from it. The subsequent RL iteration needs broader allowed TRAIN geometry and a separate official evaluation, without training on held-out/official evaluation maps.

#### 2026-09-25 checkpoint lineage explains the low-speed official finisher

Inspected the selected checkpoint's original `experiment_config.json`. Its phase is `ppo_pixel_hazard_reward_continuation` and its speed reward is the older `safe_speed_reward_target=50.0`, `safe_speed_reward_weight=0.1`; it does not use the newer one-sided speed-shortfall objective. A per-decision reward proportional to speed sums approximately to traveled distance on a fixed route, so it provides little preference for finishing that distance in less time. This is consistent with the held-out official Track 2 run: no collision, yet mean speed 40.6 m/s, max 52.9 m/s, and lap time 25.52 s. The current trainer's target-70/shortfall-0.6 term was introduced to create a direct cost for staying below cruise pace, but past Tune screens still had no sub-13 finish, and target 84 worsened realized speed/completion. Therefore the lineage explains a likely source of the low-speed policy but does not prove the replacement solves it. In the registered 0-vs-6 curve-brake pair, keep target 70/shortfall .6 common to both arms so the paired difference remains the curve-brake coefficient; attribute any shared speed change cautiously and judge both arms on lap time and completion.

#### 2026-09-25 06:40 KST curve-brake PPO run is live; first matched seed is in progress

The current root-checkout runner passed its frozen preflight: four registered custom TRAIN episodes, two custom TUNE episodes, zero held-out and official episodes loaded, strict actor initialization, and 34 runtime/source files pinned. The fixed reward configuration includes speed target 70 and shortfall penalty 0.6; only `curve_brake_reward` changes between 0 and 6. The deterministic SOTA observation bank contains 1,040 TRAIN-only pixel states; the SOTA actor completed all four TRAIN episodes without collision/off-track but had no sub-13-second completion.

The actual four-arm PPO run is running as PID 56580 in `artifacts/haic/curve-brake-reward-ppo-2seed-u8-20260925T0529KST`. Seed 8104/coefficient 0 completed all 8 updates (8,192 decisions) in 571 seconds; its final checkpoint SHA-256 is `77C764925E7354CE01A9C1E21FF1C7B3687FA12F7693C817FAE050B2C052822E`. The matched seed 8104/coefficient 6 arm has reached its U2 snapshot. Seeds 8105 and the post-training TUNE evaluation remain pending. The runner loads TUNE only after all four arms pass the checkpoint/source integrity gate. No lap-performance result, SOTA promotion, official evaluation, packaging, or submission is claimed yet.

#### 2026-09-25 06:55 KST PPO screen update: three arms complete, final matched arm started

The live runner has completed the seed 8104 pair (curve-brake coefficients 0 and 6) and seed 8105/coefficient 0, each with 8 PPO updates and 8,192 TRAIN decisions. The final seed 8105/coefficient 6 arm started at 06:55 KST under runner PID 56580. The TUNE gate remains closed; this update contains no comparative driving result. Continue to the final arm, verify all four checkpoint/source records, then run the registered two-episode TUNE comparison and same-image actor replay.

#### 2026-09-25 07:10 KST curve-brake screen result: completion is seed-specific; urgent braking is still too weak

The frozen four-arm PPO run completed with a passing source/checkpoint integrity gate. On the registered two TUNE seed labels for one custom geometry, curve-brake coefficient 0 finished 0/4 episodes and coefficient 6 finished 2/4. Both finishes came from PPO seed 8104 and were identical 20.64-second laps; seed 8105 failed 0/2 under both coefficients. No policy achieved the 13-second target. The apparent 2/4 gain is a mechanism signal on one PPO seed and one repeated TUNE map, not a generalization result or SOTA candidate.

The raw seed-8105 traces identify the remaining failure more precisely. With either coefficient, the same TUNE route reaches progress .753; the first collision occurs at decision 192 or 193 at speed 12.2 or 10.3 m/s while urgency is 1.0 and brake is zero. After impact, the car stays at zero speed for about 100 decisions with urgency 1.0, gas around .086, and brake zero, then retires off-track at decision 293. The coefficient-6 treatment changes some fixed-image actions only slightly: U8 urgent-visible brake-active fraction rises from .378 to .432 for seed 8105, while immediate pre-visible braking remains zero in all TUNE arms. Thus curve-conditioned braking alone did not produce the sustained urgent-cue response or recover the failing seed.

The next RL-only discriminator is the already registered `OBSTACLE_BRAKE_REWARD` change 6→12, with curve-brake coefficient fixed at 6. Keep the actor, optimizer, speed objective, hazard risk, maps, seeds, and U8 budget unchanged. If higher reward reduces collisions and improves completion across both PPO seeds, keep the learned change for a broader allowed TUNE screen. If it only prevents contact but leaves a pinned stall, test a train-only progress/clearance recovery reward next. Do not promote from the current TUNE result.

#### 2026-09-25 07:10 KST follow-up plan registered: paired obstacle-brake reward 6 vs 12

Plan: `docs/superpowers/plans/2026-09-25-obstacle-brake-reward-2x-screen.md`. The only between-arm reward change will be the training-only obstacle-brake coefficient. Use PPO actor-only, seeds 8104/8105, four registered custom TRAIN cells, two registered custom TUNE episodes, fresh Adam, 8,192 decisions/U8 per arm, and delayed TUNE after all source/checkpoint hashes pass. No teacher runtime, held-out/official geometry, SOTA promotion, package, or submission is part of this screen.

#### 2026-09-25 07:28 KST obstacle-brake PPO screen frozen and preflight passed

Implemented the train-only `obstacle_brake_reward` coefficient from `shape_transition_reward()` through `collect_rollout()` and `train()`, with finite/non-negative validation and checkpoint/result metadata. The unchanged default remains 6.0; focused trainer and screen tests passed (27 tests plus 5 subtests). The new immutable run is `artifacts/haic/obstacle-brake-reward-ppo-2seed-u8-20260925T072803KST`: PPO actor-only, seeds 8104/8105, reward 6/12, curve-brake reward fixed at 6, fresh Adam, 8,192 decisions/U8 per arm, exact registered four TRAIN and two custom TUNE episodes, and no held-out/official maps. Preflight passed strict actor loading, 4 TRAIN / 2 TUNE, 0 held-out / 0 official loaded, and 34 pinned source files. The result remains Tune-only and cannot promote SOTA. Hypothesis: doubling learned positive brake feedback on visible urgent hazards reduces collision/stall exposure across both PPO seeds without worsening finish time; reject it if gains are seed-specific, damage/collisions do not improve, or the 13-second rate remains zero.

#### 2026-09-25 obstacle-brake 6-vs-12 PPO screen: reward increase rejected

The frozen four-arm actor-only run completed in 3,254 s with all four final checkpoints passing the integrity gate. Each condition had four TUNE episode labels, but all four labels repeat one custom TUNE geometry across two PPO seeds; this is a mechanism screen, not independent-map evidence. At obstacle-brake reward 6, seed 8104 completed 2/2 at 20.64 s while seed 8105 completed 0/2. At reward 12, both seeds completed 0/2. Neither condition produced an under-13 s lap. Total collision onsets fell from 10 to 8, but mean final damage across the four repeated episodes rose from 0.50 to 0.60 and mean maximum progress fell from 0.877 to 0.753. Reject reward 12 as a candidate; keep SOTA and submission archive unchanged.

The raw TUNE traces localize the remaining failure: both reward settings can enter a high-urgency obstacle bend at progress .75, then fail to clear it. For seed 8104/reward 12, the collision at decision 192 occurs at 2.67 m/s with urgency 1.0, gas .02, and brake 0; it remains stalled until off-track at decision 292 with speed 0 and gas .09. For seed 8105/reward 12, the first contact is at decision 166 and 42.11 m/s despite brake .06, followed by contacts at decisions 191 and 193 with gas .09 and brake 0; damage reaches 1.0 at progress .753. TUNE immediate pre-onset braking is 0 for all four arms. The reward increase therefore neither establishes persistent braking nor recovery.

A fixed 1,040-image TRAIN replay provides a second diagnostic. On the same 74 urgent-visible states, U8 brake-active fractions were 0.432/0.405 for seed 8104 at reward 6/12 and 0.432/0.432 for seed 8105. The `policy_mean` head moved only 0.078%-0.114% relative L2 from the common initialization after 8 PPO updates; the full non-critic/non-auxiliary actor changed 0.307%-0.420%. This supports a concrete under-adaptation hypothesis: at LR 2e-6 and 1,024 minibatch optimizer steps per arm, changing a reward coefficient does not materially move the deterministic action head. It is suggestive, not proof, because the fixed image bank is from TRAIN and there is no separate optimizer audit.

Next single-variable PPO screen: keep shared visual trunk, critic, auxiliary heads, rewards, split, seeds, and 8,192-decision budget fixed, but compare the policy-mean output layer at LR 2e-6 versus 1e-5 while all other parameters remain at 2e-6. This isolates whether a larger update on the deterministic action head can produce a measurable urgent-state action change without repeating the globally high learning-rate experiment that previously underperformed. Primary gates remain Tune completion, damage, and lap time; reject the treatment if it increases off-track/collision or fails to change fixed-bank policy-mean actions. Both Tune seed labels remain one map geometry, so any gain still requires later map-diverse confirmation.


#### 2026-09-25 follow-up registered: isolate policy-mean learning rate

Plan: `docs/superpowers/plans/2026-09-25-policy-mean-lr-ppo-screen.md`. Added an optional PPO parameter group for only `policy_mean.weight` and `policy_mean.bias`; shared visual encoder, critic, auxiliary head, and log-standard-deviation remain at the base rate. Default training keeps the existing single-rate optimizer. RED tests first failed because the field was absent, then the focused PPO/trainer/screen suite passed (40 tests, 9 subtests). Next compare policy-mean LR 2e-6 vs 1e-5, with global LR 2e-6 and obstacle/curve brake rewards both fixed at 6, on the same actor, seeds, custom maps, U8 budget, and deferred TUNE gate. No run result exists yet.

#### 2026-09-25 policy-mean learning-rate screen frozen; preflight passed

New immutable run: `artifacts/haic/policy-mean-lr-ppo-2seed-u8-20260925T083429KST`. It compares policy-mean LR 2e-6 vs 1e-5 across PPO seeds 8104/8105 while every other parameter stays at 2e-6. Obstacle and curve brake reward are both fixed at 6; the same strict actor initialization, fresh Adam, four custom TRAIN cells, two repeated-geometry custom TUNE cells, 8,192 decisions/U8, and deferred TUNE gate are used. Focused implementation checks passed (40 tests, 9 subtests). Preflight passed 4 TRAIN / 2 TUNE, zero held-out/official loaded, strict actor loading, and 34 pinned sources. No training or TUNE result is claimed yet.

#### 2026-09-25 policy-mean learning-rate screen result: higher head LR rejected

The four-arm actor-only run completed in 2,919 s with the integrity gate passing. At policy-mean LR `2e-6`, seed 8104 finished both repeated TUNE episodes at 20.64 s, while seed 8105 finished neither: 2/4 overall. At `1e-5`, neither seed finished: 0/4. Both settings had 0/4 finishes under 13 s. The TUNE labels all use one custom geometry, so this is a single-layout mechanism screen, not map-diverse evidence.

The treatment moved the policy-mean head farther from initialization (0.622%-0.651% relative L2 versus 0.116%-0.145% for control), but did not improve the deterministic actions on the same 74 urgent-visible TRAIN images. Urgent-state brake-active fraction was 37.8%/43.2% for the two treatment seeds versus 43.2%/43.2% for control; mean brake also fell slightly under treatment. The larger head LR therefore changed parameters without learning a more useful brake response, and it removed both control finishes. Reject `1e-5`; keep SOTA and the submission ZIP unchanged.

The next recovery experiment is based on a specific gate in `shape_transition_reward()`: existing post-impact gas/steer reward applies only below hazard risk 0.5. A car pinned in front of the obstacle has high risk and receives none of that escape-action signal. Test a different, action-outcome-aligned PPO signal: when already damaged and slow, reward only a positive reduction in pixel-derived obstacle risk from the current observation to the next observation. Do not reward gas or steering merely for being commanded. Keep the collision/off-track penalties, progress reward, actor, optimizer, maps, seeds, and U8 budget fixed. This remains training-only shaping; submitted inference remains PPO actor-only. The registered screen must still be rejected as a submission candidate unless it produces a validated under-13 finish and survives later map-diverse/official evaluation.

#### 2026-09-25 recovery-clearance PPO screen frozen and running

Plan: `docs/superpowers/plans/2026-09-25-recovery-clearance-reward-screen.md`. Run: `artifacts/haic/recovery-clearance-reward-ppo-2seed-u8-20260925T095133KST`. Added an optional training-only PPO coefficient that pays only for positive reduction in the actor's pixel-derived hazard risk between the action observation and next observation, after damage, at low speed, with current risk at least 0.5, while the car is on-track and not colliding. The rollout reuses the model's existing next-observation feature pass; inference inputs and submitted actor remain unchanged. The coefficient-0 and coefficient-6 arms use the same actor, fresh Adam, seeds 8104/8105, four registered custom TRAIN episodes, two repeated-geometry custom TUNE episodes, and 8,192 decisions/U8 each; every other reward stays fixed. Focused verification passed (30 tests, 15 subtests); preflight passed with 4 TRAIN, 2 TUNE, 0 held-out, 0 official, strict actor loading, 34 pinned sources. Frozen run-config SHA-256 is `6E8E180EA36B067F7DBEAFE488241E23F8C2F9993E290E740E185A3E7EB1980E`; source-manifest SHA-256 is `565ED2DD769A2784F9516C429AE60DA29552B3BE6348B56D53B88AAD0EE1C542`. Training started under runner PID 29052; no performance outcome exists yet. This Tune-only run cannot promote SOTA or replace the existing ZIP.

Correction: the first run config accidentally inherited the prior screen's policy-mean learning-rate values (`2e-6` and `1e-5`) as clearance-reward coefficients, instead of the registered `0` and `6`. The run was interrupted as soon as its first arm reported coefficient `2e-6`; it had no completed PPO arm, TUNE evaluation, or performance result. Its frozen config and partial artifacts are preserved under the run ID above and are marked invalid in `invalid-run.json`; do not compare or cite its partial files as an experiment result. A new run ID will carry explicit `0/6` arm values and receive a fresh source/config manifest.

Corrected run: `artifacts/haic/recovery-clearance-reward-ppo-2seed-u8-20260925T095639KST`. The frozen arm list now explicitly contains `(8104, 0)`, `(8104, 6)`, `(8105, 0)`, `(8105, 6)` and no policy-mean learning-rate variable. Corrected preflight passed with the same actor/map hashes, 4 TRAIN, 2 TUNE, 0 held-out, 0 official, strict actor loading, and 34 pinned source files. Manifest SHA-256: `88B768195D051DD31336054A44BEFD5B6402FCCD72E29C8C3B408C53E071E595`. Performance training is pending start.

#### 2026-09-25 회복-clearance PPO 실행 진척 및 다음 반증 경로

교정된 실행 `recovery-clearance-reward-ppo-2seed-u8-20260925T095639KST`의 `run_config.json`을 재확인했다. 등록된 팔은 (8104,0), (8104,6), (8105,0), (8105,6)이며, 4 TRAIN/2 TUNE, held-out·official 0, U8=8,192, teacher warmup 0이다. PID 29900은 현재 살아 있다. seed 8104/계수 0 대조군은 8회 업데이트를 618.5초에 완료했고 TUNE은 계획대로 건너뛰었다. seed 8104/계수 6 treatment가 실행 중이며, 마지막 세션 heartbeat에서 450초를 보고했다. 아직 정책 성능 결과는 없다.

대조군 학습 rollout에서 gas 평균은 0.0656, 최대 0.1166, 상한 0.12 대비 포화 비율은 0이었다. 이 기록은 현재 actor가 설정된 가속 상한에 계속 막혀 있지는 않다는 기존 가설을 지지한다. 다만 이는 등록된 custom TRAIN rollout의 통계이며, 공식 맵의 최적 속도나 물리 엔진의 최대 가능 속도를 증명하지 않는다. 이 run은 나머지 보상과 action scale을 고정했으므로 clearance 계수 외 원인 비교에 사용하지 않는다.

TUNE 종료 후에는 완주율·13초 미만 유효 완주·충돌·손상·진행도·완주 랩타임·고위험 정체 길이를 4개 seed/계수 팔별로 확인한다. treatment가 실패 seed의 고위험 정체를 줄이지 못하면 같은 brake/gas/recovery 계수를 반복 조정하지 않는다. 별도 로그의 100-step 고정 행동 정체와 `Agent.act()`의 결정적 정책 평균 추론을 근거로, 다음 PPO 가설은 feed-forward actor가 동일한 정지 화면에서 같은 행동을 반복하는 fixed-point 문제다. 이때만 recurrent actor 상태를 PPO 학습 중 전달하는 단일변수 screen을 설계하고, inference는 actor-only로 유지한다. 현재 실행이 끝나기 전에는 새 학습이나 source 변경을 시작하지 않는다.

#### 2026-09-25 속도 간격 정량화와 병목 해석 보강

공식 Track 2 seed 101의 실제 actor 기록은 25.52초, 평균속도 40.615 m/s, 최고속도 52.939 m/s다. 평균속도×완주시간으로 근사한 경로 길이는 약 1,036.5m이며, 같은 경로를 13초에 주행하려면 평균 약 79.7 m/s가 필요하다. 이는 SOTA actor 평균보다 약 96% 높은 평균이고, 이번 기록의 최고속도보다도 높다. 다른 seed의 corridor 참고 주행은 평균 47.19 m/s, 최고 67.06 m/s, 20.92초였지만 서로 다른 seed/정책이므로 직접 paired 비교는 아니다. 따라서 under-13 목표는 단순히 브레이크를 줄이는 정도로 달성되지 않는다. 이 계산은 속도 gap의 크기를 나타내며 차량의 물리적 최대 속도를 증명하지는 않는다.

선택 actor의 출처 설정은 과거 `safe_speed_target=50`, `safe_speed_weight=0.1`이었다. 매 step 속도 보상 합은 고정 경로에서 대략 이동거리와 함께 증가해 완주 시간을 직접 줄이라는 신호가 약하다. 최신 PPO trainer는 이를 target 70/shortfall 0.6으로 바꿨지만, U8 대조 rollout에서도 gas 평균 0.0656/상한 0.12, 포화 0%였고 prior policy-head LR 2e-6→1e-5 실험은 actor 출력을 더 바꾸면서 TUNE 완주를 2/4에서 0/4로 낮췄다. 따라서 가속 페달 상한을 단순 확장하거나 모든 상황의 속도 목표를 84로 올리는 것은 이미 지지되지 않는다. 회복 screen 이후 별도 pace 실험이 필요하면 위험도와 곡률이 낮은 직선 state에만 빠른 pace advantage를 주어 안전 보상과 분리하고, 완주율·충돌 악화 없이 고정 이미지의 longitudinal action과 실제 랩타임이 함께 개선되는지 판별한다.

진행 갱신: seed 8104 coefficient 6도 U8=8,192를 605.6초에 완료했다. 현재 seed 8105 coefficient 0 arm이 시작되어 마지막 runner heartbeat에서 90초를 보고했다. 네 팔의 무결성 검사와 Tune 평가는 계속 보류 상태다.

진행 확인(동일 run): runner PID 29900의 실제 세션 heartbeat가 seed 8105 / coefficient 0 arm에서 315초를 보고했다. 이는 Tune/완주 결과가 아니라 학습 중 상태다. 프로세스가 살아 있는 한 재시작하지 않는다.

#### 2026-09-25 recovery 보상의 학습 노출 진단

`recovery-clearance` 실행이 고정한 SOTA actor TRAIN-only trace를 집계했다. 네 등록 TRAIN episode는 모두 완료됐고, trace 길이는 257–263 decisions이며 damage≥0.15인 행이 네 episode 전체에서 0개였다. 따라서 초기 deterministic 관측 bank에는 저속·손상·고위험 회복 상태가 없다. PPO rollout은 확률적 행동으로 다른 상태를 방문할 수 있으므로 이것만으로 treatment가 무효라고 단정하지 않는다. 다만 현재 `training-result.json`은 clearance-trigger 횟수를 저장하지 않아, Tune 결과만으로 효과가 없다고 판정하면 학습 노출 부족과 reward 가설 실패를 구분할 수 없다. Tune 이후 raw action/보상 결과와 same-image U2/U5/U8 정책 변화를 확인하고, 행동 변화가 거의 없으면 이번 screen은 recovery signal의 효과를 판별하지 못한 것으로 제한해 해석한다. 다음 회복 학습에서는 PPO가 고위험 정체·손상 상태를 실제로 수집하는지 원시 rollout에서 계측하거나, TRAIN 전용으로 그런 시작 상황을 더 자주 경험시키는 curriculum을 별도 변수로 시험한다. actor 입력·runtime은 계속 PPO actor-only로 둔다.

진행 갱신: seed 8105 / coefficient 0 arm이 8,192 decisions와 8 updates를 692.7초에 마쳤고, seed 8105 / coefficient 6 arm이 시작됐다. 현재 네 팔 중 세 팔이 학습 완료됐으며 Tune gate는 아직 닫혀 있다. seed 8104의 TRAIN rollout 집계는 coefficient 0→6에서 gas 평균 0.05934→0.06479, brake fraction 0.1162→0.0859, 평균속도 24.911→24.906으로 달랐다. 이 값은 각 arm이 방문한 서로 다른 rollout 상태들의 집계라 정책의 동일 상태 반응이나 주행 성능의 차이를 증명하지 않는다. 고정 이미지 action replay와 matched TUNE 결과를 기다린다.

#### 2026-09-25 recovery-clearance PPO screen posthoc 결과

네 PPO 팔 모두 U8 학습과 무결성 검사를 마쳤고, 각 팔에서 두 TUNE seed 레이블의 원시 에피소드가 저장됐다. 실행기는 TUNE 실행 뒤 비교 요약을 만들 때 이전 LR 실험용 키 `(seed, 2e-6)`를 조회해 `KeyError: (8104, 2e-06)`로 종료했다. 따라서 runner 상태는 `failed`지만 이는 요약 생성 단계 오류이며, 완료된 학습·TUNE 기록은 각 `tune-results.json`에 남아 있다. 학습을 재시작하지 않고 이 원시 파일들을 기준으로 판정한다.

| PPO seed | recovery 보상 | TUNE 완주 | 완주 시간 | 최고 진행도 | 충돌 시작 | 최종 손상 | 이탈 |
|---|---:|---:|---:|---:|---:|---:|---|
| 8104 | 0 | 2/2 | 20.64초 | 1.000 | 각 2 | 0.4 | 없음 |
| 8104 | 6 | 0/2 | — | 0.753 | 각 2 | 0.6 | 각 1회 |
| 8105 | 0 | 0/2 | — | 0.753 | 각 3 | 0.6 | 각 1회 |
| 8105 | 6 | 0/2 | — | 0.753 | 각 2 | 0.4 | 각 1회 |

두 TUNE seed 레이블은 서로 다른 평가 맵이 아니라 동일한 `custom-track-haic-tune-20260922` 지오메트리를 반복한다. 완주 시간 20.64초는 PPO seed 8104의 두 레이블에서 동일했고, 모든 팔의 13초 미만 유효 완주는 0건이다. 계수 6은 seed 8104의 기존 2/2 완주를 없앴고 seed 8105도 구하지 못했으므로 기각한다. seed 8105에서는 충돌·손상을 조금 줄였지만 진행도와 완주를 바꾸지 못했다. 이 결과는 새 제출 후보가 아니며 SOTA와 제출 ZIP은 갱신하지 않는다.

#### 2026-09-25 11:20 KST PPO 학습량 연장 실험 등록

실패 원인은 둘로 나뉜다. 완주 실패는 공식 Track1에서 장애물이 보인 뒤 약 5개의 결정만 남은 상태에서 brake가 사라지고 gas가 다시 올라가 충돌하는 접근 문제, 그리고 충돌 뒤 같은 고위험 정지 화면에서 약 100회 정체하는 회복 문제로 재현된다. 저속은 별도 병목이다. 공식 Track2 기록은 평균 40.6m/s·최고 52.9m/s·25.52초이며, 같은 코스를 13초에 끝내려면 근사 평균 79.7m/s가 필요하다. 넓힌 페달 상한과 speed target 상승은 완주를 악화시켰고, 높은 policy-mean LR도 안전한 action 변화를 내지 못했다.

PPO reward만 반복 조정하는 경로는 obstacle brake 6→12, curve brake 0→6, recovery-clearance 0→6에서 실패했다. policy-mean LR 2e-6→1e-5도 동일 TUNE에서 완주를 2/4→0/4로 떨어뜨렸다. 따라서 다음에 분별할 단일 변수는 새 reward가 아니라 PPO on-policy 결정 수다. recovery-clearance 실행의 PPO seed 8105 / coefficient 0 / U8 actor와 저장된 Adam 상태(SHA-256 `FF574E7CE75EEA3D473E078B04809EF7A87F9B03B9E43C717F45EFFB8BE3F9E3`)에서 같은 PPO 설정·보상·4 TRAIN / 2 TUNE·seed·episode cap을 고정하고 추가 32,768 decision / 32 update를 이어간다. TUNE 선택은 학습 중 끄고 8,192 decision마다 actor-only snapshot과 동일 TRAIN 관측 bank action을 기록한다. TUNE 결과가 더 많은 학습으로 completion·damage·progress를 개선하지 않으면 동일 설정의 budget continuation은 멈추고 policy representation 또는 실제 회복 상태 TRAIN 노출을 바꾼다. held-out·official은 이 화면에 열지 않는다.

실행 프로토콜: `docs/superpowers/plans/2026-09-25-ppo-budget-continuation-screen.md`; 원시 산출물 경로: `artifacts/haic/ppo-budget-continuation-seed8105-u32-20260925T112029KST/`. 현재 SOTA 기록의 선택 actor는 해당 8-update 학습 실행 중 TUNE 순위가 가장 높았던 update 4 checkpoint이며, update 8 정책과 동일한 파일이 아니다. SOTA 포인터와 제출 ZIP은 바꾸지 않는다.

#### PPO budget continuation 실행: 재현 TUNE 기준선 교정

실제 continuation을 시작하기 전 동일한 seed8105 U8 actor를 고정 TUNE 두 라벨에서 다시 실행했다. 두 라벨은 같은 geometry일 뿐 아니라 모든 행동·상태 기록도 동일하다. 두 에피소드 모두 decision 194에서 첫 충돌을 일으키고 decision 198에서 progress .757, damage 1.0으로 끝났으며 완주하지 못했다. 충돌 전 장애물 신호는 decision 189에 처음 보였고, urgency는 d190 .47, d191 .82, d192 1.0, d193 1.0으로 커졌다. d189–193의 gas는 .051–.078, brake는 전부 0이었다. 따라서 이 actor/지도에서 우선 실패는 저속 정체 후 회복 실패가 아니라, 마지막 장애물에서 5번의 decision 안에 지속적인 제동을 내지 못해 damage 한계에 도달하는 것이다. 각 에피소드의 종료 전 최저 속도는 3.8m/s였고, `high_risk_stall_decisions`는 0이어서 이 TUNE 기록에는 회복 shaping이 작동할 저속·비충돌 회복 구간이 없다. 앞서 언급한 100-step stall은 다른 실험 조건의 관측이며 이 actor와 TUNE 에피소드에 일반화하지 않는다.

비장애물 직선 TUNE 구간 220 decision에서 gas 평균은 .0796/상한 .12, 최대 속도는 49.1m/s였고 urgency≥.8 관측 48개에서 brake 평균은 .0180, gas 평균은 .0266이었다. 이는 속도 개선 목표가 낮은 가속 사용과 함께 남아 있으나, 같은 actor가 장애물 고위험 구간에서 먼저 완주하지 못하는 상태이므로 지금 가속을 밀어 올리면 위험을 더 키울 수 있음을 보여준다. 고정 TUNE replay와 source manifest가 통과해 추가 32,768 PPO decision 실험이 실행 중이다. 이 screen은 원인 분리용이며 TUNE 두 라벨은 독립 맵 평가가 아니다.

실패 위치의 반복성은 회복 보상만으로 해결되지 않는 고정점 가설과 맞는다. seed 8104/계수 6의 첫 충돌은 진행도 0.634 부근에서 발생했다. seed 8105의 네 조건은 모두 진행도 0.753에서 멈춘 뒤 고위험 장애물 신호가 유지되는 동안 거의 정지 속도, 가스 약 0.08, 브레이크 0으로 반복하다 이탈했다. 다음 회복 실험을 설계하기 전, 결정적 actor가 같은 정지 이미지에서 같은 입력을 되풀이하는지 원시 trace와 fixed-image replay로 확인하고, 회복 트리거가 실제 PPO rollout에서 몇 번 발생했는지 계측한다. 이 확인 전에는 reward 크기만 다시 바꾸지 않는다.

#### 실패 국면 재대조 및 학습 속도 계측

원시 기록을 다시 대조해 seed 8105의 실패를 두 국면으로 나눴다. 현재 이어 학습 실험의 coefficient 0 U8 시작 actor(SHA-256 `FF574E7CE75EEA3D473E078B04809EF7A87F9B03B9E43C717F45EFFB8BE3F9E3`)는 d189–193의 짧은 장애물 경고 구간에서 brake 0을 유지하고 d194에 충돌, d198에 damage 1.0으로 종료한다. 이 실행은 고위험 저속 정체에 도달하지 않았다. 반면 같은 이전 recovery-clearance 실행의 coefficient 6 actor(SHA-256 `370CB15703F96BEA96204EC72B81D02FB429758ECAB4212FE7692B496EACD73D`)는 d168에서 먼저 접촉한 뒤 d197부터 progress .753에서 102회 정체하고 이탈했다. 그러므로 coefficient 6의 정체 원인을 지금 시작 actor의 최초 완주 실패 원인으로 혼동하지 않는다. 전자는 충돌 전 제동 반응, 후자는 충돌 뒤 탈출을 판별한다.

같은 coefficient 0 시작 체크포인트의 이전 TUNE trace와 이번 continuation의 `tune-before.json`도 행동 궤적이 완전히 같지 않다. 초반 차이는 매우 작았지만 d127 전후에 action/state 차이가 1e-4를 넘어 이후 충돌·정체 경로가 갈라졌다. 이 작은 차이의 원인은 아직 확인되지 않았다. 따라서 이 비교에서는 “actor 체크포인트와 seed가 같다”만으로 simulator 재생이 완전히 같다고 가정하지 않고, 이번 continuation의 저장된 same-run TUNE 기준선과 후속 결과를 우선 대응시킨다. 두 TUNE seed 표기는 같은 맵 geometry를 반복하므로 독립 맵 표본으로 세지 않는다.

학습시간 원인도 기존 원시 timing으로 확인했다. 동일 8,192-decision/U8 PPO arm은 rollout 453.7초, PPO update 239.0초, 총 692.7초였다. rollout은 CPU에서 환경을 순차 실행하며 매 결정마다 관측과 다음 관측의 actor forward를 하고, simulator tick 4회를 진행한다. 이번 continuation의 첫 8,192-step actor snapshot은 시작 후 약 700초에 생성됐으며 현재 프로세스는 다음 구간을 학습 중이다. 따라서 첫 snapshot 대기는 멈춤이 아니라 이전 동일 실행 속도와 맞는다. 32,768-step continuation의 나머지 구간과 최종 TUNE 비교가 끝날 때까지 추가 학습은 겹쳐 실행하지 않는다.

현재 continuation은 학습량만 늘리는 단일 변수 screen이다. 결과가 baseline 대비 개선되지 않고 d190–193 제동 부재 및 d194 손상 종료가 재현될 때만 actor recurrence를 다음 가설로 검토한다. 그 실험은 짧은 시각적 위험 cue의 시간 누적을 actor가 활용하는지를 묻고, 현재 FF actor의 PPO budget continuation과는 다른 변수다. recurrent 처리도 PPO actor-only·픽셀 관측만 사용하며, teacher/privileged state를 제출 추론에 넣지 않는다.

#### 직선 속도 병목 진단: 조건부 PPO 가설 대기

speed 분석에서 current U8 TUNE trace를 같은 픽셀 조건으로 재집계했다. 더 엄격한 적격 조건(`visible_hazard_risk < 0.1`, `road_curve_magnitude < 0.1`)은 반복 trace 합계 160행, 즉 회당 80행을 남긴다. 여기서 평균 gas는 0.0862/상한 0.12, 평균 속도 41.58m/s, 최고 속도 48.87m/s였다. 기존 속도 shortfall은 전 구간에 걸쳐 적용되지만 이 저위험 직선에서 별도 고속 목표를 주지 않는다. 앞서 target 84 같은 전 구간 변경은 이미 안전/완주를 악화시켰으므로 같은 변경을 반복하지 않는다.

대기 중인 PPO 가설은 `α=0` 대 `α=0.6`의 저위험 직선 전용 속도 비용이다: `0.1 × α × I[risk<0.1 AND |curve|<0.1] × max(80-speed, 0)/80`. 현 trace에 α=0.6을 대입하면 적격 결정당 평균 추가 비용은 약 −0.0288로, 기존 곡률 완화 속도 비용 −0.0236과 비슷한 크기다. 적격 gate와 속도 라벨은 학습 중 보상 계산에만 쓰고, inference는 deterministic PPO actor-only로 유지하는 설계다. 다만 현재 continuation이 충돌/완주를 개선하지 못한 경우에는 이 실험을 먼저 실행하지 않고 cue 구간 제동 actor 가설을 우선한다. 이 후보는 아직 trainer에 추가하지 않았고 학습 결과도 없다.

같은 route 길이 추정에 따르면 13초 완주에는 평균 약 79.7m/s가 필요해 현재 관측한 48.87m/s보다 훨씬 높다. 뒤이은 local fixed-gas probe에서 gas .42로 3.2초 시점 80.04m/s에 도달했고 네 sampled track의 peak는 90.7–100m/s였다. 따라서 80m/s가 local 물리상한 때문에 불가능한 값은 아니라는 근거가 생겼다. 다만 조향 없는 opening straight probe라서 full-lap 달성을 증명하지 않는다. 현재 U32 continuation은 저속 보상 실험의 안전 gate를 판단하기 전 단계다.

기존 official throttle-cap 원시 sweep도 대조했다. 별도 예전 checkpoint로 Track1/Track2 두 episode를 평가할 때 gas ceiling `0.12`→`1.00`은 평균속도 41.27→42.33m/s, 최고속도 55.28→55.65m/s, median 완주시간 24.24→23.72초에 그쳤다. 그 sweep의 최고속도는 cap `.42`/brake `.98` arm에서 56.11m/s였다. 이는 이 actor와 두 경로에서 상한만 늘리는 것은 주 병목이 아니었음을 보여준다. checkpoint가 달라 현재 actor에 대한 확정은 아니며, 다음 pace screen은 pedal cap이 아닌 PPO가 같은 안전 직선에서 고른 action을 바꾸는지를 측정한다.

진행 갱신: 동일 PPO continuation에서 추가 16,384 decisions/U16 actor snapshot이 2026-09-25 11:47:16 KST에 생성됐다. PID 54040은 계속 CPU를 사용하며 U32까지 학습 중이고 TUNE 평가는 아직 보류 상태다.

진행 갱신: 2026-09-25 11:59:06 KST에 추가 24,576 decisions/U24 snapshot이 생성됐다. PID 54040은 U32 마지막 구간을 학습 중이며 TUNE은 여전히 한 번도 열지 않았다.

#### 직선 속도 물리 가능성 확인

기존 local fixed-gas 원시 probe(`artifacts/haic/lap-time-ceiling/lap-time-ceiling.json`)에서 gas .42는 조향 없는 40 decision/3.2초에 80.04m/s에 도달했고, 네 공식 geometry의 peak는 90.7–100m/s였다. 차량 코드에는 명시적 최고속도 clamp나 공기저항 항이 없고 simulator는 50Hz로 진행한다. 샘플한 track의 길이를 13초로 나눈 필요 평균속도는 74.0–92.1m/s다. 그러므로 80m/s는 local 물리상한 때문에 불가능한 값은 아니지만, 직선 probe가 한 속도를 순간 달성했다는 사실은 곡선과 장애물을 포함한 13초 완주 가능성을 입증하지 않는다. Box2D 50Hz × 2 unit/step에서 100m/s가 되는 설명은 probe 수치와 일치하는 강한 추론이며, native Box2D 버전이 평가 환경과 완전히 같다는 근거는 별도로 필요하다.

정책의 페달 상한 sweep는 물리 한계와 제어 정책 사용량을 구분해 준다. 기존 SOTA의 cap .12→.24 zero-shot 변경은 TUNE 직선 gas .0844→.0984, 속도 44.05→45.08m/s, lap 19.88→19.50초로 조금 빨라졌지만 충돌은 2→6으로 증가했다. 더 넓은 상한 sweep도 평균속도 41–42, peak 55–56m/s에 머물렀다. 즉 약 80m/s를 안전하게 유지하는 정책이 아직 학습되지 않았으며, cap만 확장하는 것은 충분한 해결이 아니었다. 이에 따라 pace reward gate는 위험도와 곡률이 모두 낮은 픽셀 관측에서만 적용하고, 완주·damage가 먼저 좋아지지 않으면 실행을 보류한다.

#### 조건부 recurrent PPO 설계 점검

현재 actor는 관측마다 독립적으로 호출되는 feed-forward network다. `RolloutStorage`는 episode 종료 표식은 저장하지만 hidden state를 저장하지 않고, PPO updater는 개별 transition을 섞으므로 GRU만 network에 덧붙여서는 시간 학습이 되지 않는다. recurrent actor를 선택할 경우 최소 구현은 actor 경로의 64-unit GRU와 2차원 zero-initialized action residual, rollout hidden-state/episode-start 저장, episode 경계를 넘지 않는 8-decision sequence PPO/BPTT, deterministic runtime episode reset이다. critic과 auxiliary head는 기존 feed-forward 경로를 유지한다. 입력은 허용된 픽셀 latent뿐이고 teacher/privileged 정보는 넣지 않는다.

U32 결과에 d189–193 제동 부재와 d194 damage-cap 충돌이 남는 경우에만 recurrent treatment를 검토한다. 그때 대조는 같은 U8 checkpoint에서 출발한 현재 feed-forward U32 결과이며 recurrent actor도 32,768 decision budget, seed8105, 동일 TRAIN/TUNE, reward/LR/action scale을 맞춘다. 기전 신호는 반복 cue 구간에서 늦어도 d191까지 brake가 0.02를 넘고 충돌·손상 없이 진행이 늘어나는지다. brake onset이 늦거나 damage-cap 종료가 유지되면 기각한다. 이것은 여러 학습/API 경로를 건드리는 architecture 변화라 현재는 설계 검토만 했고 코드 수정이나 실행은 하지 않았다.

#### PPO budget continuation U32 최종 결과 및 전략 재배치 (2026-09-25)

동일 seed8105 coefficient-0 U8 actor와 Adam 상태에서 추가 32,768 decision/32 update를 완료했다. 학습은 2,801.5초(rollout 1,852.4초, PPO update 948.8초) 걸렸다. U8 직후 같은 TUNE geometry 반복 기준은 0/2 완주, 평균 progress .7572, final damage 1.0, collision onset 2였다. U32 뒤도 0/2 완주였고 progress .7531, final damage 1.0, collision onset 4로 바뀌었다. valid-under-13은 전후 모두 0이다. U32 trace는 기준 궤적과 달라져 첫 충돌이 d194에서 d164로 앞당겨졌고, 두 라벨은 하나의 geometry 반복이므로 독립 지도 표본이 아니다.

같은 1,040장 TRAIN 관측 replay에서 clear-straight gas는 .07959→.07982로 사실상 그대로였다. urgent-visible brake 평균도 .01801→.01826에 그쳤고, brake>.02 비율은 U8 .432에서 U32 .405로 낮아졌다. 중간 snapshot의 비율은 U16 .459, U24 .405, U32 .405였으며, U32 actor parameter drift는 .01258까지 커졌다. 즉 파라미터는 움직였지만 장애물 반응이나 완주가 개선되지 않았다. 더 긴 동일 정책 학습만 반복하는 가설은 기각하며 동일 budget continuation을 추가로 실행하지 않는다.

이 결과는 새 제출 후보가 아니다. source artifact와 custom TRAIN/TUNE만 확인했고 held-out/official 평가는 하지 않았다. 기존 시작 checkpoint의 과거 학습 provenance는 별도 제한으로 남아 있으므로 restriction 상태를 `unknown`으로 기록한다. SOTA 포인터와 ZIP은 유지한다. 후속 레인은 세 가지로 분리한다: A는 PPO on-policy에서 위험 접근 초기 상태를 더 자주 경험시키는 TRAIN-only hazard-start curriculum의 구현 가능성을 확인하고, B는 고정 충돌 벌점 대신 적응 collision budget을 쓰는 Lagrangian PPO를 설계하며, C는 허용된 observation history만 쓰는 recurrent actor를 검토한다. PPO 학습은 한 프로세스만 허용하고, 다른 레인은 학습을 시작하지 않고 순서를 기다린다.

A의 최초 teacher-warm-up 제안은 기존 결과와 중복되어 취소했다. clean-start `fresh actor → TRAIN-only teacher BC → PPO` attempt-2는 U8 Tune 0/8, under-13 0이었다. 더 구체적으로 seed8105 teacher-BC actor는 PPO 전 동일 Tune geometry에서 18.96초에 완주했지만 그 actor에서 출발한 PPO U8 정책은 두 Tune reset 모두 off-track으로 끝났다. 따라서 다음 구별 가능한 변수는 teacher 재학습이 아니라 PPO rollout의 초기 상태 분포다. A는 site TRAIN map에서 hazard-proximal start를 만들 수 있는지와 그 분포 아래 PPO rollout/advantage가 올바르게 계산되는지 확인하도록 재배정했다.

초기 hazard start 구현의 첫 후보인 `geometry.start_index` 변경은 단독으로 사용할 수 없다. `SiteCustomCarRacing._ordered_centerline()`이 centerline을 start index부터 회전하고, `site_obstacle_specs()`가 회전된 track에 원래 normalized `obstacle.progress`를 다시 적용하므로 장애물의 상대 출현 위치도 함께 유지된다. 현재 `CarRacing.reset()`은 차를 `track[0]`에 배치하고 `CarEnvironment.reset()`은 첫 actor 관측 전 50 raw no-op ticks를 수행한다. 따라서 유효한 start-state curriculum은 map centerline·obstacle placement·finish line을 그대로 두고, custom TRAIN reset에서 차량 pose만 hazard-proximal track index로 바꾸는 별도 옵션이 필요할 수 있다. visited-tile/finish qualification, damage reset, no-op warmup 뒤 hazard 거리까지 보존되는지 확인하기 전에는 이 가설을 실행 준비 완료로 보지 않는다.

세 사이트 map JSON을 별도로 대조하면 obstacle progress grid는 TRAIN 두 geometry와 TUNE geometry 모두 `[.25,.37,.50,.63,.75]`로 같다. progress `.75` lateral도 TRAIN 두 번째 map과 TUNE에서 모두 `+.22`이고, `.63` lateral은 TRAIN `-.33/-.37`, TUNE `-.45`다. 그러므로 후기 장애물 위치 자체가 학습 지도에서 빠졌다고 볼 근거는 없다. 현재 U32 PPO rollout의 raw cue 노출은 저장되지 않아 노출 부족인지 정책의 cue-to-action 학습 실패인지는 미결이다. SOTA actor의 고정 observation bank를 U32 on-policy 노출로 간주하지 않으며, 실제 TRAIN rollout 노출이 부족하다는 증거가 생길 때만 hazard-start curriculum 학습을 승인한다.

#### 충돌 예산 PPO 비용 라벨 정의 점검 (2026-09-25)

B의 source 조사와 root의 원시 재확인에서 현재 안전 신호가 onset 횟수와 같지 않음이 확인됐다. `CarEnvironment.step()`은 4 raw tick의 충돌 여부를 OR해 한 decision의 `collision`으로 내보내고, 그 decision마다 `CollisionDamage.update()`가 damage를 0.2씩 누적한다. `shape_transition_reward()`는 collision=true decision마다 `-60 × 0.1 = -6`을 이미 주고 damage 비례 감점도 추가한다. damage cap에 닿는 terminal failure에는 별도 failure penalty도 걸린다.

U32 TUNE raw trace의 두 반복 episode는 동일했다. 각 episode에서 첫 충돌 cluster는 d164–166의 collision-bearing decision 3개/ onset 1개, 다음 cluster는 d191–192의 decision 2개/ onset 1개였다. 따라서 episode당 onset은 2지만 damage-bearing decision은 5개, damage는 1.0까지 누적됐다. 특히 두 번째 접근 d188–190에는 urgency .81→1, 속도 44.15→44.59m/s, gas .07→.05, brake 0이었고 d191에 접촉했다. 첫 접근은 d162–163에 brake .09/.08을 썼지만 d164부터 세 차례 접촉했다. 이는 모든 장애물에서 브레이크가 항상 0이라는 설명보다, 빠른 후반 위험 접근에서 제동·회피 행동이 충분히 학습되지 않았다는 설명을 지지한다.

Lagrangian 비교를 계속 설계한다면 cost는 onset count보다 `max(0, damage_t - damage_{t-1}) / 0.2` 누적으로 두어 실제 termination을 만든 손상 단위와 맞춘다. 고정 cost와 adaptive multiplier arm의 차이는 λ 갱신만 남겨야 하며, 기존 collision reward penalty·damage penalty·terminal failure penalty를 cost channel과 함께 어떻게 분리할지 사전 지정해야 한다. 이 가설은 충돌 budget을 맞추는 학습 목적이며 cue에 대한 새 시각 정보나 빠른 제동 반응을 직접 제공하지는 않는다. 그래서 현재는 설계 screen이고 학습을 시작하지 않았다.

원시 근거: `artifacts/haic/ppo-budget-continuation-seed8105-u32-20260925T112029KST/{tune-before.json,tune-after.json,training-result.json,learning-diagnostics.json,execution-status.json}`.

#### 추가 병렬 조사 결과: hazard 노출 계측, 충돌 예산 screen, rollout 처리량

**A — hazard-start curriculum은 계측 전까지 no-go.** 현재 reset API는 차체만 위험 지점으로 옮기는 안전한 선택지를 제공하지 않는다. custom obstacle은 wrapper의 50 no-op 및 첫 관측 준비 뒤 붙기 때문에 단순 pose 이동은 첫 관측과 visited-tile/finish 상태까지 다룰 수 없다. 더구나 U32의 on-policy hazard 노출 요약도 저장하지 않았다. 따라서 SOTA 고정 이미지 bank의 위험 비율로 hazard-start를 정당화하지 않는다. 먼저 TRAIN collector에서 실제 pre-action 픽셀을 동일한 이미지 전용 detector로 분류하고, map/progress bin별 hazard·urgency·risk 노출량, gas/brake, 다음 충돌을 PPO update별로 요약해야 한다. 이 계측은 관측/행동/보상을 바꾸지 않고 TRAIN만 다룬다.

**B — 고정/적응 Lagrangian screen 제안.** SOTA TRAIN 고정 bank의 urgent-visible 관측은 1,040장 중 74장(7.12%)이며, 8% quota와 비교하면 수량상 10장 부족하다. 하지만 반복 route의 픽셀 표본이므로 독립 위험상황 74개를 의미하지 않는다. 같은 74장 replay에서 U8→U32 brake-active는 32/74(43.2%)에서 30/74(40.5%), 평균 brake .02218→.01930으로 오히려 낮아졌다. 이는 U32 on-policy 통계가 아니다.

제안된 단일변수 paired screen은 cost를 damage 증가분 `c_t=max(D_t-D_(t-1),0)`로 두고 `C_episode=D_final`로 집계한다. 두 arm 모두 explicit collision/damage/failure penalties를 cost channel로 옮기고 같은 cost critic을 추가한다. fixed arm은 λ=30, adaptive arm은 시작 λ=30에서 8개 완전 TRAIN episode마다 `λ←clip[0,60](λ+5(C̄−0.05))`로 갱신한다. arm당 seed8104/8105 및 32개 완전 TRAIN episode를 제안했다. Collector 경계에서 잘린 episode의 cost를 multiplier 갱신에 섞지 않고 누적하는 처리가 필요하다. 이 screen은 위험 예산을 조절할 뿐 시각 cue에 대한 조기 제동이나 회피 방향을 직접 가르치지 않는다. 제안은 source/design 수준이며 구현·학습·성능 근거가 아니다.

**처리량 조사 — rollout actor forward 중복이 우선 후보.** U32 32,768 decision은 2,801.5초(rollout 1,852.4초, PPO update 948.8초)였다. rollout은 전체의 66.1%다. 매 decision의 next-value 계산은 바로 다음 action decision에서 다시 쓸 동일 관측에 full actor/value forward를 수행한다. 현재 모델은 rollout 동안 고정되고 dropout/BatchNorm도 없어, 다음 관측의 `PolicyOutput` 및 mutable `last_visual_features`를 함께 캐싱하면 같은 rollout의 연속 forward 하나를 재사용할 수 있다는 제안이다. 캐시는 episode/rollout 경계와 PPO update에서 폐기해야 하며 terminal/truncation의 GAE bootstrap 규칙을 유지해야 한다. 호출 수는 대략 `2N`에서 `N+R`(N decisions, R 연속 구간)으로 줄 가능성이 있으나, 실제 forward 비중과 초 단위 속도 향상은 아직 측정되지 않았다. 구현·동등성 확인·benchmark는 미실행이다.

#### 작업 라우팅 갱신

A의 다음 조건부 단계는 실제 TRAIN on-policy 노출 계측이다. 노출 편향이 확인되지 않으면 hazard-start PPO를 시작하지 않고 cue-to-action 쪽으로 이동한다. B의 Lagrangian screen은 별도 cost-critic 전략으로 남긴다. C에는 raw physics action-repeat 4→2 ablation 설계를 맡겼으며, 4/2 비교에서 원시 physics tick 예산과 PPO update 예산 중 무엇을 고정할지까지 명시하게 했다. 처리량 개선은 이 세 정책 변수 실험과 분리한다. 현재 이 기록의 새 내용은 source/design evidence만이며 raw 성능 결과, SOTA 승격, 제출 후보는 없다.

#### 2026-09-25 현재 소유권 및 실제 진행 상태

| Owner | Task ID | Assigned work | Latest verified state |
|---|---|---|---|
| A | `01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1` | TRAIN on-policy hazard exposure schema and curriculum go/no-go | Active task turn; source finding delivered, follow-up schema pending |
| B | `01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de` | Lagrangian fixed/adaptive λ implementation-ready experiment plan | Active task turn; design delivered, exact plan pending |
| C | `01a0cec2-4799-7303-aba2-baa80f015189` | Raw action-repeat 4→2 ablation design | Active task turn; no result message yet |
| Speed branch | `/root/ppo_speed_strategy` | TDD next-value output/visual-feature cache and equivalence/collection benchmark | Agent active; it confirmed no overlapping collector writer and is starting with RED tests |
| Speed-objective branch | `/root/ppo_recovery_strategy` | Audit gated low-risk straight-speed PPO reward against authoritative raw artifacts | Agent active; output pending |

The Codex task flags for A/B/C are active, but their current turns had no assistant message or tool marker at this snapshot; that is not evidence of completed work. The two internal analysis/implementation agents are live. A process query found no `train_policy.py`, PPO `run_experiment`, or other Python PPO training process. Therefore current work is parallel analysis/implementation preparation, not live RL training. Start only one PPO training process after the cache patch and experiment plans are ready, and preserve each branch's distinct hypothesis.

Routing update after A's return: the exposure schema is delivered. A's implementation task is queued behind the speed cache completion because both touch `collect_rollout`; it must add only aggregate TRAIN diagnostics, preserve actions/rewards/GAE, and avoid PPO training. The 8,192-decision sample-only rollout is conditional on whether the existing collector can do it without an optimizer update; no such run has started.

#### Cache patch verification update (2026-09-25)

The speed branch reports a test-first cache implementation in `training/train_policy.py` with focused collection-equivalence cases in `tests/test_train_policy.py`. The full suite completed with **202 passed, 1 skipped**. Cases cover reuse of next-state policy output and visual features, terminal reset invalidation, rollout/update boundary invalidation, uncached reference equivalence for observations/actions/log-probability/value/reward/return. An on-policy training run has not started. A controlled cache-on/off collection timing comparison is pending on the U32 checkpoint, TRAIN split, seed 8105, 1,024 decisions, with no PPO optimizer update or TUNE access. These tests verify implementation behavior; they do not establish faster wall time or improved lap performance.

#### Cache collection benchmark: provisional first arm

The speed branch reports an uncached→cached sequential collection comparison on 1,024 TRAIN decisions with exact output/state/reward diff empty: 60.5705 s and 2,048 forward calls versus 58.3709 s and 1,027 calls (1.038×). Treat this timing as provisional because another short-lived project `.venv` Python process appeared during monitoring; it may have overlapped at some point. No PPO optimizer update or TUNE was run. A clean serial rerun is pending after a process check showed no project `.venv` Python process active. Do not claim the 3.8% observed difference as a validated speedup yet.

Owner state correction: the C action-repeat design follow-up completed with no message or tool artifact, so there is no C finding to compare yet; do not treat its `idle` state as an action-repeat result. A returned its read-only collection feasibility finding: the U32 actor can be loaded into `collect_rollout()` for sample-only 1,024-decision chunks, discarding storage and omitting PPOUpdater.update(). Eight chunks can produce 8,192 TRAIN decisions with fixed actor weights. The normal manifest loader must not be used because it loads TUNE/held-out entries; construct only registered TRAIN episodes and record that omitting PPO updates means a fresh stochastic rollout, not a replay of U32's original trajectory. A has been asked to keep waiting on the cache patch before touching the collector.

Benchmark protocol correction: the speed branch confirmed that both observed timing attempts used `load_site_map_split()`, which reads TRAIN, TUNE, and held-out map entries from the manifest. Although only `split.train` was passed to collection and no PPO update/TUNE rollout occurred, these attempts are invalid for the strict TRAIN-only benchmark and are excluded from performance claims. The next run will parse only the registered `train` manifest group and open only its map files; no TUNE/held-out path may be loaded.

B's plan gate is now executable without a historical checkpoint: initialize `VisualActorCritic` from recorded per-seed random initialization with no parent, no teacher warmup/demonstrations, and no restored optimizer. The fixed/adaptive arms must share byte-identical initial model hashes within each seed. This resolves the inherited-checkpoint provenance gap without returning to teacher warmup.

#### 2026-09-25 coordinator correction: work was not actually parallel

The latest compact status check showed A still in progress but with no new message/tool marker, while B and C had completed their latest turns with no output. Only the two internal branches were actively producing work. The previous ownership table incorrectly treated task `active` flags as meaningful progress. This was an orchestration failure; no PPO training process was active at the check.

The speed branch has now completed a valid strict TRAIN-only cache benchmark. It directly loaded only the four registered TRAIN map/seed cells and used the U32 continuation checkpoint, seed 8105, 1,024 decisions, and no PPO update or evaluation. Uncached collection took 64.051 s and 2,048 policy forwards; cached collection took 57.519 s and 1,027 forwards, a 10.20% reduction in collection time. Both full 1,024-step trace hashes were identical (`f386eaa461fe9a14d486173fa3a7f48e067ad573b7df281d7228f3cf213bda0f`). Focused and full tests passed: 202 passed, 1 skipped, 30 subtests. This is rollout throughput evidence, not lap-performance or submission evidence.

The recovery-objective audit returned a no-go for the gated straight-speed reward until a safer actor exists. On the same U32 continuation seed, repeated TUNE rows on one geometry had 0/2 finishes before and after; first collision moved earlier from decision 194 to 164. The straight-state TRAIN sample did not show throttle saturation. This does not justify adding speed pressure now and is not independent-map evidence.

After the cache dependency cleared, the existing strategy tasks were reactivated with concrete, non-overlapping deliverables: A implements aggregate TRAIN-only hazard-exposure logging in `collect_rollout()` without changing observations/actions/rewards/GAE; B implements isolated Lagrangian cost accumulation and multiplier helpers outside `train_policy.py` while the collector change is underway; C implements a TRAIN-only `frame_skip` 4→2 override in its isolated worktree and prepares a paired manifest with equal raw physics-tick budget. All three are instructed to avoid long PPO training until the implementation and preflight gates are complete. Current work is finally assigned across the three strategies, but no outcome from these new turns is available yet.

#### Raw failure trace audit: visible hazard response is late and may steer toward the obstacle

Read-only analysis of `ppo-budget-continuation-seed8105-u32-20260925T112029KST/{tune-before.json,tune-after.json}` adds a sharper failure hypothesis. The two TUNE episode rows per arm are exact repeats on one geometry; the effective independent scenario count is one, not two. Before continuation, the actor first marked an obstacle at decision 189 (urgency 0.11), reached urgency 1.0 at decision 192, and collided at 194. It commanded zero brake throughout that approach. After continuation, it detected an obstacle at decision 158, reached urgency 1.0 at 161, and collided at 164. In that approach the signed obstacle offset was left/negative (`-0.15` to `-0.23`) while PPO steering was strongly negative (`-0.36` to `-0.44`); brake was only `0.02` to `0.087` in normalized trace units. Corridor-controller tests use positive steering to pass a left-side obstacle, so this pattern is consistent with steering toward the obstacle, though the camera-to-vehicle sign and simultaneous bend correction still need image-level verification.

This points to a cue-to-action credit problem more directly than an absence of detection. The PPO reward currently gives obstacle brake reward and penalizes gas under hazard risk, but its positive clearance reward is restricted to the post-damage, low-speed recovery state. It does not reward the pre-impact action that increases signed obstacle clearance. A was sent this evidence and asked to add TRAIN-only aggregate signed-offset/steer-alignment counts by urgency bin, then correlate the next progress bin with collision; no TUNE trace is to enter training or detector tuning.

#### Speed ceiling and training-time decomposition

At the requested 13-second target, the registered official route lengths in the existing ceiling probe require mean speeds of approximately 74–92 m/s (962.5–1,197 m). The current PPO actor's unexpanded throttle branch is capped at gas 0.12. The fixed-gas ceiling probe at 0.12 peaked at about 59 m/s on Track 1 seed 42 and about 71 m/s on the longer samples; the 0.42 probe reached about 80 m/s after 40 decisions. This is not a full-lap proof, but it shows the present action envelope is below the target on several tracks before accounting for turns or obstacles. The next pace hypothesis should expand the actor's throttle range and train PPO to use it, paired against the current range while keeping the brake range fixed; a whole-track run must still reject any gain that reduces completion or raises damage.

The U32 run took 2,801.5 s for 32,768 decisions. Rollout consumed 1,852.4 s (66.1%) and PPO updates 948.8 s. The validated cache speedup was 10.2% on collection; applying that factor only to the measured rollout portion predicts about 6.8% less total training time, an extrapolation rather than a full-run measurement. Thus duplicate actor forwards were real but explain only part of the slow training. The throttle envelope is a separate likely cause of slow laps.

#### 2026-09-25 implementation progress and current gates

**B Lagrangian helper implemented; trainer integration remains pending.** `training/lagrangian.py` and `tests/test_lagrangian.py` now implement transition cost components, carry/discard of collector fragments, completed-episode-only multiplier updates, and fixed/adaptive λ windows. B reports 15 focused tests passed. Its source audit confirms current/next damage labels and collision aggregation semantics; however, raw failure reward `-100` still remains in the PPO reward and the cost critic is not wired into the trainer. Therefore this is an implemented helper, not an executed Lagrangian PPO strategy and not a performance result. Keep B's integration away from `collect_rollout()` until A finishes its collector change.

**A hazard recorder is being integrated.** The shared checkout now contains `training/hazard_exposure.py` and `tests/test_hazard_exposure.py`, but its collector integration is incomplete. A source read found an undefined `pre_action_labels` reference in the recorder call and a conditional-mean denominator error in `risk_positive_mean`; both were reported to A along with the request for signed offset/steer alignment. Do not start a sample-only TRAIN rollout or use this recorder until A fixes those issues and its focused tests pass.

Correction after A's in-progress edits: the current files now use `transition.observation_labels` safely, accumulate positive-risk mass separately, and include urgency-binned steering-alignment and collision-follow-up aggregates. I also found that its first alignment sign used obstacle offset from road center without compensating for the car's offset from center; A has been asked to compute the signed obstacle position relative to the car using the same pixel-coordinate equation as `visible_hazard_risk()` and to test off-center cases. Tests and a final turn report are still pending, so no fixed-actor rollout has begun.

**C action-repeat work has not produced a verified artifact yet.** Its thread is active, but a read-only scan of the isolated worktree did not find the requested `frame_skip` override. Keep its status as pending, not complete.

**New PPO throttle-range screen registered.** A distinct Luna task is being set up in an isolated HAIC worktree to compare throttle expansion 1.0 versus 3.5 (gas ceiling 0.12 versus 0.42), with brake expansion fixed at 1.0, identical PPO actor initialization, fresh Adam, TRAIN-only learning, and deferred TUNE measurement. Creation returned a setup-pending client ID `client-new-thread:4d540427-42d7-4336-adf2-c85d15bdb9de`; no actual training result or ready thread ID exists yet. The run gate also requires no concurrent PPO training process.

**C action-repeat implementation is now visible in an isolated worktree.** C created branch `codex/action-repeat-2tick` at `C:\Users\koi\.codex\worktrees\action-repeat-2tick\HAIC`, added `training/action_repeat.py`, TRAIN override plumbing, a paired-manifest template, and `tests/test_action_repeat.py`. C reports 7 focused tests, `py_compile`, and JSON parsing passed. No PPO learning/evaluation has run. The manifest template still lacked actual map/seed and raw physics budget when first reported; I supplied the registered four TRAIN map/seed cells and specified 32,768 physics ticks per seed/arm: skip-4 gets 8,192 decisions/8 updates; skip-2 gets 16,384 decisions/16 updates with the same 1,024-decision update batch. Evaluation remains at the standard map skip of 4 after training. Preflight must confirm override affects TRAIN only.

B has resumed the Lagrangian branch on the next non-overlapping component: cost-value/GAE and `A_reward - λ*A_cost` PPO update support outside the files A is editing. It is explicitly gated from PPO training and TUNE until trainer integration and preflight are complete.

A's latest collector instrumentation passed the focused suite when run from the shared checkout: 5/5 tests, including TRAIN-manifest-only loading, cached pre-action pixel features, actual post-action outcomes, car-relative steering alignment, and neutral positions. I authorized the next fixed-policy sample-only collection: 8,192 decisions from the U32 actor over the four TRAIN map/seed cells, no optimizer/update, with aggregate JSON only. That measurement will tell us whether late-progress cue scarcity is the cause or whether the policy sees hazards but selects the wrong evasive action. No sample-only artifact exists yet.

### 2026-09-25 coordinator continuation: completed helper, failed speed-run logging, next evaluations

The current selected PPO actor still misses the requested pace and has a major official-track completion gap. `artifacts/haic/final-ppo-actor-selection.json` reports held-out completion 6/8 (0.75), median completed lap 19.32 s, P90 22.42 s, mean speed 40.41 m/s, and median per-episode peak speed 52.96 m/s. On official Track 1 seed 42 it did not finish and stopped at progress 0.1625 after two collisions; the added-obstacle diagnostic stopped at progress 0.1484 after five collisions. These records do not establish a sub-13-second lap.

A separate throttle-envelope PPO screen ran in worktree `C:/Users/koi/.codex/worktrees/0e2d/HAIC`. It saved the seed-8104 baseline and treatment checkpoints through 8,192 decisions, then failed while writing the treatment run record. `failure.json` identifies the exact cause: strict JSON rejected `VisualActorCritic.pedal_expansion = NaN`. `haic_agent/networks.py` uses that field intentionally as the legacy symmetric-scale sentinel when throttle and brake expansions differ; the branch-specific throttle/brake values remain finite. The second seed and all TUNE evaluation were not run. The failed artifact is preserved; its partial checkpoints are diagnostic material, not a performance result.

A recovery run has a separate immutable directory `artifacts/haic/throttle-envelope-ppo-2seed-u8-json-record-fix-20260925`. Its runner maps only the asymmetric legacy sentinel to JSON null, preserves the actual branch scales, and still rejects non-finite metrics. The focused regression tests passed 3/3. Its preflight passed against 42 frozen source files, four registered TRAIN map/seed cells, matched initial actor hashes across all arms, and zero held-out/official cells. No PPO training or TUNE evaluation has started in that recovery run.

A completed the pixel-only potential-shaping helper in `training/hazard_potential.py`; its focused tests pass 20/20. It remains an unintegrated reward helper, not a trained policy. B is now integrating the separate Lagrangian cost critic with the PPO trainer; C is working on fixed-actor frame-skip 4-vs-2 measurement in its isolated worktree. A has a separate read-only diagnostic on the two saved seed-8104 throttle checkpoints and a TRAIN-provenance observation bank. No result from those follow-ups is available yet.

Next run order: finish C's TRAIN-only action-repeat diagnostic; complete the fixed-observation throttle action analysis; then run the four-arm throttle PPO screen only after a fresh no-training-process check. Compare completion, under-13 valid finishes, collisions, damage, progress, and lap time; keep SOTA and the submission package unchanged unless independent evaluation improves.

### 2026-09-25 throttle-envelope screen: action-level evidence and live final arm

A's fixed-pixel diagnostic is recorded at `C:/Users/koi/.codex/worktrees/0e2d/HAIC/artifacts/haic/throttle-envelope-ppo-2seed-u8-20260925T135214KST/postfailure-throttle-action-diagnostic.json`. It used a provenance-checked TRAIN-only bank (1,040 images from two TRAIN maps and four TRAIN seeds), identical pixel tensors, deterministic actor means, and no simulator, TUNE, held-out, or PPO updates. On these inputs, seed8104 treatment gas mean rose from 0.06470 to 0.07920 (+22.4%), p90 from 0.08959 to 0.11307, with brake and steering means nearly unchanged. Neither arm reached 0.1176 (98% of the original 0.12 gas cap); treatment max was 0.11500. This proves a higher commanded gas distribution on this fixed bank, not higher vehicle speed, better lap time, or improved safety. The original cap was not saturated on these sampled frames, so the gain may come from the changed branch transform and must be judged in closed-loop episodes.

The same recovered paired PPO screen is still genuinely running: process 27548 remains alive, training the final `seed8105-treatment` arm. Its 2,048- and 5,120-step actor snapshots were written at 14:59:51 and 15:02:40 KST. `continuation-status.json` still says the last arm started at 14:57:59; therefore use the live process and checkpoint writes, not that stale status timestamp, to track progress. Three of four arms are complete; no TUNE result is available yet.

Next decision is gated on the same run reaching all four completed arms and exiting. Compare paired TUNE outcomes for completion, valid sub-13 s finishes, lap-time distribution, collisions, damage, progress, and actual speed; its two TUNE seeds share one geometry, so a promising result still needs a map-diverse held-out/official confirmation before SOTA or submission changes. If expanded throttle does not improve closed-loop speed without a safety regression, focus the next single-variable PPO screen on earlier hazard-credit/risk learning rather than widening the gas envelope again.

### 2026-09-25 throttle-envelope PPO screen: paired TUNE result

The recovered run completed all four 8,192-decision PPO arms and the registered TUNE screen. `paired-comparison.json` is the raw comparison source; all arms share the same initial model-state SHA and TRAIN budget. TUNE comprised two seeds on one map geometry, and each policy produced identical results for the two seed labels, so treat this as one trajectory per policy arm rather than four independent episodes.

| Arm | TUNE completion | Median finished lap | Mean episode speed | Collisions | Mean damage | Off-track episodes | Valid finishes under 13 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| throttle 1.0 / cap 0.12 | 2/4 (50%) | 20.64 s | 33.35 m/s | 10 | 0.50 | 2 | 0/4 |
| throttle 3.5 / cap 0.42 | 2/4 (50%) | 19.54 s | 40.89 m/s | 12 | 0.60 | 0 | 0/4 |

Paired seed 8104 improved: both TUNE runs finished 1.10 s sooner, at 2.09 m/s higher mean speed, with one fewer collision onset and 0.2 less damage. Paired seed 8105 did not finish in either arm at 75.3% progress; the widened-throttle treatment raised mean speed from 26.14 to 39.12 m/s but increased collisions from 3 to 5 and damage from 0.6 to 1.0. Overall the treatment is faster but safety is not non-regressing, completion did not improve, and no lap met 13 s. `decision.status` is `rejected`, `promotion_authorized` is false, and SOTA/submission remain unchanged.

The treatment never saturated its 0.42 gas limit. Together with the fixed-pixel diagnostic, this shows the wider throttle transform can raise speed, but speed-only pressure exposes a safety failure on the harder seed. Next paired learning test should hold throttle expansion at 3.5 for both arms and vary only the Lagrangian safety-cost PPO signal (off vs adaptive), after B's trainer integration tests pass and C's action-repeat measurement finishes. This directly tests whether PPO can retain the speed gain while reducing collision/damage; it is not a rule-based driving fallback. Keep TUNE deferred until all arms finish, use actor-only export, and require map-diverse validation before any candidate promotion.

The second-seed fixed-pixel diagnostic is now independently saved as `C:/Users/koi/.codex/worktrees/0e2d/HAIC/artifacts/haic/throttle-envelope-ppo-2seed-u8-20260925T135214KST/postfailure-throttle-action-diagnostic-seed8105.json` (SHA-256 `A03EEF198D76C1B0C8A40CD634F0AD48DE6283715A673D0EC969E4AE637AC6C1`). It passes its TRAIN provenance gate and uses the same 1,040-image bank. For PPO seed 8105, baseline→treatment gas mean was 0.06414→0.07937 (+23.75%), p90 0.08940→0.11255, and max 0.09035→0.11448; again 0/1,040 commands reached 0.1176. Brake and steering means changed little. Thus the action-scale shift repeats across both PPO seeds on identical pixels, but the completed TUNE screen shows it can raise speed while worsening collisions on the harder seed. No further throttle-cap-only PPO run is justified without a safety-learning change.

### 2026-09-25 action-repeat diagnostic: full raw check and next PPO gate

C's fixed-actor runner needed one import-path fix before any environment or actor was loaded; attempt 1 recorded zero maps/environments/actions. After the repair, the immediate process gate passed and all eight episodes completed in strict actor-only inference. The canonical actor SHA, two TRAIN map hashes, four TRAIN map-seed identities, strict 25-tensor load, PPO update count zero, and absence of TUNE/held-out/official inputs are recorded in the copied protocol and gate receipts under `artifacts/haic/action-repeat-sota-actor-2400tick-train-diagnostic-v1/`.

Raw episode records confirm: 4-tick and 2-tick controls both finished all four cells with no sub-13 lap. The 2-tick arm was 0.69 s faster by the reported median (20.08 vs 20.77 s), had the same 2/4 collision episodes, reduced mean final damage from 0.20 to 0.10, and did not retire off-track. Wall time increased from 11.46 to 16.40 s. On the obstacle geometry, 2-tick produced a normalized final tile progress of 1.0 where 4-tick produced 0.9756 despite both reporting completion at the finish line; preserve both raw labels instead of treating this as a finish-rate difference. This fixed-actor diagnostic supports a possible control-resolution effect but does not show a learned improvement; the tiny 2-geometry screen is insufficient for promotion.

The throttle-envelope TUNE screen remains rejected: larger throttle raised speed but worsened aggregate safety, and no lap approached the 13 s gate. C's action-repeat measurement is now complete. B is finishing the Lagrangian trainer integration and focused tests; after that and only after its code/preflight pass, the next PPO comparison is adaptive safety-cost PPO vs ordinary PPO with throttle expansion 3.5 and action repeat 4 held fixed. The run must remain actor-only at inference, use TRAIN-only safety labels, defer TUNE until all arms finish, and retain full paired collision/damage/speed/lap reporting. No SOTA or submission update is justified by either diagnostic.

### 2026-09-25 15:37 KST coordinator recovery

The user asked why only one strategy seemed active. The status audit showed A and C idle after finishing their preceding bounded assignments; B alone had an in-progress Lagrangian trainer integration. The orchestration gap was that these task threads finish and become idle unless the coordinator gives them a distinct follow-up. A heartbeat of the coordinator does not itself restart completed task turns.

At 15:37 KST, A was reactivated for a read-only, no-replay audit of the already saved seed8105 U32 `tune-before.json` / `tune-after.json` traces: summarize hazard cues, speed/pedal/steering, collision-bearing decisions versus onsets, damage, and progress; collapse duplicate seed labels from the same geometry; propose one falsifiable PPO reward/cost variable. C was reactivated for a separate source-level steering-coordinate audit of d158–164, tracing signed obstacle offset and PPO steering through pixel features, action scaling, simulator sign, and tests; use saved frames if present, otherwise state the evidence limit. Both tasks prohibit new evaluation, replay, training, and code edits. B remains on Lagrangian PPO trainer integration. The immediate post-dispatch snapshot marked all three turns in progress, but no assistant/tool output had arrived yet; this is assignment state, not evidence of completed work.

The measured training bottleneck remains rollout throughput: U32 took 2,801.5 s, with 66.1% in sequential environment rollout and 33.9% in PPO updates. A validated cache comparison reduced strict TRAIN collection time 10.2% with identical trace hashes, which projects to about 6.8% total-run savings if the same ratio holds; this is not enough to meet the 13 s lap target. The primary performance path remains PPO learning of safe early obstacle response under the wider speed envelope, with no SOTA or submission promotion absent valid under-13 finishes and safety non-regression.

### 2026-09-25 15:42 KST Lagrangian integration source check

A read-only source scan now finds the Lagrangian path wired through `collect_rollout()` (separate cost critic values and TRAIN episode accumulation), `train_policy()` (PPO update with the pre-window multiplier, then completed-episode-only multiplier updates), training checkpoint metadata, and CLI preflight that requires the registered TRAIN-only split. The safety mode also removes the environment's terminal out-of-bounds -100 from reward and disables the legacy direct collision/damage/off-track/failure reward deductions, leaving those safety signals in the cost channel. This is source-level implementation evidence only; the focused integration test file was last modified at 15:40 KST, no Python/pytest process was present at the 15:42 process check, and no test result or PPO training result has been reported. Keep training gated on B's explicit test/preflight result.

At the same 15:42 check, A and C's newly assigned turns were still in progress with no report; B's turn was still in progress. No active PPO or pytest process was found. Do not treat an elapsed wait as a stopped task or restart these turns; next action is to receive their evidence or a specific blocker, then decide whether the paired TRAIN screen can begin.

### 2026-09-25 15:59 KST exact SOTA seed42 failure diagnosis

A and C's source findings were paired with the exact official Track 1 seed42 trace: `artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/official-track1-seed42-feature-trace.json` (SHA-256 `BAA84F15097E00B604E2A316CC21BBA7C4DE1C3FDC8D363913983119B784B675`). The trace is 163 decisions, seed 42, progress `0.162544`, DNF `off_track`, two collision onsets and damage `0.4`. Its checkpoint hash matches the SOTA pointer (`3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`). The older scratch trace follows nearly the same opening but has inconsistent collision/damage aggregation; use the feature trace and `feature-trace-summary.json` for this diagnosis.

At decision 58 the pixel cue appears 16.016 m ahead at 38.737 m/s with urgency `0.917`; constant-speed time to contact is about `0.413 s`, or five 4-tick policy decisions. PPO brakes only `0.045` there, then brakes zero at d59–d63. Urgency rises to `1.0`; gas rises from `0.004` at d60 to `0.071` at d62 and `0.082` at d63. The exact summary counts 105 high-risk decisions with mean brake zero. Recomputing the existing pixel hazard-risk formula from trace features gives risk `.37` at d58, `.85` at d59, `.87` at d60, `.89` at d61, and `.98` at d62: the actor lets the perceived obstacle move into its lane instead of reducing risk. C's source-level sign audit maps negative obstacle offset to left-of-road and negative PPO steering to left turn; the d58–d60 negative steering therefore likely points toward the obstacle. Curve correction is a confound, so treat direction as a strong hypothesis rather than standalone causal proof.

The first collision is logged at d63 at 3.836 m and 7.631 m/s, followed by a second at d65 at 4.27 m/s; both have gas around `.08` and brake zero. From d66 to d163, all 98 decisions have zero brake, mean gas `.075`, median speed `.037 m/s`, and unchanged progress `.162544`. The episode then retires off-track. The failure chain is therefore: late high-speed visual cue → brief weak braking and likely wrong-side steering → contact → no learned lateral escape/recovery. It is not detector blindness.

The SOTA training config used `safe_speed_reward_target=50 m/s`, while the current trainer default is 70 m/s; official route lengths require about 74–92 m/s average for 13 seconds. The selected actor's held-out mean is about 40.4 m/s, median peak about 53 m/s. Widening the gas envelope raised speed but increased collisions and produced no sub-13 lap, so action range alone is insufficient. A future pace treatment should reward higher speed only on pixel-clear, low-curvature segments while the safety condition stays fixed; global high-speed pressure already regressed safety.

Next controlled sequence: finish B's current adaptive-Lagrangian-versus-off PPO screen without changing its speed/reward settings. If safety cost reduces collision but does not teach cue-to-action avoidance, the next isolated PPO variable is to integrate the existing pixel-only `training/hazard_potential.py` shaping term (`gamma * Phi(next risk) - Phi(current risk)`) into the PPO reward, with Lagrangian setting held constant. It is currently a standalone helper and has no call site in `train_policy.py`. Falsify it if pre-contact hazard risk and collision/damage do not decrease on matched TRAIN/TUNE cells, or if any safety gain comes only with worse completion/pace. Inference remains actor-only and pixel-only; no rules-based steering is added.

### 2026-09-25 16:04 KST paired Lagrangian preflight hash fix

The first `seed8104-off` attempt exited before learning with `initialized actor does not match the paired preflight hash` (exit code 1). B traced the mismatch to initialization running with different PyTorch CPU thread counts: preflight initialized with the default 8 threads, while the trainer sets `CPU_INFERENCE_THREADS=1` before seeding/actor construction. Preflight now sets the same thread count. `tests/test_lagrangian_screen_preflight.py::test_preflight_actor_hash_matches_trainer_cpu_thread_initialization` encodes this regression; the focused Lagrangian/preflight suite was reported 15/15 passing and `py_compile` passed.

The refreshed `artifacts/haic/lagrangian-off-vs-adaptive-20260925/preflight.json` is independently readable and reports `passed=true`, `execution_started=false`, matched off/adaptive actor hashes for each seed (`8104`: `cfa1dd79...`; `8105`: `c2b37f16...`), only the registered TRAIN maps opened, and no non-TRAIN or official maps opened. The prior failed console and exit code remain preserved. At 16:04 KST, the only PPO arm observed running was the repaired `seed8104-off` attempt; its Python worker had accumulated CPU time and a new `console-attempt-2.log` was open. No adaptive arm or TUNE evaluation had started. Treat this process as the live handle and do not launch another arm concurrently.

### 2026-09-25 16:38 KST TRAIN-only hazard-exposure sample and potential-shaping preregistration

#### Completed frozen-policy measurement

The previously authorized sample-only collection did run. Raw source: `artifacts/haic/hazard-exposure-train-u32-step32768-8192decisions-20260925.json` (SHA-256 `B01EFEC4EAB6B117716E3354D4603BA5A7A3FB2EC7B80D43AD3B9A2A420C9D5D`). It records exactly 8,192/8,192 decisions, zero optimizer/training updates, unchanged actor-state hash (`3fdb6937e6e22883fb4476a57749dfe94b32233e2c8cabe28ec01e236180eb28` before and after), `split_group_loaded=train`, and four custom TRAIN map/seed cells: `custom-track-haic-obstacles-20260920` seeds `20260920/20260924`; `custom-track-haic-train-20260921` seeds `20260921/20260925`. There are 32 episode aggregates across repeated rollout updates, not 32 independent maps. No TUNE, held-out, or official map episode was collected, and no image/step trace was saved. The sampled actor was at global step 40,960 (the U32 continuation's local checkpoint filename is `actor-step-32768.pt`).

Pooled urgency-band statistics from those raw aggregates:

| Urgency | Obstacle observations | Mean car-relative signed offset | Non-neutral steer samples | Steered away | Steered toward | Immediate collision-labeled transitions |
|---|---:|---:|---:|---:|---:|---:|
| `[0.00, 0.25)` | 98 | −2.590 px | 94 | 25 (26.6%) | 69 (73.4%) | 0 |
| `[0.25, 0.50)` | 100 | −1.980 px | 89 | 32 (36.0%) | 57 (64.0%) | 0 |
| `[0.50, 0.75)` | 98 | −2.174 px | 89 | 35 (39.3%) | 54 (60.7%) | 0 |
| `[0.75, 1.00]` | 2,026 | −0.894 px | 501 | 242 (48.3%) | 259 (51.7%) | 43 (2.12%) |

The alignment score is positive for steering away, negative for steering toward, using `relative_px = obstacle_lateral_offset * 12 + road_center_offset * 42`; positions within ±1 pixel are excluded. The high-urgency band contains 87.3% of all obstacle observations, but 1,525/2,026 offsets (75.3%) are too close to center to determine a left/right direction. Among the remaining 501 observations, away and toward actions are nearly balanced (mean alignment −0.025). Lower urgency bands lean more toward the obstacle. This supports inconsistent cue-to-action alignment, not a detector-blindness claim. The sign score is a geometric proxy; it does not account for heading, curvature, or counterfactual collision risk. Collision counts here are collision-labeled next transitions, not unique onsets.

Raw-aggregate outcome totals add that these 8,192 decisions came from 32 episode rows with 3 finishes, 18 off-track retirements, 47 collision-labeled transitions, and mean episode end progress `0.4524`; risk-positive decisions total 4,010 with mean gas `0.0617`, mean brake `0.0077`, brake-active rate `11.42%`, and 46 next-collision labels. These denominators must not be collapsed: `risk_positive_count` means computed risk `> 0`, while `obstacle_present_count` requires pixel feature 4 `>= 0.5`, so the 4,010 risk-positive decisions need not equal the 2,322 obstacle-present observations; the 46 risk-positive next-collision labels likewise differ from the 43 labels in the narrower highest-urgency band. Episode/bin rows repeat the same four map-seed cells and are not independent samples. The frozen actor is the U32 continuation checkpoint (global step 40,960; actor-state SHA above), not the selected SOTA actor (checkpoint SHA `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`).

The recorder also has progress-bin collision and damage aggregates. For an exploratory one-bin lag, episode-progress cells with a hazard in the source bin were matched to that same episode's immediately following progress bin:

| Source-bin urgency | Source observations / episode-progress cells | Following-bin decisions | Collision-labeled transitions | Following-bin damage delta |
|---|---:|---:|---:|---:|
| `[0.00, 0.25)` | 98 / 64 | 1,598 | 20 (1.252%) | 4.00 |
| `[0.25, 0.50)` | 100 / 63 | 1,528 | 14 (0.916%) | 2.80 |
| `[0.50, 0.75)` | 98 / 64 | 1,686 | 15 (0.890%) | 3.00 |
| `[0.75, 1.00]` | 2,026 / 98 | 2,354 | 16 (0.680%) | 3.20 |

These following-bin rates are descriptive only: the denominator is all decisions in the next bin of an episode-progress cell that had an exposure, not a per-cue matched transition. Cells overlap across urgency bands and repeat the same four TRAIN map/seed identities. For example, `[0.65,0.70)` exposure occurred in only two episode-progress cells; their next `[0.70,0.75)` bins contain 5 collision labels among 21 decisions and damage delta 1.00. This is too small and confounded to treat as an estimated hazard effect. In the unconditional progress aggregates, the `[0.70,0.75)` bin has 5/115 collision-labeled transitions and damage delta 1.00; `[0.75,0.80)` has 0/154 and zero damage. Exposure coverage is concentrated early/mid-track: progress `[0.00,0.20)` and `[0.80,1.00]` have no visible-obstacle observations, while `[0.75,0.80)` has only six.

The aggregate schema does not retain damage delta by urgency band, risk change by action, or an event-level cue-to-following-bin join. Thus it cannot answer exactly whether a particular urgency-bin cue caused a later-bin collision/damage. A future recorder change should add compact `(source progress bin, urgency band, next progress bin)` counts/sums for next collision, damage delta, and current/next visible risk; add tests for progress-boundary crossing and the sign/neutral rules. No code or trainer file was changed for this audit. The prior strategy entry reports 5/5 focused exposure tests and 20/20 potential-helper tests; this audit read the present code/tests but did not rerun them while B's shared trainer integration remains in progress. Static inspection found the previously reported undefined `pre_action_labels` reference absent and `risk_positive_mean` divided by its matching positive-risk count. No separate-file unit-test defect was identified.

#### Pixel-only hazard-potential shaping — preregistered, not run

The raw evidence makes potential shaping plausible but not proven. In the official Track 1 seed42 trace (diagnostic evidence only, never a training/tuning input), visible risk rises from `.37` at d58 to `.85/.87/.89/.98` by d59–d62 while the policy briefly brakes, then releases brake and reapplies gas; two collision onsets follow. The TRAIN-only sample independently shows that the highest-urgency steering response is nearly 50/50 away/toward. Current reward shaping already penalizes gas and rewards braking under visible risk, while its positive clearance reward is gated to damaged, low-speed recovery. `training/hazard_potential.py` is currently a tested helper only; there is no training call site.

**Hypothesis:** an additive potential difference gives PPO earlier transition-level credit for reducing visible pixel hazard risk, even before damage/recovery. Use `Phi(risk) = -risk` and `F_t = beta * (gamma * Phi(risk_next) - Phi(risk_now))`, with `gamma=0.99` and a preregistered treatment coefficient `beta=0.10`; the control is `beta=0`. Add the term once to the already reward-scaled transition reward (do not apply `REWARD_SCALE` twice). At the seed42 d58→d59 risk change, this scale would add approximately `−0.047` for the observed risk increase. This is one reward change only: no obstacle detector, action rule, steering target, brake/gas scale, or other reward coefficient changes. Both arms keep ordinary PPO with Lagrangian cost disabled (`lambda=0`) to isolate this hypothesis from B's separate arm.

Execution is gated until B closes its trainer/preflight work. Use two paired PPO seeds (`8104`, `8105`), 8,192 decisions/eight updates per arm, and only the registered custom TRAIN cells listed above. Within each PPO seed, control and treatment must start from the exact same fresh actor initialization and optimizer state; no teacher warmup or inherited checkpoint is allowed because the current SOTA/U32 ancestry records official/held-out episodes and no clean pretrained lineage has been approved. The registered `training/maps/site/site_map_split.json` has only the two custom families in its TRAIN group; do not load its TUNE/held-out groups or any official map during training. Use the last fixed-budget checkpoint, with no TUNE-based checkpoint selection or coefficient tuning.

After both paired arms are frozen, a single locked evaluation may report the same metrics for every strategy: completion rate, valid completion under 13, collision onsets per episode, final damage, and median lap time among finishers (plus DNF progress). Any TUNE/held-out/official episodes are evaluation-only after freeze and cannot change the coefficient, checkpoint, or subsequent training settings. Mechanism metrics are the fraction of visible-hazard transitions that reduce `visible_hazard_risk`, and car-relative steer-away alignment with neutral offsets reported separately. Stop before PPO if the manifest/checkpoint preflight finds any non-TRAIN episode, mismatched paired initialization, non-finite reward, or B's work is still open. Reject the shaping hypothesis if it fails to improve risk-decreasing transitions and paired collision/damage outcomes, or if an apparent safety gain is obtained only with worse completion, under-13 valid completion, or lap time. A two-seed screen is only a screening result; no SOTA promotion without independent map-diverse confirmation.

No potential-shaping PPO run, evaluation, or SOTA/RESULTS promotion has occurred. The fixed-U32 sample is a completed diagnostic measurement, not a learned-policy result. The current B task remains the scheduling gate; no environment, rollout collection, or training was started in this audit.

### 2026-09-25 16:49 KST Lagrangian preflight: fresh actor with SOTA runtime input architecture

The SOTA/U32 checkpoint lineage includes held-out/official exposure, so pretrained initialization is not approved. The selected SOTA `model_state` was not opened or loaded. Existing `lagrangian-off-vs-adaptive-20260925` seed8104 off/adaptive and seed8105 off runs remain clean-start diagnostics under their recorded `use_hud=true/use_visual_features=false` architecture; they are not results for this new setup and are not merged with it.

Added a distinct non-executing preflight at `artifacts/haic/lagrangian-fresh-runtime-architecture-20260925/preflight-seed8104.json` (SHA-256 `597afb139eaea8b4de05d6a0a474ac703704565022fb650c0f6d6dbd49df5e84`). This setup starts a fresh seed8104 actor with `use_hud=false`, `use_visual_features=true`, `use_temporal_features=false`, matching the SOTA runtime input architecture without inheriting its weights. Both `off` and `adaptive` initialize from the same actor SHA-256 `19d3ccaa2c47c1b3e0fe365d31f5c6d53bce2571aa5ef55f3538eb122d630477`; each has fresh Adam state (zero entries). Teacher warm-up is zero. Both arms record the same throttle expansion 3.5, brake expansion 1.0, 8,192 decision budget, and eight updates.

The map gate passed with only the four registered custom TRAIN cells loaded. No TUNE, held-out, or official map files were opened. The report has `execution_started=false`, defers TUNE, and records the two runner commands for later use. Focused preflight/trainer tests passed 17/17 and `py_compile` passed. No PPO training or TUNE was run in this stage; no SOTA or submission pointer changed.

### 2026-09-25 PPO update-drift audit of the earlier clean-start teacher screen

I independently re-aggregated `episodes.jsonl`, `updates.jsonl`, and both `run-manifest.json` files from `C:\Users\koi\.codex\worktrees\4fd3\HAIC\experiments\strategy-c\clean-start-train-only-teacher-bc-ppo-seeds-8104-8105-v1-attempt2\`. This is evidence from an older isolated source snapshot, not the current Lagrangian implementation. Its manifests record fresh random initialization, TRAIN-only teacher warm-up for 30 epochs, PPO learning rate `3e-4`, 8,192 PPO decisions/eight updates, HUD enabled, visual features enabled, temporal features disabled, and four registered custom TRAIN cells. The repeated evaluation records load one TUNE map geometry under two seed labels; those rows are not independent maps.

The raw PPO diagnostics show substantial policy movement: seed 8104 approximate KL was `.091`, `.109`, `.375`, `.238` through updates 1–4 while gradient norm rose `7.6 → 38.5`, reaching `67.0` by update 8. Seed 8105 reached approximate KL `.412` at update 1 and gradient norm `70.0` by update 8. On the repeated TUNE geometry, seed 8104 completed both update-1 episodes at `19.86 s`, then completed neither at update 2; seed 8105 completed none across updates 1–8. No recorded finish was under 13 s. At update 8, the seed-8105 teacher-warm-start arm averaged `0.179` progress and `32.49 m/s` in PPO rollouts. Its `0.24` gas limit was not saturated in the logged action metrics.

This makes high-LR PPO drift a credible failure mechanism for that run, but it is not a controlled causal test: both seeds, the older HUD-enabled architecture, teacher warm-up, and an older reward/configuration differ from the current `2e-6` clean-start Lagrangian screen. The repeated TUNE map also cannot establish map generalization. Do not use those checkpoints as a current training parent. The current paired Lagrangian screen tests a different question (adaptive safety cost under the pixel-only runtime architecture); after it closes, the next PPO-stability test should keep one train-only initialization fixed and compare a behavior-preserving KL/early-stop treatment against ordinary PPO, using training-side KL and completion metrics before one locked evaluation.

### 2026-09-25 coordination reset after idle strategy threads

The user correctly pointed out that only one live run was visible. A fresh Codex task snapshot showed A, B, and C idle while the root's `seed8105-off` PPO arm was running. All three existing strategy tasks were reactivated with separate, read-only questions; each was told not to edit code or start training, replay, or evaluation:

- A (`01a0b2c0-dd70-7f71-9c5e-95e74bbfaaf1`): inspect deterministic visual-to-steering conditioning and whether the feed-forward observation misses the first curve; separate policy response from sampling noise.
- B (`01a0c2fd-b049-7ef1-aae8-fa5d43fbc5de`): diagnose the 70 m/s speed objective against observed 32–44 m/s speed and unsaturated gas; distinguish reward pressure from action scaling or vehicle dynamics.
- C (`01a0cec2-4799-7303-aba2-baa80f015189`): inspect whether PPO updates erase teacher behavior after BC warm-up, using per-update action and trajectory drift.

The immediate Codex snapshot then showed all three tasks active alongside the root strategy task. These are assignments, not completed findings; no claims from them are counted until their evidence-backed reports arrive. The root retains the sole simulator slot and continues the already-running PPO arm, so parallel analysis cannot introduce simulator resource contention. The root task and simulator process were still active at the status check; the output directory for `seed8105-off` had not appeared yet, so no result or promotion is recorded here.

### Warm-up arm log reaggregation: pedal headroom versus driving failure

Before the seed8105 arm completed, I independently parsed the full raw stdout JSON for the completed `seed8104-off` and `seed8104-adaptive` arms. Across all eight PPO updates in each arm, `gas_saturation_fraction` was zero; mean sampled gas stayed about `0.146–0.216` under a `0.42` throttle limit, mean speed was `32.3–44.1 m/s`, and peak rollout speed stayed around `64–66.5 m/s`. Neither arm logged a finished training episode. Thus the actuator ceiling was available but the learned actor did not request it consistently; the unfinished/off-track trajectories also depress episode mean speed. This does not establish whether weak speed pressure or curve/obstacle failure causes the low action choice.

The registered earlier throttle-envelope comparison further argues against another global range increase as the next isolated variable: on its four evaluation episodes, expansion `1.0` and `3.5` both completed `2/4` and had zero under-13 finishes; median finished lap improved from `20.64 s` to `19.54 s`, while collisions rose `10 → 12` and damage `0.5 → 0.6`. Its geometry/protocol is distinct from the current clean-start teacher screen, so use this only as evidence that widening range alone did not achieve the target, not as a direct paired estimate for the current arms.

Current causal priority is to separate (a) early steering/perception causing off-track and short episodes from (b) a weak speed objective causing unsaturated gas. Do not retune throttle expansion again until the active A/B/C source audits and the locked post-freeze evaluation identify which mechanism dominates. No completion, under-13, or promotion claim is supported by these TRAIN rollouts.

`seed8105-off` has now closed successfully through `RULES.md` with return code 0: 8,192 PPO decisions, eight updates, 554.3 s wall time, and no TUNE selection. It logged zero finished TRAIN episodes. Across updates, mean speed was `39.2–49.0 m/s`, mean gas `0.134–0.148` under the same `0.42` ceiling, and saturation stayed zero; the largest episode progress was `0.569` mid-run and `0.343` at update 8. The seed8105-adaptive arm has since started with the same initialization seed, map split, PPO budget, reward settings, and warm-up configuration; only Lagrangian mode differs. The current paired evaluation remains deferred until that arm finishes.

### A visual-to-steering source audit and next measurement

A's read-only audit corrects the “single-frame policy” assumption. `VisualActorCritic` receives a four-frame `(B,4,84,84)` stack and seven current-frame visual features, but has no recurrent state. The optional temporal feature is one previous-obstacle-side cue, not a general frame-difference representation. The logged rollout steering standard deviation includes Normal sampling and cannot establish observation-conditioned deterministic steering. The previous TUNE actor's nearly constant `-0.121` steering is close to the initial `-0.12` bias, which is consistent with weak mean-head learning but does not prove it. The handcrafted road-center parser can drop near/far offsets when pixel rows fail its validity threshold; stored logs do not contain those row-validity flags, so cue sparsity on the first curve remains unverified. The CNN still receives the image, so this is not evidence of a missing pixel input.

A proposed a distinct four-to-eight-frame input-window PPO contrast, initializing the treatment's new frame channels to zero and copying the existing first-convolution weights to the matching recent frames so both policies start with the same function. This is not the same as enabling the existing single-value previous-obstacle-side feature, and the older `site-map-temporal-branch` run never toggled that cue on versus off. Still, the strategy lab already contains conditional recurrent/temporal experiments; do not start a new history architecture solely from this source audit. First use the already-locked deterministic TUNE traces to measure action/visual-feature dependence. A has been assigned a separate source-only check of whether the evaluator actually records enough fields for that measurement; no result is back yet.

### Existing no-warm-up TUNE trace: actor mean stays at its initialization bias

I re-read `artifacts/haic/lagrangian-fresh-runtime-architecture-20260925/tune-evaluation.json` and recomputed action statistics from every saved decision. All eight episodes across four no-warm-up arms stopped at exactly progress `0.09053498` with reason `off_track`; each contains 174–175 decisions. Deterministic `Agent.act()` steering means range from `-0.1213` to `-0.1209` across episodes, with within-episode standard deviation only `0.0001–0.0002`; gas is `0.0659–0.0660` and is effectively constant. The source confirms `Agent.act()` encodes the pixels, takes the actor `action_parameters`, bounds the policy mean, and returns it with planner disabled. The trace's `controller` field is null, so it does not preserve the seven pixel features/row-validity alongside actions. These eight rows repeat one TUNE geometry under two seed labels and are not independent map trials.

This is strong evidence that the no-warm-up deterministic actor is not using visual input to change its early steering/pedal outputs; it does not isolate whether weak pixel features, near-zero policy-mean gain, or too little PPO mean-head learning causes that behavior. It also explains why sampled TRAIN action variation cannot be used as proof of learned state dependence. The frozen warm-up screen now has a decisive next metric: compare its actor-only deterministic TUNE traces against this exact baseline. If warm-up traces remain near `steer=-0.121`, `gas=0.066`, and progress `0.0905`, BC did not fix the deployed mean policy. If actions vary materially with visual state but episodes still leave the road, correlate the recorded progress/yaw/action sequence before choosing an input-history or curve-control experiment.

### 2026-09-25 locked TUNE result: teacher warm-up changes behavior, but no candidate finishes

The frozen `ppo-teacher-warmup-route-rescue-20260925/tune-evaluation.json` completed through the RULES runner. It evaluated four frozen arms (`seed8104/8105 × adaptive/off`) on two seed labels for the same TUNE geometry. All eight episodes failed, so the two labels are repeated trajectories per arm rather than independent map samples; do not treat this as an 8-map estimate.

Compared with the no-warm-up actor, which stopped at progress `0.0905` on all eight repeated rows with nearly constant `steer≈-0.121` and `gas≈0.066`, the warm-up actors reached farther and showed changing actions over each trajectory. Mean progress by arm was `0.588` for seed8104-adaptive, `0.741` for seed8104-off, `0.551` for seed8105-adaptive, and `0.251` for seed8105-off. Across these four arms, the unweighted mean is `0.533`; this is a diagnostic gain over `0.0905`, not a successful lap result.

None completed a lap or achieved an under-13-second finish. Seed8104-adaptive retired off-track at `0.588` without collision; seed8105-off went off-track at `0.251` after one collision; seed8104-off and seed8105-adaptive crashed at `0.741` and `0.551`, respectively, with five collision events per repeated episode and final damage `1.0`. Mean speeds ranged from `16.48` to `52.01 m/s`, and peaks from `56.86` to `72.91 m/s`; these are not lap-speed measurements because every episode retired early. The four arms do not isolate a single cause because both seed/checkpoint and cost mode vary.

This rejects the narrow hypothesis that teacher-only warm-up left the deployed actor at its initialization action bias: deterministic steering and gas now vary over each trajectory, and progress improves materially. It does not show that the policy reads the right visual cue or generalizes. The trace still has `controller=null` and lacks raw pixels and visual features, so it cannot align a steering error with the image. The next experiment should target the observed failure modes after the early bend while retaining the fixed speed/action envelope; do not promote or select a checkpoint from these TUNE rows.

The RULES status output also reports a best observed held-out actor record of `19.26 s` over five completed episodes, but its restriction status is `unknown`, under-13 validity is unrecorded, and it was not promoted. An existing actor archive passes the static package audit (`forbidden_calls/imports=0`, no extra or missing files), but the submission preflight did not pass because restriction verification is incomplete; submission remains disabled. This is not a submission-ready result.

### 2026-09-25 follow-up: paired pixel-risk potential screen

The completed A/B/C audits agree that the current bottleneck is learned action selection, not a lack of theoretical pedal range. The no-warm-up actor nearly preserves its initial steering/gas bias and leaves the first bend before reaching an obstacle. The teacher-warmed actors respond with changing actions and progress farther, but the four frozen TUNE trajectories still produce zero finishes and contain either obstacle contact or later off-track retirement. The unusually low mean speed in one arm is explained by a post-contact stall (101 of the remaining 102 decisions below 1 m/s), not by a low speed ceiling. TUNE seed labels 20260922/20260926 repeat one geometry.

**Amendment to the earlier potential-shaping preregistration:** keep teacher warm-up fixed at the already user-approved 30 TRAIN-only BC epochs in every arm, followed by PPO only. The earlier no-warm-up clause was intended to avoid inheriting an actor trained on official/held-out maps; this screen instead starts every arm from a fresh actor and uses only the registered TRAIN manifest `training/maps/site/site_map_split.json`. The warm-up does not vary between arms. The current 8-update no-warm-up run's near-initial deterministic action bias, alongside the already recorded TRAIN-only BC MSE reduction, makes the fixed warm-up a more useful common starting point. This is a protocol amendment, not a performance result.

The screen changes only `beta`: ordinary PPO with `beta=0.00` versus `beta=0.10`, `Phi(risk)=-risk`, `gamma=.99`; the additive shaping term is applied after the base reward has been scaled. It uses fresh matched actors/Adam state for PPO seeds 8104 and 8105, the same four registered TRAIN map/seed cells, 8,192 decisions/eight updates per arm, no resume or TUNE checkpoint selection, and the existing action/reward settings held fixed. Per-update TRAIN summaries now record positive/urgent visible-risk exposure, risk-decrease fraction, and collision count. The later locked TUNE evaluator verifies every fixed-budget checkpoint and preflight hash before it opens TUNE, then records actor-input risk alongside actions. TUNE remains diagnostic only; two seed labels on one map cannot support SOTA promotion.

No potential-shaping training has started yet. `training/train_policy.py` now gates both coefficient arms through a TRAIN-only paired preflight; `training/preflight_hazard_potential_screen.py` binds the map/source/configuration/actor hashes; `training/evaluate_hazard_potential_tune.py` is the locked evaluator. Focused tests passed (8/8) for reward scaling, rollout risk telemetry, paired preflight validation, and checkpoint-gated risk-trace evaluation. The preflight and four-arm training are still pending; this experiment has no outcome and changes no SOTA pointer.

### Preflight and first-arm execution update — 2026-09-25

The RULES-controlled preflight completed successfully with return code 0. It loaded only the four registered TRAIN cells from `training/maps/site/site_map_split.json`, confirmed exact shared configuration and seed-matched initial actor hashes, and recorded no TUNE/held-out/official map access. The report is `artifacts/haic/ppo-hazard-potential-2seed-u8-20260925/preflight.json`; the RULES execution plan is `artifacts/haic/rules-runs/improvement-plan-20260925T114429Z.json`.

The first sequential arm (`seed8104-beta-0.00`) is running through the explicit RULES command. Its Python trainer process is live; no checkpoint or training result has been written yet, so it is still pending evidence. The three follow-up strategy tasks were reactivated with separate read-only questions; their prior completed reports remain the current evidence until new replies arrive. The best indexed actor-only held-out record remains 19.26 s over five finishes, with restriction status unknown and under-13 validity unrecorded. The package's static audit passes, but submission remains blocked by the restriction gate. No SOTA pointer changed.
