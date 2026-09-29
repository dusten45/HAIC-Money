# HAIC 하네스 운영 요약과 현재 상태

갱신: 2026-09-26. 완주율 우선 HAIC 연구 하네스에서 네 독립 방향을 탐색했고, 여러 에이전트가 후보별 근거와 활성 조건을 검토했다. Battlecode에서 가져오려는 것은 팀별 전략 탐색·중앙 통합·반복 가능한 비교 절차이며, HAIC 환경에 맞춰 고정 map/seed matched evaluation을 사용한다. 현재 dirty working tree에는 split 인자·Windows 경로·held-out tune-winner 검증 로직이 있다. 이는 이번 source review에서 확인했지만 아직 동작 검증되지 않았다. control/후보 체크포인트, exact split·seed·budget plan과 승인이 아직 없어 새 학습·평가는 실행하지 않았다.

## 합의한 운영 방식

- **완주율이 최우선**이다. 같은 조건의 후보끼리 비교하고, 완주율이 같을 때만 랩타임·미완주 진행도·충돌·손상·추론 지연을 비교한다. 공식 대회 점수와 내부 선택 기준은 구분한다.
- 중앙 담당자가 상태·우선순위·예산·역할을 조율한다. 여러 에이전트는 서로 다른 탐색 방향을 맡고, 사실·추론·미확인 사항·권고·근거 경로를 나누어 보고한다.
- 가설마다 근거, 관측 가능한 정보, 허용 행동, 성공 조건, 비교군, 반증 조건, 최소 실험, 자원 한도를 적는다.
- 설계·실행 승인을 구분한다. 설계 승인 범위의 구현에는 별도 승인이 필요하지 않다. 평가 뒤 ADVANCE / REJECT / REVISE / PIVOT 모두 게이트 검토를 거친다. **ADVANCE이며 세 게이트가 모두 PASS일 때만 릴리스 가능**하다. 공식 제출은 별도 승인 사항이다.
- 탐색 배치는 독립 방향 4개 이상, 후보 8개 이하, 방향별 2개 이하로 제한한다. 비교 가능한 유효 주기 3회 연속 비개선이면 체크포인트를 보존하고 방향을 바꾼다. 실행 장애는 성능 실패 횟수에 넣지 않는다.
- 과거 데이터는 원래 위치에 보존한다. 자동 수집·복사·이동·재분류하지 않는다. 새 체계에서 과거 최고기록을 자동으로 승격하지 않는다.
- HAIC에 필요한 운영 원칙을 적용한다. 여러 프로젝트용 포터블 에이전트 제품을 만드는 계획은 아니다.

## 문서 역할

| 문서 | 확인할 내용 |
|---|---|
| [AGENTS.md](../AGENTS.md) | 운영 규약, 상태 전이, 승인, 에이전트 역할 |
| [PROJECT_INFO.md](../PROJECT_INFO.md) | 프로젝트 목표와 실행 구조 설명 |
| [RESTRICTIONS.md](../RESTRICTIONS.md) | 대회·데이터·보안·운영 제한 |
| [RESULTS.md](../RESULTS.md) / [SOTA.md](../SOTA.md) | 기존 결과와 역사적 로컬 최고기록 |
| [sources/INDEX.md](sources/INDEX.md) | 출처와 확인 시점 |
| [sources/legacy-path-map.md](sources/legacy-path-map.md) | 과거 데이터의 위치와 용도 |
| [strategy-history.md](strategy-history.md) | 기존 전략과 판단의 역사 |
| [handoffs/REPORT_TEMPLATE.md](handoffs/REPORT_TEMPLATE.md) | 에이전트 보고 양식 |
| [experiments/INDEX.md](experiments/INDEX.md) | 새 실험 기록 위치 |
| [handoffs/completion-first-discovery-20260926/INDEX.md](handoffs/completion-first-discovery-20260926/INDEX.md) | 이번 네 방향 가설 통합과 인계 |
| [context/current-state.md](context/current-state.md) | 현재 설계 단계, blocker, 다음 승인 게이트 |
| [handoffs/completion-first-discovery-20260926/coordinator-preflight-20260926.md](handoffs/completion-first-discovery-20260926/coordinator-preflight-20260926.md) | 에이전트별 후보 활성 조건·위험·실행 준비 검토 |
| [handoffs/completion-first-discovery-20260926/battlecode-style-tournament-draft.md](handoffs/completion-first-discovery-20260926/battlecode-style-tournament-draft.md) | 네 후보를 공통 대조군과 고정 map/seed로 겨루는 미승인 설계 초안 |

## 이번에 재개된 하네스 단계

여러 에이전트가 독립 방향을 검토하고 중앙 담당자가 [인계 묶음](handoffs/completion-first-discovery-20260926/INDEX.md)에서 통합·배치 검토했다. Battlecode에서 가져오는 것은 경쟁 규칙을 정해 여러 전략을 만들고 공통 기록으로 비교하는 운영 방식이다. HAIC에서는 한 번에 정책 하나를 주행시키므로, control과 후보를 동일 map/seed 셀에서 평가하는 [완주율 우선 실행 경로](handoffs/completion-first-discovery-20260926/battlecode-style-tournament-draft.md)를 사용한다. 이 경로는 고정 분모, 순차 arm 실행, 즉시 기록, 완주율 우선 순위를 지원한다. 앞선 source audit에서 보고한 split 인자·Windows 경로·held-out winner 문제는 현재 dirty working tree에서 해당 guard가 구현된 것을 확인했다. behavior test는 아직 실행하지 않았으므로 검증 통과로 주장하지 않는다. corridor 전용 아이디어는 제출 적격성이 확인되지 않아 제외했다. 네 후보 설계 초안에는 CEM 대신 곡률 조건 제동 보상이 제안됐지만 미승인이다. control/후보 체크포인트, 정확한 map·seed, 자원 한도, run 계획·revision/hash·승인이 확정되지 않았다. 실제 대진과 성능 결과는 없다. 새 학습·주행 평가는 필요한 승인 게이트를 마친 뒤 실행한다.

## 기존 방식과 전환 상태

지원 진입점은 `python -m haic_research.cli`다. `plan`은 manifest·가설·등록 profile·typed arguments를 고정하고, `approve`는 새 계획의 설계·실행 승인을 순서대로 기록한다. `status`는 해당 실행의 상태를 조회하며, `run`은 기본 계획 미리보기이고 `--execute`가 있어야 승인된 로컬 작업을 한 번 실행한다. `report`는 평가 결과와 세 gate, 선택적으로 등록된 후보/control 비교를 기록한다. `validate`는 정해진 문서와 설정 구조만 검사한다. [README의 실제 문법](../README.md#supported-local-cli)과 [프로젝트 입력 계약](../PROJECT_INFO.md#v2-interfaces-and-operation-inputs)을 따른다.

기존 방식의 자동 탐색·RESULTS/SOTA 재생성·latest 포인터·임의 명령 실행은 퇴역 대상이며 [과거 경로 지도](sources/legacy-path-map.md)에 기록한다. 기존 결과와 연구 자료는 원래 위치에 보존한다. 논문은 가설의 근거이고 주행 측정이 성능의 근거라는 원칙, 측정 항목, 시작 문서·외부 연락·외부 전략 도입·단일 branch/dataset 과적합 주의 규약은 AGENTS/RESTRICTIONS에 보존한다.

## 실제 완료 상태와 한계

- 설정·상태·비교·기록·승인 실행 코드는 로컬 커밋으로 남아 있다. Task 7 결과 기록과 completion-first SOTA 승격 구현은 `02be453`에 커밋됐다.
- Task 7 인계 보고서에 기록된 focused 검증은 총 **131개: 130개 통과, 1개 건너뜀, 실패 0개**다. Windows 파일 symlink 권한을 사용할 수 없어 1개를 건너뛰었다. 임시 fixture와 FakeRunner를 사용했으며 실제 학습·주행 평가를 수행한 결과가 아니다.
- Task 8 구조 검증기는 `d44f1fb`에 커밋됐다. Task 9 문서 전환과 전체 새 하네스 검증은 완료했으며 기존 파일의 실제 삭제는 자동 정책 차단으로 남아 있다. 하네스 source 전환 전체가 완료됐다고 판단하지 않는다.
- 새 결과 기록은 이름이 정해진 v2 실행의 지표·비교 증거를 gate report 전에 고정한다. SOTA 승격은 사전 등록된 비교 ID·맵·split·seed·denominator·모델 식별자, 독립 반복 평가, 세 PASS gate, 실제 ADVANCE 릴리스 이력을 확인한다. 입력된 집계 지표가 실제 주행을 정확히 반영하는지는 별도 측정·검토 근거가 필요하다.
- 기존 사용자 변경과 과거 데이터는 그대로 보존한다. 기존 RESULTS 생성 구간과 SOTA의 역사적 내용도 보존됐다.
- Task 7 구현과 현재 가설 탐색에서는 실제 학습·주행 평가·패키징·공식 제출을 하지 않았다. 따라서 **새로운 완주율 개선이나 공식 성과, 새 SOTA 승격은 없다.**
- 토너먼트 실행 경로는 문법 검사, CLI 도움말 확인, `scripts/harness/validate.ps1` 구조 검증에서 통과했다. 이 검증은 시뮬레이션 실행이나 후보 간 성능 비교가 아니다. 해당 엔진의 Battlecode식 대진은 고정 map/seed 위에서 후보를 순차 실행하는 HAIC 내부 실험이며, 공식 대회 점수나 동시 다자 주행 대전과 같지 않다.
- SOTA 문서의 완주율 0.75, 중앙 랩타임 19.32초는 기존 문서에 남아 있는 역사적 로컬 기록이다. 이번에 원자료를 다시 검증한 결과가 아니다.

## Task 9 전환 검증 기록

2026-09-26에 지정된 config/state/policy/coordinator/records/commands/CLI/results/validation 전체 단위 suite **148개 중 147개 통과, 1개 건너뜀, 실패 0개**를 확인했다. 건너뛴 항목은 Windows 파일 symlink 권한에 관한 항목이다. `scripts/harness/validate.ps1`은 exit 0, `valid=true`, issue 0개였다. 단위 검증은 임시 fixture와 FakeRunner를 사용했으며 실제 학습·주행 평가·패키징·제출 작업을 실행하지 않았다. 구조 검증은 문서/설정 구조만 확인하며 공식 규칙의 최신성, 측정된 주행 성능, 입력 지표의 진실성을 보증하지 않는다.

문서의 실제 v2 명령·입력 계약과 기존 정책 이관을 완료했다. 삭제 대상으로 지정된 정확한 파일 10개 모두 저장소 내부의 regular/untracked 경로임을 확인했다. 그러나 보호 검증을 포함한 묶음 삭제와 RULES 단일 파일의 literal 삭제가 자동 승인 검토에서 "blocked by policy" 사유로 거부됐다. 추가 이유는 제공되지 않았다. 두 번째 거부 후 삭제 시도를 중단했으며 기존 진입점·정책·전용 테스트 파일 10개는 모두 남아 있다. 데이터·보고서·checkpoint·ZIP·map·연구 자료와 디렉터리는 삭제하지 않았다. 실제 source 퇴역은 이 차단이 해소된 뒤 정확히 지정된 파일만 제거하는 별도 남은 단계다.

승인 입력의 바이트 동일성을 보강한 `050982a`는 실행 계획 스키마를 v2로 올리고, 등록된 모든 `input_file` 인수와 split이 실제 사용하는 map 파일의 SHA-256을 계획에 포함한다. 실행 claim 전에 이 해시를 다시 확인하며, split·map·resume checkpoint가 달라지면 새 계획과 승인이 필요하다. v1 계획은 자동 변환하지 않고 실행 거부한다. 전용 회귀 검증은 계획 버전과 split·map·checkpoint 변경 사례를 다뤘다.

이후 전체 하네스 suite **152개 중 151개 통과, 1개 건너뜀, 실패 0개**를 확인했다. `scripts/harness/validate.ps1`은 exit 0, `valid=true`, issue 0개였다. 이번에도 실제 학습·주행 평가·패키징·제출은 실행하지 않았다. 해시 재검증 뒤 하위 프로세스가 파일을 읽기 전까지 동시에 파일을 바꾸는 상황은 snapshot/lock으로 차단하지 않는다.

## 새 점수개선 사이클 — 2026-09-27

사용자 요청에 따라 새 V2 훈련·평가 사이클 설계를 작성했다: [score-reset-batch-r1](plans/active/score-reset-batch-r1.md), SHA-256 `875F88F9F5C7EBA978F179181CDFC54AD7885A83ABEA89F1CE0449F828F51DF6`. 상태는 `DESIGN_PENDING_APPROVAL`이다. 설계는 fresh actor 학습, 명시된 SOTA actor의 읽기 전용 matched 기준평가, 네 후보와 두 독립 훈련 시드, 사용자가 만든 TRAIN 맵, 새 official tune/held-out seed 범위를 고정한다. 과거 파일·런·체크포인트를 삭제하거나 검색하지 않는다. 아직 코드·테스트·학습·시뮬레이션·패키징·제출은 하지 않았다. 다음 전이는 이 해시에 대한 설계 승인 뒤 별도 구현 계획/승인, 이후 별도 실행 계획/승인이다. 네 병렬 대화의 source 조사 결과는 설계 입력으로만 반영했으며 성능 근거는 아니다.

사용자 지시에 따라 미승인 revision 1 초안을 보존하고 revision 2를 작성했다: [score-reset-batch-r2](plans/active/score-reset-batch-r2.md), SHA-256 `BFAA1C8927BC1F56B6B1282C53C551BF40FFCF746FBF0CE53BB192C1285078DC`. revision 2는 네 가지 메커니즘을 같은 비교 틀에서 시험하고, 16/16 완주를 속도 비교의 통과 조건으로 둔다. 같은 번들을 반복하지 않으며, 세 번의 유효하고 비교 가능한 비개선 사이클 뒤에는 새 계획 해시로 `DISCOVER`에 돌아가 새 메커니즘 계열을 등록하게 했다. 이전 네 전략 대화는 모두 조사 작업을 완료했고 현재 idle이며, 성능 실험을 수행한 대화는 아니다. 유기된 대화는 확인되지 않아 보관·삭제하거나 중복 대화를 만들지 않았다. 현재 단계는 여전히 `DESIGN_PENDING_APPROVAL`; 구현·테스트·학습·주행은 수행하지 않았다.

## 직접 점수 진단과 후보 ZIP — 2026-09-28

사용자는 이 턴에서 단계별 별도 승인 없이 진행하고 기존 아키텍처보다 성적 개선을 우선하라고 지시했다. 이에 revision 2의 10개 신규 actor 학습 배치 대신, 명시된 역사적 PPO checkpoint SHA-256 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`를 고정하고 픽셀 기반 장애물 회피 제어를 결합한 단일 후보를 직접 진단했다. 아래 실행은 V2 출력 경로에 원본 행을 남긴 **직접 진단**이며, `haic_research.cli`로 등록한 4방향 배치나 그 릴리스 게이트를 통과한 실행으로 주장하지 않는다. 기존 파일과 checkpoint는 보존했다.

| 환경·분리 셀 | 기존 PPO | 고정 회피 결합 후보 | 근거 |
|---|---:|---:|---|
| Windows tune, 트랙 1·2 × seed 132–135 | 5/8 완주 | 8/8 완주 | `runs/haic-research-v2/score-probe-throttle-20260928/`, `score-probe-hybrid-control-20260928/` |
| Windows held-out, 트랙 1·2 × seed 136–139 | 3/8 | 7/8 | `runs/haic-research-v2/score-confirm-hybrid-20260928/` |
| Windows confirmation, 트랙 1·2 × seed 140–143 | 2/8 | 6/8 | `runs/haic-research-v2/score-final-confirm-hybrid-20260928/` |
| Linux CPU tune, 동일 seed 132–135 | 5/8 | 6/8 | `runs/haic-research-v2/score-linux-tune-20260928/` |
| Linux CPU held-out, 동일 seed 136–139 | 2/8 | 6/8 | `runs/haic-research-v2/score-linux-heldout-20260928/` |

각 비교는 2,000 decision 상한과 고정 8셀 분모를 사용했다. Windows held-out에서 후보는 충돌 0회 대 기존 16회였고, Linux held-out에서는 4회 대 19회였다. Linux held-out의 완주 랩타임 중앙값은 후보 23,290ms, 기존 21,890ms로 완주 후보만 보면 느리다. Linux의 같은 셀에서 Windows와 주행 궤적 및 완주 여부가 달라졌으므로 Windows 결과를 Linux 공식 성과로 환산하지 않는다. 확인 셀은 후보를 조정하는 데 사용하지 않았고 blind seed 144–147은 열지 않았다.

고정 후보의 [로컬 ZIP](../artifacts/haic-research-v2/score-confirm-hybrid-20260928/submission-obstacle-hybrid.zip)은 SHA-256 `94D5B4558FF72145B19D8D5F8942273D06EC0D14C81E79E6B8D3137713940DA5`이며 [패키지 매니페스트](../artifacts/haic-research-v2/score-confirm-hybrid-20260928/package_manifest.json)에 파일별 해시가 있다. 12개 파일, 557,859바이트다. 압축 해제한 ZIP의 `Agent`가 Windows CPU에서 동일 tune 셀 2/132를 완주했고, Linux Python 3.11 CPU 컨테이너에서는 import/생성 약 0.007초, reset 약 0초, act 약 0.002초, RSS 약 246MB, 유효하지 않은 행동 0회를 기록했다. Linux의 2/132 셀은 충돌로 미완주하여 Windows와 다르지만 Linux 직접 후보 평가의 같은 셀 결과와 일치한다. 관련 테스트 17개가 통과했다.

현재 근거는 **로컬 완주율 개선 후보**를 지지한다. 다만 단일 기존 학습 checkpoint를 사용했고, Linux Docker 이미지는 공식 서버 자체가 아니며 대회 웹사이트는 2026-09-28에도 접속되지 않았다. 따라서 공식 점수, 공식 적합성 PASS, V2 SOTA 승격 또는 제출 성공은 주장하지 않는다. 세 게이트는 `rule_compliance=UNKNOWN`, `mechanism_activation=PASS`, `competitive_or_product_outcome=FAIL`(revision 2의 16/16 완주 목표 미달)로 검토하고 `GATE_REVIEW_REVISE → STOPPED`로 기록하며 RELEASE는 하지 않는다. 별도로 고정 셀의 로컬 완주율 개선은 측정됐다. 공식 사이트에 업로드하거나 모델을 확정하지 않았다.

## 추가 로컬 완주 실험 — 2026-09-28

사용자는 대회 사이트 업로드 없이 로컬 완주율이 좋아질 때까지 반복하라고 지시했다. 기존 PPO 체크포인트를 고정하고 장애물 회피 방향 유지, 도로 인식 신뢰도, 회피 후 복귀를 순서대로 조사했다. 모든 주행은 Linux CPU 컨테이너, 트랙 1·2, 셀별 2,000 decision 상한으로 실행했다. 기존 held-out·confirmation 및 blind seed 144–147은 조정에 사용하지 않았다. 아래 새 조정용 시드와 검증 시드의 원본 행은 각 V2 경로의 `events.jsonl`에 있고, `run_manifest.json`과 `integration_report.json`을 함께 남겼다. 직접 실행한 로컬 진단이며 V2 CLI 등록·릴리스나 공식 점수가 아니다.

| 후보·분리 셀 | 후보 완주 | 같은 셀 대조군 | 판단·근거 |
|---|---:|---:|---|
| 늦은 회피 방향 고정, 기존 tune 132–135 | 7/8 | 기존 hybrid 6/8 | `score-linux-commit-late-tune-20260928` |
| 늦은 고정, fresh tune 148–151 | 8/8 | hybrid 7/8 | `score-linux-commit-late-fresh-tune-20260928` |
| 늦은 고정, fresh held-out 156–159 | 7/8 | hybrid 7/8 | `score-linux-commit-late-fresh-heldout-20260928` |
| 도로 인식 보호, 소비된 tune 132–135 | 8/8 | 늦은 고정 7/8 | `score-linux-road-guard-consumed-tune-20260928` |
| 도로 인식 보호, fresh tune 160–163 | 7/8 | 늦은 고정 7/8 | `score-linux-road-guard-fresh-tune-20260928` |
| 도로 인식 보호, fresh held-out 164–167 | 7/8 | 늦은 고정 6/8 | `score-linux-road-guard-fresh-heldout-20260928` |
| 초기 장애물 위치 근거, fresh tune 168–171 | 8/8 | 도로 인식 보호 8/8 | `score-linux-side-evidence-fresh-tune-20260928` |
| 초기 위치 근거, fresh held-out 172–175 | 7/8 | 도로 인식 보호 7/8 | `score-linux-side-evidence-fresh-heldout-20260928` |
| 초기 위치 근거, 확장 fresh tune 176–183 | 15/16 | 별도 대조군 없음 | `score-linux-side-evidence-expanded-tune-20260928` |
| 완전한 도로 윤곽 요구, 소비된 확장 tune 176–183 | 16/16 | 초기 위치 근거 15/16 | `score-linux-full-road-guard-expanded-tune-20260928` |
| 완전한 도로 윤곽 요구, fresh tune 184–187 | 7/8 | 초기 위치 근거 7/8 | `score-linux-full-road-guard-fresh-tune-20260928` |
| 완전한 도로 윤곽 요구, fresh held-out 188–191 | 8/8 | 초기 위치 근거 8/8 | `score-linux-full-road-guard-fresh-heldout-20260928` |
| 최종 ZIP 그대로, 일회성 confirmation 192–195 | 7/8 | 대조군 없음 | `score-linux-full-road-guard-package-confirmation-20260928` |

같은 주행을 반복 조정한 결과와 새 검증 결과를 구분해야 한다. 소비된 tune의 16/16과 새 held-out의 8/8은 긍정적이지만, 포장된 동일 후보의 독립 confirmation은 7/8이었다. 따라서 모든 시드에서 완주한다는 근거는 없다. 확인 주행에서 action 오류 0회, 최대 act 17.75ms, 최대 셀 p95 11.33ms, 충돌 합계 1회였다. 다른 후보인 `obstacle_slow_runtime.py`, `obstacle_slew_runtime.py`, 조기 방향 고정, 정지 후 탈출, 도로 단독 주행, 가시성 상실 시 방향 초기화는 각각 조정용 비교 또는 실패 셀 진단에서 완주율 개선을 확인하지 못해 채택하지 않았다.

현재 최선의 [로컬 ZIP](../artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip)은 SHA-256 `5C55670AB5AEF4BF64115909792650A5E76BC3576A64F3AAEB3780EFABC18E40`, 16개 파일, 560,499바이트다. [패키지 매니페스트](../artifacts/haic-research-v2/full-road-guard-candidate-20260928/package_manifest.json)에 파일별 해시가 있다. 압축 해제 후 Linux Python 3.11 CPU에서 import·생성 약 0.008초, reset 약 0초, 단일 act 약 0.002초, RSS 약 247MB였고 샘플 주행을 완주했다. 정적 금지 구문·크기 검사는 통과했다. 이 검사는 공식 서버 적합성을 보증하지 않으며 공식 사이트 업로드는 하지 않았다. 남은 목표는 충돌 없는 코스 이탈과 조기 장애물 실패를 줄여 새로운 독립 확인 세트에서 8/8을 재현하는 것이다.

추가로 근거리 고속 장애물 제동을 `obstacle_guard_brake_runtime.py`에서 시험했다. 소비된 tune 184–187은 제동 8/8 대 현 ZIP 7/8, 새 tune 196–199도 8/8 대 7/8이었다. 그러나 새 held-out 200–203은 양쪽 모두 7/8이었고 제동 후보의 완주 중앙값은 30,660ms로 현 ZIP의 28,140ms보다 느렸다. 따라서 완주율 우선·동률 시 속도 비교에 따라 제동 후보를 채택하지 않았다. 근거는 각각 `runs/haic-research-v2/score-linux-guard-brake-consumed-tune-20260928/`, `score-linux-guard-brake-fresh-tune-20260928/`, `score-linux-guard-brake-fresh-heldout-20260928/`에 있다. 사용한 held-out와 confirmation 셀은 조정 자료로 되돌리지 않는다.

## 직선 가속과 도로 여유 폭 확인 — 2026-09-28

사용자 요청에 따라 장애물 없는 안정적인 직선에서 전가속하는 `straight_sprint_runtime.py`를 만들었다. 일곱 이미지 행의 도로 중심이 화면 중심에서 모두 2.5픽셀 이내이고 중심 변화가 2픽셀 이내인 상태가 8 decision 이어지며 픽셀 속도가 58 미만일 때 가속한다. 비교는 Linux CPU, 트랙 1·2, 셀당 최대 2,000 decision, 동일 셀의 full-road-guard 대조군으로 수행했다. 새 tune 208–211에서 양쪽 모두 8/8 완주했고 완주 랩타임 중앙값은 전가속 23,950ms, 대조군 25,480ms였다. 새 held-out 212–215에서 전가속 8/8, 대조군 7/8, 완주 중앙값은 각각 24,660ms와 27,220ms였다. 그러나 패키지 그대로 새 confirmation 216–219에서는 전가속이 **5/8**만 완주했다. 이 확인 셀은 조정에 사용하지 않는다. 전가속 ZIP은 `artifacts/haic-research-v2/straight-sprint-candidate-20260928/submission-straight-sprint.zip`에 보존했으나 현재 후보로 채택하지 않는다. 관련 원본은 `runs/haic-research-v2/score-linux-straight-sprint-tune-20260928/`, `score-linux-straight-sprint-heldout-20260928/`, `score-linux-straight-sprint-package-confirmation-20260928/`에 있다.

같은 직선 조건에서 가속 명령을 0.4로 제한한 `straight_boost_runtime.py`도 새 tune 220–223에서 비교했다. 대조군은 8/8, boost는 7/8 완주했고 boost의 완주 중앙값은 22,580ms, 대조군은 24,170ms였다. 완주율 우선 기준에 따라 boost의 held-out은 실행하지 않고 거부했다. 근거는 `runs/haic-research-v2/score-linux-straight-boost-tune-20260928/`이다.

사용자가 지적한 도로 경계 판정을 확인했다. `env_wrapper.py`의 `wrapper_off_track`은 기하학적 경계 이탈 즉시 판정이 아니다. 네 시뮬레이터 프레임을 묶은 한 decision의 보상이 음수이면 카운터를 올리고, 0 이상이면 0으로 되돌린다. 카운터가 100을 **초과**하면 종료한다. 시뮬레이터 `core/vendor/car_racing.py`는 방문하지 않은 새 도로 타일의 보상을 주고 매 원시 프레임에 0.1을 차감하므로, 이 표시는 실제로는 장시간 새 타일을 못 밟는 정체도 포함한다. `training/evaluate_closed_loop.py`는 다른 시뮬레이터 종료에 명시적 이유가 없을 때도 결과 `retire_reason`을 `off_track`으로 대체할 수 있다. 따라서 결과 행의 그 이름만으로 101회 음수 보상 종료라고 단정하지 않는다. `progress=1.0`도 결승선 통과와 동의어가 아니다. `core/finish_line.py`가 진행률 95% 이상과 유효한 방향의 결승선 통과를 별도로 요구한다. 로컬 환경의 종료 조건을 느슨하게 바꾸면 기존 비교와 달라지므로 변경하지 않았다.

가속 조건 자체는 보수적이었다. 이미 소비한 tune 셀 1/184의 픽셀 진단에서 도로 일곱 행이 모두 보이고 장애물이 없는 254프레임 중 102프레임이 중심 2.5픽셀 조건으로 제외됐다. 그중 91프레임은 가까운 이미지 행의 화면 중앙 픽셀에도 아스팔트가 있었다. 중앙 한 픽셀만으로 차체 전체의 안전을 증명할 수 없지만, 고정된 2.5픽셀 조건이 도로 폭을 활용하지 못한다는 근거다. 진단 원본은 `runs/haic-research-v2/road-margin-diagnostic-20260928/`에 있으며 소비한 tune 셀만 사용했다. 도로 폭을 이용한 새 메커니즘은 별도의 조정용 셀과 독립 검증을 거쳐야 한다. 현재 선택된 ZIP은 위 full-road-guard로 유지한다. 공식 사이트 업로드는 하지 않았다.

## 현 대회 사이트·주행 영상·속도 실험 — 2026-09-28

사용자가 새 사이트 `https://scholarships-hardwood-headers-influenced.trycloudflare.com/`와 팀 이름 `돈만 있으면`을 제공했다. 사이트의 공개 트랙은 1·2·3이고 각 팀의 트랙별 최고 완주를 표시한다. 확인 시점의 팀 최고 기록은 트랙 1 **24.06초**(제출 #6), 트랙 2 **29.06초**(제출 #6), 트랙 3 **30.58초**(제출 #7)였다. 선두 기록은 각각 12.58초, 16.30초, 14.64초였다. 공개 순위는 변동 가능하며, 이 숫자는 로컬 시드 비교의 공식 환산값이 아니다. 로그인 후 제출 페이지에는 최신 제출 #7까지 완료된 것으로 표시됐으나 **확정 모델은 제출 #2**로 표시됐다. 확정 상태는 바꾸지 않았고 업로드도 하지 않았다.

사이트에서 제출 #6의 트랙 1·2와 제출 #7의 트랙 3 주행 영상을 확인했다. 원본 MP4 SHA-256은 차례로 `AC7E27804177FFDAB6B0BF00AC873F6F528EAF84F5404EC76429DEEB6712C3B5`, `6EBB99A5F8D6E79F9DA5C26ED520A37880B3A164BC3B6626B1F88B5BB7C2EAC7`, `AD5A86D5DDE35D4F5CDEE670B1FDC2141EE786AE160F9EE534AC0D37718A2AE6`이다. 차량은 대체로 도로를 따라 완주하며 일부 코너·가장자리에서 잔디에 접근했다. 영상은 가속 명령 자체를 보여주지 않으므로 특정 페달값을 추정하지 않는다. 공개 1차 모의평가 영상은 오래된 제출을 담아 현재 제출 진단에는 사용하지 않았다.

잔디 2픽셀 여유를 허용한 전가속 `road_margin_throttle_runtime.py`는 소비된 tune 1/184에서 166회 활성화됐지만 진행률 98.86%에서 미완주했다. 실제 최고 속도는 69.34로 대조군 50.72보다 높았다. 기존 도로 폭 가속에 속도 제한을 더한 `road_margin_speed_cap_runtime.py`도 같은 셀에서 진행률 50%로 미완주했다. 조향을 함께 바꾸는 `speed_coupled_corridor_runtime.py`는 소비된 tune 1/184에서 22.02초 대 대조군 23.44초였지만, 새 tune 트랙 1·2·3 × seed 228–231에서 **9/12 완주 대 대조군 12/12**였다. 후보의 완주 랩타임 중앙값 25.32초 대 대조군 26.86초는 완주율 손실을 보상하지 못한다. 이 새 tune 셀은 이후 조정 자료로만 사용한다.

실패 진단에서 순수 도로 제어기의 1/228, 3/228, 1/230 종료 전 조향은 0으로 고정됐고 도로 타일 진행이 멈췄다. 이 결과만으로 `wrapper_off_track`인지 플레이 영역 이탈인지는 확정할 수 없다. 도로 윤곽이 불완전할 때 기존 학습 조향으로 돌아가는 `speed_coupled_fallback_runtime.py`는 소비된 tune 1/228과 1/230을 완주로 회복했지만 3/228은 미완주였다. 먼 도로 중심을 조향에 추가한 `lookahead_corridor_runtime.py`는 1/184를 21.54초, 1/228을 23.56초에 완주했지만 3/228은 미완주였다. 최고 목표 속도를 50으로 낮춘 `launch_lookahead_runtime.py`는 3/228을 완주했으나 1/228이 미완주, 1/230은 충돌 손상으로 리타이어했다. 소비된 tune에서 실패한 후속 후보들은 독립 held-out에 올리지 않았다. 관련 원본은 `runs/haic-research-v2/road-margin-throttle-probe-20260928/`, `road-margin-throttle-trace-20260928/`, `score-linux-road-margin-speed-cap-consumed-tune-20260928/`, `speed-coupled-corridor-probe-20260928/`, `score-linux-speed-coupled-corridor-tune-20260928/`, `speed-coupled-failure-trace-20260928/`, `speed-coupled-fallback-probe-20260928/`, `lookahead-fallback-probe-20260928/`, `launch-lookahead-probe-20260928/`에 보존했다. 평가 후 확장된 두 runtime의 원래 정확한 바이트는 `research/frozen/`에 따로 보존하고 해당 run의 `events.jsonl`에 출처 정정 이벤트를 추가했다.

현재 선택된 로컬 ZIP은 full-road-guard 그대로다. 영상은 더 넓은 도로 가장자리 주행의 가능성을 보여주지만, 이번 로컬 실험은 단순 가속과 현재 도로 추종 제어만으로 완주율을 유지하지 못했다. 새 속도 후보를 선택하려면 도로 가시성 상실 전의 조향·제동과 충돌 회피를 함께 개선하고 독립 세 트랙 검증을 통과해야 한다. 대회 사이트 업로드와 모델 확정 변경은 하지 않았다.

마지막으로 직선과 코너에서 제어를 나누는 `bend_safety_runtime.py`를 소비된 tune 3/228, 1/228, 1/184, 1/230에서 활성 확인했다. 각각 미완주 17.0%, 완주 28.62초, 완주 21.94초, 미완주 85.1%였으며 마지막 셀은 충돌 3회와 손상 0.6을 기록했다. 기존 후보가 네 셀 모두 완주하므로 이 방향도 채택하지 않았다. 원본은 `runs/haic-research-v2/bend-safety-probe-20260928/`에 남겼다. 세 트랙 새 tune을 통과한 속도 후보가 없으므로 새 held-out·confirmation은 열지 않았다.

## 3위권 기준과 공개 경쟁 영상 재검토 — 2026-09-28

08:44 UTC 공개 리더보드에서 트랙 1·2·3의 3위 완주 기록은 각각 **13.62·18.94·17.00초**, 우리 팀 최고 기록은 **24.06·29.06·30.58초**였다. 같은 표의 decision 수는 각각 171/301, 237/364, 213/383(3위/우리 팀)이다. 3위 시간까지 필요한 감소는 **43.4%·34.8%·44.4%**이다. 이는 변동 가능한 공개 트랙별 최고 기록이며 최종 종합 순위는 아니다.

[공개 1차 모의평가 트랙 3 영상](https://scholarships-hardwood-headers-influenced.trycloudflare.com/mock-videos/ad8e8d53-10bf-47c5-85c3-77d65123abe8)을 다시 확인했다. 화면에는 14대의 독립 ghost 주행이 합성되어 있고 2026-09-24 당시 1위 15.62초, 우리 팀 50.56초가 표시된다. 현재 상위 제출의 조작값이나 내부 방식은 공개 영상으로 알 수 없다. 그 한계를 포함한 네 방향의 새 설계와 **결승선 통과 기준**은 [3위권 성능 재설계안](plans/2026-09-28-top-three-performance-rebuild.md)에 고정했다. 현재는 재설계·목표 설정 단계이며 새 완주 성적이나 3위권 기록을 달성한 것은 아니다. 대회 사이트 업로드나 모델 확인 변경은 하지 않았다.

첫 방향의 최소 활성 검사는 이미 소비한 tune 셀 1/228, 3/228, 1/230, 1/184에서 했다. 먼 도로를 보고 코너를 일찍 조향하고 도로가 영상 밖으로 사라지면 직전 조향을 잠시 유지하며 제동하는 `PredictiveCorridorAgent`는 미리보기 조향과 도로 상실 복구가 실제로 작동했지만 **0/4 완주**, 대조군은 **4/4 완주**였다. 후보 진행률은 각각 17.35%, 17.35%, 84.80%, 51.14%였다. 원본은 `runs/haic-research-v2/predictive-corridor-activation-20260928/`에 있다. 이 진단은 새 독립 비교가 아니지만 명백한 조기 실패이므로 이 후보를 새 tune이나 held-out으로 보내지 않는다. 특히 현재 방식의 단순 조향 기억을 계속 미세 조정하기보다, 학습용 빠른 경로 교사와 속도 상태 학습에서 **완주 가능한 고속 행동 자체**를 먼저 만들어야 한다.

학습용 경로 교사는 **train 전용** 공식 환경 셀 1/43과 2/102에서 시뮬레이터 경로·차량 상태를 읽어 시험했다. 이 상태는 제출 추론에는 사용하지 않았고 교사 결과는 제출 후보 완주율에 포함하지 않는다. 첫 순수 목표점 추종은 0/2였으며 급커브에서 과회전했다. 미리보기 거리를 줄이고 커브 속도를 낮춘 V2는 1/43을 **37.76초에 완주**했지만 2/102는 23.3%에서 미완주했다. 곡률을 바탕으로 미래 속도를 계획하고 차량 방향·도로 중심 오차를 반영한 V3는 0/2, 회전 속도 감쇠를 추가한 V4도 0/2, V2 조향과 미래 속도 계획을 결합한 V5도 0/2였다. 최대 속도는 75–91까지 도달해도 코너에서 진로를 유지하지 못했다. 따라서 현재 교사는 빠른 모방 학습 자료로 채택하지 않는다. 각 버전의 불변 manifest·episode events·integration report는 `runs/haic-research-v2/fast-route-teacher-*-20260928/`에 있고, V1/2 실패 구간의 궤적도 별도 trace run에 남겼다. 후속은 같은 조향 규칙을 더 조절하기보다 **픽셀 정책이 속도 상태를 학습하는 방향**을 우선한다.

## 고속 학습 정책 재사용과 안전 제어 — 2026-09-28

예전 고속 PPO 체크포인트의 메타데이터는 현재 로더가 예상한 평면 구조가 아닌 중첩 `model_config`였다. 첫 진단은 한 episode 후 로딩 오류로 중단되어 `historical-speed-policy-consumed-tune-20260928`에 **인프라 무효**로 기록했다. 구조를 맞춰 다시 로딩한 소비된 tune 4셀(1/184, 1/228, 3/228, 1/230)에서 기존 full-road-guard는 4/4, 고속 seed8104는 3/4, seed8105는 2/4 완주했다. seed8104는 완주 시 더 빨랐으나 1/230에서 마지막 도로 타일 방문 후 74 decision 동안 진행하지 못하고 미완주했다. `historical-speed-policy-nested-consumed-tune-20260928`와 `historical-speed-seed8104-failure-trace-20260928`에 원본을 보존했다.

픽셀 도로가 불완전하거나 가까운 도로 중심이 크게 벗어났을 때 기존 안정 정책으로 전환하고 감속하는 후보를 만들었다. 소비된 4셀에서는 대조군과 함께 4/4 완주했고 후보가 네 셀 모두 빨랐다. 그러나 **새 tune 트랙 1·2·3 × seed 260–263**에서는 후보 **10/12**, 대조군 **11/12**였다. 후보의 완주 중앙값 23.31초가 대조군 24.84초보다 짧아도 완주율 우선 기준으로 탈락시켰다. 후보만 실패한 3/262는 진행률 98.91%에서 `finish_time_s` 없이 종료했다. 소비된 해당 셀의 픽셀·행동 trace에서 마지막 새 도로 타일 방문은 step 438, 이후 100회 이상 진행률이 같았고 마지막 구간에 도로 일곱 행이 보이고 가속한 사실을 확인했다. 이는 결승선 통과 실패이지 완주가 아니다. 근거는 `fast-policy-safety-consumed-tune-20260928`, `score-linux-fast-policy-safety-fresh-tune-20260928`, `fast-policy-safety-finish-trace-20260928`이다.

다음에는 기존 정책 조향을 유지하고 고속 정책의 페달만 도로가 충분히 보이는 구간에 사용하는 방향을 시험했다. 소비된 4셀은 양쪽 4/4, 후보 완주 중앙값 23.59초 대 대조군 25.94초였다. 하지만 **새 tune 트랙 1·2·3 × seed 272–275**에서 후보 **11/12**, 대조군 **12/12**였고 후보의 충돌 6회·손상 1.2 대 대조군 충돌 1회·손상 0.2였다. 후보 완주 중앙값 24.06초 대 26.16초의 속도 이득도 완주율 손실을 보상하지 못한다. 특히 후보만 1/273을 진행률 22.7%에서 미완주했다. `fast-pedal-stable-steer-consumed-tune-20260928`와 `score-linux-fast-pedal-stable-steer-fresh-tune-20260928`에 manifest·전체 episode events·integration report가 있다. 각 후보는 새 held-out·confirmation으로 올리지 않았고 264–271 및 276–283 예약 셀을 열지 않았다. 모든 점수는 로컬 프록시이며 공식 점수가 아니다. 선택된 ZIP은 full-road-guard 그대로이고 대회 사이트 업로드·모델 확정은 하지 않았다.

## TRAIN 전용 속도 상태 PPO 재학습 — 2026-09-28

등록된 로컬 학습 프로필에서 선택 체크포인트 SHA-256 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`만 읽기 전용 재개할 수 있도록 좁은 입력 검증을 추가했다. 픽셀 특징 사용과 스로틀 범위 4배 옵션도 프로필에 등록하고 관련 테스트를 통과시켰다. 정확한 가설·TRAIN 분리·예산·후속 tune seed는 [속도 상태 학습 계획](plans/2026-09-28-speed-state-ppo-continuation.md)에 있다. 첫 run `speed-state-ppo-continuation-20260928`은 체크포인트의 `use_hud=false`와 실행 기본값 `true`가 달라 학습 전 중단됐다. 실패를 세 게이트와 함께 `REVISE → STOPPED`로 기록했고 성능 주기로 세지 않았다.

모델 설정만 수정한 `speed-state-ppo-continuation-hudfix-20260928`은 Linux/Python 3.11 CPU에서 TRAIN 맵만 사용해 4,096 environment decisions, PPO 4회를 약 151초에 완료했다. 체크포인트는 `artifacts/haic-research-v2/speed-state-ppo-continuation-hudfix-20260928/training/policy.pt`에 있다. 스로틀 상한은 0.12에서 0.48로 열렸지만, 마지막 TRAIN rollout의 **평균 gas 0.0674, 평균 속도 30.32**, 최대 gas 0.2154였다. 첫 업데이트도 평균 gas 0.0651, 평균 속도 32.11이었다. 각 업데이트의 TRAIN 코스 구성이 달라 속도 두 숫자는 직접적인 matched 개선 비교가 아니지만, 정책이 넓어진 가속 범위를 적극 쓰게 됐다는 근거는 없다. 이전 체크포인트와 새 모델의 `policy_mean.weight` 상대 L2 변화는 0.0601%였다. 이 4,096 decision으로는 빠른 순항 행동 학습이 확인되지 않아 **mechanism_activation=FAIL, REJECT → STOPPED**로 마쳤다. TRAIN 결과는 제출 후보의 완주율에 넣지 않았고 새로운 tune 284–287, held-out 288–291, confirmation 292–295는 열지 않았다.

이 결과는 단순 페달 상한 확대와 현재 속도 부족 보상만으로 학습된 정책의 속도 선호를 크게 바꾸기 어렵다는 근거다. 다음 주기는 `policy_mean` 변화와 코너 실패의 인과 경로를 분리하여 학습 신호 또는 행동 학습 방식을 바꿔야 한다. 기존 full-road-guard ZIP이 계속 선택되어 있으며 사이트 업로드·모델 확정 변경은 없다.

## 최신 1~3위 기록과 가속 전략 재검토 — 2026-09-28

09:59 UTC 무렵 공개 순위를 다시 읽었다. 아래는 각 트랙에서 팀별 **최고 단일 완주**이며 같은 날짜의 고정 결과가 아니다. 괄호 안은 시뮬레이션 단계 수다.

| 공개 트랙 | 1위 바이브코더 | 2위 외국인 | 3위 탑저그예요 | 우리 팀 돈만 있으면 |
|---|---:|---:|---:|---:|
| 1 | 12.58초 (158) | 13.58초 (170) | 13.62초 (171) | 24.06초 (301), 13위 |
| 2 | 16.30초 (204) | 17.46초 (219) | 17.88초 (224) | 29.06초 (364), 12위 |
| 3 | 14.64초 (184) | 15.44초 (193) | 17.00초 (213) | 30.58초 (383), 8위 |

3위 기록까지 필요한 랩타임 감소는 트랙별 43.4%, 38.5%, 44.4%다. 세 팀 모두 세 트랙에서 우리보다 약 39~48% 적은 단계로 완주한다. 공개 표가 보여 주는 것은 빠른 평균 진행 속도와 세 트랙의 단일 완주 성적이지 가속·브레이크·조향 명령이나 재현 완주율이 아니다. 공개 영상 목록에는 9월 24일 기준 1차 모의평가만 있었으며, 최신 1~3위 제출 영상을 확인하지 못했다. 이 기록만으로 잔디를 밟았는지, 어떤 코너에서 감속했는지, 내부 정책이 무엇인지를 단정하지 않는다.

현 로컬 정책의 가속 명령 상한을 0.12→0.48로 연 PPO 재학습은 마지막 TRAIN rollout 평균 gas 0.0674에 머물렀다. 반면 직선 전가속과 도로 폭을 이용한 가속은 실제 속도·일부 랩타임을 높였지만 독립 확인의 완주율을 잃었다. 따라서 다음 로컬 방향은 단순 상한 확대가 아니라 **가속 명령을 실제로 높이는 행동 변화**와 **코너 진입 전 예측 제동·회복**을 함께 다루는 것이다. 먼저 TRAIN 전용 셀에서 gas와 속도, 제동 선행 시간, 도로 상실·결승선 통과를 계측하고, 그다음 새 세 트랙 tune에서 완주율과 랩타임을 대조군과 비교한다. 공개 3위권 기록은 목표값으로만 사용하고 트랙별 최고 기록을 학습 데이터나 검증 성공으로 취급하지 않는다. 기존 ZIP과 대회 사이트 상태는 바꾸지 않았다.

사용자가 목표를 **상위 팀의 이동 패턴을 따라 하는 것**으로 구체화했다. 이에 페달 빈도 진단을 우선 작업에서 내리고 공개 [1차 모의평가 트랙 3 영상](https://scholarships-hardwood-headers-influenced.trycloudflare.com/mock-videos/ad8e8d53-10bf-47c5-85c3-77d65123abe8)을 프레임별로 살폈다. 영상 범례에는 현재 공개 1위 `바이브코더`(T15)와 3위 `탑저그예요`(T17)가 있지만 2위 `외국인`은 없다. 이 영상은 현재 최고 제출보다 오래된 2026-09-24 경기다. 당시 T15는 첫 U자 구간 진입 때 도로 한가운데만 고집하지 않고 바깥쪽에서 들어와 V자 구간의 안쪽에 근접한 경로를 지나갔다. T17도 V자 구간 진입에서 안쪽에 가까웠다. 화면에서 보이는 순간의 위치와 궤적에 한정된 관찰이며, 바퀴가 아스팔트를 벗어났는지, 그 선이 최신 정책에도 유지됐는지, 각 차량의 조향·페달 명령은 확인할 수 없다. 당시 선두 T15는 15.62초에 완주했지만 우리 팀의 영상 당시 차량과 최신 차량은 다르므로 그 격차를 현 정책의 직접 비교로 사용하지 않는다.

이 관찰로 다음 메커니즘을 **픽셀 기반 코너 진입 주행선**으로 바꾼다. 보이는 도로의 좌우 경계와 먼 도로의 방향으로 턴인 방향을 잡고, 진입에서는 바깥쪽 여유를 쓰며 코너 가운데에서 안쪽으로 목표점을 옮긴 뒤 출구에서 도로를 넓게 쓴다. 단순히 브레이크를 줄이거나 기존 중심선에서 풀가속하는 방식과 분리해서 검증한다. 먼저 이미 소비한 tune 셀에서 이동선의 변화와 차체의 도로 여유를 확인하고, 메커니즘이 작동한 경우에만 새 세 트랙 tune에서 결승선 통과율과 랩타임을 비교한다. 2위 차량의 실제 주행선과 최신 1·3위 제출 주행선은 여전히 미확인이다.

## 코너 안쪽 주행선 3트랙 진단 — 2026-09-28

첫 구현은 전체 진입·출구 궤적이 아니라, 픽셀에서 코너 방향과 가까운 도로 폭을 읽어 **코너 안쪽으로 조향을 추가하는 작은 단계**다. 장애물이 보이거나 도로 일곱 행이 보이지 않으면 기존 정책을 유지한다. 고속 페달과 안정 조향을 결합한 기준 후보와 별도로 비교했다. `apex-line-three-track-fresh-tune-20260928`의 첫 실행은 공유 작업 공간에서 실행 코드가 도중 바뀌어 사후 소스 해시 검증에 실패했으므로 인프라 무효다. 그 기록은 성능 근거로 채택하지 않는다.

코드를 고정 복사한 `apex-line-three-track-frozen-retry-20260928`에서 같은 tune 트랙 1·2·3 × seed 296–299를 다시 평가했다. 세 후보 모두 같은 12셀을 달렸다. 선택된 full-road-guard는 **12/12 완주, 완주 랩 중앙값 23.39초**, 고속 페달·안정 조향은 **11/12, 21.68초**, 코너 안쪽 주행 후보는 **12/12, 21.70초**였다. 코너 후보는 609번 조향을 변경했고 충돌·손상·무효 행동이 모두 0이었다. 트랙별 코너 후보는 각각 4/4 완주, 중앙값 21.70·21.68·21.76초였다. `ADVANCE → GATE_REVIEW_ADVANCE → STOPPED`, release 없음으로 기록했다. 이는 로컬 tune 진단이며 공식 점수나 SOTA 승격이 아니다. 새 held-out과 confirmation 검증, 공식 패키지 확인이 남아 있고 공개 3위권의 12~17초대 기록에도 아직 못 미친다. 고정 소스와 실행 기록은 `tmp/apex-line-frozen-20260928/`, 통합된 불변 실행 기록은 `runs/haic-research-v2/apex-line-three-track-frozen-retry-20260928/`, 전체 주행 결과는 `artifacts/haic-research-v2/apex-line-three-track-frozen-retry-20260928/report.json`에 있다.

이어 **새 held-out 트랙 1·2·3 × seed 300–303**을 같은 고정 코드로 열었다. 선택 모델은 **12/12 완주·25.06초**, 코너 후보는 **11/12·23.26초**, 고속 페달 기준 후보는 **10/12·23.37초**였다. 코너 후보는 충돌 1회·손상 0.2·무효 행동 0, 안쪽 조향 변경 651회를 기록했다. 트랙 3/303에서 진행률 60.78%를 넘지 못하고 도로 이탈로 종료했고, 선택 모델은 그 셀을 완주했다. 코너 후보와 고속 페달 후보는 그 지점에서 비슷한 궤적으로 도로를 잃었다. 완주율 우선 규칙에 따라 코너 후보는 **REJECT → GATE_REVIEW_REJECT → STOPPED**, release 없음이다. 빠른 완주 중앙값을 선택 후보 교체 근거로 쓰지 않는다. 다음 연구는 이 지점의 진입 속도와 주행선 여유를 픽셀에서 미리 판별해 완주율을 회복시키는 것이다. 새 held-out 셀은 소비됐고 confirmation은 열지 않았다. [held-out 실행 보고서](../artifacts/haic-research-v2/apex-line-three-track-heldout-20260928/report.json)와 `runs/haic-research-v2/apex-line-three-track-heldout-20260928/`에 원본 기록이 있다.

## 사용자 우선순위: 실제 상위 팀 주행 방식 확인

2026-09-28 약 22:30 KST의 공개 리더보드와 영상 목록을 다시 확인했다. 트랙 1·2는 `바이브코더`·`외국인`·`탑저그예요` 순서다. 트랙 3은 `바이브코더`·긴 이름의 T7 팀·`외국인` 순서로 바뀌었다. 공개 영상은 과거 1차 모의평가 트랙 3 한 편이며 현 `외국인` 차량은 등장하지 않는다. T15 `바이브코더`는 V자 굴곡을 화면상 도로 안에서 지나 먼저 출구에 도달했고, T17 `탑저그예요`는 굴곡의 안쪽 경계 가까이를 지났다. **선두가 항상 안쪽 경계를 깎아 달린다는 증거는 확인되지 않았다.** 이전 코너 안쪽 조향 가설을 상위 팀의 실제 방식으로 단정하지 않고, 페달·속도·조향 명령과 최신 최고 제출의 주행선은 미확인으로 둔다. 구간별 사실과 팀별 정보 한계는 [상위 팀 공개 주행 감사](plans/2026-09-28-top-three-public-driving-audit.md)에 기록했다.

기존 세 팔의 동일 tune 셀을 도달 시각으로 분해하니, 선택 모델 대비 빠른 페달 팔은 진행률 10%에서 중앙값 **0.72초**, 95%에서 **1.60초** 앞섰다. 빠른 페달 팔 대비 안쪽 조향의 추가 이득은 같은 완주 셀의 랩 중앙값 **0.04초**에 불과했다. 첫 10%의 평균 gas와 속도는 선택 모델 `0.085 / 24.4`, 빠른 페달 팔 `0.171 / 32.0`이었다. 따라서 로컬 1.7초 개선의 주된 관찰 신호는 안쪽 경계 공략보다 강한 초반 가속이다. 이것을 1위 팀의 실제 명령이라고 추론하지 않는다.

시드 무결성도 재확인했다. `apex-line-three-track-heldout-20260928`이 300–303 셀을 13:16 UTC에 먼저 등록하고 13:24 UTC에 마쳤다. 별도 `anticipatory-bend-fresh-tune-20260928`은 **같은 12셀**을 13:27 UTC에 `fresh tune`으로 등록했다. 후자 결과는 실행 성공 여부와 별개로 프로젝트 전체에서는 소비 셀 진단이며 새 독립 근거가 아니다. 원본 run은 그대로 보존하고 [시드 중복 감사](plans/2026-09-28-seed-300-303-identity-audit.md)에 해석을 기록했다.

## 앞보기 조향 ZIP 독립 확인 — 2026-09-28

사용자가 제안한 굴곡 선행 조향은 멀리 보이는 도로 방향이 일관되고 기존 정책이 아직 거의 직진할 때만 작은 조향을 먼저 준다. 가속·제동은 그대로 둔다. 중복된 300–303 결과는 성능 선정 근거에서 제외했다. 별도로 등록한 304–307 held-out에서는 후보와 대조군이 모두 **11/12 완주**했다.

고정한 후보 ZIP을 깨끗한 Linux Python 3.11 CPU 디렉터리에서 추출해 **트랙 1·2·3 × seed 308–311**로 비교했다. 후보는 **11/12**, 기존 ZIP은 **10/12 완주**했다. 후보만 미완주한 셀은 없고 앞보기 조향은 109번 작동했으며 충돌은 **0 대 6회**였다. 완주 랩 중앙값은 후보 **26.28초**, 기존 **25.93초**로 후보가 느렸다. 다만 다른 팀 실험이 같은 308–311 셀을 이 실행 전에 이미 열었음을 확인했다. 원래 보고서의 `ADVANCE`는 독립 확인 결론으로 쓰지 않고, 이 결과를 **소비 셀 작동 진단**으로만 보존한다. 무효 행동 0, 후보의 import·생성 최대 1.29초, 메모리 최대 301MB, 행동 최대 36.66ms였다. 세부 수치와 시드 중복 정정은 [실험 기록](experiments/anticipatory-bend-20260928.md)과 `runs/haic-research-v2/anticipatory-bend-package-confirmation-20260928/`에 있다. 후보 ZIP은 로컬 산출물이며 대회 사이트에 올리거나 확인 모델을 바꾸지 않았다. 308–311은 후속 독립 검증에 재사용하지 않는다.

## 앞보기 조향 후 속도 증가와 도로 이탈 점검 — 2026-09-28

앞보기 조향을 고정한 채 가속 조건만 세 차례 좁혀 확인했다. 평가 때만 바퀴의 도로 접촉과 연속 불리한 보상 상태를 읽었고, 제출 추론 코드는 픽셀과 행동만 사용했다. 넓은 가속 조건은 새 tune 1700–1703에서 후보 **11/12**, 앞보기 조향 **12/12 완주**였고 후보만 트랙 2/1700에서 도로 이탈했다. 그 가속은 실제 조기 조향 순간과 한 번도 겹치지 않아 의도한 연결이 약했다. 더 안정적인 도로 조건의 0.13 가속은 tune 1724–1727에서 후보 **12/12 대 10/12**였지만, 사전 등록한 한 조건의 바퀴 이탈 비율 한계를 0.15%포인트 넘겨 `REJECT`했다.

가속 상한을 0.12로 낮추고 작은 조향과 거의 곧은 도로가 7번 연속 유지될 때만 적용한 최종 후보는 새 tune 1800–1803에서 양쪽 **11/12 완주**했고 도로 이탈 한도를 지켰다. 코드를 바꾸지 않은 held-out 1804–1807에서도 양쪽 **12/12 완주**, 충돌·손상·무효 행동 모두 0이었다. 후보 평균 속도는 **40.99 대 40.70**으로 조금 높았고 바퀴 한 개 이상 이탈 비율은 **10.41% 대 10.01%**, 바퀴 전체 이탈 최장 연속 길이는 양쪽 **47 결정**이었다. 큰 이탈 증가는 없었지만 완주 중앙값은 후보 **26.04초**, 앞보기 조향 **25.96초**로 개선되지 않아 `REVISE`다. 예약된 confirmation 1808–1811은 열지 않았다. [전체 실험 및 파일](experiments/anticipatory-steering-speed-20260928.md)에 첫 실패, 인프라 무효 재포장, 각 조건의 원본 기록을 연결했다. 공식 사이트 결과는 아니다.

**교차 실행 감사 정정:** 위 308–311은 앞보기 ZIP 확인 계획보다 먼저 `haic2-four-direction-fresh-tune-20260928`에서 13:34 UTC에 실행을 시작했다. 앞보기 ZIP 확인 manifest는 13:44 UTC에 생성됐다. 따라서 위 수치는 남기되 **독립 confirmation 통과 및 신규 release 근거로 사용하지 않는다**. 해당 ZIP을 다시 고려하려면 새 시드에서 독립 확인이 필요하다. 원본 보고서와 `ADVANCE` 기록은 변경하지 않았고 [시드 중복 감사](plans/2026-09-28-seed-300-303-identity-audit.md)에 이 정정의 근거를 남겼다.

## 네 방향 속도·코너 비교 — 2026-09-28

고정 소스의 등록된 `haic2-four-direction-fresh-tune-20260928`이 트랙 1·2·3 × seed 308–311에서 여섯 팔 72회 주행을 마쳤다. 선택된 안전 대조군은 **10/12 완주·25.93초**, 이전 apex 대조군은 **9/12·24.64초**였다. 새 네 후보는 코너별 가속 하한 `speed_envelope` **7/12·22.90초**, 출구 가속 `exit_burst` **6/12·23.95초**, 접근 제동 `approach_brake` **9/12·25.76초**, 진입 바깥선 `outer_entry` **9/12·23.96초**였다. 네 방향 모두 실제 행동 변경은 발생했고 무효 행동은 없었지만, 완주율이 대조군보다 낮아 `REJECT → GATE_REVIEW_REJECT → STOPPED`, release 없음이다. 특히 진행률이 100%로 표시됐어도 `finish_time_s` 없이 도로 이탈한 셀은 미완주로 세었다. 이 결과는 로컬 tune 진단이며 공식 점수가 아니다. 새 held-out과 confirmation은 열지 않았다. [원본 실행 기록](../runs/haic-research-v2/haic2-four-direction-fresh-tune-20260928/integration_report.json)과 [전체 결과](../artifacts/haic-research-v2/haic2-four-direction-fresh-tune-20260928/report.json)를 보존했다.

## 앞보기 조향의 새 독립 진단 — 2026-09-28

시드 충돌이 없음을 v2 manifest와 계획에서 확인한 뒤, 앞보기 조향을 선택 안전 모델과 같은 고정 코드에서 **새 트랙 1·2·3 × seed 312–315**로 비교했다. 양쪽 모두 **12/12 실제 결승선 통과**, 충돌·손상·무효 행동 0이었다. 앞보기 조향은 91번 작동했지만 완주 랩 중앙값이 후보 **25.01초**, 대조군 **24.40초**였다. 완주율 동률에서 시간 열세이므로 등록 기준에 따라 `REJECT → GATE_REVIEW_REJECT → STOPPED`, release 없음이다. 기존의 시드가 겹친 308–311에서 보였던 후보 11/12 대 10/12는 이 새 결과를 대체하지 못한다. 이번 실행은 진단 프로필이라 공식 패키지 점수도 아니다. [계획](plans/2026-09-28-anticipatory-fresh-confirmation.md), [원본 실행 기록](../runs/haic-research-v2/anticipatory-bend-independent-confirmation-20260928/integration_report.json), [전체 결과](../artifacts/haic-research-v2/anticipatory-bend-independent-confirmation-20260928/report.json)를 남겼다.

## 도로 여유 가속 후보와 실제 ZIP 완주 검증 — 2026-09-28

새 안정 기반 도로 여유 가속 후보는 고정 소스의 세 로컬 제어기 비교에서 tune 324–327 **12/12 대 12/12, 중앙값 23.22 대 24.70초**, held-out 328–331 **11/12 대 9/12, 22.36 대 24.98초**, one-time confirmation 332–335 **12/12 대 11/12, 22.71 대 24.26초**로 유망했다. 그러나 정확한 ZIP을 만든 뒤 새로운 336–339에서 직접 실행하니 후보 **11/12**, 선택 full-road-guard ZIP **12/12**였다. 후보의 완주 중앙값은 23.54초로 대조군 25.23초보다 빠르지만, 트랙 2/336에서 후보만 58.62% 도로 이탈로 미완주해 완주율 우선 기준으로 **ZIP은 탈락**한다. 현재 선택 ZIP을 교체하거나 사이트에 올리지 않았다.

첫 ZIP 비교는 완전한 결과를 썼지만 공유 루트에 실행 중 새 소스 파일이 추가되어 하네스의 사후 소스 검증이 실패했다. 별도 고정 스냅샷 비교도 같은 336–339 셀과 동등한 제어기에서 동일한 11/12 대 12/12를 보였고 `REJECT → GATE_REVIEW_REJECT → STOPPED`를 기록했다. 두 실행을 독립된 두 확인으로 세지 않으며, 336–339는 소비된 확인 셀이다. 원본과 정확한 시각·해시·규칙 준수 한계는 [실험 기록](experiments/stable-risk-exact-package-20260928.md)에 있다. 다음 후보는 이 실패 셀을 조정에 쓰지 않고 새 TRAIN 조건에서 도로 재획득과 급커브/장애물 위험 신호를 실험한다.

## 공개 1~3위 방식 우선 재확인 — 2026-09-29 00:17 KST

사용자 요청에 따라 새 로컬 후보의 결과 해석보다 공개 상위 팀의 **실제 이동 패턴** 확인을 우선했다. [트랙 1](https://scholarships-hardwood-headers-influenced.trycloudflare.com/api/ranking?trackId=1), [트랙 2](https://scholarships-hardwood-headers-influenced.trycloudflare.com/api/ranking?trackId=2), [트랙 3](https://scholarships-hardwood-headers-influenced.trycloudflare.com/api/ranking?trackId=3) 순위 API와 [영상 목록](https://scholarships-hardwood-headers-influenced.trycloudflare.com/api/mock-videos?page=1&limit=12)을 읽기 전용으로 다시 확인했다. 순위 응답의 생성 시각은 2026-09-28 15:17 UTC다.

| 트랙 | 1위 | 2위 | 3위 | 우리 팀 |
|---|---:|---:|---:|---:|
| 1 | 바이브코더 12.48초·157단계 | 탑저그예요 13.04초·164단계 | 외국인 13.58초·170단계 | 24.06초·301단계 |
| 2 | 바이브코더 16.14초·202단계 | 외국인 17.46초·219단계 | 탑저그예요 17.88초·224단계 | 29.06초·364단계 |
| 3 | 바이브코더 14.36초·180단계 | 긴 이름의 T7 팀 15.32초·192단계 | 외국인 15.44초·193단계 | 30.58초·383단계 |

1위의 세 기록은 같은 제출 ID 29에서 나왔다. 공개 영상 목록에는 9월 24일의 트랙 3 모의평가 한 편만 있고 2차 모의평가 영상은 없다. 따라서 현재 최고 제출의 코너별 경로, 가속·제동·조향 명령은 확인되지 않는다. [과거 영상 관찰](plans/2026-09-28-top-three-public-driving-audit.md)에서 1위 차량은 U·V자 굴곡을 도로 위로 지나 출구에 먼저 도달했고, 당시의 탑저그예요 차량은 V자 안쪽 경계에 가까웠다. 외국인 차량은 그 영상에 없다. 이를 현재 정책의 재현 영상으로 취급하지 않는다.

단계 수는 3위권과 우리 기록의 속도 격차가 대략 1.6~2배임을 보여 주지만, 감속을 덜 했는지 또는 어느 주행선을 택했는지는 분리하지 못한다. 따라서 다음 로컬 가설은 **코너 진입·출구의 화면상 이동선과 구간별 도달 시간**을 실제로 기록하고, 더 강한 가속이 그 이동선을 유지하면서 완주하는지 검증하는 것이다. 상위 팀의 공개 영상을 새로 확인할 수 있을 때까지 특정 페달 전략을 상위 팀의 방식으로 주장하지 않는다. 사이트 업로드·모델 확인은 하지 않았다.

## 잔디 주행 제한과 완주 우선순위 재검토 — 2026-09-29

완료된 `speed-coupled-preview-tune-20260929`의 불변 [통합 결과](../runs/haic-research-v2/speed-coupled-preview-tune-20260929/integration_report.json)와 [셀별 사건](../runs/haic-research-v2/speed-coupled-preview-tune-20260929/events.jsonl)을 다시 확인했다. 같은 12개 tune 셀에서 속도 후보는 **12/12 결승선 통과**, 고정 미리보기 기준은 **11/12**였고 완주 랩 중앙값은 각각 23.75초와 24.30초였다. 유효하지 않은 행동은 양쪽 모두 0, 후보 충돌·손상은 0, 기준 충돌 2·손상 0.4였다. 그럼에도 사전 등록한 바퀴 도로 접촉의 셀별 상한 위반 때문에 원래 결정은 `REJECT`이며 그대로 보존한다.

문제 셀인 트랙 1/2002에서 후보는 모든 바퀴가 아스팔트 밖인 최장 연속 구간이 49결정, 어느 바퀴든 밖인 비율이 23.2%였으나 27.18초에 완주했다. 기준은 같은 셀에서 최장 2결정·5.9%였고 충돌 2회 뒤 진행률 61.6%에서 중단됐다. 이 셀은 도로 접촉 상한이 **실제 완주와 반대로 후보를 탈락시킨 사례**다. 다만 49결정 이탈이 다른 시드에서 안전하다는 증거는 아니다. 다음 새 비교에서는 도로 접촉을 진단으로 기록하고, 결승선 통과율·충돌·손상으로 위험을 검증하는 별도 계획을 세운다. 소비한 2000–2003 셀을 새 독립 근거로 재사용하거나 종전 결정을 소급 변경하지 않는다.

별도로 실행 중이던 고정 소스 4방향 위험 회복 tune도 종료됐다. [원본 비교 보고서](../tmp/haic2-risk-recovery-batch-frozen-20260929/artifacts/haic-research-v2/haic2-risk-recovery-four-direction-tune-20260929/report.json)의 트랙 1–3 × seed 344–347에서는 `stable_risk_envelope`가 **12/12**, 코너 접근·도로 가장자리·횡방향 이동 후보가 각각 **10/12**, 장애물 접근 후보가 **11/12** 완주했다. 네 후보 모두 완주 우선 기준에서 뒤졌으므로 이 결과만으로 다음 후보를 선택하지 않는다. 별도 작업의 통합 게이트와 소스 검증 결과는 아직 여기서 주장하지 않는다.

## 고속 ZIP과 현재 선택 ZIP의 새 held-out 비교 — 2026-09-29

[사전 계획](plans/2026-09-29-speed-coupled-exact-zip-heldout.md)에 후보 SHA-256 `54c117287390704acc52505d680c31574fdcf53ac062cbb0ce5258a57058c324`, 현재 선택 full-road-guard SHA-256 `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40`, 트랙 1–3 × 미사용 held-out seed 2004–2007, 12개 짝지은 셀, 2,000결정 상한을 등록했다. Linux CPU에서 계획 해시 `9bf35dcd9a93689063e6c5aca049ce1b96c65fb15f0095d92afa8ef2b8daa53c`에 대한 설계·구현·실행 승인 기록 뒤 정확한 ZIP 파일을 추출해 주행시켰다. [원본 보고서](../artifacts/haic-research-v2/speed-coupled-exact-zip-heldout-20260929/report.json)에 24회 전 결과가 있다.

고속 후보는 **11/12 실제 결승선 통과**, 현재 선택 ZIP은 **12/12**였다. 후보의 완주 랩 중앙값은 24.14초, 기준은 25.15초였지만, 후보만 트랙 3/2006에서 진행률 77.86%에 멈추고 `off_track`으로 종료됐다. 그 셀의 후보 충돌 2회·손상 0.4, 기준 충돌·손상 0회였다. 무효 행동은 양쪽 0, 후보의 최대 import+생성 1.01초·최대 act 29.33ms·최대 RSS 약 302MB였다. 도로 접촉 자체에 새 자동 탈락 기준을 두지 않았는데도 **완주율에서 기준에 못 미쳤다**. 이 held-out 셀은 이후 튜닝 데이터로 바꾸지 않는다.

등록된 통합 결정은 [REJECT → GATE_REVIEW_REJECT → STOPPED](../runs/haic-research-v2/speed-coupled-exact-zip-heldout-20260929/integration_report.json), release 없음이다. `rule_compliance=UNKNOWN`, `mechanism_activation=PASS`, `competitive_or_product_outcome=FAIL`로 남겼다. 따라서 현재 선택 ZIP은 변경하지 않았고, 예약한 confirmation 2008–2011도 열지 않았다. 사이트 업로드·공식 제출·모델 확정은 하지 않았다.

## 고속 속도 예산 후보의 독립 완주 검증 — 2026-09-29

기존 고속 후보의 실패 뒤, [네 독립 메커니즘 설계](plans/2026-09-29-high-speed-recovery-pivot.md)를 새 사이클로 등록했다. 이미 소비한 TRAIN 1/43·2/102에서 `path_exit`와 `speed_budget`은 각각 **2/2 완주**하고 행동을 99·255회 바꿨다. 도로 상실 기억은 행동 변화 0회, 장애물 통과선은 0/2 완주여서 새 tune으로 보내지 않았다. 새 tune 2012–2015의 첫 실행은 다른 작업이 공유 소스 디렉터리에 파일을 추가해 종료 시 해시 검증에 실패했으므로 인프라 무효다. 같은 셀을 소비 셀 재시도로 명시하고 [고정 소스 복사본](../tmp/high-speed-pivot-frozen-retry-20260929/docs/plans/high-speed-recovery-pivot-20260929/retry-design.md)에서 동일 코드·조건으로 다시 실행했다.

고정 tune의 `speed_budget`과 선택 full-road-guard는 모두 **12/12 결승선 통과**, 랩 중앙값은 **24.43초 대 25.47초**였고 충돌·손상·무효 행동이 모두 0이었다. `path_exit`은 6/12로 탈락했다. 코드를 변경하지 않은 새 held-out 2016–2019에서도 `speed_budget`과 기준은 **12/12**, 랩 중앙값은 **24.78초 대 25.74초**, 충돌·손상 0이었다. 정확한 원본은 고정 복사본의 [tune 보고서](../tmp/high-speed-pivot-frozen-retry-20260929/artifacts/haic-research-v2/high-speed-recovery-two-survivors-frozen-retry-20260929/report.json)와 [held-out 보고서](../tmp/high-speed-pivot-frozen-retry-20260929/artifacts/haic-research-v2/high-speed-speed-budget-heldout-20260929/report.json)에 있다. TRAIN·tune은 제출 후보 독립 완주율로 합치지 않는다.

측정된 제어 파일을 바이트 고정해 [로컬 ZIP](../tmp/high-speed-pivot-frozen-retry-20260929/artifacts/haic-research-v2/high-speed-speed-budget-package-20260929/submission.zip)을 만들었다. ZIP SHA-256은 `6b6c675e4e30dc51af061b7c42a9c73058222cf1873adc7ed6530774f7f8cda9`이며 19개 파일·563,702바이트다. ZIP을 그대로 추출해 서로 겹치지 않는 두 confirmation 블록에서 현재 선택 ZIP SHA-256 `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40`과 비교했다.

| 독립 confirmation | 새 고속 ZIP | 선택 ZIP | 완주 랩 중앙값, 후보 / 기준 | 해석 |
|---|---:|---:|---:|---|
| 트랙 1–3 × 2020–2023 | **11/12** | 9/12 | 26.32 / 27.52초 | 후보 완주 우세. 후보의 트랙 1/2023은 진행률 1.0이어도 결승선 통과가 없어 미완주. |
| 트랙 1–3 × 2024–2027 | **10/12** | 10/12 | 23.16 / 24.29초 | 완주 동률에서 후보가 빠름. 후보만 트랙 1/2024에서 미완주·충돌 2회, 기준만 트랙 2/2027에서 미완주. |

두 정확한 패키지 확인을 합치면 후보 **21/24**, 기준 **19/24** 결승선 통과다. 후보는 트랙별로 1번 **6/8**, 2번 **8/8**, 3번 **7/8**이다. 두 run 모두 `ADVANCE → GATE_REVIEW_ADVANCE → STOPPED`, release 없음이며 [첫 확인](../tmp/high-speed-pivot-frozen-retry-20260929/runs/haic-research-v2/high-speed-speed-budget-exact-zip-confirmation-20260929/integration_report.json)과 [독립 재확인](../tmp/high-speed-pivot-frozen-retry-20260929/runs/haic-research-v2/high-speed-speed-budget-exact-zip-replication-20260929/integration_report.json)에 원본 게이트가 있다. `rule_compliance=UNKNOWN`이므로 현재 로컬 SOTA 선택 ZIP을 자동 교체하지 않았다. 후보도 24셀 중 3개를 완주하지 못했고 공개 3위권 12~18초대 기록에는 여전히 못 미친다. 대회 사이트 업로드·공식 제출·모델 확정은 하지 않았다.

## 2026-09-29 geometry-speed teacher TRAIN pivot

The frozen TRAIN-only geometry-coast teacher finished 11/12 versus pixel control 11/12 on 5000–5003; its preregistered absolute 12/12 gate failed. A separate 5004–5007 comparison also finished 11/12 each but had a teacher-only obstacle/off-track failure at 1:5005 despite a 5.0% median lap advantage on ten jointly completed cells. A visual obstacle lockout activated on 1,881 decisions in a fresh 5008–5011 comparison but again finished 11/12 each with a teacher-only off-track failure at 1:5008 and only 2.36% paired median lap gain. Its registered outcome is PIVOT. All teacher cells are consumed TRAIN diagnostics, not candidate or official scores; no teacher model was distilled or submitted. See [the fixed-teacher outcome](../plans/haic2-geometry-speed-relative-train-20260929/outcome.md) and [the visual-guard outcome](../plans/haic2-geometry-speed-visual-guard-20260929/outcome.md).

## 고속 후보의 1번 트랙 막판 실패 재현 진단 — 2026-09-29

이미 소비한 confirmation 1/2023·1/2024를 [고정 소스의 등록된 진단 실행](../tmp/high-speed-pivot-frozen-retry-20260929/runs/haic-research-v2/high-speed-finish-posthoc-diagnostic-20260929/integration_report.json)에서 다시 추적했다. 픽셀 runtime 고속 후보는 두 셀 모두 결승선을 통과하지 못했고, 선택 기준은 모두 완주해 원래 정확한 ZIP 결과와 실패 여부가 일치했다. 이는 **사후 원인 조사**이며 새 성능 비교나 독립 반복으로 세지 않는다.

1/2023은 진행률 100%에 도달해도 유효한 방향의 결승선 통과가 없었고, 이후 101번의 판단 동안 새 도로 타일을 밟지 못했다. 충돌은 없었고 차는 계속 움직였으므로 단순 정지 실패가 아니다. 1/2024는 진행률 약 90%에서 기준과 같은 위치에 있었지만, 마지막 코너에 더 높은 속도로 접근해 반대 방향으로 크게 돌아 나갔다. 새 타일 방문이 끊긴 뒤 다시 97.85%까지 갔으나 결승선을 통과하지 못하고 마지막에는 충돌 후 거의 정지했다. 이 진단은 마지막 코너의 속도·방향·도로 가시성 상호작용이 다음 개선 대상임을 보여주지만, 어느 한 요소가 단독 원인인지는 아직 확정하지 않는다. [단계별 위치·행동 분석](../tmp/high-speed-pivot-frozen-retry-20260929/docs/plans/high-speed-finish-diagnostic-20260929/analysis.md)과 [원본 주행](../tmp/high-speed-pivot-frozen-retry-20260929/artifacts/haic-research-v2/high-speed-finish-posthoc-diagnostic-20260929/report.json)을 보존했다. 이 두 셀은 새 후보 튜닝에 사용하지 않는다.

## 2026-09-29 pixel speed student: held-out gain, exact ZIP rejection

The TRAIN-only shadow-label pixel student activated on successful control trajectories, then the first TUNE attempt failed on checkpoint serialization before student driving. Its loader-only retry on consumed 5200–5203 finished 11/12 versus 10/12 control and improved paired median lap 6.81%, but failed that plan's extra candidate-only-nonfinish veto. A prospectively registered second TUNE block 5204–5207 finished 10/12 versus 8/12 and was 6.87% faster on seven jointly finished cells. Untouched held-out 5208–5211 finished 12/12 in both arms, student median 22.20 s versus 22.92 s (3.14% faster), although student was slower on three track-2 cells and had two collisions in 3:5211. The exact local 19-file ZIP SHA-256 `4308ab2b900aa3fd2615c8b9f84c3455dde5d9f5f1711da5d31ff200f00d5e27` passed static/runtime limits. Extracted-ZIP confirmation on fresh 5212–5215 then finished only 10/12 versus selected control ZIP 11/12, with eight candidate collisions versus zero control, despite 12.74% faster paired median lap. The exact ZIP decision is **REJECT** on completion first; it was not released, confirmed on the site or uploaded. All used split identities remain consumed. See [TRAIN](plans/haic2-safe-speed-distill-20260929/outcome.md), [second TUNE](plans/haic2-safe-speed-student-second-tune-20260929/outcome.md), [held-out](plans/haic2-safe-speed-student-heldout-20260929/outcome.md), and [exact ZIP confirmation](plans/haic2-safe-speed-student-exact-confirm-20260929/outcome.md).

## 2026-09-29 visual speed saturation: activation, then fresh TUNE rejection

A fixed HUD-speed ceiling of 54 suppressed 129 extra-gas decisions on three already consumed TUNE cells and repaired one original student off-track failure, passing its diagnostic activation gate. On fresh TUNE tracks 1–3 x seeds 5216–5219, valid finishes were selected control 10/12, original student 12/12, and speed-saturated student 10/12. Saturation suppressed 439 extra-gas decisions and retained 1,457, but produced five collisions and two candidate-only nonfinishes. The fresh cycle is **REJECT** by completion first. No model was released or uploaded. The fresh TUNE identities are consumed; retain original student as a research checkpoint but its earlier exact ZIP rejection still stands. See [activation](plans/haic2-visual-speed-saturation-activation-20260929/outcome.md) and [fresh TUNE](plans/haic2-visual-speed-saturation-fresh-tune-20260929/outcome.md).

## 마지막 코너 픽셀 계측과 네 방향 재시험 — 2026-09-29

소비된 1/2024의 픽셀 도로를 사후 진단했다. 고속 후보는 decision 270에서 가까운 도로 중심 x≈38, 먼 중심 x≈23.5인 왼쪽 굽이를 보았지만 조향은 −0.02였고 속도는 약 49였다. decision 273에는 도로가 일부만 보일 때 상속 조향이 +0.17로 바뀌었고, decision 276 이후 약 40번의 판단에서 도로 중심이 검출되지 않았다. 이 계측은 시뮬레이터 상태를 추론 입력에 넣지 않았다. 첫 진단 실행 뒤 다음 후보 소스를 같은 복사본에 추가해 등록된 파일 목록이 바뀌었으므로 첫 실행의 통합 기록은 인프라 무효로 남겼다. 동일한 소비 셀을 [별도 고정 소스에서 재실행·등록](../tmp/final-bend-pixel-frozen-20260929/runs/haic-research-v2/final-bend-pixel-trace-frozen-retry-20260929/integration_report.json)했고, 이는 독립 성능 반복이 아니다.

이 근거로 [네 독립 메커니즘](../tmp/finish-bend-research-20260929/docs/plans/final-bend-four-mechanisms-20260929.md)을 만들었다. TRAIN 1/43·2/102·1/5005·2/5005에서 코너 방향 유지와 진입 감속은 각각 **4/4 실제 완주**, 선택 기준은 **3/4**였고, 직선 가속은 **3/4**였다. 행동 변경은 각각 234·95·244회였다. 도로 소실 감속은 행동 변경 20회였으나 공동 실패 셀만 작동해 새 tune으로 보내지 않았다. [TRAIN 원본](../tmp/finish-bend-research-20260929/artifacts/haic-research-v2/final-bend-four-direction-train-20260929/report.json)은 제출 후보 독립 성적에 합치지 않는다.

소스와 조건을 고정한 [새 tune 트랙 1–3 × 2900–2903](../tmp/finish-bend-research-20260929/docs/plans/final-bend-batch-20260929/tune-design.md)에서 선택 기준은 **12/12 결승선 통과·완주 랩 중앙값 25.02초**였다. 진입 감속 **11/12·23.60초**, 직선 가속 **10/12·22.61초**, 코너 방향 유지 **5/12·22.38초**로 모두 완주율에 못 미쳤다. 코너 방향 유지의 충돌은 14회, 나머지 두 후보는 0회, 무효 행동은 모두 0이었다. 고속 기준 속도 예산도 **10/12·23.37초**였다. 따라서 [REJECT → GATE_REVIEW_REJECT → STOPPED](../tmp/finish-bend-research-20260929/runs/haic-research-v2/final-bend-three-survivors-tune-20260929/integration_report.json), release 없음이며 예약 held-out 2904–2907과 exact-ZIP confirmation 2908–2911은 열지 않았다. [전체 결과](../tmp/finish-bend-research-20260929/artifacts/haic-research-v2/final-bend-three-survivors-tune-20260929/report.json)를 보존했다. 이 tune에서는 강한 조향이 여러 중간 구간에서 충돌을 만들고, 감속·직선 가속은 기존 고속 경로의 막판 미완주를 모두 고치지 못했다. 후보 ZIP이나 현재 선택 ZIP은 변경하지 않았고 대회 사이트에 올리지 않았다.

## 코너 강제 조향의 장애물 간섭 확인 — 2026-09-29

소비된 tune 기록을 분석하니, 코너 방향 유지 후보가 처음 충돌한 7개 셀 모두 가장 가까운 시뮬레이터 장애물까지 3.80–4.06 거리였다. 반면 기존 고속 후보가 완주한 열 셀에서도 먼 도로 굽이와 약하거나 반대인 조향이 **135번** 나타났다. 따라서 그 영상 패턴만으로 조향을 강제하면 정상적인 장애물 회피를 방해할 수 있다. 이 거리와 위치는 평가 전용이며 추론 입력으로 사용하지 않았다.

이미 소비한 1/2902·2/2903만 [별도 고정 소스의 등록된 픽셀 진단](../tmp/bend-hazard-diagnostic-20260929/runs/haic-research-v2/bend-hazard-pixel-posthoc-diagnostic-20260929/integration_report.json)으로 다시 보았다. 1/2902에서는 충돌 일곱 decision 전에 기존 감지기가 화면 장애물을 포착했고, 그 다음 decision에 새 조향이 상속 조향 +0.06을 −0.18로 뒤집었다. 2/2903에서는 감지가 충돌 다섯 decision 전부터 시작됐지만 강제 조향은 이미 작동 중이었다. 이 결과는 장애물 픽셀 신호로 조향 개입을 막거나 해제할 근거를 준다. 실제 완주 개선은 아직 검증되지 않았으며, 이 두 소비 셀은 새 성적에 합치지 않는다. [단계별 분석](../tmp/bend-hazard-diagnostic-20260929/docs/plans/bend-hazard-pixel-diagnostic-20260929/analysis.md)과 [원본 trace](../tmp/bend-hazard-diagnostic-20260929/artifacts/haic-research-v2/bend-hazard-pixel-posthoc-diagnostic-20260929/report.json)를 보존했다.

## 2026-09-29 obstacle precursor and acceleration burst diagnostics

Two more predeclared mechanisms were checked on the already consumed 1:5200, 3:5201 and 2:5205 TUNE cells. A fixed far-bright-object detector suppressed 10 learned gas decisions and repaired 3:5201, but lost the original 1:5200 finish and had zero suppressions in the registered twenty-decision window before the original collision. A fixed 16-decision acceleration burst with two-decision cooldown suppressed 19 decisions, including two before that collision, and likewise repaired 3:5201 while losing 1:5200. Both cycles are **REJECT** with no release; neither provides fresh competitive evidence. See [obstacle precursor](plans/haic2-long-range-obstacle-activation-20260929/outcome.md) and [burst budget](plans/haic2-acceleration-burst-activation-20260929/outcome.md). The remaining registered independent direction is visual finish reacquisition; it requires an observability diagnostic before changing actions.

## 2026-09-29 finish-view sensor and continuous boost pivot

The fourth direction in the prior pixel-student batch failed observability: its fixed return-to-start pixel signal fired at decisions 136 and 130 in valid finishes that ended at 257 and 346, as well as at 124 in the missed-finish 2:5205 run. It did not distinguish a missed finish early enough to drive a safe correction and was formally PIVOT, with no action change or release. See [sensor outcome](plans/haic2-finish-view-observability-20260929/outcome.md).

The next batch tested continuous gas blending rather than suppressing the student boost. A fixed midpoint between inherited gas and 0.28 preserved the reference finish and repaired one failure on three consumed TUNE cells, passing activation only. On fresh TUNE tracks 1–3 × seeds 5220–5223, however, valid finishes were selected control **11/12**, original student **10/12**, and half-blend **10/12**. The half-blend median finished lap was 24.38 s versus selected 25.74 s, but completion first made the fresh cycle **REJECT**; there were candidate-only failures on 1:5222 and 3:5220. No new model or ZIP was released or uploaded. These seeds are consumed TUNE. See [activation](plans/haic2-continuous-boost-activation-20260929/outcome.md), [fresh comparison](plans/haic2-continuous-boost-fresh-tune-20260929/outcome.md), and [new-batch directions](plans/haic2-post-four-direction-pivot-20260929.md).

## 2026-09-29 learned pixel coasting: TRAIN activation then driving rejection

A new three-class hold/boost/coast pixel student was trained from shadow geometry-teacher advice on successful stable pixel-control runs only. On TRAIN tracks 1–3 × seeds 5016–5019, all twelve controls finished; successful trajectories supplied 107 coast training and 31 within-TRAIN validation labels. At the fixed acceptance threshold, the pixel model made 18 coast predictions with 0.611 precision and 0.355 recall and saved a tensor-only checkpoint SHA-256 `f1afc4016f00ac699e8212f0007dde27f829831294f84b9fd3236a6cd0e3ab86`. This passed training activation but did not measure student driving.

On separately registered TRAIN tracks 1–3 × seeds 5020–5023, the student actually made 1,519 boosts and 64 coasts. It validly finished **8/12** versus selected stable pixel control **9/12**, with 14 versus 7 collisions. Median finished laps were 22.51 s student versus 23.96 s control, but the completion-first gate is **REJECT**. No new TUNE cells were opened, no ZIP released and no external action occurred. The TRAIN identities are consumed. See [training](plans/haic2-coast-class-train-20260929/outcome.md) and [closed-loop outcome](plans/haic2-coast-class-driving-train-20260929/outcome.md).

The next diagnostic is [pixel obstacle time-to-contact observability](plans/haic2-visual-ttc-observability-20260929.md) on already consumed TRAIN cells, with no action change before its signal is checked.

## 2026-09-29 visual TTC rejection and bounded gas authority activation

A pixel time-to-contact warning on already consumed TRAIN cells fired only two decisions before one student collision and produced three false alarms in a selected-control success. The registered sensor cycle was REJECT; no action or fresh score changed. See [sensor outcome](plans/haic2-visual-ttc-sensor-20260929/outcome.md).

The next fixed mechanism gave the stable pixel driver authority over learned gas: it vetoed boosts when inherited gas was below 0.08 and otherwise limited each boost to inherited gas plus 0.04. On four already consumed TRAIN cells, the original coast student finished 1/4 and the bounded variant 3/4. It repaired 2:5021 and 3:5020, retained the 1:5021 success, made 239 vetoes and 388 bounded boosts, and had zero invalid actions. This passed the preregistered activation gate and authorizes only a fresh TRAIN comparison. These four repeated cells do not count as fresh candidate completion evidence. No ZIP, release or external action. See [registered outcome](plans/haic2-base-authority-activation-20260929/outcome.md).

## 2026-09-29 시각 기반 안전·고속 전환 후보 — 정확한 ZIP까지 검증

네 픽셀 메커니즘을 별도 고정 소스에서 시험했다. TRAIN 네 셀의 행동 활성 확인 뒤, 새 tune 1–3번 트랙 × 3000–3003에서 가시성 변화 감속은 **12/12**, 안전·고속 차량 전환은 **11/12**, 기존 선택 차량은 **10/12** 완주했다. 별도 held-out 3004–3007에서는 전환 후보 **11/12**, 기존 선택 **9/12**, 기존 고속 기준 **11/12**였다. 가시성 감속은 held-out **10/12**로 두 셀에서 기존 선택 차량만 완주해 다음 단계에서 제외했다. [tune 원본](../tmp/hazard-flow-batch-20260929/artifacts/haic-research-v2/hazard-flow-four-survivors-tune-linux-20260929/report.json)과 [held-out 결과](../tmp/hazard-flow-batch-20260929/docs/plans/hazard-flow-heldout-20260929/outcome.md)를 보존했다.

전환 후보의 실제 로컬 ZIP SHA-256 `b0dc7a24e6ef994cee6e663ec6605f5aa088dbb9b3ce4510f90ccd16dc9d7fc4`를 만들어 남겨 둔 confirmation 3008–3011의 12개 셀에서 압축 해제해 실행했다. 후보 **12/12**, 기존 선택 ZIP **12/12**, 기존 고속 ZIP **11/12** 결승선 통과였다. 후보의 완주 랩 중앙값은 **24.07초**, 기존 선택은 **25.96초**였으며 후보가 12개 일대일 셀 모두에서 빨랐다. 후보 import+생성 최대 1.78초, act 최대 83.3ms, RSS 최대 302.1MB, 무효 행동 0으로 로컬 제한 안이었다. [정확한 ZIP](../tmp/hazard-flow-batch-20260929/artifacts/haic-research-v2/hazard-mode-switch-exact-package-20260929/submission.zip)과 [전체 확인 결과](../tmp/hazard-flow-batch-20260929/docs/plans/hazard-mode-confirm-20260929/outcome.md)를 보존했다. 현재 공식 사이트 URL이 열리지 않아 규칙 적합성 gate는 UNKNOWN이고, 진단 profile은 SOTA release 대상이 아니다. 공개 3위 기록과는 여전히 차이가 크다. 대회 사이트 업로드·제출·모델 확정은 하지 않았다.

다음 속도 반복을 위한 사후 관측에서는 held-out 전환 주행 3,483 decision 중 안전 모드가 2,190번이었다. 이 가운데 화면에 완전한 도로가 보이고 장애물이 없는데도 안전 가속이 고속 가속보다 0.1 이상 낮은 판단이 225번 있었다. [네 방향의 가설과 측정 한계](plans/2026-09-29-mode-switch-speed-opportunities.md)를 기록했으며, 이 소비 셀은 새 성적이나 튜닝 데이터로 재분류하지 않는다.

## 2026-09-29 bounded gas authority: fresh TRAIN improvement

The fixed bounded-authority pixel student was compared with the unchanged selected pixel control on fresh TRAIN tracks 1-3 x seeds 5024-5027. Candidate validly finished **12/12** versus selected **11/12**, repairing selected's 3:5027 nonfinish. Both arms had zero collisions and invalid actions. Candidate median finished lap was 23.90 s versus 24.00 s selected; in eleven jointly finished cells it was 23.74 s versus 24.00 s, faster on ten. The mechanism made 700 low-base vetoes and 1,248 bounded boosts. The preregistered TRAIN gate is **ADVANCE** to a separately registered fresh TUNE comparison; no release. The 5024-5027 identities are consumed TRAIN. See [outcome](plans/haic2-base-authority-fresh-train-20260929/outcome.md).

## 2026-09-29 초기 가속과 앞보기 조향의 직접 검증

출발 가속만 사용한 ZIP 위에 화면의 먼 도로를 보고 미리 조향하는 고정 코드를 결합했다. 강한 직선 추가 가속 결합은 소비된 진단에서 **10/12 대 12/12 완주**로 실패했고, 커브 제동 결합은 새 tune에서 **11/12 동률**이지만 중앙시간 **27.32 대 27.28초**로 느렸다. 가속·제동을 추가하지 않는 조향 결합 ZIP `533c3a74f43313ace925998830d3c9b948a2185d4866e761025978f44c1542d7`은 출발 가속 단독 대비 새 tune **12/12 동률**, held-out **12/12 대 11/12**, confirmation **12/12 동률**이었다. confirmation 중앙시간은 **24.18 대 24.38초**였다. 로컬 workflow는 세 게이트 PASS로 release state를 기록했지만 이 profile은 SOTA 대상이 아니고 공식 사이트 조치는 없었다.

현재 선택된 full-road-guard ZIP과 새 tune 7040–7043에서 직접 비교하자 조향 결합 후보가 첫 10%에 **2.96 대 3.92초**로 빨리 도달하고 완주 랩 중앙값도 **25.74 대 26.85초**로 짧았다. 그러나 후보 **11/12**, 선택 ZIP **12/12** 완주였다. 트랙 3/7040에서 후보가 91.62% 진행 후 도로를 이탈해 완주 우선 기준으로 **REJECT**했다. 뒤의 held-out·confirmation은 열지 않았고 선택 ZIP도 바꾸지 않았다. [세 수정안, 원본 주행과 시드 이력](experiments/launch-preview-20260929.md)을 보존했다. 약 2배 속도 목표는 달성되지 않았다.

소비된 3/7040·3/7041을 정확한 ZIP으로 다시 분석하니, 앞보기 조향은 launch-only가 실패한 3/7041의 초반에는 완주에 도움이 됐지만 launch-only가 완주한 3/7040에서는 후반 경로를 바꿨다. 픽셀 속도 45 초과 때 조향을 생략한 revision 4 ZIP은 이 두 셀을 완주했으나 같은 소비 진단의 1/7040에서 후보만 충돌 5회 후 미완주했다. 따라서 **revision 4도 REJECT**이며 새 tune 7052–7055는 열지 않았다. [행동 경계·ablation·게이트 결과](experiments/launch-preview-20260929.md)를 보존했다. 현재 선택 ZIP과 사이트 상태는 그대로다.

## 2026-09-29 bounded gas authority: fresh TUNE improvement

On fresh TUNE tracks 1-3 x seeds 5224-5227, the fixed bounded-authority pixel student finished **11/12** versus unchanged selected pixel control **8/12**. It repaired selected's 1:5227, 2:5225 and 3:5225 nonfinishes; both failed 3:5227. Candidate had zero collisions and damage versus selected five collisions and 1.0 damage, and zero invalid actions in both arms. Candidate median finished lap was 22.28 s versus 22.75 s selected. Its 574 low-base vetoes and 1,149 bounded boosts confirmed activation. The preregistered TUNE gate is **ADVANCE** to untouched held-out, with no release; 5224-5227 are consumed TUNE. See [outcome](plans/haic2-base-authority-fresh-tune-20260929/outcome.md).

## 2026-09-29 bounded gas authority: held-out rejection

On untouched held-out tracks 1-3 x seeds 5228-5231, the unchanged selected pixel control finished **12/12** while the fixed bounded-authority student finished **11/12**. Candidate-only 2:5231 reached reported progress 1.0 but retired off track without a valid lap. Both arms had zero collisions and invalid actions. The candidate made 678 vetoes and 1,257 bounded boosts, so the mechanism remained active. Its all-finish median of 21.62 s versus selected 22.49 s omits the selected control's slow successful 2:5231; on eleven jointly finished cells the candidate median was **21.62 s versus selected 21.38 s**. Completion-first held-out gate is **REJECT**, no exact ZIP, release or upload. These held-out cells are consumed and cannot be tuned on. See [outcome](plans/haic2-base-authority-heldout-20260929/outcome.md).

The next registered search batch is [four independent post-held-out mechanisms](plans/haic2-post-authority-four-directions-20260929.md): bend-phase gas veto, visual route-health recovery, continuous teacher speed target and early pixel episode routing. The consumed held-out trace motivates observation-only diagnosis; it will not be used for tuning or fresh scoring.

## 2026-09-29 mode-switch acceleration screen and fresh TUNE rejection

An isolated frozen copy screened four pixel speed mechanisms against the exact-package-derived visual mode-switch control. Three directions produced identical actions on the four designated TRAIN cells and each finished 3/4 versus control 4/4. Curve-exit surge changed 50 actions and tied control 4/4, so it alone advanced to fresh TUNE. On tracks 1–3 × seeds 3040–3043, the unchanged control finished **12/12**, the speed-budget reference **12/12**, and curve-exit surge **11/12**. Surge median among finished laps was 23.32 s versus control 23.43 s, but its candidate-only off-track at 3:3041 makes the completion-first decision **REJECT**. The run is integrated as `REJECT → GATE_REVIEW_REJECT → STOPPED`, no release. The existing exact local mode-switch ZIP remains the strongest confirmed candidate from this branch; reserved held-out and confirmation cells stayed unopened. [TRAIN design](../tmp/mode-speed-batch-20260929/docs/plans/mode-speed-four-directions-20260929.md), [TUNE outcome](../tmp/mode-speed-batch-20260929/docs/plans/mode-speed-tune-20260929/outcome.md), [integration](../tmp/mode-speed-batch-20260929/runs/haic-research-v2/mode-exit-surge-fresh-tune-20260929/integration_report.json).

Post-hoc trace inspection of the consumed 3:3041 cell shows both cars at similar positions and headings near decision 295, then sharply different headings by decision 305 while both were using the safe controller. The candidate's added gas at decisions 313–314 occurred **after** this separation. This supports testing route stability near visually ambiguous final bends, but does not identify a single causal intervention. [Step-level diagnostic](../tmp/mode-speed-batch-20260929/docs/plans/mode-speed-tune-20260929/posthoc-3041.md). No site action occurred.

## 2026-09-29 visual route-stability batch: fresh TUNE gain, held-out rejection

Four independent pixel mechanisms were screened against the unchanged visual mode-switch control on four designated TRAIN cells. Steering continuity and visibility-loss coast made no actual action changes; rendered-speed extra gas finished 3/4 versus control 4/4. Temporal far-road-flow preview changed 79 actions, tied control at 4/4, but was slower on TRAIN median (23.83 versus 23.39 s). Its unchanged code advanced under the registered activation and completion gate. [Batch design](../tmp/mode-route-batch-20260929/docs/plans/mode-route-four-directions-20260929.md) and [TRAIN result](../tmp/mode-route-batch-20260929/docs/plans/mode-route-train-20260929/outcome.md).

Fresh TUNE tracks 1–3 × 3052–3055 finished **11/12 in both arms**; preview median finished lap was **21.48 versus 21.94 s**, with zero collisions and 241 changed actions. It advanced to untouched held-out 3056–3059 without any source or threshold edit. There the unchanged control finished **11/12**, preview **10/12**, with candidate-only off-track failures at 1:3059 and 2:3058. The completion-first held-out outcome is **REJECT → GATE_REVIEW_REJECT → STOPPED**, no release. Held-out candidate median among finished laps was 25.10 s versus control 25.08 s, also no speed gain. [TUNE report](../tmp/mode-route-batch-20260929/docs/plans/mode-route-tune-20260929/outcome.md), [held-out report](../tmp/mode-route-batch-20260929/docs/plans/mode-route-heldout-20260929/outcome.md), [registered decision](../tmp/mode-route-batch-20260929/runs/haic-research-v2/mode-road-flow-heldout-20260929/integration_report.json). Exact ZIP confirmation 3060–3063 remains unopened; the previously confirmed mode-switch ZIP stays unchanged. This is the second consecutive valid non-improving speed/route cycle in this branch. No competition-site action occurred.

## 2026-09-29 continuous target and bend-sensor rejections

The continuous target-speed pixel regressor collected successful stable-control TRAIN trajectories on 5032-5035, with 235 low-target training labels and 84 within-TRAIN validation labels. It predicted 95 low targets but achieved validation MAE 2.2134 versus a constant-60 baseline 2.4362: **9.14%** improvement, below the preregistered 10% activation gate. Cautious-target MAE was 7.04. This checkpoint is **REJECT** before student driving; no candidate completion evidence. See [training outcome](plans/haic2-continuous-target-train-20260929/outcome.md).

An observation-only bend-phase sensor replayed consumed TRAIN 1:5020 failure and two bounded-student successes without changing actions. Both fixed warnings fired zero times. On failed 1:5020, the large visible bend appeared only three decisions before collision, after learned extra gas had stopped. The preregistered lead-time gate is **REJECT**, no fresh completion evidence or action change. See [sensor outcome](plans/haic2-bend-observability-20260929/outcome.md).

## 2026-09-29 pivot to the strongest confirmed local package

An audit of the frozen visual mode-switch ZIP confirmed SHA-256 `b0dc7a24e6ef994cee6e663ec6605f5aa088dbb9b3ce4510f90ccd16dc9d7fc4`, 20 archive members, and the registered extracted-ZIP report: candidate 12/12 versus prior selected exact ZIP 12/12, median 24.07 versus 25.96 s, with valid import, action latency, memory and invalid-action measurements. A later hand-designed mode-speed batch failed fresh TUNE completion (curve-exit surge 11/12 versus unchanged mode switch 12/12). Subsequent speed search will use this stronger exact package as frozen local control and a [four-direction data-driven batch](plans/haic2-mode-switch-data-driven-batch-20260929.md). No official site action or model confirmation occurred.

## 2026-09-29 temporal pixel risk feasibility rejection

The registered paired TRAIN run on tracks 1-3 x seeds 5044-5047 collected 2,101 eligible fast-policy training states and 730 within-TRAIN validation states, but zero matched the predeclared next-12-decision collision/visibility-loss label. No classifier checkpoint was trained; the activation gate is **REJECT**. The unchanged fast-only runtime finished 12/12 while the frozen visual mode-switch runtime finished 9/12 on this one TRAIN block. These are local policy observations, not candidate promotion evidence and not a reason to reuse consumed cells as fresh results. The [outcome](plans/haic2-temporal-risk-train-20260929/outcome.md) and [early-routing observability note](plans/haic2-early-routing-observability-20260929.md) explain why a future router needs a common first-50-decision prefix before comparing continuations.

## 2026-09-29 route-preview failure diagnosis and forward-model feasibility

Observation-only review of the two candidate-only held-out failures found distinct mechanisms. In 1:3059, the preview candidate hit an obstacle at decision 288 and remained nearly stationary near 93.9% progress after repeated contacts. In 2:3058, it lost the visible road around decisions 300–310 and continued moving without a collision. These simulator signals are post-run diagnostics, never policy inputs, and the consumed held-out cells will not be retuned. [Step-level account](../tmp/mode-route-batch-20260929/docs/plans/mode-route-heldout-20260929/posthoc-failures.md).

For the separate data-driven direction, a read-only audit of prior TRAIN-only mode-switch traces found 3,214 full-road decisions with five-decision targets. The far-road-center persistence baseline has 7.98-pixel MAE where the target is visible, while only 5 of those 3,214 targets lose most visible road rows. This supports testing a **dense action-conditioned road-motion prediction** before another rare-event risk classifier. The proposed feasibility gate holds back whole TRAIN cells and requires improvement over persistence plus action-ablation evidence before candidate driving. At the time of this audit, no model had yet passed that gate. [Feasibility note](plans/2026-09-29-action-conditioned-forward-feasibility.md).

## 2026-09-29 action-conditioned road prediction passed TRAIN feasibility

A new specifically registered local diagnostic read the prior immutable TRAIN report by read-only mount, without copying raw data or opening TUNE/held-out cells. It fitted one fixed five-decision ridge model on nine whole TRAIN cells (2,267 rows) and tested on three held-back TRAIN cells (882 rows). The pixel-road-and-action model achieved **4.637-pixel MAE** versus persistence **8.123**, constant motion **13.188**, and the same model without actions **4.986**. Permuting validation actions raised error to **9.187**, so the preregistered feasibility gate passed. This is predictive association, not proof of safe counterfactual actions or a qualified-finish improvement. The registered outcome is `ADVANCE → GATE_REVIEW_ADVANCE → STOPPED`, release none, with `rule_compliance=UNKNOWN`, `mechanism_activation=PASS` and `competitive_or_product_outcome=UNKNOWN`. [Frozen design](../tmp/forward-feasibility-batch-20260929/docs/plans/forward-feasibility-design-20260929.md), [result](../tmp/forward-feasibility-batch-20260929/docs/plans/forward-feasibility-run-20260929/outcome.md), [integration](../tmp/forward-feasibility-batch-20260929/runs/haic-research-v2/forward-road-feasibility-train-20260929/integration_report.json). The exact visual mode-switch ZIP and competition site remain unchanged.

The next common-prefix steering intervention falsified using that fixed model to choose actions. Twelve fresh TRAIN cells produced paired short rollouts; an integer-versus-string parser mismatch invalidated the first analysis and was corrected through a separate SHA-pinned read-only run on the immutable episode report. **10** pairs had matching prefixes and visible targets, but only **4** exceeded the registered one-pixel actual-difference threshold (minimum eight), and **0/4** had the predicted sign. The model predicted about −11.91 pixels for the fixed steering difference while those four actual differences were +1.5 to +4.0 pixels. The correction is `REJECT → GATE_REVIEW_REJECT → STOPPED`, no release. This removes the passive ridge as a justified counterfactual speed/route chooser; the earlier held-back MAE remains an on-policy prediction result. No candidate completion score or site action occurred. [Frozen correction outcome](../tmp/forward-intervention-correction-20260929/docs/plans/forward-intervention-correction-run-20260929/outcome.md), [integration](../tmp/forward-intervention-correction-20260929/runs/haic-research-v2/forward-intervention-correction-20260929/integration_report.json).

## 2026-09-29 common-prefix fast routing: activation then fresh TRAIN rejection

On six consumed TRAIN cells, keeping the visual mode-switch actions for 50 decisions and then using the fast actor repaired three prior failures: 6/6 versus control 3/6. This was an activation result only. A separately registered fresh TRAIN comparison on tracks 1-3 x seeds 5052-5055 finished **12/12 in both arms**, but the candidate was slower on **11/12** matched cells. Median finished lap was **24.51 s candidate versus 24.11 s control**; median paired slowdown was 0.80 s. The mechanism changed 1,164 post-prefix actions, with identical first-50 traces and zero collisions or invalid actions. Decision: **REJECT** at the predeclared completion-tie speed gate; no TUNE or package. An earlier run on this same plan stopped before any episode due to omitted frozen checkpoints and is retained as infrastructure-invalid, not score evidence. [Activation](plans/haic2-early-route-activation-20260929/outcome.md), [fresh TRAIN](plans/haic2-early-route-fresh-train-r2-20260929/outcome.md).

## 2026-09-29 pixel-road direction memory: TRAIN gain, TUNE rejection

A pixel-only road-direction memory rule altered steering for up to 20 decisions when fewer than three road-center rows were visible, using the side last seen in the image and leaving gas/brake unchanged. On consumed TRAIN activation cells it shortened three long visibility-loss streaks from 38–48 to 7–12 decisions, preserving 6/6 finishes. On fresh TRAIN 5056–5063 it finished **22/24 versus control 21/24**, median finished lap **23.65 versus 24.64 s**. It repaired three control failures but lost two control finishes. On independent fresh TUNE 5064–5071 it finished **22/24 versus control 23/24**; despite a 23.69 versus 24.04 s median among finishes, the completion-first gate is **REJECT**. The candidate repaired 2:5067 but lost valid control finishes 2:5066 and 2:5071. No held-out or exact ZIP was run; the exact visual mode-switch ZIP remains the local control. [Activation](plans/haic2-road-memory-activation-20260929/outcome.md), [TRAIN](plans/haic2-road-memory-fresh-train-20260929/outcome.md), [TUNE](plans/haic2-road-memory-fresh-tune-20260929/outcome.md), [paired road-loss diagnosis](plans/haic2-road-loss-paired-diagnosis-20260929.md).

## 2026-09-29 residual road-memory steering rejected at activation

A fixed halfway blend between inherited steering and the remembered road direction changed 60 actions on six consumed TRAIN diagnostic cells. It left three ordinary finishes unchanged, but all three long visibility-loss cells retired off track: candidate **3/6** versus control **6/6**, with low-visibility streaks worsening from 38–48 to 101–103 decisions. The registered activation is **REJECT**, with no fresh scoring. This falsifies the assumption that a weaker unconditional correction is automatically safer. Combined with the hard road-memory TUNE completion loss, the next search pivots away from this family rather than sweeping its blend on consumed data. [Outcome](plans/haic2-road-memory-residual-activation-20260929/outcome.md).

The next registered search is a [four-direction recovery pivot](plans/haic2-post-memory-recovery-pivot-20260929.md): TRAIN-only privileged recovery teacher, visual motion odometry, explicitly interventional action-value labels, and temporal finish-crossing localization. Its first gate asks whether a privileged teacher can recover from the same TRAIN prefix without harming valid finish. No teacher, student, or package from this pivot has passed a driving gate yet.

## 2026-09-29 05:39 KST — 공개 상위 3위 우선 확인과 후속 가속 검증

사용자 요청에 따라 공개 순위 API 세 트랙과 영상 목록을 읽기 전용으로 다시 확인했다. 트랙 1 상위 기록은 12.48/13.04/13.58초, 트랙 2는 16.14/17.46/17.88초, 트랙 3은 14.36/15.32/15.44초다. 1위 바이브코더의 세 기록은 같은 제출 #29다. 공개 영상은 과거 트랙 3 모의평가 한 편뿐이다. 그 영상에서 1위 차량은 코너 폭을 활용하며 포장 노면을 통과한 뒤 진행률이 크게 정체되지 않았고, 다른 차량의 안쪽 잔디 횡단이 선두 추월로 이어지지는 않았다. 현재 1~3위 최고 제출의 페달·브레이크·조향 명령은 공개되지 않았으며 `외국인` 차량은 그 영상에 없다. 상위 우선순위 대회 주소는 DNS 해석 실패였다. [근거](plans/2026-09-29-top-three-driving-check.md), [이번 출처 확인](sources/INDEX.md).

이 관찰을 가설로 사용한 고정 픽셀 코너 탈출 가속 후보는 TRAIN 6/6 대 6/6, 새 TUNE 12/12 대 12/12, 미사용 HELD-OUT **11/12 대 10/12**로 완주율을 지켰다. HELD-OUT 완주 중앙값은 후보 23.20초, 기준 23.53초다. 다만 후보 충돌과 P90 악화, 양쪽 모두 실패한 1:5106이 남아 있다. 진단 런의 `ADVANCE`는 별도 exact ZIP 확인 단계로만 이어지며 로컬 선택 ZIP·공식 결과·사이트 상태를 바꾸지 않았다. [전체 조건별 결과](../tmp/wide-speed-batch-20260929/docs/plans/wide-speed-heldout-20260929/outcome.md).

## 2026-09-29 privileged recovery teacher feasibility

A TRAIN-only teacher used simulator road geometry and car pose only when pixel-road visibility fell below three rows, leaving the exact visual mode-switch controls elsewhere. The first consumed diagnostic retained 6/6 finishes and shortened three 38–48-decision visibility-loss intervals to 3–7, but its preregistered absolute 20-action activation floor failed because only 16 changes were needed; it is formally **REVISE**, not a retroactive pass. On separately registered fresh TRAIN 5072–5079, the outcome-linked gate **ADVANCED**: all 21 control finishes were preserved, the teacher repaired one additional finish (**22/24 versus 21/24**), and all eight long control visibility-loss intervals shortened by at least ten decisions. Median finished lap was 23.29 versus 23.92 s. This is privileged teacher feasibility, not a pixel runtime or submission-candidate score. Next step is a separately registered pixel-only student feasibility study; no ZIP or external action. [Initial revision](plans/haic2-recovery-teacher-feasibility-20260929/outcome.md), [fresh TRAIN result](plans/haic2-recovery-teacher-fresh-train-20260929/outcome.md).

A read-only student-data audit found the passed teacher report has 110 recovery decision labels but only eight independent recovery episodes; 108 actions steer left and two steer right. Last visible pixel-road side agrees with teacher sign on only 81/110 decisions. This is insufficient diversity for a credible pixel student validation. The next registered study will collect raw four-frame pixel observations and naturally occurring right-turn recovery episodes on TRAIN only, partition by whole episode, and use horizontal mirroring only inside training. [Data sufficiency note](plans/haic2-recovery-student-data-feasibility-20260929.md).

## 2026-09-29 06:11 KST — 코너 탈출 가속 ZIP 독립 확인

앞서 HELD-OUT에서 앞섰던 코너 탈출 가속을 고정된 모드 전환 ZIP 위에 추가한 로컬 후보를 만들었다. 후보 ZIP SHA-256 `8f7e87c29e92caf49db0070842dbc30f21a20ff98a3f48bf17d9599490559e49`. 압축에서 직접 가져온 두 Agent를 독립 새 확인 조건 5110–5113에서 비교한 결과 후보 **11/12**, 기준 **10/12** 완주, 완주 중앙값 **21.94초 대 22.13초**다. 후보 188회 행동 변화가 있었고 무효 행동·충돌은 양쪽 0이다. 후보의 가져오기·생성 최대 0.926초, 행동 최대 39.43ms, RSS 302.2MB였다. 트랙 3의 5113에서는 둘 다 결승선 직전 잔디로 나가 미완주했다. [모든 조건 결과](../tmp/wide-speed-batch-20260929/docs/plans/wide-speed-exact-confirm-20260929/outcome.md), [로컬 ZIP](../tmp/wide-speed-batch-20260929/artifacts/haic-research-v2/wide-bend-rebound-exact-package-20260929/submission.zip). 현재 공개 3위 기록과는 여전히 격차가 있고, 공식 규칙 최신 확인이 불가하므로 `rule_compliance=UNKNOWN`, 릴리스·SOTA 승격·사이트 업로드·모델 확정은 하지 않았다.

## 2026-09-29 06:24 KST — 두 번째 속도 배치

새 TRAIN 여섯 조건에서 출발 가속 후보는 84회 행동이 바뀌고 완주 중앙값 21.54초로 기준 25.26초보다 빨랐다. 그러나 완주는 양쪽 5/6이며 후보만 실패한 3:5120이 생겼다. 이 차량은 약 32% 진행 지점 장애물에 부딪혀 탈락했고, 기존 차량만 실패한 3:5121은 후보가 완주했다. 따라서 사전 등록한 6/6 완주 기준은 통과하지 못해 `REVISE`로 기록했다. 연속 직선 가속과 코너 조향은 각 4/6 완주, 브레이크 해제는 행동 변화 0회였다. [조건별 결과](../tmp/wide-speed-batch-20260929/docs/plans/speed2-four-directions-20260929/outcome.md). 기존에 독립 확인된 코너 탈출 ZIP은 그대로 보존했다.

## 2026-09-29 recovery-student collection interrupted by Docker

The registered TRAIN-only collection under the stronger bend-rebound ZIP started on tracks 1–3 × seeds 5120–5135. Docker Desktop's Linux engine exited during track 3. The process handle and experiment container ended before the 48-cell report was written. Eleven partial recovery-label files remain preserved, including 137 labeled decisions, but only one naturally right-steering episode with three positive decisions. The run is formally `REVISE → GATE_REVIEW_REVISE → STOPPED` for infrastructure failure; the partial files are excluded from student fitting and completion scoring. Docker restarted successfully. [Frozen outcome](plans/haic2-recovery-student-data-wide-train-5120-5135-20260929/outcome.md) and [integration record](../runs/haic-research-v2/haic2-recovery-student-data-wide-train-5120-5135-20260929/integration_report.json).

The next TRAIN-only fixed-pulse diagnostic produced real unmirrored recovery labels in both directions on all six fresh 5140–5141 cells: 304 finite labeled decisions, zero invalid actions, and all 12 NPZ hashes verified. Yet the predeclared requirement for an uninterrupted eight-decision pulse failed 0/6 in both arms: the privileged teacher took over after four to six decisions. The registered outcome is **REVISE**, no student training or candidate score. A new independent activation gate must define the early pulse-to-teacher transition before execution. [Frozen outcome](plans/haic2-balanced-recovery-pulse-train-5140-5141-20260929/outcome.md), [collection design](plans/haic2-balanced-recovery-collection-design-20260929.md).

The revised gate was registered before running fresh TRAIN 5160–5161. All six cells in each pulse direction produced a ≥3-decision pulse followed by the intended opposite-sign teacher steer, exceeding the preregistered 4/6 floors. There were 18 episodes, 248 finite pixel labels, 12 valid NPZ hashes and zero invalid actions. The formal outcome is **ADVANCE only to a larger TRAIN data collection**, with `rule_compliance=UNKNOWN`, no pixel student or candidate driving score, and no release. [Outcome](plans/haic2-recovery-transition-train-5160-5161-20260929/outcome.md).

## 2026-09-29 09:40 KST — 출발 가속 재검증 탈락

5120–5121 학습 기록은 다른 로컬 TRAIN 등록과 겹쳤다는 사실을 확인해 [정정 기록](../tmp/wide-speed-batch-20260929/docs/plans/speed2-four-directions-20260929/overlap-correction-20260929.md)에 남겼다. 이후 별도로 비어 있던 6200–6201에서 네 방향 다섯 후보를 비교했다. 출발 가속 제한 후보는 기준과 6/6 완주하며 중앙값 21.35초 대 21.55초로 빨랐다. 그러나 새 TUNE 6202–6205에서는 후보 11/12, 기준 12/12였다. 후보가 3:6204에서 85.9% 진행 후 도로를 벗어났으므로, 빠른 완주 중앙값 22.22초 대 22.66초에도 완주 우선 기준으로 `REJECT`했다. [전체 조건별 결과](../tmp/wide-speed-batch-20260929/docs/plans/speed3-launch-fresh-tune-20260929/outcome.md). 앞서 정확한 ZIP으로 독립 확인한 코너 탈출 차량은 보존했고 대회 사이트를 변경하지 않았다.

## 2026-09-29 larger pixel recovery data: natural coverage short

The SHA-pinned TRAIN 5170–5177 collection completed 72 teacher-only episodes on 24 independent cells. It produced 917 finite pixel/teacher labels in 48 hashed NPZ files, with zero invalid actions. Both induced directions met the registered 20-cell floor (left 24/24, right 22/24), but unpulsed natural recovery occurred in only **2/24** cells against the preregistered minimum four. The formal outcome is **REVISE**, no pixel-student fit or candidate driving score. The immutable data can be combined only with a separately registered natural-recovery supplement and whole-cell partitioning; the earlier Docker-interrupted partial files stay excluded. [Outcome](plans/haic2-balanced-recovery-data-train-5170-5177-20260929/outcome.md), [pixel observability note](plans/haic2-recovery-student-observability-20260929.md).

A new unpulsed 96-cell TRAIN supplement on 5260–5291 was registered and started, but Docker Desktop exited before it could write a complete report. One partial NPZ remains excluded; the run is formally infrastructure-invalid **REVISE**, not a score observation. Docker restart initially failed on a stale local inference socket; a normal WSL shutdown and Docker restart restored the engine without deleting the socket. [Frozen outcome](plans/haic2-natural-recovery-train-5260-5291-20260929/outcome.md). Natural coverage remains two completed cells.

A distinct short pixel-motion recovery rule was then tested on consumed TRAIN seed 5176. It changed nine steering actions, repaired a track-1 nonfinish and shortened two road-loss streaks, but lost the valid track-3 control finish near 95% progress. The no-lost-control-finish gate is **REJECT**; the observed 2/3 completion tie exchanges failures and is not a qualified improvement. No fresh TRAIN/TUNE driving or ZIP followed. [Outcome](plans/haic2-motion-recovery-consumed-activation-20260929/outcome.md).

The first shorter natural-recovery collection block completed on fresh TRAIN 5292–5299: 24/24 episodes executed, four independent natural recovery cells across two tracks, both steering signs, 30 finite labels in four matching-hash NPZ files and zero invalid actions. It met the registered data-coverage gate and formally **ADVANCED only to whole-cell offline student feasibility**. Six completed natural recovery cells now exist when combined with the earlier 5170–5177 data; the Docker-interrupted partial files remain excluded. There is still no pixel student driving score. [Outcome](plans/haic2-natural-recovery-chunk1-train-5292-5299-20260929/outcome.md).

## 2026-09-29 09:44 KST — 상위 1~3위 방식 우선 재확인

공개 순위를 다시 읽자 1위 바이브코더의 최고 기록은 트랙별 12.22·16.06·14.10초로 빨라졌고, 트랙 1·3은 제출 #31, 트랙 2는 #30이었다. 공개 영상 목록은 여전히 과거 트랙 3 모의평가 한 편뿐이다. 당시 선두의 도로 폭 활용과 코너 출구 이후 진행률 증가만 영상으로 확인되며, 현재 1~3위의 가속·제동·조향 명령과 반복 완주율은 확인할 수 없다. [최신 트랙별 표와 영상 관찰](plans/2026-09-29-top-three-driving-check.md). 별도로 탈락한 출발 가속 후보의 기존 TUNE 궤적을 읽기 전용으로 진단했다. 후보는 80% 진행까지 기준보다 9결정 빨랐으나 85.9%에서 약 100결정 동안 진행하지 못하고 미완주했다. 이 실패는 최고 속도 자체보다 후반 주행선 이탈·복귀의 문제라는 가설을 세우지만, 인과는 확인되지 않았다. 소비한 TUNE에서 재조정하지 않고 다음 메커니즘은 새 TRAIN에 등록해야 한다. [진단 기록](../tmp/wide-speed-batch-20260929/docs/plans/speed3-launch-fresh-tune-20260929/failure-diagnostic.md). 사이트 업로드·모델 확정은 하지 않았다.

## 2026-09-29 10:45 KST — 직선 가속 후반 완주 검증

새 네 방향 TRAIN 6300–6301에서 도로가 전부 보이는 직선에만 gas 최소 0.55를 주는 후보가 기준과 **6/6 완주**, 중앙값 **21.88 대 23.66초**로 앞섰다. 다른 세 방향은 활성 부족 또는 속도·충돌 열세였다. 고정 후보를 새 TUNE 6302–6305로 보냈을 때도 양쪽 **12/12 완주**, 후보 **19.62 대 21.29초**, 모든 셀에서 빨랐다. 다만 후보에게 충돌 4회가 생겼다. [TRAIN 결과](../tmp/wide-speed-batch-20260929/docs/plans/speed4-path-speed-train-20260929/outcome.md), [TUNE 결과](../tmp/wide-speed-batch-20260929/docs/plans/speed4-straight-fresh-tune-20260929/outcome.md).

미사용 Linux HELD-OUT 6306–6309는 실행 시작 직후 Docker Desktop Linux 엔진이 종료돼 결과 파일 없이 중단됐다. 해당 셀은 예약된 채 보존했고 성적으로 계산하지 않았다. 복구 과정에서 오래 남은 Docker 소켓 파일 제거 명령이 자동 승인 검사에 차단되어 Linux 검증을 마치지 못했다. [엔진 실패 기록](../tmp/wide-speed-batch-20260929/docs/plans/speed4-straight-heldout-20260929/engine-failure.md).

별도 Windows 보조 진단 6314–6317은 양쪽 **11/12 완주**였지만 실패 셀이 달랐다. 후보만 실패한 2:6314는 진행률 100%를 찍고도 정상 결승선 통과 없이 650스텝 제한에 도달했다. 기준만 실패한 1:6317은 후보가 완주했다. 후보의 완주 중앙값은 **20.96 대 23.26초**로 빨랐지만 사전 등록한 후보 단독 완주 손실 금지 기준을 어겨 `REJECT`했다. Windows 결과는 Linux 검증을 대체하지 않는다. [보조 진단 결과](../tmp/wide-speed-batch-20260929/docs/plans/speed4-straight-windows-stress-20260929/outcome.md). 기존 정확한 코너 탈출 ZIP은 유지하고 대회 사이트는 변경하지 않았다.

이후 Docker 엔진이 다시 연결되는 것은 확인했다. 중단된 HELD-OUT 실행은 여전히 결과가 없고 6306–6309를 새 조건으로 재사용하지 않는다. 탈락한 직선 가속 버전도 유리한 시드를 찾기 위해 재평가하지 않는다. 새 TRAIN 조건에서 가속 이득과 중후반 주행선·결승선 접근 안정화를 함께 검증하는 별도 메커니즘을 설계한다. [엔진 복구 후 기록](../tmp/wide-speed-batch-20260929/docs/plans/speed4-straight-heldout-20260929/engine-recovery-followup.md).

## 2026-09-29 fixed high speed with road constraints

사용자 요청대로 최종 속도 제어를 목표60으로 고정하고, 네 조향 방향의 30회 TRAIN 비교와 포화 구현 수정6회·도로 경계 개입6회를 실행했다. 흔들림 억제 방식은 3/6 일반 완주(17.24–17.54초, 충돌0, 고속 유지)를 보였으나 도로 밖 시간2.39–4.87%로 엄격 조건은0/6이다. 다른 방식도 엄격 통과0/6. 속도나 이탈 판정 기준을 낮추지 않았고, 후속의 같은 셀은 독립 점수로 계산하지 않았다. 세 실행은 모두 STOPPED/released=false, 선택 ZIP·사이트 변경 없음. [결과와 다음 진단 지점](experiments/fixed-high-speed-steering-20260929.md).

## 2026-09-29 user-requested high-speed damping ZIP delivery

사용자가 측정된 짧은 이탈을 허용하고 원래 damping의 제출 코드를 요청했다. [ZIP](../artifacts/haic-research-v2/fixed-high-speed-damping-package-20260929/submission-high-speed-damping.zip)은 7,205바이트, 5파일이며 SHA-256 b196cfb653a0ceeae0779e5e3832f879a27f24cc662b7202a1a0c39888d469f9이다. 압축에서 직접 가져온 Agent가 여섯 소비 TRAIN 셀 모두 원래 행동·궤적·결과와 일치했다(3/6완주). [확인 결과](../artifacts/haic-research-v2/fixed-high-speed-damping-package-20260929/package_report.json). 새 성능 증거·자동 SOTA승격·사이트 업로드는 없으며, 이전 1%이탈 기준을 사용자 요청에 반해 제공 차단에 적용하지 않았다. 가져오기 시간은 NumPy warm 상태의 수치이며 cold import 인증은 아니다.

## 2026-09-29 high-speed collision recovery

네 방향 여섯 후보를36회 비교했다. 충돌 후 장애물 회피 조향을 잠시 제외하는 후보는 기존 실패1:38201을17.86초 완주로 바꿨고, 기존 세 완주를 유지해 소비 TRAIN4/6을 기록했다. 고정 ZIP을 새 TRAIN 여섯 조건에서 기존 ZIP과 비교한12회에서는 양쪽3/6, 중앙값18.84초로 동일했다. 새 충돌 실패는 복구하지 못했다. 따라서 실험용 개선은 보존하되 일반적인 완주 개선으로 주장하지 않는다. 속도 목표와 가속·제동은 그대로다. 실행 상태는 STOPPED, release=false이며 사이트 변경은 없다. [원인·모든 후보·ZIP·결과](experiments/fixed-high-speed-recovery-20260929.md).

## 2026-09-29 픽셀 복구 학생 오프라인 검증

완료된 TRAIN 5170–5176 셀의 교사 자료로 두 픽셀 조향기를 학습하고, 별도 TRAIN 5177 및 자연 복구 5292–5299 셀에서 검증했다. 자연 복구 네 에피소드의 전체 30판단에서 두 학습기는 각각 16개 방향만 맞혀 사전 기준 75%에 못 미쳤다. 첫 방향 고정은 22/30이고 이전 주행에서 완주 손실도 확인됐다. 계획 해시 `94b8d9fa402ba75cdf68d071654179fe82e17d3bdc470bacbfc32a6b96a36380`의 정식 결과는 `PIVOT`, 배포 없음. 소비된 TRAIN 셀은 유지하며, 실제 완주·랩타임을 겨냥한 다른 메커니즘으로 방향을 바꾼다. [결과](plans/haic2-recovery-offline-student-train-20260929/outcome.md).

후속 원본 자료 확인에서 위의 방향 전환 집계 중 자연 복구 한 건은 연속 복구 도중이 아니라 100여 결정 떨어진 두 복구 구간 사이의 변화로 밝혀졌다. 자연 복구 다섯 구간 안에서는 방향 전환이 없었다. 원본 실행·결정은 보존하고 [정정 기록](plans/haic2-recovery-offline-student-train-20260929/correction-20260929.md)에 구간 기준 수치를 남겼다. 다음에는 도로가 다시 보일 때 방향 기억을 초기화하는 방식만 별도로 사전 등록해 평가한다.


## 2026-09-29 out-in-out 및 가속 비교

out-in-out 목표를 구현하고 5차례 수정해 총 126회 비교했다. 최신 후보는 기존 6/6 대비 4/6 완주였으며, 공통 완주 네 조건에서만 차이 중앙값 0.25초 빨랐다. 실제 바깥→안쪽→바깥 궤적도 일관되게 확인되지 않아 제출본 교체 없이 PIVOT/STOPPED로 기록했다. 완주와 안정적인 속도 개선 목표는 미달성이다. 기존 체크포인트와 모든 실패 자료를 보존했다. [구현·전체 결과·한계](experiments/out-in-out-fast-20260929.md).


## 2026-09-29 최종 비교 및 제출 권장 상태

`fast-clearance-cone-20260929`의 72회 주행이 완료되었다. 같은 TRAIN 12조건에서 안정화 control 12/12, 가속 current 10/12, cone_commit 9/12, cone_bearing 4/12, cone_shock 8/12, cone_rate 8/12 완주했다. 모든 원자료 해시가 일치하고 invalid action 및 실행 오류는 0이다. 조향 개입은 실제 발생했지만 완주율을 유지하지 못했다. REVISE → GATE_REVIEW_REVISE → STOPPED, released=false로 기록했다. 근거: `artifacts/haic-research-v2/fast-clearance-cone-20260929/report.json`, `batch_audit.json`, 해당 run의 integration_report.

현재 제출 권장은 기존 `artifacts/haic-research-v2/fast-completion-row-repair-20260929/submission-fast-completion.zip`이다. SHA-256: `7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835`. 비교한 로컬 12조건에서 12/12 완주한 안정화 코드이며, 공식 비공개 트랙 완주를 보장하는 결과는 아니다. 새 가속 ZIP은 10/12이므로 승격하지 않는다. 랩타임 절반 단축 목표는 미달이다. 사이트 업로드나 공식 제출은 하지 않았다.

사용자가 요청한 반복 개선은 매시간 heartbeat `haic` (HAIC 완주율·고속 주행 개선)에 등록했다. 기존 실행과 중복하지 않고 회차당 2CPU/2GiB/20분 이내에서 진행하며 의미 있는 변화가 있을 때 알린다. 다음 진단은 관측 기반 조향 지연·회전 관성·차체 통과 영역 및 장애물 추적이다. 순간 목표점 조향만으로 실제 차체 여유를 보장하지 못한다는 가설을 평가해야 하며, 아직 원인을 확정하지 않았다.

## 2026-09-29 가속·감속 연구

[연구 문서](experiments/acceleration-braking-research-20260929.md)에 기존 24개 TRAIN 주행과 물리 코드, 레이싱 논문 분석을 기록했다. 장애물 검출 시 거리와 무관한 속도 44 제한, 작은 gas/brake 중복, 회피 경로·회전 지연과 속도 계획의 결합 필요성을 확인했다. 제동거리 기반 감속과 코너 탈출 조기 가속을 우선 연구하며, 네 방향 비교 설계는 아직 미등록 초안이다. 새 주행과 성능 승격은 없고 기존 12/12 완주 ZIP 권장을 유지한다. 다음 자동 개선 회차는 이 연구 문서도 읽고 실험을 구체화한다.

## 2026-09-29 앞보기 가감속 90회 비교 완료

[앞보기 가감속 구현 결과](experiments/preview-pedal-20260929.md): 즉시 앞보기→코너 정보 유지→가감속 설정 상향의 세 버전, 각30회 비교를 완료했다. 최신 조향 부담 후보6/6완주·최고74.74·중앙23.78초, 기존6/6·중앙21.59초라 미승격. 신규 ZIP/업로드 없음. 모든 raw해시 및 control 재현 일치, invalid0. 세 run 모두 STOPPED/released=false. 다음 자동 개선에서는 코너 제약 해제와 실측 회전·경로 진행을 결합해 저속 체류를 줄이는 가설을 우선 확인한다. 기존 안정화ZIP을 기준으로 유지하고 R1 PIVOT 이후R2/R3두 번의 비개선 이력을 이어간다.

## 2026-09-29 사용자 지적에 따른 앞보기 우선순위 변경

[관측 범위 진단](experiments/forward-visibility-audit-20260929.md): 현재 Agent입력에전체미니맵없고초기광각도전달되지않음. 기존주조향row42/54는속도70에서직선기준약0.18초앞, 새pedal corridor는row10약0.45초앞이다. 최신90회는원거리페달+근거리조향 조합으로 장거리경로계획이 아니다. 단순pedal상수재조정보다2차원연결도로추적/선행조향/회전이동보상국소지도/공유경로속도계획을 다음연구우선순위로한다. 새설계초안만기록했으며주행·성능승격없음. 미관측전체맵을아는것처럼가정하지않는다.

## 2026-09-29 14:25UTC heartbeat 결과

[연결 경로 선행 조향](experiments/connected-preview-20260929.md): R1 30회는 control6/6, pursuit1/6,dual5/6,memory1/6,shared3/6로PIVOT/STOPPED. 원거리조향이약0.32초앞점을사용했지만장애물항과상쇄되고근거리보정을잃는사례발견. 기억경로재사용0회도발견해현재관측과맞는과거점의혼합으로보완했다.

R2 connected-guarded-budget-20260929는20회/4소비TRAIN조건의부분screen이다. dual_preview4/4 중앙21.47초 vs 같은4조건control4/4 21.68초,충돌0 vs1. 나머지3/4,3/4,2/4. 아직넓은조건과ZIP재현없어개선확정/승격없음; 기준ZIP보존. 특히R1실패조건2:38302가이번부분screen에없으므로다음회차는그조건과3:38302를포함한전체6조건부터확인해야한다. memory혼합381회중조향허용296회였으나완주3/4로효과미확정. gate UNKNOWN/PASS/UNKNOWN,REVISE/STOPPED,released=false. 모두rawhash/control재현/first10일치,invalid0. R1실행332.40초,R2실행210.58초,각2CPU2GiB. 현재fixed60_completion프로파일timeout240초이므로다음실행등록전에회차20분예산과적절히맞춘다. 초기 connected-guarded-20260929는등록만하고승인/실행하지않은대체계획이므로실행하지않는다. 이번회차더이상주행없음.

## 2026-09-29 지속 개선 범위 확대

사용자 최신 지시: 기존 방법을 유지할 필요 없이 반복해서 최선의 주행 결과를 추구한다. 기존 heartbeat `haic`를 갱신했다. 조향·가감속·인식·경로 계획·학습 정책·혼합 구조까지 변경할 수 있고, 고정 속도·항상 높은 gas·out-in-out을 필수 조건으로 삼지 않는다. 역사 기준 ZIP을 보존하면서 검증된 현재 최선 후보를 별도로 관리한다. 절반 단축은 중간 목표이며 사용자가 중단/변경할 때까지 매시간 계속한다. 회차당 등록 실행 예산 2CPU/2GiB/20분, 동일 조건의 완주율 우선 비교, 실패/검증 split 보존, 정확한 ZIP 재현 원칙은 유지한다. 외부 업로드/공식 제출/모델 확정/외부 메시지는 허용되지 않는다. 이번 변경 자체는 예약 작업 설정이며 새 주행 성능을 의미하지 않는다.


## 2026-09-30 00:26KST heartbeat / guarded full12

[Connected preview results](experiments/connected-preview-20260929.md): unchanged guarded actor completed full6 and additional6,60episodes565.93seconds total,2CPU2GiB. Dual12/12 matches stable12/12 but aggregate median20.65s vs20.52s is slower;10faster/2slower cells do not override median rule. Collisions3 vs1 (three dual contact flags in1:38303 steps98-100), no promotion/ZIP. Pursuit11/12,memory10/12,shared10/12. All60hashes,invalid/error0,first10equal;all12controls exactreplay. R1failed2:38302 nowcomplete, but additional1:38303 exposes contact while preview alreadydisabled. Next investigate pre-obstacle state and controller handoff continuity with four mechanisms; no threshold sweep. OriginalZIP7bc38a3... preserved. Full6 run3904a9c7... REVISE/STOPPED gatesUNKNOWN/PASS/UNKNOWN; extra6run71bf243b... REVISE/STOPPED gatesUNKNOWN/PASS/FAIL. Full6 omitted formal predecessor reference; append-only correction preserves limitation, extra6 formally linked. Allcells consumedTRAIN, repeated prior4 not independent. Currentfixed60timeout600seconds; no active run. User allows any compliant method; no external actions.


## 2026-09-30 01:27KST heartbeat / handoff-path

[Handoff/path72episode report](experiments/handoff-path-20260930.md). New four directions on frozen dual plus stable and dual references, full12consumedTRAIN each. Stable12/12median20.52s,dual12/12median20.65s;handoff11/12,earlygap8/12,heading12/12butzeroactivation,straightacceleration9/12peak69.00. No improvement/ZIP. Direct trace finds early-gap lockedright command conflicting with parent's mutableleft avoidance at1:38300steps255-256;nextreplace competing obstacle terms with one geometric path, evaluate motion-prediction/lateral-velocity/sharedspeed directions separately. Handoff active onlytwo decisions nearobstacle2:38302y60 thenofftrack; heading missingroad condition absent inall12. Do not count inactive heading as successful repair.

runhandoff-path-20260930 planhashc196aeaf10b00dc00a9ac8f30887aab2c4b2d143d332dfb6a63e692e079b636a formally linked predecessor;bothapprovalsrecorded. REVISE/STOPPED,releasedfalse,gatesUNKNOWN/UNKNOWN/FAIL. Execution671.59s2CPU2GiB,peakRSS315969536bytes,maxact42.014ms;all72rawhashvalid,invalid/error0,first10equal,24control/dualexactreplays,12headingexactdual. Staticreviewcountertiming/side-lock/fallbackfixes madebeforefreeze. Second fullcomparable guardednonimprovement;partial4/full6+extra6notextraindependentcycles. OriginalstableZIP7bc38a3...verifiedunchanged. NoactiveHAICrun;fixed60timeout1050s. Nextregisterfreshdesign beforeimplementation and run, no frozen source edits or external actions.


## 2026-09-30 02:27KST heartbeat / unified passing PIVOT

[60-run unified passing result](experiments/unified-passing-20260930.md). Stable12/12median20.52s; unifiedgap10/12, predictedgap8/12, lateralobserver12/12median20.59s, distance-speed12/12median23.28s(max70.145,zero contacts). Noimprovement/ZIP. All4mechanismsactualactive; duplicateopposingavoidtermremoved, prior1:38300finisheswithoutcontact, butsidechosenearliercanremainwrongrelativecurvingroad(2:38301firstcontactstep86). Need feasibility of fullpassingpath, notconstanttuning/anotherresidual.

rununified-passing-20260930 planhashe6ecf8ecd654ea27f9be752054d66e3f14c50c65ae8c76f7fdf3b2f4e0a0ab65,PIVOT/STOPPED,releasedfalse,gatesUNKNOWN/PASS/FAIL. Thirdvalidcomparablenonimprovement:guardedfull12,handofffull12,unifiedfull12. CheckpointoriginalZIP7bc38a3...preserved. 60rawhashvalid,invalid/error0,first10equal,12controlexactreplays,367.84seconds2CPU2GiB,RSS339279872bytes,maxact9.902ms. NoactiveHAICrun,profiletimeout900s. NextbeginfreshDISCOVER/plan with formalPIVOTpredecessorandpreservedcheckpoint: evaluate road/obstacle-clearance paths, observedturningresponse, finite-horizon actions, speed onselectedfeasiblepath as fourindependentmechanisms. Noactualnextimplementationyet. ConsumedTRAIN38300..38303unchanged; donotuseconfirmation/blindfortuning.


## 2026-09-30 stronger acceleration request /120 comparisons

[Acceleration evidence](experiments/acceleration-envelope-20260930.md). Userasks더가속/영상처럼. Implementedfourpropulsionmechanismsgas1: launch,exitburst,sweptpathsprint,horizonbrake. R1acceleration-envelope-20260930plan4c33eee1...:12control,11launch,10exit,10sprint,9horizonof12. Sprintmax75.276vs60.651andall10commonfinishes.76-1.42sfaster,but2failures. R2acceleration-braking-20260930plan4590a862...retainsgas1andstrongthreatbrakinginclwheel-lock:8/12,8/12,9/12,6/12,slower;rejected. BothREVISE/STOPPEDgatesUNKNOWN/PASS/FAIL.120rawhashvalid,invalid/error0,24controlexactreplay,nonlaunchfirst10equal(launchintentionaldifference). Runtime800.794s,total2CPU2GiBperrun,nonewZIPoroverallimprovement,original7bc38a3...preserved.

Important bottleneck: pixel_features.nearest_bright_object searchesrows22..61 only. Straightgeometricfrontdistance≈24.1,~.32secondsatspeed75. Laterstrongbrakingcausedoverbraking/wheel-lock,notfix. Nextdesign earlierpixelhazarddetection+tracking+feasiblepassingpaths+farhazardspeedtrajectory (fourmechanisms), retainR1highgasandavoidnewblanketbraking. Two validpostpivotnonimprovementcycles. Exactuser-referencedvideounconfirmed;priorhistoricvideoobservationsre-read,webpriority/alternatevideolistinaccessible, don'tclaimfreshvideoorcompetitorpedals. NoactiveHAICrun,currentfixed60timeout780s. Registerfreshhashandseparateapprovalsunderstandingauthorizationbeforeoperations.


## 2026-09-30 03:56 KST — arrival-speed ZIP local ADVANCE

[108-episode evidence](experiments/far-hazard-20260930.md). Four-direction screen: stable12/12 median20.52s; arrival envelope12/12 median20.13s, peak75.27. Other candidates lost completion or duplicated actions. Frozen ZIP exactly reproduced all12 TRAIN trajectories. New held_out49300–49303 xtracks1–3: original10/12 median19.34s vs candidate10/12 median18.46s; identical two failures, all10common finishes0.66–1.16s faster. These held_out identities are consumed, never tuning/fresh data. Confirmation/blind untouched.

Current local best is `artifacts/haic-research-v2/far-hazard-package-20260930/submission-arrival-speed.zip`, SHA256 `b168a17dac5fe5d6b28a265fb4adeeba6b3346391b35acd4a5818e7dc0b9693b`. Original historical checkpoint remains `artifacts/haic-research-v2/fast-completion-row-repair-20260929/submission-fast-completion.zip`, SHA256 `7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835`.

Search run far-hazard-20260930 plan70eefa094cb3738357c5153a47d601cfebd4e37353ac938f514aec5e632988ce REVISE/STOPPED UNKNOWN/PASS/UNKNOWN. Package run far-hazard-package-20260930 plane8a5d6debacaf9b52f4b475ea2744f7384ffeebcbf7a7353525ed89d4ddfa4ff ADVANCE→GATE_REVIEW_ADVANCE→RELEASE_IF_GATE_PASS→STOPPED, all3gatesPASS; local release only. No official upload/submission/model confirmation. Both runs703.068s,2CPU2GiB,108raw hashes valid/errors0. Source audit uses current official Participants fallback; site inaccessible. Wrapper differences are telemetry-only, physics matches. No active HAIC run.

Next: preserve both reference ZIPs, register fresh TRAIN robustness/predictive-control study with4independent directions (<=8candidates,<=2/direction), then untouched validation. Do not optimize on held-out failures. Measured improvement resets consecutive non-improvement count; full objective remains ongoing. Current profile650s, settings dispatch frozen package helper (do not blindly rerun; output exclusive). See current-best-validated.json for current candidate. No automation changes required.


## 2026-09-30 04:28 heartbeat — completion improvement, new local ZIP

[144 episodes and causal evidence](experiments/contact-continuity-20260930.md). New TRAIN50300–50303 frozen diagnostic: original9/12, arrival10/12, two repeated-contact failures. At near obstacle4, five damage increments occur in0.32s; brake-history veto coincides with pixel speed collapse and recovery suppression. Side persistence rescues2:50302 by preventing late avoidance reversal; crossing prediction rescues2:50301 but earlierinterventions also change approach. Contact-aftercare alone insufficient in thisbatch.

Four-direction screen contact-continuity-20260930: current10/12; impactoverride10/12, coast10/12, side11/12median19.12, crossing11/12median18.86. Crossing selected and frozen. ExactZIP all12TRAIN traces/outcomes reproduce. New held_out51300–51303 x3tracks: currentarrival11/12median19.06, crossing12/12median18.96, historical12/12median19.67. All12historicalmatched laps faster. Versuscurrent11commonfinishes,7slower/3faster/1tie; pairedmedian+40ms, so improvement is completion, not pairedspeed. No universalcompletion claim; older held_out49300–49303 untested for newZIP and never reused as tuning.

New local best: `artifacts/haic-research-v2/contact-package-20260930/submission-crossing-projection.zip`, SHA256 `4447c8dfad1392ddbbb83f0b3d23b8d5237af2b9be785cdfd08edaf6c4623f64`. PriorarrivalZIPb168a17... and originalhistoricalZIP7bc38a3... preserved. `current-best-validated.json` records allthree. Package plan62fd7b9e2b8738d2be2c74caa542ec6af4974c856325db02d266e202a041e670 ADVANCE, all3gatesPASS, localRELEASE_IF_GATE_PASS→STOPPED. No formal SOTA table update, modelconfirmation, submission, upload or externalmessage.

Three runs: completion-discovery-20260930(24,140.007s), contact-continuity-20260930(72,427.805s), contact-package-20260930(48,382.190s). Total144episodes950.002s,2CPU2GiB,allrawhashesvalid/errors0. MaxpackageRSS284237824bytes,coldimport.3023s,reset.000123s,act22.38ms. Exact24screenreference replays plus12packagereplays. Latestprofile550s/settingscontact_package, outputexclusive; do not rerun blindly. No activeHAICrun. Validimprovement keepsnonimprovementcount0. TRAIN50300–50303 consumed; held_out51300–51303 consumedprotectedfromtuning; confirmation/blind untouched.

Next boundedcycle: retain crossingcurrent andoriginalhistorical as distinct controls, study remaining TRAIN failure and/or independently validate frozen sidecandidate. New search needs4independentdirections and fresh approvals understandingauthorization, no newuserquestion. Diagnosticroot registration firstattempt --previous-run afterADVANCE wasrejected beforeexecution; finalrootmanifest hasnoformalpredecessor, documented inbatch_audit. No failed simulation hidden. Continue broaderoptimization; fullgoal remainsongoing.
