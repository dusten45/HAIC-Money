# HAIC 하네스 운영 요약과 현재 상태

갱신: 2026-09-26. 최신 요청으로 HAIC 하네스 작업을 재개했다. 현재 여러 에이전트가 완주율 개선과 관련한 서로 다른 코드 경로를 읽기 전용으로 조사하는 DISCOVER/HYPOTHESIZE 단계다. 구체적인 실행 가설과 실험 설계는 중앙 통합 후 별도 승인 게이트를 거친다.

## 합의한 운영 방식

- **완주율이 최우선**이다. 같은 조건의 후보끼리 비교하고, 완주율이 같을 때만 랩타임·미완주 진행도·충돌·손상·추론 지연을 비교한다. 공식 대회 점수와 내부 선택 기준은 구분한다.
- 중앙 담당자가 상태·우선순위·예산·역할을 조율한다. 여러 에이전트는 서로 다른 탐색 방향을 맡고, 사실·추론·미확인 사항·권고·근거 경로를 나누어 보고한다.
- 가설마다 근거, 관측 가능한 정보, 허용 행동, 성공 조건, 비교군, 반증 조건, 최소 실험, 자원 한도를 적는다.
- 설계·구현·실행 승인을 구분한다. 평가 뒤 ADVANCE / REJECT / REVISE / PIVOT 모두 게이트 검토를 거친다. **ADVANCE이며 세 게이트가 모두 PASS일 때만 릴리스 가능**하다. 공식 제출은 별도 승인 사항이다.
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

## 이번에 재개된 하네스 단계

여러 에이전트가 독립 방향 4개를 대상으로 가설을 탐색하는 DISCOVER/HYPOTHESIZE 단계다. 현재 서로 다른 코드 경로를 읽기 전용으로 조사하고 있으며, 중앙 담당자가 출처, 관측 정보, 허용 변경, 비교군, 반증 조건, 최소 실험, 자원/위험 게이트를 갖춘 가설로 통합한다. corridor 경로의 제출 적격성은 아직 미확인이다. 탐색 결과가 구체적인 실험 설계·구현·실행 승인을 대신하지 않는다.

## 기존 방식과 전환 상태

지원 진입점은 `python -m haic_research.cli`다. `plan`은 manifest·가설·등록 profile·typed arguments를 고정하고, `approve`는 설계·구현·실행 승인을 순서대로 기록한다. `status`는 해당 실행의 상태를 조회하며, `run`은 기본 계획 미리보기이고 `--execute`가 있어야 승인된 로컬 작업을 한 번 실행한다. `report`는 평가 결과와 세 gate, 선택적으로 등록된 후보/control 비교를 기록한다. `validate`는 정해진 문서와 설정 구조만 검사한다. [README의 실제 문법](../README.md#supported-local-cli)과 [프로젝트 입력 계약](../PROJECT_INFO.md#v2-interfaces-and-operation-inputs)을 따른다.

기존 방식의 자동 탐색·RESULTS/SOTA 재생성·latest 포인터·임의 명령 실행은 퇴역 대상이며 [과거 경로 지도](sources/legacy-path-map.md)에 기록한다. 기존 결과와 연구 자료는 원래 위치에 보존한다. 논문은 가설의 근거이고 주행 측정이 성능의 근거라는 원칙, 측정 항목, 시작 문서·외부 연락·외부 전략 도입·단일 branch/dataset 과적합 주의 규약은 AGENTS/RESTRICTIONS에 보존한다.

## 실제 완료 상태와 한계

- 설정·상태·비교·기록·승인 실행 코드는 로컬 커밋으로 남아 있다. Task 7 결과 기록과 completion-first SOTA 승격 구현은 `02be453`에 커밋됐다.
- Task 7 인계 보고서에 기록된 focused 검증은 총 **131개: 130개 통과, 1개 건너뜀, 실패 0개**다. Windows 파일 symlink 권한을 사용할 수 없어 1개를 건너뛰었다. 임시 fixture와 FakeRunner를 사용했으며 실제 학습·주행 평가를 수행한 결과가 아니다.
- Task 8 구조 검증기는 `d44f1fb`에 커밋됐다. Task 9 문서 전환과 전체 새 하네스 검증은 완료했으며 기존 파일의 실제 삭제는 자동 정책 차단으로 남아 있다. 하네스 source 전환 전체가 완료됐다고 판단하지 않는다.
- 새 결과 기록은 이름이 정해진 v2 실행의 지표·비교 증거를 gate report 전에 고정한다. SOTA 승격은 사전 등록된 비교 ID·맵·split·seed·denominator·모델 식별자, 독립 반복 평가, 세 PASS gate, 실제 ADVANCE 릴리스 이력을 확인한다. 입력된 집계 지표가 실제 주행을 정확히 반영하는지는 별도 측정·검토 근거가 필요하다.
- 기존 사용자 변경과 과거 데이터는 그대로 보존한다. 기존 RESULTS 생성 구간과 SOTA의 역사적 내용도 보존됐다.
- Task 7 구현과 현재 가설 탐색에서는 실제 학습·주행 평가·패키징·공식 제출을 하지 않았다. 따라서 **새로운 완주율 개선이나 공식 성과, 새 SOTA 승격은 없다.**
- SOTA 문서의 완주율 0.75, 중앙 랩타임 19.32초는 기존 문서에 남아 있는 역사적 로컬 기록이다. 이번에 원자료를 다시 검증한 결과가 아니다.

## Task 9 전환 검증 기록

2026-09-26에 지정된 config/state/policy/coordinator/records/commands/CLI/results/validation 전체 단위 suite **148개 중 147개 통과, 1개 건너뜀, 실패 0개**를 확인했다. 건너뛴 항목은 Windows 파일 symlink 권한에 관한 항목이다. `scripts/harness/validate.ps1`은 exit 0, `valid=true`, issue 0개였다. 단위 검증은 임시 fixture와 FakeRunner를 사용했으며 실제 학습·주행 평가·패키징·제출 작업을 실행하지 않았다. 구조 검증은 문서/설정 구조만 확인하며 공식 규칙의 최신성, 측정된 주행 성능, 입력 지표의 진실성을 보증하지 않는다.

문서의 실제 v2 명령·입력 계약과 기존 정책 이관을 완료했다. 삭제 대상으로 지정된 정확한 파일 10개 모두 저장소 내부의 regular/untracked 경로임을 확인했다. 그러나 보호 검증을 포함한 묶음 삭제와 RULES 단일 파일의 literal 삭제가 자동 승인 검토에서 "blocked by policy" 사유로 거부됐다. 추가 이유는 제공되지 않았다. 두 번째 거부 후 삭제 시도를 중단했으며 기존 진입점·정책·전용 테스트 파일 10개는 모두 남아 있다. 데이터·보고서·checkpoint·ZIP·map·연구 자료와 디렉터리는 삭제하지 않았다. 실제 source 퇴역은 이 차단이 해소된 뒤 정확히 지정된 파일만 제거하는 별도 남은 단계다.
