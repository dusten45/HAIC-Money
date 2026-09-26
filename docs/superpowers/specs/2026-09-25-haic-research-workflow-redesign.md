# HAIC Research Workflow Redesign

- 상태: 사용자 수정사항 반영, 구현 계획 검토 전
- 날짜: 2026-09-25
- 범위: HAIC 저장소의 연구·실험·실행 운영 구조 재설계
- 기준 문서: 사용자가 제공한 「이식 가능한 에이전트 하네스 운영 문서」

## 1. 의도와 경계

이 저장소에 여러 프로젝트에서 재사용할 포터블 에이전트 제품을 만드는 것이 목적은 아니다. 제공된 하네스 문서를 HAIC에 적용할 운영 기준으로 삼아, 문서 책임·작업 절차·보고 형식·기록 구조를 가능한 한 충실히 적용한 HAIC 전용 연구·실행 체계를 만든다. 규칙, 명령, split, 지표는 HAIC의 공식 계약과 사용자 목표에 맞게 채운다.

기존 연구 오케스트레이션 방식을 장기 지원하거나 새 방식과 병행하지 않는다. HAIC의 현재 실행·기록 구조를 문서화한 뒤 새 운영 코드와 필요한 실행 경로를 만든다. `research_ops`와 `training/improvement_loop.py`는 새 체계의 기반이나 필수 의존성이 아니며, 전환이 확인되면 기존 오케스트레이션·색인·자동 SOTA 동기화 역할을 퇴역시킨다. HAIC 학습·평가·패키지 기능은 개별적으로 검토해 재사용하거나 재작성한다.

### 연구 목표 우선순위

제출 자격을 충족한 후보 사이의 내부 연구 선택에서 **완주율이 최우선 성과 지표**다. 같은 사전 등록 평가 집합과 동일 조건에서 완주율이 높은 후보를 우선한다. 완주율이 같을 때만 완주 랩타임 중앙값, 미완주 진행도, P90 랩타임, 충돌·손상, 추론 비용 순으로 비교한다. 규칙 준수는 성과 순위와 별개의 필수 게이트다.

이 우선순위는 사용자의 연구 목표이며 공식 리더보드 점수 공식을 대신하지 않는다. 보고서에는 공식 규칙상 점수와 내부 완주율 우선 비교를 구분한다. Teacher·smoke·단일 seed 기록은 PPO 제출 후보의 완주율로 합산하지 않는다.

## 2. 제공된 하네스 문서를 HAIC에 적용하는 정도

제공 문서의 구조를 최대한 따른다. 프로젝트별 값을 채우되, 공통 운영 아이디어와 요구 산출물은 임의로 생략하지 않는다.

- 중앙 오케스트레이터가 전체 상태·우선순위·탐색 배치를 관리한다.
- 독립 에이전트는 서로 다른 탐색 축을 맡고 공유 문서를 동시에 수정하지 않는다.
- 모든 가설은 공식 규칙 또는 관측 근거, 성공 endpoint와 실패 조건에 연결한다.
- 문서에 지정된 상태 머신과 설계·구현·실행 승인 게이트를 사용한다.
- 배치당 독립 방향 4개 이상, 전체 후보 8개 이하, 방향별 후보 2개 이하를 기본으로 한다.
- 메커니즘이 실제 행동이나 상태를 바꾸기 전에는 threshold·가중치 미세 탐색을 하지 않는다.
- 비교 가능한 유효 주기 3회 연속 비개선이면 pivot한다. 인프라 실패와 무효 실행은 횟수에 포함하지 않는다.
- 에이전트 보고에서 `fact`, `inference`, `unknown`, `recommendation`을 구분하고 근거 경로를 붙인다.
- `run_manifest.json`, `events.jsonl`, `integration_report.json`과 세 판정 게이트를 구현한다.
- blind·confirmation 자료를 보호하고 과거 원자료를 복사하거나 덮어쓰지 않는다.

여기서 “최대한 따른다”는 문서에 있는 항목을 HAIC 구조에서 구현·운영한다는 뜻이다. 교차 프로젝트용 설치 절차나 범용 core/adapter 제품을 만든다는 뜻은 아니다. 실제로 적용할 수 없는 항목이 생기면 validator/설계 문서에서 이유와 대체 방식을 명시한다.

## 3. 현재 HAIC 구조와 바꿀 이유

| 영역 | 현재 역할과 관찰 |
|---|---|
| `README.md` | 실행 환경·에이전트 계약·패키지와 PPO/CEM/teacher 실험 설명을 함께 담는다. 시작 안내와 연구 기록이 길게 섞여 있다. |
| `COMPETITION_INFO.md` | 관측·행동 계약, 제한, 종료·점수, ZIP 구성을 요약한다. 공식 원문이 아닌 로컬 요약이다. |
| `RESTRICTIONS.md` | 제출 계약, 누수 방지, data split과 기록 제한을 담는다. |
| `RULES.md` | `research_ops.cli improve`를 단일 실행 흐름으로 지정하고 DB 색인·비교·SOTA 갱신 및 선택적 명령 실행을 연결한다. |
| `research_ops/` | artifact 검색·정규화, 비교·승격, 계획 생성, 결과 동기화 및 CLI를 구현한다. |
| `training/improvement_loop.py` | 별도로 결과 검색·감사·다음 실험 제안·RESULTS/SOTA 작성·선택적 명령 실행을 구현한다. |
| `RESULTS.md` | 사람이 쓴 요약과 `research_ops`가 생성한 과거 JSON 색인 구역이 함께 있다. |
| `SOTA.md` | 현재 문서상 held-out 완주율 0.75, 중앙 랩타임 19.32초의 PPO actor-only 로컬 기록을 가리킨다. 공식 순위와 동일하지 않다. |
| `docs/` | 현재 HAIC 설계·실험 계획은 `docs/superpowers/specs/`, `docs/superpowers/plans/`에 있다. 저장소 루트에 단일 운영 규약인 `AGENTS.md`는 없다. |

중복된 오케스트레이션과 자동 결과 재생성을 걷어내고, 운영 규칙·실험 기록·실행 코드의 단일 책임자를 둔다. 기존 문서는 제공 문서의 책임표에 맞춰 정리하되, 동일 목적 문서를 여러 개 만들지 않고 유일한 역사 정보는 보존한다.

## 4. HAIC에 적용할 파일·폴더 구조

사용자 제공 문서의 구조를 기본으로 적용한다.

```text
project-root/
├─ AGENTS.md
├─ PROJECT_INFO.md
├─ RESTRICTIONS.md
├─ RESULTS.md
├─ SOTA.md
├─ harness.config.json
├─ docs/
│  ├─ experiments/INDEX.md
│  ├─ handoffs/
│  ├─ sources/
│  ├─ strategy-history.md
│  └─ report.md
├─ runs/
├─ artifacts/
├─ output/pdf/
└─ scripts/harness/validate.ps1
```

`harness.config.json`와 `scripts/harness/validate.ps1`는 HAIC 전용 설정·검증기다. 공용 패키지나 다른 프로젝트를 위한 adapter framework로 만들지 않는다. `README.md`는 빠른 시작과 위 문서들의 경로 안내를 맡는다.

### 문서별 책임

| 문서 | HAIC에서 맡을 책임 |
|---|---|
| `AGENTS.md` | 단일 운영 규약: 출처 우선순위, 상태 머신, 중앙 오케스트레이터·에이전트 역할, 승인 게이트, 검색 규칙, 보고 형식 |
| `PROJECT_INFO.md` | HAIC 목적, 공식 출처, 성공 endpoint, 완주율 우선순위, 허용된 실행·평가 진입점 |
| `RESTRICTIONS.md` | 제출·runtime·보안·정보 누수·split·외부 실행 제한 |
| `RESULTS.md` | append-only 결과 원장. 과거 생성 표와 설명은 역사 자료로 보존하고 새 기록은 추가 |
| `SOTA.md` | 독립 평가와 필수 게이트를 통과한 제출 가능 후보. 완주율을 먼저 비교 |
| `harness.config.json` | HAIC 전용 경로, 정책, split ID, 예산, 명령 profile, pivot 기준. 비밀값 금지 |
| `docs/experiments/` | 가설, control, 실험 조건, 성공 endpoint, 반증 조건, 비용, 판정, 원자료 경로 |
| `docs/handoffs/` | 독립 에이전트의 범위·결과·불확실성·추천을 중앙 오케스트레이터에 전달 |
| `docs/sources/` | 공식 규칙·사양·외부 근거의 URL, 버전/확인일, 적용 사실 |
| `docs/strategy-history.md` | 채택·폐기 전략과 근거의 append-only 이력 |
| `docs/report.md` | 검증 성과, 제한, 실패 원인, 다음 우선순위의 최신 종합 |
| `runs/` | 신규 실행 manifest, events와 실행별 원자료 위치 |
| `artifacts/` | 신규 checkpoint, package, 비교 산출물과 hash |
| `output/pdf/` | 신규 보고서 PDF. 기존 `report.pdf`는 이동하거나 덮어쓰지 않음 |
| `scripts/harness/validate.ps1` | 필수 문서·설정·경로·정책 간 구조 검증. 공식 평가기를 복제하지 않음 |

현재 `COMPETITION_INFO.md`의 고유 사실은 `PROJECT_INFO.md`/`RESTRICTIONS.md` 책임에 따라 옮기고, 기존 경로 참조가 필요하면 호환 안내만 둔다. `RULES.md`의 유효 정책은 `AGENTS.md`로 옮기고 활성 실행 절차로는 남기지 않는다. 기존 `RESULTS.md` generated 구역은 동결한다. 이동 전 전체 링크·참조를 확인하며 고유한 내용은 보존한다.

## 5. 권위와 출처

- 작업 범위와 승인 권한은 최신 사용자 지시를 따른다.
- HAIC 대회 규칙·제출·일정 사실은 최신 공식 대회 사이트, 공식 Participants 저장소, 로컬 요약, 역사 자료 순으로 확인한다.
- 공식 자료가 로컬 문서와 충돌하면 외부 제출·모델 확인을 중단하고 공식 자료에 맞게 로컬 기준을 정리한다.
- 결과 문서·오래된 계획·과거 SOTA는 특정 실행의 근거일 뿐 현재 규칙이나 새 실행의 기본값이 아니다.
- 모든 `docs/sources/` 기록에 URL, 버전/게시일 또는 확인일, 사용한 주장을 남긴다.

## 6. 중앙 조정과 에이전트 계약

HAIC 중앙 오케스트레이터는 전체 연구 주기를 관리한다.

- 현재 상태, 활성 주기, 우선순위와 실행 예산을 관리한다.
- 에이전트별로 서로 겹치지 않는 조사 축을 배정한다.
- 중복 가설, 상충 근거, 규칙 위반과 권한 문제를 확인한다.
- 실행 전에 control·split·완주 지표·실패 조건·예산 승인을 확인한다.
- 개별 보고를 통합해 다음 방향과 SOTA 승격 여부를 결정한다.

에이전트는 지정된 조사 범위와 산출물을 지키고 공유 문서를 동시에 수정하지 않는다. 보고는 다음 형식을 사용한다.

```text
fact:
inference:
unknown:
recommendation:
source_paths:
```

코드가 존재하거나 점수가 높다는 사실만으로 전략 활성화나 인과를 확정하지 않는다. handoff는 중앙 판정을 대체하지 않는다.

## 7. 가설, 상태 머신과 탐색 정책

신규 가설은 아래 필드를 갖는다.

```text
hypothesis_id:
source_ref:
rule_or_requirement:
observable_information:
allowed_action_or_state_change:
expected_success_endpoint:
eligible_state:
control:
falsifier:
smallest_decisive_experiment:
resource_and_risk_gate:
```

인과 설명은 `공식 규칙/요구사항 → 에이전트가 볼 수 있는 정보 → 허용된 행동/상태 변화 → 성공 endpoint`를 이어야 한다.

모든 작업은 다음 상태를 따른다.

```text
STOPPED
→ DISCOVER
→ HYPOTHESIZE
→ DESIGN_PENDING_APPROVAL
→ IMPLEMENT_PENDING_APPROVAL
→ EXECUTE_PENDING_APPROVAL
→ EVALUATE
→ ADVANCE / REJECT / REVISE / PIVOT
→ RELEASE_IF_GATE_PASS
→ STOPPED
```

- `ADVANCE` is the only evaluation outcome that can enter `RELEASE_IF_GATE_PASS`; all three gates must pass before that state returns to `STOPPED`.
- `REJECT`, `REVISE`, and `PIVOT` return to `STOPPED` without releasing the current candidate. Revised or pivoted work starts a new cycle at `DISCOVER` with a new plan hash and fresh approvals; `PIVOT` preserves the prior checkpoint.
- `STOPPED`가 기본이다. 문서 조사만으로 학습·평가·제출 실행 상태에 들어가지 않는다.
- 설계, 구현, 실행 승인은 서로 분리하고 승인한 계획 revision/hash를 기록한다.
- 배치에는 최소 4개 독립 메커니즘 방향, 전체 최대 8개 후보, 방향별 최대 2개 후보를 둔다.
- 메커니즘이 실제 행동이나 상태를 바꾸기 전에는 threshold·가중치 작은 값 비교를 하지 않는다.
- 같은 split·control·프로토콜로 비교 가능한 유효 주기 3회 연속 비개선 시 pivot한다. infra invalid는 세지 않는다.
- train/tune/held-out/confirmation/blind를 분리한다. 사용된 confirmation/blind는 새 자료가 아니며 blind를 튜닝하지 않는다.
- 외부 브랜치·논문·결과는 가설과 반례를 찾는 데만 쓴다. 이름·순위·빈도로 활성 동작이나 인과를 추정하지 않는다.

## 8. 완주율 우선 성과 판정

완주율은 제출 자격이 있는 후보 사이의 **primary outcome**이다.

1. `rule_compliance`가 `PASS`가 아니면 승격할 수 없다.
2. 사전 고정한 같은 map/seed 집합과 평가 조건에서 candidate와 control을 비교하고, 완주 수/전체 수와 완주율을 가장 먼저 본다.
3. 완주율이 더 낮은 후보는 더 빠른 완주 랩타임만으로 우선할 수 없다.
4. 완주율이 같을 때 완주 랩타임 중앙값, 미완주 진행도, P90 랩타임, 충돌·손상, 추론 비용 순으로 비교한다.
5. 단일 seed나 불일치 split은 replicated matched improvement가 아니다. 표본 수·seed·map geometry·split을 함께 기록한다.
6. 공식 score와 완주율 우선 내부 선택 순위를 보고서에서 구분한다.

Teacher, smoke, fixed-actor 진단은 제출 후보의 완주율과 별도로 기록한다. `SOTA.md`의 기존 0.75 PPO 결과는 새 체계로 다시 검증되기 전까지 historical local reference이며 새 SOTA로 자동 승격하지 않는다.

## 9. 실행 기록과 판정 게이트

각 신규 실행은 다음을 남긴다.

- `run_manifest.json`: run ID, 목적, 가설·승인 hash, candidate/control revision·package hash, tool/runtime version, data/map/split ID, 자원·권한, 결과 파일과 원자료 경로·hash
- `events.jsonl`: 상태 변화, 승인, 실행 시작·종료, 오류·충돌·timeout, 자원 사용, 메커니즘 활성 신호와 성공 endpoint
- `integration_report.json`: 근거 경로를 포함한 세 독립 gate 판정

세 gate는 `PASS`, `FAIL`, `UNKNOWN`, `NOT_APPLICABLE` 중 하나다.

현재 설정에는 `NOT_APPLICABLE` 릴리스 면제 항목이 없으므로 릴리스는 세 gate 모두 `PASS`일 때만 가능하다. 미래에 면제가 필요하면 명시적 설정 검증과 transition API 지원을 함께 추가해야 하며, 규칙 준수 gate는 면제할 수 없다.

- `rule_compliance`
- `mechanism_activation`
- `competitive_or_product_outcome`

`UNKNOWN`은 통과가 아니다. 점수만 바뀌고 등록 메커니즘이 작동하지 않았으면 개선으로 인정하지 않는다. 원자료 전체를 매번 복사하지 않고 선택 필드·hash·원자료 위치를 남긴다. 정정은 기존 event를 덮지 않고 참조하는 새 event로 추가한다.

새 실행은 과거 산출물과 충돌하지 않는 경로를 쓴다. 초안은 `runs/haic-research-v2/<run-id>/`와 `artifacts/haic-research-v2/<run-id>/`를 제안하며, 최종값은 `harness.config.json`에서 한 번만 정의한다.

## 10. 실행·보안 경계

- 기본은 plan-only이며, 등록된 HAIC operation과 승인된 계획 hash가 있을 때만 학습·평가·package 작업을 실행한다.
- command profile은 명령 종류, typed arguments, timeout, read/write 경로를 구조화한다. 임의 shell 문자열 실행은 허용하지 않는다.
- secret은 config, manifest, log, report에 저장하지 않는다.
- 공식 제출, 모델 확인, 대회 사이트 업로드는 CLI operation이 아니다. 실행 직전 별도 명시 승인이 필요하다.
- `validate.ps1`는 필수 문서, 설정 schema, 경로, 완주율 우선순위, split 보호, command profiles와 plan-only 기본값을 검사한다. 공식 package/runtime/evaluator 검사를 중복 구현하지 않는다.

## 11. 데이터 보존과 전환 순서

- 기존 `runs/`, `artifacts/haic/`, `submissions/`, raw JSON·checkpoint·ZIP·PDF는 현재 경로에 둔다. 추가 복사본 약 1.8 GiB를 만들지 않는다.
- 새 시스템은 과거 경로를 자동 스캔하거나 결과 DB로 가져오지 않는다.
- 읽기 전용 legacy path map만 두며, 과거 자료를 연구에 참고하려면 사람이 선택한다. 과거 기록을 신규 실행 결과인 것처럼 사용하지 않는다.
- 과거 JSON 내부 경로, manifest, checksum, `RESULTS.md`의 generated 구역을 다시 쓰지 않는다.
- 문서 구조와 validator를 먼저 만들고, HAIC workflow·기록·명령 profile을 그 contract대로 구현한다.
- 새 흐름을 계획 기반으로 확인한 뒤 기본 경로로 전환하고 구형 orchestration/index/promotion entry point를 퇴역시킨다. 불필요한 옛 오케스트레이션 코드는 계획에 따라 제거한다. 과거 실험 데이터는 보존한다.

이 설계는 데이터 복사·이동·삭제, 학습·평가 실행, 공식 제출·모델 확인을 승인하지 않는다. 각 실행은 해당 계획의 별도 게이트를 따른다.
