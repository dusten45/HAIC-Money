# DAgger recovery-timing V2 adapter design

## 상태

- 단계: `DESIGN_PENDING_APPROVAL`
- revision: `1`
- design SHA-256: 최종 내용 고정 후 기록
- 대상: 기존 144셀 확인 실험을 V2의 등록된 로컬 프로필과 산출물 경로에 연결하는 준비 작업
- 이 문서는 범위 승인용 설계다. 아직 실행 승인이나 시뮬레이션 권한을 부여하지 않는다.

## 가설 등록

- `hypothesis_id`: `H-DAgger-tail5-v2-adapter`
- `source_ref`: D 작업의 보존 스냅샷 `81a759bf8102d7fddbfcad46dba5c60d4dfef000`; 기존 실험 프로토콜 SHA-256 `19B9DA2789736F192104425C521C4C5C6567DD3FD0FCE4024597B3679CC7C645`
- `rule_or_requirement`: 새 실행은 V2의 명시된 로컬 프로필을 통해서만 시작하고, manifest·event log·integration report는 `runs/haic-research-v2/<run-id>/`, 새 실험 산출물은 `artifacts/haic-research-v2/<run-id>/` 아래에 기록한다. 과거 `runs/`, `artifacts/haic/`, `submissions/` 자료는 자동 검색·복사·이동·수정하지 않는다.
- `observable_information`: 동일한 여섯 고정 actor와 여덟 로컬 geometry에서 control, 즉시 teacher steering, detection 이후 5-decision teacher steering tail의 144개 셀 및 action trace.
- `allowed_action_or_state_change`: V2에 전용 local diagnostic profile과 출력 경로 어댑터만 등록한다. 144셀 조건, 순서, 체크포인트, map/seed, teacher 동작은 바꾸지 않는다. 기존 셀 실행기와 `training/evaluate_closed_loop.py`, 공식 대회 harness 및 제출 코드는 수정하지 않는다.
- `expected_success_endpoint`: profile이 V2의 plan-only 경로에서 manifest/input identity를 고정하고, 추후 별도 승인된 실행은 raw cell/freeze/receipt를 V2 artifact root에만 write-once로 남긴다. 이 확인의 성능 endpoint는 tail5 완주 수 대 control 및 즉시 steering의 사전 등록 비교다.
- `eligible_state`: 내부 local confirmation만 해당한다. 모델 학습·승격, SOTA 갱신, 공식 평가·제출에는 적격하지 않다.
- `control`: 같은 actor × geometry × episode budget에서 actor-only control 및 즉시 teacher steering.
- `falsifier`: profile이 공유 evaluator 또는 공식 harness 동작을 바꾸거나, 출력이 V2 root 밖으로 나가거나, source/checkpoint hash·144셀 분모·trace 검증이 맞지 않으면 어댑터 설계를 기각한다. 하나라도 유효하지 않은 셀이 있으면 confirmation은 inconclusive이며 셀 대체·재시도를 하지 않는다.
- `smallest_decisive_experiment`: 먼저 등록된 profile의 plan-only 생성/미리보기와 출력 경로·입력 hash 확인만 수행한다. 주행 144셀 실행은 별도 immutable run plan 및 design/implementation/execution 승인 후에만 가능하다.
- `resource_and_risk_gate`: 실행 한도는 기존 프로토콜 그대로 CPU 1 thread, deterministic inference, 최대 2,000 decisions/episode, 5,400초 실행 budget, 144 cells, 학습 0 step이다. held-out/blind/official 자료, 새 checkpoint, 패키징, 업로드는 금지한다. V2 plan은 해당 run의 명시 입력·소스 hash를 고정해야 한다.

## 설계

1. 보존 스냅샷에서 기존 protocol, plan, 확인 실행기와 tail 구현을 읽기 전용 기준으로 삼는다. 아카이브된 D 작업공간을 되살리거나 그 안의 run 결과를 복사하지 않는다.
2. 실험 동작은 보존된 확인 실행기와 같게 유지하고, 새 V2 adapter에서 freeze·cell·receipt 출력 위치만 profile이 받은 run-specific artifact directory로 매핑한다. 새 protocol revision은 조건·순서·분모를 유지하고 V2 경로와 현재 허용된 metadata/checkpoint hash만 새로 고정한다.
3. `dagger_obstacle_recovery_timing_confirmation_v2` local diagnostic profile을 등록한다. 필요한 profile allowlist/schema 연결만 추가하고, V2 workflow/state machine과 공통 evaluator는 수정하지 않는다. profile은 SOTA 승격 대상이 아니다.
4. 기존 여섯 checkpoint를 V2 plan 입력 경로로 복사하지 않는다. 새 plan은 명시 checkpoint manifest에 경로와 기대 SHA-256을 고정하고 adapter가 실행 직전 및 실행 중 원본 바이트를 검증한다. 현재 확인한 보존 스냅샷과 원 체크아웃에는 이 여섯 checkpoint가 없어, 해당 파일이 명시 경로에서 복구되기 전에는 실행 plan을 만들거나 실행하지 않는다.
5. V2 CLI의 manifest/events/integration report와 adapter가 만든 raw confirmation artifact는 각자 지정된 v2 root에 기록한다. legacy run 경로는 입력 출처로만 명시하며 쓰지 않는다.

## 계획된 파일 범위

- 수정: `harness.config.json` — 전용 local profile 등록
- 수정: `haic_research/config.py` — 해당 profile/module/typed-argument allowlist만 추가
- 수정: `README.md`, `PROJECT_INFO.md` — 새 profile 사용법과 plan-only 동작을 문서화
- 추가: `training/run_dagger_obstacle_recovery_timing_confirmation_v2_adapter.py` — V2 출력 경로 연결, 명시 입력 hash 검증, 기존 셀 규약 유지
- 추가: 새 revision protocol 및 checkpoint manifest (체크포인트 원본의 존재/hash가 확인된 뒤 생성)
- 추가: V2 run plan/report artifact (계획 단계 승인 후)
- 수정 금지: `haic_research/commands.py`, `training/evaluate_closed_loop.py`, `training/benchmark_corridor.py`, 공식 Participants 코드, `agent.py`, 제출 패키징 경로, 과거 run/checkpoint/receipt 파일

## 보존·승인 원칙

- 보존된 Git snapshot ref는 `refs/codex/snapshots/3693b7378da3c32fedc142e117d8faeb74e8fd91`이다. 해당 tree에는 protocol/plan/source가 있지만 여섯 checkpoint blob은 없다.
- snapshot의 protocol JSON은 Windows CRLF 바이트 기준으로 SHA-256 `19b9da2789736f192104425c521c4c5c6567dd3fd0fce4024597b3679cc7c645`와 일치한다. Git blob은 LF 정규화 때문에 `5dc98b3a2a194a0c542aa62e0962d7c3e9bb9cf007bdf89c32a07c52226fb431`이다.
- snapshot에 남은 16개 declared metadata 중 14개만 protocol의 파일 hash와 일치했다. `docs/context/current-state.md`와 `docs/evaluation/generalization-policy.md`는 불일치하므로 snapshot을 동결 가능한 provenance로 간주하지 않는다. 새 protocol에서 허용된 metadata만 다시 검사·고정해야 하며, held-out/blind 결과를 열어 seed를 고르지 않는다.
- 기존 프로토콜이 지정한 여섯 checkpoint 경로는 main checkout과 보존 snapshot tree 모두에 없다. 따라서 adapter 구현과 별개로, exact checkpoint 바이트를 허용된 명시 경로에서 복구하고 plan 입력 hash를 고정하기 전에는 run plan/freeze/실행이 불가하다.
- 사용자 범위 승인: “로컬 어댑터만 추가”. 이 답변은 공유 evaluator·공식 harness 변경 및 실험 실행을 승인하지 않는다.
- 위 승인에는 이 문서의 hash가 없었으므로 V2 workflow의 hash-bound design/implementation/execution approval 기록으로 소급 등록하지 않는다.
- 이 설계 revision을 승인받기 전 구현하지 않는다. 구현 완료 후 새 run plan/hash를 만들고, 해당 run plan에 design, implementation, execution 승인을 각각 기록한다. 실행 승인은 144셀을 한 번 실행하는 권한으로만 사용한다.
