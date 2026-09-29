# HAIC 실격 및 실험 제한

현재 공식 규칙과 충돌하면 [AGENTS.md](AGENTS.md)의 출처 우선순위를 적용하고 외부 조치를 멈춘다. 아래 수치와 금지 목록은 로컬 스냅샷이며 제출 직전 공식 대회 사이트와 Participants 저장소에서 다시 확인한다.

## 제출 계약

- ZIP 최상위에 `agent.py`가 있어야 하고 `Agent.act()`는 shape `(3,)` 유한값을 반환해야 한다.
- import+생성 10초, reset 5초, act 5초, 메모리 1,024 MB 제한을 지킨다.
- ZIP 500 MB, 파일 수 1,000개, 압축 해제 후 2 GB, 개별 파일 500 MB 제한을 지킨다.
- 금지 import: `ctypes`, `importlib`, `multiprocessing`, `os`, `pathlib`, `resource`, `shutil`, `signal`, `socket`, `subprocess`, `sys`
- 금지 동적 실행: `compile`, `eval`, `exec`, `__import__`
- `.dll`, `.dylib`, `.exe`, `.so` 등 실행 파일을 넣지 않는다.

## 정보 누수와 runtime

- train 외 tune/held-out map·seed·obstacle 정답·geometry를 학습/증강에 사용하지 않는다.
- map 좌표와 simulator state를 제출 추론 입력으로 전달하지 않는다.

## V2 연구 운영 제한

- `train`, `tune`, `held_out`, `confirmation`, `blind`, `official` split을 분리한다. 소비한 confirmation/blind cell을 새 평가 자료로 재명명하지 않고, 예약 blind cell로 반복 튜닝하지 않는다. 공식 private track은 최종 일반화 목표이며 이용 가능한 holdout이 아니다.
- 사전 등록된 같은 map/seed, denominator, protocol에서만 후보와 control을 matched comparison으로 부른다. 단일 seed나 teacher/smoke 기록을 제출 후보의 replicated completion rate로 계산하지 않는다.
- `rule_compliance=PASS`와 `mechanism_activation=PASS`가 없는 결과는 SOTA로 승격하지 않는다. `UNKNOWN`도 통과가 아니다. 내부 지표는 공식 점수와 구분한다.
- 설계 승인과 실행 승인을 각각 정확한 plan revision/hash에 기록한다. 설계 승인 범위의 구현에는 별도 구현 승인이 필요하지 않다. 기본은 plan-only이며 허용된 로컬 profile만 실행한다. 임의 shell 명령을 profile로 등록하지 않는다.
- 기존 `runs/`, `artifacts/haic/`, `submissions/` 자료는 자동 검색·복사·이동·재작성하지 않는다. 새 출력은 `harness.config.json`의 v2 roots에 쓴다.
- 공식 제출, 모델 확인, 대회 사이트 업로드는 CLI 작업이 아니며 실행 직전 별도 명시적 사용자 승인을 받는다.

## 기록

- smoke나 teacher 단독 완주를 제출 후보 성능으로 기록하지 않는다.
- 실패 실험도 원본 JSON과 함께 남기며 유리한 시드만 보고하지 않는다.

## 비밀정보와 제출 로그

- 문서·로그·산출물에 비밀값, 인증 정보, 개인정보를 기록하지 않는다.
- 제출물의 표준 출력에 임의의 디버그 로그를 추가하지 않는다. 분석은 별도 기록과 계측으로 남긴다.
