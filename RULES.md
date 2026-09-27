# HAIC 성능 개선 운영 규칙

## 1. 목표와 우선순위

이 파일은 HAIC 에이전트의 조사, 실험, 검증, 실행, 제출, 기록을 총괄한다.
목표는 **공식 규칙을 완전히 준수하면서, 처음 보는 트랙에서도 안정적으로 완주하고,
완주한 경우 가능한 한 빠른 에이전트**를 만드는 것이다. 우승은 목표이지 보장된 결과로
표현하지 않는다.

의사결정 우선순위는 항상 다음과 같다.

1. 실격 방지와 공식 규칙 준수
2. 미지 트랙에서의 유효 완주율과 실행 안정성
3. 공식 평가 지표
4. 완주 시간
5. 보조 지표와 구현 편의성

속도 향상이 완주율, 일반화, 규칙 준수 또는 실행 안정성을 훼손하면 성능 개선으로
간주하지 않는다. 현재 구현과 알고리즘은 임시 수단이며, 근거가 있으면 교체할 수 있다.

## 2. 지시 및 근거의 우선순위

프로젝트 작업 지시는 `AGENTS.md`를 따른다. 현재 작업 범위와 활성 실험, 동결된 계약,
사용한 seed 및 다음 우선순위는 작업 시작 전에 `CONTEXT.md`에서 확인한다.

대회 사실의 근거 우선순위는 다음과 같다.

1. 현재 공식 대회 사이트와 공식 participant 저장소의 명시적 규칙
2. `RESTRICTIONs.md`의 실격·금지 조건
3. `COMPETITION_INFO.md`의 환경, 평가, 일정, 제출 절차 및 baseline 정보
4. 재현 가능한 공식 코드와 실행 결과
5. `RESULTS.md`, `SOTA.md`, `report.pdf` 및 인용된 1차 문헌
6. 추정, 관행 또는 현재 코드의 동작

공식 자료와 로컬 문서가 충돌하면 공식 자료를 기준으로 하고, 조회 일시, URL, commit 또는
버전을 기록한 뒤 관련 로컬 문서를 즉시 고친다. 불명확한 규칙은 유리하게 해석하지 않는다.
공식 근거로 해소하기 전에는 해당 실험 또는 제출을 중단한다.

공식 확인의 시작점은 다음 두 곳이다. 링크 자체도 변경될 수 있으므로 `AGENTS.md`와 공식
공지에서 현행 주소인지 함께 확인한다.

- participant 저장소: <https://github.com/2026-HAIC/Participants>
- 대회 사이트: <https://ships-duo-ethical-saver.trycloudflare.com/>

## 3. 필수 자료의 역할

- `RESTRICTIONs.md`: 금지 사항, 실격 조건, 자원·의존성·인터페이스 제한과 각 조건의 검증법
- `COMPETITION_INFO.md`: 최신 공식 규칙, 평가 지표, 환경, baseline, 일정 및 정확한 제출 절차
- `RESULTS.md`: 성공·실패·중립 결과를 모두 보존하는 실험 DB
- `SOTA.md`: 동일한 평가 계약으로 검증된 최고 기록과 재현 명령
- `report.pdf`: 인용 문헌, 방법, 가정 및 제출 주장의 근거
- `CONTEXT.md`: 현재 상태, 활성 실험, 동결 사항, 소비·예약 seed와 다음 우선순위

필수 파일이 없거나 읽을 수 없거나 최신 여부를 확인할 수 없으면 내용을 추측하지 않는다.
공식 자료와 원본 artifact를 찾아 복구하거나, 출처가 표시된 문서를 먼저 구축한다. 그 전에는
학습, blind 평가, 제출 또는 SOTA 갱신을 진행하지 않는다. 파일명은 대소문자를 포함해 위 표기를
정본으로 사용하며 임의로 유사 이름을 만들지 않는다.

## 4. 한 번의 성능 개선 사이클

각 후보는 아래 게이트를 순서대로 통과해야 한다. 뒤 단계의 좋은 결과로 앞 단계의 실패를
무시할 수 없다.

### 4.1 범위와 기준선 동결

1. `AGENTS.md`, `CONTEXT.md`, `RESTRICTIONs.md`, `COMPETITION_INFO.md`,
   `RESULTS.md`, `SOTA.md`를 읽고 `report.pdf`의 관련 절과 참고 문헌을 확인한다.
2. 공식 사이트와 participant 저장소에서 규칙, 환경, 평가 및 제출법의 최신성을 확인한다.
3. 현재 branch, source commit, 환경 버전, baseline artifact hash, 평가 protocol과 자원 제한을
   기록한다.
4. 진행 중인 실험과 동결 artifact는 건드리지 않는다. 소비한 seed는 재사용하지 않고, geometry
   seed는 모든 track ID에 걸쳐 예약하며, 예약한 confirmation/blind seed는 사전 등록된 목적
   외에는 열지 않는다.
5. 현재 요청이 문서·분석만을 요구하면 학습, 평가 또는 제출로 범위를 넓히지 않는다.

### 4.2 실패 원인 조사와 문헌 검토

1. `RESULTS.md`에서 같은 현상, 유사 변경, 실패 사례와 비용을 먼저 검색한다. 실패한 접근을
   새 이름으로 반복하지 않는다.
2. `SOTA.md`의 현 champion이 어떤 protocol과 지표로 선정되었는지 확인한다.
3. 공식 구현, 원 논문, 저자 코드 등 1차 자료를 우선 검색한다. 각 자료에 URL/DOI, 조회일,
   핵심 주장, 적용 조건, 이 대회 환경과의 차이를 기록한다.
4. 검색한 주장을 `report.pdf`의 설명 및 인용과 대조한다. 논문에서 입증하지 않은 효과를
   사실처럼 확대하지 않으며, 환경 차이가 크면 실험 가설로만 취급한다.
5. 관찰된 병목을 인지/제어/학습/일반화/자원/패키징 중 어디에 속하는지 구분하고, 측정 가능한
   실패 원인 하나를 우선 선택한다.

### 4.3 실험 사전 등록

실행 전에 `RESULTS.md` 또는 별도 version-controlled protocol에 다음을 기록한다.

- 고유 experiment ID, 가설과 예상 메커니즘
- 비교할 champion/control과 정확히 한 가지의 주된 변경점
- 고정할 설정, 학습 budget, 데이터 및 source commit
- development/confirmation/blind track과 seed의 사전 분리
- 공식 지표, 완주율, progress, 완주 시간, 충돌·이탈·timeout, latency·memory·오류율
- 성공, 실패, 조기 중단, rollback 및 SOTA 승격 기준
- 필요한 ablation, 반복 횟수, 예상 계산 비용
- 모든 제한 조건별 준수 방법과 생성할 증거

가능하면 한 번에 한 요인만 바꾼다. 여러 변경을 묶어야 하면 이유를 기록하고, 효과를 분리할
후속 ablation 없이는 원인을 단정하지 않는다. 결과를 본 뒤 seed, 지표, 제외 조건 또는 성공
기준을 바꾸지 않는다.

### 4.4 구현 및 실행 전 준수 감사

구현은 기존 사용자 변경과 동결 파일을 보존하고, 작은 단위 테스트와 재현 가능한 설정을
포함한다. 실제 장기 실행 전에 다음을 증거와 함께 확인한다.

- 공식 observation/action/reward/termination 및 reset 계약
- 허용된 데이터, 모델, 라이브러리, 파일, network 및 실행 방식
- 초기화·step timeout, CPU/GPU 조건, memory, process 수, package 크기
- 제출 archive 구조, entry point, dependency와 금지 import
- 숨은 상태, track 식별자, test seed 또는 평가 결과의 누출 여부
- 결정론적 재실행, 예외·NaN·멈춤 및 안전한 fallback

`RESTRICTIONs.md`의 각 항목은 `통과/실패/미확인 + 검증 명령 또는 artifact`로 판정한다.
하나라도 실패하거나 미확인이면 장기 실행과 제출을 금지한다. 제한을 우회하는 구현은 성능과
관계없이 폐기한다.

### 4.5 단계적 평가

평가는 비용과 정보 누출이 작은 순서로 진행한다.

1. 정적 검사, 단위 테스트, interface 및 package smoke test
2. 짧은 local smoke와 오류·자원·latency 검사
3. 사전 등록된 development 평가
4. screen 결과만으로 finalist를 사전 선택하고 source/config/actor hash를 동결한 뒤 fresh
   confirmation 평가
5. 모든 승격 조건을 만족한 최종 후보만 예약된 blind/final 평가
6. 공식 제출 container와 동일한 조건의 최종 dry run

각 단계는 champion과 동일한 protocol, budget 및 집계법을 사용한다. 다른 protocol의 기록은
별도로 표시하며 직접 우열을 주장하지 않는다. 평균 하나만 보지 말고 track/seed별 결과,
분산, 실패 유형과 최악 조건을 확인한다. blind 결과를 본 뒤 같은 후보를 조정하거나 다시
선정하지 않는다. confirmation은 사전 선택된 한 후보의 가설을 확인하거나 기각할 뿐 후보 간
선택에 사용하지 않으며, 진단용 결과는 blind 실행이나 승격 권한을 새로 만들 수 없다.

트랙 일반화 검증은 공식 범위 안에서 직선, 급커브, 연속 커브, 좁은 구간, 장애물 배치 등
서로 다른 형태와 보지 않은 geometry seed를 포함한다. 특정 track ID, 화면 위치 또는 test
seed 암기에 의존하는 방법은 금지한다.

### 4.6 제출

현재 작업이 제출을 명시적으로 허용하고 모든 앞 단계가 통과한 경우에만
`COMPETITION_INFO.md`의 최신 절차로 제출한다. 제출 전에 다음을 수행한다.

1. 최종 source commit, model/package hash와 설정을 동결한다.
2. 공식 runtime에서 처음부터 재실행하고 시간·memory·크기·interface 제한을 확인한다.
3. quota, 마감, 파일명, metadata와 대상 track/phase를 재확인한다.
4. 검증한 동일 artifact를 수정 없이 제출한다.
5. 제출 시각, 명령/절차, hash, server 응답과 receipt를 `RESULTS.md`에 기록한다.

제출 권한, 인증, quota 또는 규칙이 불명확하면 추측하거나 반복 제출하지 않고 중단해 보고한다.

## 5. 자동 SOTA 비교 및 승격

눈대중 비교를 금지한다. 첫 후보 평가 전에 재실행 가능한 comparator를 마련하고, 평가 결과를
구조화된 artifact로 저장한다. comparator는 최소한 protocol/version, source와 model hash,
track/seed 목록, episode 수, 제외 항목, 지표와 준수 감사 결과를 검증해야 한다.

공식 순위 산식이 있으면 그 산식을 그대로 최우선으로 사용한다. 공식 순위 지표와 내부 안정성
승격 지표가 다르면 둘을 별도 필드로 계산하고 어느 하나로 다른 하나를 대체하지 않는다. 내부
반복 평가에서는 동일한 track/geometry/seed cell을 가능한 한 paired 비교하고 여러 training
seed에서 재현성을 확인한다. 같은 cell의 process reload나 반복 실행은 독립 표본 수를 늘리지
않는다. 별도 공식 산식이 없을 때는 현재 프로젝트 계약에 따라 다음 사전식 순서로 비교한다.

1. 모든 제한과 실행 게이트 통과 여부
2. fresh confirmation/unseen track의 유효 완주율
3. 미완주 episode를 포함한 progress
4. 완주 episode의 시간
5. 충돌·이탈·timeout과 운영 실패율
6. latency, memory와 package 크기

우연한 최고값 하나는 SOTA가 아니다. 사전 등록된 반복 모두에서 개선이 재현되고, 안정성
하락이 없으며, 승격 기준을 통과해야 한다. development 결과, 서로 다른 protocol, 진단용
실행 또는 blind를 열어 선택한 후보는 champion을 교체할 수 없다. 정확한 동률이면 기존
champion을 유지한다.

비교가 끝나면 comparator는 다음 다섯 상태 중 정확히 하나를 반환한다.

- `INVALID`: 규칙·실행·자원·누출 gate 실패. 결과를 성능 비교에 사용하지 않음
- `INCOMPARABLE`: protocol, 환경 또는 필수 증거 불일치. 탐색 결과로만 기록
- `INCONCLUSIVE`: 유효하지만 표본 부족, seed 간 충돌 등으로 사전 기준상 결론 불가
- `REJECT`: 비교 가능하지만 승격 기준 미달. champion 유지
- `PROMOTE`: 모든 gate와 사전 승격 기준을 통과한 엄격한 개선

`PROMOTE`일 때만 `SOTA.md`에 이전값, 새 값, delta, protocol, 불확실성, hash, 재현 명령과
날짜를 갱신한다. 나머지 상태는 champion을 유지하고 전체 결과와 이유를 `RESULTS.md`에
기록한다. `INCOMPARABLE` 또는 `INCONCLUSIVE` 결과로 우열을 주장하지 않는다.

`SOTA.md`는 최고 기록의 간결한 정본으로 유지한다. 실패와 중립 결과를 삭제하거나 숨기지
말고 `RESULTS.md`에 남겨 중복 탐색을 막는다.

## 6. 병렬 작업과 독립 검증

상당한 작업은 가능한 범위에서 서로 독립적인 subagent로 병렬화한다.

- 자료 담당: 공식 규칙·논문·baseline 최신성 조사
- 실험 담당: DB 분석, 가설·protocol 및 구현
- 준수 담당: `RESTRICTIONs.md` 기반 독립 감사와 package 재현
- 결과 담당: 통계 집계, comparator 실행과 SOTA 승격 검토

주 agent는 결과를 교차 검증해 하나의 결정으로 통합한다. 동일 정본 파일을 여러 agent가
동시에 수정하지 않으며, 감사 담당은 실험 담당의 주장만 복사하지 않고 원본 artifact와 명령을
직접 확인한다.

## 7. 기록, 보고 및 버전 관리

모든 결론은 source, config, seed, 환경, 명령, artifact hash와 원시 결과까지 추적 가능해야
한다. `CONTEXT.md`에는 현재 상태와 다음 행동만 간결하게 유지하고, 상세 결과는
`RESULTS.md`와 run artifact에 둔다. 제출 방법이나 과학적 주장이 바뀌면 `report.pdf`의 원본
문서와 인용도 함께 갱신하고, PDF를 다시 생성·검증한다.

의미 있는 구현, protocol, 실험 결과, 문서 갱신은 각각 검증 후 작은 단일 목적 commit으로
보존하고 현재 upstream에 push한다. 임시 출력, 비밀정보, 대형 학습 artifact와 무관한 사용자
변경은 commit하지 않는다. 최종 보고에는 변경 내용, 검증 범위, 결과, 남은 위험과 다음으로
가장 가치 있는 실험을 명시한다.

## 8. 즉시 중단 조건

다음 중 하나라도 발생하면 성능 실행 또는 제출을 중단하고 원인과 보존된 상태를 보고한다.

- 공식 규칙 또는 실격 조건이 불명확하거나 변경됨
- 필수 문서, baseline, source lineage, seed 분리가 없음
- 평가 누출, 결과 사후 선택 또는 비교 protocol 불일치가 발견됨
- runtime, 자원, package, interface 또는 금지 의존성 gate 실패
- 재현되지 않는 개선, 완주 안정성 저하 또는 예상 밖의 안전 회귀
- 활성 실험·동결 artifact·사용자 변경과 충돌

중단은 실패를 숨기는 수단이 아니다. 실행 가능한 artifact와 로그를 보존하고 `RESULTS.md`와
`CONTEXT.md`에 상태를 기록한 뒤, 가장 작은 검증 가능한 복구 단계부터 재개한다.
