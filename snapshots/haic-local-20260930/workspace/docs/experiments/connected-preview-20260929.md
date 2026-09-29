# 화면 연결 경로와 선행 조향 — 14:25 UTC heartbeat

기준은 SHA7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835 안정화 ZIP이며 변경하지 않았다. 전체 목표(완주 유지와 랩타임 절반)는 미달이다. 최신 사용자의 앞보기 지적을 따라 페달 상수 조정에서 관측·조향 구조로 우선순위를 옮겼다. 공식 환경과 관측 입력을 보존하고 사이트 작업은 하지 않았다.

## 첫 비교: connected-preview-20260929

화면의 도로 색 범위에 연결요소와 거리장을 적용하고, 차량 앞 연결 도로를 유한 beam으로2차원 추적했다. 도로가 옆으로 이어질 수 있다. 네 방향은 원거리 pursuit, 근거리+원거리 보완, optical-flow 경로기억, 조향과 가속의 경로공유다. 모두 픽셀과 과거 행동만 입력으로 사용한다. 현재 보이지 않는 전체 트랙 지도를 안다고 가정하지 않는다.

tracks1–3 × seeds38300,38302의 소비TRAIN6조건, 5종×6=30회:

| 후보 | 완주 | 완주한 주행 중앙 시간 |
|---|---:|---:|
| 기존 control | 6/6 | 21.59초 |
| connected_pursuit | 1/6 | 21.58초 |
| dual_preview | 5/6 | 21.66초 |
| motion_memory | 1/6 | 21.58초 |
| shared_path | 3/6 | 20.64초 |

부분완주 시간만으로 향상이라 판정하지 않는다. 원거리 경로가 조향에 쓰인 시점의 직선도착시간 중앙값은약0.32초였지만 신뢰할 수 있는 코너 선행 검출이 개선됐다는 증거는 별도로 필요하다. shared_path최고76.08은 완주 손실을 보상하지 못한다.

실패 진단: track2/38300 pursuit step41에서 우측목표(9.32,14.56)지만 최종steer-.05, step42우측목표(11.99,12.96)지만steer-.01이었다. 장애물회피 항과 약한 원거리 조향이 상쇄되고 횡이탈이 커지는 현상이 관측됐다. 가까운 위치 보정을 대체하는 방식이 위험하다는 근거이며 모든 실패의 단일원인이라고 확정하지 않는다. dual은2/38302에서step120충돌후총5충돌로미완주했다.

motion_memory는1/38300에서88번 유효변환을 얻었지만 경로 재사용은전체0회였다. 현재추적이8점이상 반환하면 기억을쓰지않는조건이 활성화를 막았다. 광류 계산이나 코드 존재만으로 지도 기억이 작동했다고 주장하지 않는다.

30raw해시와첫10행동 동일,control6개 전체행동/궤적이 직전 기준과일치,invalid/error0. 실행332.40초,2CPU/2GiB,최대RSS318844928bytes, 후보p95act약20.5–22.6ms. 정확한 ZIP검증없음. gates=ruleUNKNOWN,mechanismUNKNOWN(메모리미활성),outcomeFAIL. PIVOT→gate review→STOPPED,releasedfalse. 이전pedalR2/R3비개선이력을보존한세번째유효비개선이며기준ZIP유지.

planhash43aeff2dd647a9244d67e1848720c55961ac8ece2c07bbe238f7806a2ed0f5ee. 원자료와batch_audit는artifacts/haic-research-v2/connected-preview-20260929, 불변manifest/events/integration은동일run_id의runs/haic-research-v2에있다.

## 보완 비교 설계

connected_guarded_preview_runtime.py는 원래 안정화 행동 주변±.12의 원거리 보완만 허용하며, 장애물이 보일 때와 사라진뒤3프레임에는 보완하지 않는다. pursuit/memory/shared는차량앞정렬도확인한다. optical-flow기억은현재경로와4pixel내대응하는점이70%이상이면25%혼합해 실제경로에반영한다. 이는원거리정보를없애는것이아니라가까운제어와충돌하지않게권한을구분하는가설이다.

초기 connected-guarded-20260929는Windows문서읽기encoding오류후메타데이터가예정대로갱신되지않은상태로등록되었다. 승인/실행하지않고보존했으며 connected-guarded-budget-20260929로대체했다. 실행이없는등록을실패주행/비개선주기로세지않는다.

보완비교는남은20분회차예산에맞춰(1,38300),(1,38302),(2,38300),(3,38300)의4조건을미리등록했다.5종×4=20회,프로파일timeout240초,2CPU/2GiB. 6조건전체와같은분모로비교하거나이4조건만으로승격할수없다. 모든조건은소비TRAIN이다. planhash9fd88d57b0a0bb6f2a2499b26d431973cada60e0e13c2cdcfae365c1aab641f8. 정확한별도설계/실행승인이standingauthorization을인용한다.

정적검토에서 impact-clear의관측행조건누락과장애물hold의한프레임오차를각각등록전수정했다. 공개기준프로파일fixed60_completion timeout은회차예산집행을위해현재240초로줄어있으므로다음회차는등록전에필요한예산과일치시켜야한다. 현재실행중인source/config를바꾸지않는다.


## 보완 비교 완료

{"control": {"completed": 4, "denominator": 4, "median_finished_ms": 21680.0, "interventions": 0}, "connected_pursuit": {"completed": 3, "denominator": 4, "median_finished_ms": 21940, "interventions": 704}, "dual_preview": {"completed": 4, "denominator": 4, "median_finished_ms": 21470.0, "interventions": 505}, "motion_memory": {"completed": 3, "denominator": 4, "median_finished_ms": 21920, "interventions": 667}, "shared_path": {"completed": 2, "denominator": 4, "median_finished_ms": 20860.0, "interventions": 659}}

실행 시간 210.58220169999913초.20raw해시일치,invalid/error0,control4개exactreplay및first10동일. 메모리 활성 통계는batch_audit.json. 설계의21.59초는이전6조건값이남아있는것으로,이번4조건시간비교에는사용하지않는다. 각후보는동일4조건control과만비교한다. 전체6/12조건과정확한ZIP재현전개선/승격을주장하지않는다. 원래추천ZIP유지. 초기connected-guarded계획은미승인·미실행으로대체되었으므로실행하지않는다. 다음회차는이partialscreen을full6으로확대하기전에실패원인과후보별지표를확인한다.

최종R2결정: REVISE/STOPPED,released=false; gates UNKNOWN/PASS/UNKNOWN. dual4/4 중앙21.47초 vs matchedcontrol21.68초는부분screen 신호다. R1실패조건2:38302가포함되지않아그실패가고쳐졌다고말할수없다. 다음회차는해당조건과3:38302를포함한전체6조건을먼저확인하고그뒤12조건/정확한ZIP으로확대한다. 원본ZIP유지;새패키지/업로드없음.


## 2026-09-30 full6 and additional6 continuation

Full6 `connected-guarded-full6-20260930`, plan `3904a9c7a8784977db218737f2c21caf941e4567f206589b18a638d233b34869`: 30 episodes/301.49 seconds, 2CPU2GiB. control6/6 median21.59s; dual_preview6/6 median21.30s, collisions0 vs1; pursuit5/6, memory4/6, shared4/6. All raw hashes valid, invalid/error0, all six control action/trajectory/progress exact replays, first10 equal. Source unchanged from previous partial4. Those repeated4 are not additional independent cells. Gate UNKNOWN/PASS/UNKNOWN; REVISE/STOPPED, no release or ZIP.

Read-only reviewer review_fixed_speed found omitted formal predecessor fields in full6 registration. An append-only CORRECTION references the original DISCOVER event and prior budget run. Immutable manifest preserved; annotation is not retroactive preregistration repair. Additional6 uses explicit --previous-run with predecessor run/decision IDs. Reviewer also notes memory fusion plus authority does not establish memory-specific action contribution, and source-file package hash is not a ZIP hash. Accepted; no memory-specific improvement or ZIP-compliance claims.

Additional6 `connected-guarded-extra6-20260930`, plan `71bf243b8a472fc79a4732cb58ee058fcbe8c4880968fc740c479fa7fe4cf3a5`: same frozen five arms, tracks1-3 x38301/38303, 30 episodes,600-second bound. These cells were previously consumed TRAIN, not heldout. Separate standing-authorization design/execution events. Full6+additional6 aggregate twelve unique cells only. Additional6 is currently running; source/config/settings frozen until integration.


Full6 failure trace observations (diagnostic only, not causal proof): pursuit1:38302 first collision step245, large lateral departure step260; preview authority already false near departure. memory2:38302 departs without collision: step254 normalized lateral -0.85 at39.6speed, step255 -1.16 with missing near center, step256 -1.60; authority false. shared1:38300 first collision step246 drops45.2 to12.5; shared1:38302 step229 drops44.3 to1.5; authority false near both. Disabling preview does not reset the altered physical state. Hypothesis for next design: controller handoff continuity / heading-aware recovery, rather than simply more aggressive residual clipping. These traces do not isolate a unique cause or prove memory-specific harm.


## 2026-09-30 additional6 completed / twelve unique TRAIN cells

Both runs finished, no process pending. Additional6 264.43 seconds; total registered rollout time565.93 seconds, each2CPU2GiB, no timeouts. Thirty additional raw hashes valid, invalid/error0, first10 equal. Six additional control full action/trajectory/progress replays exactly match prior clearance-cone controls (aggregate_audit.json). Source actor remained unchanged for all60episodes. No new package or external action.

Aggregate unique12 completion: control12/12; pursuit11/12; dual12/12; memory10/12; shared10/12. Control median20.52s versus dual20.65s, therefore registered completion-first/median ranking does NOT improve. Dual is faster on10matched cells and slower on2, but count of wins and within-half medians do not override aggregate median. Control collisions1/damage.2; dual collisions3/damage.6. Aggregate linear-interpolated P90 control21.712s,dual21.646s; secondary metric cannot override median. All statistics are consumed TRAIN, not generalization evidence.

Dual1:38303 takes20.38s versus20.10s. Contact flags atsteps98,99,100 occur while preview_authorized=false and obstacle hold3. This is three contact records, not proof of three separate impacts. Already-altered vehicle state persists when preview hands back to baseline. Correlation motivates examining pre-obstacle trajectory and handoff continuity, not claiming sole cause. Previous failed2:38302 nowfinishes21.34s versuscontrol21.54s.

Additional6 outcome REVISE/STOPPED,released=false; gates UNKNOWN/PASS/FAIL. Full6 remains REVISE/STOPPED UNKNOWN/PASS/UNKNOWN; additional comparison closes its previously partial outcome evidence without rewriting old records. Combined evidence is one guarded-mechanism twelve-cell cycle, not independent full6 and extra6 optimization failures. No SOTA change; original7bc38a3...ZIP retained.

Next cycle: preserve dual source checkpoint9903d6eee51351af1421c4dfc60bc702fc12a42f8318c638d8c715af50641240 and stableZIP. Review pre-contact trajectory1:38303 and previous1:38302 failures; design four independent mechanisms around anticipatory obstacle/path coupling, handoff continuity, heading-aware recovery, and observed curvature/speed planning. Do not simply sweep residual weights. Full12 matched evaluation before exactZIP/new heldout; formal predecessor run/decision linkage required. No source modifications made during this heartbeat.
