# 최종 확인 요약 — 2026-09-29

out-in-out 조향 목표와 가속을 구현하고 5차례 수정, 총 126회 비교 주행을 실행했다. 같은 소비 TRAIN 6조건을 반복 사용했으며 새로운 일반화 성능으로 해석하지 않는다.

| 버전 | 기존 완주 | 주행선 후보 완주 | 가속 후보 완주 |
|---|---:|---:|---:|
| 첫 주행선/가속 | 6/6 | 3/6 | 4/6 |
| 출구 전환 수정 | 6/6 | 4/6 | 3/6 |
| 회피 방향 유지 | 6/6 | 5/6 | 4/6 |
| 장애물 이동 예측 | 6/6 | 5/6 | 2/6 |
| 화면 밖 예측 제외·직선 추가 가속 | 6/6 | 4/6 | 4/6 |

최신 가속 후보의 공통 완주 4조건은 기존보다 각각 0.42, 0.30, 0.20, 0.06초 빨랐다(차이 중앙값 0.25초). 그러나 트랙 2/시드 38302에서 95.54%, 트랙 3/시드 38300에서 71.69% 진행 후 이탈했다. 완주 손실이 있어 개선 후보로 채택하지 않았다. 기존 제출 코드와 ZIP은 교체하지 않았고 사이트 업로드도 하지 않았다.

코드는 바깥 진입→안쪽 꼭짓점→바깥 탈출 목표를 생성한다. 실제 차량 위치를 별도로 측정했을 때 이 순서를 일관되게 실현하지 못했다. 따라서 **실제 out-in-out 주행과 안정적인 속도 개선은 아직 미달성**이다. 현재 중앙 추종 조향에 좌우 보정만 더하는 구조의 한계가 드러났다. 후속 구조에서는 화면에 보이는 도로 경계 안에서 미래 경로를 먼저 만들고, 그 경로의 방향과 곡률을 따라가도록 해야 한다. 장애물 대응도 그 경로와 함께 결정해야 한다.

모든 126개 원본 파일 해시와 첫 10회 행동 일치를 확인했다. 잘못된 행동 출력은 0회, 최대 행동 계산 시간 68.11ms, 최대 RSS 362,725,376바이트였다. ZIP cold import 검증 및 공식 규칙 인증은 이번 비교에 포함되지 않는다. 최신 상태는 PIVOT → STOPPED, release=false. 보존된 기준 체크포인트: `artifacts/haic-research-v2/fast-completion-row-repair-20260929/submission-fast-completion.zip`.

현재 연구 구현: `haic_agent/out_in_out_runtime.py`, `haic_agent/out_in_out_flow_runtime.py`, `haic_agent/out_in_out_visibility_runtime.py`. 실험 전용이며 최종 제출 Agent로 연결하지 않았다. 알려진 진단 카운터 및 드문 제동 복원 분기 불일치는 아래 리뷰 기록에 남긴다. 후속 재설계 전에 해결해야 한다.

근거: `artifacts/haic-research-v2/out-in-out-visibility-20260929/batch_audit.json`, 각 실행의 `report.json`, `runs/haic-research-v2/out-in-out-visibility-20260929/integration_report.json`.

---

# Out-in-out steering and speed — 2026-09-29

User requests outside-entry/inside-apex/outside-exit and higher speed, retaining completion objective. Local standing authorization applies; no external upload or official submission.

## Completed first probe

Registered run `out-in-out-fast-probe-20260929`, six consumed TRAIN cells (tracks1–3 ×38300,38302),36 full episodes. Control is preview_row_repair. Exact first10-action/pose prefixes match. No invalid actions or execution errors.

| Mechanism | Finished | Median finished time |
|---|---:|---:|
| Control |6/6|21.59s|
| Out-in-out line |3/6|22.04s|
| Out-in-out plus fast pedals |4/6|20.48s|
| Unwind |4/6|21.80s|
| Curvature speed |3/6|20.02s|
| Exit boost |5/6|20.66s|

Medians use different finished subsets and are not evidence of matched improvement. On1:38300 line_fast20.82s vscontrol22.10s, bothfinished andcollisionfree; this one cell is not generalization. Two line_fast failures1:38302 and2:38302. Do not promote.

Review found apex persists through observed straightroad25–30untiltimeout37 on1:38300. Applied smoothed offset could exceed newly narrowed width; unequal-spacing fallback could misread heading as curvature (not observed in inspected trace). These defects are addressed by separately registered phase repair. Physical racing-line activation not established by phase labels or near-field pixel offset.

## Phase correctness repair

Run `out-in-out-phase-repair-20260929`, same36episode design. Exit on2consecutive existing straight-road cues, clamp actual offset to width, use equal-spacing fallback. Add evaluator-only pre-action signed track lateral distance, never used as policy input. Results pending.

## Earlier completion ZIP interruption

The previous row-repair ZIP comparison stopped when Docker Desktop exited, at21of24episodes; all21completed, including the2targeted failures. Recorded `EXECUTION_FAILED` and integratedREVISE/STOPPED. No24/24claim. Frozen ZIP and raw episodes remain in `artifacts/haic-research-v2/fast-completion-row-repair-20260929/`. Current firstprobe baseline6/6 is a new matched check on its registered6cells.

## Completed phase repair

36/36episodes executed successfully, all first10prefixes match, invalidactions0, maxact34.49ms, peakRSS362725376bytes. Control6/6;line_only4/6;line_fast3/6;unwind4/6;radius_speed3/6;exit_boost3/6. Both line variants lose1:38302and2:38302;line_fast additionally loses3:38300. Paired lap deltas among line_only finishes+380,+420,+420,+320ms (slower);line_fast−960,−1240,−1320ms but loses half the finishes. IntegratedREVISE/STOPPED,notreleased.

Actual pre-action signed road distance does not consistently follow outside→inside→outside within each inferred bend. Exit can be one decision before phase resets; phase intent is not proof of achieved racing line. Large off-road association distances cannot establish correctly identified bend sides. Two-line-only avoidance contributions flip−.336→+.340beforecontact;line_fast−.193→+.312. Control also flips, so causality needs separate comparison.

## Avoidance continuation

`out-in-out-avoidance-20260929` records third iteration. Continue two line variants from the four-direction batch,3arms×6cells18episodes. Camera-relative commitment isolates avoidance in line_only; line_fast adds heading-aware baseline target+6 while retaining original visible-obstacle target. Results pending. No automatic release.

## Completed avoidance continuation

18/18episodes completed execution;first10prefixes match;invalid0,maxact30.404ms,peakRSS328699904bytes. Control6/6,line_only5/6,line_fast4/6. Line_only finishes are300–520ms slower;fast's four common finishes are620–800ms faster. Fast fails2:38302at37.58% collision and3:38302at49.68% offtrack. Line_only fails2:38302at95.54% offtrack. Keep completed/incomplete denominators separate. IntegratedREVISE/STOPPED,notreleased.

Read-only review found impact-clearing replacement suppression lacks a complete-road-row predicate (parent clearing requires42and54). No such missing-row impact case was observed in inspected2:38302. `committed_obstacle_term` is proposed, not guaranteed applied; interpret `avoidance_replaced` and final action. No claim of full physical racing-line activation.

## Optical-motion continuation

`out-in-out-flow-20260929` subclasses the preservedr3 controller. At first sight use road-relative side; at next positive vertical image displacement predict objectx atvehicle row63 and commit opposite side. Motion forecast remains an unvalidated hypothesis pending18episodes. Runtime consumes only images and own state. No release or official submission.

## Completed optical-motion comparison

18/18episodes executed;control6/6,line_only5/6,line_fast2/6. Line_only now completes2:38302but loses3:38300. The lost3:38300episode andfast2:38300 contain invalid offscreen forecasts clippedto83 then treated as confident side decisions. Infast2:38300dx4,dy0.8projects beyondtheimage. Other risks: detector switches objectidentity and perspective is not calibrated collision geometry. IntegratedREVISE/STOPPED,no release.

## Bounded prediction and clear-road acceleration

`out-in-out-visibility-20260929` preserves previous controllers. Rejectrawprojectedxoutside0..83bykeepinginitialcamera-side fallback. Fastincrement+6onlywhenexistingstraightcueholds;otherwisebaselinepedalsandactualbrakehistoryare restored.18episodespending. Physicalracinglineandperformancegatesremainseparate.

## Review limitations for bounded prediction

`completion_changed` is inherited before final pedal restoration, so it can overcount interventions; interpret returned pedals and matched traces. `extra_acceleration` means eligibility, not effective pedal difference. Conditional baseline restoration mismatch: when target60coincides with incomplete zero-spread geometry, restoredbrakecap.28differsfromoriginalfixed60cap.15. No such >.15case in the first three completed fastcells; full audit pending. Parent clearing eligibility and detector objectidentity remain unresolved. These controllers are research candidates, not replacements for the proven6/6control on this set.

## Architectural finding

Adding signed offsets to an existing centerline-following steering law does not reliably produce physical outside-entry/inside-apex/outside-exit. Turning phases can advance faster than vehicle lateral response. All measured phase intent must be distinguished from physical trajectory. Subsequent work should explicitly construct a feasible future path from visible road boundaries and track its heading/curvature, while maintaining obstacle identity. Preserve this batch and baseline checkpoint rather than further coefficient sweeps or claiming out-in-out success from names.
