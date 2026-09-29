# 출발 가속과 앞보기 커브 제어 결합

출발 가속 ZIP의 행동과 앞보기 조향·속도 제어를 결합하는 동일한 메커니즘의 순차 실험이다. 공식 사이트 업로드나 제출은 하지 않았다. 모든 주행 비교는 실제 결승선 통과를 먼저 판정하고, 기존에 사용한 2900–2903은 활성 진단으로만 해석한다.

## 수정안 1: 직선 추가 가속 포함

ZIP SHA-256 `5a41ff6a9ff4decfd9642726b3ec047951ceebfe92972122a12680a6062c5ab4`. 정확한 기존 `LaunchSurgeAgent` 위에 `AnticipatoryBendAgent`, `SpeedCoupledPreviewAgent`를 결합했다. 2900–2903 진단에서 출발 가속 120회는 유지됐고 앞보기 조향 622회, 커브 감속 92회, 직선 추가 가속 560회가 발생했다. 그러나 후보 10/12, 출발 가속만 적용한 control 12/12 완주다. 트랙 1 시드 2903과 트랙 3 시드 2901에서 후보만 미완주했다. 충돌 5 대 3, 손상 1.0 대 0.6이었다. 메커니즘은 작동했지만 완주 기준으로 REJECT다. 원본은 `artifacts/haic-research-v2/launch-preview-consumed-activation-2900-2903-20260929/report.json`과 동일 run ID의 V2 manifest/events에 있다.

## 수정안 2: 직선 추가 가속 제거

revision 2 설계 `docs/plans/active/launch-preview-composition-r2.md`는 원래 출발 가속 및 앞보기 조향을 유지하고, 이전과 같은 화면 커브·속도 조건의 제동만 남긴다. 새 ZIP SHA-256 `0c5fc1348c509ba9ac83fbd51d1641f46ceaa5bdcc39b0c4077c1950024ab795`.

기존 2900–2903 진단에서 후보와 출발 가속만 사용한 control이 모두 12/12 완주했다. 후보의 출발 가속은 120회 유지됐고 앞보기 조향 103회, 커브 제동 97회, 직선 추가 가속 0회였다. 충돌은 후보 1회, control 3회, 손상은 0.2 대 0.6이었다. 완주 중앙시간은 후보 23.88초, control 23.85초로 후보가 0.03초 느렸다. 원본 comparator의 기계적 decision은 이 작은 시간 열세 때문에 `REJECT`지만, 이 셀들은 독립 비교 자료가 아니므로 coordinator는 메커니즘 활성 진단으로만 받아들여 `REVISE`와 `competitive_or_product_outcome=UNKNOWN`으로 기록했다. 원본은 `artifacts/haic-research-v2/launch-preview-bend-only-diagnostic-2900-2903-20260929/report.json`에 있다.

새 tune 7012–7015에서는 양쪽 모두 11/12 완주했고 후보 중앙시간 27.32초, control 27.28초였다. 트랙 1 시드 7013은 양쪽 모두 같은 진행률 약 13%에서 멈췄다. 후보 앞보기 조향 107회와 커브 제동 121회, 직선 추가 가속 0회로 의도한 동작은 확인했다. 충돌·손상·무효 행동은 양쪽 모두 0이다. 시간 이득이 없어 `REJECT`이며 예약된 held-out 7016–7019와 confirmation 7020–7023은 열지 않았다. 원본은 `artifacts/haic-research-v2/launch-preview-bend-only-tune-7012-7015-20260929/report.json`에 있다.

## 수정안 3: 앞보기 조향만 결합

revision 3 설계 `docs/plans/active/launch-preview-composition-r3.md`는 원래 출발 가속과 앞보기 조향만 결합하고 추가 gas·brake 명령을 모두 제거한다. ZIP SHA-256은 `533c3a74f43313ace925998830d3c9b948a2185d4866e761025978f44c1542d7`이다. 소비된 2900–2903 진단에서 양쪽 모두 12/12 완주했고 후보의 출발 가속 120회, 앞보기 조향 105회, 추가 감속·가속 0회였다. 후보 중앙시간은 23.75초, control은 23.85초였다. 충돌 2 대 3, 손상 0.4 대 0.6, 무효 행동 0이다. 이 시간 차이는 이미 사용한 셀의 관측으로만 남겼다. 원본은 `artifacts/haic-research-v2/launch-preview-steer-only-diagnostic-2900-2903-20260929/report.json`이다. 새 tune 7024–7027을 별도 등록하고 실행했다.

새 tune 7024–7027에서 양쪽 모두 12/12 완주했다. 후보 완주 중앙값은 24.27초, control은 24.32초였으며 충돌·손상·무효 행동은 양쪽 0이다. 후보 앞보기 조향 106회, 출발 가속 120회로 메커니즘이 작동했다. 다만 트랙 1의 네 쌍 중 세 쌍은 후보가 0.28–0.36초 느렸고, 트랙 2·3에서는 대체로 빨랐다. 0.05초의 전체 중앙값 차이는 작아 일반화가 미확인이다. 원본은 `artifacts/haic-research-v2/launch-preview-steer-only-tune-7024-7027-20260929/report.json`이다. ZIP을 변경하지 않은 held-out 7028–7031 비교를 등록했다.

변경 없는 ZIP의 held-out 7028–7031에서는 후보 12/12, 출발 가속만 적용한 control 11/12 완주했다. 후보 완주 중앙값은 23.63초, control 완료 셀 중앙값은 23.70초다. 후보의 앞보기 조향은 114회, 출발 가속은 120회였으며 추가 가속·제동은 0회다. 충돌 3회와 손상 0.6은 양쪽 동일하다. 트랙 1 시드 7031에서 후보만 결승선을 통과했다. 반면 트랙 2 시드 7029에서는 후보가 4.52초 느렸으므로 안정적인 속도 우위로 단정할 수 없다. 원본은 `artifacts/haic-research-v2/launch-preview-steer-only-heldout-7028-7031-20260929/report.json`이다. ZIP을 변경하지 않은 confirmation 7032–7035를 별도 등록해 실행했다.

변경 없는 ZIP의 confirmation 7032–7035에서는 후보와 출발 가속 control 모두 12/12 완주했다. 후보 완주 중앙값 24.18초, control 24.38초, P90은 26.60초 대 26.80초였다. 후보 출발 가속 120회와 앞보기 조향 90회가 작동했고 추가 gas·brake는 0회였다. 충돌·손상·무효 행동은 양쪽 모두 0이었다. 후보 import·생성 최대 1.41초, reset 1ms 미만, act 최대 42.4ms, RSS 302MB 미만으로 로컬 제한 안이었다. 원본은 `artifacts/haic-research-v2/launch-preview-steer-only-confirmation-7032-7035-20260929/report.json`이다. 세 게이트 PASS로 로컬 workflow의 `RELEASE_IF_GATE_PASS`를 거쳐 STOPPED가 기록됐지만, 이 비교 profile은 SOTA 대상이 아니고 사이트 업로드·공식 확인·공식 제출은 하지 않았다. 공개 3위권의 약 12–18초 기록이나 사용자의 2배 속도 목표는 달성하지 못했다.

상대 비교 대상인 출발 가속만 쓴 ZIP이 과거 다른 셀에서 현재 선택 ZIP보다 완주가 적었으므로, 정확한 조향 결합 ZIP을 현재 선택된 full-road-guard ZIP과 새 7040–7043 셀에서 직접 비교했다. 후보의 첫 10% 도달 중앙시간은 2.96초, 선택 ZIP은 3.92초였고 첫 10% 평균 gas는 0.175 대 0.086이었다. 완주한 랩 중앙값도 후보 25.74초, 선택 ZIP 26.85초로 빨랐다. 그러나 후보 **11/12**, 선택 ZIP **12/12** 완주다. 후보만 실패한 트랙 3 시드 7040은 진행률 91.62%에서 `off_track`으로 종료했고 충돌 2회, 손상 0.4였다. 후보는 약 90% 진행부터 선택 ZIP과 다른 궤적·방향으로 진입했고 끝내 복귀하지 못했다. 출발 속도와 조향 변화 중 어느 쪽이 실패를 일으켰는지는 이 trace만으로 분리할 수 없다. 따라서 현재 선택 ZIP 교체는 `REJECT`, 예약 held-out 7044–7047 및 confirmation 7048–7051은 열지 않았다. 원본은 `artifacts/haic-research-v2/launch-preview-selected-tune-7040-7043-20260929/report.json`에 있다.

현재 결론: 초기 가속과 픽셀 앞보기 조향은 실제로 작동했고, 출발 가속 단독보다 안정적인 로컬 ZIP을 만들었다. 하지만 현재 선택 ZIP과의 새 직접 비교에서 완주율을 잃었으므로 현재 선택 ZIP을 유지한다. 2배 속도 또는 공개 상위권 도달은 검증되지 않았다. 다음 수정은 트랙 3/7040의 후반 도로 이탈을 픽셀에서 사전에 감지해 강가속의 이득을 유지하면서 속도를 줄이거나 조향 개입을 제한하는 원인 분리에서 시작해야 한다. 7040–7043은 이미 사용한 tune 셀이며 새 held-out으로 재사용하지 않는다.

평가 전용 행동 경계 진단 `launch-preview-7040-boundary-trace-20260929`은 같은 트랙 3/7040의 후보 `off_track` 91.62%와 선택 ZIP의 완주를 재현했다. 후보 앞보기 조향은 여섯 번 개입했고, 후반 진행률 89.5%에서 조향을 −0.03에서 −0.09로 바꿨다. 그 뒤 진행률 91.6% 부근에서 후보는 장애물 픽셀 신호가 보일 때 +0.08 방향으로, 선택 ZIP은 비슷한 위치에서 −0.15 방향으로 조향했다. 이는 후보가 다른 회피 경로로 갈라지는 시점을 좁혔지만, 선행 조향의 개입이 직접 원인인지는 증명하지 않는다. 원본은 `artifacts/haic-research-v2/launch-preview-7040-boundary-trace-20260929/report.json`이다. 다음 소비 셀 ablation은 정확한 launch-only ZIP을 같은 7040–7043에 다시 달려 트랙 3/7040 실패가 조향 없이도 발생하는지 본다.

`launch-only-consumed-ablation-7040-7043-20260929`에서 출발 가속만 적용한 ZIP은 트랙 3/7040을 26.62초에 완주했고, 앞보기 조향 결합 ZIP은 위와 같이 실패했다. 반대로 출발 가속만 적용한 ZIP은 트랙 3/7041에서 진행률 29%의 도로 이탈로 실패했지만, 조향 결합 ZIP은 완주했다. 현재 선택 ZIP은 두 셀 모두 완주한다. 이 ablation은 앞보기 조향 개입이 **셀에 따라 복구와 실패를 모두 만들 수 있음**을 보여 주지만, 화면 조건 중 무엇이 두 경우를 가르는지는 아직 모른다. 7040–7043은 소비 셀 진단일 뿐 새 완주 근거가 아니다. [ablation 전체 trace](../../artifacts/haic-research-v2/launch-only-consumed-ablation-7040-7043-20260929/report.json)를 보존했다.

복구된 3/7041의 평가용 [행동 경계 trace](../../artifacts/haic-research-v2/launch-preview-7041-rescue-boundary-trace-20260929/report.json)는 앞보기 조향 아홉 번을 기록했다. 시작 진행률 5.1–5.4%의 두 개입은 픽셀 속도 약 42.3이었고, 실패한 3/7040의 후반 개입은 진행률 89.5%·속도 47.3이었다. 도로 굴곡 크기는 겹쳐 굴곡 임계값만으로 구분할 근거가 약했다.

이에 픽셀 속도 45를 넘을 때만 앞보기 조향을 건너뛰는 revision 4 ZIP `8fcf21ad282b1cc441f4bbb6313707634c6285dba2cdeac4d1c9508933a05a32`를 별도로 고정했다. 소비된 7040–7043 진단에서 목표 셀 3/7040과 3/7041은 모두 완주했지만, 후보만 트랙 1/7040의 진행률 약 93%에서 충돌 5회 후 `crash`로 끝났다. 전체 후보 11/12, 선택 ZIP 12/12이므로 revision 4는 `REJECT`다. [원본](../../artifacts/haic-research-v2/launch-preview-speed-gate-diagnostic-7040-7043-20260929/report.json)을 보존하고 예약한 새 tune 7052–7055, held-out 7056–7059, confirmation 7060–7063은 열지 않았다. 이 comparator는 실제로 억제된 개입 횟수를 계측하지 않아 고속 게이트의 메커니즘 활성 판정은 UNKNOWN이다. 새 ZIP으로 모델을 바꾸거나 사이트에 올리지 않았다.
