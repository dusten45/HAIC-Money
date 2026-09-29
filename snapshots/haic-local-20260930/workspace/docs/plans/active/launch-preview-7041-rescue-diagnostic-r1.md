# 트랙 3/7041 앞보기 조향 복구 조건 진단

정확한 launch-only ZIP은 이미 소비된 3/7041에서 진행률 29% 후 도로를 이탈했지만, launch+preview ZIP은 완주했다. 반면 3/7040에서는 launch-only가 완주하고 preview 결합이 91.62% 후 이탈했다. 원본은 `artifacts/haic-research-v2/launch-only-consumed-ablation-7040-7043-20260929/report.json`, `artifacts/haic-research-v2/launch-preview-selected-tune-7040-7043-20260929/report.json`이다.

정확한 preview 결합 ZIP `533c3a74f43313ace925998830d3c9b948a2185d4866e761025978f44c1542d7`과 선택 ZIP `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40`을 이미 소비된 3/7041 **한 셀**에서 평가용 wrapper 계측으로 재주행한다. 앞보기 개입 순간의 현재/직전 도로 중심, 픽셀 속도, 장애물, 기본·최종 조향을 3/7040의 동일 계측과 비교한다. 두 ZIP과 환경·행동 코드는 변경하지 않는다. 2 CPU/2 GiB, 각 팔 한 셀, 최대 2,000 decision, 600초. 이 결과는 원인 진단이며 새로운 완료율 표본이 아니다.

복구 셀과 실패 셀의 개입 조건에 분명한 구분이 없으면 단일 픽셀 임계값으로 안전한 개입을 보장한다고 주장하지 않는다. 구분이 보이더라도 새 후보는 별도 계획과 새 tune/held-out/confirmation에서 검증한다. 정확한 plan hash에 설계·실행 이벤트를 나눠 기록하며 2026-09-29 사용자 상시 로컬 승인을 인용한다. 사이트 조치는 없다.
