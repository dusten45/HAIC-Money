# HAIC 성능 개선 실행 규칙

이 파일은 성능 개선 작업의 단일 실행 계약이다. 사용자가 **`성능 개선해 줘`**를 입력하면 결과 DB, 논문, 대회 규칙, 로컬 실행 결과를 한 흐름으로 연결한다.

## 실행 순서

1. `artifacts/haic/`, `runs/`, `submissions/`의 allowlist JSON을 읽고 `RESULTS.md`를 갱신한다.
2. `research/papers.json`과 `report.pdf`의 참고 문헌·가설을 대조한다. 논문은 실험 가설의 근거이고, 성능 판정은 주행 기록으로 한다.
3. 같은 train/tune/held-out 분할에서 세 전략을 비교한다.
4. `COMPETITION_INFO.md`와 `RESTRICTIONS.md`를 확인한다. 계약·시간·메모리·ZIP·정적 검사·정보 누수를 통과하지 못하면 승격하지 않는다.
5. 실행은 명시된 로컬 명령만 수행한다. 기본은 계획 모드이며 `--execute --command`를 지정한 경우에만 학습·평가·패키징 명령을 실행한다.
6. 결과와 실행 로그를 `artifacts/haic/rules-runs/`에 저장하고 `SOTA.md`와 비교한다.
7. Tune에서 선택된 후보만 held-out 및 공식 장애물 진단으로 보낸다. 독립 평가가 엄격히 좋아진 경우에만 SOTA 포인터를 갱신한다.

## 계속 비교하는 세 전략

| ID | 전략 | 제출 runtime | 용도 |
|---|---|---|---|
| S1 `ppo_actor_only` | 시각 PPO actor-only | 허용·기본 | 제출 주체. train-only obstacle-risk reward와 learner-state DAgGER는 이 lane의 학습 변형으로만 기록한다. |
| S2 `ppo_cem` | PPO actor + latent dynamics CEM | 비교 전용 | 짧은 후보 행동을 dynamics로 평가한다. actor-only보다 좋아질 때만 후보로 유지한다. |
| S3 `vision_corridor_teacher` | pixel corridor 교사 | 제출 금지 | train-only teacher·진단 기준선. 제출 ZIP과 SOTA runtime에는 넣지 않는다. |

각 전략은 다음 지표를 같은 시나리오에서 남긴다.

- 완주율, 완주 주행의 중앙/P90 랩타임
- 미완주 평균 진행도, 충돌·손상
- `act()` p50/p95/max, invalid action, import/reset 시간, RSS
- 공식 Track1 seed42와 `official_plus_custom` 장애물 맵의 완주·진행도·충돌
- rollout, PPO update, 평가 시간과 checkpoint/ZIP 경로

비교 순서는 `완주율 → 중앙 완주 시간 → 평균 진행도 → P90 완주 시간 → 충돌 → 손상 → act p95`다. teacher 단독 기록은 진단 표에만 쓴다.

## 명령

```powershell
# DB 색인·논문/문서 감사·전략 비교·SOTA 기록(실행하지 않음)
python -m research_ops.cli improve "성능 개선해 줘"

# 지정한 로컬 학습/평가/패키징 명령 실행 후 동일 기록 저장
python -m research_ops.cli improve "성능 개선해 줘" --execute `
  --command "python training/train_policy.py --help" `
  --submission artifacts/haic/submission/haic-obstacle-risk-ppo-actor.zip
```

외부 대회 사이트 업로드는 자동화하지 않는다. ZIP 생성과 smoke 결과를 기록한 뒤 제출은 참가자 사이트에서 수행한다.

## 상태

- `candidate`: DB에 있으나 독립 평가가 끝나지 않음
- `sota-candidate`: Tune에서 가장 높지만 held-out/공식 진단 중
- `rejected-tune`: 동일 프로토콜에서 기준보다 낮음
- `diagnostic-only`: 제출 불가 진단 lane

실패한 실험도 원본 JSON과 함께 보존하며 유리한 시드만 남기지 않는다.
