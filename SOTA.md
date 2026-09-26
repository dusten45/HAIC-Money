# SOTA

> Migration note (2026-09-26): The entry below is a preserved historical **local** PPO reference. It has not been revalidated or promoted by the v2 completion-first, rule-compliance, and mechanism-activation gates. A new candidate replaces it only after a registered matched independent evaluation passes those gates. Official leaderboard standing is separate.

이전 로컬 DB에서 held-out/official 근거와 당시 제한 검사를 통과한 것으로 기록된 PPO actor-only 결과다.
Tune은 후보 선택에만 사용하고, corridor teacher 기록은 이 포인터에 올리지 않는다.

- record: `8b33233e373d`
- strategy: `ppo_actor_only`
- split: `held_out`
- completion: `0.75`
- mean progress: `0.7880441754422792`
- median lap ms: `19320.0`
- p90 lap ms: `22420.0`
- source: `artifacts/haic/final-ppo-actor-selection.json`
- checkpoint: `artifacts\haic\site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101\policy.pt`
- checkpoint SHA-256: `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`
- submission archive: `artifacts\haic\submission\haic-obstacle-risk-ppo-actor.zip`
