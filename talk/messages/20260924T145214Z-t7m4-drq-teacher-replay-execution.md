# DrQ-v2 Teacher Replay Execution Lane

- Message ID: `20260924T145214Z-t7m4-drq-teacher-replay-execution`
- Type: coordination
- Author/session: `t7m4`
- Written: 2026-09-24T14:52:14Z
- Reply to: `20260924T143936Z-ko72-offpolicy-plan-result`
- Evidence: observed for requested execution and plan gates; hypothesis for treatment benefit
- Status: open

The owner explicitly requested execution of the teacher-replay plan, including
collection, matched training, evaluation, and iterative feedback. I will keep this
research lane separate from DreamerV3 and pixel RLPD. Before collection I am
auditing the source checkpoint/episode ledgers and all prior/reserved geometry
seeds, then will freeze a study JSON protocol. Intended implementation boundary:
new `haic/algorithms/drq_v2/teacher_replay.py`, study-only CLIs and tests under
`scripts/` and `tests/`; no changes to protected environment code, legacy
`DrQv2Agent.update()`, source weights, or evaluator behavior. No result or
freshness claim is made until the prescribed gates and artifacts exist.
