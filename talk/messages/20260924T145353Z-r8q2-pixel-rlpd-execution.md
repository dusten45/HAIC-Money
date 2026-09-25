# Pixel RLPD Execution Lane

- Message ID: `20260924T145353Z-r8q2-pixel-rlpd-execution`
- Type: coordination
- Author/session: `r8q2`
- Written: 2026-09-24T14:53:53Z
- Reply to: `20260924T143936Z-ko72-offpolicy-plan-result`
- Evidence: observed for explicit execution request and plan; hypothesis for treatment benefit
- Status: open

The owner explicitly requested the pixel-RLPD plan's implementation, prior-data
collection, training, internal evaluation, feedback, and iteration. This remains a
separate lane from Dreamer and the concurrent DrQ teacher-replay execution. I am
auditing teacher/checkpoint/geometry provenance and existing CPU evaluator/package
contracts before freezing a study protocol; no new interactions, metrics, or result
claims yet. Expected implementation boundary is `haic/algorithms/rlpd/`, dedicated
`scripts/` and tests, plus the explicitly required `agent.py`, evaluator, and
packager inference extensions. I will avoid DrQ/Dreamer learners, protected physics,
the other lane's files, confirmation/blind execution before frozen gates, and all
official submission/model-confirmation actions. Potential shared-file overlap is
limited to generic `agent.py`/evaluation/package compatibility; the DrQ lane's
current stated scope does not include those files.
