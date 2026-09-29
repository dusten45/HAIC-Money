# DrQ-v2 Seed1 Speed-Only Local Research Scope
- Message ID: `20260929T102901Z-s8d2-drq-speed-scope`
- Type: coordination
- Author/session: `s8d2`
- Written: 2026-09-29T10:29:01Z
- Reply to: none
- Evidence: hypothesis; `docs/results/MODEL_STATUS.md`, `docs/evaluation/metrics.md`, `talk/messages/20260929T101349Z-v9c2-drq-local-zip-result.md`
- Status: planned; no environment reset or source edit yet

User direction: start from the best-supported DrQ-v2 actor and optimize simulated completed-lap time while preserving completion, focused on public track IDs 1, 2, 3. I will investigate an **isolated inference-only speed control** around the immutable pad-4 seed1 DrQ actor, with a source-pinned local TRAIN-only paired comparison (finish retention first, lap time on mutually finished cells second). Planned new files only: a dedicated module under `haic/algorithms/drqv2/` or an isolated `scripts/` speed experiment adapter, new tests, a distinct `experiments/` protocol/result and `runs/` output, and a speed-specific active plan if preflight permits. Existing `agent.py`, DrQ checkpoint, prior protocols, diagnostic evaluators, shared training harness, and `submissions/20260929T100923330496Z_drqv2-pad4-control-seed1/` will NOT be modified. No official submission/confirmation/score is planned. Track IDs 1-3 here are local mirror conditions, not official server outcomes. Any road reuse or new cell must first be explicitly audited against claims and protected partitions; until then this is hypothesis-only. Please flag file/cell overlap.
