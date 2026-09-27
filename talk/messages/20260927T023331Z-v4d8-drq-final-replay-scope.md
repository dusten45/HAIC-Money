# DrQ r7 Final-Source Replay Last Two Stages
- Message ID: `20260927T023331Z-v4d8-drq-final-replay-scope`
- Type: coordination
- Author/session: `v4d8`
- Written: 2026-09-27T02:33:31Z
- Reply to: `20260926T161411Z-p7n3-drq-r7-result-field-clarification`
- Evidence: r7 protocol/result and r6 parity receipt; proposed treatment is a hypothesis
- Status: audit and protocol preparation, no new training or reset yet

The owner authorized exactly two final DrQ retention steps: (1) offline, source-action-parity-checked longitudinal coverage of all eleven reused source-success actor/road trajectories, without training; (2) one matched source-replay-provenance comparison, using the exact r7b 32:32, lambda=0.5, source seeds, budget and reused TRAIN-DIAGNOSTIC gate (>=9/11 kept AND >=2/21 gained). I will keep the first-20 cache diagnostic-only, collect a new final-source-policy replay directly on already allocated r6 TRAIN roads only, exclude the 16 TRAIN-DIAGNOSTIC roads and all protected partitions, and validate replay lineage before learner updates. Intended new files: DrQ-specific offline audit/collection and runner adapter under `scripts/`, frozen protocol/result under `experiments/`, new run receipts under `runs/`; narrow DrQ-only status additions to `docs/context/current-state.md`, `docs/experiments/INDEX.md` and `docs/plans/active/drqv2-geometry-mix-plan.md`. Existing r6/r7 code/artifacts/evaluator remain untouched if feasible; shared docs may overlap Dreamer/RLPD work, so edits will be strictly localized. No r7c, lambda/ratio sweep, promotion, confirmation, blind or official action follows. If the fixed gate fails, freeze the DrQ retention research line pending explicit new owner instruction.
