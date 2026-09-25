# Teacher Replay Source and Seed Audit

- Message ID: `20260924T150214Z-t7m4-source-seed-audit`
- Type: warning
- Author/session: `t7m4`
- Written: 2026-09-24T15:02:14Z
- Reply to: `20260924T145214Z-t7m4-drq-teacher-replay-execution`
- Evidence: measured for source artifacts and recorded cells; unknown for undocumented pilot use
- Status: open

Both intended pad-4 source checkpoints, actor exports, configs and `episodes.jsonl`
ledgers exist under `runs/20260922-drq-augmentation-pad-v1-restart/`. The
recorded checkpoint SHA-256 values are learner 0 `4248750c7114afdf955ea85a60c842565498aba3a64e380fa81164c7eb340979`
and learner 1 `c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4`;
actor hashes match the model ledger. Ledgers contain 367 and 321 reset records and
their content hashes are recorded in the run's `frozen_candidates.json`.

The prior 180-seed pad reservation alone is incomplete: recorded later geometry
includes Dreamer smoke seeds 0/1, Dreamer B1 development/partitions, residual-options
seeds `4000000001-4000000008` and prefix-branch seeds `4200000001-4200000004`, plus
the executed legacy blind seeds `20201-20208`. The old DrQ promotion pilot schedule
is incomplete, so any clean allocation can claim only “no known recorded overlap,”
not global non-use. Evidence is in `docs/evaluation/generalization-policy.md`, the
linked `experiments/*drq*.json` and `experiments/dreamerv3-b1-*.json` protocols,
`evaluations/20260922T145956295352Z_dreamerv3-smoke-screen/protocol.json`, and the
source run ledger manifests. No new cells have yet been frozen and no collection
has started.
