# TD-MPC2 100k first attempt stopped before 20k checkpoint
- Message ID: `20260928T161418Z-k3p7-tdmpc-long-v1-partial-failure`
- Type: failure
- Author/session: `k3p7`
- Written: 2026-09-28T16:14:18Z
- Reply to: `20260928T154254Z-k3p7-tdmpc-long-run-freeze`
- Evidence: `runs/tdmpc2-long-20260928-v1/training.jsonl` SHA-256 `b80fa66d48174b31c26f9d5a17438d0830e1423eb527cf7432d93f35929f3ae1`, `steps.jsonl` SHA-256 `2a5eae82c089894110fba47c71a72d6736d65fe9f35da56bdd4b45cbbc5eb429`; source/protocol pinned in earlier freeze
- Status: fail-closed partial; diagnosis in progress

The first separate 100k TRAIN run is NOT ongoing. Its immutable ledger ends
with `event=partial,reason=ValueError`: 10,020 decisions and 10,020 optimizer
updates after exactly 10,000 random seed decisions/10,000 pretraining updates;
28 completed TRAIN episodes, 279 decisions in its unfinished episode. Wall
time was 1,144.63s, process CPU 1,141.01s, peak RSS 1,768.47 MiB and peak
CUDA allocated 251.98 MiB. Process stderr ended `ValueError: TD-MPC2 action
must be in [-1,1]`. No 20k/40k/70k/100k checkpoint or learner outcome was
produced, and there is no exact resume. The scope is only the four consumed
TRAIN road cells; confirmation/blind/official cells were not touched. Root
cause (environment action versus newly added same-state MPPI mean diagnostic)
is under independent source/ledger audit; neither candidate is yet confirmed.

Preserve original protocol, run dir, partial ledgers and source hashes.
An isolated, newly source-pinned correction with a deterministic regression
test is required before ANY new run/reset. Do not relabel this as failure of
TD-MPC2 learning or reuse the partial checkpoint that does not exist. The
user has explicitly requested continued iterative TRAIN research, not a
model confirmation or official submission.
