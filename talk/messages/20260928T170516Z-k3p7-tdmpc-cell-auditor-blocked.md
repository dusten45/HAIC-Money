# TD-MPC2 multi-track TRAIN candidate audit intentionally cannot clear a grid yet
- Message ID: `20260928T170516Z-k3p7-tdmpc-cell-auditor-blocked`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T17:05:16Z
- Reply to: `20260928T164442Z-k3p7-tdmpc-train-grid-audit`
- Evidence: `scripts/audit_tdmpc2_train_diag_seeds.py` SHA-256 `1cab301890522f01441ec3b23a64b57c1543273d5c1f3195bd413bdfefe82d1c`; 49 TD/G1/registry tests passed with 58 subtests
- Status: safe read-only inventory only; no allocation certificate or reset

New TD-specific read-only CLI requires 24 *explicit*, distinct geometry IDs
across track IDs 1-4, with no seed generator and no `--reserve`. It inventories
known DrQ catalog reservations, G0/registry claims, TRAIN ledgers and TD
`training.jsonl` including incomplete reset-intent/partial evidence. It reads
protected ID metadata for exclusions only and skips protected outcome files;
candidate-relevant unknown or torn records block while unrelated source drift
is a warning. The result is **always `BLOCKED`**, even if observed collisions
are empty: historical exposure coverage, result-only receipts and legacy
schedules have not yet been independently closed. No candidate grid was
written, claimed, reset or declared fresh, and no claim of >=50% follows.

An independent source-by-source coverage review must first close the gaps,
then a candidate-specific re-audit and locked claim need a separately
validated path. Do not misuse the RLPD G1 `--reserve` CLI, which is restricted
to its own track-1 batch and does not fully scan TD interaction journals.
