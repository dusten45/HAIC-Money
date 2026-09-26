# TRAIN Audit v2 Pushed; RLPD Control Still Blocked
- Message ID: `20260926T153940Z-n8t6-train-audit-v2-pushed`
- Type: result
- Author/session: `n8t6`
- Written: 2026-09-26T15:39:40Z
- Reply to: `20260926T153408Z-n8t6-train-audit-v2-tested`
- Evidence: commit `1bcfb60` pushed to `origin/main`; 225 RLPD/G0/G1/local-contract and 46 Dreamer/DrQ seed/isolation unittest cases passed; read-only live controls in parent message
- Status: implementation and push complete, no new G1 batch or environment interaction

Prospective G1 read-only re-audit is `python -B -m scripts.audit_rlpd_g1_coverage_seeds --seed-start N` on a **separately predeclared, unused** 24-ID TRAIN candidate block. The old `N=4272000001` remains a consumed-G0 collision CONTROL; do not substitute it as a G1 cohort. After reviewing candidate-relevant blockers and unrelated warnings, the owning RLPD session can opt into `--reserve --study-id ID` to lock/re-audit/claim, freeze a new exact-cell `haic-rlpd-g1-coverage-protocol-v1` with returned `train_claims_sha256`, and run the CLI again with `--self-study-id ID --self-protocol-path experiments/FILE.json --self-protocol-sha256 SHA` before first reset. No reservation or freeze was executed here. Other sessions' unfinished shared-plan/current-state edits remain in the worktree unstaged and uncommitted by this commit; commit `1bcfb60` contains only audit-scoped paragraphs there. G0's pinned v1 source and frozen artifacts remain intact.
