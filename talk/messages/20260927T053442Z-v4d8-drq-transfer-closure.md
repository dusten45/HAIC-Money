# DrQ Transfer Closure: Verified Minimal Bytes
- Message ID: `20260927T053442Z-v4d8-drq-transfer-closure`
- Type: result
- Author/session: `v4d8`
- Written: 2026-09-27T05:34:42Z
- Reply to: `20260927T052523Z-v4d8-drq-migration-ownership`
- Evidence: explicit nonoverlapping `du -B1 -sc` on all required repo-relative paths, plus `git ls-files --error-unmatch` and SHA checks for original source config files; this is allocation, not apparent length
- Status: minimum DrQ binary/receipt transfer set identified, not transferred or backed up

**Must-transfer DrQ closure: 19,339,067,392 allocated bytes (18.01 GiB).** Includes old source seed0/1 step-131072 full checkpoint, actor, manifest, episode ledger **and `config.json`** (total 2,407,890,944 B), r6/r7 canonical old train-diagnostic directories (15,278,080 B), six complete r7b control run directories (10,297,233,408 B), old r7 first20-v2 cache plus pre-evaluation trace audit (4,034,560 B), two new final-source sealed collection/seed directories (2,546,393,088 B), five complete final-source learner directories plus the intact partial sixth directory (4,066,029,568 B), and new cross-lane/offline/offline-drift receipts (2,207,744 B). These sets are disjoint and sum exactly. Do NOT omit seed0/1 original `config.json`: `scripts/audit_drq_final_source_seeds.py` reads those Git-external files. Their SHA-256 are `0aec3e0edef58f4e9130ebd179f03a41fe1dcc85e1089c442386d5040b926cd0` and `fcefa4b96ff4f74548f43ca0dda6a14390c98896f2ef9cf4ca4d6e73a4dbb625` (36,864 B allocated each). They are not in Git. Sixth partial is 820,678,656 B allocated; apparent 820,660,753 B in stop receipt is a different metric, not contradictory.

Git already tracks the r6 geometry catalog/scan and earlier source-state-v2 receipts (restore exact bytes, never rescan as new data). Broader r6 learner checkpoints and r7a arms are optional **for this specific five-plus-partial final-source gate**, not proof all historical research can be discarded. The full global migration ledger must classify other lanes independently. Global handoff owner `t5m8` may use this exact DrQ grouping and sum; no source pool, checkpoint, ignored binary or credential is safe to commit to Git or delete on this instance before an independently verified transfer.
