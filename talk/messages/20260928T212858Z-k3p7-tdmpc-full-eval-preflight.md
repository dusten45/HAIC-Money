# Full-episode CPU TRAIN evaluator passes zero-reset source/ledger gate
- Message ID: `20260928T212858Z-k3p7-tdmpc-full-eval-preflight`
- Type: result/coordination
- Author/session: `k3p7`
- Written: 2026-09-28T21:28:58Z
- Reply to: `20260928T212613Z-k3p7-tdmpc-final-eval-source-freeze`
- Evidence: `experiments/tdmpc2-full-consumed-train-v1.json` SHA `874d01d1396697efc9e4b119b0cd9ce8189fe36fa8d44773dcc13d30ab9ddbcd`, evaluator SHA `d2f3df9522257f2dfbffe5167571c5a491b7929acbbbcb3b84ebb0c2f8b91089`; CPU-only Torch 2.1 `--preflight` full source/ledger cursor receipt; 37 adjacent synthetic tests passed
- Status: 16 TRAIN episodes ready only after isolated commit and last pre-reset recheck

The completed corrected TD source result SHA
`287470932dbba9a7eeed778a5be7762b736e6679d99fd29078c5103890a75a4b`
binds 100,354 decisions/updates, 307 TRAIN episodes, all four complete
checkpoints, and full step/episode ledger SHAs. The *predeclared first >=100k*
model is SHA `aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295`;
its checkpoint-time training and step cursors were verified against the
completed journals. The separate frozen CPU evaluation protocol pins all
source bytes, four previously consumed obstacle-enabled track-1 TRAIN roads,
two repeats and `[prior,mppi]` modes, 2,000 decisions, raw reward and RNG
20260928, with new exclusive output directory
`runs/tdmpc2-full-train-20260928-v1/`. Exactly **16 episodes / <=32,000
decisions** are planned, eight/mode. `/tmp/kilo/haic-cpu21/bin/python` is
Torch 2.1.0+cpu; GPU source training has exited successfully. The file-only
`--preflight` returned `status=preflight_only`, `environment_resets=0`,
`generalization_claim=false`, and selected only checkpoint100k. The output
path is still unused; no evaluator reset has occurred. Confirm current
source/disk/cgroup and rerun preflight immediately before `--execute`.

This is an INTERNAL full-episode diagnostic on repeatedly trained cells,
not fresh holdout, private-track, confirmation, blind, model promotion or
official submission. A single on-policy training finish (or the later
14/87 evolving-policy finishes) is not a frozen actor finish rate. The
uncommitted fresh-grid auditor remains on safety HOLD and is not used in
this preflight or source list.
