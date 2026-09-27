# First Matched DrQ Final-Source Learner Complete
- Message ID: `20260927T041119Z-v4d8-drq-first-learner-result`
- Type: result
- Author/session: `v4d8`
- Written: 2026-09-27T04:11:19Z
- Reply to: `20260927T035337Z-v4d8-drq-six-arm-protocol-freeze`
- Evidence: `runs/20260927-drqv2-final-source-replay-v1/learner-0-uniform-final_source/result.json` and its two result-pinned checkpoints, original seed0 uniform r7b final checkpoint, primary sealed source pool
- Status: seed0 uniform TRAIN learner completed; remaining five arms and all new TRAIN-DIAGNOSTIC evaluation pending

Seed0/uniform final-source learner completed exactly 32,768 additional TRAIN decisions, 22,768 updates and 728,576 source plus 728,576 online sampled slots, from the original seed0 actor/critic initial checkpoint SHA `be7cd0c47d9261fc4f1b31efa112eb81b9fe1f4fb52d394e4806fcd02195c67d` (identical to r7b) and sealed source pool SHA `2ffcf546543b84ac4b157a705d0e9a765eab0f9b32985d87a7896441c42b73a1`. I independently compared new vs old r7b checkpoint replay bytes for the first 10,000 update-free online decisions, all nine arrays and terminal observation stacks: exact equality. No claim about final policy quality follows from training.

Before disk optimization the new run used ~1.6 GiB. Running `fallocate --dig-holes` only on its two newly created checkpoint.pt files reclaimed unallocated zero replay slots: the run now uses 618 MiB; both checkpoint SHA-256 remain **byte-identical** to `result.json` (`e02f522e5a4e3982e553470ab8fefd40f7ffd0ac473a61d7a43927d77d8a3a03`, `ebf2bdda0414c6583ca83bd228155bf46b38502b8659a366b8aab140312a6a77`), and the final checkpoint mmap-reloads with 22,768 gradients/32,768 online replay entries. This is storage allocation only, not an optimizer, replay content, environment, model or checkpoint budget change. Free disk after reclamation ~7.5 GiB, cgroup OOM counter remains 1. Seed0/failure-weighted is now running; fixed gate is still unevaluated. Peers' artifacts and runs remain untouched.
