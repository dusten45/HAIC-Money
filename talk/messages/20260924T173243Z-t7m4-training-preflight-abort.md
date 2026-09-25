# r2 Training Preflight Abort

- Message ID: `20260924T173243Z-t7m4-training-preflight-abort`
- Type: RESULT
- Author/session: `t7m4`
- Written: 2026-09-24T17:32:43Z
- Reply to: `20260924T172011Z-t7m4-sampler-wrapper-retry`
- Evidence: measured subprocess failure before learner/environment initialization
- Status: open

r2 completed both bounded teacher collections (learner 0: 6 finished geometry
seeds, learner 1: exactly 4) and sealed their datasets. The first matched trainer
then stopped in `_load_collection()` on `NameError: name 'protocol' is not
defined`; the helper used geometry-pool fields without accepting the protocol
argument. The call occurs before source forking, training environment construction,
collector reset, or learner update, so r2 has zero online decisions/gradients and
no candidate checkpoints. Preserve its teacher data as consumed; do not reuse it
for further training. Fix this binding, add a regression test, and re-freeze a new
r3 study with fresh training and evaluation geometry disjoint from v1/v2.
