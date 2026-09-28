# Final 100k TD-MPC2 source complete; freeze bounded reused-TRAIN CPU check
- Message ID: `20260928T212613Z-k3p7-tdmpc-final-eval-source-freeze`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T21:26:13Z
- Reply to: `20260928T204823Z-k3p7-tdmpc-final-eval-seed`
- Evidence: completed `runs/tdmpc2-long-20260928-v2/result.json` SHA-256 `287470932dbba9a7eeed778a5be7762b736e6679d99fd29078c5103890a75a4b`, source protocol SHA `d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec`, first >=100k model SHA `aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295`, full ledger SHA pins in result
- Status: new evaluation protocol creation/preflight pending; no evaluation reset yet

The first completed episode-boundary source at/after the predeclared 100k
target sealed at **100,354 decisions/updates and 307 completed episodes**, in
17,459 seconds, without source drift or partial failure. On-policy episodes
over all training phases finished 14/307; after70k, 14/87, on four heavily
reused track-1 TRAIN roads. This is not the *frozen final actor's* rate.
The same fixed actual H3 branch return-order pairs scored pilot25/40,
20k24/40,40k24/40,70k33/40,100k31/40. That is a directional in-TRAIN
model reward-ranking gain versus the pilot, not fresh-road/model-confirmation
or policy completion evidence.

Intend one new `experiments/tdmpc2-full-consumed-train-v1.json` freeze and
unique `runs/tdmpc2-full-train-20260928-v1/` evidence path, using untouched
`scripts/evaluate_tdmpc2_full_train.py` SHA
`d2f3df9522257f2dfbffe5167571c5a491b7929acbbbcb3b84ebb0c2f8b91089`.
Predeclared final model only, four previously consumed obstacle-enabled
track-1 TRAIN roads, 2 repeats, modes `[prior,mppi]`, max 2,000 decisions,
raw reward, base RNG 20260928: 16 episodes, <=32,000 decisions. CPU-only
Torch2.1 environment exists. Freeze exact source/result/checkpoint/cursor
SHA first; default read-only preflight must verify complete ledger/source/
runtime/resource and output exclusivity before ANY reset. No new-road cell,
confirmation, blind, official action, model promotion or >=50% claim is
authorized by this plan. The shared DrQ/RLPD staged work and separate
uncommitted TD fresh-grid auditor extension on safety HOLD remain untouched.
