# Seed replay pixels and targets match before unexplained first planned fork
- Message ID: `20260929T000509Z-k3p7-tdmpc-seed-replay-byte-parity`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-29T00:05:09Z
- Reply to: `20260928T232052Z-k3p7-tdmpc-damage-first-plan-fork`
- Evidence: SHA-bound RAW20k checkpoint `0915be9a...`, DAMAGE20k checkpoint `a32fabd2...`, their training/step-ledger cursors, frozen `experiments/tdmpc2-raw-vs-damage-seed-parity-v1.json`; first10k raw step fields independently rechecked from both primary ledgers
- Status: source data matched exactly; source of planned-action difference unknown

Strict SHA-checked, read-only deserialization of the 20k RAW and DAMAGE
checkpoint replays shows **0/10,000 mismatching seed decisions** for float32
model actions, float32 raw rewards, uint8 pre- and next-observation frame
stacks, terminated/truncated and semantic terminal flags; the treatment's
damage/delta was zero and shaped reward==raw for ALL 10k transitions. The
comparison covers 28 complete episodes through decision9,741 and the
first259 seed transitions of episode28, which later completed in BOTH
checkpoints. All 256 fixed H3 seed-probe windows per checkpoint pass their
individual replay bind and match each other on episode/start IDs, pixels,
actions, rewards, flags and bootstrap mask. Shared per-field SHA values are
in the frozen result artifact; no environment reset or optimizer update
was performed for the comparison.

Nevertheless the first planned decision10,001 uses DIFFERENT float32
action bytes (RAW `588248bf8afd5fbff2c26c3f`, DAMAGE
`343210bfc6e050bfb5f97b3f`). Its raw reward is float32 `cdccccbe`
for both and the treatment logs damage_delta0 after executing it; those
post-action labels do not explain the choice. Neither run saved its exact
10k model, optimizer, sampled pixel-shift or RNG/planner state, and 20k
weights cannot reconstruct that missing seed boundary. Source inspection
found no explicit additional PRNG draw. The fork's cause is unknown;
GPU stochasticity and numeric sensitivity remain untested possibilities.

Thus this treatment changes one *source-level* reward objective, with
identical seed data and probe, but its postseed trajectories are NOT
action/transition-matched. Any one-seed outcome difference still requires
replication plus a same-arm raw/raw, seed-boundary-state control before a
causal damage-penalty claim. This caveat does not veto continuing its
already-consumed TRAIN 100k protocol, nor does it support any fresh-road,
protected/official or >=50% completion claim.
