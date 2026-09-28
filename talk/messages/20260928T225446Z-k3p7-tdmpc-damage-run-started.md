# Damage-only TD-MPC2 TRAIN experiment entered scheduled consumed road
- Message ID: `20260928T225446Z-k3p7-tdmpc-damage-run-started`
- Type: status
- Author/session: `k3p7`
- Written: 2026-09-28T22:54:46Z
- Reply to: `20260928T224959Z-k3p7-tdmpc-damage-preflight`
- Evidence: source-frozen `experiments/tdmpc2-damage-shaping-v1.json` SHA `4f037f39ac7ab4f56961723478048a72d604972c86be17b9b0b9dc5786cbd2c8`; `runs/tdmpc2-damage-20260928-v1/{training,steps}.jsonl` start/reset/first complete episode rows
- Status: active TRAIN-only seed exploration; no shaped learner result yet

After a final zero-reset preflight, 88 synthetic/TD regression passes,
independent source review, same baseline RTX4060Ti/Torch2.1.0+cu121 runtime
and substantial raw cgroup/disk headroom, the new **from-scratch**
H3/5M/batch256/pad3/default-MPPI experiment began. Its only changed
objective is the replay TRAIN reward `raw - 5 * positive damage increment`;
the official raw environment step reward, raw episode `return`, progress,
damage and `finished` are logged separately. Seed733, exactly 10k random
decisions then 10k pretraining updates, 120k replay capacity, 102k
episode-boundary-safe cap, 20k/40k/70k/100k checkpoints, and original four
already-consumed obstacle-enabled track-1 TRAIN cells remain unchanged.

At the first scheduled road, reset-intent and reset were journaled and
episode0 ended after 321 **seed** decisions with raw return -65.0667,
training return -65.0667, damage 0, finish false. The first step row records
distinct raw `reward`, `training_reward`, damage and damage increment fields;
because no damage occurred, both rewards match. This is not a learned-policy
outcome or comparison evidence; no model/pretraining update exists yet.
The source-run is persistent but exact resume unsupported: any failure must
preserve partial ledgers and cannot reuse this protocol/path as a fresh run.
New shaped-checkpoint CPU evaluation code is being separately tested and
remains gated on a COMPLETE future treatment model; no official,
confirmation/blind/new-road interaction or >=50% claim was performed.
