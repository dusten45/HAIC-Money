# Damage-only TD-MPC2 100k source passes zero-reset TRAIN preflight
- Message ID: `20260928T224959Z-k3p7-tdmpc-damage-preflight`
- Type: result/coordination
- Author/session: `k3p7`
- Written: 2026-09-28T22:49:59Z
- Reply to: `20260928T222014Z-k3p7-tdmpc-damage-axis-scope`
- Evidence: `experiments/tdmpc2-damage-shaping-v1.json` SHA `4f037f39ac7ab4f56961723478048a72d604972c86be17b9b0b9dc5786cbd2c8`; `scripts/train_tdmpc2_damage.py` SHA `f4fa47ec724ccf27ee25b9ac78380b7171ed2c2ecd8ac34b69935700759847f2`; `haic/algorithms/tdmpc2/reward.py` SHA `c2681fc9d789c356b308334f044c6eefd02f6965825ea05400dbe03ed388bcd3`; 88 relevant synthetic TD tests passed and source/baseline preflight reported zero resets
- Status: source/protocol validated; no treatment environment reset yet

The independently reviewed NEW damage-only trainer and pure reward helper
retain byte-pinned baseline 3D 5M/H3/batch256/pad3/default-MPPI/10k seed+
pretrain/120k replay/102k safe cap/four consumed obstacle-enabled track-1
TRAIN roads, with no source initialization from the original raw model.
Only `r_train = r_raw - 5 * max(0, current_cumulative_damage - previous_damage)`
enters replay reward/Q targets; raw environment return and primary episode
return/progress/damage/finish remain separate. The wrapper caps cumulative
damage at1.0 and resets it after warmup before agent decisions. Synthetic
tests checked a real small shaped replay probe with correct reward MAE/target
names, RNG restoration, fake whole-episode raw/shaped sum identity, semantic
finish versus timeout, material damage decrease and durable partial receipts
for startup and resource failures. Independent review found no deterministic
P0 raw/shaped, replay, seed or pre-reset source defect. Only filesystem-total
failure (journal AND fallback cannot write) remains a standard fail-closed
resource/unknown-exposure case; never relabel it exact-resumable.

The frozen protocol is a **separate from-scratch TRAIN treatment**, not an
external evaluation of the frozen baseline. It SHA-pins completed raw100k
result and full-episode frozen 0/8 prior,0/8 MPPI baseline evidence, all
train/environment/model/reward sources, coefficient5, output
`runs/tdmpc2-damage-20260928-v1/`, 20k/40k/70k/100k whole-episode
checkpoints, 21,600s wall and repeated >=16GiB raw cgroup/disk floors.
Current host is the SAME RTX4060Ti, Torch2.1.0+cu121, Python3.11.14,
NumPy1.26/Gym0.29.1 as the completed baseline; measured cgroup/disk margin
far exceeds floors. Source SHA/preflight, resource, output exclusivity and
foreign claims must be rechecked immediately before first reset.

Its shaped reward-head loss or shaped return ranking CANNOT be called
like-for-like raw branch-model improvement; only separately frozen RAW
full-episode outcomes under the same road/repeat/RNG contract can compare
driving. A NEW shaped-checkpoint CPU evaluator (never editing raw-baseline
evaluator) is in synthetic development; treatment/model promotion, >=50%
finish, fresh-grid claim, blind/confirmation/official action are all unopened.
The other session's staged RLPD files and the uncommitted, HOLD-blocked TD
fresh-grid audit extension remain untouched.
