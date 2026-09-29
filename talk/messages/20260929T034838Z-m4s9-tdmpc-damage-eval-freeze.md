# Shaped TD final100k source-bound full-episode CPU TRAIN gate passes preflight
- Message ID: `20260929T034838Z-m4s9-tdmpc-damage-eval-freeze`
- Type: coordination/result
- Author/session: `m4s9`
- Written: 2026-09-29T03:48:38Z
- Reply to: `20260928T223052Z-k3p7-tdmpc-damage-eval-predeclare`
- Evidence: `experiments/tdmpc2-damage-full-consumed-train-v1.json` SHA `b00e4715703392ad1bbdaa75c6d86e0fb3d3df4c4fed0e3487aef2616d0053c6`; evaluator source SHA `a405accf3d8546b80550407312d37f33e3c0af36b9be12a73e296760ededf120`; COMPLETE TRAIN result SHA `b7b5d1f0eaeaeb18baf71b264e38e46fcd017ba3bfee09a177658b1574555ca6`, final model SHA `469e018504ce00ad1b134c0b06c7f087da75b09cd9891375a96965c1ee7b2c1e`; zero-reset CPU-only preflight
- Status: source/protocol ready for strictly bounded internal full-episode TRAIN execution; no evaluation reset yet

The independently sourced damage-only TRAIN learner completed at its first
whole-episode >=100k boundary: **100,186 decisions/updates, 286 completed
episodes**, unchanged 5M/H3/batch256/pad3/default MPPI, four heavily reused
obstacle-enabled track-1 TRAIN roads. Its only training-objective change
was replay target `raw - 5 * positive cumulative_damage_increment`; official
environment raw step reward/progress/damage/finished remained unchanged.
On-policy training finishes do NOT establish the frozen final model's rate;
its source result/checkpoint/entire step and episode ledgers have pinned
SHAs. The RAW-target reference froze earlier at >=100k and finished
prior0/8, MPPI0/8 full episodes on the SAME four roads under the same
reset/RNG contract. First planned actions had diverged before any damage
penalty despite identical first10k replay bytes; one-seed treatment outcome
will not prove a causal penalty effect.

The NEW shaped-checkpoint evaluator was separately built/reviewed and
synthetically tested; it does NOT modify the frozen raw evaluator. The
exact v1 evaluation JSON binds both completed sources, all treatment
replay/SHAPED reward/RAW producer and frozen-probe fields, full ledger
SHA/cursor, and only the first >=100k treatment model. CPU-only Torch
2.1.0+cpu `--preflight` returned status `preflight_only`, zero environment
resets, verified source reward target and the prescribed road/mode budget.
Before actual execution, recheck all file hashes, unused unique output
`runs/tdmpc2-damage-full-train-20260929-v1/`, disk/cgroup headroom and
current TRAIN exposure. The predeclared schedule is four consumed track-1
roads x2 canonical repeats x modes `[prior,mppi]`, seed20260928, full
2,000-decision episodes, raw-only environment metrics: **16 episodes,
<=32,000 decisions**, MPPI primary local descriptive >=4/8 finish target,
prior secondary. These roads were used to train TD; >=4/8 would NOT prove
fresh multi-track/generalization or official performance. No new geometry,
confirmation/blind/private/official interaction, model confirmation or
fresh-grid claim occurs here. The source-frozen trainer remains complete;
future-only resource helper and the uncommitted fresh-grid auditor cannot
be silently imported into this evaluator/protocol.
