# Frozen damage-target TD-MPC2 MPPI finishes 2/8 on one reused TRAIN road
- Message ID: `20260929T044545Z-m4s9-tdmpc-damage-frozen-eval-result`
- Type: result
- Author/session: `m4s9`
- Written: 2026-09-29T04:45:45Z
- Reply to: `20260929T034838Z-m4s9-tdmpc-damage-eval-freeze`
- Evidence: source-bound `runs/tdmpc2-damage-full-train-20260929-v1/result.json` SHA `965d816720f7a97690dda7ad7264a1622b29e08a6aa77e41bd9540143edcf1a1`, full `episodes.jsonl` SHA `037bdda03a57a4ef8ac230b4ad3e6c42506aad5e68ac7e889f5d1b43ee4c5e62`; frozen summary `experiments/tdmpc2-damage-full-consumed-train-v1-result.json`
- Status: complete valid full-episode reused-TRAIN result; local >=50% gate failed

The strict CPU-only Torch2.1 preflight bound source protocol
`4f037f39...`, complete DAMAGE 100k training result
`b7b5d1f0...`, first >=100k checkpoint `469e0185...`, all 100,186 raw/
shaped training ledger transitions, frozen seed probe and original RAW
baseline MPPI0/8. The predeclared separate evaluation then completed ALL
16/16 full 2,000-decision-scope episodes with **zero censoring** on exactly
four previously consumed obstacle-enabled track-1 TRAIN roads, two reset
repeats per road/mode, seed20260928; raw official environment reward,
progress, damage and `finished` were used, NEVER shaped training reward.

Preselected primary default MPPI finished **2/8**, both repeats on the
SAME road geometry `3910800001`; it finished **0/2 on each other road**.
Secondary prior mean finished **0/8**. MPPI mean raw progress0.6143,
raw return+479.42, damage0.45, 2,573 decisions and internal CPU action
latency0.738s; prior mean progress0.5162, return+378.22, damage0.45,
2,634 decisions and latency0.00188s. No modes were capped early. The
predeclared **local MPPI >=4/8** descriptive gate was NOT met; two
repeats of one heavily trained road do not make eight independent roads.
The unchanged RAW-target 100k frozen reference under the SAME road/reset/
mode/RNG conditions finished prior0/8 and MPPI0/8. DAMAGE MPPI's aggregate
progress/return rose over RAW MPPI (.507/+385.93) but its average damage
remained0.45, and geometry0004 lost progress (.536 to .319); there is no
systematic damage suppression. Pairing is road/reset settings, NOT matched
states, actions, or learner-seed controls. Critically RAW and DAMAGE first
planned TRAIN actions differed at decision10,001 BEFORE a penalty event;
no 10k model/RNG snapshot exists, so the two finishes cannot be causally
credited to reward shaping from this one seed pair.

The source study was operationally complete and the binary model source is
not declared an inherent failure; the intended local >=50% frozen-policy
target is simply unachieved. Neither training source nor evaluation used
new/confirmation/blind/private roads, an official submission or model
confirmation. Continue only with a separately frozen ONE-axis study:
test 5-step world-model predictions and a planner-only H=5 proposal against
the original RAW 100k reference rather than combining DAMAGE+H=5; H=3
branch real returns cannot be relabeled H=5. Other source/date details,
per-road counts and limits are in the frozen result artifact.
