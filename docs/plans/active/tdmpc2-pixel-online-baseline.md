# TD-MPC2 Single-Task Pixel Baseline (Independent Lane)

## User-Requested Focus Pause (2026-09-30)

**Current status: temporarily PAUSED to focus on other ideas.** This is
the user's explicit direction at 2026-09-30 00:04:27 UTC, superseding
all earlier autonomous-continuation and hourly resource-check instructions
in this plan. It is not a declaration that TD-MPC2 or the unexecuted
raw-MSE objective is ineffective. Resume only if the user explicitly
reopens this research line; improved memory availability is NOT permission.

The 00:25 UTC resource wakeup was cancelled. No TD trainer, adaptation,
benchmark, branch or policy process was active at the stop check, and all
research subagents had completed. Do not create another wakeup or respond
to a delayed cancelled alert by running research. Code, checkpoints,
partial/failed receipts and previously consumed TRAIN records are preserved.
The sections below are historical experimental context, not live commands.

### What Was About To Be Tested

The proposed experiment was an **objective-only reward-head control**.
The completed categorical-CE head pilot reduced fitting CE but worsened
positive raw-reward calibration even on fitting data. The next hypothesis
was that optimizing the decoded raw reward directly might better align
the head objective with the observed raw calibration criterion. It is
untested: categorical CE is not proven inherently wrong, nor is MSE
proven to improve planner action ranking, finishes or generalization.

| Item | Frozen intended choice |
|---|---|
| Parent | ORIGINAL completed overshoot100k checkpoint, not the failed CE-adapted head |
| Parent path | `runs/tdmpc2-overshoot-20260929-v1/checkpoint-at-least-100000-step-100159.pt` |
| Parent SHA-256 | `51b1da8c4e2c2584524a3d51ccbd82daa1121479c123237d2e809cbde1bcdaa0` |
| Replay | Original RAW100k complete replay/episode/step ledgers on the four consumed obstacle-enabled track-1 TRAIN geometries |
| Replay checkpoint | `runs/tdmpc2-long-20260928-v2/checkpoint-at-least-100000-step-100354.pt`, SHA `aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295` |
| Single changed objective | `mean((two_hot_inv(reward_logits, cfg) - raw_reward)^2)` instead of categorical `soft_ce` |
| Unchanged model | Original model size, action dimension 3, reward categorical support and standard symlog decoder |
| Trainable tensors | Only `WorldModel._reward`; encoder, dynamics, actor, online/target Q, termination and all non-head buffers frozen |
| Fitting split | Original RAW episode IDs 44..306, inclusive |
| Fitting-excluded logged split | Original RAW episode IDs 12..43, inclusive |
| Additional exclusions | Episodes 0..7 already inspected branches; episodes 8..11 reserved in the prior plan for a possible third branch cohort, not fresh or protected cells |
| Batch | 256 unique transitions within each update: 16 positive and 48 nonpositive raw-reward labels per each of four roads |
| Sampling and augmentation | Fixed seed 20260929; unchanged road/sign sampler and isolated per-update augmentation seed schedule |
| Optimizer | Fresh reward-head-only Adam, learning rate 3e-5, weight decay 0, gradient norm clip 10 |
| Budget | Exactly 512 successful updates; full execution hard wall 1800 seconds; no resume of a partial attempt |
| Environment interaction | None during head adaptation or its logged calibration screens |

These source episodes were already exposed TRAIN data, including prior
model training and diagnostics. Fitting exclusion is not independent-road
validation or a fresh confirmation/blind allocation. Source binding must
keep ORIGINAL RAW307 replay identities separate from overshoot309 model
producer metadata; earlier diagnostic prototypes mixed them and were
rejected before real execution.

### Completed Evidence Motivating It

- [Overshoot100k archived action score](../../../experiments/tdmpc2-overshoot-old-branch-score-v1-result.json):
  H5 reward rank improved 56/91 to 74/91 on identical real outcomes,
  but normalized absolute error worsened 0.396677 to 0.406912. Its
  four-way gate FAILED, so its default full-policy evaluator stayed closed.
- [Disjoint episode4..7 Q0 real branch](../../../experiments/tdmpc2-overshoot-q0-branches-v1-result.json):
  H5 Q0 ranked 68/93 versus H3 Q0 70/93, tied-best 7/12 below 8/12,
  with greater H5 regret. Gate FAILED; no H5 Q0 full episodes were run.
- [Categorical-CE head-only v2 pilot](../../../experiments/tdmpc2-head-only-adaptation-v2-result.json):
  all 512 updates completed, non-head tensors stayed bitwise unchanged,
  but excluded positive MAE worsened on all roads and 0/4 roads qualified.
- [Frozen-head fit/transfer audit](../../../experiments/tdmpc2-head-fit-transfer-v1-result.json):
  96,510 unique RAW transitions; fitting natural CE decreased
  1.5066 to 1.2208 while positive raw MAE/bias worsened on all four
  fitting roads. Its fixed mechanism gate rejects a transfer-only
  explanation, without establishing the precise causal defect.

### Code And Exact Execution Boundary

| File | SHA-256 at pause |
|---|---|
| `haic/algorithms/tdmpc2/reward_head_mse.py` | `4f49ee9b513bc0b7a0403048611e96035200124e0e0967b6c9b0045ee7d1c8d5` |
| `scripts/adapt_tdmpc2_reward_head_mse.py` | `783cc53ae025c55ec331d55bc255992fd6457c5bf0fe2210ae1245810c4407eb` |
| `tests/test_tdmpc2_reward_head_mse.py` | `70ff5eaf1ed5d7ffabe7068616391a84a9f49eb9021eb7eb5288ed213769fa54` |
| `tests/test_adapt_tdmpc2_reward_head_mse.py` | `451ab6142738cc5ddd53e1e1b667f8e6aaddff4c92686d262c7f507c41efee45` |

At pause, 91 combined synthetic/parent tests and pyright passed, and
independent finalization VETOs were resolved. These are code-safety
evidence only. Catchable timeout/interruption cleanup stages and fsyncs
`result.pending.json`, captures its inode before exclusive promotion,
and removes only an owned completion on failure; it does not guarantee
recovery from SIGKILL, power loss or denied filesystem cleanup.

**Actual raw-MSE source-checkpoint loads, adaptation updates and environment
resets are all ZERO. No raw-MSE performance result exists.** The following
planned artifacts are absent and must not be described as frozen/executed:

- `experiments/tdmpc2-head-only-raw-mse-v1.json`: prospective exact protocol.
- `runs/tdmpc2-head-raw-mse-benchmark-20260929-v3.json`: final-source generated CUDA benchmark.
- `runs/tdmpc2-head-raw-mse-20260929-v1/`: prospective exclusive real adaptation output.

Generated benchmark v1/v2 receipts do exist, but were produced by
superseded runner bytes before finalization fixes. Preserve them as
historical operational evidence; do not reuse them to release final code.
The generated benchmark deliberately uses nonzero synthetic final-head
weights to exercise MSE gradients: the unchanged decoder has a zero
derivative at exactly zero symlog expectation. A real attempt would still
start from the original trained parent, never those synthetic weights.

### Pending Budget Question

The previous resource gate required:

```text
available = min(host MemAvailable, cgroup memory.max - memory.current)
required  = max(source full-100k peak RSS, synthetic head benchmark peak RSS)
            + 24 GiB
          = 16,898,007,040 + 25,769,803,776
          = 42,667,810,816 bytes (42.67 GB, about 39.74 GiB)
```

The 16.898 GB input is the FULL overshoot100k process peak, not a
measured head-only loading/training peak. The extra 24 GiB is an
agent-selected fixed safety reserve, not an official rule or measured
workload need; raw cgroup headroom does not credit reclaimable cache.
The user questioned its conservatism, and the explanation acknowledged
that actual head-only need is unmeasured. No budget change was authorized
or implemented. The last blocking observation was 35.39 GB raw headroom.
Do not present 42.67 GB as a technical minimum. A future explicit restart
would need to resolve or deliberately re-freeze this budget before running;
the current pause is to focus on OTHER IDEAS, not to wait for memory.

### Gates And Conditional Path If Explicitly Reopened

1. Re-read current source and peer context; verify all hashes and absence
   of duplicate/partial outputs. Reconsider the unresolved memory estimate
   with loading-inclusive measurements and document any budget revision
   before execution, without confusing operational limits with performance.
2. Generate a final-source benchmark receipt with zero real replay/environment
   use. Forecast includes 512 full MSE updates, 686 FIT and 86 excluded
   inference batches, plus 600 seconds I/O allowance; require <1800 seconds,
   source/runtime/body hash and measured RAM/disk/VRAM gates. Old receipts
   are not substitutes. Re-run zero-load source/replay/split preflight.
3. Freeze the exact prospective protocol only after those gates. Real
   execution would produce 512 finite head-only updates and bitwise
   non-head parity from the original parent, with journaled intents,
   exclusive output, source rechecks and non-resumable partial evidence.
4. Require BOTH natural FIT and fitting-excluded logged screens: positive
   raw MAE and absolute signed bias improve by >=0.10 on >=3/4 roads,
   and no road's nonpositive MAE worsens by >0.05. Failure ends this
   objective axis before real branch or full-policy evaluation.
5. Only following both PASS outcomes consider the previously exposed
   episode4..7 actual-action preservation screen: H3 Q0 >=70/93,
   H5 Q0 >=68/93, tied-best >=7/12 each, H3/H5 regret no worse than
   1.028467882472486 / 1.279246663025931. This is TRAIN preservation,
   not independent validation. It is not yet implemented/released for MSE.
6. Only following preservation PASS could a separately frozen, audited
   episode8..11 x steps16/50/100 five-action protocol consume <=72
   new TRAIN reset intents. Planned screen: 60 full suffixes, >=40
   informative actual H5 pairs on >=3 roads, H5 Q0 >=70% and >=H3
   Q0 on identical actual returns, tied-best >=8/12 across >=2 roads,
   regret <=H3 Q0 and below all ten held-fixed Q1 pairs. Then, and
   only then, consider a distinct eight-full-episode internal CPU study
   with >=4/8 uncensored finishes across >=2 roads as a LOCAL signal.

None of these steps is authorized while paused. Prior FAILED gates
are not retroactively changed, protected/official resources remain
untouched, and no future test can relabel exposed roads as fresh.

## Instance-Migration Stop (2026-09-27 05:10 UTC)

The user stopped new experiments for Vast.ai instance migration. Do **not** run
the v2 protocol on this instance; it has zero environment resets, no checkpoint,
and no learner/evaluation result. The only TD-MPC2 TRAIN interaction is the
v1 partial attempt: 10,061 decisions, 10,000 pretraining plus 60 later
updates at its last valid boundary in
`runs/tdmpc2-reused-train-20260927-v1/boundary.pt` (SHA-256
`d3502e430a0c4bb3be9a993ab16434d6b5400afdc201177080f8cc4d035d8b4c`).
Its journal SHA-256 is `ff909545fe87fac34e649e016586b8babc22ab296aa33ad935ca4a1e13871961`;
the final `reset_intent` and `partial` after that boundary mean **exact resume
is prohibited**. Transfer the checkpoint and journal together for provenance,
not as a deployable or completed policy. No complete planned episode, paired
prior/MPPI result, confirmation, blind, or official action occurred.

After a new clone receives both committed code/protocol and separate Git-external
artifacts, inspect the migration handoff in `docs/context/`, verify pinned hashes
and resources, and obtain a new user instruction before opening any TRAIN cell.
The only predeclared follow-up is a **new from-scratch** v2 attempt using the
frozen protocol and consumed TRAIN roads, never continuation of v1. None of
the ordered research gates below authorize action during migration.

## Post-Migration Resumption (2026-09-28)

The user's new instruction to resume TD-MPC2 satisfies the plan's requirement
for a fresh instruction before opening the predeclared v2 TRAIN pilot. It does
not permit resuming v1: the restored v1 `training.jsonl` and `boundary.pt` SHA-256
values match the handoff, but the ledger ends after `reset_intent` and `partial`,
so that checkpoint remains provenance-only. V2 is a distinct from-scratch run;
its protocol is unchanged at SHA-256
`209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb`.

The source-pinned v2 `--preflight` has passed on the restored host with zero
environment resets. Its four scheduled cells are consumed TRAIN development
roads, not fresh validation: they have prior DrQ/Dreamer/TD-MPC2 interactions,
including exact track-1 rows in DrQ r6/r7 and final-source run ledgers. The
current host (RTX 4060 Ti, Torch 2.1.0+cu121) differs from the former training
host (RTX 5070 Ti, Torch 2.11.0+cu128); any v2 result must identify the new
runtime and must not be described as exact training-environment reproduction.
The separate `/tmp/kilo/haic-cpu21/bin/python` now imports Torch 2.1.0+cpu; a
two-process synthetic model-only export/MPPI check reproduced identical traces.
That check does not establish trained-checkpoint or whole-road behavior.

The immediate pre-reset source/resource `--preflight` was rerun on 2026-09-28
and returned `preflight_only` with `environment_resets: 0` for the unchanged
protocol SHA above. The v2 from-scratch TRAIN run ended at a valid episode
boundary with `boundary_decision_budget`, not a reset or wall-time failure. Its
[`result.json`](../../../runs/tdmpc2-reused-train-20260927-v2/result.json)
records 12,058 decisions, 37 completed episodes, 12,057 updates (10,000
pretraining, minimum post-seed budget met), and 1,629.3 seconds. It stopped
because only 1,942 decisions remained under the 14,000 cap, fewer than the
2,000-step maximum episode. All 37 TRAIN episodes ended without a finish, and
the prior next-road planner-reset failure did not recur during this run.

The terminal [`boundary checkpoint`](../../../runs/tdmpc2-reused-train-20260927-v2/boundary.pt)
SHA-256 is `08988417fec38b2a5940f3ad3037fc3e67b625735bf13ea90825470073bceee1`.
The separately pinned Torch 2.1.0+cpu export passed with zero resets and binds
that checkpoint, the protocol/source map, and the `cpu-model.pt` weights. The
[`paired TRAIN diagnostic`](../../../runs/tdmpc2-reused-train-20260927-v2/evaluation-result.json)
completed 16 episodes (8 prior, 8 MPPI; two repeats across the same four
consumed TRAIN cells). Prior recorded 0/8 finishes, 0/8 censored, mean progress
0.05252 and reward -46.904; MPPI recorded 0/8 finishes, 1/8 censored at 500
steps, mean progress 0.03487 and reward -50.155. The evaluator explicitly sets
`capped_finish_comparison_valid: false` and notes trajectories differ despite
matching road/reset seeds. These small reused-TRAIN proxy values are descriptive
only; neither an MPPI advantage nor a generalization result is established.

The v1 reset regression is resolved for this single run, but no driving signal
or basis for automatic scale-up was established. The pilot's frozen >=16 GiB
raw cgroup headroom, >=5 GiB disk and 7,200-second wall cap were **local
source-pinned conditions for that experiment only**. They were not official
HAIC local-training limits, user-requested minima or a requirement for every
newly authorized study. Preserve the pilot protocol/result bytes, but budget
new studies using measured remaining memory/disk writes and concurrent-job
growth under [the current workflow](../../workflows/run-experiment.md),
with a separate source/protocol. The later long and currently running damage
studies separately froze 16 GiB raw cgroup and disk plus 21,600 seconds;
those executable checks remain in force **for their own runs** until complete
and must not be loosened mid-run. All four cells are consumed TRAIN, not
fresh validation; confirmation, blind, official, promotion and changes to
another lane remain excluded.

## Longer-Budget Follow-up (2026-09-28)

The user reclassified v2 as an **operational but undertrained early pilot**:
10,000 of its 12,057 updates were seed-boundary pretraining, leaving only 2,057
post-seed updates. Do not call v2 a failed TD-MPC2 learning test or infer that
more MPPI search is the remedy. Its episode and capped prior/MPPI results are
still the consumed-TRAIN proxies above.

First, the [separately frozen exploration comparison](../../../experiments/tdmpc2-exploration-v2.json)
used the same four already-consumed track-1 TRAIN geometries, two repeats each,
and matched steering random draws. Its [measured result](../../../experiments/tdmpc2-exploration-v2-result.json)
selects **independent 3D** under the predeclared progress-first rule: mean final
progress 0.06897 versus exclusive 2D 0.03682, with 7/8 paired progress wins,
zero damage/finish in both arms. The 3D episodes were longer (321.625 versus
203.75 decisions), confounding the whole-episode raw-return comparison
(-65.95 versus -50.95); 3D reward per decision was -0.205 versus -0.250.
Absolute 10m spatial occupancy slightly favored 3D (12.25 versus 11.125
bins), whereas occupancy per 100 decisions favored 2D (3.82 versus 5.55).
These are eight episodes per arm on only four previously viewed roads; 3D's
unphysical overlapping pedals remain an explicit limitation, not a proven
general advantage. The first collector had one reset and zero decisions before
a metadata-check failure; its [receipt](../../../runs/tdmpc2-exploration-20260928-v1/failure.json)
is preserved rather than merged with the valid v2 comparator.

The **new, separately source-pinned** from-scratch study should therefore keep
the 3D action representation and the 5M model-size class, H=3, batch 256,
pad 3, rho 0.5, 10,000 seed decisions/10,000 pretraining updates and default
MPPI settings. Budget about 100,000 total decisions with sufficient replay
capacity; collect whole-episode checkpoints near 20k/40k/70k/100k. At each
checkpoint report the TRAIN road/episode denominator, progress, raw return,
damage and finishes, and separately identify interval-mean minibatch
consistency/reward/value/termination losses versus fixed-replay one-step fit
proxies, Q scale, policy entropy and both last-action and interval-aggregate
same-observation prior/MPPI action differences. No additional
confirmation/blind or official interaction belongs to this study; the four
reused TRAIN roads cannot measure unseen-road generalization.

The [source-frozen longer protocol](../../../experiments/tdmpc2-long-reused-train-v1.json)
at SHA-256 `bc1a2746845cbef89275c9b51163c273955ef1664fe833da35ed17e534e8c885`
passed a zero-reset preflight on the current host and started a new TRAIN run
at 2026-09-28 15:44 UTC. It uses exactly 10,000 uniform actions before
pretraining (v2's `<=` condition issued 10,001), retains four immutable
whole-episode checkpoints and no exact resume mode. The 102,000 cap reserves
one maximal episode beyond 100k; the 21,600s wall, 120k replay capacity,
repeated >=16GiB cgroup/disk floors, and process/runtime receipts are distinct
from v2 rather than retroactive changes to its frozen source. The prior-MPPI
delta uses the planner's weighted-elite distribution center before action
exploration noise; a separate applied-action delta includes that noise, and
neither is the isolated Gumbel-selected elite action. The first run **stopped
partially** at 10,020 decisions/updates after pretraining, with 28 completed
TRAIN episodes and an action-bounds `ValueError` in the open episode. There is
no 20k+ checkpoint or exact resume. Preserve the partial ledger/run and do
not infer a TD-MPC2 learning failure. A newly pinned source/protocol and
action-bounds regression test are required before restarting. The
[frozen failure receipt](../../../experiments/tdmpc2-long-reused-train-v1-failure.json)
binds the partial ledgers: all 10,020 actual applied actions were valid, and
an independently reproduced float32 weighted-elite diagnostic mean can exceed
one by a single ULP and cause the exact adapter error. The unrecorded next
mean cannot be directly verified. Clip only the diagnostic copy within a
1e-6 tolerance; leave applied actions, planner state and H=3 training
unchanged, and reject material overflow. Source-regression tests now cover
both one-ULP tolerance and material overflow; the corrected runner requires
a separately source-pinned retry protocol/run directory before any reset.
The [retry v2 protocol](../../../experiments/tdmpc2-long-reused-train-v2.json)
at SHA-256 `d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec`
passed zero-reset preflight and started from scratch in the distinct
`runs/tdmpc2-long-20260928-v2/` directory at 16:27 UTC. Its corrected
runner source is pinned at `9121fee37b5beeeaf7f22f253ab0500896e514fa7e5a140a8257483116df9849`;
88 relevant synthetic/TD regression tests passed before restarting. The
[20k result](../../../experiments/tdmpc2-long-reused-train-v2-20k-result.json)
now binds a sealed model at 20,099 decisions/updates, 67 completed TRAIN
episodes (0 finishes, mean progress 0.06746, return -58.81, damage 0.00299),
and SHA-bound read-only fixed-probe and same-anchor ranking receipts. The
256-window in-replay probe has 0/768 positive terminal labels, so its
termination BCE is not a positive-event test. Its reward MAE is 0.05899
versus 0.37949 constant, but the same 40 informative H3 branch pairs ranked
24/40 concordantly, versus 25/40 for v2's shorter pilot: no world-model
ranking improvement at 20k. The same-observation prior versus MPPI weighted-
elite *center* had mean L2 1.445 over 10,099 planned decisions; the applied
exploration-noised action gap was 1.475. These are dependent reused-TRAIN
signals and not full-episode evaluation or reason to tune search/H. The
[40k result](../../../experiments/tdmpc2-long-reused-train-v2-40k-result.json)
at 40,024 decisions/updates now reports 0/124 cumulative TRAIN finishes;
57 completed episodes since 20k averaged progress 0.182, raw return +36.36,
damage 0.158 and length 349.56, versus the first interval's 0.0675, -58.81,
0.003 and 299.99. These episodes are not matched independent evaluations,
and damage and episode length rose. On the *same* frozen seed replay probe,
consistency MSE decreased 0.00444 to 0.00373 and value CE 0.59159 to
0.54907, while reward MAE worsened 0.05899 to 0.08775 (reward CE 0.37125
to 0.37645); zero of 768 labels were positive terminal events. Q scale
rose 35.89 to 63.37 and same-observation prior versus weighted-elite
center mean L2 gap 1.445
to 1.477. On the SAME 40 informative H3 real-return pairs the model remains
at **24/40** (pilot 25/40), so a core world-model ranking survival signal
has not improved. Keep H3/default MPPI unchanged through 70k/100k before
considering any single-axis change. Training remains active; no full-episode
frozen-policy evaluation or >=50% conclusion is available.
For another view of the SAME 60 branch action sequences,
[predicted-return magnitude error](../../../experiments/tdmpc2-long-v2-branch-calibration-v1.json)
has MAE 0.904 at the short pilot, 1.190 at 20k and 0.900 at 40k; the
40k model merely returns to pilot-level calibration. This probes different
states/returns from the frozen seed replay reward MAE, and neither magnitude
trend establishes the still-absent within-anchor *ordering* improvement.
This reward-only H3 ranking is the world-model diagnostic, **not** the
planner's full action score: MPPI adds alive-gated `gamma^3 * Q` from a
sampled policy after the H3 rewards (`planner.py:101-140`). `q_scale` is
an EMA of the policy Q 5th-to-95th percentile *spread*, not mean or absolute
Q (`learner.py:182-190`). Thus 24/40 alone does not determine the planner's
candidate ordering or whether its bootstrap dominates; a separately bound
read-only reward/termination/Q decomposition on the SAME fixed anchors is a
post-100k hypothesis check, not permission to raise H or MPPI samples now.
The [balanced terminal-label read-only probe](../../../experiments/tdmpc2-long-v2-balanced-terminal-20k40k.json)
now closes the original frozen seed probe's 0-positive-label blind spot for
*raw termination*, not for finish/timeout: 67 positive and 335 ordinary
negative transitions from identical 20k replay windows were rescored at
20k/40k. `termination(encode(true next image))` recognized 0/67 positives
at both checkpoints, while `termination(predicted latent)` recognized
42/67 then 43/67; all four paths rejected 335/335 ordinary negatives.
The termination head's latent input distributions and the evolving encoder
differ, so this does not establish an implementation bug; both are in-sample
TRAIN measures, not evidence of 50% driving completion. No finish or
time-limit examples occur, and their detection remains unavailable.

The [70k checkpoint result](../../../experiments/tdmpc2-long-reused-train-v2-70k-result.json)
binds a third model at 70,361 decisions/updates. It still has 0/220
cumulative completed TRAIN episode finishes. Among 96 episodes since 40k,
mean progress rose to 0.3463 and raw reward/decision to +0.676, while mean
damage rose to 0.3979 (maximum progress 0.89); these different evolving
trajectories are not matched frozen-policy evaluation. Crucially, the SAME
40 informative reconstructed-anchor H3 raw-return pairs went **25/40 short
pilot, 24/40 at 20k, 24/40 at 40k, 33/40 at 70k**. Three of four road
geometries improved versus the pilot and one was unchanged, but pairs share
only seven informative anchors and four repeatedly viewed roads. This is
the first source-verified *within-TRAIN* directional world-model ranking
survival signal; no broader generalization or planner benefit is proved.
Fixed 256-window seed replay consistency MSE decreased to 0.002738 and
value CE to 0.46489; reward MAE is 0.06378 versus 0.37949 constant, with
0/768 positive terminal labels. The separate balanced, fixed positive/
negative probe's model-predicted-latent raw termination recall rose to
53/67 at 70k, while true-next-image latent recall remains 0/67 (no sampled
finish or timeout). Q spread scale rose to 133.12. Do not increase planner
complexity or tune reward based on the 70k signal alone: finish the frozen
100k boundary and predeclared full-episode reused-TRAIN prior and MPPI check
first.
An [on-policy first finish after the 70k checkpoint](../../../talk/messages/20260928T200236Z-k3p7-tdmpc-first-training-finish.md)
occurred at decision/update 74,216 on one of the four reused track-1 TRAIN
roads: one 646-decision episode, raw return 692.61, progress 0.95677,
damage 0.2, semantic `finished=true`; as of that decision the cumulative
evolving-policy denominator was **1/232** completed episodes. The 70k
snapshot remains 0/220 and no frozen model was saved at the actual finish.
This is a one-road TRAIN observation, not a replicated completion rate or
reason to retune before the predeclared 100k/full-episode gates.

The [completed 100k model result](../../../experiments/tdmpc2-long-reused-train-v2-100k-result.json)
binds the first safe boundary at 100,354 decisions/updates, 307 completed
TRAIN episodes and all source/checkpoint/ledger SHA receipts. The evolving
collector finished 14/307 episodes, all **14/87 since 70k**, spread across
all four repeatedly trained road geometries. This is not the frozen 100k
actor's finish rate. The SAME 40 informative fixed H3 raw-return pairs
ranked 31/40 at100k (pilot25/40,20k/40k24/40,70k33/40), so a directional
reused-TRAIN world-model reward-order improvement persists but does not
monotonically rise; 100k H3 magnitude MAE 1.033 is worse than pilot0.904
and70k0.887 on those sixty fixed sequences. Frozen 256-window seed probe
consistency MSE 0.002425, value CE 0.42066 and reward MAE 0.05184 improve
versus earlier snapshots, but its positive termination labels remain 0/768.
The separate fixed 20k-window balanced probe's planner-relevant *predicted
latent* raw-ending recall is now 59/67, ordinary specificity 335/335,
true-next-image recall 0/67; no finish or timeout is sampled. Q spread scale
rose to 140.61 while prior versus weighted-elite search-centre mean L2
gap is 1.344 across the final 29,993 planned actions. These are dependent
in-TRAIN fit/ranking diagnostics, not proven generalization or policy benefit.
The first >=100k model was chosen **before** seeing its outcome for the
[frozen full-episode reused-TRAIN CPU protocol](../../../experiments/tdmpc2-full-consumed-train-v1.json)
at SHA-256 `874d01d1396697efc9e4b119b0cd9ce8189fe36fa8d44773dcc13d30ab9ddbcd`:
four consumed roads x2 repeats x prior/MPPI, full 2,000 decisions, 16
episodes at most 32,000 decisions. Source/ledger/checkpoint SHA and CPU-only
runtime zero-reset preflight passed; the
[separate frozen result](../../../experiments/tdmpc2-full-consumed-train-v1-result.json)
is **16/16 valid, uncensored episodes, prior 0/8 finishes and MPPI 0/8**.
MPPI mean progress 0.507/raw return +385.93/damage 0.45 versus prior
0.327/+237.73/0.35, with mean CPU action inference 0.739s versus 0.00193s.
The road/reset settings are matched by mode, not their action trajectories.
The collector's 14/87 late *on-policy* TRAIN finishes cannot be substituted
for either frozen 0/8 mode. This four-road failure to meet a local >=50%
finish target is not a fresh-road, matched-treatment, official or intrinsic
TD-MPC2 algorithm-failure verdict. Preserve the 100k baseline model and
raw metrics before any next treatment.
The [source-bound no-reset final-model parity check](../../../experiments/tdmpc2-final-100k-freeze-parity-v1.json)
found all nine CPU/GPU output types within predeclared tolerances on four
archived TRAIN H3 windows after supplying identical *replacement* pixel
shift fixtures (latent max difference 5.60e-6; decoded-Q max 1.45e-4).
With identical CPU RNG and warm start, toggling MPPI `eval_mode` selected
the same elite; the training-only final Gaussian changed one final action
by L2 0.0823. This demonstrates intentional action noise, not that the
noise caused the 14/87 evolving-policy versus 0/8 frozen-policy gap.
CPU evaluation stored action hashes, not failure-action or pixel bytes,
and historical random shifts were not reproduced. Do not call parity a
causal explanation or change the baseline model on its strength.

The first *outcome-untested* one-axis follow-up is a separate scratch learner with
`r_train = r_raw - 5 * max(0, damage_t - damage_{t-1})`, using cumulative
damage reset to zero at each episode start. One ordinary +0.2 damage event
would cost one *training-target* reward unit; the chosen coefficient is a
frozen hypothesis, not a tuned causal remedy. Keep the official environment
step reward, primary logged raw return/progress/damage/finish, 3D 5M/H3,
batch256, pad3, rho0.5, 10k seed/pretrain, replay capacity120k and default
MPPI unchanged. Only the reward saved to replay for the reward head and Q
bootstrap differs; label all shaped rewards separately from raw rewards.
Use new source/protocol/run paths and synthetic source/terminal/timeout
tests before any reset; never reinitialize a shaped-target learner from the
raw-target 100k checkpoint. The existing *raw-return* fixed H3 branch ranking
is not directly comparable to a shaped-target reward head; raw full-episode
frozen-policy outcomes remain the primary comparable metric on the same
declared reused TRAIN cells. Damage rising on failed baseline episodes is an
observation, not proof of the hypothesized mechanism. H=5, additional MPPI
samples and actor-MPPI BC remain separate, later hypotheses only.
If this single shaping axis fails its own frozen full-episode comparison,
return to the original **raw-target** H3 baseline when separately studying
H=5 or MPPI sample count; do not stack shaping with another change and then
attribute an unmatched improvement to either one.
The prospective treatment comparison preselects its first whole-episode
>=100k checkpoint and unchanged default MPPI as the **primary** mode, prior
mean secondary, before any treatment reset. Only after a complete result and
strict CPU source gate, freeze the same four consumed TRAIN roads x2 repeats
per road/mode, raw reward, 2,000 decisions, base evaluation RNG20260928 as
the baseline. At least **4/8** uncensored MPPI finishes would be a *descriptive
local* >=50% threshold on four repeated training geometries, never eight
independent roads or established unseen-road generalization. Report each
geometry and the 0/8 baseline mode separately; keep strict source/receipt
hashes and no posthoc checkpoint/mode substitution. No such treatment model
or evaluation protocol exists yet. The
[source-pinned training protocol](../../../experiments/tdmpc2-damage-shaping-v1.json)
SHA `4f037f39ac7ab4f56961723478048a72d604972c86be17b9b0b9dc5786cbd2c8`
passed zero-reset source/baseline/host/resource preflight and 88 synthetic
TD regressions; the new `runs/tdmpc2-damage-20260928-v1/` source logged
its first 321-decision **seed** episode on a consumed TRAIN road. A separate
raw-metric CPU full-episode evaluator for the shaped checkpoint format is
now [reviewed as dormant code](../../../talk/messages/20260928T232547Z-k3p7-tdmpc-damage-evaluator-code-ready.md):
strict replay/probe-to-step reward/action/terminal, original producer raw
telemetry, exact reset-intent transitions, baseline checkpoint/ledger SHAs
and fsynced partial fallback passed 123 synthetic/adjacent tests and a
separate read-only review. Its fixes are isolated from the running trainer
and do not alter the frozen raw baseline evaluator. Its completed source,
checkpoint and separate CPU evaluation protocol subsequently passed SHA-
bound zero-reset preflight; the valid full-episode result is reported below.
Before pretraining, all 28 completed random seed episodes of the two runs
matched action hashes, road, episode length and raw return; no damage was
observed. The [first planned action already forked](../../../talk/messages/20260928T232052Z-k3p7-tdmpc-damage-first-plan-fork.md)
at decision10,001 although its observed damage increment was zero. Source
inspection found no explicit additional PRNG call/changed pretrain update
order, but neither run saved model/optimizer/pixel/RNG state at the seed
boundary. Therefore the treatment is a **single reward-axis source design**
on matched road/seed settings, NOT matched action/transition trajectories;
one later finish-rate difference will not identify a penalty effect on its
own. Keep this experiment's raw and shaped receipt intact. If needed after
first20k, compare seed replay/probe bytes without resetting roads; future
causal replication needs same-arm raw/raw reproducibility and seed-boundary
state/RNG snapshots plus independent learner seeds.
The later [20k replay-byte audit](../../../experiments/tdmpc2-raw-vs-damage-seed-parity-v1.json)
verified matching first10k seed observation pixels, action float32, raw
reward float32, semantic flags and 256 identical frozen seed-probe windows
under both SHA-bound checkpoints. There was no damage exposure in that
prefix, yet the decision10,001 planned actions still differ. No 10k
model/optimizer/shift-RNG checkpoint exists to isolate why; exact input
parity alone does not make later treatment trajectories matched.

The [first sealed 20k treatment checkpoint](../../../experiments/tdmpc2-damage-shaping-v1-20k-result.json)
at 20,149 decisions/updates has 68 completed on-policy TRAIN episodes,
0 finishes, mean raw progress 0.0663 and damage 0. In all **20,149 step rows**,
damage increment is zero and sum of raw and training-target rewards is
identical (-3,976.78); the penalty had **zero exposure** before this
checkpoint. The fixed seed replay probe's training-target MAE 0.06312
versus constant 0.37949 has zero positive terminal labels. Rolling
minibatch consistency/reward/value/termination losses, Q spread/entropy
and same-state prior/MPPI gap are in the frozen result. None is a shaping
outcome yet; first logged +0.2 damage occurred later, at decision 22,123
(raw -0.4 versus training -1.4). Keep H3/default MPPI unchanged to40k/70k/
100k; only after a complete model may source-bind the predeclared raw
full-episode CPU actor comparison.

The [40k damage-target checkpoint result](../../../experiments/tdmpc2-damage-shaping-v1-40k-result.json)
then seals 40,154 decisions/updates. Across all primary step rows, 65
positive damage increments totaled 13.0; raw reward sum -1,003.155 and
separate replay-target sum -1,068.155 differ by exactly **5 x 13 = 65**.
Every increment occurred after the 20k zero-dose boundary, proving the
new training target was actually applied without modifying environment
raw reward. The 63 episodes ending since20k averaged progress0.18050,
raw return+47.20, shaped training return+46.17, damage0.20635, length317.54;
**0/63 finished** (0/131 cumulative). Rolling consistency0.004816,
training-target reward CE0.41010, value CE0.66132, terminal BCE0.002708,
Q-spread scale41.72, actor differential entropy-26.48 and prior-to-MPPI
weighted-centre action L2 gap1.475 are on-policy/in-replay proxies. Its
fixed seed probe had no damage or terminal positives, so training-target
MAE0.04874 probes only raw-equivalent early states and is not comparable
to the baseline's raw H3 branch-return ranking. Source-level one reward
axis is now exercised, but trajectories forked before penalty, and no
one-seed causal finish advantage or >=50% full-episode result follows.
Maintain H3/default MPPI and the frozen budget through70k/100k; wait for
complete shaped source and independently pinned RAW CPU full episodes.

The [70k damage-target result](../../../experiments/tdmpc2-damage-shaping-v1-70k-result.json)
seals 70,481 decisions/updates and 203 completed evolving-policy TRAIN
episodes. Raw finish is **2/203 cumulative, 2/72 since40k**, with both
events on ONE of four repeatedly consumed road geometries. The later
72 episodes averaged raw progress0.402, return+226.98, damage0.394,
length421.21; this is not a frozen policy or replicated-road finish rate.
Cumulative primary step rows bind 207 positive damage increments totaling
41.4 damage and exactly **207 training-target penalty units**; 142 units
occurred since40k. The 30,327-update rolling training-target reward CE
was0.42011, consistency0.003553, value CE0.52270, terminal BCE0.001896;
Q-spread scale107.52, policy differential entropy-27.31 and same-state
prior/MPPI weighted-centre L2 gap1.520. Frozen seed probe has 0/768
positive terminal labels and no damage exposure, so MAE0.048996 does not
test penalized transitions or compare with raw-baseline H3 branch ranking.
The RAW baseline's separate late TRAIN interval had 0/96 finishes, but
trajectories diverged before the first reward penalty and are **not** a
matched/causal treatment contrast. Source settings were preserved to the
100k boundary; separately frozen RAW full-episode policy outcomes now
supersede this 2/72 on-policy sample for a finish-rate judgment.

The [completed damage-only 100k source](../../../experiments/tdmpc2-damage-shaping-v1-100k-result.json)
sealed at 100,186 decisions/updates with 286 completed repeated-TRAIN
episodes, 10 evolving-policy finishes (8/83 since70k on three roads), and
391 positive damage steps totaling78.2, hence 391 replay-only penalty
units while raw reward remained unchanged. Rolling consistency/reward-head
(SHAPED training target)/value/termination losses, Q spread, entropy
and prior-MPPI gap are separately named; its fixed seed probe has no
damage or positive terminal labels. These are not frozen-policy finishes
or like-for-like RAW H3 branch-model ranking evidence.

The separately [predeclared frozen full-episode CPU result](../../../experiments/tdmpc2-damage-full-consumed-train-v1-result.json)
completed 16/16 uncensored episodes on the same four consumed track-1
roads x2 repeats/mode with source-bound RAW environment metrics: prior
**0/8**, primary MPPI **2/8**, both MPPI finishes on one geometry and
zero on each of the other three. Its explicit >=4/8 local target failed;
even 4/8 on four training geometries would not prove cross-road private
generalization. MPPI mean progress0.614/raw return+479.42 increased
versus RAW-target baseline MPPI0.507/+385.93, but mean damage remained
0.45, one road's progress regressed, road/reset matching does not imply
action/trajectory matching, and FIRST planned training actions already
diverged at10,001 before the first damage penalty. Do not claim a causal
effect from one learner seed, a replicated 50% rate, model promotion or
official performance. Preserve both immutable source/result families and
isolate any subsequent H=5 or MPPI-sample axis against the original RAW
100k reference, not by stacking it with damage shaping.

The separate source-bound, parity-checked prefix-replay diagnosis compares
H=3 predicted *reward-only* discounted return rankings with actual
same-anchor action-suffix return rankings and records per-pair counts and
tie/terminal handling. Existing DrQ and RLPD prefix helpers are **not**
automatically valid for TD-MPC2: the TD wrapper differs and historical v2
step ledgers contain hashes, not full cell-bound action traces. An offline
within-replay logged-sequence ordering is a weaker observational proxy, not
the counterfactual same-state survival gate. A source-verified improvement
in the reused-TRAIN model-return ranking (first observed at 70k) is necessary
but not sufficient to raise H from 3, increase MPPI samples, or add planner
complexity. The 100k run and predeclared full-episode frozen-policy
comparison are complete. Their 0/8 finishes/mode permit considering ONE
separately protocol-frozen axis rather than stacking shaping, H=5, and MPPI
sample changes.

The [read-only v2 fallback](../../../experiments/tdmpc2-v2-logged-ranking-diagnostic.json)
bound all 37 completed replay episodes to the TRAIN ledger and scored the first
H=3 actions at each episode start without an environment reset. All 666
cross-start real-return pairs are ties within 1e-6 (logged range below 1e-15):
there is **no identifiable return ranking**, let alone a parity-verified
same-state counterfactual ranking. Do not treat its model predictions or
roundoff-sized raw-return differences as survival or failure evidence.

The [bounded same-reconstructed-state branch result](../../../experiments/tdmpc2-v2-prefix-branches-v1-result.json)
then used 12 predeclared anchors and five fixed H=3 action suffixes each,
with all 72 resets restricted to those four consumed TRAIN roads. Each branch
passed source-bound full-prefix pixel/reward/flag/raw-command and accessible
state parity before its suffix. Of 120 within-anchor return pairs, 80 were
actual ties and the v2 model ranked 25/40 remaining pairs concordantly
(62.5%; 15 discordant) across seven informative anchors. The 40 pairs share
four roads and five candidates/anchor and are not independent replicates.
This is an initial usable **internal** reward-ranking signal, neither a
significance test nor improvement from v2: historical hidden Box2D state
equality is not proven, 20k/40k each scored 24/40, and this
in-development proxy cannot replace new-road evaluation. Preserve the exact
anchor pixels, action bytes and real returns when comparing later checkpoints;
do not pick new branches based on this observed 62.5%.

After the ~100k milestones, if weak learning persists, consider raw-reward
shaping, then H=5, then increased MPPI samples **one at a time**, with separate
frozen protocols and same development denominator. No such change is approved
by an in-replay loss decrease alone. A longer-term
[TD-M(PC)^2](https://arxiv.org/abs/2502.03550)-family hypothesis is
planner-policy mismatch. Its main treatment regularizes the actor using the
stored planner *action distribution* (a log-likelihood term), whereas direct
behavior cloning of planner actions is a related ablation, not identical to
the paper's main update. Candidate HAIC extensions could separately test a
bounded planner-distribution regularizer and MPPI-action BC at the *same
observed states*. Neither is part of this baseline or an established causal
remedy here: first measure same-state action gaps and world-model ranking,
then compare actor imitation and its potential off-policy/model-bias harms
under a new isolated protocol.

## Five-Action Decision Gate (2026-09-29)

The [no-reset logged H5 quality screen](../../../experiments/tdmpc2-h5-logged-v1-result.json)
passed only its weak, necessary in-replay reward/termination gate. A separate
[predeclared real five-action branch protocol](../../../experiments/tdmpc2-h5-branches-v1.json)
then strictly bound the unchanged RAW100k H3-trained checkpoint, complete
replay and 12 reconstructed anchors on the same four consumed track-1 TRAIN
roads. The [primary result and decision](../../../experiments/tdmpc2-h5-branches-v1-result.json)
records 72 resets, accessible-state/pixel/raw-prefix parity, all 60 complete
five-step candidates and 120 within-anchor pairs, 29 of them real H5 ties.
The original *reward-only* H5 necessary gate barely passed: 56/91 concordant,
35 discordant, across four roads (>60% and >=40 informative). It authorizes
consideration, not automatic launch, of longer-horizon MPPI.

The user's more relevant action-choice comparison scored H3 and H5 on the
SAME new five action suffixes and observed H5 returns: H3 reward-only 63/91,
H5 reward-only 56/91; terminal-gated sampled-Q planner scores H3 **65/91**,
H5 **51/91**. H3/H5 sampled-planner top choice tied the best actual H5
candidate on 8/12 and 7/12 anchors; mean H5-return regret was 0.795 vs
0.799. H3 versus the actual *three-step prefix* was separately 31/40; this
is a distinct target and should not be conflated with the earlier historical
31/40 on a different old three-action suffix set. These five fixed candidate
scores are not a full 512-proposal MPPI policy evaluation or independent
road replicates; H3/H5 top-choice differences were small despite pairwise
rank deterioration. The raw H5 reward order deteriorated with depth, and
on these same 60 counterfactual suffixes the mean absolute predicted-minus-
actual reward error per discounted step rose from H3 0.330 to H5 0.397.
This error is a secondary mechanism probe, **not** the action-choice gate.
The sampled-Q/terminal score did not rescue H5 in this fixed sample. However,
the scored H3/H5 horizons drew different unordered Q-head pairs at 10/12
anchors and different bootstrap actions, and the mean top-choice regret
differed by only 0.00395. This cannot isolate horizon or establish which full
MPPI policy would finish more roads. Attribution to dynamics, reward head,
terminal, Q calibration or training coverage remains a hypothesis. A
separate [zero-reset fixed-critic control](../../../experiments/tdmpc2-h5-fixed-q-v1-result.json)
then held each of all ten Q-head pairs and bootstrap-policy RNG fixed
across horizons on these same candidates: H3 ordered 63-72/91 actual H5
pairs against H5 42-53/91, with top-choice tied-best 8-10/12 versus
4-6/12. The first read-only attempt stopped on a concurrently amended
source-result SHA; its refrozen retry passed source and final rehash. This
removes the critic-pair draw confound for *fixed-suffix score comparison*,
not the difference between full stochastic MPPI policies or competing
rollout/terminal/reward mechanisms. Historical hidden Box2D solver identity
is still unproven.

**Hold H5 full-episode MPPI.** The necessary H5 reward threshold passed,
but action-selection evidence is not sufficiently positive versus the
same-candidate H3 score. Do not activate the prepared dormant H5 evaluator
or spend eight policy episodes merely on that weak reward pass. The next
separately [source-frozen H3 final-action Gaussian test](../../../experiments/tdmpc2-h3-noise-full-consumed-train-v1-result.json)
kept original RAW100k model, H3, raw reward, 3D action and 512/24/6/64
MPPI unchanged, toggling only `eval_mode=True` to `False` as in TRAIN
collection. Its isolated CPU21 source/protocol/complete-ledger preflight
used **zero** resets; 147 combined TD diagnostic/evaluator synthetic tests
and independent read-only review checked code safety, not driving skill.
The real fixed eight-episode evaluation on the same four consumed TRAIN
roads x2 completed uncensored and yielded **2/8 finishes across two roads**,
below the predeclared **>=4/8 across >=2 roads** gate. Mean progress 0.7024,
raw return 546.38, damage0.35 and peak action latency 3.994s <5s are
internal metrics. Historical frozen H3 no-Gaussian MPPI finished 0/8 on
the same reset IDs, but actions and subsequent planner RNG differ, so no
matched-trajectory causal attribution to the late evolving-TRAIN 14/87
finishes or fresh-road benefit is justified.

**Next one-axis hypothesis: data coverage.** Both RAW/DAMAGE 100k learners
recycled only four track-1 geometries, while H3 frozen planning still
misses the local finish gate and H5 longer rollout rank worsens on their
counterfactual branches. A possible separate *cross-lane-consumed TRAIN*
track-1 coverage learner would retain H3/raw reward/model/planner/decision
budget and change only the deterministic geometry schedule to 24 catalog
TRAIN roads, four per six families including the original four. This is
NOT an allocated run or fresh diagnostic. First complete a candidate-specific
read-only audit of exact track+seed+obstacle conditions, catalog membership,
protected-ID metadata, ALL typed claims/partial/reset-intent cross-lane
exposure, diagnostic disjointness and current reservations. The old fresh
grid auditor is BLOCKED and cannot certify reuse. Re-audit/claim the
TRAIN-only cells immediately before a new protocol freeze and reset, bind
an isolated new runner (never relax the original four-road runner), and
reserve independent TRAIN evaluation rows separately without relabeling
trained roads fresh. If audit cannot resolve candidate-specific ambiguity,
do not reset or train; choose another single-axis model intervention on
already-consumed data instead of stacking damage/H5/noise or touching
protected cells.

**Coverage BLOCKED, next feasible H3 axis (2026-09-29):** The
[read-only consumed-TRAIN reuse inventory](../../../scripts/audit_tdmpc2_consumed_train_reuse.py)
and 47 passing auditor tests (42 subtests) distinguish candidate-specific
claims from unrelated lane drift, but even an otherwise clean 24-cell
manifest would return `BLOCKED`: legacy result-only interaction and
protected-ID metadata are not exhaustive. The metadata-only
[24-road catalog-order proposal](../../../experiments/tdmpc2-consumed-train-24-proposal-v1.json)
has four TRAIN cells in each of six families; the read-only inventory found
921 typed historical exposures but also conservatively flagged numerous
known prior TD/DrQ TRAIN records and its unconditional blocker. It cannot
certify any cell OR prove all cells collided. No claims or 24-road learner
reset exist; coverage efficacy remains untested. Do not waive the blocker
or claim a catalog road as fresh. The next one-variable
intervention that needs **no new cells** is to replace H3 MPPI's random
two-of-five Q-head bootstrap average with the arithmetic mean of all five
heads. First use the already consumed five-candidate real-H5 branches to
score that change with a common bootstrap RNG. Before seeing this score,
require >=66/91 informative-pair concordance, >=9/12 tied-best observed
choices and mean regret <0.7953492685094498 (the original one-draw H3
proxy). Only if all three TRAIN-development gates pass, source-freeze a
distinct complete eight-episode H3 all-Q CPU evaluator on the original
four consumed roads, keeping checkpoint/reward/horizon/action/MPPI search
settings otherwise fixed. An improved fixed-candidate score is not full
policy performance or a fresh confirmation cell.

The [all-five-Q no-reset result](../../../experiments/tdmpc2-h3-all-q-v1-result.json)
scored 65/91 real H5 pairs against original H3's 65/91, with 9/12
tied-best choices and mean regret 0.79416 (original 8/12, 0.79535).
The predeclared >=66/91 requirement failed, so do **not** create an
all-Q full-episode evaluator. The next model-level single-axis candidate
is multi-step counterfactual reward/latent target quality on the original
four consumed TRAIN roads, whose H5 predicted reward error rose from
H3 0.330 to H5 0.397 per discounted step. Distinguish untested reward-
head, dynamics, terminal and coverage mechanisms; only a new pinned
learner with replay-sufficient synthetic checks can test one target.

**Reward-target feasibility gate (2026-09-29):** The distinct [read-only
source-replay audit](../../../experiments/tdmpc2-reward-overshoot-target-audit-v1-result.json)
confirmed all 100,354 step rows/307 complete episodes and found 99,126
same-episode step-4/5 extensions among 99,740 H3-eligible windows,
99.3844% versus the predeclared >=95%; every old TRAIN road had
discounted suffix-reward range >0.1. The isolated `OvershootReplay` and
`OvershootLearner` in `haic/algorithms/tdmpc2/reward_overshoot.py`
passed 38 focused synthetic/base tests proving original H3 sampler/RNG
and disabled-aux one-update byte parity, boundary masking and nonzero
depth-5 reward/dynamics gradient. Do not count these tests as model
performance. Original pinned base learner/replay/model/planner are
unchanged. A NEW from-scratch source-bound trainer must preserve the
original four consumed roads/100k budget, pass a CUDA throughput forecast
of <20,600s under the original 21,600s wall cap, and pin a dedicated
result/partial ledger BEFORE starting a single reset. Only then freeze a
new 100k learner and check the predeclared old-branch reward-order gate
before considering any eight-episode policy evaluation.
The new isolated trainer at `scripts/train_tdmpc2_reward_overshoot.py`
passed 63 synthetic/base tests; a separate read-only review found no
blocking four-road/seed/update/checkpoint/resource violation. The
CUDA throughput operator was held BEFORE its first run when an `os.getpid`
versus NVIDIA host-PID comparison proved invalid in this container;
that and the own-GPU-utilization false alarm were corrected BEFORE the
first [synthetic CUDA PASS](../../../experiments/tdmpc2-reward-overshoot-throughput-v1-result.json).
The measured full-update-with-replay-sampling means 0.07749s original
versus 0.08978s overshoot; forecast 18,692.85s <20,600s. This synthetic
434-transition fixture does not guarantee long-run throughput. The
[frozen variant protocol](../../../experiments/tdmpc2-reward-overshoot-train-v1.json)
passed zero-reset strict source/runtime/benchmark/resource preflight.
Independently re-audit/declare the four ALREADY-consumed track-1 TRAIN
cells against current claims and protected-ID metadata before any first
reset; do not relabel a consumed cell as fresh. The independent 4/4
exact-cell review found no active exclusive/protected conflict, and a
non-exclusive consumed-TRAIN reuse intent was posted. The source-frozen
learner started 2026-09-29 12:26 UTC under
`runs/tdmpc2-overshoot-20260929-v1/` and [completed](../../../experiments/tdmpc2-reward-overshoot-train-v1-result.json)
at its first >=100k episode boundary: 100,159 decisions/updates,
309 episodes and 19,902.393 seconds under its unchanged 21,600s cap.
No collector finish rate is a frozen-policy rate. The
[predeclared old-branch score](../../../experiments/tdmpc2-overshoot-old-branch-score-v1-result.json)
reused exactly the 12 anchors, 60 five-action suffixes, 91 informative
H5-return pairs and 40 H3-prefix pairs without environment resets:
H5 reward concordance **56/91 -> 74/91**, H3-prefix **31/40 ->
34/40**, 4/4 road H5 rank nonregression, but H5 absolute return
error/discounted step worsened **0.396676986 -> 0.406911553**.
Thus only 3/4 mandatory gates pass. Keep its prepared H3/default-MPPI
eight-full-episode CPU evaluator DORMANT: do not posthoc waive the error
threshold or promote the single-seed repeatedly trained model. The rank
improvement on identical actual action suffixes is evidence worth
explaining; larger signed underprediction (-1.084 -> -1.368) and Q/
terminal effects are hypotheses to diagnose next using TRAIN-only data
and a distinct one-axis protocol. Do not equate local rank with policy
completion or official generalization.

**Post-100k mechanism and next Q-only gate (2026-09-29):** The
[zero-reset score-term result](../../../experiments/tdmpc2-overshoot-planner-terms-v1-result.json)
separated reward, termination gating and Q bootstrap on the SAME
12-anchor/60-candidate real H5 branches. All predicted termination
probabilities stayed <=0.5; new H5 reward rank 74/91 dropped to
**31-36/91** across EVERY one of ten fixed Q-head pairs and the
best-of-five mean H5 regret worsened from 0.502 to 2.076. This is
an arithmetic fixed-candidate Q bottleneck, not full MPPI policy
proof or a causal diagnosis of critic target learning. Independently,
the [source-pinned reward-blind positive-event probe](../../../experiments/tdmpc2-overshoot-positive-events-v1-result.json)
found positive-event underprediction on true encoded observations
on 4/4 original TRAIN roads. Its original script mistakenly mixed
RAW307 and overshoot309 episode lists; zero-load preflight FAILED
before a checkpoint was opened. Corrected source/tests and independent
review passed, then the 8,192-window no-env diagnostic passed the
separate head-only *mechanism* screen, not a policy screen. Bias -1.15
to -1.37 reward/step may reflect the new model being tested on OLD
replay, so head-only adaptation remains a hypothesis.

The next prioritized single planner axis retains the completed
overshoot100k checkpoint, track1/raw reward/3D actions/H5 default
512/24/6/64 MPPI, changing ONLY terminal Q-bootstrap weight from
1 to 0 while consuming the same policy/Q RNG draws. Do NOT reuse the
already inspected twelve branch outcomes to authorize a policy run.
Before real resets, freeze `QWeightedPlanner` as an isolated new module
and an exclusive H5 five-action branch protocol on disjoint ORIGINAL
RAW TRAIN replay episodes4..7 at steps16/50/100 (12 anchors, same
four consumed geometry seeds; episode lengths293-319), candidate-
specific cell exposure/reservation review, raw prefix pixel/action/
native parity and <=72 resets. Require 60 complete H5 candidate
branches, >=40 informative actual H5 pairs on >=3 roads, H5 Q0 score
concordance >=70% and strictly above ALL ten Q1 fixed-head scores;
H5 Q0 concordance >= H3 Q0 on SAME returns; Q0 top choice tied-best
>=8/12 on >=2 roads and mean H5 regret <every Q1 pair and <=H3 Q0.
Only if ALL pass, freeze a new **internal consumed-TRAIN** H5 Q0
eight-complete-episode evaluation with local >=4/8 finishes across
>=2 roads. This separate TRAIN-development path neither waives the
overshoot reward-error FAIL nor confirms fresh/official performance.

That [disjoint Q0 real-branch gate](../../../experiments/tdmpc2-overshoot-q0-branches-v1-result.json)
now **FAILED** with its complete 72/72 journaled reset intents and
60/60 H5 actual suffixes, zero optimizer updates. The corrected
RAW307 (not overshoot309) episode4..7 replay binder and before-EACH-
reset raw ledger/checkpoint rehash passed source/110 synthetic tests;
initial flawed source hashes never ran a reset. On 93 informative
same-anchor actual H5 pairs (27 ties), H5 Q0 scored **68/93** versus
Q1 ten-pair range **39-41/93**, but H3 Q0 scored **70/93** on the
SAME outcomes, Q0 best-choice was only **7/12** (<8), and mean real
H5 regret was 1.279 versus H3 Q0 1.028. Pairwise Q1 degradation
was addressed but extra H5 horizon still lacked enough action-choice
evidence. Keep `scripts/evaluate_tdmpc2_overshoot_h5_q0.py`
DORMANT; do not retroactively drop H3/regret/best-choice conditions
or claim a complete MPPI/independent-road result. A separate head-only
logged positive-event adaptation is worth assessing after its 4/4
read-only mechanism PASS, but must freeze its own one-axis model,
TRAIN-only data/split/score/failure protocol before any optimizer
update or environment reset. Road-coverage inventory is still
BLOCKED, not proven ineffective; no confirmation/blind/official cell
is available for iterative tuning.

**Head-only follow-up data gate:** The predeclared adaptation-excluded
original RAW ep12..19 split (two episodes per road) had only
38/87/38/39 positive raw reward labels per old road. This FAILED
the >=100 each threshold before any optimizer update; do not lower
it or relabel that v1 split as adequate. A distinct, fixed v2
split reserves old RAW ep8..11 for a possible third five-action real
branch, excludes ep12..43 from head fitting, and fits on ep44..306.
The eight-episode-per-road excluded cohort has 144/199/137/151
positives and >2,000 nonpositives per road; the fit pool has
>6,000 positives and >13,000 nonpositives per road. This passes only
a source-data-count screen, NOT head/critic calibration, action
ranking, policy completion or generalization. Before a new 512-update
B256 head-only pilot, isolate source and optimizer, freeze encoder/
dynamics/Q/actor/termination, and validate a measured CUDA/cgroup/
disk budget and adaptation-excluded positive/nonpositive MAE gate.
Then require old ep4..7 actual choice-rank preservation before even
considering a separately frozen ep8..11 real branch screen. The
overshoot four-way and H5 Q0 failure receipts remain unchanged.

The [head-only v2 pilot](../../../experiments/tdmpc2-head-only-adaptation-v2-result.json)
has now completed512 updates, zero environment resets, non-head
bitwise parity and367.66s elapsed under1800s. It **FAILS** its fixed
adaptation-excluded logged screen:0/4 roads meet positive-event
MAE/bias improvement, all four positive MAEs worsened and3 roads
exceeded0.05 nonpositive MAE deterioration. Do NOT score old branch
preservation, create ep8..11 real branch protocol or release full
episodes for this checkpoint. CE decreased on fit batches while
excluded error worsened; inspect fit/excluded data/action-policy
phase and encoder/target distribution before another single-axis
intervention. Preserve original model and failed adapted weights;
the previous v1 sample insufficiency and earlier overshoot/Q0 gates
are still FAIL, not overwritten by code/throughput safety PASS.

Next READ-ONLY mechanism gate: compare parent and adapted heads on
identical original RAW fit44..306 and excluded12..43 targets, separating
randomdecision<=10000 from early planneddecision>10000. Fit positive
events28518/86581 (32.94%) differ sharply from excluded631/9929
(6.36%); the early-planned subset243/3773 still has low frequency,
so random versus MPPI is not a complete causal explanation. Reject
successful-fit/transfer-only explanation unless fitting positive MAE
AND absolute signed bias improve>=0.10 on>=3/4 roads and nonpositive
MAE deteriorates<=0.05 onALL roads. Report natural and25:75 sampler-
weighted categorical CE separately from rawMAE, with fixed augmentation
and no optimizer/reset/tuning. This diagnosis never releases a failed
head checkpoint for actual branch or policy evaluation.
The [completed fit/transfer audit](../../../experiments/tdmpc2-head-fit-transfer-v1-result.json)
found fitting naturalCE1.5066->1.2208, balancedCE1.4002->1.1403,
but positive rawMAE/bias worsened onALL4 fitting roads. Fixed
fitmechanismgate0/4 rejects successful-fit/transfer-only explanation.
Next objective-only control keeps original overshoot parent and fixed
v2 data/sampler/augmentation/Adam/512 updates, changes ONLY loss from
categoricalCE to decoded rawrewardMSE (original two_hot_inv decoder),
nonhead frozen. Require measured resources and BOTH fit/excluded
rawcalibration gates before old actual-branch preservation; no FAIL
is retroactively overridden or third real branch outcome spent.
The MSE source is code-ready (91 combinedtests/pyright0), after
independent VETOs for final result release beyondwall and interruption
afterlink were resolved by staging+fsync/ownedinode-based cleanup.
Prior generated benchmarkv1/v2 receipts cannot validate revised source;
newv3 benchmark and source-bound protocol remain absent. Actual cgroup
rawheadroom35.39GB is below unchanged requirement42.67GB (sourcepeak
16.898GB plus24GiB). Hold before any checkpointload/optimizer;
the previous hourly resource-wait route was cancelled by the 2026-09-30
user-requested focus pause at the TOP of this plan. No resource check,
benchmark or adaptation follows automatically. The budget was acknowledged
as conservative rather than a measured head-only minimum; future explicit
reopening must resolve/re-freeze it before final-source benchmark and
no-load schema. Both fixed FIT/excluded gates remain prerequisites, not
permission to execute while paused.

Parallel source-only actor-imitation inventory verified14 completed
RAW finishes in307 evolving TRAIN episodes,6,229 executed-action
transitions across4 roads. These are not frozen-policy demonstrations:
collector labels include final Gaussian noise/clipped MPPI actions.
Episode counts alone do not establish same-pixel conditional action
consistency or actor learnability; actor-only BC remains HOLD until a
separate source-bound pixel/action consistency gate. Do not pool
overshoot successes to hide missing road coverage or imitate the failed
head-adapted model; no actor optimizer or policy evaluation has run.

## Full-Episode Finish Gate (user continuation, 2026-09-28)

The user requests continued training/evaluation iterations toward at least
50% **full-episode** finishes and the strongest project model. This is an
experimental target, not a license to use protected cells or to claim success
from a few repeatedly trained roads. The first source-frozen 100k attempt
stopped at 10,020 decisions with no 20k checkpoint; repair its action-bounds
failure in a new pinned protocol before inferring learning trends. Preserve
that partial run; there is no exact learner/replay resume. Finish the unchanged
H=3 baseline's planned 20k/40k/70k/100k diagnostic sequence before considering
reward, H=5 or MPPI changes one at a time.

Progress, return and world-model loss/ranking on the four TRAIN geometries
remain **training-development signals**. At candidate checkpoints, separately
freeze and run full 2,000-decision CPU actor/prior and MPPI episodes on allowed
TRAIN development cells, tracking finish/n, uncensored n, track ID, geometry
seed, raw return, progress, damage and latency. The v2 500-decision capped
prior/MPPI check is invalid for a finish claim. Reject this bounded baseline
iteration if operational feasibility fails or there is no completed-road
signal, rather than silently adding training interactions or escalating
planner compute.

A >=50% *generalization-development* gate needs a **separate, predeclared,
training-excluded TRAIN-only diagnostic grid** with multiple track IDs,
distinct road geometries, full-episode finish counts, and canonical repeats.
Before allocating a single new cell, audit cross-lane prior exposure,
reservations, protocol exclusions, and TRAIN-diag/confirmation/blind splits;
claim only eligible TRAIN cells. Freeze TD-MPC2 and an unchanged independent
reference actor (RLPD seed 11 is the strongest currently fully evaluated
reference) on exactly the same diagnostic cells. RLPD's 12/24 screen,
12/32 confirmation and 9/24 blind are **different cohorts**, not a matched
TD comparison or a replicated 50% result; protected cohorts remain retired.
Even an observed >=50% on the new TRAIN diagnostic would be internal evidence
only, not an official/private-track score or model confirmation. Iterate
distinct hypotheses only when the previous frozen TRAIN evidence supports
them; never retune against confirmation or blind results.

The separate `scripts/evaluate_tdmpc2_full_train.py` operator is prepared
for *reused TRAIN* full-episode CPU prior/MPPI comparisons. Its default
preflight requires an externally frozen evaluation protocol, a **completed**
long-run result, SHA-bound checkpoint and full step/episode ledger cursors;
it cannot treat an in-flight 20k checkpoint or the v1 partial as an evaluation
source. `--execute` requires isolated Torch 2.1.0+cpu and writes an exclusive
receipt with per-road/mode/checkpoint finish, censor, damage, progress, raw
return and action-latency denominators. The operator currently permits only
the four already-consumed TRAIN roads. It has passed synthetic tests but **no
evaluation protocol or real full-episode reset exists yet**. Any distinct
training-excluded multi-track TRAIN gate requires a new audited protocol and
cannot reuse this four-road operator as proof of generalization.

The cross-lane [TRAIN-cell audit finding](../../../talk/messages/20260928T164442Z-k3p7-tdmpc-train-grid-audit.md)
is stricter than simply checking the four TD roads. **No fresh multi-track
TRAIN-DIAGNOSTIC grid is currently certified or claimed.** The old DrQ
512-ID catalog was previously reserved; its 120 selected TRAIN roads across
tracks 1-4 have been consumed, while its separate 16 track-1 diagnostic roads
were already inspected. Geometry seed controls the road across track IDs,
so the same ID on different tracks is not a distinct geometry. Do not run
the RLPD G1 `--reserve` CLI for TD-MPC2: it only handles its own fixed
track-1 cells and does not fully parse TD `training.jsonl` reset-intent and
partial exposure. The shared claim library alone also cannot certify its
caller's incomplete exposure inventory.

For a prospective, not yet allocated gate, one bounded design is 24 unique
geometry IDs (six each on tracks 1-4) with 2,000-decision canonical repeat0,
then repeat1 as a reproducibility check. Predeclare the *primary* TD mode
before outcomes; a descriptive >=50% canonical-road target is at least
12/24 finishes, reporting per-track and censored counts. TD prior, TD MPPI
and an unchanged RLPD reference with two repeats would require up to 144
episodes / 288,000 decisions, so capacity and CPU feasibility must be checked
before claiming a grid. This is a study design, **not an ID allocation**.
Only after a promising completed TRAIN checkpoint, independently audit
candidate-specific cross-lane exposure, protected-ID exclusions, G0 claims,
all partial reset/interaction ledgers, registry and reservations; under a
shared lock re-audit and claim, freeze exact cell and source hashes, then
recheck immediately before the first reset. Candidate-relevant ambiguity
fails closed. No confirmation/blind outcomes are tuning inputs and no new
multi-track road has been driven by this planning step.

The new read-only `scripts/audit_tdmpc2_train_diag_seeds.py` inventories a
*separately predeclared* 24-cell candidate for known DrQ reservations,
cross-lane claims, TD partial reset-intent records and protected **ID**
exclusions. It has no seed generator or `--reserve` command. Its status is
intentionally always `BLOCKED`: prior result-only receipts and legacy
training exposure are not yet comprehensively certified. Synthetic
multi-track/protected/partial/torn-ledger tests pass, but the tool cannot
clear a candidate, claim a new road, or substitute for a later independent
audit and locked re-audit. Do not create an allocation from a zero-collision
report while its coverage blocker remains.

**Lower-tier reused TRAIN alternative, not fresh-grid clearance:** The
[read-only feasibility note](../../../talk/messages/20260928T192157Z-k3p7-tdmpc-reused-multitrack-path.md)
identifies a potential TD-training-excluded subset within the DrQ catalog's
120 *already-consumed* TRAIN geometries. A future metadata-only selector
could exclude all four TD training geometry IDs across every track and
predeclare four unique IDs from each of six road-shape families, one assigned
to each track 1-4, for 24 distinct geometries. No list of cells is selected
or cleared. All these geometries had cross-lane DrQ interaction on track 1;
that does not prove the exact track-2/3/4 obstacle variants were driven.
Classification must remain **TD-training-excluded, cross-lane-consumed TRAIN
development**, not fresh, independent, private-track or official evidence.
The shared new-seed claim API cannot claim these already-consumed roads;
use a *separate* lineage-bound reuse audit with protected-ID, current
foreign-claim, exact-cell reset-history and candidate-relevant ambiguity
checks, then recheck before a frozen reset. The existing four-road CPU
evaluator rejects these cells by design, so any 24-road assessment needs a
new explicit-cell evaluator and protocol after the 100k model/result are
sealed. Preselect one TD mode and 24 canonical episodes first; descriptive
12/24 finishes would be an internal reused-TRAIN threshold, **not** the
fresh-grid >=50% generalization gate above. Extra modes, matched reference
and repeats have separate denominators and CPU costs; no such road was
selected, claimed, reset or scored by this feasibility study.

## Hypothesis And Sources

TD-MPC2's learned latent dynamics and short-horizon planner may improve HAIC
driving over its own stochastic policy prior, but this is an untested hypothesis.
Start with the paper [TD-MPC2](https://arxiv.org/abs/2310.16828) and the
[official implementation](https://github.com/nicklashansen/tdmpc2) as primary
sources; pin the inspected revision before interpreting a run. Do not import
DrQ-v2/PPO training logic or introduce CarRacing-specific model, loss, reward,
action, or planner improvements in this first implementation. Adapt only the
official observation/action/episode boundary contracts to HAIC.

Pinned sources: TD-MPC2 `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`
and official Participants `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`.
Participants delivers four oldest-to-newest grayscale `float32` 84px images at
decision boundaries after 50 raw no-op reset ticks and up to four physics ticks
per decision; upstream TD-MPC2 instead expects three RGB 64px frames as 9
`uint8` channels. The TD-MPC2-only adapter retains all four grayscale planes,
resizes 84->64, quantizes to `uint8`, and configures the upstream four-conv pixel
encoder for 4 input channels (512 latent unchanged). This is an irreducible
observation adaptation, not literal upstream pixel parity. Symmetric `[-1,1]^3`
actions map to HAIC `[s,(g+1)/2,(b+1)/2]` without pedal exclusivity; symmetric
zero is half gas and half brake. The wrapper sums raw unshaped reward. Either
done flag resets planner/episode; a finish has HAIC `truncated=True` and
`info.finished=True` but is a semantic terminal with no bootstrap, unlike a
plain horizon truncation.

Official inference is CPU-only, <=1,024 MB, <=10s import/init, <=5s per reset
and per act. MPPI is not categorically prohibited; actual cold/warm planner
latency and process RSS must be measured before claiming deployability.
An [untrained-shape CPU preflight](../../../experiments/tdmpc2-cpu-preflight-v1.json)
measured 25 full MPPI actions under Torch 2.1.0+cpu after repairing a 2.1-only
tuple-axis reduction incompatibility. It cannot substitute for trained-checkpoint
reload, whole-episode latency, package validation, or an official run.
An independent [model-only synthetic export check](../../../experiments/tdmpc2-cpu-export-preflight-v1.json)
also loaded 2.11-produced weights in two fresh CPU-only 2.1 processes and
reproduced four-action seeded MPPI traces; it is still not a trained checkpoint.

For the official-repository 5M **model size** (not five million environment
steps), select `model_size=5`: 512 latent/MLP, five Q heads, 101 symlog bins,
batch 256, horizon 3, rho .5, random shift pad 3, 512 MPPI candidates (24
policy-prior), six refinements and 64 elites. The current repo's 10M-step run
default is not this pilot's budget. At the HAIC local 2,000-decision episode
limit, upstream's formula yields discount .995 and seed steps 10,000; its loop
performs 10,000 pretraining updates at the seed boundary on already completed
episodes, then one update per subsequent decision. A shorter pilot that omits
or reduces this work must be labeled an explicit computational deviation, not
quietly called a faithful upstream training schedule. The current optional
episodic termination mode must be enabled because HAIC has actual terminals.
The upstream default 9-channel non-episodic model has 4,865,540 trainable
parameters in this implementation; the required 4-channel HAIC input plus
optional termination head produces 5,385,573. Thus "5M" names the upstream
size class, not identical weights/parameter count across environments.

## Upstream Mechanism And Evidence

At [upstream `world_model.py`](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py),
the encoder embeds stacked pixels into 512 SimNorm coordinates. The deterministic
dynamics learns the next latent from current latent/action against a detached
encoder target; the reward model learns symlog two-hot reward, and five Q heads
learn symlog two-hot bootstrapped return. Two randomly chosen Q heads are
decoded: their minimum builds the TD target and their average values actor and
planner actions. An independently optimized tanh-squashed **Gaussian** prior
maximizes scaled detached Q plus weighted sampled entropy; it is neither a
truncated normal nor the action selection rule by itself. The planner draws 24
trajectories from that prior and 488 from a Gaussian, evaluates learned
reward+terminal Q through dynamics for horizon 3, repeatedly re-fits weighted
64 elites, then Gumbel-selects an elite first action. Evaluation suppresses
only its final exploration Gaussian; pixel ShiftAug and elite selection remain
stochastic, so deterministic evaluation requires reset-specific RNG replay.

[Latest upstream `tdmpc2.py`](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py)
targets Q using detached encodings of *actual next observations*, although
[paper Eq. 3](https://arxiv.org/html/2310.16828v2#S3.E3) writes a dynamics-
predicted next latent; this lane follows the newer code. Paper section 3.2
writes a Gaussian sampled action and shifts both its mean and std; latest code
selects an elite first action and shifts only its mean. The paper predates
current opt-in episodic termination. Strict H+1 replay slices sample only
completed episodes and apply per-frame independent pad-3 random shift **inside
the encoder**. This distinction prevents replay-side double shifts.
The operator deliberately updates from episodes completed *before* the current
decision, then admits the current transition/just-ended episode after updates;
the seed-boundary pretraining therefore never sees a newly ended episode from
that same action. The CPU-only development evaluation must load a separately
source-bound model-only export under official-version CPU Torch, not merely move
the CUDA training model to CPU inside the same process.

## Ordered Gates

1. Audit official 5M single-task online/pixel defaults and current episodic
   termination implementation; verify local HAIC pixels/history, reward, action
   mapping, terminated vs truncated, and official inference constraints. Record
   any irreducible differences from upstream before environment interaction.
2. Synthetic and local parity tests must cover environment observation/action
   bounds and inversion; correct episode reset/bootstrap semantics; no sequence
   crossing and matched temporal image augmentation; finite and connected
   encoder/dynamics/reward/Q/policy losses and target EMA; MPPI action bounds and
   prior comparison; exact checkpoint/replay/optimizer/RNG resume and deterministic
   CPU evaluation. Failed tests block any pilot.
3. Freeze a separate small TRAIN-only pilot after a candidate-specific cross-lane
   geometry/track/conditions audit (and shared TRAIN claim if allocating *new*
   cells). Record executable and
   environment hashes, exact reset cells, seed-step, decision/update counts,
   diagnostics, CPU/GPU resources, and stop criteria before first reset. Preserve
   partial attempts as consumed; never substitute cells silently. No confirmation,
   reserved blind, or official submission/evaluation is authorized here.
   No generic new-lane four-cell freshness auditor currently exists; the proposed
   fallback is reused obstacle track-1 r6 TRAIN seeds `3910800001`,
   `3910800004`, `3910800034`, `3910800085`, already reset by Dreamer. These
   are consumed development cells, not fresh claimable holdouts.
   The frozen pilot protocol is
   [`experiments/tdmpc2-reused-train-pilot-v1.json`](../../../experiments/tdmpc2-reused-train-pilot-v1.json),
   SHA-256 `e671cb916ea07c7c902b029393d5728560cfe9b48b04bade3a8c5604d54ef748`;
   its read-only source/resource/cell preflight passed with zero environment
   resets. It caps 14,000 decisions, 14,000 updates, 7,200 training seconds,
   and pairs the prior with MPPI only in 500-decision-censored reused-TRAIN
   episodes after source-bound CPU-only model export.
   The first v1 attempt stopped after 10,061 decisions and 10,000 pretraining
   plus 60 later updates: an inference-mode MPPI warm-start tensor caused
   `planner.reset()` to fail on a subsequent TRAIN road reset. Its
   [partial receipt](../../../experiments/tdmpc2-reused-train-pilot-v1-failure.json)
   and boundary checkpoint are preserved; ledger ends after `reset_intent`, so
   it cannot be presented as a complete pilot or resumed exactly. A separate
   v2 attempt may only open after the replacement-tensor reset regression,
   source re-pin, reused-TRAIN audit, resource preflight, and new frozen path.
   The separate [v2 protocol](../../../experiments/tdmpc2-reused-train-pilot-v2.json)
   (SHA-256 `209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb`)
   and [v1-exposure receipt](../../../experiments/tdmpc2-reused-train-pilot-v2-exposure.json)
   pin that exact reused-cell history. At the 2026-09-27 04:49 UTC recheck,
   full source/resource preflight passed with **zero v2 resets** (raw memory
   ~17.1GiB free), but shared disk had only 5.5GiB free, ~0.5GiB above the
   immutable 5GiB floor. A v2 boundary checkpoint alone is expected to take
   ~358MiB while independent checkpoint writes continue; do not launch into
   that transient margin. Wait for safe sustained headroom; do not lower
   either floor, delete other lanes' artifacts, or describe v2 as run.
4. Assess world-model held-sequence predictions and reward/Q calibration on
   TRAIN-only experience, actor-only vs MPPI on identical TRAIN development
   road/seed cells, and progress/finish/raw-reward proxies with denominators.
   Finite losses alone and within-replay fit do not establish driving improvement.
   Pilot failure should separate verified adapter/loss/replay errors from possible
   algorithm-environment mismatch; neither implies a general TD-MPC2 limit.
5. Only if the pilot is operationally correct and exhibits learning/driving signal
   design a NEW protocol with a decision/update budget comparable to DrQ r6/r7
   (32,768 additional online decisions after pre-trained source initialization,
   with 10,000 update-free startup decisions and 22,768 updates). A from-scratch
   TD-MPC2 run has different initialization, prior data, replay exposure and
   compute; equal marginal steps do not make it matched. No automatic expansion.

## Failure Boundaries

Stop on a mismapped action or pixel stack, impossible evaluation latency, invalid
terminal targets, cross-episode replay, nonfinite model/critic/policy gradients,
broken replay/resume, candidate-relevant seed collision, insufficient cgroup
headroom, or planner output outside the environment action contract. Preserve
the failure and source context rather than tuning the baseline ad hoc.

The corresponding standalone code belongs under `haic/algorithms/tdmpc2/`, a
one-off operator under `scripts/`, and run receipts under `runs/`. Report proxy
metrics as local TRAIN results only; no model is nominated for official action.
