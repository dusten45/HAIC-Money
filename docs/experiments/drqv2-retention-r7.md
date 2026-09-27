# DrQ-v2 r7 Source-Behavior Retention

This report concerns only previously designated, reused TRAIN-DIAGNOSTIC roads.
It is not a fresh generalization estimate or official score. The frozen
[`r7 protocol`](../../experiments/drqv2-retention-r7.json) has SHA-256
`774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9`;
the existing [`r6 control protocol`](../../experiments/drqv2-geometry-mix-v1-r6.json)
has SHA-256 `d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab`.
The r6 source actors and all r4/r5/r6 artifacts were not replaced or rewritten.

## A. Executive Summary

**No arm established the requested source-success retention contract.** All
twelve frozen arms completed the same 32,768 decisions/22,768 updates; the
separate 384-episode TRAIN-DIAGNOSTIC finished with all 192 repeat pairs
identical. From 11 original successful source-actor/road cells, r7a retained
**1/3/1** and r7b **4/1/5** in uniform/failure-weighted/easy-retention order.
Each arm gained at least two successful **actor/road cells** formerly failed by
its corresponding source actor, but none approached the preregistered strong
signal of >=9 kept **and** >=2 gained. The corresponding final finishes are
r7a **3/10/5 of 32** and r7b **9/8/10 of 32**, versus unchanged-source 11/32
and existing r6 **1/4/3 of 32**. These totals are **not** model-selection or
generalization scores; in particular, failure-weighted r7a's 10/32 consists
of just three preserved source wins plus seven new wins. No setting changed
after seeing any source action or driving result.

The three requested direct answers are: **(1)** 50:50 source replay improves
some total finishes and gains but does **not** reliably prevent catastrophic
loss of original source successes (r7a still loses 8-10 of 11). **(2)** The
explicit action term reduces offline source-state action drift in all six
matched treatment pairs, and improves retention for uniform/easy-retention,
but **worsens** failure-weighted retention, so there is no uniform additional
finish benefit. **(3)** Preserving most of the 11 old successes *while* adding
new ones remains **unproven**: best observed retention is 5/11 with five gains.
These answers apply only to reused development roads. Nothing here authorizes
model promotion, held-out/confirmation/blind, or official evaluation.

## B. Hybrid Actor, Encoder And Critic Diagnosis

The offline [`hybrid result`](../../runs/20260926-drqv2-retention-r7/hybrid/first20-v2/result.json)
(SHA-256 `5614b8c2a1c7b261857c3fde4f0a2896b525ba217a4a50acf28c1931efd84369`)
replayed the source actors' **5,998 archived official actions** on the eleven
source-success actor/road cells; no hybrid action controlled the environment.
The action trace, 666 previous sampled source-observation instances, and each of
the cached first twenty observations per cell passed parity checks. The cache
[`first20-observations.npz`](../../runs/20260926-drqv2-retention-r7/hybrid/first20-v2/first20-observations.npz)
has SHA-256 `2679b63c5569b9c2b6a3ecbc9f3461563560a5e44b0581103e2366fa46dc7f28`.
It includes 220 float32 observations from 11 source-actor/road cells on nine
distinct roads, reused for **each** variant; those frames are serially dependent
and never enter training, preservation loss, or lambda selection. The stopped
`first20-v1` float32/uint8 parity failure remains preserved, not counted as
valid data.

Mean/median absolute native **steering** difference relative to full source
encoder + head on those same 220 frames per variant is below. Parentheses give
the number of frames with steering difference >=0.10; they are not independent
road successes. Source full actor is exactly zero by definition. Official
steering/gas/brake actions, native actions, all three axes' mean/median/threshold
values, and every observation-level value are retained in the hybrid result.

| Frozen variant | Full r6 encoder + head | Source encoder + r6 head | R6 encoder + source head |
|---|---:|---:|---:|
| Uniform | .2084 / .0032 (67/220) | .0913 / .0004 (51/220) | .2217 / .0074 (64/220) |
| Failure-weighted | .3667 / .0161 (87/220) | .2001 / .0008 (60/220) | .2316 / .0026 (72/220) |
| Easy-retention | .3136 / .0083 (84/220) | .1629 / .0003 (64/220) | .2473 / .0148 (86/220) |

The full r6 mean absolute native **brake** differences are .1950, .2737, .1963
for uniform, failure-weighted and easy-retention, respectively (>=.10 on
75/220, 98/220, 88/220 frames); the corresponding native gas differences are
.000016, .000136, .000016 (none >=.10). Official gas/brake map the native [-1,1]
actions to their environment action axes. These early measurements are highly
skewed: small medians coexist with substantial means and frequent large turns.

Across all three variant comparisons, the 11 source-success actor/road cells
cover five of the six families: two easy-curvature, three mid-road reversal,
one mid-road sustained, three opening-delayed and two opening-short-entry;
finish-approach has **zero** source-success cells in this cache. Pooling variant
comparisons only descriptively, full r6 versus source-encoder/r6-head versus
r6-encoder/source-head mean native steering gaps by family are: easy-curvature
.320/.149/.213, mid-road reversal .317/.161/.247, sustained .272/.202/.275,
opening-delayed .253/.115/.172, opening-short-entry .320/.170/.306. Each
family's 20-frame cells are repeated across three variants, not new roads.

Both hybrid swaps are valid DrQ-v2 actor encoder/head combinations. Neither one
uniformly reproduces the source actor, so both learned components can contribute;
their nonlinear interaction is not identified by this intervention. For example,
the source encoder + r6 head is relatively close on uniform, whereas changing
only the encoder while retaining the source head is not. This motivates, but
does **not** insert, encoder freezing into r7a/r7b; an isolated r7c is only a
future hypothesis.

The identical cache also supports matched Q1 and Q2 cross-forward evaluations
for source encoder + source critic head, source encoder + r6 head, r6 encoder +
source head, and r6 encoder + r6 head; both critics' four-way scores for each
source and r6 action are stored per frame in the same result. A descriptive Q1
ordering check of **source action versus r6 action** on the same 220 frames
per variant gives the counts below in that critic-combination order:

| Variant | Source E + source Q1 | Source E + r6 Q1 | R6 E + source Q1 | R6 E + r6 Q1 |
|---|---:|---:|---:|---:|
| Uniform | 127 | 111 | 110 | 80 |
| Failure-weighted | 145 | 109 | 120 | 79 |
| Easy-retention | 131 | 96 | 88 | 54 |

The analogous Q2 source-action preference counts are uniform 123/98/103/89,
failure-weighted 137/116/112/92, easy-retention 108/78/96/72. Q values under
different encoder/head parameterizations are not calibrated against each other;
these are same-input **orderings**, not ground-truth advantages, causal critic
explanations or official scores. The observed minority reversals cannot explain
all lost source finishes on their own.

## C. r7a Implementation And Exact Replay Ratio

[`retention.py`](../../haic/algorithms/drq_v2/retention.py) samples separately
from the existing source **training** checkpoint's rolling replay and the new
r7 online replay, then constructs **exactly 32 source + 32 online rows** for
every r6-sized batch of 64. Odd batch sizes are rejected rather than guessed;
the r6 batch size is fixed at 64. Pool-internal sampling is without replacement
per update. Source data are loaded from a SHA-pinned original checkpoint into
a separate read-only in-memory replay; there is no call to insert new rows into
it and no checkpoint is overwritten. r7a retains the original Q1 actor
objective, separate twin-critic/target updates, optimizer and trainable actor
and critic encoders; it has **no** preservation term or encoder freeze.

Source seed 0 replay is the original checkpoint SHA-256
`4248750c7114afdf955ea85a60c842565498aba3a64e380fa81164c7eb340979`,
episode ledger SHA-256 `d38f84661f92836a8d164f48d48cceb8e8c43c5f8e9a3f29f50257f1e52fe3e6`;
seed 1 checkpoint SHA-256
`c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4`,
episode ledger SHA-256 `568b07056916435bb53db0f3694a445f97728d4cb5fe2e316bdf4715788eeecd`.
Each replay contains the original 100,000 retained source-training transitions,
sequence IDs 31,072..131,071 and 99,994 valid 3-step starts. These original
source actions reflect an **evolving source-training actor**; they are neither
11 source-success-trajectory demonstrations nor the consumed r3 dataset. The
source actor teacher export is separately hash-pinned per seed.

All twelve [`run results`](../../runs/20260926-drqv2-retention-r7/), including
r7a and r7b, reached 32,768 additional decisions and exactly 22,768 updates
each, of which the first 10,000 decisions are update-free. Their receipts and
step/sample traces record **728,576 actual source slots and 728,576 online slots
per run**, or 8,742,912 each over all twelve arms. The independent, strictly
offline [`pre-evaluation semantic trace audit`](../../runs/20260926-drqv2-retention-r7/pre-evaluation-trace-audit-v1.json)
(SHA-256 `826e6fad1a58ea5880c25b7f3354ab6cf06122099d803f465f1d7b02c95c0db9`)
decoded every `(22,768,64)` trace and verified every per-update tag quota,
within-pool uniqueness, original source and online replay n-step/episode joins,
online availability by sampling time, TRAIN-only roads, and unchanged source
checkpoint bytes. This proves the **sampling contract**, not coverage of the
particular source-success states or driving performance. Relative to r6's 64
online rows per update, r7a also halves online sampled-row exposure while
keeping the update count, an important causal confound.

Each arm's own `run-config.json`, `result.json`, two checkpoint manifests,
`replay-sample-trace-step-*.npz`, `step-metrics.jsonl`, and `drift.jsonl` under
`runs/20260926-drqv2-retention-r7/learner-<seed>-<variant>-<condition>/`
record its protocol and executable SHA map, source actor/checkpoint/replay
identity, initial/final weight and checkpoint hashes, original source ledger,
online replay insertion ledger/hash, both sampled slot counts, target and
actor RNG seeds, lambda, optimizer/LR, encoder-update status, exact
decision/update counts and diagnostic trace SHA. All twelve final checkpoint,
sample-trace, insertion-ledger and run-result SHA values are independently
listed by arm in the linked trace-audit receipt. The original source actor
export SHA is seed 0
`433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37`
and seed 1
`c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954`.
The **initial r7 weight-only snapshot** `initial-weights.pt` has SHA-256
`be7cd0c47d9261fc4f1b31efa112eb81b9fe1f4fb52d394e4806fcd02195c67d`
for every seed-0 arm and
`c141a493dac9a1488da52da3b281056287022152a7a4ffab073e309358f43b42`
for every seed-1 arm; it is not an optimizer/replay-resumable full checkpoint.
Both optimizers are Adam, actor/critic LR each `1e-4`, with trainable actor
and critic encoders. The fixed 120-road TRAIN catalog SHA is
`a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b`.
The protocol pins the r7 learner code hashes (`retention.py`
`1c60e9fd951d9645ed1adf3a6db6ced628ba05a27d9a76e212a6a451062e9ff7`,
trainer `a34877ddba0275d639347dda72825af19732d82e01e05a02b2bf3fc490f8c6b2`)
in addition to all inherited r6 executable source SHA values. The offline
auditor code hash is
`b283580c92445702b34152c436f515cb1481ebcd1d6a6572cb138ec31239d726`.

## D. r7b Preservation Loss And Gradient Scale

r7b changes **only** r7a's actor objective: on the 32 source-replay states in
the batch, add `lambda_preserve * MSE(pi_r7(s), stopgrad(pi_source(s)))` in
deterministic **native action** space. The teacher export is the immutable
source actor corresponding to that learner seed, set to eval with no trainable
parameters or teacher optimizer. The ordinary DrQ Q1 actor objective still
uses the full, augmented 64-row batch; preservation uses unaugmented source
observations, so it constrains the actual observation-to-action function.

Before the first r7 training decision, the separately SHA-pinned
[`TRAIN-only gradient probe`](../../runs/20260926-drqv2-retention-r7/gradient-scale-v1.json)
(SHA-256 `3069899b49aaaaff7156ea2cdff9b054bb78ac0dd1765924dd3b5246abc79d18`)
used one 32:32 source/original r6 online diagnostic batch from **each** seed's
existing r6 **uniform** checkpoint. At that post-drift reference, native source
action MSE has a nonzero gradient; at the initial fork the two actors are
identical and a zero-lag preservation-gradient probe would be vacuous. There
were no new environment decisions or learner updates and no TRAIN-DIAGNOSTIC
observations or finish results in weight selection:

| Source seed | Q1 actor gradient L2 | Unweighted preservation gradient L2 | Resulting `lambda * preserve / Q1` |
|---|---:|---:|---:|
| 0 | 5.9091 | 5.5717 | .4714 |
| 1 | 32.2263 | 9.7967 | .1520 |

The **single preregistered rule** rounded `0.25 * median(Q-gradient / preservation-gradient)`
to the nearest power of two, fixing `lambda_preserve = 0.5` for **all six r7b
arms** before any r7 outcome. This balances a potentially material, non-dominant
term on two source seeds; it does not assert that these two batches predict
training-wide gradient proportions. r7a uses zero, and no post-result lambda
sweep or encoder freeze occurred.

## E. Source-Policy Drift During Training

The frozen first-20 cache was evaluated **forward-only**, never rolled out or
inserted into learner replay. The prespecified update checkpoints were
`0, 1, 2000, 6384, 11384, 14576, 22768`; the seven records in every run's
`drift.jsonl` are bound by its `result.json` diagnostic trace SHA-256. All twelve
actual file hashes match those receipts. Every variant begins at zero source
action difference at updates 0 and 1; the differences below already exist by
update 2000. This bounds when divergence became visible on this **early-frame**
cache, not its exact first update or driving consequence. Each entry pools 120
seed-0 and 100 seed-1 serially correlated frames of the **same eleven**
source-success actor/road cells; these are descriptive observation averages,
not 220 independent geometries.

| Variant / arm | Native action L1 at update 2000 | L1 at 11384 | L1 at 22768 | Final mean abs steering | Final any-axis >=.10 |
|---|---:|---:|---:|---:|---:|
| Uniform r7a | .224 | .342 | .308 | .172 | 114/220 |
| Uniform r7b | .194 | .144 | .193 | .122 | 86/220 |
| Failure-weighted r7a | .249 | .324 | .343 | .200 | 111/220 |
| Failure-weighted r7b | .260 | .169 | .228 | .108 | 102/220 |
| Easy-retention r7a | .308 | .305 | .328 | .211 | 107/220 |
| Easy-retention r7b | .143 | .246 | .197 | .087 | 101/220 |

At the final checkpoint the r7b **per-seed** native L1 gap is lower than its
corresponding r7a gap in all six arm pairs: uniform seed 0 `.289/.131`, seed 1
`.330/.267`; failure-weighted `.376/.181` and `.302/.284`; easy-retention
`.327/.161` and `.329/.241` (r7a/r7b). The last differences for seed 1 are
small. Final brake difference and >=.10 any-axis frequency also decrease in
each r7b-vs-r7a seed/mixture pair, though the remaining 86-102/220 large-axis
deviations in r7b show that copying source actions is **not exact**.

The actor-encoder source/current feature cosine is 1 at initialization and at
final update lies approximately `.839-.871` for r7a and `.858-.900` for r7b
across the six paired seed/mixture runs. Both current and frozen source Q1
values for **source action and current action** are also logged at every
prespecified checkpoint and by family, not used for a training decision. For
example, final seed-0/uniform r7a current Q1 means for source/current action
are `71.510/71.641`, while frozen source Q1 means are `57.918/57.857`;
seed-0/uniform r7b gives `71.410/71.471` and `57.918/57.913` respectively.
These are different critics and should not be equated as calibrated returns.
Neither representation cosine nor same-state Q ordering proves why a road is
finished or lost. The batch ledger and semantic audit fix the actual concurrent
source/online replay ratio at 32/32 throughout all 22,768 updates per run.

## F. Paired TRAIN-DIAGNOSTIC Results

The SHA-256 of the new
[`manifest`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/manifest.json)
is `0e4662102bd1d2f10536a04e9c7dfa8e48ef3952ba4d0782e6e4c89791835678`;
its accompanying [`summary`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/summary.json)
and [`paired cells`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/paired.json)
have SHA-256 `a127bdf1a179e97f134b43acd4e0a113de80106ee0e9ba1c0bfabcdf408d48f1`
and `8f10b5dae1701c5832079bfec2298fd8f9f916cffc5a839ea0a4669bea9d5f37`.
The CPU21 evaluator reused **exactly** the original 16 TRAIN-DIAGNOSTIC
roads, source-seed roles, track 1, obstacles, frame-skip 4, raw reward,
1200-decision limit and two repeats. It evaluated twelve candidate actors:
16 roads x 12 actor roles x 2 repeats = **384 episodes**. Repeat 1 duplicated
repeat 0 in all 192 actor/road pairs; this confirms deterministic replay, **not**
384 independent test roads. The 32 canonical repeat-0 source actor/road cells
contain 11 original successes and 21 source failures **per treatment**.
The evaluator's final executable SHA-256 is
`ce36a8c0c769f9e646224823207a92b1b7d9a398c45d80122070c01669e310f2`,
also bound inside its manifest. An independent read-only post-evaluation audit
verified **all 384 trace-file SHA values**, recomputed each trace's
deterministic digest, compared all 192 repeat pairs, and independently
recomputed all paired records and 126 summary groups against the original r6
and r7 episode traces. All matched; there are no extra or missing trace files.

| Frozen variant / condition | Both finish (kept) | Source-only (lost) | r7-only (gained) | Both fail | Final finishes / 32 |
|---|---:|---:|---:|---:|---:|
| Uniform r7a | 1 | 10 | 2 | 19 | 3 |
| Uniform r7b | 4 | 7 | 5 | 16 | 9 |
| Failure-weighted r7a | 3 | 8 | 7 | 14 | 10 |
| Failure-weighted r7b | 1 | 10 | 7 | 14 | 8 |
| Easy-retention r7a | 1 | 10 | 4 | 17 | 5 |
| Easy-retention r7b | 5 | 6 | 5 | 16 | 10 |

The source originally finished 6/16 cells at seed 0 and 5/16 at seed 1.
Seed-specific `kept / lost / gained / final` values show why pooling can
mislead: uniform r7a seed 0 `1/5/1/2`, seed 1 `0/5/1/1`; uniform r7b
`2/4/1/3` and `2/3/4/6`; failure-weighted r7a `2/4/4/6` and `1/4/3/4`;
failure-weighted r7b **`0/6/1/1` versus `1/4/6/7`**; easy-retention r7a
`1/5/3/4` and `0/5/1/1`; easy-retention r7b `2/4/4/6` and `3/2/1/4`.
These are paired actor/road cells, not twelve or 32 distinct geometries.

Family finish counts over both source seeds, one canonical repeat each, are
descriptive. Each variant's family denominator is `6,4,6,4,6,6` in the row
order below; sample sizes are too small for family generalization.

| Family | Uniform a / b | Failure-weighted a / b | Easy-retention a / b |
|---|---:|---:|---:|
| Easy-curvature anchor (n=6) | 1 / 3 | 3 / 1 | 2 / 5 |
| Finish-approach turn (n=4) | 0 / 0 | 1 / 0 | 0 / 0 |
| Mid-road reversal (n=6) | 1 / 1 | 2 / 1 | 1 / 0 |
| Mid-road sustained turn (n=4) | 0 / 2 | 2 / 1 | 1 / 2 |
| Opening delayed turn (n=6) | 0 / 1 | 0 / 2 | 0 / 3 |
| Opening short-entry turn (n=6) | 1 / 2 | 2 / 3 | 1 / 0 |

Here `a / b` denotes r7a/r7b *for the same frozen geometry mixture*, not a
comparison of geometry-family difficulty. The paired artifact contains the
complete per-family retained/lost/gained cell lists, visited-tile progress,
rewards and repeat trace hashes.

## G. Previously Successful Source Actor/Road Cells

The original source succeeded on **eleven actor-seed/road cells, on nine unique
road IDs**. The full six-arm keep/loss matrix is given below; `K` denotes the
same source-success cell still completed, `L` a regression. The first column
is the *source actor seed*, not the environment's geometry seed.

| Source seed | Road seed | Family | U-a | U-b | FW-a | FW-b | ER-a | ER-b |
|---:|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 3910800148 | Opening short-entry | K | K | L | L | K | L |
| 0 | 3910800153 | Opening short-entry | L | K | K | L | L | L |
| 0 | 3910800160 | Opening delayed | L | L | L | L | L | K |
| 0 | 3910800163 | Easy curvature | L | L | L | L | L | K |
| 0 | 3910800172 | Mid-road reversal | L | L | K | L | L | L |
| 0 | 3910800182 | Mid-road reversal | L | L | L | L | L | L |
| 1 | 3910800124 | Opening delayed | L | L | L | K | L | K |
| 1 | 3910800134 | Opening delayed | L | K | L | L | L | K |
| 1 | 3910800163 | Easy curvature | L | K | K | L | L | K |
| 1 | 3910800169 | Mid-road sustained | L | L | L | L | L | L |
| 1 | 3910800182 | Mid-road reversal | L | L | L | L | L | L |

`U`, `FW`, `ER` stand for the original uniform, failure-weighted and
easy-retention geometry mixtures, respectively; `a/b` are r7a/r7b. The full
family strings and SHA-linked actor/road trace pairs are in
[`paired.json`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/paired.json).
Roads `3910800163` and `3910800182` occur at **both** source seeds; counting
their two original source successes as two distinct roads would be wrong.

## H. New Successes On Source-Failure Cells

The gained actor-seed/road pairs below come exclusively from the 21 source-
failure **cells** per treatment. An entry is `source actor seed / road seed
(family)`, using `E` easy-curvature, `F` finish-approach, `R` mid-road reversal,
`S` mid-road sustained, `D` opening-delayed and `O` opening-short-entry.
These are reused TRAIN-DIAGNOSTIC roads, not newly allocated geometries.

| Treatment | Newly finished source-failure actor/road cells | Distinct gained road IDs |
|---|---|---:|
| Uniform r7a | `0/3910800175 (R)`, `1/3910800171 (E)` | 2 |
| Uniform r7b | `0/3910800169 (S)`, `1/3910800162 (S)`, `1/3910800171 (E)`, `1/3910800172 (R)`, `1/3910800187 (E)` | 5 |
| Failure-weighted r7a | `0/3910800164 (F)`, `0/3910800169 (S)`, `0/3910800175 (R)`, `0/3910800187 (E)`, `1/3910800153 (O)`, `1/3910800162 (S)`, `1/3910800187 (E)` | 6 |
| Failure-weighted r7b | `0/3910800134 (D)`, `1/3910800045 (O)`, `1/3910800148 (O)`, `1/3910800153 (O)`, `1/3910800162 (S)`, `1/3910800171 (E)`, `1/3910800172 (R)` | 7 |
| Easy-retention r7a | `0/3910800169 (S)`, `0/3910800171 (E)`, `0/3910800175 (R)`, `1/3910800187 (E)` | 4 |
| Easy-retention r7b | `0/3910800162 (S)`, `0/3910800169 (S)`, `0/3910800171 (E)`, `0/3910800187 (E)`, `1/3910800171 (E)` | 4 |

The repeated source-failure gain on **one** road is `3910800187` for the two
failure-weighted r7a actors, and `3910800171` for the two easy-retention r7b
actors. Some other gained cells occur on roads the **other** source actor had
already finished. The road IDs **neither source actor had finished** among each
treatment's gains are: uniform r7a `{3910800171, 3910800175}`, uniform r7b
`{3910800162, 3910800171, 3910800187}`, failure-weighted r7a
`{3910800162, 3910800164, 3910800175, 3910800187}`, failure-weighted r7b
`{3910800045, 3910800162, 3910800171}`, easy-retention r7a
`{3910800171, 3910800175, 3910800187}`, and easy-retention r7b
`{3910800162, 3910800171, 3910800187}`. These distinctions supplement,
not revise, the frozen >=9 kept plus >=2 gained **actor/road-cell** criterion.

## I. Existing r6 Control Versus r7a And r7b

The existing six completed r6 runs, **not** a new control run, retain the same
source actors, geometry variant schedule, training decision/update budget and
TRAIN-DIAGNOSTIC roads. This is a paired **actor/road evaluation** but not
identical sampled replay: r6 trains on 64 online replay rows per update, r7
replaces half of them with original-source checkpoint rows. The preexisting
r6 source-success lost/gained counts are uniform `11/1`, failure-weighted
`10/3`, easy-retention `10/2`.

| Variant | r6 kept + new = total | r7a kept + new = total | r7b kept + new = total |
|---|---:|---:|---:|
| Uniform | 0 + 1 = 1/32 | 1 + 2 = 3/32 | 4 + 5 = 9/32 |
| Failure-weighted | 1 + 3 = 4/32 | 3 + 7 = 10/32 | 1 + 7 = 8/32 |
| Easy-retention | 1 + 2 = 3/32 | 1 + 4 = 5/32 | 5 + 5 = 10/32 |

The r7b-versus-r7a comparison holds the 50:50 batch rule, initial weights,
mixture, budget and lambda selection protocol fixed; only the actor objective
term differs. The learned online trajectories subsequently diverge, as they
must for an interactive policy intervention. These are just two source actor
seeds on sixteen previously inspected roads; the six totals are not six
independent seeds or six separate holdouts.

## J. Evidence For And Against Replay-Retention Hypothesis

**Observed:** exact 32/32 quotas are verified for every update, and r7a gains
source-failure finishes on 2/7/4 actor/road cells compared with r6's 1/3/2.
Its aggregate finishes rise from r6's 1/4/3 to 3/10/5. The strongest r7a
retention is only failure-weighted **3 of 11** versus r6's 1 of 11; uniform
retains 1 versus 0, easy-retention **1 versus 1**. r7a still loses 10/8/10
of the 11 prior wins. The prespecified retention signal (>=9 kept + >=2
gained) and even the partial-retention band (9-11 kept) are not reached.

**Interpretation:** this falsifies the **sufficiency** of this original-source
replay 50:50 contract for preventing the observed catastrophic source-success
regression at the frozen budget. It does *not* establish replay dilution as the
sole r6 cause or show that source retention has no role: r7a replaces half of
r6's online samples with data collected by an evolving historical source
actor, rather than simply restoring identical warmup rows. In particular,
r6's reported roughly 77% to 35% warmup share refers to a **different**
within-online pool. No matched r7-runner 64-online-only control exists, and
online sample exposure per update also halves. Increased new-road success
and poor old-road retention can therefore coexist without a causal attribution
to one pool or critic mechanism.

## K. Evidence For And Against Explicit Actor Preservation

**Observed on the offline cache:** r7b's final native source-action L1 gap is
smaller than r7a's in **all six** same-seed/mixture pairs; the pooled early
cache's final mean native L1 is uniform `.308 -> .193`, failure-weighted
`.343 -> .228`, easy-retention `.328 -> .197`. This verifies an effect on
the *function being regularized* but not complete action identity.

**Observed on driving:** uniform retains **4 versus 1** old wins and gains
**5 versus 2** new wins in r7b/r7a; easy-retention retains **5 versus 1**
and gains **5 versus 4**. Failure-weighted, however, retains **1 versus 3**
despite lower action drift, with **7 versus 7** gains; seed-0 failure-weighted
r7b retains **zero of six** old wins and finishes just 1/16. Thus the added
loss improves retention in two of three mixtures, **not** uniformly across
mixtures or source seeds, and every r7b still loses at least six old wins.
The `11/11 retained but zero gained` over-preservation pattern is *not*
observed; every r7b has five to seven gained **cells**, yet behavior
preservation also does not supply a working retention contract.

**Hypotheses, not findings:** the old replay may miss key source-success
observations, actor-encoder/critic adaptations may perturb decisions outside
the cached first twenty frames, and stochastic interaction may magnify
modest source-state action deviations. Reducing action MSE can coexist with
loss of completion; no post-outcome lambda increase or model selection is
justified by these results.

## L. Remaining Uncertainty

- The diagnostic reuses only sixteen development roads, two source actors,
  and correlated frames and repeats. No fresh held-out/generalization inference.
- The fixed drift cache contains each source success's first **twenty** frames,
  not preselected middle/late source frames; offline agreement there does not
  guarantee action or Q agreement at later dangerous turns.
- The original source checkpoint replay comes from its evolving training
  policy, not final-source-success demonstrations. Per-batch source quota
  and nearly complete sampled replay-start coverage do **not** prove relevant
  source-success-state coverage.
- r7a versus r6 changes both source pool presence/provenance and the number
  of online rows sampled per update. r7b's 0.5 lambda was fixed using two
  short uniform TRAIN-only gradient batches, not a training-wide scale study.
- Hybrid encoder/head Q changes are representation-sensitive, noncausal
  same-state comparisons; neither source-action imitation nor Q ordering
  establishes why an environment trajectory terminates.
- Actor/road-cell gains should be distinguished from independent new road
  geometries. Family strata include just 2-3 roads each and there are only
  nine distinct originally source-success roads.

## M. Next Minimal Experiments (Not Executed)

1. **Offline source-state coverage and longitudinal drift audit:** reconstruct
   source-success *early, middle and late* frames using the already consumed
   parity-checked traces; audit how closely the immutable source-training
   replay and r7's sampled rows cover those states and source-action behavior.
   No new environment interaction, learner update, or protected partition.
2. **Separately frozen provenance-only replay treatment:** compare the
   original evolving source checkpoint replay with an explicitly attributable
   final-source-policy TRAIN replay pool, preserving the same 32:32 quota,
   r7 learner, fixed lambda and budget. Predeclare the pool's capacity,
   terminal-safe 3-step sampling and action/noise identity; do not substitute
   consumed TRAIN-DIAGNOSTIC success observations. Check coverage **before**
   any training, and treat it as a different source-data variable.
3. **Isolated r7c encoder-freeze candidate:** only if the offline coverage
   evidence supports encoder drift as the remaining bottleneck, freeze the
   actor encoder under a separately frozen, same-budget 50:50 treatment;
   do not combine it silently with a new source pool or lambda sweep. A
   separate matched 64-online sampler-control is also needed before a
   replay-specific causal attribution; it is **not** part of these runs.

No next experiment, fresh screen/confirmation/blind evaluation, official
score lookup, or model promotion has been started or authorized by the
TRAIN-DIAGNOSTIC results above.

### Reproduction Commands

From the repository root, the input/preflight, one-arm training, and final
diagnostic CLIs were:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/kilo/haic-cpu21/bin/python -B -m scripts.diagnose_drq_r7_hybrids --output-root runs/20260926-drqv2-retention-r7/hybrid/first20-v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m scripts.probe_drq_r7_gradient --output runs/20260926-drqv2-retention-r7/gradient-scale-v1.json
.venv/bin/python -B -m scripts.train_drq_retention_r7 --protocol experiments/drqv2-retention-r7.json --preflight-only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.train_drq_retention_r7 --protocol experiments/drqv2-retention-r7.json --source-seed 0 --variant uniform --condition r7a
.venv/bin/python -B -m scripts.audit_drq_retention_r7
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/kilo/haic-cpu21/bin/python -B -m scripts.diagnose_drq_retention_r7 --protocol-sha256 774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9 --output-root runs/20260926-drqv2-retention-r7/train-diagnostic
```

Substitute source seed `0/1`, mixture `uniform/failure_weighted/easy_retention`
and condition `r7a/r7b` for the other eleven **separately initialized** arms;
the protocol's exact twelve-row matrix records their RNG seeds and run paths.
These CLIs intentionally refuse to overwrite existing outputs. Reproduction
requires an isolated checkout with the pinned source checkpoints/catalog,
the pinned GPU training and CPU21 evaluation runtimes, and new output paths
under a separately frozen protocol; the listed commands document the original
execution and are not invitations to rerun on protected cells. No r6 control
run was retrained or reinterpreted.
