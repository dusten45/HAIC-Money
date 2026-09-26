# DrQ-v2 Geometry-Mix r6 Regression Diagnosis

**Scope:** reused, unranked TRAIN-DIAGNOSTIC evidence only. All 16 roads, both
source actors and the six r6 final actors were evaluated *before* this diagnosis.
The paired analysis is entirely offline. One additional, parity-checked diagnostic
replayed only the eleven already-successful source-action traces on those same
roads (5,998 recorded decisions); no alternative policy controlled a car, and
there were no learner updates, fresh evaluation cells, or held-out/blind access.
The 11 source finishes are 11 of **32 source-actor/road cells**, repeated for
each variant; the 96 paired rows are not 96 independent observations.

## A. Key Conclusions

1. The matched collapse is predominantly **lost source successes**, not absence
   of newly acquired successes: uniform lost 11 and gained 1, failure-weighted
   lost 10 and gained 3, easy-retention lost 10 and gained 2. Only one source
   success was retained by each of the latter two variants. The net changes
   from the same 11/32 source cells are -10, -7 and -8 finishes, respectively.
2. All 31 lost-success pairs first have a sustained native-action discrepancy
   within decisions 1-8 (predeclared >=0.10 on some axis for three consecutive
   decisions). On *identical source observations* at the first decision, 5/11,
   4/10 and 7/10 lost cells already differ by >=0.10. Steering and brake differ;
   gas is effectively saturated at +1 in the first 20 source-success states.
   All 31 losses exceed the **same-state** >=0.10 gap by decision 5 on the
   parity-checked source-action replay frames. Thus the trained policy function
   has changed, but the remainder of the closed-loop divergence can include
   amplification of earlier smaller differences; large drift also occurs on
   retained wins.
3. Both actor and critic encoders, not just a policy head, changed. Across
   same-source-state samples, median actor-encoder feature cosine is 0.794,
   0.783 and 0.767 for the three variants. This change also occurs in the two
   retained source-success cells; representation drift is **observed**, not
   identified as the cause of their different outcomes.
4. Early source-Q1-versus-r6-Q1 **preference reversal** is heterogeneous, not a
   universal explanation. A fixed >=10/20-early-states reversal marker holds
   in 0/11, 2/10 and 3/10 losses. Several individual reversals and local
   steering-Q slopes are compelling leads, but source and r6 Q values are
uncalibrated estimators rather than counterfactual returns; Q1 trained the
  actor, while Q2 and twin-min also must be inspected.
5. Source-style state retention was **not guaranteed**: every run used online
   replay only, 10,000 exploratory update-free warmup decisions, then 22,768
   updates. Early transitions were not evicted (32,768 < capacity 100,000), but
   their sampled share declined from about 77% to 35% over the update window.
   Individual lost diagnostic states cannot be joined to TRAIN replay: the
   TRAIN and TRAIN-DIAGNOSTIC road sets are disjoint and replay provenance has
   no observation images. Inadequate retention is a plausible contributing
   mechanism, not a demonstrated per-episode cause.

## B. Paired Source-to-r6 Transitions

The canonical repeat-0 transition table pairs the **same road, track 1 and
source learner seed**. `Lost` is source finish -> r6 failure; `gained` is source
failure -> r6 finish; `kept` is both finished; `neither` is both failed.

| Variant | Source finished | Lost | Gained | Kept | Neither | r6 finishes |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 11/32 | 11/32 | 1/32 | 0/32 | 20/32 | 1/32 |
| Failure-weighted | 11/32 | 10/32 | 3/32 | 1/32 | 18/32 | 4/32 |
| Easy-retention | 11/32 | 10/32 | 2/32 | 1/32 | 19/32 | 3/32 |

Family breakdown, **lost/gained/kept/neither**, all numbers over the n paired
source-actor/road cells *per variant* (not percentages of the 11 successes):
Each cell's four rates are its four displayed counts divided by the row's `n`;
for example `2/1/0/3` with `n=6` means 33.3% lost, 16.7% gained,
0% kept and 50% neither, in that order.

| Family | n | Source finished | Uniform | Failure-weighted | Easy-retention |
|---|---:|---:|---:|---:|---:|
| Easy-curvature anchor | 6 | 2/6 | 2/1/0/3 | 2/1/0/3 | 2/0/0/4 |
| Finish-approach turn | 4 | 0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 |
| Mid-road left-right reversal | 6 | 3/6 | 3/0/0/3 | 2/1/1/2 | 3/1/0/2 |
| Mid-road sustained/same turn | 4 | 1/4 | 1/0/0/3 | 1/0/0/3 | 0/1/1/2 |
| Opening delayed high turn | 6 | 3/6 | 3/0/0/3 | 3/1/0/2 | 3/0/0/3 |
| Opening short-entry left turn | 6 | 2/6 | 2/0/0/4 | 2/0/0/4 | 2/0/0/4 |

Every one of the 11 source-success cells, with `L` lost, `K` kept, followed by
**first sustained action discrepancy decision / candidate maximum progress**:

| Source seed / road | Family | Source max progress | Uniform | Failure-weighted | Easy-retention |
|---|---|---:|---|---|---|
| 0 / 3910800148 | short-entry | .962 | L 1/.935 | L 1/.702 | L 1/.329 |
| 0 / 3910800153 | short-entry | .952 | L 1/.456 | L 1/.338 | L 1/.607 |
| 0 / 3910800160 | delayed-opening | .952 | L 6/.182 | L 4/.586 | L 8/.134 |
| 0 / 3910800163 | easy-anchor | .961 | L 1/.273 | L 2/.514 | L 1/.149 |
| 0 / 3910800172 | reversal | .985 | L 5/.939 | K 1/.967 | L 5/.473 |
| 0 / 3910800182 | reversal | .997 | L 1/.317 | L 2/.287 | L 1/.160 |
| 1 / 3910800124 | delayed-opening | 1.000 | L 7/.456 | L 5/.812 | L 4/.912 |
| 1 / 3910800134 | delayed-opening | .962 | L 3/.137 | L 1/.510 | L 3/.916 |
| 1 / 3910800163 | easy-anchor | 1.000 | L 4/.830 | L 4/.273 | L 1/.582 |
| 1 / 3910800169 | sustained-turn | .971 | L 4/.307 | L 4/.386 | K 1/.993 |
| 1 / 3910800182 | reversal | .987 | L 4/.317 | L 4/.403 | L 1/.800 |

Gained-success cells are uniform `(source 0, road 3910800171)`;
failure-weighted `(0, 3910800124)`, `(0, 3910800171)`, `(0, 3910800175)`;
and easy-retention `(1, 3910800162)`, `(1, 3910800172)`. All other
source-failed cells are accounted for by `neither`. Each paired row and its
source and r6 trace SHA-256 appears in
[`paired-regression-v1.json`](../../runs/20260925-drqv2-geometry-mix-v1-r6/paired-regression-v1.json).

## C. First-Divergence Analysis

Native action is `(steer, gas, brake)` in `[-1,1]`; `official_action` applies
the per-axis action adapter (official gas/brake in `[0,1]`). The diagnostic
uses **no temporal action smoothing**. The deterministic DrQ actor outputs
`tanh(logits)`, not a fitted probabilistic output distribution. The first
*sustained action discrepancy* is the first of three consecutive common
decision indices with native-action maximum absolute axis difference >=0.10.
This threshold and the ten-decision preceding action history were recorded
before inspecting paired outcomes. The analyzer also retains the first
numerical difference >1e-6 and the full pre/post action, reward, progress,
damage, collision, off-track-counter, terminal and sparse road trace.

Among lost pairs the sustained discrepancy occurs at decisions 1-7 for uniform,
1-5 for failure-weighted, 1-8 for easy-retention. The very first numerical
difference is at decision 1 in 30/31 lost pairs but may be negligible; in the
entire 96-pair grid 36 initial differences are <.01. Conversely, a single
>=.10 action difference can precede the three-step landmark. The preceding
ten-action history is **not** a shared-control prefix: earlier actions already
differ in some cases. Reward/progress at decision `t` are post-action values,
not pre-action state. After the first nonzero action difference, index-aligned
closed-loop traces no longer guarantee identical frames; the landmark is not
proved to be the earliest causal accident or road location.

Lost r6 trajectories reached lower maximum visited-tile progress in all 31
matched cases. Mean paired **loss-only** candidate-minus-source max-progress
changes were -0.507, -0.493 and -0.470 (uniform, failure-weighted,
easy-retention); terminal decision count was earlier in 8/11, 9/10 and 8/10.
Higher decision count in the others is not evidence of better road progress.
Four lost cases reached >=.90 max progress (uniform two, easy-retention two)
but did not finish; most losses were not finish-line-only failures.
All eleven uniform losses retired via the wrapper's `off_track` condition
(a sustained negative-reward counter, not a proven geometric road exit);
failure-weighted and easy-retention each had eight `off_track` and two crash
retirements. Median final damage in these paired losses was source 0 versus
r6 .2/.4/.2, respectively. Mean paired raw-reward deltas were -437/-420/-397;
raw reward, progress and damage remain internal proxies, **not** HAIC scores.

Two contrasting complete cases:

- `source 0 / road 3910800163 / uniform`: identical reset input, first native
  source/r6 steer `-.986/+ .848`, gas approximately `1/1`, brake
  `-.999/-.999` (official brake approximately `.0005/.000005`). Both step-1
  rewards `-.4`, progress `.0071`, damage `0`, off-track counters `1`;
  source/r6 ultimately finish/fail at 580/232 decisions with max progress
  `.961/.273`. Source replay reports pre-action nearest sampled-road-point
  distance `.00038 m`,
  heading error `.0057 rad`; these do not localize the later off-track event.
- `source 1 / road 3910800163 / failure-weighted`: first same-state steer
  `1/.999` (gap `.001`); sustained discrepancy begins at decision 4. At that
  decision source/r6 native steer `.460/.645`, brake `-.717/-1.000` (official
  brake `.142/~0`); both receive raw reward `-.4`, progress `.0106`, damage
  `0` and off-track counter `1`. Source pre-action nearest sampled-road-point
  distance is
  `1.709 m`, heading error `-.084 rad`; candidate heading was not recorded.
  At decision 5 the r6 actor on the **source state** steers `.595`, whereas
  its actual already-diverging trajectory steers `.254`: action differences
  and state-feedback differences have begun to separate. Max progress ends
  `1.000/.273` and the candidate retires off-track at decision 227, versus
  the source finish at 474.

Original sparse road telemetry is at step 1 and every 25 steps (and episode
end), so the landmarks at steps 1-8 have only the step-1 candidate road sample.
The 11 exact source-action replays add *source-only* pre-action position,
distance to the nearest sampled road point and road-tangent-relative heading
on sampled observations. The artifact's `centerline_distance_m` field is
Euclidean distance to the nearest discrete track point, **not** lateral
distance to a projected road segment or proof of off-road travel. They do not recover
the r6 actor's unrecorded candidate-state heading.

## D. Actor Drift Evidence

The exact source-action replays reproduce **all 5,998 archived source steps**:
official action, reward, progress, damage, off-track counter, terminal and
sparse road telemetry match the original records. At decisions 1-20 plus every
25th and the final 20, source and all three r6 actors are evaluated on the
same source frame (666 sampled frame instances, 664 unique observation hashes,
1,998 variant-frame records). Initial
candidate-model actions reproduce their independently recorded diagnostic
step-1 actions. No alternative actor drives this replay.

| Lost-case same-source-state metric, decisions 1-20 | Uniform (11 roads) | Failure-weighted (10) | Easy-retention (10) |
|---|---:|---:|---:|
| First action native max-axis gap >=.10 | 5/11 | 4/10 | 7/10 |
| Same-source-state max-axis gap >=.10 by decision 5 | 11/11 | 10/10 | 10/10 |
| Mean absolute steering difference | .208 | .363 | .311 |
| Mean absolute gas difference | ~0 | ~0 | ~0 |
| Mean absolute native brake difference | .195 | .255 | .194 |
| Initial steering sign changed | 4/11 | 1/10 | 4/10 |

These measure **functional** change on the baseline's actually visited states;
parameter norms alone cannot establish forgetting. In other losses the first
action is similar yet actions grow apart on the same source observations within
five decisions; the two closed loops then amplify differences. The t4/t5 case
above illustrates
both policy drift and its subsequent state-feedback effect. No heading-aligned
common policy inputs exist later on the r6 trajectory, so this cannot prove
that all early failures were caused by an actor-only defect.

## E. Critic Q Reordering Evidence

For each sampled source-success state `s`, `a_s` is the source deterministic
action and `a_r` is the variant's deterministic action on *the same `s`*.
The 2x2 comparison evaluates source critic Q1/Q2 and r6 critic Q1/Q2 on
**both** `a_s` and `a_r`. DrQ actor updates optimize online critic **Q1**, not
the twin minimum; Q2 and `min(Q1,Q2)` are recorded as uncertainty checks.

| Early lost-case Q1 evidence | Uniform | Failure-weighted | Easy-retention |
|---|---:|---:|---:|
| Source-Q1 prefers `a_s`, r6-Q1 prefers `a_r` (source states) | 63/220 | 78/200 | 75/200 |
| Roads with that reversal on >=10 of first 20 states | 0/11 | 2/10 | 3/10 |
| Q2 reversal on same early states | 47/220 | 59/200 | 54/200 |
| Twin-min reversal on same early states | 62/220 | 80/200 | 73/200 |
| Median within-state r6-Q1 minus source-Q1 **at `a_s`** | +9.97 | +5.63 | +8.75 |
| Median of lost-road medians: r6-Q1 minus source-Q1 **at `a_s`** | +10.44 | +5.45 | +10.86 |
| Median of lost-road medians: r6-Q2 minus source-Q2 **at `a_s`** | +10.84 | +5.38 | +10.63 |
| Median source/r6 twin gap at their own actions | 1.376/1.154 | 1.376/1.314 | 1.278/1.129 |

Those per-decision denominators are serially correlated and should not be used
for significance. Absolute critic levels can shift without calibrated
improvement: same-action Q increases do **not** mean actual returns increased.
No general increase in twin disagreement is observed here.

Two concrete early Q1 comparisons, expressed as
`Q_source(a_s), Q_source(a_r) ; Q_r6(a_s), Q_r6(a_r)`:

- Lost `uniform / source 0 / road 3910800163`, decision 1:
  `55.231, 55.030 ; 68.084, 68.742`. Source-Q1 orders source action above
  variant; r6-Q1 reverses. Source Q2 is `55.380,55.412` (not the same order),
  r6 Q2 is `69.257,69.587`; the twin minimum shares the Q1 reversal.
- Lost `failure-weighted / source 1 / road 3910800163`, decision 4:
  `69.373,69.353 ; 70.916,71.700`. Source Q2 is `69.663,69.707`,
  r6 Q2 `73.240,74.068`. The action/critic change is substantial by t4,
  whereas both first-step Q1 rankings and actions were nearly identical.

Steering-only +/-0.10 native-action probes at the frozen steps 1, 25 and the
first of the final 20 show that at step 1, Q1's *local* slope from `a_s` toward
the r6 steering direction opposes that direction under the source critic and
favors it under the r6 critic in 4/11, 4/10 and 3/10 losses. This is a bounded
local landscape, **not** an accurate return oracle or an actor/critic causal
intervention. As counterevidence, `failure-weighted / source 0 / road
3910800172` has a large initial actor-action gap (1.53) and 10/20 early
Q1 reversals, yet **both actors finish**. A ranking reversal is neither
necessary nor sufficient for a lost finish in this cohort.

## F. Replay and Distribution Shift Evidence

All six runs start from the source weight-only fork and contain only catalog
TRAIN interactions, not source demonstrations or TRAIN-DIAGNOSTIC roads.
`step-metrics.jsonl` records **inserted decisions**; the final frozen replay
NPZ contains **sampled n-step starts**, `(22,768 updates x 64)=1,457,152`
slots per arm (8,742,912 total). The 6,384-update checkpoint NPZ is a
cumulative prefix, never added to the final sample. Every sampled sequence
index maps to a prior same-run insertion/episode/seed/track and every insertion
has contiguous metadata; no teacher/source-code transitions were loaded.

The independent, previously measured TRAIN source diagnostic has 31/120 roads
finished once by source 0 and 23/120 by source 1 on **track 1 only**. Same
source-seed/road/track-1 replay episodes are an *exact geometry-plus-obstacle
cell*, not the same policy observation, because r6 had exploration, ongoing
updates and a different episode cap. Tracks 2-4 match road geometry only.

| r6 source seed / variant | Prior source-success TRAIN roads encountered / available | Exact track-1 episodes | Source-success geometry inserted decisions /32,768 | Online completed finishes / ended episodes | Anchor / delayed-opening inserted share |
|---|---:|---:|---:|---:|---:|
| 0 / uniform | 19/31 | 5 | 8,779 | 17/72 | 24.5% / 16.8% |
| 0 / failure-weighted | 20/31 | 6 | 9,407 | 10/74 | 25.0% / 18.9% |
| 0 / easy-retention | 14/31 | 7 | 10,396 | 13/67 | 58.8% / 1.5% |
| 1 / uniform | 15/23 | 5 | 6,555 | 12/82 | 21.4% / 17.0% |
| 1 / failure-weighted | 20/23 | 6 | 8,870 | 4/83 | 24.2% / 18.3% |
| 1 / easy-retention | 15/23 | 8 | 7,846 | 6/84 | 49.0% / 10.4% |

The exact-track-1 inserted decision counts, not just episode counts, and
per-family insertion shares over four fixed 8,192-decision windows and sample
shares over the three windows with updates (starting at decision 10,001) are
in [`replay-distribution-v1.json`](../../runs/20260925-drqv2-geometry-mix-v1-r6/replay-distribution-v1.json).
Easy-retention concentrated on the anchor: source 0 inserted 19,269/32,768
decisions on that family, but only 479/32,768 on delayed-opening and
1,419/32,768 on sustained-turn. These are **length-weighted decision** shares,
not violations of frozen family-per-reset weights. Those episode-draw targets
were uniform `1/6` each; failure-weighted anchor `.20`, sustained and finish
`.125` each, the remaining three `.55/3` each; easy-retention anchor `.45`,
finish `.15`, the other four `.10` each. Short episodes and the sampler's seed
cycle affect exposure. In late updates, warmup-origin sample
share is 35.1-35.2%, down from 77.3-77.4% in the first update window.
Capacity 100,000 exceeds all 32,768 insertions, so this is **dilution, not
erasure** of warmup rows. A noisy source-initialized actor on a TRAIN road
also does not guarantee a source-success *state* in that warmup replay.

Episode ledgers provide only *terminal*, not dense/maximum progress and damage.
Across arms, completed-episode finish and terminal-high-progress counts vary;
20 inserted decisions before each completed off-track retirement are only a
retrospective near-terminal proxy, not an observed off-road-state flag. Exact
source-success TRAIN-DIAGNOSTIC state overlap is unidentifiable from replay
metadata. The proposition that 32,768 decisions suffice to change the actor
but are insufficient for robust new-geometry adaptation is **plausible** given
the many lost finishes and few gained ones, but an alternative frozen budget
was not run and this study cannot identify the causal horizon.

## G. Encoder and Representation Change

Across the six source/variant checkpoint pairs, parameter relative L2 changes
are actor encoder **0.200-0.212**, actor trunk **0.230-0.256**, actor policy head
**0.085-0.110**, critic Q1 encoder **0.258-0.275**, Q2 encoder **0.257-0.278**,
and Q1/Q2 heads approximately **0.235-0.292**. These are parameter-space norms,
not perceptual distances or performance measures. On identical sampled source
observations, median actor-encoder feature cosine is 0.794 / 0.783 / 0.767
for uniform / failure-weighted / easy-retention, with Q1-encoder cosine
0.798 / 0.790 / 0.798. The predeclared `<.90 in >=10/20 early states`
actor-feature-shift marker occurs in **all 31 lost cases**, but also in
**both retained-source-success cases**. Thus co-drift is clear, while encoder
causality versus downstream actor-head change remains unseparated. No frozen
hybrid encoder/head actor was evaluated.

## H. Family Collapse Pattern

`opening-short-entry-left-turn`: source succeeded twice out of six paired
cells, and **each variant lost both**; all three finished zero. By contrast,
`finish-approach-turn` already had **zero source finishes out of four** and
all three variants remained at zero, so this family contributes no observed
source-success loss. Reversal drops from source 3/6 to uniform 0/6,
failure-weighted 2/6 and easy-retention 1/6; delayed opening drops from 3/6
to 0/6, 1/6 and 0/6. The sustained-turn family (source 1/4) is the sole
family where easy-retention both kept one win and gained another (2/4), but
it does not rescue that variant's other ten lost wins.

Failure-weighted's four wins comprise two reversal, one easy-anchor and one
delayed-opening; all **four** belong to source seed 0, whereas seed 1 finished
0/16. It kept one reversal source win and gained three previously failed
cells, also all seed 0. Easy-retention finished 0/16 at source seed 0 and
3/16 at source seed 1. On 16 reused roads/two correlated source actors,
neither a reproducible family effect nor chance can be ruled out.

The requested seven-category frequency audit below is **not a causal
classification**. Positive rows are nonexclusive, predeclared descriptive
markers among the **lost source-success cells**; lack of evidence for a
category does not refute its mechanism. `N/A` explicitly means the preserved
data cannot identify that mechanism *per lost episode*.

| Failure category / operational evidence | Uniform lost n=11 | Failure-weighted lost n=10 | Easy-retention lost n=10 |
|---|---:|---:|---:|
| 1. Early policy drift, same-reset native action gap >=.10 | 5/11 | 4/10 | 7/10 |
| 2. Critic Q1 mis-ranking **marker**, reversal on >=10/20 early source states | 0/11 | 2/10 | 3/10 |
| 3. Representation shift **marker**, actor-encoder cosine <.90 on >=10/20 | 11/11 | 10/10 | 10/10 |
| 4. Insufficient source-**state** retention, per diagnostic road | N/A | N/A | N/A |
| 5. New-geometry adaptation failure *within lost source-success cells* | N/A | N/A | N/A |
| 6. Finish/late-stage marker, r6 fails despite max progress >=.90 | 2/11 | 0/10 | 2/10 |
| 7. Other: no positive marker | 0/11 | 0/10 | 0/10 |

Category 5 can only be described separately for **source-also-failed** cells:
uniform 20/32, failure-weighted 18/32, easy-retention 19/32; neither actor
finished, which does not establish that r6 newly failed to adapt. Category 7
counts zero merely because all lost cells have the non-discriminating feature
shift marker; **the actual causal explanation remains undetermined for all
31 lost variant/road pairs**. For category 4, even exact TRAIN track-1 source-
success road exposure cannot identify corresponding disjoint TRAIN-DIAGNOSTIC
images. No post-outcome threshold was adjusted to make a marker frequent.

## I. Mechanism Ranking

1. **Actor function/representation co-drift (strong observation, incomplete
   causal separation).** Exact first-state and later same-source-state action
   comparisons, loss timing, actor-head/encoder changes and action-replay
   parity directly show changed control. Which network component produced the
   harmful change is not identified; tiny first actions may be amplified by
   dynamics and new observations.
2. **Training-state coverage and family-specific online distribution (moderate
   structural support, no exact lost-state overlap).** Only a small subset of
   prior TRAIN source-success roads got exact track-1 exposure, warmup samples
   were diluted, and easy-retention markedly favored anchors. This can permit
   forgetting but cannot explain each disjoint diagnostic cell causally.
3. **Critic Q1 reordering (specific, limited support).** Strong first-state
   examples and 2/10 plus 3/10 persistent early lost-road markers implicate
   value ordering in some cases; uniform has 0/11 persistent markers despite
   losing every prior success. Q2/twin-min and a successful reversed control
   constrain a critic-only explanation.
4. **Insufficient adaptation horizon and finish-specific failure (unresolved
   horizon, small finish subset).** One to three new successes per variant
   cannot compensate for ten or eleven losses; no different budget was tested.
   Only four of 31 losses reached >=.90 max progress, so a finish-only defect
   is not a general explanation.

## J. Counterevidence and Remaining Uncertainty

- Source itself failed 21/32 cells; reused 16 roads are not fresh validation.
  Outcomes share actors and roads, so no independent 96-cell significance or
  official/generalization ranking is available.
- All initial same-state action differences are **not** large (15/31 lost
  cases have first-decision gap <.10). First sustained divergence is a
  predeclared descriptor, not the first physical/causal branch. Only source
  trajectories have recovered dense pre-action road heading; candidate
  trajectory after divergence does not share those states.
- Feature shift also occurs on the two maintained successes; an encoder
  similarity threshold cannot prove catastrophic forgetting. Q1 reversal
  also occurs in a maintained success, and Q estimates are not validated
  returns or counterfactual actor-branch outcomes. Twin disagreement does not
  systematically rise. Some Q sign reversals have very small margins near
  a tie; the prespecified sign test did not tune a margin to outcomes. No
  actor-only/critic-only/encoder-only causal swap
  was run or justified as new evaluation.
- Both maintained source-success cells also have large initial same-state
  action gaps (1.530 and .374). An early actor change is observed in every
  lost cell by decision 5, but is not sufficient on its own to predict failure.
- Archived diagnostic traces did not save observation pixels. Replaying the
  same official actions under the pinned CPU21 runtime reproduces every saved
  simulator telemetry value and sampled source actor output, but cannot prove
  byte-for-byte pixel parity with the original unsaved frames. Each newly
  recovered frame is separately SHA-256-hashed in the same-state record.
- TRAIN replay stores geometry, track, episode and n-step sample indexes, **not
  frame-level state identity, dense training progress/damage or demonstrator
  success**. Joining prior source-success labels to tracks 2-4 is geometry
  only; joining to track 1 is still a different, exploratory/updating policy.
  Terminal-high-progress bins are not in-episode progress bins.
- No parameter, family weight, seed, source, protocol, previous r4/r5/r6
  output or official environment was changed. The first attempted model probe
  failed at CPU21 `mmap` argument preflight *before any reset*; its
  [`v1 failure receipt`](../../runs/20260925-drqv2-geometry-mix-v1-r6/source-state-replay-v1/failure.json)
  is preserved (0 decisions). The corrected `v2` performed the one bounded
  5,998-step replay, not a repeat experiment or fresh cohort.

## K. Next Minimal Uncertainty-Removal Experiments

These are **proposals only**, not authorized training/evaluation work or model
selection. Freeze any new tool/inputs before running; continue to use consumed
TRAIN-DIAGNOSTIC roads only, no new validation cells.

1. To separate encoder from actor-head drift, capture at most **11 x 20 = 220**
   early observations by replaying recorded source actions on the same 11 roads
   once, parity-check each step, then evaluate only offline frozen hybrid
   `source encoder + r6 trunk/head` versus `r6 encoder + source trunk/head`.
   Compare both to source/r6 actions on exactly those frames. No hybrid rollout
   or training; this directly tests which component changes early controls.
2. On that same <=220-frame frozen cache, score both source and r6 critics on
   hybrid/source/r6 actions and the same predeclared +/-0.10 steering probes.
   Check whether observed Q1 preference and Q2 disagreement persist when the
   encoder/head component alone changes. This tests the largest remaining
   actor-versus-critic ambiguity without an environment action.
3. Without a reset, inspect a bounded, protocol-hashed sample of existing
   checkpoint replay observation tensors against source replay tensors using
   the **frozen source encoder**. Count source-like feature-neighbor coverage
   by insertion time/family/track and compare to the geometry-only proxies
   above. Do not call feature-neighbor similarity ground-truth source-state
   retention; use it to test whether exact image-level coverage is worth a
   later, separately justified experiment.

## Provenance and Reproduction

The [frozen r6 protocol](../../experiments/drqv2-geometry-mix-v1-r6.json) SHA-256 is
`d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab`;
[TRAIN catalog](../../runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json)
`a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b`;
original 256-episode [TRAIN-DIAGNOSTIC manifest](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json)
`fb14fe9eab14f61cdecc45453b69e44d34fdb967368c44c255e827292804b380`.
The original source checkpoints are seed 0
`4248750c7114afdf955ea85a60c842565498aba3a64e380fa81164c7eb340979`
and seed 1
`c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4`.
All 27 frozen runtime source files were checked before source-action replay.
Each run's `result.json` binds both checkpoints, the final actor and both
replay-sample trace hashes; the [active plan](../plans/active/drqv2-geometry-mix-plan.md)
lists all six final checkpoint/replay SHA-256 pairs. The derived audit files
also record each r6 checkpoint, replay sample, 32,768-row insertion ledger,
episode ledger, individual diagnostic trace and/or sampled observation hash.
The paired reader was subsequently hardened with separately pinned diagnostic
manifest/episodes hashes and path containment; its current source reproduced
the preserved paired JSON **byte-for-byte** without changing any cutoff.

Artifact SHA-256 values:

| Derived read-only evidence | SHA-256 |
|---|---|
| [`paired-regression-v1.json`](../../runs/20260925-drqv2-geometry-mix-v1-r6/paired-regression-v1.json) | `29b6688413738f5a0ae7faaa950d202c53c73c1c14735f9e205d3ec01ed2cdd7` |
| [`replay-distribution-v1.json`](../../runs/20260925-drqv2-geometry-mix-v1-r6/replay-distribution-v1.json) | `2fbfcef1cb7c19bc9ae6aa334df75f0e6c57d490485cfc8576ce99ded4f25135` |
| [`source-state-replay-v2/preflight.json`](../../runs/20260925-drqv2-geometry-mix-v1-r6/source-state-replay-v2/preflight.json) | `b2fe774cc756a39fc2a6c9d586f7d11bab0085f9b98a012cb2a3e00cc14be431` |
| [`source-state-replay-v2/result.json`](../../runs/20260925-drqv2-geometry-mix-v1-r6/source-state-replay-v2/result.json) | `2c1e1a3ce21137c00bc2042d069a9a410b77f92eb76bac3334c2c548888c07cc` |
| [`regression-attribution-v1.json`](../../runs/20260925-drqv2-geometry-mix-v1-r6/regression-attribution-v1.json) | `f34ba3c63cdff4da7279d02ceeb293e0b72a0787e762a8357172101458d705f9` |

Commands used from repository root; **these CLIs reject existing outputs, so
do not rerun into the already-consumed paths**. The source-action replay is
an additional environment action on reused diagnostic cells; offline audit
commands may use *new* empty output paths when independent regeneration is
required. No action below resumes a training run.

```bash
/tmp/kilo/haic-cpu21/bin/python -B -m scripts.diagnose_drq_geometry_regression --output runs/20260925-drqv2-geometry-mix-v1-r6/paired-regression-v1.json
/tmp/kilo/haic-cpu21/bin/python -B -m scripts.diagnose_drq_geometry_replay --output runs/20260925-drqv2-geometry-mix-v1-r6/replay-distribution-v1.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/kilo/haic-cpu21/bin/python -B -m scripts.diagnose_drq_source_states --output-root runs/20260925-drqv2-geometry-mix-v1-r6/source-state-replay-v2
/tmp/kilo/haic-cpu21/bin/python -B -m scripts.summarize_drq_geometry_regression --output runs/20260925-drqv2-geometry-mix-v1-r6/regression-attribution-v1.json
/tmp/kilo/haic-cpu21/bin/python -B -m unittest tests.test_diagnose_drq_geometry_regression tests.test_diagnose_drq_geometry_replay tests.test_diagnose_drq_source_states tests.test_summarize_drq_geometry_regression
```

**Direct answer:** Source 11/32 fell to 1-4/32 because the 22,768 online-only
updates produced different, often immediately divergent steering/braking on
states where the source finished; 31 variant/source-win pairs were lost but
only six new wins were gained. Actor/encoder drift is directly observed,
replay coverage/dilution plausibly failed to protect source-style behavior,
and critic-Q reordering appears on a subset, not all, of the losses. Their
relative causal shares and whether a longer horizon would prevent collapse
remain **unidentified** by this small reused development diagnostic.
