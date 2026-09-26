# Generalization Policy

## Separate Evidence Scopes

| Scope | Meaning | Allowed use |
|---|---|---|
| Training | Episodes and maps used to optimize a candidate | Never describe as validation or fresh generalization. |
| Screen/development | Fixed internal cells used for checkpoint/candidate selection | Consumed development evidence; not fresh confirmation. |
| Confirmation | Predeclared internal cells evaluated after screen-selected actors are frozen | Pass/fail, or a one-time finalist choice only if its rule was frozen in the study protocol; no ad hoc tuning. |
| Reserved internal blind | Internal cells protected until a confirmation gate passes | One terminal test, never iterative tuning. |
| Official public tracks | Competition-server evidence for submitted models | External evidence, distinct from local cells. |
| Official private tracks | Unavailable final-evaluation target | Never treat as an internal holdout dataset. |

## Recorded Internal Partition History

This table is a selective routing aid, **not an exhaustive seed registry or a
reusable grid**. Before allocating any new cell, search every relevant frozen
protocol, run receipt, source training ledger, and retired allocation across all
research lanes. Exact cells and cross-track exclusions remain in those artifacts;
absence from this table is not evidence that a seed is fresh.

| Study | Screen | Confirmation | Blind status |
|---|---|---|---|
| DrQ-v2 promotion v1 | `101-103` x `31001-31008`, consumed | `111-114` x `31101-31108`, consumed as diagnostic-only | `121-123` x `31201-31208`, reserved and not run |
| DrQ-v2 steering-logit L2 | Reused development screen `101-103` x `31001-31008` | `211-214` x `32101-32108`, consumed | `221-223` x `32201-32208`, reserved and not run |
| DrQ-v2 pad 4 vs 1 | Reused development screen `101-103` x `31001-31008` | `311-314` x `33101-33108`, consumed | `321-323` x `33201-33208`, reserved and not run |
| [DrQ-v2 teacher replay r3](../../experiments/drqv2-teacher-replay-v1-r3.json) | Teacher training collection consumed; A3 stopped before screen | Reserved, not opened | Reserved, not opened; r1/r2 allocations were also retired |
| [Pixel RLPD off-policy pilot v1](../../experiments/pixel-rlpd-offpolicy-pilot-v1.json) | Protocol allocation retired before any environment interaction; 0 decisions | Proposed allocation retired, not recycled | Proposed allocation retired, not recycled |
| Pixel RLPD off-policy pilot v2 | `101-103` x `4000006001-4000006004`, consumed | `311-314` x `4000006011-4000006018`, reserved and not run | `321-323` x `4000006021-4000006028`, reserved and not run; conditional full-stage screen/confirmation/blind allocations `4000006031-6058` also remain unopened |
| Pixel RLPD long-horizon follow-up v1 | `201-203` x `4000013001-4000013008`, consumed | `211-214` x `4000013011-4000013018`, consumed | `221-223` x `4000013021-4000013028`, consumed |
| Pixel RLPD entropy-target ablation v1 | `261-263` x `4000023001-4000023008`, allocation retired before interaction | `271-274` x `4000023011-4000023018`, allocation retired before interaction | `281-283` x `4000023021-4000023028`, allocation retired before interaction |
| Pixel RLPD entropy-target ablation v2 | `261-263` x `4000025001-4000025008`, allocation retired before interaction | `271-274` x `4000025011-4000025018`, allocation retired before interaction | `281-283` x `4000025021-4000025028`, allocation retired before interaction |
| Pixel RLPD entropy-target ablation v3 | `291-293` x `4000027001-4000027008`, allocation retired before interaction | `301-304` x `4000027011-4000027018`, allocation retired before interaction | `311-313` x `4000027021-4000027028`, allocation retired before interaction |
| [Pixel RLPD entropy-target ablation v4](../../experiments/pixel-rlpd-entropy-target-ablation-v4.json) | `291-293` x `4000029001-4000029008`, retired unopened after the pre-screen source-hash stop | `301-304` x `4000029011-4000029018`, retired unopened | `311-313` x `4000029021-4000029028`, retired unopened |
| [Pixel RLPD entropy-target ablation v5](../../experiments/pixel-rlpd-entropy-target-ablation-v5.json) | `351-353` x `4000033001-4000033008`, consumed | `361-364` x `4000033011-4000033018`, consumed for four frozen screen-selected actors | `371-373` x `4000033021-4000033028`, consumed once for the predeclared winning-target finalist |

The RLPD v1 pilot was retired after a runtime preflight abort with no environment
interaction; those proposed numbers are not represented as consumed outcomes.
Nevertheless, its allocation was not recycled. The v2 and long-horizon
training/data pools `4000004001-4000004032` and `4000012001-4000012064`
were used by teacher collection and/or student training and are consumed training
geometry, not validation. V2's
separately reserved teacher pool `4000005001-4000005064` and full evaluation cells
were not opened but remain excluded from new experiments. Entropy-target v1
aborted during zero-interaction metadata preflight; its
`4000022001-4000022064` teacher pool and `4000023xxx` evaluation allocations were not
used and have been retired, not recycled. Exact prior allocation and cross-track
exclusions live in each protocol JSON. The following entropy-target protocol also
aborted before environment interaction because its internal study name remained v1
while the protocol filename/run paths said v2. Its `4000024001-4000024064` teacher
pool and `4000025xxx` evaluation allocation were also retired, not recycled.
Entropy-target v3 froze `4000026001-4000026064` teacher cells and the `4000027xxx`
partitions, but its learner-seed validator/source hashes were inconsistent; it too
aborted before environment interaction. Retire, do not reuse, those allocations.
V4 then consumed teacher/student training experience from `4000028001-4000028064`
but stopped on a source-hash mismatch before screen candidate evaluation. Its
`4000029001-9008`, `4000029011-9018`, and `4000029021-9028` screen/confirmation/blind
allocations were never opened and are retired rather than repurposed. V5 used a
new `4000032001-4000032064` teacher/student TRAIN pool; that pool and its
`4000033xxx` evaluation partitions are consumed, not available for reuse. Any
continuation requires a new protocol, fresh audited data, and disjoint cells.
Dreamer B1 development geometries,
[DrQ training-only geometry catalogs](../experiments/drqv2-geometry-augmentation-v1.md),
and residual-options development cells also need protocol/receipt audits even
though their exact ranges are not reproduced here.

Read the protocol/result artifacts before using any listed seed. In particular, a
geometry seed may be excluded across all training track IDs even when a table shows
only one evaluation track-ID range.

The legacy `checkpoint-v1` blind receipt at IDs `7-9` and seeds `20201-20208` was
executed. Therefore, do not describe all internal blind cells as untouched; name the
specific study partition. The unopened L2/pad blind partitions in the table remain
reserved for their own studies, while the historical pilot's exact screen schedule is
incomplete and cannot prove global non-use of every other cell.

## Track Lab and Custom Maps

`training/maps/site/site_map_split.json` separates local custom maps into train,
tune, and held-out groups. Those maps are local tools for generalization research;
they are neither official public tracks nor official private tracks. A held-out map
from one designed geometry with multiple environment seeds is not multiple
independent road layouts.

## Public-Track Discipline

When a new official public track appears, it is useful external evidence but also a
new overfitting risk. Record its result for the same immutable candidate alongside
all accessible public tracks. Do not create track-specific models and infer that a
leaderboard aggregate proves one model will generalize to the eventual final-model
run, whose exact track composition remains officially unverified.

## Prohibited Interpretations

- Do not call a consumed confirmation or blind cell fresh.
- Do not tune on internal blind results.
- Do not call different cells, source revisions, or candidates a matched comparison.
- Do not use official private tracks as if their examples were available locally.
- Do not equate a leaderboard's per-track historical best with a single model.
