# Generalization Policy

## Separate Four Things

| Scope | Meaning | Allowed use |
|---|---|---|
| Training | Episodes and maps used to optimize a candidate | Never describe as validation or fresh generalization. |
| Screen/development | Fixed internal cells used for checkpoint/candidate selection | Consumed development evidence; not fresh confirmation. |
| Confirmation | Predeclared internal cells evaluated after candidate freeze | One decision-stage test; do not select on it. |
| Reserved internal blind | Internal cells protected until a confirmation gate passes | One terminal test, never iterative tuning. |
| Official public tracks | Competition-server evidence for submitted models | External evidence, distinct from local cells. |
| Official private tracks | Unavailable final-evaluation target | Never treat as an internal holdout dataset. |

## Recorded Internal Partition History

The following describes evidence records, not a universal reusable grid. Exact seed
lists and training exclusions remain in their protocol JSON files.

| Study | Screen | Confirmation | Blind status |
|---|---|---|---|
| DrQ-v2 promotion v1 | `101-103` x `31001-31008`, consumed | `111-114` x `31101-31108`, consumed as diagnostic-only | `121-123` x `31201-31208`, reserved and not run |
| DrQ-v2 steering-logit L2 | Reused development screen `101-103` x `31001-31008` | `211-214` x `32101-32108`, consumed | `221-223` x `32201-32208`, reserved and not run |
| DrQ-v2 pad 4 vs 1 | Reused development screen `101-103` x `31001-31008` | `311-314` x `33101-33108`, consumed | `321-323` x `33201-33208`, reserved and not run |

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
