# Model Status

This is a candidate/provenance ledger, not a leaderboard. A per-track historical
best may come from different submissions and is never presented as a single-model
result.

## Current Validated Baseline

| Field | Value |
|---|---|
| Configuration | Native DrQ-v2 control, augmentation pad 4, raw-reward four-frame contract |
| Unique actors | Training seed 0: `433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37`; training seed 1: `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954` |
| Evidence | Two unique control actors were evaluated on two fresh confirmation cohorts. Seed 0 recorded 6/32 (L2) and 4/32 (padding); seed 1 recorded 7/32 on both. These are two training seeds, not four independent trained policies. |
| Run/checkpoint scope | `runs/20260922-drq-steering-l2-v1-fast/control-seed{0,1}/checkpoints/step-000131072/actor.pt` and `runs/20260922-drq-augmentation-pad-v1-restart/control-seed{0,1}/checkpoints/step-000131072/actor.pt`; detailed run directories are local artifacts. |
| Source revisions | L2 execution `8e5fa46`; padding execution `a28ef02`. Result records are [`L2`](../../experiments/drqv2-steering-logit-v1-result.json) and [`padding`](../../experiments/drqv2-augmentation-pad-v1-result.json). |
| Confirmation / promotion | Internal confirmation receipts were operationally valid, but both studies rejected their treatments. No blind candidate, official package release, or official confirmation was authorized. |
| Status | Internal validated baseline only; not an official submitted or confirmed model |

## Best Single-Model Candidate

| Field | Value |
|---|---|
| Candidate | Internal pixel RLPD long-horizon follow-up v1, training seed 11, 131,072 student decisions; actor SHA-256 `f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1` |
| Provenance | `runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt`; protocol [`pixel-rlpd-long-horizon-followup-v1`](../../experiments/pixel-rlpd-long-horizon-followup-v1.json), result [`follow-up v1`](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json) |
| Internal evidence | On the same 24-cell screen, seed 11 reached 12/24 finishes versus 3/24 for its online-only SAC control; on fresh 32-cell confirmation, 12/32 versus 7/32; on the protocol-selected 24-cell blind, the RLPD actor finished 9/24 with 0.679 mean progress. All screen/confirmation/blind CPU reload, repeat determinism, and operational checks passed. These are internal CarRacing proxies, not official HAIC scores. |
| Limitations | Two learner seeds, three screen and blind track IDs, four confirmation track IDs, eight geometry seeds per partition, and one final blind-tested actor. V2 failed its pilot gate and remains separate. The larger prior-data cap and student horizon were changed together in follow-up v1, so the result does not isolate a cause. Entropy-target V1–V3 stopped at preflight. V4 collected new data and trained four students, but its post-freeze source-hash gate stopped before evaluation; V4 teacher/student data and checkpoints are consumed, and the V4 screen/held-out cells were never evaluated and are retired. The separately frozen V5 used another audited split and completed all internal stages: author target `-1.5` beat `+1.5` on both matched confirmation seeds (17/32 vs 6/32, 7/32 vs 6/32); its seed-50 finalist finished 7/24 blind cells. That blind grid differs from follow-up v1's 9/24, so these are not a matched cross-study comparison or an automatic candidate replacement. See the [V5 result](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json). |
| Status | Best current internal candidate only; not an official submitted or competition-confirmed model. No package release, server confirmation, or official score is recorded. |
| Next requirement | Any further internal intervention requires a separately frozen hypothesis, fresh audited partitions, and evidence sufficient to justify changing the designated candidate. Official public-track evaluation/model confirmation/submission requires separate explicit user authorization immediately before the external action. |

## Official Submitted Candidates

| Local evidence | Official submission identifier | Server validation/public result | Status |
|---|---|---|---|
| `submissions/20260919T135436Z_baseline1-final/manifest.json` | Not recorded in this repository | Not recorded in this repository | Local package artifact only; official submission status is unverified |

## Current Confirmed Model

No confirmed competition-site model is recorded locally. Do not infer confirmation
from a local ZIP, a local evaluator receipt, or a per-track leaderboard result.

## Per-Track Official Historical Best

No official per-track result ledger is recorded locally. If future site results are
entered here, label them `may come from different submissions; not a single-model
result` unless the same package hash is proven for every row.

## Update Rule

Update this file only for a real status transition: a validated baseline change, a
single candidate designation, an official upload/result, or a competition-site
confirmation. Every update must link the run/checkpoint, source commit, model/package
hash, evaluation scope, result artifact, and confirmation status.
