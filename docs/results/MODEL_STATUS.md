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
| Candidate | No formally designated best single-model candidate |
| Why | The stored evidence identifies two reproducible DrQ-v2 controls, including one actor with 7/32 on two internal confirmation cohorts, but no single immutable package has a recorded official public-track result or confirmation state. The repository must not infer one from local results or a mixed leaderboard. |
| Next requirement | One checkpoint/package evaluated under the current declared local scope and, when authorized, recorded official public-track scope. |

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
