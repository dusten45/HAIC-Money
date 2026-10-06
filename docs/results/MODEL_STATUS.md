# Model Status

This is a candidate/provenance ledger, not a leaderboard. A per-track historical
best may come from different submissions and is never presented as a single-model
result.

## Current Submission Baseline

- **crossing_projection + collision-shield v1**, explicitly designated by the user
  at2026-10-02T01:36:47Z. User-reported official result: Track4 finished in **18.4s**,
  **6th overall** at report time.
- User-bound package identity: ZIP SHA-256
  `c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801`.
  No site submission ID, server receipt or independent package-to-result
  verification is available; this is not a new upload or server model confirmation.
- Frozen reference: [`baseline archive`](../../submissions/20261002-crossing-projection-collision-shield-v1-baseline/).
  All11 source-member hashes match the original receipt; isolated exact ZIP copy
  and source-only rebuilding both reproduce the pinned ZIP hash without Agent or
  simulator execution. Self-contained restore instructions are in its manifest and
  the submission ledger; dependency versions and parameter expressions are retained.
  Original experimental ZIP/receipt, source/parameters, standalone crossing and
  previous candidates remain unchanged.
- The original [research result](../../experiments/koi-collision-shield-v1-result.json)
  stays **NOT_ADOPTED**, with its failed **+20ms efficiency gate** unchanged.
  Submission selection does not convert it into a research-gate pass, demonstrate
  ordinary-overavoidance reduction, or authorize retuning/evaluation.

## Joint Temporal Successor (Submission Candidate Excluded, 2026-10-05)

- **Research CLOSED by user direction on 2026-10-06.** Keep the existing champion
  for this submission; end further H4-successor tuning, driving, branch extensions
  and promotion. See the [closure decision](../decisions/INDEX.md#close-the-current-h4-cost-successor).
- The actual-feedback H4 benefit did reproduce; the cost/horizon/terminal-assessment
  combination nevertheless misorders the measured laps. This is not rejection of
  joint-control generally or a proven steering/braking mechanism. Preserve the
  observer and small physics predictor as research assets. Long-term terminal-state
  value assessment is a separate, not-yet-authorized research task, not same-road
  weight retuning or an active replacement campaign.
- **Current `EnvelopeSuccessor` is excluded from submission candidates.** User's
  required evidence is actual feedback lap-time reduction, not local cost or
  intervention count. The [single-intervention branch result](../../experiments/joint-single-branch-v1-result.json)
  provides no such improvement; the current submission baseline above is unchanged.
- Matched consumed TRAIN3/3184000005: A champion21.720s, B onlydecision34thenfresh
  championfeedback21.860s, C continuedsuccessor22.100s. Allfinish; B-A+140ms,
  C-A+380ms, C-B+240ms. A/C reused only after exactsource/input/prefix checks;
  one new Breset with actual-issued history feedback, never storedAfutureactions.
- Interventions0/1/6; all-wheel-road-loss spells0/1/2; collision/contact/damage0.
  Shortfixedtail and actualfeedbackH4 cost gains reproduce, but B's GTprogress
  gain at.32s reverses by1s. Localbenefit doesnot establish wholefeedback utility,
  and laterinterventions cannot be the sole explanation forC's regression.
- B is a fixed-step ablation, **not a generalized deployable candidate**. Neither
  B nor C is promoted; no outcome-selected tuning, furtherepisode or official
  upload/confirmation. This doesnot reject joint-control generally or prove a
  unique observer/dynamics/feedback mechanism from one road.
- Preserve championZIP`c9e376a0...`, distillation, physics1/1/1, actionbounds,
  H4/.32s, all costweights/calibrations and frozen prior outcomes. See the
  [ABC summary](../../runs/joint-single-branch-v1/summary.json) SHA`cc51986f...`
  and [same-scene feedback costs](../../runs/joint-single-branch-v1/feedback-cost.json)
  SHA`5004b48a...`. Generated primary artifacts remain local.

## KOI Collision Shield v1 (Historical Packaging, 2026-10-01)

- User designation on2026-10-01: package the exact frozen v1 for imminent obstacle-
  collision DNF prevention; ordinary-overavoidance reduction remains separate.
- Version: `koi-collision-shield-v1-experimental-submission`.
- [Local ZIP](../../submissions/koi-collision-shield-v1-experimental-submission.zip),
  SHA`c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801`;
  [packaging receipt](../../submissions/koi-collision-shield-v1-experimental-submission.receipt.json).
- Policy source`ad772bde...` and crossing ZIP`a4b35c56...` are unchanged. Only a static
  Agent entrypoint is added around the unchanged runtime. No parameter adjustment.
- [Original research result](../../experiments/koi-collision-shield-v1-result.json)
  remains **NOT_ADOPTED** under its failed+20ms efficiency gate. This separate user
  packaging designation did not reinterpret the result or replace any baseline at
  that time; the2026-10-02 submission selection above is a separate user decision.
- Motivation includes user-reported official Track4 corner/obstacle DNF; no official
  receipt or package-to-result binding was independently verified in this task.
- Local static/ZIP checks and short CPU21 synthetic parity pass; no new simulator
  evaluation, upload, official acceptance or confirmation. Prior ZIPs/root/v2 kept.

## KOI Steering Release v2 (Frozen, Generalization Rejected)

| Field | Value |
|---|---|
| Policy freeze | v2 ZIP`b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce`; runtime/parameters/manifest unchanged. Crossing baseline ZIP`a4b35c56659555f5e60a372261df722628060e4afcd73c88ed9e9688cdb82bb8` and root Agent unchanged. |
| Evidence | [Frozen TRAIN protocol](../../experiments/koi-steering-generalization-v1.json), [guarded original result](../../experiments/koi-steering-generalization-v1-result.json), [matched diagnosis](../../experiments/koi-steering-generalization-v1-diagnosis-result.json). Old24 development cells were not re-evaluated. No protected/blind/official action. |
| Operational limitation | Planned24 roads/72 pairs/144 episodes ended95 valid completions, one bootstrap timeout and48 unrun slots. No continuation/reset retry. Original attempt remains incomplete, not passed. Compare only47 complete pairs on16 roads, never48 baseline episodes against47 candidate episodes. |
| Matched rejection | Baseline41/47 versusv2 40/47 finishes, with36 kept/5 lost/4 gained/2 neither. Five natural losses span four geometries; two new baseline-clean obstacle hits independently violate the prospectively fixed safety gates. Remaining unexecuted cells cannot erase those counterexamples. |
| Proxy limitation | Conditional preserved-window lateral/avoidance integral reductions remain, but17 baseline windows and20 baseline return followups are lost, common return censors rise37->43, and whole-episode max lateral rises29.9527->260.0025m. Better aggregate damage/collision or kept-lap timing cannot offset per-cell safety/completion regressions. |
| Independent validation | [Postrun audit](../../experiments/koi-steering-generalization-v1-postrun-audit.json) SHA`c4bd5b11...` exactly reproduces the frozen summary/source chains; [five-failure audit](../../experiments/koi-steering-generalization-v1-failure-audit-result.json) SHA`55897acf...` verifies identical pre-divergence controls/poses and release-only first divergence. Selected released objects clear; later unreleased objects/road-heading recovery fail. Generation suppression is dormant in all five losses. |
| Status | **Not a generalized internal adoption candidate.** Keepv2 immutable; historical consumed-r2 candidate evidence remains historical only. No replacement of crossing baseline, root Agent, designated RLPD model or official submission/confirmation. No planned-matrix/population completion claim. |
| User observation | Ordinary obstacles still cause visibly excessive lateral avoidance in v2, sometimes leaving the road; reduced versus before, but still excessive. |
| Follow-up | **Margin reduction and steering-release on hold:** they have not resolved excessive avoidance; former release road-recovery H1 deferred. Future reconsideration at **avoidance trajectory / side-selection** only. All further work/checks/evaluation stopped; documentation-only user direction. |

## Historical Validated DrQ Baseline (Research Closed)

| Field | Value |
|---|---|
| Configuration | Native DrQ-v2 control, augmentation pad 4, raw-reward four-frame contract |
| Unique actors | Training seed 0: `433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37`; training seed 1: `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954` |
| Evidence | Two unique control actors were evaluated on two fresh confirmation cohorts. Seed 0 recorded 6/32 (L2) and 4/32 (padding); seed 1 recorded 7/32 on both. These are two training seeds, not four independent trained policies. |
| Run/checkpoint scope | `runs/20260922-drq-steering-l2-v1-fast/control-seed{0,1}/checkpoints/step-000131072/actor.pt` and `runs/20260922-drq-augmentation-pad-v1-restart/control-seed{0,1}/checkpoints/step-000131072/actor.pt`; detailed run directories are local artifacts. |
| Source revisions | L2 execution `8e5fa46`; padding execution `a28ef02`. Result records are [`L2`](../../experiments/drqv2-steering-logit-v1-result.json) and [`padding`](../../experiments/drqv2-augmentation-pad-v1-result.json). |
| Confirmation / promotion | Internal confirmation receipts were operationally valid, but both studies rejected their treatments. No blind candidate, official package release, or official confirmation was authorized. |
| Status | Internal validated baseline only; not an official submitted or confirmed model |
| Research direction | **CLOSED by user direction on 2026-09-29.** Preserve these actor/checkpoint and local package records as historical evidence only; no further DrQ work, promotion, confirmation or submission is planned. See the [closure decision](../decisions/INDEX.md#close-the-drq-v2-research-line) and [speed-only negative result](../../experiments/drqv2-speed-reused-development-v1-result.json). |

## Historical Internal-Study Best Single-Model Candidate

| Field | Value |
|---|---|
| Candidate | Internal pixel RLPD long-horizon follow-up v1, training seed 11, 131,072 student decisions; actor SHA-256 `f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1` |
| Provenance | `runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt`; protocol [`pixel-rlpd-long-horizon-followup-v1`](../../experiments/pixel-rlpd-long-horizon-followup-v1.json), result [`follow-up v1`](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json) |
| Internal evidence | On the same 24-cell screen, seed 11 reached 12/24 finishes versus 3/24 for its online-only SAC control; on fresh 32-cell confirmation, 12/32 versus 7/32; on the protocol-selected 24-cell blind, the RLPD actor finished 9/24 with 0.679 mean progress. All screen/confirmation/blind CPU reload, repeat determinism, and operational checks passed. These are internal CarRacing proxies, not official HAIC scores. |
| Limitations | Two learner seeds, three screen and blind track IDs, four confirmation track IDs, eight geometry seeds per partition, and one final blind-tested actor. V2 failed its pilot gate and remains separate. The larger prior-data cap and student horizon were changed together in follow-up v1, so the result does not isolate a cause. Entropy-target V1–V3 stopped at preflight. V4 collected new data and trained four students, but its post-freeze source-hash gate stopped before evaluation; V4 teacher/student data and checkpoints are consumed, and the V4 screen/held-out cells were never evaluated and are retired. The separately frozen V5 used another audited split and completed all internal stages: author target `-1.5` beat `+1.5` on both matched confirmation seeds (17/32 vs 6/32, 7/32 vs 6/32); its seed-50 finalist finished 7/24 blind cells. That blind grid differs from follow-up v1's 9/24, so these are not a matched cross-study comparison or an automatic candidate replacement. See the [V5 result](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json). |
| Status | Historical internal-study candidate only, not the current submission baseline or a competition-confirmed model. No package release, server confirmation, or official score is recorded for this RLPD actor. |
| Next requirement | Any further internal intervention requires a separately frozen hypothesis, fresh audited partitions, and evidence sufficient to justify changing the designated candidate. Official public-track evaluation/model confirmation/submission requires separate explicit user authorization immediately before the external action. |

## Official Submitted Candidates

| Local evidence | Official submission identifier | Server validation/public result | Status |
|---|---|---|---|
| Shield v1 ZIP`c9e376a0...`; [current baseline record](../competition/submissions.md#current-submission-baseline-2026-10-02) | Not supplied | User report at2026-10-02T01:36:47Z: Track4 finish18.4s,6th overall; no server receipt or independent verification | User-designated submission baseline; package identity user-bound, not server-confirmed |
| `submissions/20260919T135436Z_baseline1-final/manifest.json` | Not recorded in this repository | Not recorded in this repository | Local package artifact only; official submission status is unverified |

## Current Confirmed Model

No confirmed competition-site model is recorded locally. Do not infer confirmation
from a local ZIP, a local evaluator receipt, or a per-track leaderboard result.

## Per-Track Official Historical Best

The user-reported Track4 finish18.4s is recorded above; no server-verified per-track
best ledger is available. The reported6th overall is a rank at report time, not
proof of a single model's all-track generalization. Future per-track best rows must
be labeled `may come from different submissions; not a single-model result` unless
the same package hash is proven for every row.

## Update Rule

Update this file only for a real status transition: a validated baseline change, a
single candidate designation, an official upload/result, or a competition-site
confirmation. Every update must link the run/checkpoint, source commit, model/package
hash, evaluation scope, result artifact, and confirmation status.
