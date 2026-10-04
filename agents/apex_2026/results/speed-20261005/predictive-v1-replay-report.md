# Predictive V1 exact diagnostic replays

The frozen V1 controller reproduced the original mandatory SCREEN's emitted
actions and all 15 nonruntime outcome fields exactly on track 2/seed 644062
and track 4/seed 18800. During these actual diagnostic replays, prediction
was active for 99 of 468 decisions (21.15%); unsupported camera geometry
caused 329 fallbacks. Track 4's contact occurred during an unsupported
fallback, before its first innovation warning. These findings diagnose a
control handoff gap; they do not establish a correction or a safety proof.

The original four-track SCREEN remains 15.18/19.34/16.96/18.06 seconds with
0/0/0/1 contacts. The original <=13-second and very early 10-second goals
remain unmet. This is no new candidate gate, no formal adoption, and no new
holdout or additional geometry was opened.

## Frozen inputs and exact replay checks

- Candidate: `agents/apex_2026/fast_predictive_agent.py`,
  SHA-256 `d2bd8a0855876f5cc4d281bd1183646ad29bd27e7f9231a4af6d917e901b844e`.
- Original SCREEN: `predictive-v1.json`,
  SHA-256 `78dcfaf1c98a88ce5de440ef8bc69c2b44caa1a54935efda070fecf7953df3c2`.
- Diagnostic helper: `research/speed_20261005/predictive_replay.py`,
  SHA-256 `20bae2a06835b0b0492b7be4a049d515c1e376239357b801c091afcec1ef946d`.
- Helper tests: `tests/test_predictive_replay.py`,
  SHA-256 `cf43a2b39e0772e9378bd40edf212abbd9cd6dc7337f31b5950169b4b561f112`.
  A fresh focused run passed all 18 tests. The tests cover source/receipt
  binding, missing semantic fields, action order/count and single-ULP
  differences, bounded actions, and passive diagnostic capture.
- Structured evidence: `predictive-v1-replay-summary.json`,
  SHA-256 `b82678e8da5efa370b9c170a277d082466f4351985d54e6fc846a7f1a04f1ad6`.

Both worlds ran serially after the frozen SCREEN finished, with numerical
library threads limited to one. The helper validated source, frozen
parameters and all 11 protected environment/evaluator hashes before
loading the agent and after the run. The evaluator stayed at
`39a280612b3f5e7402b7a475c447a7cb9c465ddc0bc9e5c5f98300cdcad0325c`.
Each camera stack was saved before the single act call; diagnostics were
copied afterward. Simulator truth was recorded separately for comparison
and never supplied to the agent.

Each replay independently passed float32 action hash, action count, all
15 semantic fields, finite bounded cameras/actions, exact previous-action
memory, and exactly one observer advance per emitted action. Large camera
and step records remain ignored under `.haic-artifacts`; their file and
raw camera-stack hashes are in the structured summary.

| Actual replay | Steps | Lap (s) | Contacts | Predictive | Unsupported | Low speed | Innovation | No feasible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Track 2 / 644062 | 242 | 19.34 | 0 | 58 (23.97%) | 178 | 6 | 0 | 0 |
| Track 4 / 18800 | 226 | 18.06 | 1 | 41 (18.14%) | 151 | 21 | 12 | 1 |

There were no yaw or timeout fallbacks. Track 2's action hash is
`6106c0de9dd81c4b0560052dd72c770cca05c5e7000cab978b64769cf0586a5c`;
track 4's is
`fab87df3819976b62bf66668abacf3782ccda20b2182f8c9d457007979c8d2b0`.

The actual replay action latency was 1895.24ms maximum / 1710.74ms P95 on
track 2 and 2102.54ms maximum / 1666.37ms P95 on track 4. Per-decision model
calls reached 4475 and 4402 respectively (206547 and 143597 total).
These are diagnostic-process timings, all below the five-second action
limit; they are not official container certification or a memory result
for the submitted agent.

## Geometry gates and contact handoff

An independent cached-input study reconstructed the camera observer from
zero, advanced it only with the saved actual actions, and restored passive
parent camera state. Every recorded sensor, post-action observer state and
eligible geometry branch matched on both tracks. It executed no Agent.act,
planner search, simulator step or new lap.

| First unsupported branch | Track 2 | Track 4 |
| --- | ---: | ---: |
| Entire dense reference below strict 1.9m road support | 165 | 140 |
| Legacy 5.8m center-depth prefix too short | 9 | 4 |
| Parent road missing | 3 | 2 |
| Initial full-body road guard | 1 | 5 |

Nearest unknown pixels were predominantly image boundaries: 141/178 on
track 2 and 104/151 on track 4. Circle paint/halo accounted for 23 and 35.
These are spectral/context diagnoses, not privileged semantic ground truth.

On track 4, decision 149 was predictive at actual pre-action speed
80.095m/s. The selected action was approximately [0.002842, 1, 0]. Its
three-scenario planner accepted a model backup starting after the executed
0.08-second block, lasting 0.4 seconds and ending at reference progress
22.036m. At decision 150, prediction stopped because a far dense-reference
sample [-4.5473, 35.1445]m was only 1.8925m from the image's row-zero
unknown boundary. The nearest pixel was asphalt gray 0.411765; there was
no circle, innovation flag, or parent road loss. The emitted fallback
was approximately [-0.009834, 0.421321, 0] at actual pre-action speed
82.646m/s.

Decisions 158-162 failed the full dense-reference gate on circle paint/halo,
before initial-body and planner checks. At decision 159 the current
detector saw no circle, but transported pass memory retained one circle
constraint. At contact decision 162 the actual status was
`fallback_unsupported`, action [-0.4, 0, 0.04], actual pre-action speed
38.285m/s, HUD speed 38.964m/s, and decoded body yaw 4.375rad/s.
The active pass circle was [-3.1494, 1.1758]m; road-loss count was zero
and all sensor rejection flags were false. The first innovation fallback
was decision 164, after the contact.

This shows that the executed parent fallback is not the earlier checked
emergency tail. A future causal design could retain a bounded stop tail
and revalidate it against current camera geometry, observer state, and
the union of visible and remembered circles before each action. Prevention
has not been tested; old model feasibility does not justify executing a
cached tail after new observations invalidate it. Three representative
uncertainty scenarios also do not certify the complete observation/state
error box. No guard, road-margin, or frozen-source change was made here.

The independent support study is bound by:

- `research/speed_20261005/physics_predictive_support.py`,
  SHA-256 `739b99c517397ab89ef35fe8ac2a4f6d49165365617f61c978c2eca477578f7b`.
- `physics-predictive-v1-support-track2.json`,
  SHA-256 `16fc31ff35732056c2c06486c6f3e5fc8ab7f924637005f060fd6319df2a52f8`.
- `physics-predictive-v1-support-track4.json`,
  SHA-256 `a51d37f7cafa98c4fee9f0a71d6a3b8849dab8c94c73d76db80078ea870c6dad`.

## Attribution limits

The original evaluator did not save predictive internal statuses, model
calls or latency details. Those measurements describe the actual
diagnostic replays. Exact action hashes and outcomes establish identical
emitted trajectories in these two repeats, but they do not prove identical
internal planning branches in the original SCREEN, especially with a
wall-clock planner budget. The cached support classifier localizes gates
on those same replay inputs; it provides no new driving-performance claim.
