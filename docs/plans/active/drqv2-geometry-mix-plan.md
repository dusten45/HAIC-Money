# DrQ-v2 Geometry-Mix Fine-Tuning

## Hypotheses

- **Uniform:** equal sampling across six selected seed-measured official-road families is the catalog-only baseline.
- **Failure weighted:** increasing exposure to short-entry opening turns and mid-road reversals improves those already documented failure neighborhoods while retaining easy and late-approach roads.
- **Easy retention:** a 45% lower-curvature anchor, 40% moderate opening/mid families and 15% finish-approach family better preserves route competence while introducing harder training roads.

These are distribution hypotheses, not claims that a family label proves policy
difficulty or a particular mechanism. The generated road catalog is fixed and
does not depend on actor scores.

## Matched Design

- Start six runs from the two immutable pad-4 DrQ-v2 source checkpoints with a weight-only fork; source critic targets are copied and all optimizers, online replay and RNG streams are new.
- Use the identical raw-reward DrQ-v2 configuration across the six runs: 10,000 source-policy online startup decisions with final `.05` exploration noise and zero updates, followed by 32,768 additional decisions and exactly one update per post-warmup decision (22,768 expected updates).
- Online replay is only the run's online experience, batch 64, 3-step return, gamma .99 and padding 4. There is no teacher replay, reward shaping, L2 term, or alternative environment.
- Geometry sampling uses only the frozen 120 TRAIN catalog roads and obstacle track IDs 1-4; TRAIN-DIAGNOSTIC rows are prohibited from replay.
- The fixed variant weights, sampler RNG state, source/model hashes, decision/update counters and per-update replay provenance are recorded. Checkpointing is at 16,384/32,768; only a verified full episode-boundary checkpoint can resume.

## Development Diagnostics

Run the two source actors and six final variants on the same 16 TRAIN-DIAGNOSTIC
roads with two CPU21 repeats. These roads are previously designated development
diagnostics, not fresh validation, confirmation, or blind. Diagnostics cannot
delete roads, change training weights, or be presented as generalization evidence.
Report finishes, visited-tile maximum progress, terminal progress/damage, raw
reward, off-track/collision/action/road-location traces by family and source.
No significance claim is made from two source seeds and one geometry per row.

## Stop And Promotion Boundaries

Reject the matched matrix if source identity, fork parity, training budget,
gradient count, replay quota, RNG restore, catalog split, or receipt/hash fails.
Any such failure stops that frozen run; do not add training steps, substitute
source actors, alter family weights, or reuse confirmation/blind IDs. A result
on TRAIN-DIAGNOSTIC can nominate a development hypothesis only. A fresh
screen/confirmation pool and a separate authorization are required before any
generalization or promotion claim. Blind geometry and outcomes remain unopened.

## Completed r6 Results

All six runs completed the frozen 32,768 additional online decisions and 22,768
updates, from 10,000 update-free warmup decisions, on the frozen TRAIN catalog.
Each `result.json` contains both checkpoint records (16,384 and 32,768), their
actor/checkpoint/manifest hashes, and replay-sample trace hashes. The final
checkpoint and final replay-sample trace SHA-256 values are:

| Run | Full run result | Final checkpoint SHA-256 | Final replay-trace SHA-256 |
|---|---|---|---|
| learner-0-uniform | [`result.json`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-uniform/result.json) | `36f93656d9f92639c230175ce7593f6e818e9d0e0d54f7fdb4ea8f1584ad0a7a` | `45fe530eeb22e2639f667353f32fd44aebdf3da55bba4f2d76cfd53ed359236a` |
| learner-0-failure_weighted | [`result.json`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-failure_weighted/result.json) | `4545024b867937cedd1864a2b4085fec6b90f7ba49ef67e843c54a51eb30202b` | `ca9ea5f6e428105174c92d0b26e60a3751083e244330bb3fc765421b7aa6d085` |
| learner-0-easy_retention | [`result.json`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-easy_retention/result.json) | `461c21bb3bf9809271207445d797c4785c62a42685de079b1c5a3afc0d52da1c` | `17b58fdfa683c7a00e594308d4f30fd34f54c22158bd364d49afa6f53bacf619` |
| learner-1-uniform | [`result.json`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-1-uniform/result.json) | `1fba2420cb54fd26427e3a21d69b0b651ec718b4c3f58f8b09658206739637cf` | `c80659060a2fd69160bba1dcafbc3fc3e340f895ddbe1a7ee35ba9b588ed4d5d` |
| learner-1-failure_weighted | [`result.json`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-1-failure_weighted/result.json) | `48d5d5047943e82deee31ddc99ec249f1a48a9584ac28c30e2c00b36fc527a30` | `316879ca1ee49d2019345eadcef904f0e90bc055c955100511b1259c348fb260` |
| learner-1-easy_retention | [`result.json`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-1-easy_retention/result.json) | `685ba1dee586aa5e906e7968fc47caff9488d2df3ad9df6c6f191c75794a4d11` | `7c31a4ea41f27b68c8d5ccdc5d729f9d7af3a6230ad36bb5a3719977975a0ddc` |

The frozen CPU21 post-training diagnostic used only the 16 designated
TRAIN-DIAGNOSTIC roads and produced 256 episodes: eight actor roles (the two
unchanged sources and six final variants) x two repeats. Repeat 1 matched repeat
0 for all 128 actor/road pairs. The immutable [`manifest`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json),
[`summary`](../../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/summary.json),
and `episodes.jsonl` bind all trace SHA-256 hashes. The manifest SHA-256 is
`fb14fe9eab14f61cdecc45453b69e44d34fdb967368c44c255e827292804b380`.

Canonical repeat 0 finish counts on the same 16 roads were source learner 0
6/16 and source learner 1 5/16; uniform learner 0 1/16 and learner 1 0/16;
failure-weighted learner 0 4/16 and learner 1 0/16; easy-retention learner 0
0/16 and learner 1 3/16. Across both source seeds, the candidate totals were
uniform 1/32, failure-weighted 4/32, and easy-retention 3/32, versus 11/32 for
the unchanged-source roles. These are descriptive outcomes from two actors on
16 previously designated development roads, not an independent sample or a
generalization estimate.

| TRAIN-DIAGNOSTIC family | Uniform finishes / n (mean max progress) | Failure-weighted finishes / n (mean max progress) | Easy-retention finishes / n (mean max progress) |
|---|---:|---:|---:|
| `easy-curvature-anchor` | 1/6 (0.476) | 1/6 (0.470) | 0/6 (0.456) |
| `finish-approach-turn` | 0/4 (0.755) | 0/4 (0.706) | 0/4 (0.610) |
| `mid-road-left-right-reversal` | 0/6 (0.468) | 2/6 (0.766) | 1/6 (0.640) |
| `mid-road-sustained-or-same-turn` | 0/4 (0.331) | 0/4 (0.414) | 2/4 (0.795) |
| `opening-delayed-high-turn` | 0/6 (0.337) | 1/6 (0.525) | 0/6 (0.431) |
| `opening-short-entry-left-turn` | 0/6 (0.455) | 0/6 (0.551) | 0/6 (0.727) |

`n` pools both learner seeds over the family roads; mean max progress is the
episode maximum visited-tile fraction, not physical arclength. The full summary
and episode/trace records also retain reward, progress, damage, termination,
collision, action, and sparse road-location telemetry. The diagnostic is
unranked, development-only evidence: no variant is selected or promoted, and no
fresh held-out, confirmation, blind, or official HAIC evaluation occurred.
