# Minimal guided-hazard protection on Preview V4

`guarded_preview_agent.py` preserves the robust and metric definitions from
Preview V4 source `f0ccb11d…`, whose complete R4 development run finished all
required tracks and 12/16 extras, with three finishes on each extra track ID.
Only the public guided hazard route changes: it retains robust gas/brake
requests and the minimum robust/metric speed target.

Frozen source SHA256:
`76863032117175870fbf7ef1a18b49fb2b966ca6b53dd58adcd79a1d2ee46bb1`.
The definitions before the literal `class Agent:` are byte-identical to Preview V4.
`probes/guarded-preview-v1-lineage.json` records their hash and embedded defaults;
`probes/preview-v4-to-guarded-public.patch` records the complete source change.
Source snapshots are preserved in the ignored artifact directory.

## Verified scope

Three guided-hazard tests failed on the unchanged Preview V4 copy, then passed
after the public correction. All 17 existing preview behaviors are preserved.
Default/reset equality and extracted-package source/default equality also
pass: **22 focused tests**. Independent read-only review confirms source-slice
identity, the tested public changes and byte-identical packaged `agent.py`.

Defaults remain parameters `{}`: cruise 100, preview 0.19, braking 100, lateral
acceleration 145, curve window 12 and step 4. No heading estimator, curved-path
checker, orange classifier or halo change is substituted into this variant.
Inference remains standalone NumPy plus allowed standard-library modules.

## Preserved model limits

The original metric planner's sparse localized-bend fallback remains a known
research limitation. The independent physical-gauge probe at decoded 67.73 m/s
gives a sparse target 66.82 m/s and gas 0.2499 without braking. Dense support gives
target 59.94 m/s and brake 0.1403. Public hazard protection applies when the robust
route identifies a hazard; it does not correct that separate road model.
Configured braking overrides also retain the original planner's hardcoded
visible-horizon assumption. The selected default 100 matches that assumption.

These limitations preclude a complete safety or generalization claim. The
previous heading/corridor guarded combinations and their failed driving
receipts remain preserved separately.

## Required-lap screen

The single cold required-lap screen completed with the frozen source and
parameters `{}`. Exact receipt `probes/guarded-preview-v1-mandatory.json` retains
source, environment and action hashes. All operational error fields are null.
Benchmark screens do not advance the complete-development rejection count.

| Required track | Lap time | Contacts |
|---|---:|---:|
| 1, seed 516237 | 21.92 s | 0 |
| 2, seed 644062 | 27.64 s | 0 |
| 3, seed 1007 | 26.14 s | 0 |
| 4, seed 18800 | 23.74 s | 0 |

The four laps total 99.44 s. The earlier frozen Preview V4 R4 receipt totals
98.66 s: this isolated guided-pedal change preserves mandatory completion but
does not establish a pace gain. The fresh hybrid baseline totals 90.54 s.
All sampled full off-road counts are zero; track 4 has four partial samples,
and the other tracks have none. These samples do not exclude events between
decisions.

Measured local maxima are initialization 20.71 ms, reset 0.067 ms, action
55.30 ms and RSS 97.36 MiB. These satisfy the stated runtime bounds without
establishing official container certification. The independently checked ZIP
contains byte-identical `agent.py` with the same embedded defaults.

The complete R7 prospective development study is running separately under the
18-second fallback profile after six complete rejected studies. Its fresh
extra-track completion results remain pending; this benchmark alone provides
no generalization claim. Each required lap already exceeds both the original
10–13-second goal and the 18-second fallback limit.

**Original target met: false; promoted: false.** This study uses no holdout
simulations, and the source, tests and evaluator remain unchanged during R7.
