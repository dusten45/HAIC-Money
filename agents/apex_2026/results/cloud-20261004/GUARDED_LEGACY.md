# Guarded legacy-camera envelope study

The corridor-prefix guarded combination failed its required-lap screen. This
separate variant preserves the earlier robust camera/corridor definitions and
uses the corrected metric speed envelope and guarded public routing.

Frozen source SHA256:
`0146dab0bffd43d7bc7d246c72d71ab6492f04f00fe38dd1870c3ddd18e0b05b`
(134,315 bytes). Inference imports NumPy and allowed standard-library modules;
it imports no other candidate or simulator.

## Exact source derivation

`guarded_legacy_agent.py` copies the definitions before `_MetricAgent` exactly
from safety V2 source
`7b783070f26b108957f4d9c4c5658b411a7f03cb26690055cf2bdba13dde4c37`.
That prefix SHA256 is
`fb3d708c7d0d7cac2961f0602fe479fde03abfd8514c606b87cda2b55f85a794`.
The metric/public suffix is byte-identical to the reviewed guarded source
`c0e0443f…` after changing its descriptive class docstring. Slice equality and
hashes are recorded in `probes/guarded-legacy-v1-lineage.json`.

The new curved-path hull checker, orange-core classifier and halo topology
from corridor V4 are absent in this variant. It retains the earlier bounding
box/side-padding behavior. This is an isolated comparison of perception and
route definitions; it does not establish either prefix as safe on every road.

Embedded defaults are cruise 95, preview 0.23 and braking 60, with lateral
acceleration 145, heading window 10 and step 4. The suffix retains sparse local
headings/chords, the near-edge braking budget and 1.2 curvature peak reserve,
configured visible-horizon braking, robust hazard pedal/target protection,
and bounded malformed-camera recovery.

Eleven camera/guided/default/malformed/reset regressions pass. The initial
safety-only source failed seven extension/default/parser cases before the
tested suffix was composed. The corridor-specific radius-checker regression
is inapplicable because that checker is absent. No unrelated public margin
or transition controls were added.

## Bounded driving evaluation

Exact receipts `probes/guarded-legacy-v1-mandatory.json` and
`probes/guarded-legacy-v1-target.json` use the frozen source and parameters `{}`.
They retain source, environment and action hashes. These benchmark probes do
not advance the consecutive complete-development rejection count.

| Required track | Result | Contacts |
|---|---:|---:|
| 1, seed 516237 | 21.42 s | 0 |
| 2, seed 644062 | DNF, progress 0.68452 | 0 |
| 3, seed 1007 | 25.30 s | 0 |
| 4, seed 18800 | 23.24 s | 0 |

The known failure `(1,1764402399)` remains DNF: progress 0.33916, one contact,
damage 0.2, official retirement reason `off_track`. Sampled full/partial
off-road events are zero, which does not exclude events between decisions.
All five operational error fields are null.
Measured local maxima are initialization 20.71 ms, reset 0.095 ms, action
75.59 ms and RSS 97.49 MiB. These pass the stated runtime bounds without
establishing official container certification.

Independent read-only review confirms the exact copied prefix and guarded
suffix, all eleven focused regressions, and a real obstacle camera returning
robust target/brake before clear frames restore the metric route. This verifies
specific behavior; the fresh screen still fails mandatory completion and pace.
No speed gain, general completion or dynamic path certificate is established.
**Original target met: false; promoted: false.** This study did not use holdout
simulations.
