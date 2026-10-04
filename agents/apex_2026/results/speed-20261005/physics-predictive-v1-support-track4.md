# Exact V1 track-4 unsupported geometry localization

This read-only study rebuilds the camera observer from zero on the completed
exact V1/d2bd replay, then restores recorded passive parent camera state for
deterministic geometry checks. All 226 camera sensor records, post-action
observer states and eligible geometry branch decisions match exactly. It
runs no Agent.act, planning search, world step, new lap or holdout access.
The original source-bound lap remains 18.060s with one contact; this analysis
does not create a new driving result.

The replay contains 41 predictive, 151 unsupported and 34 other decisions.
The unsupported decisions have these first blocking branches:

| First blocking branch | Count |
| --- | ---: |
| Full dense reference below strict 1.9m support | 140 |
| Initial full-body road guard | 5 |
| Legacy center-depth 5.8m prefix too short | 4 |
| Parent road missing | 2 |

Nearest unknown pixels across the 151 unsupported decisions are 104 image
visibility boundaries, 35 circle paint/halo pixels and six bright unknown
pixels; six failures have no localized pixel. These are spectral/context
diagnoses, not privileged semantic labels. No dark car raster outside its
fill box is the first cause. Existing field and body margins stay unchanged.

At step 149, geometry is supported, initial body-road slack is +0.991m and
reference-front slack is 32.4m. The selected action is steer 0.002842, gas 1,
brake 0; the planner accepts a model backup beginning after the executed
0.08s block, lasting 0.4s and ending at reference progress 22.036m. At step
150, the entire dense reference fails because its first unsupported sample
[-4.5473,35.1445]m is only 1.8925m from image row 0. Its nearest pixel is
asphalt gray 0.411765, but the image boundary remains unknown. No circle,
innovation or parent-road-loss flag causes this handoff. The emitted parent
action instead has steer -0.009834, gas 0.421321 and brake 0 at true speed
82.646m/s (truth is a comparison label, never observer input).

The inherited fallback is not the previously checked emergency tail: source
lines 1492-1493 select the parent's current action, and the checked backup
action is neither retained nor executed. This is a concrete policy
consistency limitation. It does not prove that executing an old backup
would avoid a later contact, because future cameras can reveal obstacles
or disagreement that was absent in the previous view. A future design
would need causal backup retention, coordinate/state transport and explicit
invalidation rather than treating earlier model feasibility as a guarantee.

At steps 158-162, the first blocking condition is circle paint/halo on the
dense center reference, before initial-body or control-search checks:

| Step | First unsupported depth (m) | Unknown pixel gray | Emitted steer | Gas | Brake |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 158 | 1.880921 | 0.580392 | -0.370306 | 0 | 0.099687 |
| 159 | 1.601112 | 0.596078 | -0.383797 | 0.139728 | 0 |
| 160 | 0.593527 | 0.521569 | -0.400000 | 0 | 0.072631 |
| 161 | 0.115226 | 0.572549 | -0.400000 | 0 | 0.066309 |
| 162 | 1.513023 | 0.603922 | -0.400000 | 0 | 0.040000 |

All five decisions have parent-road-loss count zero and no dynamics
innovation flag. At step 159 the current circle detector misses, but the
already transported active pass memory is correctly retained as one
constraint; this is not the earlier dropped-memory bug. Step 162 is the
recorded contact action. The first innovation fallback appears at step 164,
after contact. V2's strict-supported-prefix change can address the distant
boundary failure at 150; it cannot establish that a near circle-truncated
prefix is sufficient for a fast action and stop tail. No grass filling,
body-margin relaxation or collision-guard relaxation follows from these
observations. Fresh source-bound evaluations remain necessary.
