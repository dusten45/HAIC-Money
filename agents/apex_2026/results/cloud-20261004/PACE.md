# Clear-road propulsion study

`pace_agent.py` replaces the historical hybrid's fixed turn throttle cap with
`speed² * abs(tan(emitted_steer)) / (3.24 * lateral_accel)` demand gates. Public
low/high thresholds apply to metric and guided routes. All steering magnitudes
are covered; positive braking suppresses gas. Inference remains NumPy-only.

The mild-turn test first failed with gas 0.16 despite useful grip, then passed.
Independent review found small steering could bypass the new gate and guided
mode ignored custom thresholds; three further failing camera-level tests
reproduced these issues before v2 fixed them. Eight new tests and nine legacy
hybrid tests pass; independent review found no blocker in v2.

## Fresh complete development result

`development/r2-pace.json` binds current v2 source SHA256
`7fc32399068e80a35aefb7c2b4475663a6162a698db59f9341dd9c4ed9df3b6e`,
parameters `{}`, evaluator `39a28061…`, and the first prior development reject.

Required laps were **18.76 / 25.14 / 21.78 / DNF seconds**; track 4 retired
`off_track` at progress 0.76703 after three contacts. Gains on three tracks do
not compensate for the missing fourth finish. The prospectively selected
13-second profile failed; this is the second complete new development reject.
The full receipt includes all sixteen extra cells and their DNFs. This
candidate is not promoted.

Earlier v1 benchmark screens are `probes/pace-v1-p{1,2,3}.json`, bound to source
`49d8c08b…`. Default required laps were 18.76/25.14/21.78/DNF; faster 125/180/.22
parameters gave 17.94/23.36/DNF/DNF, and lower demand gates gave
18.84/25.14/22.28/DNF. These screens do not count toward prospective relaxation.
Reverse `probes/pace-v1-to-v2.patch` against the current v2 file to reconstruct
and hash the exact screened v1 source.

Pace alone is not a completion improvement. Holdout simulations remain unopened;
the root agent, historical candidates, and official simulator are unchanged.
