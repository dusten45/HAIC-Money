# Rejected NumPy camera ridge controller

`fast_ridge_agent.py` embeds the exact graph V2 source and adds a camera-only
parametric road reference. The inference imports only NumPy. Metric chamfer
distance propagates from observed non-road pixels; image boundaries remain
unknown, and enclosed car/obstacle holes of at most 110 pixels are filled.
A short ridge walk can follow a bend whose tangent becomes horizontal, unlike
x(y). Pursuit and curvature planning use path arclength. Frozen prior pass
selection and memory are retained; their normal lane offsets are transferred
onto the ridge. This transfer is an approximation and does not certify route
clearance, tyre sweep, or reachability.

Saved-camera causal regression: track4 step65 has a nearly horizontal bend.
The prior parsed-row reference averages 2.54 m from the actual local forward
centerline; the NumPy ridge averages 0.27 m. Step66 improves 2.51→0.18 m.
Across eight selected track4 bend frames, mean errors improve from 2.49–2.93 m
to 0.18–0.97 m. These are privileged offline pose-based measurements of a
camera-only reference. No privileged input enters inference. Chamfer distance
is compared with a SciPy exact-distance research oracle; it is not imported
by inference. Reconstruction took 5.3–6.6 ms in those saved frames.

Already-offroad track2 frames170–172 have no eligible forward ridge near the
camera origin. Reverse-facing frames173–174 can still choose the wrong route
direction. Accurate selected bend geometry does not establish robust recovery.

Ten focused corridor/graph/ridge tests pass. Fresh source-frozen four required
mandatory benchmark:

| track/seed | official lap | progress | contacts |
| --- | --- | --- | --- |
| 1 / 516237 | 15.04 s | 1.0 | 0 |
| 2 / 644062 | DNF | 0.866071 | 5 |
| 3 / 1007 | DNF | 0.462783 | 4 |
| 4 / 18800 | DNF | 0.741935 | 2 |

Maximum measured action latency was 18.80 ms; source initialization maximum
0.515 ms and RSS approximately 95 MiB. This candidate is rejected and not
adopted. The original goal remains unmet. No holdout was opened. Corrected
center geometry alone fails to provide fast, collision-free vehicle motion;
obstacle geometry and route tracking require new causal investigation.
