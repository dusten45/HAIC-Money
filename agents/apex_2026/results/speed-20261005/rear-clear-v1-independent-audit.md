# RearClear V1: source-bound pace audit

This is a read-only analysis of the frozen source and its four consumed mandatory traces. It runs no new episode and does not inspect holdout data. Source SHA256 is `093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc`; the published ZIP SHA256 is `844d19e11a5f56c59ad2864aaa8a36fda38168b2ff432711989707037733653b`, and its sole `agent.py` member equals the source byte for byte. The four official finish receipts are 15.54/18.72/16.56/18.14 s, all with zero contacts. Reaching 13 seconds requires gains of 2.54/5.72/3.56/5.14 simulated seconds, respectively. The trace position sums, 969/1159/1070/953 m, are *observed travel*, not route-length lower bounds; covering those same distances in 13 s would require means of 75/89/82/73 m/s, before accounting for launch.

| Track | Actions | Median actual speed m/s | Actions with target below 70 m/s | Brake actions | Exact +6 m/s target-ramp edges | Recovery-signature actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 195 | 64.7 | 70 | 45 | 78 | 4 |
| 2 | 235 | 61.4 | 97 | 56 | 87 | 1 |
| 3 | 208 | 65.3 | 79 | 49 | 81 | 0 |
| 4 | 227 | 54.3 | 113 | 55 | 79 | 25 |

The most frequent visible normal-driving restriction is a low final target, often followed by braking, especially on track 2. The target combines quadratic curve caps, a visibility cap, pursuit-curvature cap, a `last_target + 6` ramp, measured yaw demand, and immediate obstacle stop. These traces record only the final target, so they do **not** identify which cap dominates any individual action. A useful next diagnostic is to record each cap, current road horizon, selected mode, current circle, and gas limiter on already consumed development geometries before tuning a single cause. The source's 190 m/s² lateral budget and 150 m/s² lookahead braking assumption are controller policy values, not measured full vehicle feasibility bounds.

Track 4 has a separate, concrete delay at steps 144–168: 25 consecutive actions at progress 0.738–0.749 hold steering near −0.30 and the previous target at 37.8 m/s, alternating brake 0.35 with 0.08 gas. Actual speed repeatedly falls near 2 m/s. This matches the `_recover` branch and consumes 2.0 simulated seconds; the other tracks show only 4/1/0 actions with that signature. Fixing this alone would leave track 4 near 16 seconds and would not address track 2.

The visibility formula cannot explain the typical 50–65 m/s speeds by itself when the full roughly 34 m road horizon is present: at 60 m/s it allows about 100 m/s, and even at 100 m/s about 95 m/s. For visibility alone to cap 70 m/s, road detection must end around 17–19 m at speeds 60–80 m/s. The current traces lack horizon values, so the frequency of that condition is unknown. Final targets below 45 m/s (9/23/7/48 actions) also require another limiter when road inference is valid, since the visibility cap has a 45 m/s floor.

The +6 m/s ramp binds exactly on 78/87/81/79 transitions, but only 22/21/17/21 of those ramp actions have a target within 5 m/s of the pre-action HUD speed. It is frequent yet not proved to be the main time loss. Both ridge and inherited confidence paths impose this ramp. A trace-only change to the ramp cannot predict the resulting trajectory or safety.

Even when actual speed is below 80 m/s and the final target exceeds HUD speed by more than 12 m/s, emitted gas is below 0.60 on 22/21/17/50 actions. The source's turn-demand cap and rear-spin latch can cause this; track 4's count includes its recovery interval. Traces do not expose the selected limiter, so these counts are a reason to instrument gas decisions rather than an argument to remove grip protection.

The pre-action HUD speed agrees with the preceding trace's post-action speed to about 0.7–0.8 m/s mean absolute error, so gross speed-decoder bias is not evident. Displacement-versus-body-heading sideslip estimated from 0.08-second trace positions has 90th-percentile absolute value about 3.2–3.6° above 40 m/s, with one track-4 sample above 5°; this coarse estimate does not isolate steering or tire dynamics. Missing lateral-velocity feedback can still matter at narrow clearances, but these traces do not establish it as the primary lap-time limit.

There is at most one `_route` state transition per action; recovery actions have none. The clear-ridge branch first computes an inherited action and then replaces it, while the arc guard can recompute base steering and ridge selection recomputes distance fields. This costs wall time, but mandatory maximum action latency is 32–37 ms versus the official 5-second action limit. Official lap time is simulated physics time; removing duplicate camera work alone cannot reduce the reported lap seconds.

**Priority:** first measure the normal-driving target decomposition on consumed track 2 and the track-4 recovery entry, then test a source-bound correction with exact fresh receipts. Current evidence does not support adopting this candidate for the four early-10-second goal, nor does it prove that goal physically impossible.
