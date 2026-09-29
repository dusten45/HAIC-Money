# Road visibility loss in the fresh common-prefix comparison

Read-only diagnosis of the completed TRAIN report `artifacts/haic-research-v2/haic2-early-route-fresh-train-r2-20260929/report.json`, SHA-256 `5338593d10e5df6ed684e04549774eb1b1b189c6cc585aa1d551e6429ae9aed1`. This is not a new score run. All tracks 1–3 × seeds 5052–5055 are consumed TRAIN; they cannot be reused as fresh candidate evidence.

Among the unchanged mode-switch control's twelve valid finishes, three slow laps contained at least 20 consecutive decisions with fewer than three visible pixel-road center rows:

| Cell | Finished lap | Low-visibility decisions | Progress during interval | Speed during interval |
|---|---:|---:|---:|---:|
| 2:5053 | 28.24 s | 51–98 (48) | 0.157→0.160 | 42.0→48.5 |
| 2:5054 | 28.68 s | 110–147 (38) | 0.343→0.346 | 37.5→46.4 |
| 3:5055 | 28.26 s | 109–154 (46) | 0.351→0.351 | 40.8→47.2 |

The fixed fast-after-50 arm shared the same low-visibility interval on 2:5053 and 3:5055, finishing slower by 0.46 and 0.54 s. On 2:5054 it avoided the interval and finished 4.68 s faster. It was nevertheless slower on 11/12 fresh matched cells, so unconditional fast continuation remains rejected. On the three control intervals, mean commanded gas was 0.036–0.054 and mean steer about +0.09; visited-tile progress barely advanced even as simulator-measured speed rose. Speed and progress are evaluator-only diagnostics, not inputs available to a runtime policy.

**Inference:** preventing a long visibility-loss detour could improve lap time, but the paired trajectories do not identify whether steering, gas, or an earlier route difference caused the one repaired case. A road-loss intervention must be triggered by pixels before the road disappears, preserve valid finishes, and be compared on fresh matched cells. Earlier road-memory and braking variants on a different controller had mixed or failed results, so these consumed cells do not justify retuning that family. The separate action-conditioned forward-model feasibility study may provide an earlier visual signal; predictive association alone is insufficient to claim a safe counterfactual action.

## Later independent TUNE result

The fixed direction-memory intervention passed fresh TRAIN 22/24 versus 21/24, then failed independent TUNE 22/24 versus 23/24. On TUNE it repaired 2:5067 but lost control finishes 2:5066 and 2:5071. The first changed actions in those failures occurred around decisions 138 and 55, respectively, when fewer than three road-center rows were visible. In both cases visible road returned within 12 decisions of the first intervention, but the eventual drive retired off track. This temporal sequence does not prove that the correction itself caused the final retirement; it does establish that a short visibility-recovery metric cannot substitute for valid completion. These TUNE cells are consumed and cannot set a revised direction or duration. See the [registered TUNE outcome](haic2-road-memory-fresh-tune-20260929/outcome.md).
