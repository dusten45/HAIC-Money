# Apex fresh tune read-only audit — 2026-09-28

Source: `artifacts/haic-research-v2/apex-line-three-track-frozen-retry-20260928/report.json` and its v2 `integration_report.json`. This note did not open new evaluation cells.

On the registered tracks 1–3 × seeds 296–299, `apex_line` and the selected `full_road_guard` each finished 12/12. Their respective median finished lap times were 21.70 s and 23.39 s. In all twelve paired cells, the apex policy was faster, by 1.54–1.98 s; the median paired difference was −1.71 s. It changed steering in 609 decisions total. The prior `fast_pedal_stable` arm finished 11/12; on its eleven common completed cells, apex minus fast-pedal lap-time difference had median −0.04 s. Thus the measured gain against the selected control mainly reflects the faster pedal policy, while apex steering appears to preserve completion on this tune set. That causal decomposition is an inference, not a controlled ablation of identical trajectories.

The 296–299 tune cells are consumed. The `rule_compliance` gate remains `UNKNOWN`; this is a local Linux CPU diagnostic and not an official score or package confirmation. Further work should use a separately registered fresh split and a persistent frozen source snapshot until the v2 gate report is written. Avoid modifying the apex module while the concurrent anticipatory-bend run is active.
