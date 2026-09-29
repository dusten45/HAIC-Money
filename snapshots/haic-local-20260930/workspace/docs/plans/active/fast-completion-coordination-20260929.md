# Completion-first high-speed coordination pivot

The latest user asks all runs to complete. Steering-only fixed-pedal batch (eight candidates) did not improve completion: initial control7/12, best new5/12; corrected selected probe still no rescue. Preserve the impact_clear checkpoint. Pivot to pedal/steering coordination while retaining target60 for normal straight driving and fast launch. This deliberately relaxes the preceding experiment's constant final pedals; do not claim constant60 in bends. User's original acceleration/control objective and latest completion priority, plus standing local authorization2026-09-29, cover local work without another question. No external action.

Four independent directions:
1. preview_budget: choose a temporary target from visible bend spread and obstacle proximity, braking before rather than after loss. Target60 on straight unobstructed road, floor38 in corners, obstacle cap44. Restore60 when cues clear.
2. actuator_rate: keep fixed pedals but bound action steering changes by the physical actuator's0.24rad per0.08s decision. This prevents requests outrunning wheel motion.
3. impact_traction: on the inherited pixel-speed-drop proxy, temporarily reduce rear-wheel gas to0.18 for the same12-decision window to leave grip for steering. No normal-road speed change.
4. recovery_budget: keep target60 normally; when near-road error exceeds6pixels or row42 disappears, cap target40 until the near error stays below4pixels with full road rows for three decisions. Pixel-only state, no map/speed ground truth.

Probe control plus four candidates on consumed TRAIN track1 seeds38200,38210,38211 (15episodes), requiring3/3 to advance to all12 historical cells. Evidence below3/3 only justifies at most one causal continuation per direction; no sweep. Full12 comparison uses frozen impact_clear control run from fixed60-completion-train-20260929 and verifies exact first10prefixes, hashes and protocol. Then frozen exactZIP on separate fresh cells under its own plan. Completion precedes time; report speed and lap cost honestly. No brief-road-excursion rejection. 2CPU2GiB1800s1200decisions, invalid0, trace/latency/RSS. Three-gate review, official rules UNKNOWN, no release.

New runtime module fast_completion_coordination.py builds on unchanged impact_clear. Existing registered evaluator gains an explicitly declared actor selector in its frozen settings; settings/source hashes registered. Standalone code checks are not added; full driving is the user-requested verification.
