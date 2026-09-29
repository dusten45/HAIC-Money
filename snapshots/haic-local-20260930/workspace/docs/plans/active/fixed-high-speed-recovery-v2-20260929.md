# Recovery continuation at unchanged target60

Continue the original four-direction batch with a second candidate in its memory and impact directions (six candidates total, at most two/direction). First direction screen keeps all three original finishes but has not repaired the observed stuck cells. Keep original code and ZIP. User accepts short excursions; completion and continued progress matter, not old1% road rule.

## Two stateful interventions

Memory reacquisition: when row42 disappears, latch last trustworthy road direction once. Maintain a turn of magnitude clip(abs(last road steer),0.35,0.7) until row42 and54 are within8pixels of center for3 consecutive decisions with all7road rows visible, or80decisions expire. Do not re-arm until that stable view is seen. Update remembered direction only outside active recovery. Identical first10actions, same final pedals.

Impact clear: same pixel speed-drop proxy as initial impact candidate (speed<50 and drop>10 from previous4readings). For12decisions use road steering plus existing temporal correction without inherited obstacle term, only when row42and54are visible; otherwise retain original damping. This tests whether stale/local avoidance steers away from recoverable road after collision. It does not claim every speed drop is a collision. Record actual collision timing from evaluator.

Same six consumed TRAIN cells,12episodes total,1200decisions each,2CPU/2GiB/1200seconds. No sweep or speed reduction. Compare each to immutable original damping, verify10decision prefixes and effective action changes, measure progress after actual collisions and preserve original three finishes. Advance only if completion>3/6 with all3original finishes kept and invalid0. Then a separately frozen fresh comparison is required before a new ZIP. No external action.

Standing user authorization2026-09-29 plus current instruction permits implementation and local operations; exact design/execution events before run. Original first-screen outcome remains historical; this continuation does not overwrite it.
