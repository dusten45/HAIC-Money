# Corner-cutting passive gate closed
- Message ID: 20261004T123740Z-c7t2-corner-cutting-closed
- Type: result
- Author/session: c7t2
- Written: 2026-10-04T12:37:40Z
- Reply to: 20261004T122120Z-c7t2-corner-cutting-scope
- Evidence: experiments/koi-corner-cutting-diagnosis-v1.json and primary raw traces
- Status: closed at diagnosis; no candidate or A/B

Eight existing champion episodes/five geometries:62 sharp-corner observations,
55 complete, of which44 actual paths are already shorter than centerline arcs.
Fifteen complete windows lack recorded obstacle/recovery interference. Four on
two geometries have2.00-3.27-unit geometric savings in the bounded inward-shift
family, but none passes combined fixed-speed steering/lateral-demand/reentry
screens. Positive geometric savings are preserved, not relabeled absent.

Road boundaries were SOFT. For1/3184000015/corner284, the predicted.234s offroad
is allowed by the .5s screen, but31.44deg reentry and estimated wheel demand fail.
All path timing/contact/steering predictions are offline proxies, not simulator
counterfactuals or proofs of impossibility. Chords are unconstrained lower bounds.
Grass changes force capacity, not speed directly; unique-tile finish coverage also
limits repeated cutting. Details/limitations and all220 alternatives are in the
artifact and KOI analysis section30.

No policy/Agent edits, new resets, candidate/A-B, fresh/blind/official action.
Frozen champion ZIP/all11 sources unchanged; four focused tests pass. Previous
exit-throttle, target-speed and overavoidance closures remain. No main integration,
commit or push; existing unrelated staging preserved on research/koi-corner-cutting.
