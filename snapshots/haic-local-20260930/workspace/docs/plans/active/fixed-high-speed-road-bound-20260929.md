# Fixed-speed damping continuation: projected road-bound obstacle intervention

Question: does inherited obstacle steering cause the road departures in the high-speed damping controller? TRAIN traces on 1:38200 show inherited steering opposite to the displaced road center before excursions; this is association, not established causality.

This is the second candidate in the temporal-steering direction of the original four-direction batch. Together with pursuit repair the batch has six candidates, at most two per direction. Same six consumed TRAIN cells, no fresh score. Baseline is the original immutable damping result, not the failed pursuit variant.

Pixel-only intervention: compute projected near-road center as current row54 center plus twice its inter-frame change. After identical first ten actions, if an obstacle exists, absolute projected center offset exceeds six pixels, and obstacle steering points away from that projected road center, remove that outward obstacle contribution. Recompose road steering + the original temporal correction and clip once. Otherwise return damping unchanged. Record activation and projection. This fixed geometric intervention is not a threshold sweep. It may cause collision; zero collisions remains required.

Target60, final governor, road tolerance (1% total and <=8 raw ticks continuous), post-launch high-speed fraction >=90%, zero invalid actions, and real finish remain unchanged. No inherited slowing, no qualification relaxation. Compare first-ten action/state hashes to original damping, six full episodes max1200 decisions, 2 CPU/2GiB, timeout600s. Success requires a qualified finish increase with no loss of the three prior ordinary finishes; otherwise retain failure evidence and revise the mechanism, not speed. Final promotion still requires separate untouched evaluation; none here.

Authorization: user's explicit setup-and-proceed instruction and 2026-09-29 standing local authorization. New exact plan plus design/execution records before run. No external action. Implement subclass, focused assertions, registered local profile, six replays, review three gates.
