# Fast clear road with committed hazard reference

Early-circle V1 finishes15.20/22.72/18.92/19.22s, contacts0/2/0/0.
Committed-preview V3 finishes19.42/25.22/21.50/22.74s, contacts0/0/0/0,
but track4 still has13 full off-road samples. Neither meets the original goal.
The new experiment embeds both exact sources with public class renames only.
These measurements motivate a hypothesis; they cannot be combined into a lap.

Both controllers observe every camera. Select committed-preview only when its
current accepted pass plan exists; otherwise select early-circle. Synchronize
actual emitted steering into both controllers and preview's metric/robust
steering memory. Preserve each controller's internal proposed target ramp;
the public target reports the selected controller. Reset clears both states.
Revalidation can reject an old commitment after the faster controller moves.
No remembered invisible plan is asserted safe, and no new clearance or tyre
certificate is claimed. This selection may still fail dynamically.

Meaningful selection, steering-memory and reset regressions precede the new
standalone source. Then one frozen four-cell mandatory benchmark is required.
No development expansion, holdout or adoption is authorized by this screen.
