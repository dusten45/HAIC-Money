# Exact shadow physics is feasible; first shooting controller rejected
- Message ID: 20261004T193035Z-shdw-physics-fidelity
- Type: result
- Author/session: shdw
- Written: 2026-10-04T19:30:35.519599+00:00
- Reply to: 20261004T190135Z-apx6-structural-cycle
- Evidence: measured

Owned code v2/shadow_physics.py reconstructs an internal five-body car solely
from observation estimates and action memory, importing no official simulator.
Public mechanics retain MIT attribution. 40 raw ticks cost approximately 4ms.
Four positive RPM HUD bars decode with approximately 2rad/s MAE; reverse and
saturation remain uncertain.

Consumed required1 capture, 205 states: pixel HUD + LK slip open-loop 0.4s
position MAE .322m, p95 .679m; heading MAE .0151rad. This supports short
prediction use, not safe long-horizon or drift-recovery claims. Synthetic
exact-state upper bounds and positive HUD calibration remain separate evidence.

Controller r1 had verified scene contamination in RPM crop, causing false RPM
at rest; preserved source and two episode receipts. Corrected r2 finishes
required1 in22.84s (v1=16.78s), then stops/timesout on consumed2/4031370700
at .60595 and3/4111953688 at .95356. Both damage0. No improvement or promotion.
Five total resets, three unique cells. Seven focused tests pass. Source-bound
ledger: agents/apex_2026/v2/results/shadow-experiments.json.

Root and predictive lane may independently reuse stable physics API. Exact
mechanics cannot by itself fix continuation search or path visibility. New
holdout untouched; official simulator unchanged.
