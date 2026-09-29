# Preview pedal implementation and comparison

Goal: implement the user's accelerate-on-visible-straight, short anticipatory braking, early acceleration pattern while preserving completed laps. Standing 2026-09-29 local authorization applies, no approval question. No external upload.

Bounded extension of stable preview_row_repair steering. Four independent pedal mechanisms, no steering algorithm changes: distance envelope (backward braking reachability from pixel curvature/obstacles), time budget (time to bend determines acceleration/coasting/braking), steering budget (current steering demand adds lateral grip reservation to preview envelope), response tracking (recent HUD acceleration adapts pedal feedforward to the same distance envelope). Control is unchanged stable actor.

All read only current pixels, HUD and previous issued actions. Extended road corridor rows54..10 maps to car coordinates x=(px-42)/1.3608,y=(63-row)/1.701. Local quadratic fits estimate curvature, capped by speed85, nominal lateral acceleration100. Unknown horizon terminates at60, to avoid assuming unseen straight. Obstacle target40 at pixel-derived distance with3m body allowance. Preview uses40 nominal braking units and0.16s delay; these are explicit hypotheses, not calibrated guarantees. Steering budget constrains available lateral acceleration using current commanded turn. Pedal logic has mutually exclusive throttle/brake, full gas below target, brake cap0.5 (avoids wheel-lock >=0.9), separate small error region. Invalid/short corridor or initial camera zoom uses original action; no simulator-state input.

Implementation steps:
- Add haic_agent/preview_pedal_runtime.py with target/phase/constraint/response diagnostics and unchanged control mode.
- Add actor import routing to training/fixed60_completion.py; retain existing measurement machinery.
- Register five arms x six consumed TRAIN cells tracks1–3/seeds38300,38302 =30 episodes,1200decisions each,2CPU/2GiB/1200seconds (profile outer timeout1800). Original stable checkpoint SHA7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835. Exact plan hash and design/execution approvals recorded before run. Same first10 actions required.
- Audit hashes, invalid actions, control replay consistency, actual pedal activation, finish/time distributions and failures. No unrelated unit-test additions. Read-only review before execution.
- Integrate three gates. Improvement requires6/6 matching control then faster median finished time; broader12cell comparison and exactZIP validation needed before delivery. No confirmation/blind tuning. Three consecutive valid non-improvements trigger pivot; the distinct preview-pedal architecture begins from prior failures, preserves checkpoint and does not erase history.

Ruling: use existing working checkout because it contains the required uncommitted research infrastructure; preserve unrelated changes. No broad commit/reset. Parameter sweeps are deferred until the intended behavior is observed. A failed mechanism gets diagnosis before revision, not a blanket return to low-speed driving.
