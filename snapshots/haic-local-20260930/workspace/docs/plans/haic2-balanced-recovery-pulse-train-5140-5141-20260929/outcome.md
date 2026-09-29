# Fixed-pulse recovery activation: REVISE

Plan hash `0aa4eb5a54913c7f73f3f7d728bc23284f2ed8f58aa4068151e18df438955041` ran 18 registered TRAIN teacher-only episodes on tracks 1–3 × seeds 5140–5141. All 18 finished executing; there were zero invalid actions. Twelve hashed NPZ files hold 304 finite four-frame pixel/teacher-label decisions, and every SHA-256 matched its report.

Both directional label endpoints appeared in real, unmirrored simulator states: positive teacher steering in all six left-pulse cells, and negative steering in all six right-pulse cells. The registered activation required full eight-decision pulses in at least four cells per arm. Teacher takeover occurred after four to six decisions in every cell, so full pulses were **0/6 on each arm**. The predeclared activation gate therefore fails. The formal outcome is `REVISE → GATE_REVIEW_REVISE → STOPPED`, no release. These teacher laps are not submission-candidate results.

The observed early takeover suggests a revised mechanism: stop the pulse as soon as pixel-road visibility falls below three rows, and gate on a completed pulse-to-teacher transition with the intended opposite sign rather than eight forced steering decisions. Register that revision on fresh TRAIN cells before using data for student fitting.
