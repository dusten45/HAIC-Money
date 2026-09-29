# Implementation ledger

Design and plan written under standing local authorization. Architectural batch implemented inline with separate runtime; previous candidates and baseline ZIP preserved. Read-only reviewer confirmed camera calibration, steering sign, wheelbase, control invariance and no simulator leakage. No blocking static defect. Registered plan hash8ef2ffb09c52a91d4bf65fa72f6040a045dcf8f3d4319e4ea88ad8b22d13e23a has separate design and execution events;30episodesrunning.

Model assumptions to inspect: constant speed rollout ignores acceleration/slip/damage; lateral acceleration bound150isapproximate; near-carroadboundsareextrapolated; finiteHUDROIcanstill saturate. Inherited target_speed is inactive for fullgas candidates; returned gas/brake are authoritative. First10action equality intentionally disabled because launch acceleration is treatment. No unrelated unit tests.

Prior offset batch did not establish physical out-in-out. This batch constructs an actual future path for rollout and evaluates trajectory; a planned path is still not proof the vehicle followed it.

Completed:30episodes, control6/6, each full-gas candidate0/6. Integrated REVISE/STOPPED without release before subsequent source edits. See docs/experiments/high-acceleration-path-20260929.md.
