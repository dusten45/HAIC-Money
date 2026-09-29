# Pixel speed envelope: TRAIN activation

Goal: keep the apex policy's valid completion and reduce lap time. The current apex trace has 1,673 decisions with no nearby obstacle, steer magnitude below 0.15 and speed below 50; median gas is 0.146 on those decisions. This is an opportunity, not evidence that more gas will improve completion.

Mechanism: on a fully visible road without a detected obstacle, use the grayscale HUD speed and far-minus-near road-center displacement to set a speed target. If the inherited apex action is not braking, has small steering, and the car appears below that target, raise gas to 0.28. The constants are one preregistered mechanism probe, not a sweep. No simulator state or track geometry is used at inference.

Control: the exact frozen `apex_line` controller and checkpoint hashes. Activation cells: TRAIN 1/43 and 2/102, already used for prior teacher diagnostics. Both arms use the same official local environment, 2,000-decision limit, and full trace. Activation is at least one `envelope_decisions` event and higher mean gas on both paired cells. It fails if the action is unchanged, invalid, or either arm cannot finish where the control finishes. Lap time on these consumed TRAIN cells is diagnostic only.

If activation and completion hold, the next research batch will compare four independent directions on new tune cells: (1) this speed envelope, (2) exit-only acceleration after a detected bend, (3) a brake-distance envelope driven by bend approach rate, and (4) an outer-entry to inner-apex line. Each needs its own frozen implementation and shared control. Do not claim improvement from the TRAIN probe; no held-out or confirmation cell is opened here.

The implementation and v2 run live in this persistent frozen source copy until the gate report is complete. Source, checkpoint, and package hashes are bound by the v2 plan. Local design, implementation, and execution records cite the user's standing 2026-09-28 authorization for repeated local work. No site upload, model confirmation, or submission is in scope.
