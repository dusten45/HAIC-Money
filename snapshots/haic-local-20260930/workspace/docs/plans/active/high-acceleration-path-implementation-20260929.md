# High Acceleration Path Implementation Plan

Use superpowers:executing-plans inline. Standing authorization applies.

Goal: high default acceleration with future-path steering and measured completion/lap-time endpoint.
Architecture: pixel corridor → metric path → pursuit/tangent/predicted rollout → finite bounded action. Existing coordinator is control and missing-geometry fallback.
Tech: NumPy, existing official Linux evaluation container.
Spec: high-acceleration-path-design-20260929.md.

- [ ] Implement haic_agent/high_acceleration_path_runtime.py with four directions and immutable control behavior.
- [ ] Extend training/fixed60_completion.py actor selection and explicit prefix requirement setting; retain all telemetry.
- [ ] Register manifest/hypothesis/arguments and approvals, run30matchedTRAIN episodes.
- [ ] Read-only independent code review, inspect evidence and actual actions; correct implementation defects in separately registered revision if needed.
- [ ] Integrate exactly three gates; preserve checkpoint and report target status honestly.

Review focus: camera anisotropic calibration; motor steering response sign and joint limits; HUDspeed80clip; obstacle clearance versus car footprint; no hidden geometry used in policy; no artificial success from shortened run or partial finish.
Verification consists of user-authorized driving and static review, not an unsolicited unit suite. No commits of unrelated dirty workspace. BaselineZIP stays preserved. Source changes only between stopped runs.
