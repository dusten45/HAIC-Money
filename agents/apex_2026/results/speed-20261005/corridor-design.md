# Camera corridor smoothing experiment

The root lane tests a new standalone `fast_corridor_agent.py` derived from
frozen path V5. Existing path, preview, envelope and root sources stay intact.
The observed problem is a short obstacle approach/return that creates a large
curvature request and changes as the remaining distance shrinks.

A bounded local quadratic path solve will minimize curvature and path length
inside current camera road spans, with obstacle-relative one-sided constraints.
It anchors the current position and forward heading, retains the chosen pass
side, and biases weakly toward the road center. The existing measured-yaw
steering and grip/speed control remain for the first experiment, isolating the
route construction. Unknown or infeasible corridors retain the old path and
pedals. These sampled camera bounds are not a tire or full-hull certificate.

Required regressions: a clear bend should avoid an unnecessary centerline
excursion; a visible circle must retain explicit clearance; an infeasible
corridor must fall back; finite input/recovery/reset behavior must hold. Run
these against the previous route before adding the new solver, then perform
one fresh frozen four-track mandatory benchmark. No holdout or privileged
inference input is used. The original four early-10-second goal remains open.
