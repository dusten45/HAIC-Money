# Fixed high speed steering: design and implementation plan

Goal: maintain a high target speed while finding steering that stays on the road. Slowing down, cutting the course, or reporting only successful lap times cannot qualify as improvement.

Authorization: user requested setup and execution in this conversation; 2026-09-29 standing local authorization covers implementation and local evaluation. Record exact design and execution events before driving. No external action.

## Frozen first experiment

TRAIN only, tracks 1–3 × seeds 38200–38201, six cells per arm. Five arms: fixed-speed corridor control; farther preview; curvature feedforward; temporal damping; image-space pursuit. These are four independent steering mechanisms, one candidate each. No weight sweep. First ten decisions use identical control steering in all arms; all five use the same final pedal governor throughout. No inherited pedal command can override it.

Target is 60 in the existing pixel speed estimator's units, a provisional high-speed target, not a claim of twice baseline speed or a calibrated physical unit. Governor gas=clip(0.12+0.04*(60-estimated_speed),0,0.6); brake=clip(0.02*(estimated_speed-62),0,0.15). Braking is only overspeed regulation; curves, road confidence and obstacles cannot lower target. No simulator state enters an agent.

Compare unmodified corridor steering at that same speed. Preview uses the farther visible center. Curvature uses change in centerline slope. Damping uses inter-frame lateral motion. Pursuit uses an image-space lookahead bearing. Missing road geometry keeps inherited steering; record availability. Preserve inherited obstacle steering. Clip steering at the same inherited limit in all arms.

Qualification: actual finish AND no invalid action AND zero collisions AND center outside asphalt for at most 1% of evaluated raw ticks AND no continuous center-off-road stretch longer than eight raw ticks (0.16 s at 50 Hz) AND speed at least 90% of target on at least 90% of post-launch decisions. Launch exclusion is fixed first 25 decisions for every arm; never exclude later failures. Both pixel-estimated and evaluator physical speed are recorded. Candidate must pass physical-speed maintenance too; estimator errors cannot count as high speed. Less than 25 decisions cannot qualify. Record all road contact counts and center-road membership at raw physics cadence, including near finish. No teleport, changed dynamics or simulator input to policy. Road membership is actual road-fixture containment, not progress reward. Qualified finishes rank first, then ordinary completions and full trace diagnostics; no promotion from this TRAIN diagnostic.

Budget: five × six = 30 episodes, max 1,200 decisions per episode, 2 CPU / 2 GiB Docker, timeout 1,800 seconds. Stop for infrastructure errors and retain partial evidence. A failed candidate stays available for same-cell diagnostics; it does not trigger automatic speed reduction. First failing boundary and identical-prefix hashes guide the next registered cycle. No tune/held-out/confirmation opened.

## Implementation steps

- [ ] Add isolated runtime controller with fixed final governor and four steering mechanisms.
- [ ] Add raw-tick evaluator instrumentation, per-cell immutable reports and aggregate strict qualification.
- [ ] Register typed local diagnostic profile; freeze plan/source hashes and approval events.
- [ ] Run full registered batch and record three gates, limitations and next causal diagnostic.

Ruling: work in current checkout using new files and narrow profile additions; both active experiments use independent frozen directories. Preserve unrelated changes. No commit of the shared dirty tree.
