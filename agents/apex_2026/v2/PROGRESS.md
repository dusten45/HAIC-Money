# Structural cycle: measured limitations

Research remains active; no v2 candidate is selected. Original four-track
10–13-second target remains unmet. New holdout is still unallocated.

The geodesic r1 source finishes all four required cells, but mean20.37s is
slower than v1's18.53s. Its consumed regression result is22/24: it rescues both
old failures but loses two old successes. This is not a completion improvement.
See [full source-bound comparison](results/geodesic-r1-regression.json).

Isolated rolling force-budget replacement finishes only2/4 required cells.
[Force results](results/force-experiments.json) retain all six new episodes.
Stronger existing brake authority alone finishes3/4 required cells and rescues
only one of two old failures. [Brake results](results/brake-experiments.json)
therefore reject this change as a standalone improvement.

The rolling predictor also uses the commanded steer as an instantaneous wheel
angle, although physical steering saturates at0.4rad and moves at up to3rad/s.
Large commands can exaggerate predicted coast drag and cancel needed braking.
Restricting use to the measured command domain fixes that branch but finishes
required4/4 at mean18.96s, slower than v1, and rescues only one of two old
failures. [Actuator results](results/actuator-experiments.json) retain all six
episodes. Independent review finds40 frames with command inside0.2rad while
HUD wheel angle is outside it, and524 frames using the model without valid slip.
Thus a command-domain guard is not a validated physical-state estimator.

Four focused tests pass for the brake/actuator probes. They check branch behavior,
not physical guarantees; actual simulation receipts determine rejection.
Ongoing work targets duplicated obstacle clearance and state-aware physical
prediction. Neither has yet established a matched improvement.

Further isolated screens preserve the same conclusion:

| Candidate | Required finishes | Required laps, seconds | Prior failure probes |
|---|---:|---|---:|
| Current calibrated HUD | 3/4 | 16.48 / 20.04 / 18.38 / DNF | 1/2 |
| HUD + measured-yaw/wheel throttle cap | 4/4 | 16.54 / 20.32 / 18.44 / 18.52 | 0/2 |
| Internal physics pedal allocator | 1/4 | DNF / 21.50 / DNF / DNF | 0/2 |

These are new simulations, recorded in speed-experiments.json,
yaw-experiments.json and exact-pedal-r0.json. The yaw variant's required mean
18.455s is a small speed gain, not a matched generalization improvement.

No-reset [speed-budget diagnosis](results/speed-budget.json) attributes71–83%
of required driving time to path/tracking curvature limits. Apparent unnecessary
braking occupies only0–0.40s perlap, so a scalar pedal fix cannot explain the
3.78–7.68s reductions needed for13s. Racing-line and continuation planning are
the next structural hypotheses; observation accuracy alone is insufficient.

The independent internal physics helper uses the public force equations, but its
state reconstruction is approximate. Reverse or saturated wheel indicators are
ambiguous; joint angular rates, contact and damage are not fully observed.
The published0.4s residual audit uses zero slip when flow fails, while controllers
may propagate modeled slip. It does not validate long dropout sequences.

The complete [scoreboard](EXPERIMENTS.md) includes all source/configuration
variants, execution defects, pending cells and reused receipts. It separates
small screens from full regression coverage.

Frenet refinement plus single clearance (r6) finished its initial8/8 screen,
required mean18.60s, but the full consumed regression finished only20/24.
It rescued both old failures while losing four old successes. Directional
footprint r7 also finished8/8 with required mean18.575s, but broader validation
finished19/24. All four r6 losses remained and another prior success was lost.
Neither result meets the18.53s required-mean or24/24 selection gates.

Recovery r1 preserves all required P1 actions and laps exactly, and rescues the
consumed track2 failure in19.96s without damage. Track3 remains trapped after a
collision. This is a verified recovery mechanism, not a full24 result or adopted
candidate. New holdout remains unopened.

An obstacle-triggered joint steering/braking filter preserves four required
finishes with mean18.485s and retains the track2 recovery, but loses the same
track3 cell earlier through a different road-departure failure. Its5/6 screen
is not enough for selection. Diagnostic replay is authorized on that consumed
cell to distinguish obstacle perception from candidate road feasibility.

Active hypotheses retain earlier negatives: spatial curvature memory addresses
a measured disappearance of a still-upcoming bend constraint; beam completion
rollouts address a reproduced early-braking branch-pruning counterexample;
bounded terrain planning examines short shoulder use under normal95%-tile and
finish-crossing rules. None has yet established a matched improvement.
