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
