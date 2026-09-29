# Traction-aware positive acceleration continuation

Goal: preserve all finishes and reduce every matched lap time toward half the control. No arbitrary fixed low speed target. Full pedal trial completed: all four directions 0/6 versus control 6/6, with initial acceleration followed by first-bend loss.

Hypothesis: full rear-wheel engine torque consumes combined longitudinal/lateral tire force. Use the public physics torque equilibrium to request useful positive drive while allocating grip to steering. This remains a hypothesis until wheel-slip telemetry is collected.

Keep four independent steering directions: existing pixel steering, metric pursuit, tangent feedforward, and constrained path rollout. Apply common positive traction governor, no brake, no fixed speed cap. Accelerator uses chassis HUD speed, requested steering curvature, public engine/radius/friction constants. Rear-wheel and steering HUD decoders are diagnostic only until compared against evaluator truth; do not feed an unvalidated wheel decoder into control.

Torque budget: gas = .0054*(speed/.54+5)*sqrt(max(.01,1-lateral_fraction**2)), clipped [.03,1]. Lateral fraction = min(.995,speed**2*abs(tan(clipped steer))/3.24/150). Constants 400 N, .54 radius and40000 engine power are public;150 lateral acceleration is an explicit approximation, not measured certification. Low gas can still mean full available traction at low speed. This is an actuator allocation experiment, not a parameter sweep.

Evaluator records pre-action wheel omega, wheel joint angles, actual velocity, yaw rate, mass, and last applied wheel gas. Agent sees only pixels. HUD diagnostic rear wheel sum uses two bars together, approx grayscale44/255, width4.2 and height .021 peromega. Rasterization/finite ROI caveats reported.

Comparison: same consumed TRAIN tracks1-3 x seeds38300,38302; 5 arms including unchanged control;30episodes;1200 decisions each;2CPU2GiB1800s. Launch differs intentionally, no equal-action-prefix requirement. Falsifier: loss of finishes, no actual grip improvement, invalid actions, unvalidated data leak. Completion first; all matched lap ratios <=.5 is full target, partial gains labeled. No release or site action. Prior checkpoints preserved.

Implementation: separate traction runtime; evaluator telemetry; frozen manifest and separate design/execution events under standing local authorization; read-only review; full batch integration before revisions.
