# Fixed full-brake prediction tails

Four handselected uniform-asphalt conditions compared the four-tire model with
the unchanged official simulator for36 raw steps (.72s). Initial state is
privileged and exact. This is a model diagnostic, not a legal controller, a
map/road-clearance certificate or a timed lap. No new driving episodes or
holdouts were opened.

Each tail holds the actual initial front-joint angle and requests gas0,
brake.9. This resets wheel spin before every tire-force calculation. The
model integrates the compound center with updated velocity and reconstructs
hull position using its fixed center offset, matching the semiimplicit motion
used by Box2D. All144 raw-step pose/error/speed/yaw comparisons are retained in the tracked
`physics-brake-tail-compact-v1.json`. Relative heading integrals are derived
from recorded endpoint yaw and explicitly labeled. Full force/state vectors
remain byte-for-byte in the ignored artifact
`.haic-artifacts/apex-speed-20261005/physics-brake-tail/physics-brake-tail-v1.json`
(SHA25689501b4d73794db63dcacb654f7259641f155871bfa831696dbfe74a7bc0c026).
The compact receipt records its raw-source hash and all faithful-copy checks;
no dynamics were rerun during compaction.

| Fixed condition | Actual initial speed | Actual/predicted stop time |
| --- | ---: | ---: |
| Straight nominal70m/s |70.000m/s |.32/.32s |
| Straight nominal100m/s |100.000m/s |.46/.46s |
| Settled .06rad turn from nominal70 |75.298m/s |.36/.36s |
| Settled .06rad turn from nominal100 |99.968m/s |.46/.46s |

Stop means speed at most .5m/s and absolute body yaw at most .1rad/s. Turn
initial conditions use fixed low-gas warmups, so their measured initial speed
is reported rather than being called exactly70/100m/s. Four distinct states
were used; two earlier model cases had the same initial state and would add
no independent tail coverage.

All prospective fixed-case checks pass. Across the full .72s paths, the worst
hull-position error is .0001724m, speed error .001646m/s, yaw error
.017425rad/s and front-joint error .000000835rad. Every predicted stop tick
matches the official simulator tick. The straight100m/s stop travels about
22m; reaction and geometric hull margins must be added separately.

This supports testing a sampled emergency stop after the executed .08s
control block on confirmed asphalt. It does not justify running a whole
.32s action block before reacting, or replacing coupled tire/brake dynamics
with a constant stopping-distance formula. Camera state uncertainty, grass,
contacts, damage, road curvature and circle clearance are absent here.
The held-joint locked-wheel tail changes turning behavior, so a planner must
sample the predicted hull and obstacles all along the tail rather than
checking only its endpoint.
