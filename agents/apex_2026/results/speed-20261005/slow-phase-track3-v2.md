# Exact track-3 slow-phase diagnosis

One prefix replay of `(3,1007)` reproduced all 120 saved actions and post-step
speeds exactly. The source stayed frozen at
`e5ce693a4b9ec00d7ecbab8abf71d4a2c350ab95b4f816ae2b6547be91cfc00b`.
No controller, official simulator or holdout changed. Simulator pose, velocity
and wheels are diagnostic observations; inference received only the camera.

The slowdown begins before road confidence is lost. Actions 67–79 retain the
base confidence controller because a completed obstacle pass starts a 12-action
cooldown. Only action 80 uses ridge; a new circle switches action 81 back to
the base controller. Spin frames are zero throughout actions 67–105, contacts
and damage remain zero, and pre-recovery sideslip at actions 69–81 is only
0.68–3.40 degrees. Front-wheel angles reach each emitted command by action end;
this is not steering saturation or persistent joint lag.

In the tight bend, row centers and the clipped-edge reconstruction produce a
far lateral reference. At camera 78, a 16.146 m preview selects `x=-37.169 m`,
giving curvature `-0.045267` and steer `-0.119839`. Its near road center at
8 m is `x=-8.160 m`; using that point gives curvature `-0.124975` and bounded
steer `-0.4`. The distant reference weakens the current turn as the near road
turns left. The unmodified speed target already brakes to 32.705 m/s.

An offline camera check sampled the constant-turn center arc over 2–8 m and
the rigid vehicle corners over 0–8 m (longitudinal -2.4/+2.6 m, lateral ±1.6 m):

| Camera | Original/8 m-preview steer | Center minimum depth, original/short | Corner minimum depth, original/short |
| --- | --- | --- | --- |
| 77 | -0.20049 / -0.4 | 3.627 / 5.284 m | 0.643 / 3.294 m |
| 78 | -0.11984 / -0.4 | 1.561 / 4.176 m | 0 / 2.118 m |
| 79 | -0.16432 / -0.38047 | 0.089 / 2.096 m | 0 / 0.109 m |

These are camera-model measurements, not a dynamic or tyre-sweep certificate.
They identify an intervention before the exit; waiting until frame 79 is late.
Guard the base steering reference with the actual emitted near arc, shorten
preview only when needed, and retain the longitudinal and lateral force bounds.
The saved camera 78 provides a regression before physical road support is lost.

After action 80, road offset is +5.42 m with all four wheels touching road;
after 81 it is +6.77 m with three. At input 82, ego support becomes an isolated
two-pixel component with one-row span, so confidence correctly invokes recovery.
Actions 82–96 spend 1.20 s in recovery with target frozen at 50.705 m/s;
brake 0.35 and low gas cycle the actual speed down to 1.94 m/s. Removing the
confidence warning would conceal the preceding tracking failure.

`slow-phase-track3-v2.json` binds source/script/trace/cameras and contains all
120 compact records plus detailed onset checkpoints. The complete diagnostic
records and all cameras are saved under `.haic-artifacts`; actual float32 camera
fixtures 72/76/77/78/79/80/81/82 are in `slow-phase-track3-cameras.npz`.
