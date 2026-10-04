# Stationary rear-wheel HUD calibration

This is a privileged synthetic HUD raster study: **0 world steps, 0 driving
episodes**. Official indicator rendering, resizing and grayscale preprocessing
are unchanged. The two receipts bind 452 pooled-meter renders and 250 isolated
bar renders. Existing agents, gates and prior research results remain frozen.
The previously failed torque formula remains negative dynamics evidence.

## Existing pooled measurement

The frozen V5 formula is
`sum(frame[74:83,19:24] where pixel<=.20) / .01485 * .54`.

| Equal rear omega | True rolling speed | Decoded speed, front omega zero |
| ---: | ---: | ---: |
| 0 rad/s | 0.0 m/s | 0.00 m/s |
| 40 | 21.6 | 23.53 |
| 80 | 43.2 | 44.35 |
| 120 | 64.8 | 65.03 |
| 160 | 86.4 | 85.70 |
| 200 | 108.0 | 106.67 |
| 240 | 129.6 | 129.34 |
| 300 | 162.0 | 162.00 |

The nearest front-right blue gauge bleeds into column 19. Changing its omega
from zero to 400 rad/s shifts the pooled rear estimate by as much as 4.706 m/s.
Changing the front-left gauge alone has no measured effect. With front omega
160 and equal rear omega sampled at each integer from 120 to 200, rear rolling
speed error ranges from +0.368 to +4.259 m/s. The body-speed gauge's bias at
physical 70 m/s adds to the wheel/body subtraction: estimated slip error is
+2.030 to +5.921 m/s in these states.

The two rear bars have unequal column weights in that crop. Swapping rear
omegas 120/200 to 200/120 changes the pooled estimate by 2.710 m/s despite an
identical true mean. The existing >15 m/s gate cannot be explained solely by
these errors; this study does not justify removing that frozen gate.

At rear omega above roughly 350 rad/s the cropped bar begins to lose height.
The sampled omega300–400 interval has error as low as -12.79 m/s, so the
operating-range error figures must not be applied to arbitrarily large spin.

## Isolated columns 20 and 22

Direct measurements show that these two columns separate the purple rear
bars and exclude the front gauge. An offline linear fit over independent rear
omega grids 80, 97, …, 250 gives mass responses of 0.003539897 and 0.003233721
per rad/s, with intercepts 0.020195028 and 0.007968982. Cross terms are numerical
zero. No front change among zero, 200 and 400 rad/s affected either column.

On new equal-omega states sampled at every integer from 120 to 200, fitted
mean-speed error is **-1.875 to +1.309 m/s**. Maximum individual rear-wheel
error is 1.933 m/s. Swapping rear120/200 preserves the fitted mean within
0.5 m/s. These are stationary sensor results, not a validated torque controller.

Both crops retain identical raster patterns for five consecutive integer
omega samples: a measured true-speed span of **2.16 m/s**. The original
800-pixel indicator height moves one raster pixel per 5 rad/s, corresponding
to a nominal continuous speed interval of **2.7 m/s**. Integer sampling
understates that continuous plateau. One frame cannot distinguish positions
within the same plateau. The observed fit error is therefore not evidence
for sub-m/s wheel-slip sensing.

The body gauge used for slip subtraction is a separate source of uncertainty.
Its same-render decoded speed is about 1.14–2.32 m/s below physical speed over
the tested 50–100 m/s values. Wheel-forward speed also differs from hull speed
during yaw and sideslip. A pooled mean can hide one rear wheel's saturation.

## Force scale and tests

Official tire longitudinal force before saturation is
`82 * (0.54 * omega - wheel_forward_speed)` N. Its maximum zero-lateral-force
slip budget is `400 / 82 = 4.878 m/s`, and the budget shrinks in a curve.
An individual sensor error of 1.933 m/s corresponds to about **159 N** of
longitudinal-force uncertainty before considering body-gauge or pose error.
That is a large fraction of the available rear force in a fast turn. This
calibration supports coarse spin detection, not a confident 1–2 m/s torque
target or adoption of the already failed torque formula.

The tools execute behavioral assertions against actual official rasters:
monotonic finite pooled readings; zero-frame zero readings; fixed rear meter
independence from body speed; isolated-column independence from front bars;
new-state mean/individual error under 2 m/s; swapped-wheel mean consistency;
well-conditioned bar response; and unchanged official/helper source hashes.
All checks passed. Receipts and source hashes are in
`physics-followup-provenance.json`; raw trials for the separate side study are
kept under ignored artifacts.
