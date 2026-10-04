# Prospective single intervention: antialias-aware circle segmentation

Exact clear-supported V2 camera replay first contacts track2 obstacle atstep97.
The real circle is visible atstep94,26.47m ahead, but a narrow brightness floor
.655 gives only two connected pixels,width1; atstep95 it gives three pixels,
width1. Existing area≥4/width≥2 gates reject both. Atstep96 raster phase gives
four pixels,width2 and detection begins at13.73m true distance and80.4m/s HUD
speed. The controller has only about10m front clearance before contact; it
first brakes0.206 and emits steer0.145, then emergencybrakes0.65 next action.
Camera planned center separation3.26m is not actual reachable separation.

Lower only the component brightness floor from.655 to.60. Preserve upperfloor
.705, roadROI, four-connectivity, component area/shape, surrounding-asphalt
checks, route selection, confidence and controls exactly. Real savedframes94
and95 must fail beforeimplementation and detect afterward. Grass/HUD/whitepaint,
longbrighttextures/tinyspecks must retain rejection. Inspect realprefix detections
against offline obstacle labels before the one fresh mandatory screen.

This accepts antialiased inner orange pixels (.60–.65) and is intended to gain
two actions of maneuver time. It is not a clearance guarantee or a claim that
local image accuracy meets the early10s pace target. No holdout is opened.
