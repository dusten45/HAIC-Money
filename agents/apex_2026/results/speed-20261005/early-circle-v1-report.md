# Antialias-aware early circle V1

Exact clear-supported V2 source is preserved as a renamed prefix. The public
Agent overrides only the original `_circles` method, with the sole method
changebrightnessfloor.655→.60. The upperceiling.705, roadROI, connectivity,
area/shape and surrounding-asphalt predicates are unchanged. No inference
imports other thanNumPy or privileged simulator inputs are used.

Actual saved track2frames94/95 are RED with the old narrow core and GREEN
with antialiased core segmentation. Earlier detections are26.75/20.50m, with
offline label errors0.58/0.49m. Grass,Hud,whitepaint,longbrighttextures and
specks retain rejection. Thirteen focused tests pass. Auditing all118 saved
prefix frames finds zero detections more than1.5m from an official obstacle;
this finite local audit does not certify detection precision on all tracks.

Fresh four required source-frozen benchmark:

|track/seed|lap|contacts|prior clear-supported V2|
|---|---|---|---|
|1/516237|15.20s|0|15.18s|
|2/644062|22.72s|2|30.24s,3contacts|
|3/1007|18.92s|0|18.50s|
|4/18800|19.22s|0|18.38s|

All four officially finish, but none meet13s or the original early10s goal.
Track2 improves7.52s and one contact; the other tracks are slightly slower.
The old first contact atstep97 is replaced by contacts at98/99 on the same
approaching obstacle. Earlier detection reduces the severity/delay but does
not establish a collision-free feasible pass. Candidate is not adopted.

Exact source transformation and sole method change are verified in lineage
JSON, with a zero-context patch. No holdout is opened. Full action traces
remain ignored artifacts, and follow-up causal replay investigates remaining
contact rather than introducing an unvalidated parameter grid.

The frozen source includes one extra blank line at EOF, reported by Git
whitespace inspection. Its evaluated bytes are retained exactly.
