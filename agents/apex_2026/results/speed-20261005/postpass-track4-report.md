# T4 post-pass cooldown blocks a supported bend reference

One Arc clear V2 prefix replay reproduced all 140 actions and post-step speeds
exactly, with source `15478c179ff1c8fa2c7dfe12bf65eaba91ad215fc250bde893ce65d7eebcd5ba`
and unchanged official files. This is a diagnostic prefix, not a new lap.

At action 101 the active pass clears with missing count **2**, below the
expiry threshold of more than 4. Parent camera transport predicts the circle
behind `y=-5 m`; the actual diagnostic circle is at `y=-6.647 m`, with
**3.047 m rear clearance** after including its radius and the car's rear.
At action 100 actual rear clearance is still -0.049 m. This is a completed
rear pass, not an unconfirmed missing-object expiry.

Actions 101–104 have no active pass or current circles. Their supported ridge
has 14/12/11/11 points, minimum chord depth at least 5.989 m and maximum
diagnostic road-center error at most 1.344 m. The 12-action cooldown nevertheless
keeps them in the row-based confidence controller (remaining 11/10/9/8).
At 103 its extrapolated row center has 11.042 m error at `y=20 m`; the supported
ridge's maximum center error is 0.538 m. The near-horizontal bend is poorly
represented by a single `x(y)` reference.

| Action | Actual offset after action | Wheels touching road | Actual rear clearance before action |
| --- | --- | --- | --- |
| 101 | -0.09 m | 4 | 3.05 m |
| 102 | +1.82 m | 4 | 6.74 m |
| 103 | +3.70 m | 4 | 10.16 m |
| 104 | +5.80 m | 3 | 13.35 m |
| 105 | +7.88 m | 1 | 16.47 m |

Input 106 loses ego support (two pixels/one row) and correctly starts recovery.
Actual speed then reaches 1.539 m/s at 111; cooldown remains frozen through
invalid road decisions. No contact occurs in this prefix.

An offline same-camera counterfactual clearing only cooldown selects ridge
at 101–104 and strengthens the same negative turn: -0.400/-0.318/-0.290/-0.284
versus -0.362/-0.178/-0.156/-0.147. This is not a driving or dynamic-safety
result. It supports one bounded intervention: release cooldown only on an
observed pass transition whose unchanged camera transport clears the rear;
retain expiry, invalid-road, current-circle and all existing ridge checks.
World rear labels quantify the diagnosis and must never enter inference.

The complementary current-circle audit finds only cameras 94/95 offer an
ungated supported short turn, and both reverse turn sign. Their rejection is
consistent with the previous causal correction; no same-sign benign-circle
example appears in the examined parent cameras 80–104.

`postpass-track4-arc-clear-v2.json` binds source/script/helper/trace and full
diagnostic records. Camera fixtures include input 101 and the onset around
103–106. The two `postpass-*-camera-audit.json` files use saved cameras only,
with zero new driving episodes. No source changes or holdout were made here.
