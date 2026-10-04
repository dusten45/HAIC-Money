# Offline ridge tangent, curvature and support uncertainty

This study labels saved camera references using official track center points
and the pre-action hull pose from already consumed track2/4 prefix traces.
It resets the maps for geometry, performs **zero new driving episodes**, and
changes no frozen inference source. Track2 frames0–169 and track4 frames0–149
are considered; ego center must lie within4.8m of the actual center and an
inferred ridge must exist. Forty-six frames are excluded. There are1087
point windows,1085 with enough samples for a camera curvature fit.

Truth is the official center polyline resampled at1m arclength, with a weighted
quadratic parametric fit over±7m around the nearest actual-route projection.
This smooth reference is not exact tyre-required curvature. The camera fit
uses the same halfwindow and Gaussian sigma5m. Speed caps are bounded at100m/s
using190m/s² lateral acceleration. Measurements are reference errors, not lap
results or a physical feasibility certificate.

| reference | fitted windows | curvature MAE,1/m | tangent MAE | mean cap error,m/s |
| --- | --- | --- | --- | --- |
| raw ridge |1085|0.00845|1.30°|−1.31|
| coordinate smoothing±5m |1085|0.00795|1.26°|−1.20|
| coordinate smoothing±7m |1085|0.00716|1.22°|−0.89|
| depth-supported prefix≥5.8m |1037|0.00763|1.12°|−0.94|

Seven-meter camera-only coordinate smoothing reduces curvature error about15%,
but improves average speed-cap bias by only0.42m/s. Jitter does not explain
the full lap-time gap. Actual turns (|truth curvature|>.012/m,331 windows)
have raw speed-cap error tenth/ninetieth percentiles−9.1/+10.1m/s despite near
zero mean bias. Smoothing retains tails near−9.1/+9.2m/s. Removing braking
based on average curvature bias would hide equally important underspeed and
overspeed errors.

The clear/confident subset excludes current circles and circles seen in the
previous12 frames. Its565 windows have raw curvature MAE0.00715/m, tangent
MAE1.23°, mean cap bias−1.39m/s. Smoothing7m gives0.00633/m,1.18°,−1.10m/s.

## Geometry errors beyond quantization

The worst camera ridges include false local bends, not just0.2m walk-grid
noise. On clear/confident track2 frame162, ego center gap is0.54m and no circle
has been seen for64frames. Initial ridge depth falls7.20→5.57→5.05m and the
path develops a false S bend. At arc10m the fitted camera curvature is−.096/m
versus the local truth+.016/m. Frame163 has a related startup error. The
forward-start assumption and partially occluded inner road distort the field.
At other clear straight frames, a partially visible top-edge obstacle or
boundary occlusion lowers distance support and bends the far ridge tail.
Smoothing preserves those geometric errors; see the worst-frame overlay.

A camera-only supported prefix ends at the first point whose boundary-distance
field value falls below5.8m. This is a prospective heuristic relative to the
known6.667m road halfwidth, not a tuned lap improvement. It rejects frame162's
false startup while retaining the valid nearly horizontal track4 frame65 path
unchanged (depth6.49–7.03m). It can trim a false far tail before it influences
curvature fits. Five analytic/saved-camera research tests pass.

The supported clear subset retains533/565 eligible windows (94.3%): curvature
MAE0.00583/m, tangent MAE0.97°, mean cap bias−0.64m/s. Windows with camera cap
more than5m/s below truth fall48→34, and more than5m/s above truth fall30→26.
These comparisons include selection and shortened-fit effects; rejected ridge
windows would require the preserved fallback controller, not an assumed safe
or fast action. No driving improvement has been validated here.

## Residual confidence caveat

An independent quadratic-fit residual model gives approximately91.2% raw-turn
coverage within three estimated curvature standard deviations (302/331).
After smoothing, coverage falls to70.1% (232/331): smoothing lowers residuals
without eliminating systematic geometry bias. Residual/error correlation is
only0.19 raw and0.25 smoothed. A small fit residual should not certify a high
speed cap or justify removing curve braking.

Full1087-window data is stored in ignored artifacts; its hash and representative
worst cases are in the compact JSON. The helper, frozen inference source and
method parameters are bound by hashes. No candidate adoption, additional lap,
or holdout evaluation is claimed.
