# ArcClear plus memory constraint: track4 joint regression

The combined sourceb716564e… finishes15.44/19.48/16.54/39.08s, versus its
ArcClear V2 parent15.44/19.08/16.52/20.92s. Track4 grows18.16s and two
contacts instead of one. This diagnosis uses trace comparison and one exact
141-action saved-camera replay of the combined track4. Offline truth labels
use the official map and pre-action trace pose only; no new inference input,
holdout, source change or parameter grid is introduced.

First action divergence is26: a valid transported pass constraint changes
steer−.040→−.031. QP is feasible and solved there; the near-arc guard is
excluded by active pass. This small change shifts subsequent trajectory.

The long delay begins around94–103, not at the first contact. At94/95 a
visible circle and pass_side0 prevent ArcClear's near correction; road-center
gap grows1.0→3.2m. At96 a new pass starts, but the QP falls back because the
ego lower roadbound is positive (2.64m), so the zero ego anchor is outside
the parsed supported corridor. Road-center gap is5.8m. At97 visible forward
road extent is2.35m, below the6m action gate, and recovery begins. At98/99
extents11.8/8.2m invoke early QP fallback and return steers−.037/−.277.
At100 forward extent4.7m again enters recovery, latching steer−.184. By105
road is unavailable; by125 lost_frames26 stops recovery gas, with16.2m
actual center gap. Memory pass_y freezes−2.38m because recovery bypasses
route update. Later contacts426/427 are consequences of the delayed recovery,
not the cause of the initial20-second delay.

No transported memory constraint is added at94–99: either current circles
supply constraints or route selection uses an early fallback. The joint
failure therefore does not establish that preserving a valid remembered
hazard is wrong. It exposes a trajectory-sensitive curve/hazard/partial-road
recovery weakness that the earlier memory correction did not address.

## Near-arc counterfactual

On actual combined frame94, bypassing ArcClear's exclusions selects a short
emitted turn−.133→+.161, improving sampled8m body-footprint roaddepth
0.054→2.774m. This is an **opposite turn**, so both the blanket current-circle
exclusion and preserved no-reversal exclusion block it. At95/96 even the
ungated guard finds no supported short arc (depth0); that opportunity is gone.
Merely allowing guards around benign circles cannot fix94 while retaining
no-reversal. This is an instantaneous camera-model result, not proof that a
rapid direction change is physically safe with steering latency.

The compact probe records actual route invocation separately from hypothetical
bounds diagnostics: recovery frames do not transport pass memory or run QP.
Actual frames94/95 and exact pre-action scalar state are preserved as fixtures
for prospective controls. Counterfactual guard evaluation changes no actual
replayed actions. Source, trace and saved-image hashes bind the probe.
