# Clear-road supported ridge V2

Standalone `fast_clear_supported_agent.py` embeds exact frozen clear-ridge V1
(source86230b49…) with only its public class renamed `_ClearRidgeReference`.
The appended public `Agent` overrides only `_ridge`. The original `act`,
confidence gate, hazard/recovery decisions, controls and all other methods
are byte-preserved in the prefix; lineage checks verify this transformation.
Inference imports only NumPy and uses camera observations only.

The raw ridge is truncated strictly before its first node with boundary-distance
support below5.8m. Fewer than five remaining nodes returnsNone, retaining the
original confidence decision. This is a camera heuristic, not a vehicle-motion
or clearance guarantee.

Saved startup frame162 and horizon frame92 regressions fail on V1 and pass
with the guard. Valid horizontal-bend frame65 keeps its exact route and action.
Obstacle and unsupported-ego decisions remain exact, with finite action, input
immutability, reset and import checks. Seventeen focused tests pass.

Fresh source-frozen required benchmark (warmup excluded, official finishes):

| track/seed | V2 lap | contacts | V1 lap |
| --- | --- | --- | --- |
|1/516237|15.18s|0|15.18s|
|2/644062|30.24s|3|29.62s|
|3/1007|18.50s|0|19.60s|
|4/18800|18.38s|0|DNF,progress0.534|

The guard restores a zero-contact track4 finish and improves track3 by1.10s.
Track2 becomes0.62s slower with three contacts. This is a measured geometry
improvement on some tracks, not general pace improvement. All four finish,
but none meet the strict13s goal. The original early10s target remains unmet;
no adoption or holdout evaluation is claimed.

Sourcee5ce693a… and receipt are bound in the lineage JSON and zero-context
source patch. Full action traces remain ignored artifacts.
