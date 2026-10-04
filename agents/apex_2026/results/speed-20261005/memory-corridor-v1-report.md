# Remembered-circle corridor V1

Standalone source preserves exact early-circle V1 prefix5d9958d1… as a renamed
class. The appended public Agent overrides only `_route`. It calls the base
pass routine once, then adds the resulting transported circle to optimizer
constraints while pass_side remains valid and pass_missing>0. Existing expiry,
road limits, clearance ellipse, fallback, optimizer, detections and controls
are unchanged. Lineage verifies the exact prefix and two method transformations.

Saved missed-detection frames95/97 expose the old corridor bug: reference
clearance to the remembered circle falls2.73/2.40m, even though the base pass
keeps3.70/3.72m. Adding the valid transported constraint restores3.78/3.79m
and changes miss97 steering from−.042 to+.051. Tests verify one transport,
exact visible-circle routes, exact expired-memory behavior, reset and clear/
invalid actions. Fourteen focused tests pass. Camera reference clearance is
not actual vehicle clearance or a collision guarantee.

Fresh four required source-frozen official benchmark:

|track/seed|lap|contacts|early-circle V1 lap/contacts|
|---|---|---|---|
|1/516237|15.20s|0|15.20s/0|
|2/644062|19.72s|0|22.72s/2|
|3/1007|18.78s|0|18.92s/0|
|4/18800|21.14s|0|19.22s/0|

All four finish without contacts. The correction removes both track2 contacts
and improves that lap3.00s; track3 improves0.14s, while track4 slows1.92s.
These are fresh results from one candidate, not best laps selected across
sources. Strict13s and the original early10s goal remain unmet. The candidate
is not adopted, and no additional development or holdout gate is claimed.

Source377b925b… is bound to its receipt, lineage and zero-context patch.
Exact111-action causal camera replays and component/path/steering attribution
are preserved separately. Full traces remain ignored artifacts. Official
simulator and prior inference sources are preserved.
