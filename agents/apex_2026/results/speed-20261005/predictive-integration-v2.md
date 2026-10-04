# Predictive integration V2 supported horizon

V2 is a separate source and builder. V1d2bd8a08 and its builder remain byte
unchanged. This change removes the inherited pursuit reference's center
clearance5.8m endpoint truncation from the predictive branch. That confidence
heuristic was reducing available reference length in addition to the new
full-body1.9m support and sampled emergency-tail checks.

The branch now obtains the raw ridge, then retains only the first strict1.9m
supported prefix. At most.25m dense chord samples find the first unsupported
point. Original ridge nodes and their minimum-five-original-node requirement
are preserved; an optional final observed supported sample extends the last
partial chord. The method never skips an unknown gap to a later road branch.
All camera field, grass/body support, circle, projection, stop-tail, observer,
scenario, control and runtime constraints are unchanged.

The exact consumed track-3 frame40 fixture demonstrates the cause. V1 ends
at31m. The raw ridge continues to[-2.9748,36.8246]m with only.2124m strict
depth near the image boundary, so using the entire raw reference would be
unsupported. V2 ends at[-2.7214,35.1040]m with1.9330m strict depth and excludes
that unseen tail. With a public synthetic100m/s initial state on the saved
camera geometry, a constant coasting.32s trajectory and locked stop after the
executed first.08s pass the unchanged complete-body/endpoint/tail checks.
This is a visibility/model unit test, not an actual100m/s run on that frame.

Three new tests were observed RED against V1, then GREEN in V2; a fourth
existing grass/body rejection defense stayed GREEN. Combined V1/V2
integration tests pass14/14. Grass islands and missing body support are
still rejected. The builder reconstructs exact bytes, preserves the original
RearClear prefix and verifies V1/source-builder hashes. The zero-context patch
in `predictive-integration-v1-to-v2.patch` is reversible to exact V1 bytes.
No integrator laps, world steps or holdouts were opened. A fresh root benchmark
is required before claiming speed or reliability improvement.
