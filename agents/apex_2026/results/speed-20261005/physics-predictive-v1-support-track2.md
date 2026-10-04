# Exact V1 track-2 unsupported geometry localization

This read-only study uses the completed exactV1/d2bd track-2 replay. It does
not call Agent.act, run a world, search controls, change candidate sources or
open a holdout. A camera-only observer is rebuilt from zero with saved frames
and logged legal actions. Every one of242 sensor records and post-action
observer states matches the recorded replay exactly. Passive parent camera
state is restored only to recompute deterministic perception/geometry.
All geometry support decisions reproduce the replay's branches exactly.

The replay has58 predictive,178 unsupported and6 low-speed decisions. The178
unsupported cases separate as follows:

| First blocking branch | Count |
| --- | ---: |
| Full dense reference below strict1.9m support |165 |
| Legacy centerdepth5.8 prefix too short |9 |
| Parent road missing |3 |
| Initial full-body road guard |1 |

Nearest unknown pixels for the dense failures are141 image-visibility
boundaries,23 detected circle paint/antialias regions, and1 bright unknown.
The initial-body failure adds one bright boundary pixel. No body failure is
attributed to dark car pixels outside its exact fill box; widening that box
has no evidence here. Pixel classifications are spectral/context diagnoses,
not privileged semantic grass labels.

At step38, speed86.49m/s, the reference ends near37m. Its first rejected
sample at35.227m has1.810m depth because image row0 is a visibility boundary,
even though the row0 pixel is asphalt. This supports the separately prepared
V2 strict-supported-prefix fix. It does not support inferring road beyond the
image or changing the actual full-body margin.

At step208, speed81.46m/s, the first rejected reference sample is[-1.5,15.667]m.
The closest unknown pixel is[34,39], gray.611765, next to the detected circle.
The source fills only classified core pixels≥.655 within1.2m, so that paint
or antialias pixel remains unknown. V2 can retain the earlier supported
prefix, but that prefix may still be too short for a fast control plus stop
tail. This is evidence for future perception/reference investigation, not
permission to fill arbitrary grass or relax collision/road constraints.

The one initial-body failure occurs at step235, speed53.97m/s, road slack
−.56055m near camera pixel[65,46], gray.59216. Existing fallback is appropriate
while support is uncertain. Geometry support counts measure opportunity to
plan, not feasible control counts or improved lap speed; fresh root benchmark
results remain required.
