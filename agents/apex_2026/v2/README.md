# Apex v2 structural improvement

User-directed continuation after the failed v1 holdout. Read [DESIGN.md](DESIGN.md)
and [benchmark.json](benchmark.json) first. V1 remains immutable and unadopted.

Three independent directions replace row-limited paths, inaccurate long-horizon
prediction, and comfort-only braking with explicit geometry/model feasibility.
All old24 geometries are consumed regression data. New holdout24 remains
unallocated until a single new candidate is frozen and pushed.

No performance claim yet; experiments will record fresh source-bound evidence.
