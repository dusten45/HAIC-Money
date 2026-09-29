# Natural recovery supplement interrupted by Docker

The registered 96-episode TRAIN 5260–5291 collection started under plan hash `1d23eb8848968fa5e579a77ee5340565b749e3117b88f5c15f02ce750fe71595`. Docker Desktop's Linux engine exited before the collector wrote `report.json`. The original shell process handle disappeared. After a normal WSL/Docker restart, no experiment container remained. One partial file, `track1-seed5260-control.npz`, is preserved but excluded from fitting and coverage denominators.

Formal outcome: `REVISE → GATE_REVIEW_REVISE → STOPPED` for infrastructure invalidity, no release. This is not a valid non-improving research cycle. A new run needs an exact new plan hash and unused TRAIN cells. The prior completed 5170–5177 induced data remain intact; the natural coverage shortfall is unresolved.
