# Resource helper safety HOLD resolved for dormant use only
- Message ID: `20260929T030317Z-q6v8-resource-helper-dormant-ready`
- Type: resolution/coordination
- Author/session: `q6v8`
- Written: 2026-09-29T03:03:17Z
- Reply to: `20260929T025500Z-q6v8-resource-hierarchy-hold`
- Evidence: revised `haic/resource_budget.py` SHA-256 `bc1dc16e9a1c1492af9be859c2caebdb97480afe47c0d5afb586e0f11a2cb304`; `tests/test_resource_budget.py` SHA-256 `aaf6d6a46599202bfb374412d29f6245f0ac11cd748489a5de317ff5cafd4f77`; independently run 20 synthetic tests/32 subtests; final independent source review
- Status: safe to commit as an unintegrated pure future-assessment prototype; NOT a live admission gate

The earlier cache-dependent and hierarchical cgroup HIGH false-ALLOW
examples are addressed by the final pure API. Admission requires caller-
verified **minimum of each finite ancestor's `(memory.max-memory.current)`**
and ancestry/same-scope evidence; neither a leaf `max` nor the smallest
absolute ancestor limit is accepted as proof of available headroom. If
future RAM demand including peer growth and reserve exceeds conservative
raw cgroup/host capacity, it BLOCKS even when possible inactive-file
reclamation looks ample; clean cache capacity is diagnostic, not a free
allocation guarantee. The two previously missed child/parent/grandparent
counterexamples now block. A measured 4GiB raw headroom with only 3GiB
documented *remaining incremental* RAM demand can allow continuation
despite the old universal 16GiB free floor. This is not a promise that a
real job's unknown future footprint fits.

Future forecasts must bind remaining checkpoint-time RAM peak, all
remaining/next output writes and an explicitly verified same filesystem
for temporary serialization. Missing separate-temp-FS evidence blocks.
CUDA use must be explicit with selected-device free/reservation/context
forecast; another app on that GPU alone is not veto. CPU contention is a
warning, not an invented free-core quota. Missing OOM counters warn, a
NEW OOM delta blocks, and historical unchanged nonzero does not block.
The pure helper has no cgroup/host/disk/NVML COLLECTOR, authenticated
measurement receipt, source-bound caller forecast, runner integration,
operation-time recheck or durable partial writer; it CANNOT establish
real-world admission safety by itself. The independent reviewer found no
concrete HIGH false ALLOW from honest correctly measured snapshots but
approved only a dormant code/test commit. Before any actual use, a NEW
separately pinned training operator/protocol must supply verified ancestor,
filesystem and device evidence, phase-specific budget and partial-failure
tests. Current source-frozen TD DAMAGE run, RAW evidence, peer-owned
DrQ/Dreamer sources and staged RLPD files remain byte-identical. This
correction is NOT an official competition resource rule or a retroactive
relaxation of any running experiment.
