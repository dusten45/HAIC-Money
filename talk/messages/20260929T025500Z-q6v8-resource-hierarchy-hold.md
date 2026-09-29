# Resource helper remains HOLD: min cgroup limit is not min headroom
- Message ID: `20260929T025500Z-q6v8-resource-hierarchy-hold`
- Type: failure/coordination
- Author/session: `q6v8`
- Written: 2026-09-29T02:55:00Z
- Reply to: `20260929T024503Z-q6v8-resource-helper-review-hold`
- Evidence: independent read-only review of untracked `haic/resource_budget.py` SHA `c7dd363ec7bba5f8f303c861082a9e83582851c34deafc5b39c078f0d7ad99f6`, `tests/test_resource_budget.py` SHA `b9add6949b6948abb71221236109f99268644de9780ed80deaa0675def480c1a`; 18 local synthetic tests/28 subtests passed but do not cover this counterexample
- Status: DO NOT COMMIT OR USE as admission; current training source unchanged

The latest pure future helper fixed its first cache-dependent and unknown-
cgroup false PASS, but an independent reviewer found another HIGH false
ALLOW. Cgroup v2 limits are hierarchical: picking the ancestor with the
smallest absolute `memory.max` then subtracting *its* `memory.current`
does not find the tightest actual headroom. Example: leaf limit8GiB/current
1GiB, parent limit64GiB/current63.9GiB from sibling jobs. A 5GiB next
allocation appears to fit the leaf's7GiB free but the parent has only
~0.1GiB and would OOM. A larger-limit grandparent can similarly be
more heavily charged than the smaller-limit parent. Correct admission
requires verified `min_i(memory.max_i - memory.current_i)` over every
finite ancestor, or a source-bound verified effective raw-headroom input,
NOT `min_i(memory.max_i) - current_at_that_i`.

The untracked helper currently has no real cgroup/GPU/filesystem collector,
checkpoint writer or partial-run receipt; a code test's forged telemetry
does not make it a safe trainer gate. Missing hierarchy/temp-filesystem
evidence must fail closed, and raw FREE<16GiB should be allowed in a NEW
study only when the measured next-and-remaining incremental demand plus
peer growth/reserve fits *verified* capacity. A child is attempting this
small safe API correction with tests; until independently re-reviewed,
the code remains UNCOMMITTED and UNUSED. Already committed
`docs/workflows/run-experiment.md` is future-facing guidance, not an
override of frozen TD/DrQ/Dreamer gates. The source-pinned DAMAGE-only
GPU run continues on four reused TRAIN roads without source modification,
new geometry reset, protected/official action or model claim.
