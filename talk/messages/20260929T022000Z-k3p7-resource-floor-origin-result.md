# Local TRAIN resource floors are agent-chosen safety gates, not official quotas
- Message ID: `20260929T022000Z-k3p7-resource-floor-origin-result`
- Type: result/coordination
- Author/session: `k3p7`
- Written: 2026-09-29T02:20:00Z
- Reply to: `20260929T021001Z-k3p7-resource-policy-audit-scope`
- Evidence: current official Participants README §2/§5 (https://github.com/2026-HAIC/Participants), inaccessible dynamic detailed website body, local `docs/competition/restrictions.md:26-34`, Git `07700fd`, `0da3bce`, `6aa3156`, historical TRAIN preflight source/protocol/talk receipts, completed raw100k primary runtime
- Status: facts verified; future documentation policy correction proposed, active training unchanged

No official text we could read mandates a local training minimum of 16GiB
FREE cgroup RAM, 5/16GiB FREE disk, idle GPU/CPU, or 2/6h wall. The official
Participants README explicitly says training environments may be freely
configured; **submitted Agent** Linux CPU runtime alone has 1,024MB process
memory, import/construct10s, reset5s, act5s, and separate install/package
limits. The current official website's public dynamic pages did not expose
a detailed resources-rules body, so this is a checked README finding, NOT
proof that an unreadable website states nothing else.

The local origin is dated source/protocol design, not the user's numeric
request. Session t5m8 froze the TD pilot at >=16GiB raw cgroup FREE,
>=5GiB disk FREE, <=7,200s wall in `scripts/train_tdmpc2.py`/v2 protocol
(commit 07700fd). V2 did NOT reset when raw cgroup free was ~14.2GiB,
despite ~25GiB `inactive_file`; later formal preflight passed with only
5.5GiB disk but the agent separately HELD launch for upcoming checkpoint
and competing DrQ writes. The original v1 run failed from planner reset,
not OOM; no primary TD resource-induced crash was found. A coding subagent
then proposed >=16GiB disk and main accepted 21,600s wall for the new100k
budget (commit 0da3bce); the active DAMAGE runner inherited 16/16GiB,
21,600s exactly (6aa3156). Git author `dusten45` is shared account
metadata, NOT proof the human owner requested these numbers. Dreamer/DrQ
use separate 8/12/16GiB RAM or8GiB GPU/no-other-CUDA gates; RLPD trainer
has no analogous numerical free-resource minimum. This is not a unified
competition policy.

The concern is concrete: `scripts/train_tdmpc2_long.py:253-277` rejects
lower declarations and recomputes `memory.max-memory.current` and FREE disk
against the same 16GiB amounts at preflight, startup, each reset, every256
updates and before each snapshot; `scripts/train_tdmpc2_damage.py` pins
and reuses them. File cache charges `memory.current`, and a run's own
growing replay/checkpoint reduces its FREE headroom, so a repeated FULL
initial-growth floor can abort a still-viable partially completed job.
But blindly dropping every guard is unsafe: completed raw100k process peak
RSS was ~16,124MiB (pre-final-save ~11,783MiB); four immutable model
checkpoints together occupied ~6.74GB, total run ~6.77GB, and no cgroup
swap is available. The reported max CUDA *allocated* was only ~252MiB,
which does not measure driver/context/reserved peak; no hard universal GPU
free quota follows from it.

For FUTURE separately sourced studies, prefer **measured per-run remaining
demand**: admit on comparable peak *incremental* RAM plus documented peer
growth/reserve; consider clean reclaimable `inactive_file` with dirty/writeback
and cgroup/host bounds rather than equating raw charged cache to irreversible
memory; budget remaining checkpoint/log bytes and competing writers instead
of demanding another16GiB FREE after files were written; check GPU measured
reserved/context plus peer growth rather than insisting no other CUDA app;
CPU occupancy/threads and throughput merit warnings, not a fabricated
CPU-free quota. Reject genuinely infeasible next work, record resource
snapshot/partial intent, and do not pretend an interrupted run exact-resumable.
These are policy design criteria, not an implemented/proven new validator.

I intend focused changes in the clean shared
`docs/workflows/run-experiment.md`, only the TD-owned paragraphs of the
already-dirty `docs/plans/active/tdmpc2-pixel-online-baseline.md` and if
needed `docs/context/current-state.md`. They must say v2/long numeric
gates belong ONLY to those frozen experiments, not "any newly authorized
study." DO NOT edit source-pinned current DAMAGE/RAW trainers/protocols,
another lane's untracked DrQ/Dreamer code, protected/official sources, or
user/peer dirty hunks. Future reusable resource helper/code requires a
separate test/source/protocol once current run ends. Current shaped TRAIN
process remains alive and 70k checkpoint sealed under its existing gate;
no reset, official action or model confirmation was performed by this audit.
