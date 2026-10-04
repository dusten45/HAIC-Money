# Bounded corner target-speed diagnosis
- Message ID: 20261004T115953Z-c7t2-corner-target-speed-scope
- Type: coordination
- Author/session: c7t2
- Written: 2026-10-04T11:59:53Z
- Reply to: none
- Evidence: user-directed hypothesis; analysis not yet run

Work proceeds on research/koi-corner-target-speed, branched from preservation
commit 2408bed without changing existing staged/unstaged/untracked work.
First analyze only the eight existing frozen_shield episodes under
runs/koi-nominal-trajectory-v1. Hypothesis: current road-shape target mapping
unnecessarily limits entry speed. Frozen submitted champion, lookahead, steering,
avoidance and shield remain unchanged; exit-throttle and overavoidance stay closed.

Intended files: separate scripts/diagnose_koi_corner_target.py, one result JSON,
focused synthetic checks, and additions to existing KOI analysis/current-state/
experiment index. Compare complete corner passages, including failures, not just
instantaneous overspeed frames. Need repeated clean above-target passages in a
consistent shape band on distinct road geometries, with positive road margin and
no contradictory near-limit evidence, before considering a narrow candidate.
Absence of that evidence closes the direction without candidate or A/B.
No fresh cells, blind/official evaluation, blanket audit or champion edits.
