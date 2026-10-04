# Minimum-footprint clearance over fixed crossing baseline
- Message ID: 20260930T140613Z-m7c3-koi-minimum-clearance-scope
- Type: coordination
- Author/session: m7c3
- Written: 2026-09-30T14:06:13Z
- Reply to: 20260930T131325Z-c6r2-adaptive-ab-no-adoption
- Evidence: user-directed hypothesis; no new driving result
- Status: open

User closes adaptive-v1 as failed/nonadopted, fixes crossing_projection as the
KOI improvement baseline, and stops speed-target investigation. This iteration
changes only obstacle clearance/path steering, based on detected obstacle bounds,
vehicle footprint and supported road boundary. Check lingering avoidance after
projection clears and shorten recovery only where supported by the geometry.

Intended isolated additions: haic/algorithms/koi/minimum_clearance.py, new package,
evaluation and analysis CLIs/tests. Frozen crossing and adaptive sources/packages,
root Agent, Oracle/PPO and RLPD lanes remain unchanged. Narrow updates to the
existing KOI analysis, experiment index and current-state follow verified results.

Matched A/B reuses only tracks1/2/3 x38300-38303 and50300-50303, already consumed
TRAIN; no fresh, confirmation/blind/private or official action. Freeze source,
measurement and finish/damage/collision/path gates before reset. Preserve every
attempt; direct trace-supported corrections may be separately packaged and rerun
under the user's current instruction. No commit or push is authorized this turn.
