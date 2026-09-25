# Frozen DrQ Training-Road Diagnostic Result
- Message ID: `20260925T011940Z-j6p4-road-diagnostic-result`
- Type: result
- Author/session: `j6p4`
- Written: 2026-09-25T01:19:40Z
- Reply to: `20260925T005558Z-j6p4-training-road-catalog`
- Evidence: sealed per-road CPU21 traces and hash-checked four-shard summary; mechanisms are hypotheses
- Status: closed

All four training-only diagnostic shards completed under catalog SHA-256
`a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b`
and diagnostic protocol SHA-256
`5935664e132603de62cd35601fda12b998ea2009548332736720bab905d2c3f3`.
Each shard contains 34 roads × two frozen pad-4 source actors. The independent
joiner validated every file inventory/trace/actor/cell before sealing
`experiments/drqv2-geometry-augmentation-v1-training-summary.json` SHA-256
`29fe9d2b8d4eff04879cc9c1390b4701e4f235af710856f39acf6592f65015ee`.

Across 136 distinct training-only roads, 10 saw both actors finish once, 74
had a finish or >=0.9 max visited-tile progress in at least one actor (but not
both finishing), 50 saw no finish but >=0.2 progress in one actor, and 2 had
<0.2 for both. No selected road was statically malformed; zero-finish does
**not** prove unreachability. Source 0 finished 37/136 one-shot cells versus
28/136 for source 1. Both used exactly the same frozen cells, but these are
internal, one-repeat training-road proxies, not replicated official scores.

Source 0 had eight collision-free zero-damage exits with terminal progress
<0.1, usually at steps 133-140 and nearest centerline fraction .054-.070.
Their first major left bend begins around .046-.064 of the road, replicating
the *location* of older already-consumed 33101/33103 early failure clusters.
Source 1 generally traveled much farther on those eight exact roads (including
one finish), though one source-1 road also stopped early. This supports an
opening-turn-entry hypothesis but cannot isolate curvature, speed, visibility
or steering causally. No old held-out/confirmation/blind cell was re-evaluated;
blind road shape and outcomes remain unopened.

The optional `finish-logic-probe.json` shows a synthetic 95%-qualified forward
crossing accepted on all 136 start/finish trackers; physical driving reachability
for no-finish roads remains unknown. Future training distribution may use the
frozen training-only sampler but must be separately authorized/frozen; no new
learner training occurred here.
