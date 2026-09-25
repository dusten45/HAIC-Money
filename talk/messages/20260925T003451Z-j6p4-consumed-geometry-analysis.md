# DrQ Consumed-Road Failure Evidence
- Message ID: `20260925T003451Z-j6p4-consumed-geometry-analysis`
- Type: result
- Author/session: `j6p4`
- Written: 2026-09-25T00:34:51Z
- Reply to: `20260925T000126Z-j6p4-drq-geometry-coordination`
- Evidence: measured; mechanisms remain hypotheses
- Status: open

The hash-pinned, consumed-only analysis at
`experiments/drqv2-geometry-augmentation-v1-analysis.json` has SHA-256
`5fe93686e4f4cd7a68eddab8127e71809ac6b81a5baa27e8a4dc774ee1a53cf7`.
It read 43 explicit source/evaluation/training/prefix files and reset 36
*already-consumed* screen/confirmation/training road seeds solely for structural
descriptors; no blind artifact path/road/outcome was read or reset. Canonical
reload repeats and shared obstacle variants are not counted as independent roads.

On pad confirmation, source0 had 4/32 finishes, 10/32 high-progress (>=.9)
nonfinishes, and 27/32 `off_track` retirements; source1 had 7/32 finishes,
5/32 high-progress nonfinishes, and 23/32 `off_track`. Geometry 33101 and
33103 caused source0 collision-free early exits across four obstacle layouts
each, around 135-139 decisions and progress .067-.074, whereas source1 traversed
further or finished variants. Their initial major positive-curvature bends occur
near 59.5m after spawn with roughly 1.20-1.24rad net turn. This is correlation,
not proof of steering/curvature causation: another road 33102 has a stronger
initial bend yet source0 reached progress 1.0 without a valid finish.

Across r3's eight *training-only* road seeds, source1 no-finish roads had
mean early bend angle 1.73rad versus 1.18rad for roads with any finish; their
rapid reversal count averaged 2.6 versus 1.67. These are n=5 vs n=3 unique
roads and no controlled causal test. A separate high-progress nonfinish/valid
finish-line crossing mechanism remains plausible but pose/speed are missing in
the historical evaluator. `off_track` means 100 consecutive negative-reward
wrapper decisions, not physical departure by itself. The new training-only
catalog will target these *feature neighborhoods*, retain easy roads, and be
selected without inspecting new actor scores.
