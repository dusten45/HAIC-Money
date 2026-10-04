# Local curvature braking study

`preview_agent.py` estimates curvature in overlapping metric windows and
permits braking distance before a visible turn, preserving pursuit and actual
steering speed ceilings. Its first two required-lap screens finished all four
without contacts but slowed to 20.76–30.60 seconds. Independent review then
reproduced excessive braking from a one-pixel row shift and an unsafe sparse
road fallback. V4 adds bounded median filtering, a global fit when local fits
lack support, and a reserve of vehicle length plus one decision's travel.

Seventeen preview tests and preserved Apex contracts pass. The current source
SHA256 is `f0ccb11d8bcae690cb1a5b3ac98ea0ab9f727f85b9d7d488263c86760f92302d`.

## Fresh complete development result

`development/r4-preview.json` binds v4, parameters `{}`, evaluator `39a28061…`,
and three prior complete development rejections. Its prospective limit was
**15 seconds**, without changing the original 13-second criterion.

- Required laps: **21.94 / 27.16 / 25.12 / 24.44 seconds**, all zero contacts.
- Extra finishes: **12/16**, three on every track ID. This improves completion
  over the fresh hybrid's 10/16 but slows all required laps.
- No lap passed either 13 or 15 seconds. The profile failed; v4 is not promoted.

All four remaining extra DNFs and their contacts/progress remain in the
receipt. In particular, `(1,1764402399)` still stalls at 0.33916. V1/v2 screen
receipts and reversible `preview-v{1,2,3}-to-v4.patch` files under `probes/`
preserve byte provenance; those earlier screens do not count as rejections.

This is a measured completion/pace tradeoff, not a robust speed solution. A
new independent heading-change estimator investigates the noise of very
short second-derivative fits. Holdout remains unopened; root and physics stay
unchanged.
