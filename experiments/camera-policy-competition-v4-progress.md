# Camera Policy V4 Progress (2026-10-04)

This is a development checkpoint, not a V4 fresh-geometry result or a
submission claim. The live `Agent` selector is still
`_CompoundClearingBrakeCarryController`. V3 blind data remain sealed.

## Basis and completed evidence

- V3 screen retained with 29/32 candidate finishes versus 18/32 control.
  Confirmation rejected with 46/64 versus 31/64 because five control finishes
  were lost against a fixed budget of four. Its action/outcome repeats and
  receipt seals validated. See `camera-policy-competition-v3-result.json`.
- Action-exact V3 loss analysis identified two clear-road exits on track 3,
  two post-contact on-road stalls, and one five-contact crash. The first
  full-corridor bend guard and distant-obstacle speed-cap probes failed broader
  consumed-data checks and were excluded from V4.
- The first unconditional row-42 dropout probe used frozen source SHA256
  `bb8c2b669848cdf395d1ac410d659fab83e6af28264dcb83b4458be920a76dc3`.
  Across all 96 consumed V3 screen/confirmation cells, finishes rose 75 to 82
  with no loss of an earlier finish; contacts rose 43 to 44 and damage 8.6 to
  8.8. Crashes stayed 2 to 2 and the shared-finish time ratio was 0.999857.
  It failed the fixed no-contact-increase development gate. The canonical
  action hashes were checked against the V3 receipts. Its ignored source,
  cell receipts and summary are under
  `.haic-artifacts/clear-road-row42-v4/`.
- A refined camera-only recent-obstacle-or-HUD-speed predicate, frozen for its
  current replay at source SHA256
  `7b0b97ea5e27a7922431f587ff5c0d25ba4a5bb4c4f4a41b094fca6c86be1351`,
  was checked on the nine changed-action cells identified by the first sweep.
  Relative to the unchanged V3
  candidate, those cells yielded six finish gains, zero lost finishes, zero
  contact increases, and kept both known no-contact track-3 road-exit rescues.
  One finish gain of the unconditional probe reverted. This targeted result
  does not establish the full development gate. Its 96-cell source-bound replay
  was interrupted after 57 receipts and must be resumed from the frozen source
  under `.haic-artifacts/clear-road-row42-v4/refined-obstacle-history/`.
  Those 57 distinct receipts have matching frozen source hashes and V3 baseline
  action hashes; their partial count is 48 to 54 finishes, zero prior finish
  losses, and 12 to 12 contacts. These are incomplete, ordered development
  cells and do not establish the 96-cell gate.

The inactive candidate and focused tests were committed at `e727438`; 15 new
tests and the 36-test focused agent/submission set passed, as did static source
validation. The unbound V4 runner, byte-identical gate, template and tests were
committed at `bec6063`. Their tests passed 27 with two intentionally skipped
until the final candidate commit/blob pins are set. No V4 protocol or fresh
geometry seed has been bound or evaluated.

## Current blocker and next actions

An independent audit found that remembered obstacle evidence can remain stale
through lost-road frames. Diagnose that state, revise the candidate if needed,
and rerun all 96 consumed cells from frozen source bytes. Require no loss of a
previous V3 candidate finish, both known no-contact road-exit rescues, no
increase in total crashes or contacts, and shared-finish time ratio at most
1.10. Inspect every changed outcome and compare canonical V3 action hashes.

The tracked `_ClearRoadRow42DropoutController` is inactive and contains the
refined predicate; its stale-memory edge is unresolved. The V4 gate, runner,
tests and template are committed but unbound with pending final implementation
commit/blob pins. After a complete passing development audit, finish focused
tests and source review, commit and push the revised candidate and pinned V4
inputs, bind new nonoverlapping geometry seeds, then
run V4 screen, confirmation and blind only as their fixed gates allow. Do not
open V3 blind. Official participant HEAD checked on 2026-10-04 was
`dfb7a2de2178825ca5c5ce20bab01ba67052ba31`. Current local testing is
Windows CPU only; official Linux container certification is unavailable.
