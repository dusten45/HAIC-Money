# Guarded Camera Policy V4 Plan

## Goal and evidence

Develop and independently evaluate a camera-only controller that preserves the
large finish gain of V3 while reducing its finish regressions. V3 screen retained
(29/32 versus 18/32), but confirmation rejected (46/64 versus 31/64) because
five control finishes were lost against a fixed budget of four. Its blind phase
stays sealed. All opened V3 geometry is development data for V4.

Action-exact V3 replays split the five confirmation losses: two no-contact road
exits, two post-contact on-road stalls, and one five-contact crash. A first
current-frame full-corridor bend guard rescued two losses in a prespecified
15-cell development panel without losing any sampled gain. A full 96-cell
consumed V3 matrix is required before implementing or freezing the candidate.

## Candidate and development gate

- Add one `_GuardedBendCorridorController` subclass of
  `_BoundedSideHoldController` in `agent.py`, leaving the live bare `Agent`
  selector on `_CompoundClearingBrakeCarryController` during development.
- Apply the first bend guard only when a full `_corridor_plan` exists, the
  obstacle bounding box bottom is at most 35, and the current-frame corridor
  centers contain a far row (first of 30/34/38) and near row (54/50) at least
  3 pixels apart. The measured bend and base road steer must agree in sign;
  `abs(base_steering)` must be at least .03; the planned pass side must oppose
  the base; and the proposed obstacle steer must retain the base sign but fall
  below half its magnitude. Then return the base road steer through the normal
  slew/pedal pipeline. In diagnostics, verify `_corridor_centers` against a
  direct extraction from `observation[-1]`. No track ID, seed, simulator state
  or hidden vehicle telemetry may enter inference. Do not change the 3-pixel
  threshold based on V4 fresh phases.
- On all 96 consumed V3 screen/confirmation canonical pairs, require no loss
  of a previous V3 candidate finish, at least two of the five V3 control-only
  finish losses rescued, no increase in total crashes or contacts, and new
  shared-finish total time no more than 1.10 times the V3 candidate total on
  the same jointly finished cells. Preserve exact baseline
  receipts and action hashes. If this fails, diagnose and revise only on
  consumed development data, then rerun the full matrix before freezing.
- Treat any distant-only speed-cap relaxation as a separate prototype. It is
  excluded from the first V4 candidate and needs its own full-matrix safety
  proof and fresh evaluation before activation.

## Fresh evaluation contract

- Copy the fixed V3 comparator byte-for-byte (SHA256
  `0e2ea38793d980605c607e1b490eede26d32a60e223e3c70b6386a8cdb2a7a02`)
  into a separately hashed V4 decision
  module. Use the same phase and combined gates: 8/16/8 new geometry seeds
  crossed with track IDs 1–4, 2/4/2 exact repeat pairs, phase net finish gains
  at least 3/6/3, lost control finishes at most 2/4/2, aggregate candidate
  crashes/contacts/damage no greater than control, mean candidate progress at
  least control, and shared-finish total time ratio at most 1.10 (skip only
  this pace check if there are no shared finishes), the geometry-seed cluster
  veto, combined net gain at least 18 across
  128 pairs, and combined noninferiority for each track ID.
- Reconstruct the control snapshot from the exact `f12c5b6` commit and blob
  SHA256 `291d93081a64d49f18507ec7baaba411e06510bb533221ce07ae585bc4e77b61`.
  Reconstruct a distinct candidate base from a new pinned implementation
  commit/blob with the live selector still on the control route; derive the
  candidate snapshot by exactly one selector swap to
  `_GuardedBendCorridorController`. Verify both committed blobs and snapshots
  at bind, restore and before/after every phase. Do not require candidate to
  equal a selector swap of the old control blob. Pin the root model, helpers,
  simulator, cold harness, runner, decision module, template, protocol and
  runtime versions. V4 code, artifacts and seals have their own namespace and
  do not import the bound V3 runner/gate.
- Commit the candidate, runner, decision module, tests and unbound template
  before generating a one-time salt. Exclude every documented historical seed,
  including V3's unopened blind seeds. Commit and push the bound protocol
  before any V4 episode. Test the history scanner against all 32 V3 reserved
  seeds, including unopened blind, and exclude only V4's own correctly
  hash-matched protocol/result. All worker receipts are cold CPU runs; exact repeat
  actions and outcomes must match; unexpected, invalid or missing receipts
  prevent phase acceptance.
- Open confirmation only after an independently audited screen RETAIN and
  finalist seal. Open blind only after confirmation RETAIN and its seal. Never
  alter candidate, gate, runner, source hashes or seeds after binding.
- Promote the exact frozen candidate selector only if every fresh phase and
  the combined gate RETAIN. Check live Agent parity on registered repeat cells,
  run source/package checks and available tests, then create an immutable new
  submission record without overwriting user files. Otherwise record the
  rejection and keep the current live selector.

## Work sequence

1. Finish the 96-cell development audit of the guard and inspect every changed
   finish, crash, contact and shared-finish time.
2. Repair the diagnostic tracer's current-frame `road_centers` attribution so
   future camera claims cannot use a past frame. Verify replay action hashes.
3. Implement the selected subclass with focused behavior tests; run the full
   consumed matrix again from committed bytes and record a development result.
4. Implement and review the separate V4 gate, runner, template and tests. Bind
   fresh seeds only after all code and input hashes are final.
5. Run screen, confirmation and blind sequentially subject to their fixed
   seals, audit all receipts, then promote and package only on final RETAIN.

The official participant rules place every DNF behind a finish and rank
completed runs by lap time. This plan prioritizes finish reliability while
retaining a shared-finish pace limit. The Windows local simulator is evidence,
not Linux submission-container certification.
