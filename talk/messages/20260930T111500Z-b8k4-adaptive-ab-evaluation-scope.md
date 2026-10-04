# Unchanged paired A/B on already consumed TRAIN roads
- Message ID: `20260930T111500Z-b8k4-adaptive-ab-evaluation-scope`
- Type: coordination
- Author/session: `b8k4`
- Written: 2026-09-30T11:15:00Z
- Reply to: `20260930T104753Z-b8k4-adaptive-avoidance-result`
- Evidence: direct user evaluation instruction; no new outcomes yet
- Status: preflight; no model edits

User explicitly requests actual paired evaluation first: frozen crossing source
baseline versus adaptive ZIP `5f7057a4...`, identical already-consumed TRAIN cells.
Proposed complete development cohorts: tracks1/2/3 x38300-38303 and50300-50303,
subject to source/exposure/active-allocation audit. Exclude old held-out49300 and
51300 cohorts and all confirmation/blind/private cells. No freshness claim.

Models remain unchanged for the first full48-slot evaluation. Reuse frozen
`c4e224d` run_episode/env_factory in the restored CPU21 runtime with full traces;
external read-only telemetry records detector events, true speed, stable world
obstacle IDs and passage timing without exposing privileged state to either Agent.
If existing traces lack the requested mapping, add an isolated evaluation-only
operator under scripts, not an algorithm/runtime change. Freeze hashes, conditions,
measurement and completion/safety/time adoption rule before any reset.

At most one corrected full comparison is authorized, only after a clear trace-
supported defect. Preserve original outcomes and every incomplete attempt. No
training, official submission/confirmation, or automatic candidate promotion.
New experiment/run artifacts and narrowly scoped evidence-router updates are
planned; root Agent, both model implementations and other lanes stay untouched.
