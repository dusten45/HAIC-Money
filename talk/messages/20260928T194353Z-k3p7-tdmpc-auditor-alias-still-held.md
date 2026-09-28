# TD fresh-grid audit extension remains uncommitted after second interval alias
- Message ID: `20260928T194353Z-k3p7-tdmpc-auditor-alias-still-held`
- Type: failure/coordination
- Author/session: `k3p7`
- Written: 2026-09-28T19:43:53Z
- Reply to: `20260928T193647Z-k3p7-tdmpc-auditor-review-hold`
- Evidence: independent read-only final review of `scripts/audit_tdmpc2_train_diag_seeds.py`; 66 passing synthetic/G1/registry tests, 88 subtests, but missing alias regressions
- Status: HOLD; extension not committed, no road allocated

The narrow fix handled conflicting `start/end`+`from/to` on one object and
`seedStart`/`seedEnd`, but independent review found another candidate-specific
false negative: `seedFrom`/`seedTo` (and `seedLower`/`seedUpper`) with a
candidate inside the interval is not recognized as road-range metadata and
can be marked typed with no per-source warning. The tool still *always*
returns `BLOCKED`, has no claim/reset path, and did not open a protected
outcome; nonetheless its `collisions` list is not a trustworthy complete
inventory. The large safety extension is left **uncommitted** and must not
be used to clear a fresh road. Chasing only the next alias is not a proof of
coverage; future work should restrict to independently reviewed format/
field schemas and conservatively hold unknown candidate-possible sources.
The older committed auditor also remains unconditionally BLOCKED. No
confirmation/blind/official action, candidate geometry list, TRAIN claim or
new-road reset has occurred. This safety HOLD does not affect the separately
source-pinned TD 70k/100k learner on four already-consumed TRAIN roads.
