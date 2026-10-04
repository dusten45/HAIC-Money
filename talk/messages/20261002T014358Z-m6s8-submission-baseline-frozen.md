# Submission baseline preserved and restoration checked
- Message ID: 20261002T014358Z-m6s8-submission-baseline-frozen
- Type: result
- Author/session: m6s8
- Written: 2026-10-02T01:43:58Z
- Reply to: 20261002T013812Z-m6s8-submission-baseline-freeze-scope
- Evidence: frozen manifest, original package receipt, exact restoration output hashes
- Status: completed; no further action

User-designated submission baseline crossing_projection + collision-shield v1 is
preserved at submissions/20261002-crossing-projection-collision-shield-v1-baseline/.
submission.zip remains byte-identical SHA
c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801.
All11 exact source members, parameter expressions, dependency versions, original
packaging/research provenance, SHA256SUMS and a standalone stdlib restore.py are
included. Original ZIP/receipt, standalone crossing, root Agent and every previous
submission ZIP retain their recorded hashes. Research result25819a46... remains
NOT_ADOPTED under its original+20ms efficiency gate.

Only short file checks were performed: member/ZIP hashes, an isolated exact-copy
restore and a source-only rebuild both reproduce the pinned ZIP byte-for-byte.
Helper runs with python -I outside the repository, refuses existing outputs and
checks the fixed ZIP hash. Rebuild fails closed on Python/zlib codec drift; direct
copy is the preferred restoration. No Agent import, environment construction,
tuning, performance experiment, large audit, upload or confirmation.

Track4 finish18.4s and6th overall are recorded as the user's official-result report
at2026-10-02T01:36:47Z, with user-bound package identity, not an independently fetched
server receipt or site confirmation. Current-state, candidate/submission ledgers
and decision index record this submission-baseline designation separately from
the unchanged research verdict. All old candidates remain available.
