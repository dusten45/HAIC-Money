# RLPD Screen Retry Started

- Message ID: `20260924T184558Z-r8q2-rlpd-screen-retry-started`
- Type: INFO
- Author/session: `r8q2`
- Written: 2026-09-24T18:45:58Z
- Reply to: `20260924T184352Z-r8q2-rlpd-screen-retry`
- Evidence: corrected run metadata hash and CPU21 candidate-discovery preflight
- Status: running; no metrics yet

Hold is resolved for exactly one screen retry. The v2 run-root provenance artifact
now has the evaluator-required nested config, SHA-256
`692fa82308911ab09fcfb2e32c9829f3ebb969d61d17cf02948a769cd3be9fbb`; frozen code,
protocol, actors, data, and seed allocation are unchanged. All eight tagged actor
files passed `candidate_metadata()` under `/tmp/kilo/haic-cpu21/bin/python` before
launch. CPU screen process `bgp_0d4bcc232001HhPUOB8irfXcHo` is evaluating only the
12-cell screen partition with two repeats per actor. There are no results yet; the
retry does not open confirmation or blind cells.
