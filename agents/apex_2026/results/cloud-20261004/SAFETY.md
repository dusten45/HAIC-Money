# Guided hazard routing study

`safety_agent.py` is a standalone copy of the historical hybrid. Planned
hazard steering retains the robust corridor command. Guided hazard pedals
retain robust braking and cap gas by robust gas and the bounded speed target.
Only clear-route transitions apply the configurable steering average.

The v1 regression demonstrated that the historical hybrid could replace a
robust brake request with full gas. Independent review then found v1 could
average steering toward the opposite corridor and bypass low-speed crawl
gas; both behaviors were reproduced with failing tests and fixed in v2.
Eleven safety tests and the unchanged legacy contracts pass; two independent
readers found no blocking defect in v2. This is a control-routing correction,
not a measured completion-rate improvement.

## Fresh development result

The complete 20-cell `development/r1-safety.json` binds v2 source SHA256
`7b783070f26b108957f4d9c4c5658b411a7f03cb26690055cf2bdba13dde4c37`,
parameters `{}`, and evaluator `39a28061…`.

- Required laps: **19.62 / 26.46 / 23.96 / 21.82 seconds**, all finished,
  zero contacts.
- Extra cells: **10/16 finishes**, with per-track counts 1/4, 4/4, 2/4, 3/4.
- No required or extra lap meets 13 seconds. The prospectively selected
  13-second profile failed; this is the first complete new development reject.
- Completion count equals the fresh hybrid benchmark. Safety is not promoted.

Earlier v1 benchmark receipts are under `probes/safety-v1-*.json`. Their source
was `ef64c728…`, not the current v2 source. The saved
`probes/safety-v1-to-v2.patch` can be reversed against this v2 file to recover
the exact v1 bytes for hash checking. Neither mandatory-only benchmarks nor
the two targeted failures contribute to the prospective relaxation count.

Holdout simulations remain unopened. The root agent and official simulator
are unchanged.
