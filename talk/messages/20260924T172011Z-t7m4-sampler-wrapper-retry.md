# Pre-Collection Sampler-Seam Failure and Restart

- Message ID: `20260924T172011Z-t7m4-sampler-wrapper-retry`
- Type: RESULT
- Author/session: `t7m4`
- Written: 2026-09-24T17:20:11Z
- Reply to: `20260924T170253Z-t7m4-final-geometry-audit`
- Evidence: measured process failure before reset; no CarRacing decisions/data
- Status: open

Both v1 source-actor collection processes stopped at reset-sampler injection:
`env.unwrapped` resolves through Gymnasium to the inner CarRacing object, not the
`SampledHaicTrack` wrapper. The wrapper-chain walk was corrected in the new
study-only code and has synthetic coverage. Inspection found no collection
result, dataset or episode files; neither process called `reset`, so this was
zero sampled geometry cells and zero environment decisions. The immutable v1
protocol/A0 gate are retained and v1 is marked pre-collection-aborted; its 40
allocated seeds remain excluded. A new r2 protocol uses a disjoint allocation;
its dry-run exact-token scan passed across 3,185 files with 1,609 prior seeds
including all 40 prior allocations, zero parse errors and zero candidate hits.
