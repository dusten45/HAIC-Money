# Latest KOI baseline identified and analysis documented
- Message ID: `20260930T075853Z-b8k4-koi-baseline-analysis-result`
- Type: result
- Author/session: `b8k4`
- Written: 2026-09-30T07:58:53Z
- Reply to: `20260930T073800Z-b8k4-baseline-identity-correction`
- Evidence: frozen source inspection, included ZIP/report hashes, synthetic-only tests and independent read-only audits
- Status: analysis complete; no improvement execution

Durable analysis: `docs/architecture/koi-baseline-analysis-2026-09-30.md`.
Under the user's future-baseline instruction, the latest archived local candidate
is `ContactContinuityAgent('crossing_projection')`, not its arrival predecessor.
Final ZIP identifier is `4447c8dfad1392ddbbb83f0b3d23b8d5237af2b9be785cdfd08edaf6c4623f64`;
available contact source module hash is
`bd0518542d0c9165c2d5c87be81d4c77bb82183c4d0334269389154143d34e80`.
These identifiers are different artifact types. Original crossing ZIP and primary
report are recorded in missing external `haic-artifacts-01.zip`; its performance
is therefore labeled archival summary evidence, not locally reverified episodes.

Available arrival ZIP `b168a17d...` and all nine extracted sources match the
manifest exactly. Its included primary report was recomputed independently.
Eleven existing base/perception synthetic tests passed; seven mocked crossing
branches preserve parent pedals. Full-source synthetic reset smoke passed.
No environment was constructed/reset and no new performance was measured.

Document covers the selected inheritance/action priority, pixel/HUD assumptions,
crossing extrapolation and missing-row behavior, diagnostic caveats, precise
cohort/geometry counts, completion/timing/parity semantics, missing evidence and
baseline preservation contract. It distinguishes completion-first adoption from
a universal speed claim, and consumed local evidence from official/private results.

Pull fetched but refused merge over existing edits. User-approved immutable
`c4e224d` extraction was used; existing staged/unstaged/untracked work is untouched.
Only a new analysis document and separate talk notes were written. No runtime,
shared status/plan, model, training, tuning, official action, commit or push changed.
