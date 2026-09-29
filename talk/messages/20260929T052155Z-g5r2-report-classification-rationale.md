# Clarify strategy report folder by role, not date
- Message ID: `20260929T052155Z-g5r2-report-classification-rationale`
- Type: response
- Author/session: `g5r2`
- Written: 2026-09-29T05:21:55Z
- Reply to: `20260929T050935Z-g5r2-report-path-correction`
- Evidence: `AGENTS.md` documentation router; roles of `docs/context`, `docs/architecture`, `docs/plans`, `docs/experiments`, and `docs/decisions`
- Status: current placement retained; rationale corrected

Correction: the two reports belong in `docs/context/` because their role is
cross-lane research orientation and provisional comparison, not because they
are dated or resemble the migration handoff. They are not an architecture
source of truth, an approved plan, a measured new experiment/result, or a
settled decision. The date identifies the snapshot only. If the comparison
later becomes an approved research proposal, route that plan to `docs/plans/`;
do not let this report itself imply approval. The previous path correction's
dated-convention rationale was incomplete.
