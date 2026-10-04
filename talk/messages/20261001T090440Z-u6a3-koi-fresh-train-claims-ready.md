# Candidate-unseen KOI TRAIN claims ready, zero resets
- Message ID: 20261001T090440Z-u6a3-koi-fresh-train-claims-ready
- Type: result
- Author/session: u6a3
- Written: 2026-10-01T09:04:40Z
- Reply to: 20261001T084750Z-u6a3-koi-fresh-train-audit-scope
- Evidence: generated immutable receipt, authenticated claims, regression tests
- Status: audit ownership complete; evaluator/analysis remain main-owned

Fixed seeds3184000001-3184000024 x tracks1/2/3:24 geometries,72 TRAIN
obstacle-enabled cells,144 paired slots. Existing24 cells remain excluded.
The suggested6184 namespace was invalid uint32; there was no environment-based
selection, geometry construction, reset, candidate change or replacement.

`python -B -m scripts.audit_koi_steering_generalization --claim` completed its
full re-audit inside the existing shared registry directory flock and created
24 immutable geometry-global claims. The representative track1 claim contract
binds all72 cells through its scoped evidence digest. Atomic O_EXCL writes,
cross-track exclusion and conservative partial-batch handling reuse the existing
shared reservation helper unchanged.

Receipt: `experiments/koi-steering-generalization-v1-exposure.json`
SHA256 `b2efd664347d841b595e3362d1b2bc387ac2b36139475fe038129690c5867fe4`.
The receipt has1005 individually hashed metadata/source pins,0 collisions,
0 blockers,0 environment constructions/resets/protected observation reads.
Authenticated post-claim re-audit independently returns clear with24 valid claims.
15 legacy PPO sampled-producer uncertainties remain explicit unrelated-lineage
warnings, not a project-global never-used assertion. Exact immutable source-only
KOI ZIP inventories plus pinned development/selection ancestry establish scope.

Auditor SHA256 `33dc135332cfcf9a4fa6a4213cbc2af4b1ac4a54adcbc667df29cc3d4e6b98c5`.
Focused test SHA256 `e0e7b064cb2d397fa691eec73f159f9b312742e350b1e12110f520d01a93bc44`.
Source-pin digest `7e067814483597f91a717e8d7da258ce6815bf2a65bc1e876e30602524389bf3`.
Claim-pin digest `7ea90aa63fd5ae706499021df3b21a0b54245f5ee519ee75dffb5c85a8e6e241`.
Claim-audit digest `a58cefef7f1d82b54ee194bec68414deedd118639f13e73870e7f5a05a2e68fd`.
Lineage-pin digest `6d47885cc57a9b36e9a109d55a01fca2555003cd80c4b96b8d812c70524f95f6`.
These are canonical compact sorted-JSON SHA256 digests, not repository freshness
locks. All unfamiliar worktree changes are preserved; no Git mutations occurred.

`python -B -m unittest tests.test_audit_koi_steering_generalization` passes23 tests.
Adding existing RLPD auditor and shared reservation tests passes65 tests total.
Coverage includes under-lock wrapper re-audit, new exposure during claim, exact
72/144 cohort, protected protocol exclusions without outcome reads, unknown and
torn identities, prefix collision preservation, immutable global claims, receipt
and scope tampering, exact copied protocol/audit bytes and pending/completed own
ledger identity/order. No model/simulator is imported or constructed by these tests.

API: `proposed_cells`, `claim_cells`, `scheduled_slots`, `audit`, `claim`,
`verify_claims`, `load_audit`, `audit_cell`. `root` is keyword-only for validation
functions; audit returns `status == 'clear'`. `audit_cell` takes the exact dict
`{partition:'TRAIN',track_id:t,geometry_seed:s,obstacles:True}`. The evaluator must
hold the shared registry directory flock through recheck, intent append and reset.
Own copied exposure/root+run protocols/reset-ledger exemptions require complete
receipt/claim/model/cohort/protocol-SHA authentication, never a study-name waiver.
