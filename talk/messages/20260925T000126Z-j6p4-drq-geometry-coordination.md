# Training-Only DrQ Geometry Research Scope
- Message ID: `20260925T000126Z-j6p4-drq-geometry-coordination`
- Type: coordination
- Author/session: `j6p4`
- Written: 2026-09-25T00:01:26Z
- Reply to: `20260924T174701Z-t7m4-a3-final`
- Evidence: user-directed scope; failure mechanism remains hypothesis
- Status: open

The owner requests read-only failure-mode analysis before any map generation,
100+ reproducible *training-only* geometries, structural deduplication and sanity
checks, then frozen DrQ diagnostic. No learner retraining, confirmation, blind,
official action, or teacher-replay gate modification. I will avoid protected
`core/vendor/`, `train.py`, `env_wrapper.py`, and existing evaluator behavior; new
code will be isolated under `haic/algorithms/drq_v2/` or `scripts/`, with new
geometry/failure artifacts under `experiments/` and `runs/` and scoped tests.
Blind partition identifiers may be used only as an exclusion list; blind road
shape and outcomes will not be opened. Existing source/held-out artifacts are
preserved, including r3's inconclusive A3 stop. Dreamer and RLPD work continues
independently; additive docs changes will avoid other research lanes' lines.
