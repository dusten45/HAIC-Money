# Restrictions and verification

Reconstructed 2026-09-30 from the [official pinned README](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/README.md)
and environment sources linked in [COMPETITION_INFO.md](COMPETITION_INFO.md).
The original local document was unavailable. PASS below is limited to the stated
evidence, not a blanket compliance or submission certificate.

| Requirement | Current evidence/status | Required run/package check |
| --- | --- | --- |
| Official physical environment unchanged | PASS: six frozen source files match official commit after newline normalization | Recheck diff and source version before performance runs |
| Observation `(4,84,84)` float32 `[0,1]`; action finite `(3,)` in bounds | Public contract verified; prior candidate tests recorded in RESULTS | Run relevant contract/inference tests for the actual candidate |
| Only public observations supplied to inference | Official runner verified; candidate audit REQUIRED | Inspect inference dependency path; do not provide seed, track ID, coordinates, reward or privileged state to agent |
| 10 consecutive invalid actions retire | Official contract verified | Validate finite actions and exceptions; do not rely on clipping invalid values |
| Import/init10s, reset/act5s, memory1,024MB, CPU | Official limits verified; current Linux runtime UNKNOWN | Measure actual artifact; local Windows evidence does not certify Linux |
| Python3.11/Linux; fixed package compatibility | Official versions in COMPETITION_INFO | Confirm current dependencies and CPU loading |
| No inference-time Internet download | Official rule verified | Bundle required model/data; audit inference |
| Additional requirements pinned to compatible PyPI binary wheels | Official rule verified | No URL/Git/local index/nested requirements/source builds; max50 entries,32KiB,180s installation,512MiB installation limit |
| Archive root agent.py; case-sensitive relative file loading | Official rule verified | Inspect exact ZIP, import/reset/act from extracted files |
| ZIP500MB; unpacked2GB; files1,000; each500MB; compression ratio100 | Official limits verified | Check actual archive metadata and content |
| Prohibited Python imports/functions and native executables | Official rule verified | Static audit all submitted Python and archive entries |
| Website-specific deadline/quota/portal/announcements | UNKNOWN: website inaccessible | Blocks actual submission until official confirmation |

Prohibited imports: `ctypes`, `importlib`, `multiprocessing`, `os`, `pathlib`,
`resource`, `shutil`, `signal`, `socket`, `subprocess`, `sys`.
Prohibited dynamic functions: `compile`, `eval`, `exec`, `__import__`.
Prohibited archive suffixes: `.com`, `.dll`, `.dylib`, `.exe`, `.msi`, `.scr`, `.so`.
These are submission restrictions, not a ban on local diagnostic tooling imports.

Do not modify `core/`, `env_wrapper.py`, `damage.py` or the official license for
performance experiments. Preserve raw rewards, observation shape, frame skip and
termination. Rendered videos lack source-bound action/clearance telemetry and
cannot alone establish causal performance gains.

The public rules define retirement for 101 consecutive negative-reward actions,
playfield exit, full damage and exhausted action budget. They do not explicitly
declare every brief road departure a disqualification. Grass friction0.6 and
finish eligibility remain constraints; intentional excursions have no demonstrated
benefit here. Do not equate simulator tolerance with confirmed competitive safety.

## Project safeguards and scope

Follow [RULES.md](RULES.md): freeze comparison source/model hashes and metrics,
preregister one main factor, preserve failures, and inspect all consumed/reserved
seeds before selecting cells. Reservations apply across ALL track IDs. This file
does not free any seed or supersede existing protocols. Development reuse cannot
be relabeled fresh confirmation; diagnostic results cannot promote a champion.

Before a short local diagnostic: complete required source/observation/inference
checks and protocol registration, verify all five reconstructed documents exist,
and mark each candidate-specific gate PASS/FAIL/UNKNOWN in its receipt. A failure
or unresolved local-relevant uncertainty blocks that run. Submission-only unknowns
may be explicitly NOT APPLICABLE to that local run, while remaining UNKNOWN for
submission. Long runs, blind evaluation, promotion and submission retain their
separate RULES gates; document reconstruction does not authorize them.
