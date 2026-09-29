# Audit local resource floors versus official runtime rules before changing gates
- Message ID: `20260929T021001Z-k3p7-resource-policy-audit-scope`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-29T02:10:01Z
- Reply to: none
- Evidence: current user explicitly challenges unexplained RAM/disk/GPU/CPU training headroom stops; completed RAW TD and active DAMAGE-only source-bound TRAIN runs
- Status: read-only origin/official-rule audit; no resource code edit or run interruption

The user asks where local training resource minima came from, why they were
added, whether any are OFFICIAL HAIC limits, and prefers loosening gates that
halt feasible concurrent experiments. Independently verify the current
official website and Participants README versus local frozen protocols,
working source, primary resource-failure receipts, documentation and Git
authors/talk provenance. Distinguish official submitted-Agent CPU/process
limits from optional LOCAL GPU training operator floors and observed OOM/
disk failure risks. Findings are not a license to claim there is no safety
margin or to relax another lane's source based on silence.

Two focused read-only code/history agents and a read-only official-source
research run are investigating distinct questions. After evidence, I intend
only a minimal future-facing resource policy/document/test change if safe,
with any shared workflow/current-state/active-plan edits coordinated by
scope. Exact affected source files are deliberately undecided until audit;
current SHA-pinned `scripts/train_tdmpc2_damage.py`, its running
`experiments/tdmpc2-damage-shaping-v1.json`, RAW benchmark/evaluation
artifacts, official environment and peer-owned staged RLPD/DrQ code stay
BYTE-IDENTICAL. The active four-road TRAIN learner continues independently
to the next checkpoint; no new cell, protected/official action or model
confirmation is opened by this resource-policy request.
