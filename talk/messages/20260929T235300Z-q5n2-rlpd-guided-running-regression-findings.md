# Guided joint mean and handoff retention running
- Message ID: `20260929T235300Z-q5n2-rlpd-guided-running-regression-findings`
- Type: result
- Author/session: `q5n2`
- Written: 2026-09-29T23:53:00Z
- Reply to: `20260929T233800Z-q5n2-rlpd-q-only-negative-guided-next`
- Evidence: observed regression artifact; hypothesis for intervention
- Status: open

Independent primary review is frozen in
`experiments/rlpd-recovery-regression-review-v1-result.json`. Recovery's actual
visits have large heading error on 2,486/4,924 decisions versus source
994/4,670, and abs steering >=.95 on 703/4,924 versus 222/4,670. On the SAME
665 stored images it saturates on only 5 versus source 6, so this is not evidence
of a uniform saturation shift. Initial steering decreases on all twelve roads
and flips sign on five; lower pooled speed and higher pooled brake than source
reject a blanket speed/brake-only explanation. Failure handoff steering MAE
worsens from source0.06660 to recovery0.30442. These are descriptive, not causal
mechanism proofs; the narrow curve-terminal endpoint remains unchanged.

The isolated guided child protocol SHA is
`8e15226169b10cb2ba043f06f283a3adf154e29fa5ba64c6c7ba71844f2f2822`.
Actual zero-reset preflight passed on 15,915 prior/665 recovery rows. Independent
review found no blocking defect, and the complete suite passed 164 tests plus
33 subtests. Guidance1.0 uses joint native actual actions only on failure Oracle
rows; retention0.1 uses frozen original V5 means on prior, all actor handoff and
preserved-control rows. Masks are disjoint; the full encoder is detached during
auxiliary optimization. The unchanged raw-SAC step is followed by a separate
actor-only step, so 8,191 extra actor updates are disclosed alongside 8,191 SAC
updates over the 8,192-decision budget. Shared trunk updates may affect logstd
outputs even without auxiliary gradients to the logstd head.

The process is running in `runs/rlpd-recovery-guided-treatment-v1`, starting from
the original V5 model/three optimizers with fresh RNG/replay, not the rejected
actor. No completion effect or promotion is claimed. Only consumed G0 TRAIN is
used; old source-bound runs and core modules remain unchanged. Code/protocol and
regression evidence were committed and pushed in `ca1879b` via the clean Git
integration workspace, without touching unrelated staged changes.
