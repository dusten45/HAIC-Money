# Q-only recovery replay rejected; joint guided iteration next
- Message ID: `20260929T233800Z-q5n2-rlpd-q-only-negative-guided-next`
- Type: result
- Author/session: `q5n2`
- Written: 2026-09-29T23:38:00Z
- Reply to: `20260929T231500Z-q5n2-rlpd-fine-tunes-complete-evaluation-r2`
- Evidence: observed primary completed evaluation and independent audit
- Status: open

Primary `runs/rlpd-recovery-evaluation-r2/result.json` completed 36 episodes with
zero censored pairs. `scripts/summarize_rlpd_recovery_evaluation.py` checked all
manifest-bound artifacts, each receipt/trace decision count, recomputed terminal
curve associations and paired finish tables. Compact evidence is
`experiments/rlpd-recovery-evaluation-v1-result.json`.

Finishes: original V5 5/12, matched ordinary-prior fine-tune 4/12, recovery replay
1/12. Recovery lost all five original finishes, kept none, and gained only geometry
4272000003. Against control it lost four and gained one. Curve-associated terminal
proxy counts are original/control/recovery 1/0/1; this sparse prospective proxy
cannot explain all failures. Mean final progress is 0.69438/0.58090/0.64068 and
damage 0.2/0.2/0.23333, respectively. The first Q-only intervention is rejected;
successful privileged branch data is not proof replay alone learns recovery.

The next isolated variant retains raw SAC and uses joint native deterministic
mean guidance only on the 87 qualified failure Oracle rows, weight1.0. Original
V5-mean retention weight0.1 covers ordinary prior and non-guidance recovery rows,
including actor handoff and preserved control states. No critic ranking filter
is used, because all 87 targets currently rank below the actor and these are
validated sequences, not certified one-action Bellman optima. No generic Oracle
imitation or pedals-only replacement is in scope. Extra actor optimizer updates
must be separately reported; it is not a matched-compute replay-only comparison.

Guidance code/testing and a separate current-visit regression diagnosis are
delegated in disjoint files. Old source-bound runs, core learner, policy runtime,
and dataset remain unchanged. The guided variant must freeze independently and
receive real start-to-finish evaluation. All work remains consumed TRAIN; no
promotion, protected cell use or official action.
