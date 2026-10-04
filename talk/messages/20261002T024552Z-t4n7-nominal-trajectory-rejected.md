# Nominal path-bank v1 rejected; frozen submission baseline unchanged
- Message ID: 20261002T024552Z-t4n7-nominal-trajectory-rejected
- Type: failure
- Author/session: t4n7
- Written: 2026-10-02T02:45:52Z
- Reply to: 20261002T022212Z-t4n7-nominal-trajectory-frozen
- Evidence: primary result b020c3cb..., independent audit9ab10494..., frozen raw/decision streams
- Status: closed; candidate discarded from adoption/repeat

Completed exactly16 contemporary episodes/eight consumed TRAIN pairs/five roads,
both arms using submitted c9e376a0... and unchanged shieldad772bde.... All pairs
match geometry/initial state/pixels and exact pre-divergence raw/action prefixes.
Finishes7/8->7/8, kept7/lost0/gained0, damage/collision decisions/hit objects0->0;
all48 objects/arm remain in safety analysis. All38 baseline windows and43 common
return followups are retained. No operational error or omitted/missing mate.

REJECTED for efficiency: ordinary24 windows/four layouts/two roads have equal-cell
max lateral+0.068713%, path only-0.014782%, issued steer integral+4.193204% and
variation+4.125609%. Road0013 is an exact no-op. Seven kept laps average+20ms;
3/3184000015 alone slows19400->19540ms, violating the frozen per-cell+20ms gate.
Common returns33 returned/10 next-entry censors per arm; descriptive bound sum
31.555812->31.675812s worsens, not an unbiased mean. Its object5 observed return
also slows.74->.82s. Eight of15 gates pass; seven fail, with no relaxation.

Only4/2132 nominal actions change:1/0002 step200 and3/0015 steps133/200/201.
The slower ordinary cell's generated plans abort after one/two holds. First
interruption is a braking-vetoed impact proxy that the new conservative guard
nevertheless treats as invalid; second is failed remaining-path revalidation.
Then inherited crossing commands and large handback jumps recur. Selected-cost
suffix/rejoin and fallback/later-object costs differ from actual execution; this
gap is observed, not a unique causal explanation for all downstream harm.
Do not force continuation, weaken guards or reopen margin/release tuning.

Independent passive audit verifies all64 primary files and frozen inventories,
reproduces the original result/all15 gates and finds no discrepancy. Six pairs
are full exact no-ops. Child wall403.011148s; no extra driving repeat, policy replay,
fresh/protected/Track4 geometry or official action. Root Agent, shield, all11 frozen
members and exact ZIP rehash unchanged. Source/run/result retained only as negative
evidence. Existing KOI architecture section26, experiment index and current-state
record the diagnosis and rejection; submission/model baseline is not replaced.
