# Interval reference gains support; empirical cost uncertainty prevents confident anchors
- Message ID: 20261005T112308Z-r6t9-interval-paired-result
- Type: result
- Author/session: r6t9
- Written: 2026-10-05T11:23:08Z
- Reply to: 20261005T110518Z-r6t9-scope-time-correction
- Evidence: runs/joint-temporal-interval-v1/paired-analysis.json and calibration.json
- Status: paired replay complete, before pilot resets

Primary paired analysis SHA19e2fc4ee205b491fc228adf0644ab6093c50a3d96be119c41dcf0c296254f43;
calibration SHA5c8fe665e21688a88e286ded8771577d51b1c1849db2a744c91f1905cf3bf1b0.
Same eight consumed starts: old/new paired cost support2/8 ->6/8, with4 new,
2 retained,0 lost,2 unsupported. Supported actual suffix orders are3 baseline,
3 alternative. Reference-only predicted orders are3 baseline/2 alternative/
3 ambiguous; after fixed CAL paired-difference allowance .44961874671412616,
all8 are ambiguous. No weight, physics, hold or horizon tuning follows this result.

Natural cost support changes straight2/25->9/25, left0/2->1/2, right2/5->3/5,
braking2/13->9/13, with original denominators. Calibrated eligible gains remain0.
Left2 retains incomplete reference support despite17/17 footprint-known ticks;
brake2 retains invalid observer/history. Neither is shortened to obtain cheap cost.

Three CAL roads/five starts fit a central total-error envelope, including dynamics:
H4 position1.095715344103732 world/yaw.09249519316075627rad, plus the original
19 shared initial-state hypotheses. On two roads/three starts excluded from this
fit, whole-path same-scenario misses0/6 but central-path misses2/6. Separately,
held-out mapping misses are0/38 measured and0/15 valid inferred intervals after
CAL-only widening. Small consumed TRAIN samples: empirical ranges, NOT confidence
levels, safety guarantees or untouched validation. No additional paired resets.

The user-authorized six full episodes still test actual supported opportunities
with strict baseline fallback; no all-regime prerequisite or forced intervention.
If zero interventions, record candidate effect NOT EVALUATED. Integration reviews
fixed support diagnostics and a JSON tuple/list initial-parity error before data.
Champion ZIP/all11 sources unchanged. Comparator/prerequisites committed and pushed
as907fcff on the existing upstream; unrelated changes remain untouched.
