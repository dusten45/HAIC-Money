# Rejected stationary recovery experiment

The diagnostic behind `recovery_agent.py` found that four hybrid `off_track`
development retirements were actually stationary collisions followed by the
official negative-reward watchdog, with wheels still on asphalt. A camera
aspect gate removed a nearer elongated curb fragment, and a bounded escape
used a nearby side cross-section after six low-speed frames. It limited each
burst to 24 decisions, gas 0.18, steering 0.40 with 0.07 slew, and a cooldown.

Five failing-first camera tests and the preserved Apex tests pass. The tests
prove bounded/reset behavior and detector selection, **not a safe escape**.
Independent review identified that the side slice can contain interpolated
road edges and does not check the swept approach footprint. The aspect test
rejects wide curb fragments but not all tall ones. These limitations prohibit
claiming a verified corridor or integrating this escape as an improvement.

## Fresh results

`development/r3-recovery.json` binds unchanged source SHA256
`63a4f8537a227f8ee52c898265e0cba4b2b2abdba4d64418603305ad669be1d3`,
parameters `{}`, evaluator `39a28061…`, and the first two complete rejects.

- Required laps: **19.62 / 26.20 / 22.90 / 21.82 seconds**, all zero contacts.
- Extra finishes: **10/16**, per track 1/4, 4/4, 2/4, 3/4; same coverage as
  the fresh hybrid benchmark. None met 13 seconds.
- This is the third complete 13-second development rejection. The next
  prospective profile is 15 seconds; the original 13-second verdict stays false.
- The diagnostic target `(1,1764402399)` still stalled at progress 0.33916.
  Escape steering/gas activated, but could not move the frontal obstruction.

Mandatory/target benchmark receipts are in `probes/recovery-v1-*.json` and
do not count toward relaxation. The source is preserved as a failed experiment;
it is not promoted. Earlier curve-aware avoidance is the next independent
hypothesis. Root/official sources remain unchanged; holdout remains unopened.
