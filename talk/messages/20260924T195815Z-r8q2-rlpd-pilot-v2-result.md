# Pixel RLPD Pilot v2 Result

- Message ID: `20260924T195815Z-r8q2-rlpd-pilot-v2-result`
- Type: result
- Author/session: `r8q2`
- Written: 2026-09-24T19:58:15Z
- Reply to: `20260924T195334Z-r8q2-rlpd-result-followup-coordination`
- Evidence: measured frozen screen/run artifacts; mechanism is not identified
- Status: v2 closed as stop/hold failure

V2 used teacher actor SHA `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954`
and eligible prior-data SHA `a4e5dccc96bb14f06658ae39a3f69df3a9b15168619766bc99f8f529f3dacd3f`
(8,192 decisions spent, 8,170 transitions stored, 19 episodes, 3 distinct-geometry
finishes). All four matched student runs completed 16,384 decisions/15,384 updates.

The custom CPU21 evaluation executed all 192 planned screen episodes over the 12
track/seed cells and two repeats for each of eight actors. All eight actors were
eligible, repeat-deterministic, CPU reload matched, had zero operational failures;
max RSS was 361,873,408 bytes and max action time 2.341 ms. There was one canonical
finish in 96 canonical episodes: RLPD seed 1 at 8,192 decisions finished 1/12
(mean progress 0.4801; lap 40,680 ms), RLPD seed 0's selected 8,192 actor finished
0/12 (progress 0.3313). The matched SAC screen actors selected at 16,384 finished
0/12 for both seeds (progress 0.3540 and 0.2800). Seed 0 fails the preregistered
RLPD minimum of one finish; seed 1 passes and is not lower than SAC. Thus the
two-seed pilot gate is **failed** and no RLPD candidate is promoted. These are
internal CarRacing proxies on a small screen, not official HAIC scores.

Evidence: protocol `experiments/pixel-rlpd-offpolicy-pilot-v2.json` SHA
`ba2375426de3d1d690ea598dd14835523290791b0e91d0c889fae4799460bed0`; result
`experiments/pixel-rlpd-offpolicy-pilot-v2-result.json`; screen manifest SHA
`6c48578fa46a57c747f45f6f73166c411560ab953231c4b1006a193d2a11b2b4`; evaluation
directory `evaluations/20260924T184536689584Z_pixel-rlpd-offpolicy-pilot-v2-screen`.
The first evaluator attempt failed before discovery (zero cells), and the retry
used the same still-fresh v2 screen. Confirmation, blind, and v2's conditional
full-stage cells remain unopened. A runner-only candidate-step field assumption
was caught while deriving the already complete screen result; it is being fixed
for future runs without changing the v2 evaluation receipt.

Per the failed gate, v2 does not proceed to its full/confirmation/blind stage. The
user requested continued feedback and retraining; a distinct follow-up will have a
new hypothesis, fresh teacher data, new code/protocol hashes and fresh screen/held
out geometry. The mechanism (short pilot horizon/data, seed variance, or other
factors) remains unknown; the expansion is not a causal diagnosis.
