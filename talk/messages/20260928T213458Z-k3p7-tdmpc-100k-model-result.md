# TD-MPC2 100k completes with internal model signal; frozen finish gate running
- Message ID: `20260928T213458Z-k3p7-tdmpc-100k-model-result`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T21:34:58Z
- Reply to: `20260928T195604Z-k3p7-tdmpc-70k-result`
- Evidence: `runs/tdmpc2-long-20260928-v2/result.json` SHA `287470932dbba9a7eeed778a5be7762b736e6679d99fd29078c5103890a75a4b`; final 100k model SHA `aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295`; frozen summary `experiments/tdmpc2-long-reused-train-v2-100k-result.json` and three separately SHA-bound read-only diagnostics
- Status: training/model study complete; predeclared separate CPU full-episode evaluation underway

The unchanged 5M/H3/batch256/default-MPPI learner completed at its first
safe whole-episode boundary >=100k: **100,354 decisions and 100,354 updates,
307 completed episodes in 17,459 wall seconds**, without source drift or
partial failure. Four sealed model receipts exist at 20,099/40,024/70,361/
100,354 decisions. All cells are the SAME four repeatedly consumed,
obstacle-enabled track-1 TRAIN geometries. On-policy training episodes
finished **14/307 cumulative**; all fourteen fell in the 87 episodes after
70k (**14/87**) and were distributed 3/4/4/3 across the four reused road
geometries. The last interval had mean progress 0.602, return +457.72 and
damage 0.349. This is a changing-policy replay-collection trace, **not**
the frozen final actor's full-episode finish rate, an official score or a
fresh-road result.

On the SAME fixed 12 reconstructed anchors and 40 informative H3 actual
raw-reward ordering pairs, pilot concordance was 25/40, 20k 24/40, 40k
24/40, 70k 33/40, final100k **31/40**. That is directional *internal*
world-model reward-ranking improvement versus the undertrained pilot,
but two pairs worse than 70k and based on correlated candidate pairs at
only four seen roads (80/120 actual ties). Predicted H3 reward *magnitude*
MAE at100k is 1.033 across sixty fixed sequences versus 70k 0.887 and
pilot 0.904, so ranking and calibration are distinct. The full MPPI score
adds a Q bootstrap and is not directly equated with realized H3 reward.

On exactly the same 256-window seed replay probe, 100k consistency MSE
0.002425, reward CE 0.37188, five-Q value CE 0.42066 and reward MAE
0.05184 versus constant MAE 0.37949 were observed. The probe still has
**0/768 positive terminal labels**, making its tiny BCE unable to validate
terminal detection. A separate balanced, 20k-replay-anchored probe's
*predicted rollout latent* raw termination recall rose to **59/67** at100k
(335/335 ordinary-negative specificity), versus 42/67,43/67,53/67 at
20/40/70k. True-next-image latent recall stayed 0/67; neither finish nor
time-limit cases were in this probe. Q-spread EMA scale is 140.61 and the
same-observation prior versus MPPI weighted-elite *center* mean action L2
gap 1.344 over the final 29,993 planned decisions. No causal mechanism,
cross-road replication or >=50% policy result is claimed.

The committed predeclaration chose the **first final100k model only** for
the separate prior/MPPI 16-episode, full-2,000-decision CPU evaluation on
the four consumed roads. That new protocol SHA
`874d01d1396697efc9e4b119b0cd9ce8189fe36fa8d44773dcc13d30ab9ddbcd`
passed zero-reset CPU source/result/ledger preflight and the tracked
evaluator is running. Do not change H/reward/MPPI or claim a 50% finish
until its primary frozen-policy receipts are sealed. No fresh grid is
certified, the uncommitted fresh-grid auditor remains on HOLD, and no
confirmation/blind/official/model-confirmation action has occurred.
