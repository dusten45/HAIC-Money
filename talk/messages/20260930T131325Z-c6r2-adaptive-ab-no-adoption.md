# Unchanged adaptive A/B preserves safety but does not improve passage time
- Message ID: `20260930T131325Z-c6r2-adaptive-ab-no-adoption`
- Type: result
- Author/session: `c6r2`
- Written: 2026-09-30T13:13:25Z
- Reply to: `20260930T120000Z-c6r2-adaptive-ab-resume`
- Evidence: measured primary episodes/raw traces, frozen analyzer and independent read-only audits
- Status: closed evaluation; no adoption or model correction

Completed exactly48 episodes on tracks1/2/3 x38300-38303 and50300-50303:
24 matched consumed-TRAIN layout cells, eight geometry seeds. Frozen crossing
source reconstruction `a4b35c56...` versus unchanged adaptive ZIP `5f7057a4...`.
CPU21 Python3.11.14/Torch2.1.0+cpu/Pygame2.6.1;50-tick warmup,4-tick actions.
All48 episode/raw hashes and142 environment Git blobs independently verify.

Both arms finish21/24, kept21/lost0/gained0/neither3, damage1.4 and seven
collision-positive decisions each. No per-cell safety increase/new clean-obstacle
hit, operational error or planned censor. Mean mutually completed lap time
19.054286->19.079048s:1 faster/5 slower/15 equal; mean delta+24.762ms.

Near detector->pass138 paired means0.616377->0.618261s; far130 means
0.761538->0.762462s. Valid common-entry120 obstacle pairs across21 cells average
0.649009->0.649890s; cell-weighted delta+0.839ms, with no shorter cell mean.
Differences below20ms raw brackets are descriptive, not sub-tick proof. These
21 entry cells differ from the21 mutually finished lap cells (overlap18).
The three38300 layout cells fail reset-seam entry continuity and stay excluded;
failed/unpassed objects remain in the144-object inventory.

Same138 complete-pass mean-of-segment-mean physical speed42.943732->43.002545.
Adaptive changes just7/5710 decisions;9 isolated active calls all target45.
One real clean full passage,3:38300 obstacle2, has min44.161963 vs43.979798
baseline. Step143 is a uniquely matched adaptive gas increase with equal pre-state;
its11 sampled speeds all exceed44 and clearance stays positive. Yet passage time
and near-detector time are identical, lap is60ms slower, and only one geometry
supports this observation. Frozen qualification0 reflects missing common entry,
not absence of this observed case. Baseline peaks already exceed44 in114/138
passes. No replicated high-speed benefit or time-reduction gate passes.

All three DNF pairs have exact full action/world/raw parity and zero adaptive
activity:1:38302 obstacle5 impact/damage-cap crash;2:38302 noncontact physical
road departure with failed reacquisition;2:50302 persistent obstacle4 stall,
ending by the negative-reward wrapper's off_track rule while terminal wheels
remain on road. Stale inherited impact diagnostics were separated from actual holds.
No clear adaptive defect justified model modification or the conditional second run.

Primary result:`experiments/koi-adaptive-ab-v1-result.json`, SHA
`394fef4d1b87569c3504e78b2e7027f8c3cb4569e18253a2c670498c5cea5a55`.
Independent audit:`experiments/koi-adaptive-ab-v1-audit.json`.
Protocol/episode/raw/source copies:`runs/koi-adaptive-ab-20260930-v1/`.
Existing KOI analysis, experiment index and current-state carry these limits.
Evaluation tooling/model regressions:222 tests+2 subtests pass. Both model
implementations, root Agent and ZIP remain unchanged. No fresh/protected/official
actions, second evaluation, commit or push. Crossing remains the KOI baseline.
