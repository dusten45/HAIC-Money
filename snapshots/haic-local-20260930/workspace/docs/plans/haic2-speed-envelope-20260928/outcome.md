# HAIC local repeat: speed and phase batch — 2026-09-28

## Fact

- The first consumed TRAIN probe finished 2/2 for both apex control and bend-conditioned gas floor. Gas was changed in 140 and 135 decisions; the candidate was 1.46 and 1.08 seconds faster on those two cells. This was activation evidence only.
- A registered batch of four independent directions then changed actions on both consumed TRAIN cells and all five arms finished 2/2. The four mechanisms were speed envelope, bend exit burst, approach-rate brake, and mild-bend outer entry.
- On fresh TUNE tracks 1–3 × seeds 308–311, the selected full-road-guard control finished **10/12** with median finished lap **25.93 s**. Apex finished 9/12 (24.64 s); speed envelope 7/12 (22.90 s); exit burst 6/12 (23.95 s); approach brake 9/12 (25.76 s); outer entry 9/12 (23.96 s). All six arms had zero invalid actions. The candidates changed actions in 1699, 983, 41, and 142 decisions respectively.
- The v2 result is `REJECT → GATE_REVIEW_REJECT → STOPPED`; the three gates are rule compliance `UNKNOWN`, mechanism activation `PASS`, competitive outcome `FAIL`. No release, package, held-out, confirmation, or official site action followed.

## Inference

The faster pedal mechanisms reduced completed lap time but caused too many nonfinishes. The new direction should preserve the stable full-road-guard as the default controller and use an independently justified risk estimate before granting faster pedal commands. The apex-family held-out failure on 3/303 and this tune batch show that apex steering alone is not a general safety fix.

## Unknown

The best safe throttle schedule and private-track generalization remain unknown. TRAIN and TUNE records do not establish official score, official platform compatibility, or a submission-ready candidate.

## Next research decision

Use a new frozen cycle with the stable full-road-guard as the only default controller. Derive a pixel-only risk gate from observed road-edge clearance and recent bend growth, first screen action changes on consumed TRAIN cells, then compare a new four-direction batch on fresh TUNE cells. Do not sweep the 0.28 gas floor from the rejected candidate. Keep 308–311 consumed as TUNE and 300–303 consumed as the earlier apex held-out.

## Evidence

- `runs/haic-research-v2/haic2-speed-envelope-train-activation-20260928/`
- `runs/haic-research-v2/haic2-four-direction-train-screen-20260928/`
- `runs/haic-research-v2/haic2-four-direction-fresh-tune-20260928/`
- Matching `artifacts/haic-research-v2/<run-id>/report.json` files, with full action traces.
- Persistent frozen executable source: `tmp/haic2-speed-envelope-frozen-20260928/`.
