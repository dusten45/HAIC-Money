# Initial launch surge: measured behavior and seed provenance correction

## Frozen candidate

The pixel-only one-shot launch ZIP is `artifacts/haic-research-v2/launch-surge-one-shot-package-20260929/submission.zip`, SHA-256 `4043993d9901d58b126980b7bafe4f0154305c6a0bf01e1e7e5bc02aacec2389`. It raises gas to 0.45 only while seven road rows are visible and the opening line is clear, straight, centered, and below image-speed 34. It permanently returns to the selected full-road-guard controller on the first exit signal. No site action occurred. Package run `launch-surge-one-shot-package-20260929` finished successfully, but its three gates were `UNKNOWN` until driving.

## Local observations

| Cells | Candidate finish | Selected finish | Candidate/control median finished lap | First 10% median arrival | Selection |
|---|---:|---:|---:|---:|---|
| Tracks 1–3 × 2900–2903 | 12/12 | 12/12 | 23.85/25.02 s | 2.72/3.68 s | Diagnostic improvement only; seed overlap below |
| Tracks 1–3 × 2904–2907 | 10/12 | 11/12 | 26.19/26.36 s | 2.80/3.76 s | REJECT on completion first |

The launch gate fired 120 times in each twelve-cell block and exited at the image-speed limit in every episode. It nearly doubled mean first-10% gas (`0.085→0.179` in the first block; `0.085→0.182` in the second), but finished-lap time did not halve. The second block's candidate-only nonfinish was track 1 seed 2905: it reached 99.25% unique-road progress, then spent more than 100 decisions without visiting a new tile and never crossed the finish line. Candidate and control both failed track 2 seed 2906 at 13.54% progress. There were no invalid actions; the second block reported zero collisions and damage in either arm. The candidate's faster completed-lap median cannot offset the extra nonfinish.

The track 1/2905 evaluator trace shows a different route near 93% progress: at decision 260 the launch candidate was around `(223.9, −132.6)` with heading `3.64`, whereas the control around decision 265 was `(203.2, −116.6)` with heading `4.33`. The candidate later reached 99.25% progress but drove through the final area without a qualified crossing. This is evidence of route divergence, not proof that the initial gas alone caused that particular late failure. The trace contains simulator position only for offline diagnosis; the ZIP reads pixels and inherited actions.

## Provenance correction

The 2900–2903 identities were **already registered and executed** in `tmp/finish-bend-research-20260929/runs/haic-research-v2/final-bend-three-survivors-tune-20260929/` before this launch plan was registered. That other run started `2026-09-28T17:07:46Z` and finished `17:18:22Z`; this launch comparison was registered at `17:26:05Z` and began `17:28:54Z`. The earlier plan reserved 2904–2907 for held-out and 2908–2911 for exact-ZIP confirmation. It rejected all its candidates without opening those held-out cells, but the launch work did not explicitly release or reconcile that reservation before opening 2904–2907. The first launch block is therefore reused tune evidence, and the second cannot be presented as an uncontested independent held-out block. It is now consumed diagnostic evidence. Neither block supports SOTA promotion or a new release.

The original immutable manifests, episode reports, and decisions remain in place at `runs/haic-research-v2/launch-surge-matched-tune-2900-2903-20260929/`, `runs/haic-research-v2/launch-surge-matched-heldout-2904-2907-20260929/`, and their corresponding artifact directories. Their `ADVANCE`/`REJECT` records are historical outputs; this dated correction governs interpretation. The two decision events are `dcd5d66a-e23f-4b14-ad73-07907a462182` and `14acfe2a-3857-411c-9ac6-10fefa4a32e3` respectively.

## Next mechanism

Keep the strong launch and use pixel road preview to regulate steering and speed before sharp bends, including the final bend. Freeze that composition, check activation on consumed diagnostics, then compare on wholly new registered identities across all visible V2 worktrees. Do not adjust thresholds using 2904–2907 or open 2908–2911 as if untouched confirmation for this rejected ZIP.
