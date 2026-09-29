# Anticipatory bend steering: fixed ZIP confirmation

The proposed mechanism reads the pixel road ahead and makes a small early steering correction while the base policy is nearly neutral. It does not alter throttle or brake. The source is `haic_agent/anticipatory_bend_runtime.py` at SHA-256 `27459733EDF6D845B61296F68FA82EB17AF9672E2A6C88337FF8AB6D9A2F91D2`. The packaged candidate ZIP is `artifacts/haic-research-v2/anticipatory-bend-candidate-20260928/submission-anticipatory-bend.zip` at SHA-256 `CE28AB744956BC4898D102C41FF6184EE4AC5A819A8559A656A6AEDDA4008F66`.

The 300–303 run was recorded as fresh tune, but the project-wide [identity audit](../plans/2026-09-28-seed-300-303-identity-audit.md) established those cells had already been consumed by an apex held-out run. Treat the 300–303 result as a consumed-cell diagnostic only. The anticipatory bend held-out on 304–307 used unique cells in the current v2 manifests: candidate and control each finished 11/12, with the same nonfinish, and candidate median completed lap 25.38 versus 25.92 s. This selected a fixed source for package confirmation; it did not establish official generalization.

The [approved package confirmation plan](../plans/2026-09-28-anticipatory-bend-package-confirmation.md) registered tracks 1–3 × seeds 308–311 as reserved confirmation, 12 cells per arm, with both ZIP hashes fixed. A later project-wide audit found `haic2-four-direction-fresh-tune-20260928` had already started those same cells at 13:34 UTC, before this run began at 13:44 UTC. Therefore this package run is a **consumed-cell diagnostic**, not an independent confirmation. The Linux Python 3.11 CPU runner extracted each archive into a clean directory and imported its packaged `agent.py` and `haic_agent` before using the local evaluator. The 24 episode records and exact source/package identities are in `runs/haic-research-v2/anticipatory-bend-package-confirmation-20260928/`.

| Confirmed metric | Control ZIP | Anticipatory bend ZIP |
| --- | ---: | ---: |
| Completed | 10/12 | 11/12 |
| Candidate-only nonfinishes | — | 0 |
| Median completed lap | 25.93 s | 26.28 s |
| Collisions | 6 | 0 |
| Damage | 1.2 | 0 |
| Invalid actions | 0 | 0 |
| Preview steering activations | 0 | 109 |

The run's original report recorded `ADVANCE` because the candidate had the better completion count, but the split collision invalidates independent advancement. One extra candidate completion came from track 2 seed 311. Track 1 seed 308 was unfinished for both arms. The effect on lap time varies by cell, including slower candidate laps on tracks 2 and 3. This is a local diagnostic, not an official site result or proof that the original public video turn is fixed.

The ZIP passed local static member checks and extracted-package import. Candidate max import plus construction was 1.29 s, measured peak RSS 301 MB, max action time 36.66 ms, and no invalid action occurred. This checks the current Participants README contract locally; an official upload and private-track outcome remain unverified. Do not relabel any of 300–311 as fresh. Keep the fixed ZIP and all run records unchanged. For another refinement of this same preview mechanism, use newly registered tune cells, then separate held-out and confirmation identities.
