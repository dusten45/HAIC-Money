# Rear clear V1: prospective development candidate gate

## Frozen candidate and profile

Source was committed and pushed as
`67a0fc19d066195355b04d66de8d15a1943f1a89` before this fresh run. Source SHA256:
`093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc`.
Parameters are `{}`. The unchanged evaluator SHA256 is
`39a280612b3f5e7402b7a475c447a7cb9c465ddc0bc9e5c5f98300cdcad0325c`.

This run uses **candidate** role, two workers, 700 steps per cell, and the
exact 20-cell development suite. The seven prior complete candidate receipts,
in chronological order, are `r1-safety.json`, `r2-pace.json`,
`r3-recovery.json`, `r4-preview.json`, `r5-envelope.json`, `r6-corridor.json`
and `r7-guarded-preview.json` under `results/cloud-20261004/development/`.
Their distinct source/parameter identities, cell coverage, predecessor links
and hashes were checked. Their original prospective profiles were
13/13/13/15/15/15/18 seconds, and all seven failed. Consequently this new
candidate prospectively uses **18 seconds**, with rejection streak 7 before
the run. The historical receipts supply the profile basis; they are not
fresh validation for this candidate. Earlier diagnostic benchmarks are excluded.

The process completed with exit status 0. No source/parameter changes or new
holdout episodes occurred. The extra geometry seeds are consumed development
data. The output freeze was written before any episode.

## New measurements and verdicts

| Track | Mandatory seed | Actual lap, s | Contacts | Within 18 s | Extra finishes | Extra within 18 s |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | 516237 | 15.54 | 0 | Yes | 3/4 | 1/4 |
| 2 | 644062 | 18.72 | 0 | No | 4/4 | 2/4 |
| 3 | 1007 | 16.56 | 0 | Yes | 3/4 | 0/4 |
| 4 | 18800 | 18.14 | 0 | No | 3/4 | 3/4 |

**The 18-second profile fails.** Only 2/4 mandatory cells and 6/16 extras
(37.5%) finish within that limit; the profile requires all four mandatory
cells, at least 90% fast extras, and at least 75% per track. Overall additional
completion is **13/16**, with 14 contacts and completed times **15.96–24.70 s**.

**The original 13-second and early-10-second goals remain unmet.** No mandatory
or extra cell finishes within 13 seconds. No candidate adoption or unseen
reliability claim follows from this run. This complete, distinct, error-free
candidate rejection advances the ordered streak from 7 to 8; the next profile
remains 18 seconds. Neither earlier criteria nor results were rewritten.

| Extra seed | Track 1 | Track 2 | Track 3 | Track 4 |
| --- | --- | --- | --- | --- |
| 2867319041 | 16.44 s, 0 contacts | 16.20 s, 0 contacts | 20.66 s, 0 contacts | 16.34 s, 0 contacts |
| 359018627 | DNF, progress .2578, 3 contacts | 19.92 s, 2 contacts | 24.70 s, 0 contacts | 15.96 s, 0 contacts |
| 1764402399 | 19.54 s, 0 contacts | 16.14 s, 0 contacts | 21.22 s, 0 contacts | 17.56 s, 3 contacts |
| 4029571806 | 19.46 s, 0 contacts | 20.50 s, 2 contacts | DNF, progress .6199, 1 contact | DNF, progress .5793, 3 contacts |

All three DNFs have retirement label `off_track`, with no worker errors. That
label is set by more than 100 consecutive decisions with negative total reward
in the unchanged wrapper, rather than by a direct wheel-road departure test.
The track-4 DNF has zero sampled full-off-road decisions and two partial-road
decisions. Its contacts occurred at actions 103–105; final speed is 59.86 m/s
at 16.72 s. These observations do not establish a stationary stall or explain
its control failure. Further diagnosis must distinguish poor progress and
contact history from actual road departure.

## Verification and limits

The four mandatory cold repeats match the earlier `rear-clear-v1.json` rows
on all 15 nonruntime fields, including the action hashes. Their full raw trace
JSON also matches exactly. These are new executions; the earlier four-lap
benchmark is used solely as their repeat reference.

Every one of the 20 raw action hashes was recomputed from row-major float32
bytes. All actions have the correct three-value shape, finite values and
official bounds. All trace state/debug values are finite. Steps, contacts,
final progress and per-cell receipts match. There are exactly 20 unique
declared cells, no operational errors, and unchanged source/parameters.
All 11 environment/evaluator hashes match the freeze and baseline. The root
agent and seven official files still match starting commit `14bb967` byte
for byte.

Local worker maxima: initialization **0.685 ms**, reset **0.018 ms**, action
**47.733 ms**, peak RSS **95.039 MiB**. All are finite and within the respective
10 s / 5 s / 5 s / 1024 MiB limits. NumPy is already imported by the evaluator
before its initialization timer. These measurements do not certify the
official submission container. No new ZIP driving or holdout is claimed here.

The complete source, input, predecessor, cell and trace bindings and checked
actuals are saved in `rear-clear-v1-development-candidate-lineage.json`:

- Development receipt SHA256: `44a43f3e37b5a93a6a8f2e10c6c6ef47b41e3a8e1f82d08f75de56c684e94d12`.
- Freeze SHA256: `782c583afe158e8b06f2ad8210709fa0d5c59d249569b2deda89150e4fc4e36e`.
- Lineage SHA256: `b81790c1c8c7b413115d3a8e658cc7365576caaac580ba39aec7172eebed07d2`.

## Reproduction

```sh
env MPLCONFIGDIR=/tmp/haic-mpl XDG_CACHE_HOME=/tmp/haic-cache \
 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m agents.apex_2026.evaluate \
 --source agents/apex_2026/fast_rear_clear_agent.py \
 --suite development --max-steps 700 --workers 2 --role candidate \
 --prior-receipts \
 agents/apex_2026/results/cloud-20261004/development/r1-safety.json \
 agents/apex_2026/results/cloud-20261004/development/r2-pace.json \
 agents/apex_2026/results/cloud-20261004/development/r3-recovery.json \
 agents/apex_2026/results/cloud-20261004/development/r4-preview.json \
 agents/apex_2026/results/cloud-20261004/development/r5-envelope.json \
 agents/apex_2026/results/cloud-20261004/development/r6-corridor.json \
 agents/apex_2026/results/cloud-20261004/development/r7-guarded-preview.json \
 --output agents/apex_2026/results/speed-20261005/rear-clear-v1-development-candidate.json \
 --trace-dir .haic-artifacts/apex-speed-20261005/rear-clear-v1-development-candidate-traces
```

The recorded paths now exist. Any authorized repeat must use new unique output
paths; this command records the completed run, rather than permitting receipt
overwrites.
