# Apex final research result — 2026-10-05 KST

**Performance improved on required/development cells, but the original10–13s
objective and the prospective holdout corroboration gate were not met. The
frozen candidate remains experimental and is not formally adopted or submitted.**

## Selected implementation

[Standalone Agent](candidate/agent.py) requires no model file or constructor
configuration. It uses observation pixels to optimize a continuous free-space
path, reads speed/yaw from the public HUD, and allocates pedals with a calibrated
rolling-dynamics predictor including coasting. This replaces main's learned
policy strategy; it imports no old policy and receives no simulator telemetry,
track ID, geometry seed or route tape. Main/root agent and official simulator
are unchanged.

Source SHA256: `b27142578d98eb090da8ac858eb784d48053b026142cf894b072da2c6440ac97`.
Candidate freeze commit: `cdf675e1811542f7999ebcc9510dd35e47541217`.
The [freeze](candidate/freeze.json) binds source and empty runtime configuration;
[selected settings](candidate/selection.json) are baked into defaults. Candidate
files have not changed since that freeze. Selection metadata records the state
at freeze; current status is in [benchmark.json](benchmark.json) and this report.

## Fresh final results

Times are actual official simulation finish timestamps minus post-warmup start,
not process runtime, action count or qualification time.

| Required track / seed | Earlier smooth200 | Frozen candidate | Reduction |
|---|---:|---:|---:|
|1 /516237|18.32s|16.78s|1.54s|
|2 /644062|22.80s|20.68s|2.12s|
|3 /1007|20.18s|18.56s|1.62s|
|4 /18800|18.42s|18.10s|0.32s|

All required laps are damage-free. The four-lap total falls79.72→74.12s, about7.0%.
This comparison uses measured same-session sources, not historical scores.
Fresh literal-main and recommended-release baselines were0/4 and3/4 finishes;
see [baseline evidence](results/baseline-20261005.json).

| Fresh frozen partition | Finished | Median of finished laps | Slowest finish |10–13s / all cells|
|---|---:|---:|---:|---:|
|[Required](results/final-required.json)|4/4|18.33s|20.68s|0/4|
|[Declared development](results/final-development.json)|12/12|16.28s|17.44s|0/12|
|[Required repeat](results/final-required-repeat.json)|4/4|18.33s|20.68s|0/4|
|[One-time unseen holdout](results/final-holdout.json)|10/12|17.53s|19.30s|0/12|

The first three rows are20 newly executed episodes, all damage-free. No earlier
P1 receipts were reused in this final validation. Required repeats have identical
full driving traces apart from inference timing; they are reproducibility checks,
not new geometries. Development cells were repeatedly consumed for selection.

After source/configuration freeze was committed and pushed, these20 runs passed.
Only then were12 random geometry seeds allocated, audited for prior exposure,
and committed in `41aca5f` before the first holdout reset. All12 were evaluated
once without tuning afterward. [holdout.json](holdout.json) is the immutable
allocation-time record; current consumed status is in benchmark.json.
The10 finished holdout laps range14.82–19.30s; one has damage0.2. Failures:

|Track / seed|Outcome|Final progress|Damage|
|---|---|---:|---:|
|2 /4031370700|off_track|0.6097|0.0|
|3 /4111953688|off_track|0.7461|0.2|

The weaker local fallback gate passed (required4/4 below25s and development12/12,
median below25s), but its independent holdout gate required at least11/12.
Observed10/12 fails that gate. Neither timing nor completion criteria were relaxed.
These seeds are now consumed and must never be relabeled as fresh holdout.

## What the experiments established

- Continuous path refinement removes integer-grid artificial corners.
- Instantaneous HUD yaw improves motion preview when optical flow fails. Its
  benefit combines measurement accuracy and availability, not just one cause.
- Calibrated pedal allocation permits coasting and improved required mean time
  by0.46s over HUD100 while retaining all11 development finishes and rescuing1.
- Actual-state traction rescued one cell but lost required track4: rejected.
- The separate rollout planner passed required4/4 yet only development6/12.
  Its long-horizon prediction error remained large: rejected.

All negative experiments and source snapshots are retained in the line, rollout
and pedal ledgers. [LIMITATIONS.md](LIMITATIONS.md) links the underlying analyses.
A remaining concern is limited visual anticipation around obstacles/hairpins.
The rolling model also operates with unknown slip when optical flow fails;
its low-slip applicability is not guaranteed. Row-wise path clearance does not
certify the continuous vehicle footprint. These limits were not hidden or patched
after seeing holdout results.

## Validation and use

[Verification](results/verification.json):81 focused tests passed. Packaging had
135 exact sequential synthetic action matches and independent review found no
blocking implementation defect. Across all32 fresh final episodes, process peak
memory was at most100.62MiB, maximum action inference0.080s, and maximum
import/construction0.063s; all passed the recorded resource limits.

The earlier complete repository suite had1,182 tests:1,161 passed,14 errors,
7 skipped. Thirteen errors involved missing historical artifacts; the remaining
memory test passed in isolation and was affected by inherited process peak-RSS.
The failures were preserved and documented in [baseline-tests.json](results/baseline-tests.json).
Official mirrored physics were byte-identical; the competition website was
unreachable with HTTP403, as disclosed in [RULES.md](RULES.md).

Reproduce a required cell in a fresh process with a new output filename:

```bash
.venv/bin/python -m agents.apex_2026.evaluate \
  --agent agents/apex_2026/candidate/agent.py --config '{}' \
  --track-id 1 --seed 516237 --output /tmp/apex-required-new.json
```

Do not substitute evolving `line_agent.py` defaults for the frozen candidate.
Further improvement requires a new, explicitly separated research/validation
cycle. This completed cycle does not authorize official submission or confirmation.
