# Apex 2026: opposite-strategy research

Final result: [REPORT.md](REPORT.md). The [frozen standalone candidate](candidate/agent.py)
finishes required4/4 and development12/12, but holdout10/12 misses the predeclared
11/12 gate. The10–13s objective is unmet. It remains experimental and unadopted.

Created from main `0a887f5` on 2026-10-05 KST by user authorization.
The requested prior `CONTEXT.md` and `agents/apex_2026/` did not exist on
either the initial checkout or fetched main. This is a new experiment, not a
continuation or adoption of an undocumented candidate.

Read [DESIGN.md](DESIGN.md), [RULES.md](RULES.md), and
[benchmark.json](benchmark.json) before running experiments. Root `AGENTS.md`
and `docs/context/current-state.md` remain the repository guidance.

The literal root agent is a learned image policy (restored main model is a
Baseline1Actor state dictionary). The separately recommended arrival-speed
release is a layered reactive corridor controller. Apex instead optimizes
visible free-space paths or predicted action trajectories using image pixels.
No root agent, old candidate, official environment, or scoring code is modified.

Runtime: Python 3.11.16; NumPy 1.26.0, OpenCV 4.8.1.78,
Gymnasium 0.29.1, Box2D 2.3.5, CPU Torch 2.1.0 for the baseline only.
Development uses the ignored repository `.venv`.

All candidates are experimental until the frozen validation report says
otherwise. Historical results are never contemporary measurements.
No official submission or model confirmation is authorized by this experiment.

## Evidence recorded in this session

- [Fresh root/release baseline](results/baseline-20261005.json): required finishes
  0/4 and 3/4 respectively. Release times 18.52/21.70/20.30 seconds; track 4 DNF.
- [Existing test baseline](results/baseline-tests.json): 1,182 tests,
  1,161 passed, 14 errors, 7 skipped. Thirteen errors need missing historical
  artifacts; the packaging memory check passes all 9 isolated tests and suffers
  an inherited `ru_maxrss` measurement artifact in the full-suite subprocess.
- [Line-planner experiments](results/line-experiments.json) and
  [independent rollout experiments](results/rollout-experiments.json) retain
  negative results and exact source/configuration fingerprints.
- [Conservative line development screen](results/line-development-safe.json):
  12/12 finishes, zero damage, median 26.25 seconds. This is development evidence,
  not holdout performance or the 10–13-second objective.
- [Continuous-path development screen](results/line-development-smooth200.json):
  10/12 finishes, median completed lap 17.57 seconds; two lost finishes relative
  to the slower preview controller prevent treating this as unqualified progress.
- [Rollout R10 development screen](results/rollout-development-r10.json):
  6/12 finishes despite 4/4 required finishes. Finish-only speed hid a large
  robustness regression, so this candidate does not qualify.
- [HUD dynamics calibration](results/hud-dynamics-calibration.json): yaw and
  actual wheel angle decoded from the public image; no runtime telemetry access.
  Required smooth200→HUD100 times improve on all four cells, zero damage:
  18.32→17.14, 22.80→21.42, 20.18→19.48, 18.42→17.92 seconds.
  This combines better yaw accuracy with availability during optical-flow loss.
- [Physical feasibility and image calibration](FEASIBILITY.md): measured
  diagnostics and explicitly limited approximations; no impossibility proof.

Use the research evaluator in a fresh process for each episode. For example,
the recorded conservative line configuration is:

```bash
.venv/bin/python -m agents.apex_2026.evaluate \
  --agent agents/apex_2026/line_agent.py \
  --config '{"config":{"lookahead":26,"pursuit_gain":3.5}}' \
  --track-id 1 --seed 516237 --output /tmp/apex-new-result.json
```

Source is evolving. Reproduce an archived result using its recorded source
commit/hash, not merely the latest file. Existing output receipts are never
overwritten. Evaluator/protocol/diagnostic tools are not submission runtime files.

The user removed the original 04:00 KST deadline during this session.
Continue based on experimental evidence, without time-triggered stop criteria.
