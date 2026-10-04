# Apex 2026: opposite-strategy research

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

The user removed the original 04:00 KST deadline during this session.
Continue based on experimental evidence, without time-triggered stop criteria.
