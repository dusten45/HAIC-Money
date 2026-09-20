# Task 4 report — deadline-bounded CEM inference

## Delivered interface

- `haic_agent/planner.py` implements a CPU CEM planner.  It samples from the
  PPO Normal prior, shifts the prior unconstrained action sequence by one
  decision, and autoregressively rolls every candidate through
  `LatentDynamicsEnsemble.predict`.
- Candidate scores add learned progress/reward and subtract one decision time
  cost, collision probability, off-track probability, and ensemble
  uncertainty.  Candidate population work is split into fixed 16-candidate
  chunks and checks the same absolute monotonic deadline before each chunk and
  horizon step.
- `Agent.act()` takes one monotonic timestamp at entry and uses a 4.5-second
  total budget by default.  It computes the bounded PPO action first, then
  returns it whenever planning expires, has invalid output, or has no loaded
  dynamics model. `reset()` clears the planner cache.
- Default packaged names are `policy.pt` and `dynamics.pt`; the Agent loads
  Task 2 `model_state` and Task 3 `model` checkpoints on CPU. The local runner
  accepts `--plan-budget`, `--policy-checkpoint`, and `--dynamics-checkpoint`.

## TDD evidence

1. RED: `.venv\Scripts\python.exe -m unittest tests.test_planner tests.test_agent_inference -v`
   initially produced eight expected errors: no `haic_agent.planner` module and
   no injectable Agent policy/dynamics/planner interface.
2. GREEN: the same focused suite passed 8/8 after planner and Agent were added.
3. RED: the runner argument regression failed with `ImportError: cannot import
   name 'build_argument_parser'` before the `--plan-budget` parser was added.
4. GREEN: focused suite passed 9/9 after the runner connection.
5. RED: default checkpoint regression failed because `policy.pt` and
   `dynamics.pt` constants did not exist. GREEN: it passed after default
   loading was connected and verifies both actual state dictionaries load.

## Verification

- Focused: `.venv\Scripts\python.exe -m unittest tests.test_planner tests.test_agent_inference -v`
  → 10 passed.
- Full: `.venv\Scripts\python.exe -m unittest discover -v` → 71 passed,
  1 expected server-repository skip.
- Static: `git diff --check` passed. New inference code imports only numpy,
  torch, and `haic_agent` modules; it imports neither simulator modules nor
  training labels.
- Closed-loop smoke: local variables-6 run with track 1, seed 42, two decision
  steps, 0.20-second budget, a Task 2 policy checkpoint, and a Task 3 dynamics
  checkpoint completed without timeout/invalid action. It stopped only because
  `--max-steps 2` was requested (DNF/max_steps); it is an interface smoke, not
  a driving-performance result.
- CPU timing with those actual checkpoints and default 4.5-second budget:
  three planner-backed `act()` calls were 0.157 s, 0.140 s, and 0.141 s;
  maximum 0.157 s, leaving 4.343 s under the operating budget. Planner cache
  was populated, confirming the CEM path executed.

## Limits

- The smoke uses tiny existing training artifacts and two decisions. Task 5
  must measure full held-out episodes and choose whether PPO+CEM improves over
  PPO-only.
- Legacy `Baseline1Actor` remains only for pre-existing export compatibility
  tests and is not referenced by Agent's inference path. Task 5 can remove the
  legacy export/package layout while making the final bundle include the
  `haic_agent` modules and both learned checkpoints.
