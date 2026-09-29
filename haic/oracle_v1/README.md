# HAIC Oracle v1

This is a one-time, independent export of the local privileged geometry oracle.
It is a diagnostic controller, not a competition `Agent`, submission policy, or
claim of private-track generalization. The controller reads privileged simulator
state (`track`, vehicle pose/velocity, and obstacle locations).

## Contents and Runtime

- `controller.py`: copied oracle controller and closed-polyline path geometry.
- `runner.py`: small local CLI for running the controller and recording summaries
  and action-trace hashes.
- `requirements.txt`: minimum pinned runtime dependencies for the existing HAIC
  environment. PyTorch is not required.
- `PROVENANCE.json`: source commits, source hashes, environment match, and validated
  result.

The runner deliberately uses the existing HAIC `core/`, `env_wrapper.py`, and
`damage.py`; these files were not copied. At export, those environment sources were
byte-identical to the source used by the oracle evaluation. Run commands from the
HAIC repository root. No external checkout, mount, import path, remote, or shared
data directory is used.

Install the minimal dependency set if the repository environment is not already
installed:

```bash
python -m pip install -r haic/oracle_v1/requirements.txt
```

Run a representative track/seed set. The output directory must not already exist,
and its parent must exist:

```bash
python -m haic.oracle_v1.runner \
  --track-ids 1 5 --seeds 1 3 --avoid-obstacles \
  --output /tmp/haic-oracle-v1-check
```

Run the full 50-cell grid by omitting `--track-ids` and `--seeds`. Each run writes
`metadata.json` and `summary.json` to the chosen output directory. The action-trace
SHA-256 is over the concatenated float32 actions in episode order.

The controller can also be imported directly:

```python
from haic.oracle_v1 import OracleController

controller = OracleController(base_env, target_speed=12.0, avoid_obstacles=True)
action, diagnostics = controller.act()
```

`base_env` must be the unwrapped local CarRacing environment after reset. The
returned action is float32 `[steer, gas, brake]`.

## Source and Validation

The source was inspected at Fresh commit
`a70b35950414a930d5ddaad4ac15733e65774345`. The 100-episode evaluation metadata
points to source revision `27d96562bb71dc32f7ff0c1b85bb01ac3555e10d` and recorded a
dirty worktree because documentation was being edited. Its recorded controller
SHA-256 exactly matches the exported `controller.py`; the corresponding current
Fresh controller is the same Git blob. The package runner is adapted only for
this repository's module path and provenance; the controller itself is copied
unchanged. See `PROVENANCE.json` for the complete hashes.

The original fixed evaluation used `target_speed=12`, obstacle avoidance, frame
skip 4, warmup 50, and at most 2,000 agent actions. Read-only aggregation of its
15 saved run directories verified 100/100 full episodes across 50 track/seed
configurations (track IDs 1-5, seeds 1-10), with each road finishing twice and
identical repeated trace hashes. The action range was 923-1,236 and lap times
were 73.76-98.80 simulation seconds. The result was not collision-free: 98/100
episodes had no damage; track 5 / seed 3 had one collision-positive action at
step 754 in each repeat and finished both. These exposed local cells are neither
an untouched holdout nor an official result.

The source also verifies the following behavior and limitations:

- Obstacle avoidance shifts a smoothed reference path to the obstacle's opposite
  side, using obstacle radius + 1.4 vehicle half-width + 1.2 tracking margin, with
  a 25-unit cosine blend.
- Pure-pursuit steering uses a `6 + 0.25 * speed` lookahead; speed control targets
  12 units/s by default.
- The controller does not guarantee zero collisions or successful completion on
  untested geometries. The measured evaluation is local and exposed.
- This privileged controller must not be used as a submitted competition agent.
