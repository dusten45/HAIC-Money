# Official contract recheck

Checked 2026-10-04 UTC against official Participants commit
[`dfb7a2de2178825ca5c5ce20bab01ba67052ba31`](https://github.com/2026-HAIC/Participants/tree/dfb7a2de2178825ca5c5ce20bab01ba67052ba31).
README SHA256: `5aa574e36c4be1755c0264ae7f2cd394dc3a297ac0b21d929407f61779c824be`.
Official website proxy CONNECT returned HTTP 403. Current website announcements
and public seed inventory could not be independently verified; required cells
are user-specified, not asserted to be a verified official inventory.

Independent source comparison found all official `core/` files, `env_wrapper.py`
and `damage.py` byte-identical to the repository. Local runner adds model/planning
constructor options; physics and finish timing are unchanged.

- Linux CPU, Python 3.11; environment `variables-6`.
- Input: float32 `(4,84,84)` grayscale frames in `[0,1]`, newest frame last.
- `Agent.act(observation)` returns finite `[steer,gas,brake]`, bounded
  `[-1,1]`, `[0,1]`, `[0,1]`. Optional `reset(observation)`.
- No geometry, track ID, seed, car telemetry, reward or simulator object is
  provided to the agent. Pixel-derived planning is compatible with this API;
  this is a contract interpretation, not explicit organizer endorsement.
- Import/construction <=10 s; reset/act <=5 s; process memory <=1024 MB.
- Fixed libraries: gymnasium[box2d] 0.29.1, CPU torch 2.1.0,
  numpy 1.26.0, opencv-python 4.8.1.78.
- Submitted Python must not import ctypes, importlib, multiprocessing, os,
  pathlib, resource, shutil, signal, socket, subprocess or sys. Dynamic
  compile/eval/exec/__import__ prohibited. Research evaluator is not packaged.
- No runtime downloads. ZIP root agent.py; <=500 MB compressed, <=1000 files,
  <=2 GB uncompressed, each file <=500 MB and <=100x compression ratio.

Finish requires >=95% visited tiles plus valid forward finish-zone entry,
center crossing and front exit; qualification alone is insufficient. Physics
is 50 Hz. The published local default repeats each action four physics frames,
warms up for 50 raw frames, and permits 2000 actions. Lap time uses actual
recorded center-crossing timestamp minus post-warmup start time. This document
does not assert independently verified server evaluator defaults.

Other termination includes full damage (five obstacle collision events),
101 consecutive negative action rewards, playfield exit, max steps and ten
consecutive invalid actions. Do not alter these to improve a result.
