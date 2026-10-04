# Official contract checked 2026-10-04

The official Participants `main` commit was
`dfb7a2de2178825ca5c5ce20bab01ba67052ba31` (2026-09-28). This is the
`variables-6` environment. The local wrapper, track variables, vehicle physics,
CarRacing environment and finish tracker matched that commit byte for byte.

Sources: [official README](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/README.md),
[wrapper](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/env_wrapper.py),
[environment](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/core/vendor/car_racing.py),
[finish tracker](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/core/finish_line.py),
[obstacles](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/core/track_variables.py).

## Inference and submission

- `Agent()` receives images through optional `reset(observation)` and required
  `act(observation)`: four 84x84 grayscale frames, float32 values in `[0,1]`.
- Return three finite values `[steer, gas, brake]`, within `[-1,1]`, `[0,1]`,
  `[0,1]`. The server clips finite out-of-range values; invalid actions become
  no-ops and ten consecutive invalid actions retire the car.
- Evaluation uses Linux, Python 3.11 and CPU. Import/construction: 10 seconds;
  each reset/action: 5 seconds; participant process: 1,024 MB.
- Fixed packages include Torch 2.1.0 CPU, NumPy 1.26.0, Gymnasium 0.29.1 and
  OpenCV 4.8.1.78. Additional packages require exact versions and compatible
  PyPI binary wheels; runtime downloads are unavailable.
- Put `agent.py` at the ZIP root. Maximum ZIP size: 500 MB; extracted size:
  2 GB; 1,000 files; individual file: 500 MB; compression ratio: 100.
- Submitted Python cannot import `ctypes`, `importlib`, `multiprocessing`, `os`,
  `pathlib`, `resource`, `shutil`, `signal`, `socket`, `subprocess`, or `sys`.
  Dynamic calls `compile`, `eval`, `exec`, and `__import__` are prohibited.
  Executables and native library files cannot be bundled in the ZIP.

## Driving and timing

Preserve `core/`, `env_wrapper.py` and `damage.py`. Use frame skip four and
50 warmup raw frames. Physics runs at 50 FPS; one action spans 0.08 simulated
seconds. Ranking lap time uses the recorded finish physics tick minus the
post-warmup start time, not elapsed wall time or action count.

Finishing requires at least 95% unique road tiles, a valid forward finish-line
crossing, and departure from its front region. Progress alone does not finish
a lap. Completed laps rank ahead of DNFs; completed laps compare simulation
time and DNFs compare progress. Five obstacle contacts retire the car. Also
preserve the official consecutive-negative-reward retirement rule.

Geometry is determined by seed; `(track_id, seed)` determines six obstacles.
Reserve a geometry across track IDs when splitting development and holdout.
The inference interface supplies no seed, track ID or simulator state.

## Verification limits

The [competition website](https://ships-duo-ethical-saver.trycloudflare.com/)
was inaccessible and its hostname did not resolve during this check. Therefore
the live schedule, leaderboard aggregation, hidden evaluation matrix and any
website-only fairness rules remain unverified. No explicit prohibition of
offline seed-specific training or privileged-state teachers was found in the
Participants documentation; absence of a clause does not establish permission.
The candidate uses image observations and internal memory at inference.

Local tests establish measured compatibility and performance only. They do
not establish official Linux certification, unseen-track reliability, or a
competition-winning result.
