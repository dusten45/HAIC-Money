# Official Runtime and Package Restrictions

This local transcription was verified against the pinned official README listed in
[`info.md`](info.md) at 2026-09-23T06:30:22Z and rechecked unchanged on
2026-09-26. It is a convenience mirror only; the current official source wins
on every conflict.

## Runtime and Environment

| Item | Verified current contract |
|---|---|
| Official runtime | Linux Docker, CPU-only, Python 3.11 |
| Environment version | `variables-6` |
| Fixed libraries | `gymnasium[box2d]==0.29.1`, CPU `torch==2.1.0`, `numpy==1.26.0`, `opencv-python==4.8.1.78` |
| Environment | Gymnasium `CarRacing-v2` with six physical obstacles; `(track_id, seed)` determines track and obstacle placement |
| Observation | Four latest grayscale frames, `float32`, shape `(4, 84, 84)`, values in `[0, 1]` |
| Action | Finite shape `(3,)` `[steer, gas, brake]`; steer `[-1, 1]`, gas/brake `[0, 1]` |
| Required API | `Agent` and `act(observation)`; `reset(observation)` is optional |
| Invalid action handling | Exception, non-finite values, or wrong shape become `[0, 0, 0]`; finite out-of-range values are clipped; ten consecutive invalid actions retire the run |

The environment can also retire/end for full collision damage, 101 consecutive
negative action-step rewards, leaving the play area, or max steps. Local-runner
defaults such as `max_steps=2000` and `frame_skip=4` are local defaults, not a claim
about all official evaluator internals.

## Execution Limits

| Operation | Limit |
|---|---|
| Agent import and construction | 10 seconds |
| Each `reset()` | 5 seconds |
| Each `act()` | 5 seconds |
| Participant process memory | 1,024 MB |

## ZIP Layout and Limits

- The ZIP root must contain `agent.py`; `Participants/agent.py` or any nested root
  is invalid.
- It may contain required model files, root `requirements.txt` when needed, and
  necessary user modules/data.
- Compressed ZIP limit: 500 MB.
- File limit: 1,000.
- Uncompressed total limit: 2 GB.
- Individual file limit: 500 MB.
- Individual compression-ratio limit: 100x.
- All model/weight data required at inference must be inside the ZIP; evaluation has
  no internet download.
- Use case-correct relative paths and force CPU model loading where appropriate.

## Additional Inference Dependencies

- Put only required extra inference dependencies in root `requirements.txt` as exact
  `package==version` pins. Fixed official libraries may be omitted.
- Every extra dependency must have an installable official-PyPI Python 3.11/Linux
  binary wheel.
- Unsupported: Git, URL, local-path, custom-index, nested-requirement entries,
  source builds, and dependency conflicts.
- Official README caps additional package installation at 50 packages, 32 KiB of
  requirements content, 180 seconds, and 512 MiB.

## Prohibited Content

Static inspection covers every submitted `.py` file.

- Forbidden imports: `ctypes`, `importlib`, `multiprocessing`, `os`, `pathlib`,
  `resource`, `shutil`, `signal`, `socket`, `subprocess`, `sys`.
- Forbidden dynamic functions: `compile`, `eval`, `exec`, `__import__`.
- Forbidden archive/native extensions: `.com`, `.dll`, `.dylib`, `.exe`, `.msi`,
  `.scr`, `.so`.

## Recheck Gate

Do not treat a passing local validator as official acceptance. Before any approved
upload, reopen the official README and website, compare their current version to
`docs/competition/info.md`, then follow
[`../workflows/prepare-submission.md`](../workflows/prepare-submission.md).
