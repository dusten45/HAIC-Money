# Competition information

Reconstructed 2026-09-30. The earlier document was unavailable; this is a new,
source-backed summary, not a recovered original or a submission approval.

## Sources and version

- [Official participant README](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/README.md), rechecked live 2026-10-03.
- Official main: `dfb7a2de2178825ca5c5ce20bab01ba67052ba31`; environment `variables-6`.
- [Environment wrapper](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/env_wrapper.py).
- [Vehicle physics](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/core/vendor/car_dynamics.py).
- The [competition website](https://ships-duo-ethical-saver.trycloudflare.com/)
  was inaccessible during this check. Website-only deadlines, upload procedure,
  quota and announcements remain UNKNOWN; actual submission stays blocked.

Live read-only comparison found equal text after CRLF/LF normalization for local
`env_wrapper.py`, `damage.py`, `core/track_variables.py`,
`core/obstacle_contacts.py`, `core/vendor/car_dynamics.py` and
`core/vendor/car_racing.py` against the pinned official commit. This is a source
comparison, not a Linux runtime or driving result.

## Environment and ranking

The agent receives four 84x84 grayscale float32 frames in `[0,1]`, and returns
`[steer, gas, brake]` within `[-1,0,0]..[1,1,1]`. The official runner calls
`Agent()`, optional `reset(observation)`, and `act(observation)`; it does not pass
track IDs, seeds or physical state to the policy. Default frame skip is 4 and
camera warmup is 50 raw frames. Preserve the official observation/reward contract.

Each track has six physical obstacles. Geometry and placement are deterministic
for `(track_id, seed)`; local example seed42 is not an official evaluation seed.
Each new obstacle contact adds 20% damage, reducing grip, engine and steering;
five contacts retire the vehicle. Continuous contact counts once until separation.

Finish requires visiting at least 95% of unique road tiles and a valid forward
finish crossing after leaving the starting area. Further termination conditions:
damage100%, 101 consecutive negative-reward action steps, playfield exit, action
budget exhaustion, or 10 consecutive invalid actions. The negative-reward counter
uses the sum over the action's raw frames; nonnegative reward resets it.

Brief grass contact is not stated to cause immediate disqualification. Grass
friction is0.6, so a deliberate excursion is a performance/safety hypothesis, not
an established benefit or blanket official permission. Finishing eligibility and
termination still apply.

Finished entries rank by shorter simulation lap time; DNF entries rank behind
finishers by progress; execution failures are excluded. Lap time runs from the
end of warmup to the finish-center crossing at 50Hz physics resolution. Raw reward
is not the ranking score. Local default action budget is2000.

## Runtime, baseline and submission

Official runtime: Linux Docker, Python3.11, CPU. Import/construction10s,
reset5s, act5s, process memory1,024MB. Fixed libraries: gymnasium[box2d]0.29.1,
torch2.1.0 CPU, numpy1.26.0 and opencv-python4.8.1.78. The official example agent
only accelerates; it is an installation baseline, not a competitive reference.

The ZIP root must contain `agent.py`, its required model and user modules/data.
Optional `requirements.txt` must pin additional compatible PyPI binary wheels.
See [RESTRICTIONs.md](RESTRICTIONs.md) for archive/import restrictions. Upload
portal, metadata, quota and deadline are not reconstructed from guesses.

## Local diagnostic scope

Document reconstruction does not itself approve a candidate or release a run.
A short development comparison requires a version-controlled protocol, frozen
control/candidate/model hashes, declared cells, seed-reservation audit, pass/fail
criteria and current local contract checks. Existing reserved confirmation/blind
seeds remain reserved across every track ID; their source protocols and
[CONTEXT.md](CONTEXT.md) are authoritative. Reused cells are development evidence.
Site-only submission unknowns do not describe local driving mechanics, but keep
submission blocked. Linux readiness, generalization and SOTA need their own gates.
