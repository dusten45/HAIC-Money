# Short grass excursions: official-rule and mechanics audit

Checked at 2026-10-04T19:57:39.029292+00:00. Read-only research: zero simulator resets, no holdout access,
no official simulator edits, and no candidate change.

## Conclusion

The currently fetched official Participants README does **not** prohibit brief
ordinary driving on grass or prescribe that all wheels remain on asphalt.
It explicitly defines grass friction and completion/retirement conditions.
Together with the identical official executable sources, this supports researching
small, continuous, observation-planned corner cuts that rejoin the same visible
road and still satisfy normal completion. This is an interpretation of published
rules, not an organizer's explicit endorsement of corner cutting.

The current competition website could not be rechecked: its HTTPS proxy CONNECT
returned HTTP403. Therefore an additional website-only announcement could not be
excluded. No official submission or acceptance claim is made.

This finding does not authorize environment modification, teleportation, hidden
track/seed access, exploiting finish-state bugs, or bypassing the normal finish
procedure. A planner should seek a normal nearby racing line, not an artificial
qualification/finish trick. No such experiment was executed in this audit.

## Current official evidence

`git ls-remote https://github.com/2026-HAIC/Participants.git refs/heads/main`
returned `dfb7a2de2178825ca5c5ce20bab01ba67052ba31`.
The README and six relevant code files were then fetched at that exact commit.
All six local mechanics/rule files match those official bytes. Root README is a
project document and is not asserted to match the official README.

- [Official README](https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/README.md), sections7–8:
  “잔디 마찰 계수는 표준 환경과 같은 `0.6`입니다.”
  “고유 도로 타일을 95% 이상 방문한 뒤 정상 방향으로 결승선 재통과”
  “음수 보상이 101 행동 스텝 연속 발생하여 오프트랙 판정”.
- The README explicitly says off-track is not determined solely by vehicle
  coordinates: negative total reward over the action's frame-skip interval
  increases the counter, and a nonnegative total resets it.
- Other published restrictions cover observation/action API, CPU/runtime/package
  limits, prohibited imports/dynamic execution and preserving the simulator.
  No grass-driving or ordinary corner-cut prohibition was found in this README.

## Completion, contact and retirement

- `core/vendor/car_racing.py:FrictionDetector._contact`: a wheel's ordinary road
  contact adds the tile to that wheel's contact set. The first visit to each tile
  increases unique count once and adds `1000/N` reward. Re-visiting tiles cannot
  repair a unique-visit deficit; hull-only overlap does not qualify as a wheel
  tile visit.
- `core/finish_line.py`: progress must reach at least0.95. The vehicle must have
  departed the starting region, enter the finish zone through its back with
  qualification, cross the center in the forward direction, then leave through
  its front while inside the lateral width. Forward velocity must be positive
  and lateral/forward speed ratio at most4. Qualification alone never finishes.
- `core/vendor/car_racing.py`: raw reward is minus0.1 per physics step plus newly
  visited-tile reward. There is no separate instant grass-contact disqualification.
  Leaving the playfield, `|x|>2000/6` or `|y|>2000/6`, ends the episode.
- `env_wrapper.py`: the negative counter uses the whole four-tick action reward,
  not a grass sensor. At default50Hz/four-frame skip, 100 consecutive negative
  actions span8.00s; the101st triggers retirement at about8.08s if starting with
  a zero counter. A preexisting counter reduces the remaining time. These are
  retirement mechanics, **not a recommended or guaranteed safe cutting duration**.
- Damage still accumulates through normal obstacle collisions. New-tile contacts
  usually outweigh the minus0.4 action cost on these track sizes, but rejoining
  already visited road does not necessarily reset the negative counter.

## How much cutting is possible?

There is no published universal limit in metres or a separate grass-time quota.
The only global tile-count allowance implied by qualification is
`N - ceil(0.95*N)` unvisited tiles over the entire lap. It is not a per-corner
budget, and a pixel-only agent does not receive authoritative N or visited count.

Previously recorded REQUIRED geometry sizes, read without new resets:

| Track | TilesN | Minimum unique visits | Maximum unvisited tiles |
|---|---:|---:|---:|
|1|280|266|14|
|2|336|320|16|
|3|309|294|15|
|4|279|266|13|

Nominal centreline tile spacing is3.5m. Multiplying the counts by3.5 does **not**
produce a legal or safe shortcut-distance bound: road polygons have width,
contact footprints overlap several tiles, and corner geometry changes the relation
between skipped arc and driven distance. No maximum-distance performance claim
can be made from these counts.

A conservative research design would retain an explicit cost for missing visible
road coverage and allow only a short departure with a visible return to the same
bend, keeping obstacle and physical-footprint constraints. Do not assume all5%
remain available late in a lap. The independent normal finish requirements remain
mandatory regardless of any proposed line.

## Grass physics for a planner

The official tire-force cap is400N per wheel on undamaged road. A wheel with no
road contact receives a0.6 multiplier:240N, applied to the combined longitudinal
and lateral force norm. Collision-damage grip multiplies it again. Grass does not
simply scale speed or instantly reduce the engine-power constant.

This is **per wheel**: a wheel touching road uses road friction even if another
wheel is on grass. A predictor that always assigns every wheel road contact
(such as the current ShadowCar) is optimistic for a deliberate grass excursion.
Such a trajectory must use per-wheel surface assumptions or a conservative grip
bound; current all-road prediction evidence does not establish grass feasibility.

## Evidence hashes

Downloaded source files and audit receipts are preserved under
`/tmp/apex-v2-grass-rule-audit`.

- `README.md`: `5aa574e36c4be1755c0264ae7f2cd394dc3a297ac0b21d929407f61779c824be`
- `core/finish_line.py`: `7c16dda1378dcd65aed46d9da96608aa055124c2f27d8fb00c309078361d4204`
- `core/vendor/car_racing.py`: `fcd3c36087d7b2721efcd27538b2c9576fa48a77b276ca17c2185b410b0a8c69`
- `core/vendor/car_dynamics.py`: `8c5bc529d6dd60f5f1affeafadc9036f9cb53db98da658a80db77cdc5c26fece`
- `core/track_variables.py`: `cd28fdfc79f12614262f61b1ac13d670d03fad940d753a2ec185390617c0ed58`
- `env_wrapper.py`: `8687215412da0a1f34534623d5050030d85c629a70cb922e9d2199c513a3941e`
- `damage.py`: `b74d15bd2de623cea1345e0469625d4b58f1634d073151a35edf1f05ea35fa30`
