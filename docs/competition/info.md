# Competition Information

This is a local mirror and audit aid. The competition website and official
Participants repository override it whenever they differ.

## Provenance

| Scope | Source | Last verified | Version / note |
|---|---|---|---|
| Technical contract | [Participants README at `1c11db8`](https://github.com/2026-HAIC/Participants/blob/1c11db8afc2fbfcfb610672b7ee0ecd122c97741/README.md) | 2026-09-23T06:30:22Z | Official `main` commit `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`; README blob `4fd6bc02ddf6064bd09d3bbaaa8fa57253ab65be` |
| Operating schedule, quota, model-selection policy | [competition website](https://ships-duo-ethical-saver.trycloudflare.com/) and loaded [official bundle](https://ships-duo-ethical-saver.trycloudflare.com/assets/index-YpEsg9KV.js) | 2026-09-23T06:30:22Z | SPA shell plus loaded bundle; the site provides no deployment commit |
| Current public-track snapshot | [`/api/tracks`](https://ships-duo-ethical-saver.trycloudflare.com/api/tracks) and per-track [`/api/ranking`](https://ships-duo-ethical-saver.trycloudflare.com/api/ranking?trackId=1) | 2026-09-23T06:30:22Z | Tracks 1 and 2 were exposed as public at verification time |

Refresh these sources before official submission, model confirmation, a mock event,
the final deadline, or first official evaluation after another public track is
released.

## Verified Operating Schedule

All site times below are KST (`+09:00`), with UTC shown for auditability.

| Event | KST | UTC |
|---|---|---|
| Submission window opens | 2026-09-18 09:00 | 2026-09-18 00:00Z |
| First mock competition | 2026-09-24 18:00 | 2026-09-24 09:00Z |
| Second mock competition | 2026-09-30 18:00 | 2026-09-30 09:00Z |
| Submission deadline | 2026-10-06 23:59:59 | 2026-10-06 14:59:59Z |
| Final event | 2026-10-08 18:00 | 2026-10-08 09:00Z |

The closest verified gate at this document refresh is the first mock competition.
This table is the only schedule copy; `docs/context/current-state.md` may link to it
but must not duplicate it.

### Unverified Information Carried Forward

The migration request supplied 15:00 KST model-confirmation cutoffs on 2026-09-24
and 2026-09-30, post-mock public-track releases, and a final composition of four
public plus six private tracks. Those values were not published in the accessible
official README, website bundle, public APIs, or ranking data at the verification
moment. They are retained here as **unverified task-provided information**, not
rules. Recheck the authenticated competition UI before confirmation or any decision
that depends on them.

The official sources also did not expose a numerical final-points formula,
cross-track aggregation formula, final track/seed list, or a tie policy beyond the
ordering below. Do not invent one from local scores.

## Submission Resource and Model Selection

- Each team has three official submissions per KST day; the quota resets at 00:00
  KST.
- A submission consumes quota when server validation begins, including a package that
  later fails validation. Local/browser-only checks that do not start server
  validation are not official submissions; verify the site behavior if unclear.
- The site asks each team to choose one successfully evaluated submission as its
  final model. That one model is used for mock evaluations and final evaluation and
  may be changed until the published deadline.
- A public leaderboard shows each team's best track result and warns that it can
  differ from final rankings. It is not proof that one model produced every visible
  best result.

Record all external actions in [`submissions.md`](submissions.md) and distinguish a
local package from an official upload or confirmation.

## Ranking Semantics

For one track, the verified ordering is:

1. Completed runs rank ahead of incomplete runs.
2. Completed runs rank by shorter simulated lap time.
3. Incomplete runs rank by higher progress.
4. Execution failures are handled separately and excluded from the normal result
   ordering.

The official lap time begins after the initial 50 raw-frame camera-preparation
interval and ends at a valid finish-line-center crossing, measured at 50-FPS physics
tick precision in simulation milliseconds. Completion requires at least 95% unique
road tiles and a valid forward finish crossing; 95% alone is not a finish.

Raw reward, local progress, finish rate, damage, smoothness, and custom robustness
scores remain internal proxies. See [`../evaluation/metrics.md`](../evaluation/metrics.md).
