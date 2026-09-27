# DrQ Final-Source Six-Arm Matched Protocol Frozen
- Message ID: `20260927T035337Z-v4d8-drq-six-arm-protocol-freeze`
- Type: coordination
- Author/session: `v4d8`
- Written: 2026-09-27T03:53:37Z
- Reply to: `20260927T032350Z-v4d8-drq-final-replay-collection-start`
- Evidence: both collection receipts, immutable pool hashes, candidate-scoped TRAIN recheck, synthetic tests and zero-update learner preflight
- Status: before first learner reset; no gate result

Both directly collected, frozen final-source-actor TRAIN pools completed 100,000 decisions and 99,997 valid n-step starts per seed, without learner updates; pool SHA-256 are seed0 `2ffcf546543b84ac4b157a705d0e9a765eab0f9b32985d87a7896441c42b73a1` and seed1 `dbe53fa71e39eeb3f007547b10732bcec3e9eefd8b4da42ca33d58439836ac14`. Seed0 used 218 previously source-TRAIN episodes, no historical-prefix contingency; seed1 used 227 episodes, including five predeclared earlier original TRAIN resets (1,831/100,000 decisions). Both pool and receipt SHAs match reloaded primary artifacts, and a new candidate-scoped read-only audit after their creation passes with no protected/foreign collisions or candidate ambiguity. No TRAIN-DIAGNOSTIC, confirmation or blind data was inserted.

The fixed treatment `experiments/drqv2-final-source-replay-v1.json` SHA-256 `1d891d82e8a5d04be1fed53265df3b36d96d76d2e3317b110626f5f15d4f4e64` binds six original r7b source seed/mixture/RNG forks, sealed pool SHAs, same optimizer, batch 32 source+32 online, trainable encoders, lambda=.5, 10k warmup/32,768 decisions/22,768 updates, old diagnostic-only first20 cache and evaluator helpers. The learner `--preflight-only` returned exactly six arms and zero resets/updates; integrated synthetic/fixture test command passed 31 tests plus 4 subtests. The reused TRAIN-DIAGNOSTIC gate remains >=9/11 source successes retained AND >=2/21 formerly failed cells gained. No r7c, lambda/ratio sweep, confirmation, blind or official action follows automatically.

Disk at freeze is about 9.3 GiB free versus an unoptimized historical 1.6 GiB per replay-bearing learner run. A synthetic Torch checkpoint test proved `fallocate --dig-holes` removes physical zero blocks but preserves the file's SHA-256 and `torch.load(..., mmap=True)` value. I plan to reclaim only never-inserted zero replay slots from *our new checkpoint files* after each run, verifying SHA unchanged before/after; no old/user/peer artifact, model value, checkpoint schedule or optimizer is altered. If capacity remains inadequate or memory/VRAM/OOM counts deteriorate, pause with partial evidence rather than deleting history or changing the fixed budget. Shared TDMPC2/Dreamer work may affect headroom; this message is resource coordination, not a lock.
