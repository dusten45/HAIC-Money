# Checkpoint provenance audit — 2026-09-24

## Conclusion

No existing actor checkpoint reviewed here is **clean 증명됨** under the requested lineage criterion: the starting actor must be provably free of official, tune, and held-out data exposure. Six representative artifacts are classified **오염 확인** because each records tune-based checkpoint selection, teacher demonstrations, or official-track entries in its training split. Missing metadata is recorded separately; it is not treated as evidence that an exposure did not occur.

This is a read-only artifact provenance audit. It does not assess competition legality or reinterpret `RESTRICTIONS.md`. No training, evaluation, code edits, `SOTA.md` edits, or `RESULTS.md` edits were performed.

## Hash and evidence method

- File SHA-256 was computed from the actual checkpoint bytes.
- Model-state SHA-256 uses the scheme recorded in the pilot source manifest: sort `model_state` entries by name, then hash UTF-8 name + NUL + `str(dtype)` + NUL + compact JSON shape + NUL + contiguous CPU tensor bytes. The recomputed SOTA hashes match the manifest.
- Checkpoint weights were not run through a policy or environment. Only the saved state tensors were read for hashing; provenance metadata and configuration files were inspected. No map geometry or episode evaluation file was used for these classifications.
- The audit source manifest is [source-manifest.json](C:/Users/koi/.codex/worktrees/strategy-a-curve-brake-reward/HAIC/artifacts/haic/throttle-expansion-ppo-pilot-v2/source-manifest.json), SHA-256 `9C765CE8DA05F168FFA87F8A94B7C8CF0DBFDE996529AAE994D57E2EE4A1117B`.

## Representative actor artifacts

### SOTA actor

- Checkpoint: [policy.pt](C:/Users/koi/Coding/HAIC/artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt)
- File SHA-256: `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`
- Canonical model-state SHA-256: `3AA08AF8F196FBBCD6D8DB64780E7717FF6CCF247C1CD800061753CDDD6539CD`
- Step: `19456`
- Classification: **오염 확인**
- Direct checkpoint metadata records teacher warmup enabled, 30 epochs, 3,625 demonstration steps, and a split with 22 train entries including official-track entries, 6 tune entries, and 8 held-out entries. It also records the tune selection metric.
- [experiment_config.json](C:/Users/koi/Coding/HAIC/artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/experiment_config.json) records the parent path `site-map-official-domain-speed-reward-ppo8192-u8-lr2e6-seed8101/policy.pt`. Its [training-result.json](C:/Users/koi/Coding/HAIC/artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/training-result.json) records that teacher warmup was reused from the checkpoint.
- The direct parent artifact currently hashes to file SHA `DB34782CC5A4515E07A33FD9640383FF2A0303CC72E2B26AA15EC6D31EE443E1`, model-state SHA `4DCE4E0281ED09CB38B5D588F0691321A023EEAAA134A752C431148C7985E823`, step `15360`. These are hashes of the files as observed now; the child config did not record the parent hash at training time.

### `site-map-ppo-8192`

- Checkpoint: [policy.pt](C:/Users/koi/Coding/HAIC/artifacts/haic/site-map-ppo-8192/policy.pt)
- File SHA-256: `5BCCB38610CC7F0E8B4EA2DDEAEF38CC5945B90277616366D74203CDA8BC3D26`
- Canonical model-state SHA-256: `3AC25D988F4F56B80E79D48C85FD40E373687ABB5471644727A62A24DB7BA40E`
- Step: `8192`; classification: **오염 확인** because checkpoint metadata records a tune selection metric and includes tune/held-out groups. Teacher warmup and parent checkpoint are **not recorded**; this is not proof that neither existed.

### `site-map-pedal-ppo-*` representatives

| Checkpoint | Step | File SHA-256 | Model-state SHA-256 | Classification |
|---|---:|---|---|---|
| `site-map-pedal-ppo-seed-8101/policy.pt` | 8192 | `36E84A3A652EFC88D41E1F2C08D33ED06F23322064FCF5760D5D44220DFF7A9F` | `E13099AA4B6297FD7C66AAF3BEBD01151EA4357D3E1819D90CD7DEC68FE9CECB` | 오염 확인 |
| `site-map-pedal-ppo-v2-seed-8101/policy.pt` | 8192 | `F8A3CEA8E59794002052D4845235A83B71493E171939FC73DA1CFE6C3ED1486A` | `E53F330ED2916F00086F899B2DEF1D9430139AA78492FE596424B5CAED8F1C02` | 오염 확인 |
| `site-map-pedal-ppo-v3-seed-8101/policy.pt` | 6144 | `5D033CC64BAD21E38E203E12D0F7F5F54050B7C711E49D1612A5B918F60C9E88` | `F73521CAE013A48C21D17CCB8484F01F1BD3B2F7A3E0C11490EC7ACF7AC70D18` | 오염 확인 |

For each, the saved metadata records 4 train entries, 2 tune entries, 5 held-out entries, and the tune selection metric. Teacher warmup and parent path are not recorded in these checkpoint metadata files; that absence cannot prove a clean start.

### `site-map-actuator-range-v2-ppo2048`

- Checkpoint: [policy.pt](C:/Users/koi/Coding/HAIC/artifacts/haic/site-map-actuator-range-v2-ppo2048-seed8101/policy.pt)
- File SHA-256: `843D4A2E2CF97DD44192C2CE1C70053717FF5360367502FCCB0582D2B8AFCC8B`
- Canonical model-state SHA-256: `6AF8AC2DE7DE247576AF1C23D8318E8E507D0C300361ECD426F0FD8810786F47`
- Step: `2048`; classification: **오염 확인**. Checkpoint metadata and [training-result.json](C:/Users/koi/Coding/HAIC/artifacts/haic/site-map-actuator-range-v2-ppo2048-seed8101/training-result.json) record teacher warmup enabled for 30 epochs with 1,066 demonstration steps, plus tune-based checkpoint selection. Parent checkpoint is not recorded.

## Recorded SOTA ancestry

The available `experiment_config.json` files record this parent path chain:

`obstacle-risk SOTA → speed-reward → interleaved → perception-feature continuation`

The speed-reward checkpoint metadata records reused teacher warmup (30 epochs/3,625 steps) and official-track entries in its train group. The interleaved checkpoint also records reused teacher warmup and official-track train entries. The perception-feature continuation records teacher warmup and tune selection; its train group has no official-track entries in its metadata. Its next parent is not established here because its experiment config and checkpoint parent field are absent.

The child configs contain parent paths but no parent file hashes from the time each child was trained. The parent hashes in the companion JSON are current artifact hashes only. The direct SOTA checkpoint’s own file/model-state hashes are independently recorded in the pilot source manifest.

The pilot manifest says `execution.started=false`. Its proposed active split is custom train plus custom tune, with held-out geometry not materialized and no official geometry used in training/tune. That pilot did not produce a new actor and does not remove the historical lineage of its SOTA input.

## Audit record

- Rules and database references: [RULES.md](C:/Users/koi/Coding/HAIC/RULES.md), [RESTRICTIONS.md](C:/Users/koi/Coding/HAIC/RESTRICTIONS.md), [SOTA.md](C:/Users/koi/Coding/HAIC/SOTA.md), [RESULTS.md](C:/Users/koi/Coding/HAIC/RESULTS.md).
- The companion inventory is [checkpoint-provenance-audit-20260924.json](C:/Users/koi/Coding/HAIC/research/artifacts/checkpoint-provenance-audit-20260924.json).
- No checkpoint in this representative inventory is classified `clean 증명됨`.
