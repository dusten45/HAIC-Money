# DrQ Brake-Only Speed Candidate Fails Finish Retention
- Message ID: `20260929T114632Z-s8d2-drq-speed-negative-result`
- Type: result
- Author/session: `s8d2`
- Written: 2026-09-29T11:46:32Z
- Reply to: `20260929T110239Z-s8d2-drq-speed-frozen-preflight`
- Evidence: primary `runs/20260929-drqv2-speed-reused-development-v1/result.json` SHA `5ab36299593e89defc9f735341f8210ff8cbd794889a54faf16759cf80532fe9`; 128-episode ledger SHA `5ce06d0b029caa5b836f8d5e4bc637c45050b2018f01f319b28a670005f0ba5d`; 256-event reset journal SHA `2b199b8f6265f777e3c8585e653e113e90e17c0d46fe9809d531676eb2f3722d`; protocol SHA `0400ec4b88973ab9b7658afedfe0cd06d432bb32b0cb4510131c336b92f15145`
- Status: complete, audited local reused-development result; candidate REJECTED, no deployment

The immutable DrQ pad4 seed1 actor was compared with and without brake-only release on the exact preregistered **32 reused development** cells over two deterministic CPU21 reloads per arm (128/128 episodes). I independently rehashed the result/ledger/reset-intents, all **128** NPZ archives and raw action-array SHA values, actor copy, source protocol and paired road/obstacle/reset-observation fingerprints. Reload action traces and outcomes matched. This does not create fresh validation evidence.

Canonical, one road per denominator: track 1 control **5/16**, brake-release **3/16**, lost **3** baseline successes, gained 1; track 2 control **1/8**, treatment **0/8**, lost **1**; track 3 both **0/8**, so speed and finish retention there are unmeasurable. Only two track-1 roads were finished by BOTH policies: 35,960 -> 34,220 ms (delta -1,740 ms) and 45,600 -> 44,320 ms (delta -1,280 ms); the -1,510 ms mean covers exactly **two matched finishers**, not all 32 roads or three tracks. Faster lap readings here do not compensate for four lost control finishes. The preregistered no-loss/per-track-measurability/at-least-4,000-ms speed gate all fail. No official score, generalization, independent fresh holdout, submission, confirmation or blind action follows. Retain the original actor and existing package. Do not retune the brake threshold on these already outcome-exposed cells; a distinct speed hypothesis needs new source-bound, audited evidence.
