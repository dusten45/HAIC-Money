# Stable road-clearance candidate: local package design

## Intent and scope

The user authorized local speed work and prohibited competition-site upload, official submission, and model confirmation. The unchanged pixel-only `StableRiskEnvelopeAgent` improved actual finish-line completion on fresh tune 324–327, held-out 328–331, and one-time confirmation 332–335. Package that exact controller around the frozen selected full-road-guard archive, then test the archive itself locally. The package is not a release or an official score claim.

## Frozen inputs and control

- Control archive: `artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip`, SHA-256 `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40`.
- Candidate source: `haic_agent/stable_risk_envelope_runtime.py`, SHA-256 `31eed1bc0e566d89b7b1d5fc33cb1f28f7662f383e1ae21f7ae814142997c86b`.
- The archive entry point constructs `StableRiskEnvelopeAgent(ObstacleFullRoadGuardAgent(base))`, with the same actor-only base and strict checkpoint loading as the selected control.
- Do not change the candidate's decisions, thresholds, checkpoint, or bundled control modules.

## Package operation

Register a dedicated local packaging profile with only the frozen control archive and source file as required inputs. Its output is a new ZIP under `artifacts/haic-research-v2/<run-id>/`. Verify both input hashes, archive size and inventory, prohibited imports and calls, and write a manifest with each bundled file hash. Register a complete V2 plan, separate design/implementation/execution approval events referencing the user's local-work authorization, then execute once.

## Verification

Import the exact ZIP in a clean Linux CPU environment; check import/create, reset, act, RSS, finite three-value actions, and the forbidden-source audit. Compare exact-package candidate and selected control on previously unopened tracks 1–3 × seeds 336–339 with the same map/seed conditions, 12 episodes per arm, maximum 2,000 decisions. Count only actual finish-line crossings as completed. Compare completion count first, then median finished lap time on a tie. A package failure or worse completion rejects the package; record all failures. No site action.

## Evidence limits

The 324–335 controller results are local diagnostic evidence. The package comparison will be separate fresh evidence. The current public third-place times are 13.62, 17.88, and 15.44 seconds across tracks 1–3; even a passing local ZIP must not be described as meeting the top-three target without comparable official results.
