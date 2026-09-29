# Speed coupled exact ZIP held-out comparison

Status: new cycle at DISCOVER; revision 1. This is an unchanged-package confirmation, not a new mechanism-search batch. The previous `speed-coupled-preview-tune-20260929` REJECT and its road-contact falsifier remain unchanged.

## Hypothesis and causal path

- `hypothesis_id`: `speed-coupled-exact-zip-heldout-20260929`
- `source_ref`: `runs/haic-research-v2/speed-coupled-preview-tune-20260929/integration_report.json`, its `events.jsonl`, and `docs/plans/2026-09-29-speed-coupled-preview-steering.md`
- `rule_or_requirement`: the selected local agent must cross the qualified finish line reliably; among equally complete candidates prefer shorter finished laps. The user permits some grass driving and prioritizes speed. Road contact is diagnostic, not a disqualification rule.
- `observable_information`: current and previous 84×84 grayscale observations, the base pixel controller action, visible road edges and obstacles, and image-derived speed. Evaluator-only wheel contacts do not reach the agent.
- `allowed_action_or_state_change`: use the already frozen ZIP controller to begin turns earlier and add gas on visually stable stretches, with braking for strong bends and near obstacles. No policy, threshold, or ZIP byte changes during this comparison.
- `expected_success_endpoint`: qualified finish-line crossing in at least as many matched cells as the selected control, followed by shorter median finished lap time on a completion tie.
- `eligible_state`: held-out local diagnostic only; no SOTA release from this one experiment.
- `control`: selected full-road-guard ZIP `artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip`, SHA-256 `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40`.
- `candidate`: speed-coupled preview ZIP `artifacts/haic-research-v2/speed-coupled-preview-candidate-20260929/submission-speed-coupled-preview.zip`, SHA-256 `54c117287390704acc52505d680c31574fdcf53ac062cbb0ce5258a57058c324`.
- `falsifier`: candidate has fewer actual finishes than control, invalid actions, package/runtime limit violations, or an equal finish count with no shorter median finished lap. A candidate-only nonfinish is reported separately even when aggregate completion ties. Wheel-road exposure is reported but does not alone decide the outcome.
- `smallest_decisive_experiment`: exact extracted ZIPs on tracks 1–3 × untouched held-out seeds 2004–2007, 12 paired cells per arm, fixed 2,000-decision cap. These identities were reserved before prior tune, and no v2 run manifest consumed them at this plan's preflight. Reserve 2008–2011 for later independent confirmation without opening them now.
- `resource_and_risk_gate`: Linux Python 3.11 CPU, 2 CPUs, 2 GiB, 1,800 s maximum. Static ZIP checks, exact input hashes, valid action shape/finite range, import/create/reset/act and RSS limits. Preserve all episodes and failures, source inventory, immutable manifest, append-only events, integration report, and no external action.

## Selection and interpretation

This stage compares actual finish counts first. A tie uses median finished lap, mean incomplete progress, P90 finished lap, collisions, damage, then p95 act latency. Record actual retire reasons and any-wheel/all-wheel road exposure if available, without an automatic grass veto. This result is a single held-out split; even an ADVANCE needs unchanged-package independent confirmation and rule/mechanism/product gate review before selection. Site upload, official submission, and model confirmation remain prohibited.
