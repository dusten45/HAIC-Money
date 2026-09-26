# HAIC research operating contract

## Scope and authority

This is a HAIC-only research workflow. The latest user instruction controls task scope and approvals. For competition rules, environment, submission, and schedule, use this order: (1) the current [competition website](https://ships-duo-ethical-saver.trycloudflare.com/), (2) the current [official Participants repository](https://github.com/2026-HAIC/Participants), (3) the [pinned local official README and LICENSE mirror](docs/sources/official-participants/README.md), and (4) the historical [local competition summary](COMPETITION_INFO.md), experiments, and results. Record each checked source in [docs/sources/INDEX.md](docs/sources/INDEX.md). If an official source conflicts with a local document, stop the external action, update the local rule and validator, and continue from the official source. The competition website was inaccessible on 2026-09-26; recheck it before any external action.

Use executable code and tests for current behavior, frozen protocol/result/run artifacts for a particular experiment, and current documentation as a router. Existing `RESULTS.md`, `SOTA.md`, and archived plans are evidence of their recorded time, not current competition rules. Preserve the official environment. Internal reward, progress, damage, smoothness, and robustness measures are proxies; distinguish them from official score.

## State and approvals

```text
STOPPED
→ DISCOVER
→ HYPOTHESIZE
→ DESIGN_PENDING_APPROVAL
→ IMPLEMENT_PENDING_APPROVAL
→ EXECUTE_PENDING_APPROVAL
→ EVALUATE
→ ADVANCE / REJECT / REVISE / PIVOT
→ RELEASE_IF_GATE_PASS
→ STOPPED
```

`STOPPED` is the default. Design, implementation, and execution each require a separate approval record tied to the exact plan revision/hash. Reading documents does not authorize training, evaluation, packaging, or submission. The CLI defaults to plan-only. Only registered local HAIC command profiles may execute after their approval gate. Official submission, model confirmation, and competition-site upload are separate external actions requiring explicit user authorization immediately before execution; they are not CLI operations.

After `EVALUATE`, only `ADVANCE` may enter `RELEASE_IF_GATE_PASS`; that transition requires all three gates to pass and then returns to `STOPPED`. The current config defines no `NOT_APPLICABLE` release exemptions, so `NOT_APPLICABLE` never passes a release gate. Any future exemption requires an explicit validated config value and transition support; rule compliance cannot be exempted. `REJECT`, `REVISE`, and `PIVOT` return to `STOPPED` without releasing the current candidate. A revised or pivoted effort begins a new cycle at `DISCOVER` with a new plan hash and fresh approvals. `PIVOT` preserves the prior checkpoint.

## Central coordination

The central coordinator owns the current state, active cycle, priorities, resource budget, non-overlapping agent assignments, approval records, and integration decision. Before execution it checks the control, registered split, map/seed set, completion endpoint, falsifier, resource limit, and plan hash. It reconciles duplicate hypotheses, conflicting evidence, rule compliance, and each independent report. Agents work only in assigned read/write scope and do not concurrently edit shared records. Each five-field `AgentReport` is wrapped with its owner, handoff path, and edited paths so integration can verify assignment ownership and scope. A handoff informs the central decision; it does not make it.

Every hypothesis records `hypothesis_id`, `source_ref`, `rule_or_requirement`, `observable_information`, `allowed_action_or_state_change`, `expected_success_endpoint`, `eligible_state`, `control`, `falsifier`, `smallest_decisive_experiment`, and `resource_and_risk_gate`. Its causal path must connect rule or requirement → observable information → permitted action or state change → endpoint. Every agent report separates `fact`, `inference`, `unknown`, `recommendation`, and `source_paths` using [the handoff template](docs/handoffs/REPORT_TEMPLATE.md). Do not infer activation or causality from a model name, code presence, score, or external ranking.

## Search and evaluation

- A batch has at least **4 independent mechanism directions**, at most **8 candidates total**, and at most **2 candidates per direction**.
- Do not sweep thresholds or weights until the proposed mechanism visibly changes behavior or state.
- Pivot after **3 consecutive valid, comparable, non-improving cycles**. An infrastructure-invalid cycle does not count.
- Keep train, tune, held-out, confirmation, and blind identities separate. Never relabel consumed confirmation/blind cells as fresh or tune on reserved blind cells. Private official tracks are a final generalization target, not an available holdout.
- Register control, split, map/seed set, denominator, and conditions before comparison. One seed is a local observation, not a replicated matched improvement. Unmatched cells are not a matched comparison.
- `rule_compliance`, `mechanism_activation`, and `competitive_or_product_outcome` each report `PASS`, `FAIL`, `UNKNOWN`, or `NOT_APPLICABLE` with evidence paths. `UNKNOWN` is not a pass. `rule_compliance=PASS` and `mechanism_activation=PASS` are required for SOTA promotion.
- Among eligible candidates on the same registered evaluation set, compare **completion rate first**. Only on a tie compare median finished lap time, mean incomplete progress, P90 finished lap time, collisions, damage, then p95 act latency. Report official scoring separately. Teacher, smoke, and fixed-actor diagnostic runs do not enter submission-candidate completion rates.

Each new run writes an immutable `run_manifest.json`, append-only `events.jsonl`, and `integration_report.json` under the configured v2 roots. Corrections are new events referencing the old event. Keep source paths and hashes without copying raw data. Preserve old `runs/`, `artifacts/haic/`, and `submissions/` in place; do not automatically scan, import, or rewrite them. The read-only map is [docs/sources/legacy-path-map.md](docs/sources/legacy-path-map.md).
