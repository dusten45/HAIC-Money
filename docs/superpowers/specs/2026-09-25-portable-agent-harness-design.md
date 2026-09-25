# Portable Agent Harness Design

- Status: Draft for user review
- Date: 2026-09-25
- Scope: Harness architecture and operating contract for HAIC and future projects

## 1. Purpose

Build a portable agent harness that carries forward the useful operating ideas from the current HAIC research workflow while reducing direct dependence on its existing orchestration and training code. First document how the new workflow is expected to operate; implementation follows only after this design is reviewed and approved.

The harness should make work reproducible, reviewable, and safely gated. It should separate reusable orchestration from project-specific rules, keep past research evidence intact, and make every new run identify its plan, commands, inputs, outputs, and evaluation gates.

## 2. Goals and non-goals

### Goals

- Define a reusable core with an explicit project adapter boundary.
- Implement the HAIC workflow through a new adapter and registered command profiles.
- Preserve historical run and evidence data at their existing locations without copying or rewriting it.
- Put all new harness-managed run outputs in a distinct namespace.
- Make planning the default; require an approved plan before executing an allowlisted command profile.
- Keep official submission and confirmation actions outside this CLI and require separate, immediate user authorization for them.
- Record facts, inferences, unknowns, recommendations, and source paths separately.

### Non-goals

- Port or preserve all internal code from `research_ops` or `training/improvement_loop.py`.
- Automatically discover, ingest, normalize, or re-evaluate historical runs.
- Change historical manifests, archives, reports, models, checkpoints, submissions, or experiment results.
- Duplicate official package, evaluator, or competition-rule validation already provided by the official environment.
- Claim official HAIC performance from local proxy metrics.

## 3. Existing system and migration boundary

The current checkout contains overlapping workflow behavior in `research_ops/` and `training/improvement_loop.py`, alongside project guidance and historical outputs. The new harness will not import either implementation or invoke them implicitly. They remain available as legacy references while the new architecture is developed; this design does not require preserving their internal APIs.

Existing data is authoritative only for the particular historical run represented by its frozen artifacts. The new harness does not scan those directories or merge their results into its own indexes. A future explicit import may be designed as a separate, reviewable operation, but it is outside this initial migration.

### Historical path map

The following are legacy locations and remain in place:

| Historical content | Current path | New harness behavior |
|---|---|---|
| Run records and run outputs | `runs/` | Leave unchanged; do not auto-scan or import |
| HAIC artifacts and checkpoints | `artifacts/haic/` | Leave unchanged; do not auto-scan or import |
| Submission packages | `submissions/` | Leave unchanged; never submit from the new CLI |
| Experiment summaries and indexes | Existing `results/` and `experiments/` paths | Leave unchanged; do not regenerate or merge |
| Research maps and reports | Existing root/docs report and map paths | Leave unchanged; use as historical references only |

New harness-managed outputs use `runs/harness-v2/<run-id>/` and `artifacts/harness-v2/<run-id>/`. This boundary prevents new runs from colliding with old outputs while avoiding an unnecessary copy of approximately 1.8 GiB of historical data. Existing path strings may be repository-relative, run-relative, or internal to an archive; they must not be rewritten as part of migration.

A read-only legacy path map may document how historical materials relate to the new concepts. Any import must be initiated explicitly by a human and must create new derived records without mutating the original artifacts.

## 4. Architecture

```text
Agent / User
    |
    v
Harness CLI ------> Generic Core
                         |
                         +--> Workflow state and gate evaluation
                         +--> Hypothesis and plan records
                         +--> Run manifest, event log, integration report
                         +--> Adapter and command-profile registry
                                   |
                                   v
                             HAIC Adapter
                                   |
                                   +--> HAIC policy and evaluation contract
                                   +--> Allowlisted train/evaluate/package profiles
                                   +--> Official-source references
```

### Generic core

The core owns portable orchestration concepts only:

- workflow state transitions and approval checks;
- hypothesis, plan, run, and integration-report schemas;
- append-only run events and gate results;
- adapter and command-profile registration;
- plan-only behavior and validation of approved plan identity;
- safe invocation of a registered command profile.

The core must not contain HAIC rules, import HAIC training modules, or infer project semantics from legacy directory contents.

### Project adapter

`harness/adapters/haic/` owns HAIC-specific configuration and contracts:

- official source URLs and the local source-precedence rule;
- package, runtime, action-space, and submission constraints;
- evaluation split definitions and protected-data rules;
- success criteria and approved local metric names;
- allowlisted train, evaluate, and package command profiles, with typed arguments and declared input/output paths.

The adapter may call existing lower-level project commands through stable command-line interfaces where useful. It must not import `research_ops` or `training/improvement_loop.py` as its orchestration engine. The new harness must not expose an arbitrary-shell escape hatch.

### Proposed initial code layout

```text
harness/
  core/
    state.py
    schemas.py
    events.py
    registry.py
    gates.py
  adapters/
    haic/
      adapter.toml
      policy.toml
      commands.toml
  cli.py
harness.config.json
scripts/harness/validate.ps1
```

Names and file formats may change during implementation, but the core/adapter boundary and run-output boundary are architectural requirements.

## 5. Documentation roles

The implementation should create or reconcile these canonical project documents without discarding valuable existing history:

| Document | Responsibility |
|---|---|
| `AGENTS.md` | Main operating workflow, source precedence, state machine, reporting format, approval boundaries, and agent coordination |
| `PROJECT_INFO.md` | HAIC adapter mission, official source links, success endpoint, and registered execution entry points |
| `RESTRICTIONS.md` | Project restrictions and data/evaluation boundaries; official sources outrank local transcriptions |
| `RESULTS.md` | Append-only, human-readable result log; do not destructively regenerate past entries |
| `SOTA.md` | Best currently verified candidate, updated only after required gates and review |
| `harness.config.json` | Paths, legacy path map, run defaults, policy references, and command-profile identifiers; no secrets |
| `docs/experiments/` | Durable experiment records and index for new harness runs |
| `docs/handoffs/` | Agent-to-agent handoff summaries with evidence and remaining questions |
| `docs/sources/` | Source snapshots or references and the date last verified |
| `docs/strategy-history.md` | Append-only record of meaningful strategy changes and their evidence |
| `docs/report.md` | Current synthesis of results, limitations, and next decision |

Existing `COMPETITION_INFO.md` and `RULES.md` content should be reconciled into canonical sources/workflows. During implementation, preserve useful material by converting old files into concise compatibility pointers or archiving their content before replacement. Do not delete or overwrite unique historical information without recording where it went.

## 6. Workflow and approval gates

The default workflow is:

```text
STOPPED
  -> DISCOVER
  -> HYPOTHESIZE
  -> DESIGN_PENDING_APPROVAL
  -> IMPLEMENT_PENDING_APPROVAL
  -> EXECUTE_PENDING_APPROVAL
  -> EVALUATE
  -> ADVANCE | REJECT | REVISE | PIVOT
  -> RELEASE_IF_GATE_PASS
  -> STOPPED
```

- `STOPPED` is the safe default. The CLI may inspect and validate configuration but cannot start project commands.
- `DISCOVER` records the problem, available evidence, and unknowns.
- `HYPOTHESIZE` states a falsifiable mechanism and expected behavior change.
- Design, implementation, and execution each require an explicit approval record tied to the relevant plan revision/hash. Approval of this architecture document alone does not approve any future experiment or external action.
- `EVALUATE` runs only registered evaluation profiles and records the split, metrics, and validity status.
- `ADVANCE`, `REJECT`, `REVISE`, or `PIVOT` records the evidence-based decision and next action.
- `RELEASE_IF_GATE_PASS` is a local readiness state only. It cannot submit, confirm a model, or contact an external competition service.

### Search discipline

- Each search batch explores at least four independent mechanism directions.
- A batch has at most eight candidates total and at most two candidates per direction.
- Do not sweep thresholds until a mechanism has changed the behavior or state being measured.
- Pivot after three consecutive comparable, valid, non-improving cycles. Invalid infrastructure runs do not count. The adapter defines what makes cycles comparable and which outcomes constitute improvement.
- Keep training, validation, confirmation, and blind splits distinct. Never relabel consumed confirmation/blind data as fresh or use reserved blind data for tuning.

These defaults must be configurable by the adapter when competition rules or a documented decision require a stricter policy. A weaker setting requires a recorded rationale and approval; it must not silently weaken official restrictions.

## 7. Records and observability

Each run directory contains at minimum:

- `run_manifest.json`: run ID, adapter/version, plan hash, command-profile ID, declared inputs and outputs, source revision when available, split, timestamps, and status;
- `events.jsonl`: append-only state transitions, approvals, command start/finish, and errors;
- `integration_report.json`: gate outcomes and evidence references.

The integration report has three independent gates:

- `rule_compliance`;
- `mechanism_activation`;
- `competitive_or_product_outcome`.

Each gate is `PASS`, `FAIL`, `UNKNOWN`, or `NOT_APPLICABLE`. `UNKNOWN` never counts as `PASS`. A mechanism may pass activation while failing outcome; passing compliance alone cannot justify advancing a candidate. Every factual claim in an agent report must include a source path or be labeled as an observation from the current run.

The event log and result record are append-only for a run. Corrections are new events that reference the superseded record; they do not rewrite history silently.

## 8. Configuration and command safety

- The CLI starts in plan-only mode.
- Executing a command requires an approved plan hash, an adapter-registered profile, and declared inputs/outputs.
- Profiles accept structured arguments; they do not accept arbitrary shell text.
- Secrets must come from the user's existing secret mechanism and must not be written to configuration, manifests, logs, or reports.
- The harness records the exact profile and normalized arguments used. If a lower-level executable cannot provide a stable version identifier, the manifest records that limitation explicitly.
- Official submission, model confirmation, and external evaluation are not command profiles in this CLI. They remain separate actions requiring explicit authorization immediately before execution.

## 9. Validation and verification

`scripts/harness/validate.ps1` checks harness documentation, configuration schema, required paths, adapter registration, command-profile references, and the safety defaults. It does not duplicate official package validators, runtime checks, or competition evaluators.

During implementation, verification should be proportionate to each change: schema and transition checks for the core; dry-run/profile validation for the adapter; and a no-write boundary check confirming that legacy `runs/`, `artifacts/haic/`, and `submissions/` were not modified by initialization. Do not run training, simulation, evaluation, or package generation merely to verify harness restructuring unless the user separately requests such runs.

## 10. Migration sequence

1. Add and review this architecture specification.
2. Inventory the current project guidance and execution entry points; map concepts to the new documents and adapter contract.
3. Create the canonical documentation and configuration skeleton while retaining valuable historical content and leaving run data untouched.
4. Implement the generic core and HAIC adapter behind the documented boundary.
5. Add the plan-only CLI and structural validator.
6. Verify initialization and dry-run behavior without touching historical data or launching research workloads.
7. Make the new harness the documented default only after the above evidence is reviewed. Keep legacy entry points available as explicitly labeled historical tools until their removal is separately decided.

This design intentionally does not authorize data import, training, competition submission, model confirmation, or other external actions.

## 11. Open implementation decisions

Resolve these during the implementation plan, not by weakening the architectural boundaries:

- exact Python package/module names and minimum supported Python version;
- JSON versus TOML split for schemas/configuration;
- how approval records are represented in the local workflow;
- adapter versioning and migration behavior;
- whether an explicit, opt-in historical-run importer is needed in a later phase;
- the precise definition of comparable cycles and improvement thresholds for each HAIC metric.
