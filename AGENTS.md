# HAIC-Money Agent Constitution

## Mission

Develop the strongest possible agent for the 2026 HAIC competition. The project is
an experimental research codebase: architecture, training methods, model choices,
and workflow are provisional unless evidence says otherwise. Do not treat the
current implementation as a permanent design.

## Source Precedence

### Competition rules, environment, submission, and schedule

Use this order when facts conflict:

1. Current [competition website](https://ships-duo-ethical-saver.trycloudflare.com/)
2. Current [official Participants repository](https://github.com/2026-HAIC/Participants)
3. Local mirrors in `docs/competition/`
4. Historical experiments, archived documents, and memory

Before an official submission, model confirmation, mock evaluation, final deadline,
or first official evaluation after a newly released track, recheck the official
sources. If an official source conflicts with a local document, stop the external
action, update the local rule/validator, and continue from the official source.

### Local implementation and research facts

Use executable code and tests for current behavior; use frozen protocol/result/run
artifacts for a particular experiment; use current documentation as a router and
summary. Archived Markdown and remembered results are not authoritative evidence.

## Documentation Router

| Need | Read |
|---|---|
| Current research state, blocker, next gate | `docs/context/current-state.md` |
| Architecture or runtime boundary | `docs/architecture/overview.md` |
| Improve performance | `docs/workflows/improve-performance.md` |
| Run a new experiment | `docs/workflows/run-experiment.md` |
| Investigate a regression | `docs/workflows/investigate-regression.md` |
| Prepare an official submission | `docs/workflows/prepare-submission.md` |
| Select or confirm a model | `docs/workflows/select-and-confirm-model.md` |
| Official rules, schedule, package restrictions | `docs/competition/` |
| Internal evaluation and generalization rules | `docs/evaluation/` |
| Long-lived experiment evidence | `docs/experiments/INDEX.md` and `experiments/` |
| Important settled decisions | `docs/decisions/INDEX.md` |
| Candidate and official-model status | `docs/results/MODEL_STATUS.md` |
| Current implementation plan | `docs/plans/active/` |
| Historical plan/design snapshots | `docs/plans/archived/`, `docs/architecture/history/`, and `docs/archive/` |

Do not read the whole documentation tree by default. Start substantial work with
this file, `docs/context/current-state.md`, and the relevant active plan; then read
only the workflow, evaluation evidence, decision, or architecture document needed
for the task.

## Repository Layout

- Put new reusable project code in `haic/`, with algorithm-specific reusable code
  in `haic/algorithms/<algorithm>/`.
- Put new one-off analysis, diagnosis, and experiment operator CLIs in `scripts/`.
  Run CLIs that import project modules from the repository root with `python -m`.
- Keep generated and frozen evidence in the existing `experiments/`, `runs/`,
  `evaluations/`, and `submissions/` directories; do not consolidate their paths.
- Do not create new research Python files at the repository root. Root exceptions
  are official/runtime entry points, project-level configuration, compatibility-
  critical files, and existing protected or shared files.
- Preserve teammate-owned subsystems in their existing locations. Files written or
  modified by another teammate respect that teammate's subsystem structure; a
  general repository layout rule never automatically justifies moving them. Check
  Git authorship and protected callers before relocating even an unowned file.

## Research and External-Action Rules

- Preserve the official environment. Do not claim performance from a modified local
  environment is official HAIC performance.
- Treat raw CarRacing reward, progress, damage, smoothness, and custom robustness
  metrics as internal proxies, not official ranking scores.
- Distinguish a one-seed success from a replicated, matched improvement. Do not
  describe unmatched cells as a matched comparison.
- Never relabel consumed confirmation/blind cells as fresh. Do not use reserved
  blind cells for iterative tuning.
- Treat official private tracks as a final generalization target, not an available
  holdout dataset. Do not equate a leaderboard's per-track mixed best with one
  model's generalization.
- Separate observations from causal explanations. Mark untested mechanisms as
  hypotheses.
- A promising local result is an internal candidate, not an official submission.
  Official submission and model confirmation are separate external actions and
  require explicit user authorization immediately before execution.
- Update only the documents whose source-of-truth status changed. New experiments
  update the experiment index, plan, and current state; decisions and model status
  change only when warranted by durable evidence.

## Parallel Agent Orchestration

For substantial work, actively exploit parallel agent execution. Launch focused
background subagents for independent investigation, keep the main agent productive
while they run, and incorporate their findings explicitly. Prefer several concrete
objectives over one broad investigation.

Do not use Agent Manager or separate top-level worktree sessions merely for
delegation when ordinary in-session background agents suffice. Use foreground
delegation only when no useful independent work remains. Do not duplicate delegated
investigations or idle waiting for them.

## Version Control

Use Git continuously to preserve meaningful progress. Create coherent commits for
substantial implementations, fixes, experiment setup, documentation changes, and
other validated units of work; do not combine unrelated changes.

Before committing, inspect the working tree and diff. Do not include unrelated user
changes, temporary files, secrets, credentials, generated artifacts, model files,
or training outputs. Commit messages contain a single-line head in this form:

`<tag>: <sentence>`

Push completed commits to the current branch's existing upstream regularly. Do not
force-push, rewrite published history, change remotes, or modify branch structure
unless the task specifically requires it. Diagnose a failed commit or push rather
than bypassing safeguards.
