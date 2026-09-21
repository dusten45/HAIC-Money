# Project Instructions

## Goal

Develop the strongest possible agent for the 2026 HAIC competition.

The project is in an early stage and its architecture, training methods, tools, model choices, and workflow may change substantially as experimentation progresses.

Do not treat the current implementation or project structure as permanent. Prefer evidence from experiments and official competition sources over assumptions based on the current codebase.

## Project References

Official participant repository:
https://github.com/2026-HAIC/Participants

Competition website:
https://ships-duo-ethical-saver.trycloudflare.com/

When competition rules, environment behavior, submission requirements, schedules, or other project details are uncertain or may have changed, inspect the relevant official sources directly instead of relying on assumptions or stale information.

## Project Context

Read `CONTEXT.md` before substantial work when current project state, recent experiments, active approaches, known issues, or next priorities may matter.

`CONTEXT.md` is a living project-state document and may be updated freely as the project evolves.

Keep `CONTEXT.md` concise and current. Replace stale information instead of accumulating obsolete history.

Do not put temporary implementation details, experiment-specific results, or rapidly changing project state into this file when they belong in `CONTEXT.md` or run artifacts.

## Parallel Agent Orchestration

For any substantial task, actively exploit parallel agent execution.

Do not perform all independent investigations sequentially in the main agent when they can be delegated concurrently.

Prefer background subagents for independent work so that the main agent continues making progress while delegated agents run.

When multiple independent lines of work exist:
- Launch multiple background subagents concurrently.
- Continue useful work in the main agent immediately instead of waiting.
- Assign each background agent a distinct, concrete objective.
- Incorporate their results as they arrive.
- Use additional background agents when new independent investigations become useful.

Do not use Agent Manager or separate top-level worktree sessions merely for delegation. Keep delegated work within the current session unless workspace isolation is specifically necessary.

Do not substitute Agent Manager agents for background task subagents when ordinary in-session parallel delegation is sufficient.

For non-trivial research, debugging, optimization, or design work, parallel delegation should be the default. Do not avoid delegation simply because the main agent could complete the work alone.

For substantial tasks, prefer several focused background task subagents over one broad subagent that attempts to solve the entire problem alone.

Use foreground delegation only when the main agent cannot make useful progress until that specific result is available. Otherwise prefer background execution.

The main agent must remain productive while background agents are running. Do not launch agents and then idle waiting for them if other useful work can proceed.

When background results arrive, explicitly incorporate them into the current reasoning and implementation rather than ignoring or duplicating their work.

## Version Control

Use Git continuously to preserve meaningful progress.

For substantial work, create commits at meaningful checkpoints rather than accumulating a large uncommitted working tree until the end.

Split changes into small, coherent commits by task, responsibility, or purpose. Do not combine unrelated work into a single large commit merely because the changes were made during the same session.

Commit and push after completing a coherent implementation, fix, refactor, experiment setup, documentation change, or other unit of work that has been reasonably validated.

Also consider committing before risky or large changes so that a known-good state is easy to recover.

Do not create commits for every trivial edit. Prefer commits that represent a clear and independently understandable unit of progress.

Before committing, inspect the working tree and diff. Do not include unrelated user changes, temporary files, secrets, credentials, large generated artifacts, model files, training outputs, or other files that should not be version-controlled.

Commit messages must contain only a single-line head with no body.

Use the following commit message format:

`<tag>: <sentence>`

Choose a concise tag that describes the type or role of the change, and write a concise sentence that accurately describes the committed change.

Push completed commits to the current branch's existing upstream remote regularly so that important progress is not stored only on the local machine.

Do not force-push, rewrite published history, change remotes, or modify branch structure unless the task specifically requires it.

If a commit or push fails, diagnose the cause rather than bypassing repository safeguards.
