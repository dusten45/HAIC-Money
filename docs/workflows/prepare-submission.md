# Prepare Official Submission Workflow

This workflow prepares a package; it does not send it to the competition site.
Actual submission requires explicit user approval immediately before the external
action.

## Refresh Official Sources

Before packaging, reopen the current official Participants README and competition
website. Verify the provenance recorded in `docs/competition/`, then resolve any
conflict in favor of the official source before continuing.

## Required Gate

1. Identify one immutable candidate: source commit, run/checkpoint, model hash, and
   intended inference path.
2. Verify the official environment version and Python 3.11/Linux/CPU compatibility.
3. Use the relevant existing packager and tests (`package_submission.py` or
   `training/package_submission.py`) rather than writing a parallel validator.
4. Check import/init, reset, act, memory, observation/action, relative paths, and
   repeatability requirements.
5. Check ZIP root structure, required runtime files, dependency pins and installation
   feasibility, forbidden imports/functions/native files, archive/file-count/size/
   compression limits, and exclusion of trainers, caches, `.venv`, replay, and
   unused checkpoints.
6. Run a clean isolated smoke and record the final ZIP and model hashes.
7. Add the local package provenance to `docs/competition/submissions.md`.

Read exact current limits from
[`docs/competition/restrictions.md`](../competition/restrictions.md), not this
workflow. Server-side validation failure can consume an official submission quota,
so local validation must be complete before any approved external upload.
