# Select and Confirm Model Workflow

Model selection, official upload, and competition-site confirmation are different
actions. This workflow prevents a mixed leaderboard from being mistaken for one
model's final-evaluation ability.

## Select One Candidate

1. Read the current confirmed model and candidate ledger in
   `docs/results/MODEL_STATUS.md` and `docs/competition/submissions.md`.
2. Compare the same immutable model/checkpoint/package across every currently
   available appropriate condition. Do not combine per-track best results from
   different submissions.
3. Check local generalization evidence, accessible official public-track results,
   completion/progress/lap-time semantics, package/runtime reliability, and private-
   track generalization risk.
4. Confirm provenance: source commit, run, checkpoint, package hash, model hash,
   evaluation scope, and result artifacts.
5. Record the selected model and evidence before changing competition state.

## External Gate

1. Refresh the official website and Participants README.
2. Verify that the candidate has passed the preparation workflow and that the
   official submission ledger is current.
3. Ask for explicit user authorization immediately before uploading or confirming a
   model, unless the user explicitly delegated that exact external action.
4. After the site action, record the official identifier, server validation status,
   public results, confirmation state, and any failure reason in the local ledger.

Never estimate official points from local proxies or assume that a public-track win
guarantees private-track performance.
