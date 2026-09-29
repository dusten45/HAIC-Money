# Fresh TRAIN common-prefix fast continuation

Control is frozen exact visual mode-switch ZIP SHA-256 `b0dc7a24e6ef994cee6e663ec6605f5aa088dbb9b3ce4510f90ccd16dc9d7fc4`. Candidate uses identical pixel controller actions for first 50 decisions, then executes its already-computed fast actor action for the rest of each episode. No simulator state or map/seed identity enters agent actions. Source remains frozen during the run.

Register tracks 1-3 x fresh TRAIN seeds 5052-5055, 12 paired cells and 24 episodes. No prior run manifest in main v2 roots used these seed identities. Compare completion rate first. If completion ties, require candidate median finished lap at least 2% faster on these matched cells; also report jointly finished paired lap differences, mean incomplete progress, P90 finished lap, collisions, damage, action p50/p95/max latency and invalid actions. Exact first-50 action equality and at least 50 later action changes are activation conditions. Candidate must have zero invalid actions. If these fail, reject; if fresh TRAIN passes, register new TUNE comparison separately. No threshold sweep or external action. Linux CPU, max 2,000 decisions/episode, timeout 1,200 seconds.

Infrastructure revision: exact checkpoint files are now present in the frozen snapshot and registered by SHA-256. The prior run stopped before any episode.
