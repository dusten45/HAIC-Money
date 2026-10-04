# Independent Apex 2026 agent

Build a standalone camera-only agent without changing the active root agent or
official physics. The requested target is a completed lap in 10–13 simulated
seconds on (1,516237), (2,644062), (3,1007), and (4,18800), with similar behavior
on additional geometries. Winning the competition remains an aspiration, not a
claim supported by a local benchmark.

The controller reconstructs road coordinates using the official affine camera,
estimates speed from the visible HUD, and uses speed-dependent preview steering,
curvature braking, obstacle corridors, and bounded recovery. Its inference input
is exclusively four 84x84 grayscale frames. It contains no geometry seed lookup,
environment access, root-agent dependency, or required neural checkpoint.

Alternatives considered: accelerating the existing controller retains known
obstacle robustness but inherits its restrictive speed and steering envelopes;
training a new visual policy needs substantially more compute and does not yet
have convincing local evidence. The metric camera controller provides a new,
small, measurable starting point; evidence determines which candidate is kept.

## Evaluation contract

- Use untouched official environment, frame skip 4, warmup 50, and actual forward
  finish crossing at at least 95% progress. Subtract warmup from official finish
  physics time, never use wall time or progress alone as lap time.
- Always report all four mandatory pairs, including DNFs and contacts.
- Development uses four additional geometry seeds across all four track IDs.
  A separate four-seed holdout is opened only after freezing source/parameters.
- Strict pace limit is 13 seconds. After three consecutive development rejects,
  a new prospective profile permits 15 seconds; after another three, 18 seconds.
  A profile change never edits past outcomes or claims the original target met.
- A candidate passes a profile only with four mandatory finishes within its pace
  limit and at least 90% additional finishes within that limit, with at least
  75% on every track ID. Inference bounds,
  finite actions, CPU limits, and honest completion rules are never relaxed.
- Source, parameters, environment hashes, and action traces bind every receipt.
  Exact cold repeats and an extracted ZIP run verify the final artifact.

## Execution plan

1. Independently verify official rules, baseline four targets, and camera/physics.
2. Test evaluator failures (DNF, duplicate cells, wrong action, missing coverage)
   before adding the evaluator and prospective relaxation logic.
3. Test synthetic road/obstacle/reset behavior before implementing the fresh
   controller. Run bounded parameter studies on mandatory/development data.
4. Freeze selected source/parameters, run additional held-out seeds, cold repeats,
   runtime/import checks, and extracted-package driving; review independently.
5. Commit/push coherent validated work on the existing branch. Preserve all
   unrelated user files. Record measured results and remaining target gaps.

The user explicitly requested implementation and flexible validation; these
instructions authorize proceeding in this session without additional design
approval. Folder isolation satisfies the requested separation and preserves the
existing branch structure.
