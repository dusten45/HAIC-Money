# Prospective design and decision rules

Protocol v1, written before new candidate simulation results. The user approved
creating this lane from current main because the originally named files were
absent. The criteria below are new prospective rules, not recovered historical
criteria.

## Objective and boundaries

One immutable pixel-only agent must genuinely finish all four required cells
in 10,000–13,000 ms each. Additional tracks should have high completion and
similar speed. Actual official `finish_time_s - post_warmup_start_time` determines
lap time, never action count, qualification, progress, reward, or wall time.
Failures and times outside either end of the interval fail the literal objective.
Retain original objective and fallback outcomes as separate fields.

Preserve root agents and official simulator. All new lane runtime and experiment
tools live here. Agent input is only its copied observation; privileged simulator
telemetry may be recorded by the evaluator for diagnosis but never fed to policy.
No track ID/seed dispatch, memorized action tapes, environment mutation, alternate
finish logic, or per-track best-policy selection.

## Opposite approach

Main root: trained CNN direct actions / checkpoint-dependent learned planning.
Main recommended release: chained road-center/obstacle reaction rules.

Primary independent probes:

1. Whole visible free-space mask → smooth path search with obstacle clearance →
   pure-pursuit tracking and curvature/braking speed schedule.
2. Whole visible free-space distance field → joint steering/speed trajectory
   rollout and collision/progress optimization.

These are explicit online geometry/motion optimization methods, without training
weights or inheritance from the existing reactive controller. The simpler
temporal-feedback alternative is reserved for a documented architectural change,
not silently mixed into selected per-track behavior.

## Experimental sequence

1. Reproduce existing full tests and root required-cell episodes; separately
   rerun the recommended frozen release as a relevant comparator.
2. Test new image/action behavior synthetically. Run each meaningful candidate
   change on required cells, recording failures and source/configuration hashes.
3. Use only required and declared development cells for iterative work. Once
   all four finish, prioritize speed within the original target, with robustness
   on all declared development cells preventing four-seed specialization.
4. Freeze exactly one candidate's source and configuration, commit and push its
   freeze manifest before generating or opening this lane's holdout cells.
5. Run all required and development cells with the frozen candidate and repeat
   the required four unchanged. Open the holdout once, without further tuning.
   A post-holdout runtime change invalidates that candidate's holdout claim and
   requires a new independently approved protocol; no recycling cells.

## Predefined failure response and gates

- An architectural round is a distinct controller family or structural revision,
  not every numeric configuration probe. Keep a visible ledger of all attempts.
- After three architectural rounds fail the literal four-cell objective, or at
  02:45 KST, whichever happens first, activate completion-first fallback.
- Stop new architectures at fallback. Select the existing family using
  lexicographic gates: required finish count, development finish count, combined
  median genuine finish time, lower damage, then lower worst inference latency.
  Explicitly disclose selection on consumed development data.
- Fallback qualification: required 4/4 finish, development at least 11/12 finish,
  all required laps <=25 s, and median of completed development laps <=25 s.
  This is weaker than the original objective and never implies its achievement.
- Holdout corroboration requires >=11/12 finishes. Report the fraction of all
  holdout episodes in 10–13 s and timing over finished episodes separately;
  never omit DNFs from completion denominators.
- If no candidate meets fallback, keep the best investigated candidate as
  experimental and unadopted. Still freeze for one terminal diagnostic if time
  permits; failed criteria cannot be rewritten after observing results.
- If fresh tests expose unrelated baseline failures, preserve/report them;
  do not modify protected existing subsystems to make the suite green.
- Every substantial validated unit is a separate commit and push. Check source
  preservation and include compact primary receipts, not bulky generated traces.

## Time budget

Deadline is 2026-10-05 04:00 KST (Oct 4 19:00 UTC). At 03:35 stop launching
experiments; by 03:45 stop remaining evaluations with explicit censored receipts;
by 03:55 finish report, commits and pushes. Earlier completion is allowed.

## Holdout isolation

This lane has no inherited holdout file. The manifest specifies 12 new terminal
cells: four track IDs times three independently sampled 32-bit geometry seeds.
Generate them only after candidate freeze, audit exact seeds against repository
allocations/previous exposure without inspecting results, reject collisions,
write the manifest, then run once. Do not inspect other studies' holdout data.
Random allocation is recorded for reproducibility; it is not outcome-selected.
