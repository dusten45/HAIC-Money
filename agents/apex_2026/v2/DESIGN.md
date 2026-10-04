# Structural research cycle v2

The user's renewed request after the v1 report explicitly authorizes further
creative performance work. This is a separate cycle, not post-holdout tuning of
the v1 frozen artifact. Preserve v1 source, freeze, results and failed conclusions.
The24 previously exposed development/holdout geometries are consumed regression
cells; they can inform this cycle but can never become fresh holdout again.
No other study's reserved/blind cells may be inspected or used.

## Intended improvement and architectures

The original four required10–13s target remains unchanged. Seek materially better
completion and speed by fixing model/representation limits, not outcome-specific
track dispatch or an arbitrary gain sweep. Only4x84x84 public pixels and action
memory are runtime inputs. Official physics/scoring/root policy remain unchanged.

Independent structural axes:

1. Full two-dimensional clearance-aware geodesic path, arbitrary orientation,
   metric arclength curvature and braking envelope. Remove the row-monotone path
   restriction that cannot express a turn back down the image.
2. Coupled predictive control with measured instantaneous HUD state and calibrated
   longitudinal/tire response. Validate open-loop residuals before relying on
   long-horizon trajectory scores.
3. Braking-feasibility safety barrier based on visible drivable footprint and
   predicted vehicle motion. Distinguish comfort deceleration from emergency
   braking when newly visible geometry makes the former infeasible.

A shared diagnosis examines the two consumed v1 failures and successful traces.
Unknown slip is a measurement limitation, not proof that tire saturation caused
an episode. Timestamp-aligned kinematics must precede causal claims. Root may
investigate observation-only motion registration, with independent tests before
candidate integration. Each change preserves exact source/configuration and
all outcomes; tests precede implementation when feasible.

## Evaluation and selection

Use unchanged agents/apex_2026/evaluate.py in fresh processes. Count all failures,
use actual finish timestamps, enforce resource limits. Existing receipts are
historical comparators; candidates must execute new matched episodes. Record
repeatability separately from unique geometries. Validate source before/after.

Required4 and consumed regression24 are defined in benchmark.json before v2
resets. A matched improvement must finish4/4 and24/24, with required mean below
v1's18.53s. Candidate selection first preserves completion, then reduces actual
lap time/damage. This is weaker than original10–13s and is labeled separately.
No fixed time/experiment count forces a premature result. Stop a hypothesis when
its mechanism is falsified, retain negatives, and revisit only with new evidence.

After selecting exactly one candidate, bake defaults, commit/push source+config
freeze, execute all28 cells anew and repeat required4. Only then generate24 new
random geometry seeds (six pertrack), reject exact exposure/allocation collisions,
commit allocation, and evaluate once. Corroboration requires23/24 finishes; show
all-lap10–13s count, finished-time distribution and DNFs separately. No subsequent
policy changes or re-selection using holdout. If gates fail, report failure and
keep experimental/unadopted. No official submission or confirmation.

## Ownership

main_strategy: geodesic_agent/tests/ledger. rollout_agent: predictive_agent/tests/
ledger. official_rules: shield_agent/tests/ledger. evaluation_harness: read-only
failure diagnostic and later matched suites. Root: protocol, source archives,
motion-observation experiments, selection and final evidence. Parallel runs use
at most one worker per controller during exploration to fit available CPUs.
