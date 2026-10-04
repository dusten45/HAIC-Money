# Emitted-control guard V1

Frozen standalone source `fast_control_guard_agent.py` SHA256
`13172af1ea3ebd34b39ad3b73bf2ec23755e22bb3614d8e8443093c6dc32bdc0`
embeds exact RearClear parent
`093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc`.
It does not inherit Normal V1's rejected line optimizer.

Normal V1's source-bound173-action T2 diagnosis demonstrated that an optimized
reference path can have3.70m body clearance while the actual long-preview
emitted steering arc has only1.38m. The controller checks the confidence branch's
arc, then can replace that action with an unchecked ridge action.

This candidate keeps the original confidence action when the actual emitted
ridge8m body-corner arc has camera depth<1.9m and the confidence emitted arc has
depth≥1.9m. Supported ridge actions and both-unsupported cases keep the exact
parent action. No new steering search, shorter horizon, force threshold,
circle detector, base-ridge support condition or recovery behavior is introduced.

The new selector copies ClearRidge's action construction and calls
`_ConfidenceReference.act` once. Road/pass-memory updates therefore run exactly
once. Ridge trial steering/target/reference curvature/center/mode remain local
until accepted; only trial spin changes state, and rejection restores the
confidence spin. The public wrapper reproduces ArcGuard reset and postgas clamp,
retaining confidence ArcGuard flags on fallback.

Saved Normal149/154 camera-derived prestates were evaluated counterfactually on
the RearClear base: parent ridge depths1.482/1.014 versus confidence1.924/2.118m.
Both failed exact confidence-fallback assertions before implementation. After
implementation five new tests pass, and21 focused pytest tests pass. Tests also
retain exact valid148/both-unsupported150/155 controls, count one missing-pass
transport including cooldown, verify rejected-trial spin restoration and reset/
nonfinite recovery. These counterfactual tests are not a fresh driving result.

The fixed8m arc assumes constant emitted steer and samples body corners. Camera
sampling does not certify steering lag, tire slip, actual tracking, offscreen
hazards or dynamic feasibility. Privileged pose/track labels enter the earlier
diagnostic reports only; the standalone source uses camera/NumPy alone.

One fresh four-mandatory benchmark used workers2/maxsteps700 with this source
frozen:

| Track /seed | RearClear seconds | ControlGuard seconds | Contacts |
|---|---:|---:|---:|
|1 /516237|15.54|15.92|0|
|2 /644062|18.72|21.52|1|
|3 /1007|16.56|16.78|0|
|4 /18800|18.14|17.34|2|

All four actually finished, but the original10–13s target remains unmet.
Aggregate time increases68.96→71.56s and contacts0→3. T4 improves0.80s while
adding two contacts. The candidate is a negative research checkpoint and is
not adopted. Meaningful camera regressions passing does not establish improved
driving or collision safety. No grid or holdout opened.

Receipt `control-guard-v1.json`, frozen cells/freeze, source lineage and zero-
context parent patch preserve this exact failed candidate. Trace-only first
divergence/contact localization is recorded separately; no additional driving
benchmark or candidate change followed these results.
