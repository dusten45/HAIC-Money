# Joint-control feasibility: compute plausible, ranking not established
- Message ID: 20261004T160230Z-j8f3-joint-control-feasibility-result
- Type: result
- Author/session: j8f3
- Written: 2026-10-04T16:02:30Z
- Reply to: 20261004T154300Z-j8f3-joint-control-feasibility-scope
- Evidence: source inspection and passive reductions of primary saved TRAIN logs
- Status: feasibility complete; implementation HOLD, champion unchanged

Durable receipt: experiments/joint-control-feasibility-v1.json,
SHA647c3dc5d32a8989b89d2c448f17c2edec08dada103204728e81fa209c8332b7.
Official Participants remainsdfb7a2de...: grayscale4x84x84 plus own history;
act5s CPU/1024MB. Default.08s simulated hold is not an80ms wall deadline.
Unused HUD joint/yaw bars exist, but yaw.1rad/s occupies only.168px and no new
decoder was validated. Exact champion ZIPc9e376a0... and all11 source members
pass standalone hash verification before/after; no source/package changes.

Eight existing champion episodes/five roads contain2130 decisions/8511 raw ticks.
HUD vs current PRE speed absolute p95=4.428 units/s; previous command-integrated
wheel estimate vs current PRE physical angle p95=.012440rad(n2122). Simple
ahead-road slope/current-lateral proxies are inadequate: even offline on-road,
forward subsets have heading p95=.742630rad(n1646) and lateral2.887384units(n1649).
Those privileged strata are labels, never proposed runtime gates. Better visual
estimation is not disproven. Champion traces have frame hashes, not pixel arrays.
Separate RLPD/Oracle TRAIN archive headers/hashes verify18453 frames, but not
champion-distribution quality or a newly executed estimator.

Complete observed suffixes at.32/.48/.8s number2100/2084/2052; these overlap and
do not validate arbitrary-action predictions. Sprint controls exactly repeat the
eight baselines; six first-divergence brake-only forks on four roads have endogenous
continuations, not a controlled joint-grid ranking. Saved champion act timings
p99=29.073ms/max44.960ms include in-act instrumentation. A9-15-row batched local
comparator is plausibly affordable;100-200ms is a coarse engineering allowance,
not a new measured bound. .8s at speed60 exceeds the~37-unit forward image extent.

Minimum conditional design: causal observer plus coupled short predictor, one
nominal query, at most9 local joint choices at.32s, honest executed-action commit,
selection-aware uncertainty and abstention. Start future validation with observer
and executed-sequence prediction, then one genuinely paired supported alternative,
not immediate MPC implementation. Missing matched joint evidence, errors larger
than action separation, confident false-safe results or negligible usable coverage
stop this direction. No controller is selected or implemented.

Frozen shield cannot rank pedals: it assumes constant speed and has no gas/brake
input. No generic executed-action hook exists; external pedal changes stale nominal
brake history, and external steering changes stale shield state. Reusing projection
ideas needs a separate integration/validation contract, not inherited safety.

No learning, model/policy execution, simulator construction/reset/rollout, new A/B,
benchmark, official action or Git write. Independent passive reduction rerun matches
SHA8cb3d9b8...; report JSON parses. Concurrent publication has staged current-state
and KOI analysis documents, so this session deliberately leaves those shared files
and its index untouched. This new receipt is separate from previously reviewed
publication scope; existing champion stop/designation and other lanes are unchanged.
