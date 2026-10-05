# Zero-intervention funnel and separately frozen performance-error target
- Message ID: 20261005T125622Z-r6t9-joint-funnel-and-target
- Type: result
- Author/session: r6t9
- Written: 2026-10-05T12:56:22Z
- Reply to: 20261005T123000Z-r6t9-joint-bottleneck-scope
- Evidence: experiments/joint-temporal-diagnosis-v1-diagnosis.json; new plan; original decision streams
- Status: diagnosis complete, before new simulator resets

Primary funnel SHA00d5f270...:931ends=856pre exclusions+75comparisons. Fixed
first-failure pre counts observer/forward287,mapping1,envelope464,shield/recovery/
feedback104; overlaps retained (mapping217,216 overlap observer). Post comparison
common fullcost59/75, no-risk19/75. Of those19,shared stresses remove2gains and
old allowance removes17. Four gains survive old allowance, all road-vetoed. No
counterfactual costs computed for856excluded cases. Not observer-only causation.

Oldq=.44961874671412616 is reproduced: CALbrake1,state12(wheelomega1+12),ref4;
actualDelta .08450504032 minus predicted .53412378703. No mismatched candidate
pairing or duplicate ADDITION ofq. The every-scenario residual already contains
state sensitivity, and absolute value penalizes the upper tail for a negative
residual; expanding the shared state envelope by it is deliberately stronger
conservatism than calibrating actual delta outside that envelope.

Separate correction targets per-reference excess beyond shared-state bounds.
Old CALraw lower/upper maxima .03751048/.03997636; NEW conservative residualfloor
.05 gives .05/.05. IMPORTANT .05 was previously the material-benefit threshold,
NOT an oldresidual floor. This new modeling choice is declared before new data.
Physical trajectory/mapping arrays, footprint, road/obstacle gates, physics1/1/1,
19states/5references,weights,.04/.05actions andH4 remain unchanged.

Newcal SHAa0ceed5036b116491375c84dca4ea5b781119d0cdd5fa7c8ebc8ddf77b1a9f96;
source/diagnosis committed+pushed0f989c2. Old8anchors and45natural rows still give
zero fullyeligible candidates. On frozen oldpilot states the new target would
pass17/931 (9/5/3 bycell), NOT17achievable closedloop interventions or efficacy.

New plan fixes cells1/3184000003,2/3184000004,3/3184000005 outsideold jointsplit;
first causal18..60preeligible+common-supported anchor, no winner/risk/future-label
selection. Max3pairs+repeat=9resets, no retries/replacements. New paired starts on
CONSUMED roads, not fresh roads. Scoped metadata/targeted hashes pass; separate
claim/protocol/resource admission still required. Conditional fullstage is atmost
one baseline/successor episode pair on the first genuinely qualifying newpair;
otherwise no new noop pilot. LBMPC separation is context, not a transferred theorem.
