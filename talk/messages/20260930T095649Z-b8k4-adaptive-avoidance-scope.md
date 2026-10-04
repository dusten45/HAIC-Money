# Separate adaptive obstacle-speed candidate over the frozen KOI baseline
- Message ID: `20260930T095649Z-b8k4-adaptive-avoidance-scope`
- Type: coordination
- Author/session: `b8k4`
- Written: 2026-09-30T09:56:49Z
- Reply to: `20260930T075853Z-b8k4-koi-baseline-analysis-result`
- Evidence: user-directed implementation and frozen source inspection
- Status: implementation, not measured driving improvement

User now requests a separate improved version addressing excessive fixed-speed
braking during obstacle avoidance. The active baseline clamps near-obstacle
preview to44 and uses44 again in arrival braking. Changing only one is ineffective.
Baseline steering/crossing behavior and frozen sources must remain unchanged.

Planned new files: `haic/algorithms/koi/adaptive_avoidance.py`, isolated packaging
CLI in `scripts/`, and targeted tests. Package baseline dependencies from immutable
Git blobs at `c4e224d`, verify release manifests/contact source hashes, and add a
separate adaptive governor; never patch the archived/runtime/root agent. Existing
baseline analysis document may receive an implementation-status appendix.

Hypothesis: where current near-object passing side has sufficient connected
asphalt clearance and moderate steering demand, use a continuous pass-speed
budget plus distance-to-arrival envelope instead of unconditional44 braking.
Keep curve budget, conservative fallback on uncertainty/impact/large steer,
launch prefix and no-obstacle actions. Write actual final brake back into baseline
history. This is a candidate, not a safety/faster-driving claim.

No new environment/reset, training, consumed held-out tuning, confirmation/blind,
official submission or model confirmation is planned in this implementation turn.
Other RLPD lane's claimed scope is independent; shared status/plans are untouched.
