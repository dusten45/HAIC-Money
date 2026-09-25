# RLPD Long-Horizon Follow-up Frozen

- Message ID: `20260924T202100Z-r8q2-rlpd-followup-frozen`
- Type: result
- Author/session: `r8q2`
- Written: 2026-09-24T20:21:00Z
- Reply to: `20260924T195815Z-r8q2-rlpd-pilot-v2-result`
- Evidence: measured v2 stop gate; verified source/runtime/tests and independent seed audit
- Status: frozen before follow-up environment interaction

V2 is closed at `fail_stop_hold`; no v2 data, student checkpoint, screen, or reserved
v2 confirmation/blind/full cell enters the next run. In response to the owner's
end-to-end iteration request, I froze a separate hypothesis/protocol:
`experiments/pixel-rlpd-long-horizon-followup-v1.json`, SHA-256
`2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329`. The working
hypothesis is that a short v2 horizon and small prior cap may have limited stable
screen finishes; expanding both tests a fuller recipe, not a causal diagnosis.

The new prior corpus is capped at 16,384 decisions with four distinct-geometry
finishes; the matched SAC/RLPD students start from scratch for 131,072 decisions
each at learner seeds 10 and 11, with 65,536/131,072 screen checkpoints. New
train data uses seeds `4000012001-2064`, screen `4000013001-3008`, confirmation
`4000013011-3018`, blind `4000013021-3028`, with new disjoint track-ID groups.
An independent artifact/ledger audit found zero exact seed-token matches among the
88 candidate values; both DrQ source ledgers have zero intersections. All v1/v2
allocations, including unopened v2 reserves, are excluded. This is no-known-use,
not a global historical guarantee.

Before launch, the full 105-test learner/Agent/evaluator/package suite passed in the
main runtime; the same 105 unittest gates passed under Python 3.11/Torch 2.1.0 CPU.
The custom evaluation protocol loaded 24 screen, 32 confirmation, and 24 blind
canonical cells; its runtime/source checks pass. The protocol locks 41 source files
and the post-import dependency inventory. No follow-up environment interaction has
started. Confirmation/blind are conditional and will remain closed unless preceding
predeclared gates pass. No official score, submission, or model-confirmation action
is authorized or claimed.
