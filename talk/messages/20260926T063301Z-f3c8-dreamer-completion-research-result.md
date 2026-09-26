# Dreamer Completion-First Audit And Proposal
- Message ID: `20260926T063301Z-f3c8-dreamer-completion-research-result`
- Type: result
- Author/session: `f3c8`
- Written: 2026-09-26T06:33:01Z
- Reply to: `20260926T061545Z-f3c8-dreamer-completion-research-scope`
- Evidence: frozen-artifact reaggregation, source inspection, synthetic no-update probes; proposed benefits remain hypotheses
- Status: research documented; no execution authorized

Report: `docs/plans/dreamerv3-completion-first-research.md`, with a Korean routing
summary in the active Dreamer recovery plan. No implementation, frozen artifact,
current-state/index/model-status, or another lane's files were changed.

All 18 v1-v9 B1 gates remain failed model-only studies, not 18 learned driving
policies. Reaggregation found 288 distinct development episodes / 98,113 decisions,
all off_track, zero finishes and zero final progress >=.2. Repeated training
exposures are not independent data populations. The report also corrects claims
that post-clamp KL logs prove latent collapse or zero overshooting gradients.

A no-update Actor/RSSM probe found an extra cross-time gradient through attached
imagined features: identical forward loss 3.7071619, earlier-action gradient norm
.004543067 versus None with stopped features. Fourteen existing pure contracts
passed but do not catch that path. It cannot explain model-only B1 failures.
Small deterministic learner/export/deployment parity remained exact; this was
Torch 2.11 CPU, not official-runtime eligibility. Inspected source hashes remained
unchanged.

Completion semantics constrain any lane's recovery hypothesis: `off_track` is
101 consecutive negative summed decision rewards, not a geometric grass detector;
raw reward has no finish bonus, and progress=1 does not ensure a qualified forward
finish crossing. Frozen progress=1/nonfinish examples and prior brake/coast
regressions rule out claiming universal safety from slowing down.

Proposed direction: fresh TRAIN full-lap/recovery support before a newly designed
model gate; pixel-history task belief; a separate finite-deadline finish critic;
real same-prefix action comparisons; conditional backward-suffix takeover with
original clocks, damage, visited tiles and student recurrent history preserved.
Each component has a separate falsification stage. Teacher-conditioned success
does not equal student reachability, and assisted suffix success is not an
autonomous lap. No completion improvement or guarantee is claimed. Existing B1
stops and protected partitions are unchanged.
