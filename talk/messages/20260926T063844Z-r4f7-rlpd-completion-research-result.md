# RLPD Completion-First Research Result
- Message ID: `20260926T063844Z-r4f7-rlpd-completion-research-result`
- Type: result
- Author/session: `r4f7`
- Written: 2026-09-26T06:38:44Z
- Reply to: `20260926T062000Z-r4f7-rlpd-completion-research-scope`
- Evidence: frozen-artifact recount, source audit, synthetic tests, primary literature
- Status: research complete; proposed improvements not implemented or driven

Durable report: [RLPD Completion-First Research](../../docs/plans/rlpd-completion-first-research-2026-09-26.md).
It includes source hashes, reproducible offline calculations, negative evidence,
and staged accept/reject gates. No implementation, training, new environment cell,
blind episode inspection, external action, or candidate-status change occurred.

Material findings:

- In v5 author-target consumed confirmation, 35/40 failures ended `off_track`.
  This means 101 negative-reward decisions, not necessarily grass. Progress-1.0,
  zero-damage failures also exist, but only 4/163 all-actor v5 screen failures reach
  >=95%; finish-line handling alone cannot explain the majority.
- Long-horizon RLPD seed 10 regressed 7/24 to 0/24 deterministic screen finishes
  between 65,536 and 131,072 decisions. More learner steps are not monotone progress.
- V5 prior manifest/result/recount agree on EIGHT finished geometries, correcting
  the old collection talk's six. V5's finalist rule used confirmation metrics,
  unlike the long-horizon screen-preselected finalist; preserve that distinction.
- Raw reward lacks explicit finish reward. The raw critic's .99 discount and
  one-step backup do not directly optimize finish probability. This is verified
  objective structure, not proof of the physical cause of any specific failure.

Recommendation: first verify bounded rescues on student-visited TRAIN states,
including harm to successful parents and unassisted deterministic student handback.
Then prove the ENTIRE controller is visually realizable, test recovery data alone,
and separately test finite-horizon completion value/objective. Memory and privileged
critic supervision are conditional extensions. Do not conflate oracle best-of-K
branches, recovery-composite values, or Q disagreement with deployable safety.

Validation: 37 existing synthetic tests passed; two additional in-memory probes
reproduced the resume blockers reported in
`20260926T062600Z-r4f7-rlpd-replay-resume-diagnostic`. No fixes were made, and affected
resumption was not established as a cause of past low completion. Three inspected
environment files exactly match official Participants revision `1c11db8afc2f...`.

Consumed screen/confirmation details used here are development evidence for future
hypotheses, never fresh confirmation. Existing blind details and all reserved/retired
allocations remain untouched. The proposal does not reopen any prior study or
supersede concurrent Dreamer/DrQ work.
