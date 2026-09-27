# DreamerV3 Revival Plan

**Status: DEFERRED / LAST-RESORT ONLY.** This is not current work, a TODO, or an
authorization to resume DreamerV3. The research line was closed on 2026-09-27
because the current design and budget did not justify more work while long-horizon
prior dynamics remained unresolved. This is not a claim that DreamerV3 is
theoretically impossible.

## Reopen Only If

All conditions must hold:

- Late in the competition, RLPD, TD-MPC2, and the other serious algorithm options
  do not show sufficient performance potential.
- The owner accepts that reopening means a new, design-level research line, not
  more local tuning of the closed implementation.
- The old B1 and H8 studies are not repeated as-is; their consumed data and failed
  gates remain historical and cannot be relabeled fresh or passed retroactively.

Even then, this document is only a contingency outline. A separate explicit
decision, source/provenance audit, and newly frozen protocol must precede any
research action. No confirmation/blind partition, model promotion, or official
submission is authorized here.

## First Design Questions

- Reconsider the `4 x 84 x 84` stacked observation as the world-model input. Prefer
  investigating a newest-frame visual encoder with temporal information carried by
  RSSM memory, rather than feeding the full stack unchanged into both systems.
- Replace teacher-versus-random extremes with action-coverage data containing
  varied steering, gas, and brake outcomes from the same or closely matched
  states. Preserve the executed action and resulting transition so action-conditioned
  counterfactual consistency can be evaluated.
- Permit a small, explicitly bounded online co-training pilot: environment
  interaction -> replay -> world-model update -> imagination actor/critic update
  -> new interaction. Do not assume offline world-model completion must precede
  opening the actor, but gate each stage for stability and safety.
- Establish a new baseline for model capacity, latent size, sequence length,
  imagination horizon, replay ratio, batch composition, and update-to-data ratio.
  Do not extrapolate the old settings with one-factor weight, horizon, or update
  count sweeps.
- Center evaluation on newest-frame prediction, multi-step latent/prior stability,
  reward and termination prediction, and action-conditioned counterfactual
  consistency. Full four-frame reconstruction MSE alone is not a decision gate.
- Preserve all consumed TRAIN cells and existing fresh-evaluation boundaries.
  Resolve seed provenance and the candidate-specific audit before selecting or
  claiming any new research cell.

## Guardrail

The prior failures do not by themselves prove any listed redesign will work; each
is a hypothesis for a separately designed study. Do not reopen this line unless
the conditions above are met and the new study is explicitly authorized.
