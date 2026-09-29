# HAIC 2026 Strategy and Experiment Control Report

As of 2026-09-22

## Executive conclusion

The current submission baseline is `ppo_actor_only`. It completed 6 of 8 held-out episodes (0.75 completion rate), with a 19,320 ms median finished lap and 22,420 ms P90 finished lap. It remains the only lane currently backed by a packaged submission archive and a CPU smoke result.

The `ppo_cem` lane is not promoted. In the paired fast-budget held-out comparison it completed 0 of 10 episodes and reached mean progress 0.074530, below PPO-only at 0.077411. The result is evidence against the current planner configuration, not evidence that model-based planning is impossible.

The `vision_corridor_teacher` lane is useful as a training and diagnostic reference. Its strong local driving results must not be reported as the submission actor's performance or packaged into the submission runtime.

## Three-lane comparison contract

| Lane | Role | Evidence required for promotion |
|---|---|---|
| ppo_actor_only | submission candidate | held-out or official strict improvement over SOTA |
| ppo_cem | model-based candidate | same split improvement after planner latency and package checks |
| vision_corridor_teacher | teacher and diagnostic | teacher evidence only; never direct submission runtime |

Every candidate is compared in this order: completion rate, median finished lap time, P90 finished lap time, unfinished progress, collisions and damage, then inference cost. Tune results select experiments; held-out and official results decide SOTA.

## Evidence and provenance

| Evidence | Current observation | Source |
|---|---|---|
| PPO actor held-out | 6/8 complete; mean progress 0.788044; median 19320 ms; P90 22420 ms | `artifacts/haic/final-ppo-actor-selection.json` |
| PPO plus CEM held-out | 0/10 complete; mean progress 0.074530 | `artifacts/haic/task5-eval-fast-fullcap/summary.json` |
| Corridor teacher | 3/3 complete in the recorded benchmark; mean progress 0.996622 | `artifacts/haic/corridor-controller-candidate-v3/benchmark.json` |
| Submission smoke | planner disabled; finite action; about 203 MB RSS | `artifacts/haic/final-ppo-actor-selection.json` |

Custom Track Lab maps are generalization and obstacle diagnostics. They are not substitutes for the official final tracks.

## Literature to experiment mapping

### PPO

PPO supplies the model-free actor baseline. The relevant experiment is a fixed-seed visual actor comparison with identical maps, decision cap, CPU budget, and package smoke. The literature supports the algorithm choice; it does not prove that a particular HAIC checkpoint will generalize.

### DAgGER

DAgGER motivates collecting corrections under the learner-induced observation distribution. In this project, teacher corrections are allowed as train-only warm-start data. They must be evaluated on held-out maps and cannot leak map geometry or future track labels into submission inference.

### MBPO and short model rollouts

MBPO motivates short model-generated rollouts while warning that model error can damage policy quality. The existing PPO+CEM result therefore stays as a controlled negative result. A new planner experiment must measure model error, planner latency, and the same held-out episodes before it can approach SOTA.

### Domain randomization

Domain randomization motivates generating new local maps, obstacle layouts, appearance changes, and perturbations to reduce overfitting. These generated tracks expand train and stress-test coverage; they never replace the official evaluation protocol or authorize reading hidden map state at inference time.

## Next experiment protocol

1. Freeze the current SOTA checkpoint and the train/tune/held-out manifest.
2. Select one hypothesis, one strategy lane, and one primary metric change.
3. Generate a new timestamped artifact directory; never overwrite an old checkpoint.
4. Run train, tune, held-out, official diagnostics, package validation, and CPU smoke in that order.
5. Sync `RESULTS.md`, compare all three lanes, and update `SOTA.md` only if the restriction and strict-improvement gates pass.

The default automation is plan-only. Execution requires an explicit local command, and external submission requires an explicit command plus confirmation.

## References

1. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal Policy Optimization Algorithms." 2017. https://arxiv.org/abs/1707.06347
2. Stephane Ross, Geoffrey Gordon, and Drew Bagnell. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS 2011. https://proceedings.mlr.press/v15/ross11a.html
3. Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. "When to Trust Your Model: Model-Based Policy Optimization." NeurIPS 2019. https://papers.nips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html
4. Josh Tobin, Rachel Fong, Alex Ray, Wojciech Zaremba, and Pieter Abbeel. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." 2017. https://arxiv.org/abs/1703.06907
