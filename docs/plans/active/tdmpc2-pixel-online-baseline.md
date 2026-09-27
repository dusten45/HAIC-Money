# TD-MPC2 Single-Task Pixel Baseline (Independent Lane)

## Instance-Migration Stop (2026-09-27 05:10 UTC)

The user stopped new experiments for Vast.ai instance migration. Do **not** run
the v2 protocol on this instance; it has zero environment resets, no checkpoint,
and no learner/evaluation result. The only TD-MPC2 TRAIN interaction is the
v1 partial attempt: 10,061 decisions, 10,000 pretraining plus 60 later
updates at its last valid boundary in
`runs/tdmpc2-reused-train-20260927-v1/boundary.pt` (SHA-256
`d3502e430a0c4bb3be9a993ab16434d6b5400afdc201177080f8cc4d035d8b4c`).
Its journal SHA-256 is `ff909545fe87fac34e649e016586b8babc22ab296aa33ad935ca4a1e13871961`;
the final `reset_intent` and `partial` after that boundary mean **exact resume
is prohibited**. Transfer the checkpoint and journal together for provenance,
not as a deployable or completed policy. No complete planned episode, paired
prior/MPPI result, confirmation, blind, or official action occurred.

After a new clone receives both committed code/protocol and separate Git-external
artifacts, inspect the migration handoff in `docs/context/`, verify pinned hashes
and resources, and obtain a new user instruction before opening any TRAIN cell.
The only predeclared follow-up is a **new from-scratch** v2 attempt using the
frozen protocol and consumed TRAIN roads, never continuation of v1. None of
the ordered research gates below authorize action during migration.

## Hypothesis And Sources

TD-MPC2's learned latent dynamics and short-horizon planner may improve HAIC
driving over its own stochastic policy prior, but this is an untested hypothesis.
Start with the paper [TD-MPC2](https://arxiv.org/abs/2310.16828) and the
[official implementation](https://github.com/nicklashansen/tdmpc2) as primary
sources; pin the inspected revision before interpreting a run. Do not import
DrQ-v2/PPO training logic or introduce CarRacing-specific model, loss, reward,
action, or planner improvements in this first implementation. Adapt only the
official observation/action/episode boundary contracts to HAIC.

Pinned sources: TD-MPC2 `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`
and official Participants `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`.
Participants delivers four oldest-to-newest grayscale `float32` 84px images at
decision boundaries after 50 raw no-op reset ticks and up to four physics ticks
per decision; upstream TD-MPC2 instead expects three RGB 64px frames as 9
`uint8` channels. The TD-MPC2-only adapter retains all four grayscale planes,
resizes 84->64, quantizes to `uint8`, and configures the upstream four-conv pixel
encoder for 4 input channels (512 latent unchanged). This is an irreducible
observation adaptation, not literal upstream pixel parity. Symmetric `[-1,1]^3`
actions map to HAIC `[s,(g+1)/2,(b+1)/2]` without pedal exclusivity; symmetric
zero is half gas and half brake. The wrapper sums raw unshaped reward. Either
done flag resets planner/episode; a finish has HAIC `truncated=True` and
`info.finished=True` but is a semantic terminal with no bootstrap, unlike a
plain horizon truncation.

Official inference is CPU-only, <=1,024 MB, <=10s import/init, <=5s per reset
and per act. MPPI is not categorically prohibited; actual cold/warm planner
latency and process RSS must be measured before claiming deployability.
An [untrained-shape CPU preflight](../../../experiments/tdmpc2-cpu-preflight-v1.json)
measured 25 full MPPI actions under Torch 2.1.0+cpu after repairing a 2.1-only
tuple-axis reduction incompatibility. It cannot substitute for trained-checkpoint
reload, whole-episode latency, package validation, or an official run.
An independent [model-only synthetic export check](../../../experiments/tdmpc2-cpu-export-preflight-v1.json)
also loaded 2.11-produced weights in two fresh CPU-only 2.1 processes and
reproduced four-action seeded MPPI traces; it is still not a trained checkpoint.

For the official-repository 5M **model size** (not five million environment
steps), select `model_size=5`: 512 latent/MLP, five Q heads, 101 symlog bins,
batch 256, horizon 3, rho .5, random shift pad 3, 512 MPPI candidates (24
policy-prior), six refinements and 64 elites. The current repo's 10M-step run
default is not this pilot's budget. At the HAIC local 2,000-decision episode
limit, upstream's formula yields discount .995 and seed steps 10,000; its loop
performs 10,000 pretraining updates at the seed boundary on already completed
episodes, then one update per subsequent decision. A shorter pilot that omits
or reduces this work must be labeled an explicit computational deviation, not
quietly called a faithful upstream training schedule. The current optional
episodic termination mode must be enabled because HAIC has actual terminals.
The upstream default 9-channel non-episodic model has 4,865,540 trainable
parameters in this implementation; the required 4-channel HAIC input plus
optional termination head produces 5,385,573. Thus "5M" names the upstream
size class, not identical weights/parameter count across environments.

## Upstream Mechanism And Evidence

At [upstream `world_model.py`](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py),
the encoder embeds stacked pixels into 512 SimNorm coordinates. The deterministic
dynamics learns the next latent from current latent/action against a detached
encoder target; the reward model learns symlog two-hot reward, and five Q heads
learn symlog two-hot bootstrapped return. Two randomly chosen Q heads are
decoded: their minimum builds the TD target and their average values actor and
planner actions. An independently optimized tanh-squashed **Gaussian** prior
maximizes scaled detached Q plus weighted sampled entropy; it is neither a
truncated normal nor the action selection rule by itself. The planner draws 24
trajectories from that prior and 488 from a Gaussian, evaluates learned
reward+terminal Q through dynamics for horizon 3, repeatedly re-fits weighted
64 elites, then Gumbel-selects an elite first action. Evaluation suppresses
only its final exploration Gaussian; pixel ShiftAug and elite selection remain
stochastic, so deterministic evaluation requires reset-specific RNG replay.

[Latest upstream `tdmpc2.py`](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py)
targets Q using detached encodings of *actual next observations*, although
[paper Eq. 3](https://arxiv.org/html/2310.16828v2#S3.E3) writes a dynamics-
predicted next latent; this lane follows the newer code. Paper section 3.2
writes a Gaussian sampled action and shifts both its mean and std; latest code
selects an elite first action and shifts only its mean. The paper predates
current opt-in episodic termination. Strict H+1 replay slices sample only
completed episodes and apply per-frame independent pad-3 random shift **inside
the encoder**. This distinction prevents replay-side double shifts.
The operator deliberately updates from episodes completed *before* the current
decision, then admits the current transition/just-ended episode after updates;
the seed-boundary pretraining therefore never sees a newly ended episode from
that same action. The CPU-only development evaluation must load a separately
source-bound model-only export under official-version CPU Torch, not merely move
the CUDA training model to CPU inside the same process.

## Ordered Gates

1. Audit official 5M single-task online/pixel defaults and current episodic
   termination implementation; verify local HAIC pixels/history, reward, action
   mapping, terminated vs truncated, and official inference constraints. Record
   any irreducible differences from upstream before environment interaction.
2. Synthetic and local parity tests must cover environment observation/action
   bounds and inversion; correct episode reset/bootstrap semantics; no sequence
   crossing and matched temporal image augmentation; finite and connected
   encoder/dynamics/reward/Q/policy losses and target EMA; MPPI action bounds and
   prior comparison; exact checkpoint/replay/optimizer/RNG resume and deterministic
   CPU evaluation. Failed tests block any pilot.
3. Freeze a separate small TRAIN-only pilot after a candidate-specific cross-lane
   geometry/track/conditions audit (and shared TRAIN claim if allocating *new*
   cells). Record executable and
   environment hashes, exact reset cells, seed-step, decision/update counts,
   diagnostics, CPU/GPU resources, and stop criteria before first reset. Preserve
   partial attempts as consumed; never substitute cells silently. No confirmation,
   reserved blind, or official submission/evaluation is authorized here.
   No generic new-lane four-cell freshness auditor currently exists; the proposed
   fallback is reused obstacle track-1 r6 TRAIN seeds `3910800001`,
   `3910800004`, `3910800034`, `3910800085`, already reset by Dreamer. These
   are consumed development cells, not fresh claimable holdouts.
   The frozen pilot protocol is
   [`experiments/tdmpc2-reused-train-pilot-v1.json`](../../../experiments/tdmpc2-reused-train-pilot-v1.json),
   SHA-256 `e671cb916ea07c7c902b029393d5728560cfe9b48b04bade3a8c5604d54ef748`;
   its read-only source/resource/cell preflight passed with zero environment
   resets. It caps 14,000 decisions, 14,000 updates, 7,200 training seconds,
   and pairs the prior with MPPI only in 500-decision-censored reused-TRAIN
   episodes after source-bound CPU-only model export.
   The first v1 attempt stopped after 10,061 decisions and 10,000 pretraining
   plus 60 later updates: an inference-mode MPPI warm-start tensor caused
   `planner.reset()` to fail on a subsequent TRAIN road reset. Its
   [partial receipt](../../../experiments/tdmpc2-reused-train-pilot-v1-failure.json)
   and boundary checkpoint are preserved; ledger ends after `reset_intent`, so
   it cannot be presented as a complete pilot or resumed exactly. A separate
   v2 attempt may only open after the replacement-tensor reset regression,
   source re-pin, reused-TRAIN audit, resource preflight, and new frozen path.
   The separate [v2 protocol](../../../experiments/tdmpc2-reused-train-pilot-v2.json)
   (SHA-256 `209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb`)
   and [v1-exposure receipt](../../../experiments/tdmpc2-reused-train-pilot-v2-exposure.json)
   pin that exact reused-cell history. At the 2026-09-27 04:49 UTC recheck,
   full source/resource preflight passed with **zero v2 resets** (raw memory
   ~17.1GiB free), but shared disk had only 5.5GiB free, ~0.5GiB above the
   immutable 5GiB floor. A v2 boundary checkpoint alone is expected to take
   ~358MiB while independent checkpoint writes continue; do not launch into
   that transient margin. Wait for safe sustained headroom; do not lower
   either floor, delete other lanes' artifacts, or describe v2 as run.
4. Assess world-model held-sequence predictions and reward/Q calibration on
   TRAIN-only experience, actor-only vs MPPI on identical TRAIN development
   road/seed cells, and progress/finish/raw-reward proxies with denominators.
   Finite losses alone and within-replay fit do not establish driving improvement.
   Pilot failure should separate verified adapter/loss/replay errors from possible
   algorithm-environment mismatch; neither implies a general TD-MPC2 limit.
5. Only if the pilot is operationally correct and exhibits learning/driving signal
   design a NEW protocol with a decision/update budget comparable to DrQ r6/r7
   (32,768 additional online decisions after pre-trained source initialization,
   with 10,000 update-free startup decisions and 22,768 updates). A from-scratch
   TD-MPC2 run has different initialization, prior data, replay exposure and
   compute; equal marginal steps do not make it matched. No automatic expansion.

## Failure Boundaries

Stop on a mismapped action or pixel stack, impossible evaluation latency, invalid
terminal targets, cross-episode replay, nonfinite model/critic/policy gradients,
broken replay/resume, candidate-relevant seed collision, insufficient cgroup
headroom, or planner output outside the environment action contract. Preserve
the failure and source context rather than tuning the baseline ad hoc.

The corresponding standalone code belongs under `haic/algorithms/tdmpc2/`, a
one-off operator under `scripts/`, and run receipts under `runs/`. Report proxy
metrics as local TRAIN results only; no model is nominated for official action.
