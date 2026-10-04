# Official Submission Ledger

The competition website is authoritative. This file is a local provenance ledger;
it must never infer an upload, server validation, result, or confirmation from a
local ZIP alone.

## Current Submission Baseline (2026-10-02)

**User selection:** crossing_projection + collision-shield v1 is the new submission
baseline. This records the explicit designation and official result reported by the
user at2026-10-02T01:36:47Z, not a new upload or competition-site model confirmation.

| Field | Recorded value |
|---|---|
| User-reported official result | Track4 finished in18.4s;6th overall at report time |
| Package binding | User-bound identity, not independently verified against a site receipt |
| Selected ZIP SHA-256 | `c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801` |
| Frozen reference | [`submissions/20261002-crossing-projection-collision-shield-v1-baseline/`](../../submissions/20261002-crossing-projection-collision-shield-v1-baseline/) |
| Archive layout | Exact `submission.zip`, `manifest.json`, `source/` with all11 exact ZIP members, standalone stdlib `restore.py`, `runtime-requirements.txt`, `provenance/`, `SHA256SUMS` |
| Original package records | [`koi-collision-shield-v1-experimental-submission.zip`](../../submissions/koi-collision-shield-v1-experimental-submission.zip) and [original local packaging receipt](../../submissions/koi-collision-shield-v1-experimental-submission.receipt.json), retained unchanged |
| Shield source SHA-256 | `ad772bde9a9c3f4596fdfc742e7cac33f1b605ff52a2b3f76fbd4b75d6361d96` |
| Retained standalone crossing ZIP SHA-256 | `a4b35c56659555f5e60a372261df722628060e4afcd73c88ed9e9688cdb82bb8` |
| Site submission ID / server receipt / confirmation receipt | Not supplied; no independent official verification in this update |
| Freeze verification | Original ZIP/member hashes and preserved artifacts match. Isolated stdlib exact-copy restore and source-only ZIP rebuild both reproduce `c9e376a0...`; no Agent import, simulator reset or performance experiment. |

The original research **NOT_ADOPTED / failed+20ms efficiency gate** is unchanged.
The new submission selection does not pass that gate retroactively. Keep original
sources/parameters, root Agent and previous candidates unchanged; no retuning,
evaluation, upload or confirmation follows from this documentation update.

To restore an identical submission ZIP to a **new** output path from the repo root:

```bash
python submissions/20261002-crossing-projection-collision-shield-v1-baseline/restore.py restore \
  --bundle submissions/20261002-crossing-projection-collision-shield-v1-baseline \
  --output submissions/crossing-projection-collision-shield-v1-restored.zip
```

The helper needs only Python's standard library and works after moving the whole
bundle elsewhere; it has no worktree or `/tmp` environment dependency. `verify`
checks the bundle without writing. `rebuild` restores from frozen source instead
of copying the ZIP, checks the same exact hash and fails closed on Python/zlib
serialization drift. Prefer the preserved `submission.zip` for resubmission.
Existing outputs and writes inside the bundle are refused. These operations never
upload anything or overwrite the current root Agent.

## Recorded Local Package Artifact

| Field | Recorded value |
|---|---|
| Local record | `submissions/20260919T135436Z_baseline1-final/manifest.json` |
| Created | 2026-09-19T13:54:36.297272Z |
| Source commit | `18eacb570db02de44675320a56f6bb1f31e943ff` (`dirty: true`) |
| Packaged agent SHA-256 | `40de8b952820e3b3b39760bbda02da7063a2f576ad76c7551d3636d778720b32` |
| Packaged model SHA-256 | `c101c6964a915ee6e5be93f90dae2b5a4b798021c9aea120bc14e89992647f12` |
| ZIP SHA-256 | `2b0e1c319f4a883407bde1822d6470d99c917870c48f27fc0aa231804ee99aad` |
| Local smoke | Init 1.523 s, action 0.004 s, finite shape `(3,)` |
| Official submission identifier | Not recorded |
| Server validation/public-track result | Not recorded |
| Confirmation | Not recorded |

`package_submission.py` creates this kind of local record and does not upload to
the competition site. The current root `agent.py` differs from the archived agent;
the archived model matches root `model.pt`. Therefore this record cannot stand in for
the current source tree or an official candidate.

## DrQ-v2 Local Package (2026-09-29)

| Field | Recorded value |
|---|---|
| Local record | [`submissions/20260929T100923330496Z_drqv2-pad4-control-seed1/manifest.json`](../../submissions/20260929T100923330496Z_drqv2-pad4-control-seed1/manifest.json) |
| Submission-ready local ZIP | [`submission.zip`](../../submissions/20260929T100923330496Z_drqv2-pad4-control-seed1/submission.zip) (1,105,982 bytes; four root files) |
| ZIP SHA-256 | `88b2dcfd9d40350e23116419eef462f1f3e2833947d2f980ddfe5ae7657a5fd7` |
| Source actor | `runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/checkpoints/step-000131072/actor.pt` |
| Actor and ZIP `model.pt` SHA-256 | `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954` |
| Model tag | `haic-drq-v2-actor-v1`, verified from packed actor bytes |
| Internal selection evidence | Same actor, 7/32 fresh confirmation cells on each of two distinct DrQ cohorts; one policy, not two independent training replicas |
| Local CPU smoke | Python 3.11.14 / Torch 2.1.0+cpu; init 1.513 s, combined act/reset sequence 0.00465 s, shape `(3,)`, finite and deterministic reset |
| Independent extracted-ZIP smoke | Repeat init 1.531 s, combined act/reset sequence 0.00382 s, child peak RSS 317,012 KiB (about 310 MiB); shape `(3,)`, finite and deterministic reset |
| Official Docker/server validation | Not run; Docker is not installed on this host |
| Official submission ID, public result, confirmation | **None**; no server upload or model confirmation occurred |

This is the strongest *DrQ-v2* actor by its two internal fresh-confirmation
cohorts, not the project-wide selected model or a guarantee of official-track
generalization. The newer r7/final-source comparisons used previously
consumed TRAIN-DIAGNOSTIC roads, and the original final-source historical
warmup audit failed. The root packager's smoke is partial local validation:
it does not establish official container acceptance, all individual timeout
or memory limits, or a full driving episode. The current site bundle's fixed
text says three uploads per KST day, while a prior owner report says five;
the effective authenticated team quota is unknown. Verify it only if an
official upload is later explicitly approved, and bind any site receipt to
this exact ZIP SHA-256 rather than treating this local record as a submission.

### Local Root-Packager Boundary

The active root packager checks a root-only layout, recognized source modules,
forbidden static imports/calls, archive limits, and a clean smoke extraction. Its
smoke checks construction and a combined small reset/action sequence; it does not
independently prove every official reset/act timeout, RSS, action bounds, a full
episode, dependency installation, or official Docker behavior. It also cannot add
root `requirements.txt` or arbitrary custom modules even though the official format
permits them. Treat a passing manifest as partial local validation only.

The output directory is unique but not filesystem-immutable. Preserve its hashes and
do not edit a package after recording it; a future official ledger entry must bind
the uploaded ZIP hash to the source/checkpoint and site receipt.

## Historical Packaging Record (2026-10-01)

### Shield v1 Local Experimental Package (2026-10-01)

- Version: `koi-collision-shield-v1-experimental-submission`.
- [ZIP](../../submissions/koi-collision-shield-v1-experimental-submission.zip):19,660
  bytes,11 Python members; root `agent.py`, no extra requirements or model weights.
- SHA-256:`c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801`.
- [Receipt](../../submissions/koi-collision-shield-v1-experimental-submission.receipt.json):
  frozen shield source`ad772bde...` and crossing dependency bytes unchanged; all
  prior ZIPs/root Agent/v2 and original NOT_ADOPTED result preserved.
- Current official README/site reopened; ZIP/static/CRC/limits pass. Isolated
  Python3.11.14/NumPy1.26.0/OpenCV4.8.1.78/CPU21 smoke:64 exact frozen-policy action
  and diagnostic comparisons, deterministic reset, bounded finite float32 outputs;
  import/construct.132743s, max act.010931s, measured peak RSS509,173,760 bytes.
- Purpose: imminent obstacle-collision DNF prevention only, not overavoidance
  optimization. Research efficiency-gate failure remains unchanged.
- No simulator reset, new evaluation, upload, official server validation or model
  confirmation. User-reported prior Track4 DNF is not a verified receipt for this ZIP.

## Current Official Ledger State

A user-reported public-track result is now recorded in the current baseline entry
above. No team identity mapping, official submission identifier, team-scoped API
receipt, server validation/result receipt or confirmation receipt is recorded here.
The result and package binding remain **user-reported, not independently verified**.
The original local packaging receipt is not a server receipt. Anonymous
team/submission API endpoints required authentication at the historical migration
audit; no network request or external action is part of this documentation update.

## Required Entry for a Future Official Action

For each actual official submission or confirmation, add one immutable entry with:

- KST and UTC date/time, official submission identifier, and official-site URL or
  receipt reference.
- Source commit, run/checkpoint path, checkpoint/model hash, package hash, and local
  validation artifact.
- Server validation state, failure reason if any, and public-track results.
- Whether it is eligible for confirmation, actually confirmed, replaced, or retired.
- A clear note if public results come from a different package than another row.

Submission quota is scarce. Use the official recheck and local validation gate in
[`../workflows/prepare-submission.md`](../workflows/prepare-submission.md) before an
explicitly approved upload.
