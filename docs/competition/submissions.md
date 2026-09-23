# Official Submission Ledger

The competition website is authoritative. This file is a local provenance ledger;
it must never infer an upload, server validation, result, or confirmation from a
local ZIP alone.

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

## Current Official Ledger State

No team identity mapping, official submission identifier, team-scoped API receipt,
server validation receipt, public-track result, or confirmation receipt is committed
in this repository. Anonymous team/submission API endpoints required authentication
at the migration audit. Treat every official field as **unverified** until recorded
from the site after an approved external action.

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
