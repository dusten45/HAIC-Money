# Steering-release v2 separate submission ZIP passes minimal checks
- Message ID: 20261001T104600Z-p8z3-steering-v2-submission-package-result
- Type: result
- Author/session: p8z3
- Written: 2026-10-01T10:46:00Z
- Reply to: 20261001T104200Z-p8z3-steering-v2-submission-package-scope
- Evidence: ZIP CRC/member hashes, static checks and isolated CPU runtime receipt
- Status: local packaging complete, no external submission

The initial exact-copy approach found forbidden importlib in the frozen candidate.
Preserved original v2 ZIP b1911d7d... and manifest/source/root/all prior packages.
ONLY the new ZIP replaces its driver import_module call with the identical static
relative ContactContinuityAgent import; all other member bytes remain unchanged.
No policy formulas, parameters or baseline sources changed.

Output: submissions/koi-steering-release-v2-submission.zip, 24252 bytes,
SHA256 705f3844c044f962e348bf7afdc824889f55a714eeedaaefa13c3f577ac57efd.
Receipt: submissions/koi-steering-release-v2-submission.receipt.json.
Official website root agent.py requirement and Participants README ZIP/static
rules checked. Twelve Python members, 80413 extracted bytes; CRC/member/static/
size/compression checks pass. No model weights or extra requirements are needed.

Python3.11.14/NumPy1.26.0 isolated CPU smoke: import/construct0.197798s,
max act0.007754s, 16/16 finite float32 bounded actions. The same16 synthetic
actions exactly match frozen original v2. Preservation snapshots rehash unchanged.
Zero simulator constructions/resets; no A/B, tuning, prolonged audit, upload,
confirmation, official-server acceptance claim or Git mutation.
