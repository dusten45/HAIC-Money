# Read-only legacy path map

This map records locations only. It does not import, scan, copy, or validate their contents.

| Legacy location | Historical role | V2 handling |
|---|---|---|
| `runs/` | Prior run output | Preserve in place; select an individual source manually if needed |
| `artifacts/haic/` | Prior checkpoints, metrics, packages, and diagnostics | Preserve in place; no automatic index or rewrite |
| `submissions/` | Prior submission material | Preserve in place; no automatic index or rewrite |
| `report.pdf` | Prior root report | Preserve in place; new PDFs go under `output/pdf/` |
| `RESULTS.md` generated block | Frozen historical index | Leave byte-identical; append new human records outside it |
| `SOTA.md` current entry | Historical local PPO reference | Keep until a new candidate passes v2 gates |
