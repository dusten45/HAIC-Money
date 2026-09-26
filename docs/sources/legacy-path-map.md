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
| `research/papers.json` | Historical literature registry | Preserve as a source location; no automatic paper audit |

## Retired orchestration behavior

`research_ops.cli improve`, `research_ops.cli sync-results`, and `training.improvement_loop` are unsupported legacy entry points scheduled for removal with no compatibility shim. The replacement unit suite and structural validator passed on 2026-09-26, and all ten authorized source/test/policy deletion targets were verified as exact regular untracked files inside the checkout. Automatic approval review blocked both the guarded batch deletion and a guarded literal single-file deletion with only the reason "blocked by policy". No file was deleted; physical retirement remains pending. Directory contents outside the exact named files are preserved.

Historically, these paths scanned allowlisted summaries in the legacy roots, read a manual result registry, audited papers and `report.pdf`, rebuilt RESULTS/SOTA tables, wrote mutable `latest.json` pointers, and accepted arbitrary `--command` strings (the former CLI also accepted a submission command). None of that behavior is restored by v2. The frozen RESULTS generated block retains its original command provenance; new reports require explicit v2 run and evidence inputs.

The former three strategy lanes, PPO-specific required package files and planner restrictions, and tune-first selection with fallback to checkpoint-bearing actor records describe implementation history. They do not impose new architecture or promotion policy. The canonical contract preserves literature as hypothesis support, measured driving evidence as performance evidence, and the measurement checklist in [AGENTS.md](../../AGENTS.md). Promotion now follows registered comparisons and v2 gates.
