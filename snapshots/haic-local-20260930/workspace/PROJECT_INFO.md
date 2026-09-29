# HAIC project information

The project develops an agent for the 2026 HAIC CarRacing challenge. The research endpoint is a rule-compliant, independently evaluated submission candidate that completes the registered tracks reliably. The internal selection objective is completion rate on a matched, preregistered evaluation set. When completion rates tie, compare median finished lap time, mean incomplete progress, P90 finished lap time, collisions, damage, and p95 action latency, in that order. This internal ordering is separate from official scoring.

For current competition facts, follow the source precedence in [AGENTS.md](AGENTS.md) and check the [competition website](https://ships-duo-ethical-saver.trycloudflare.com/) and [official Participants repository](https://github.com/2026-HAIC/Participants) before external actions. The [pinned official Participants README](docs/sources/official-participants/README.md) and [LICENSE](docs/sources/official-participants/LICENSE) are the local source mirror at commit `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`. [COMPETITION_INFO.md](COMPETITION_INFO.md) is a historical local summary. [RESTRICTIONS.md](RESTRICTIONS.md) contains hard boundaries. Log source checks and claims in [docs/sources/INDEX.md](docs/sources/INDEX.md).

The supported workflow is plan-only by default. `harness.config.json` registers local HAIC profiles for training, closed-loop evaluation, submission packaging, and a diagnostic corridor benchmark. Design approval permits implementation within that design without another implementation approval. Each execution needs its own approved plan hash and an explicit execution approval. Official submission, model confirmation, and upload remain separate user-authorized actions. The v2 workflow writes runs to `runs/haic-research-v2/<run-id>/`, artifacts to `artifacts/haic-research-v2/<run-id>/`, and new report PDFs to `output/pdf/`.

Existing legacy results remain historical references. New experiments go in [docs/experiments/INDEX.md](docs/experiments/INDEX.md), agent reports use [docs/handoffs/REPORT_TEMPLATE.md](docs/handoffs/REPORT_TEMPLATE.md), and the current evidence summary is [docs/report.md](docs/report.md). A local candidate is not an official submission.

## V2 interfaces and operation inputs

The supported entry point is `python -m haic_research.cli`. [README.md](README.md) provides `plan`, `approve`, `status`, plan-only `run`, explicit `run --execute`, `report`, and `validate` syntax. No compatibility shim is planned. Legacy source files remain locally pending the blocked deletion step; their unsupported behavior is archived in [docs/sources/legacy-path-map.md](docs/sources/legacy-path-map.md).

`plan` registers a complete manifest, a complete `Hypothesis`, a profile ID, and typed argument JSON. The manifest needs run/cycle identifiers; purpose and hypothesis reference; candidate/control revisions and package hashes; tool/runtime versions; data/map/split identities; resource/permission limits; and source hashes. Leave approval and plan hashes unset: the harness computes them. Output paths are derived from the profile and run ID. A promotion comparison also preregisters `comparison_id`, unique `seed_ids`, and `comparison_episode_count`. A continuation uses `--previous-run` plus the checkpoint and predecessor decision references required by its prior outcome.

| Registered profile | Required operation arguments |
|---|---|
| `train_policy` | Existing train-only site-map split, `defer-tune=true`, positive `total-steps` |
| `evaluate_closed_loop` | Explicit policy and dynamics checkpoint paths |
| `package_submission` | Explicit policy and dynamics checkpoint paths; creates a local ZIP |
| `benchmark_corridor_diagnostic` | Diagnostic-only profile; optional track/map/profile selections; ineligible for SOTA |

Use the exact argument keys and types in `harness.config.json`. Path inputs must be explicit, permitted repository inputs; plan registration fingerprints operation source and inputs. Source/input/profile changes invalidate the stored plan. `run` without `--execute` previews it without launching a process. New plans advance through design approval, then execution approval; execution approval is consumed once. The retired implementation state remains readable for historical runs.

`report` takes an `IntegrationReport` with all three named gates and evidence paths. Optional `ExperimentResult` candidate/control records bind measured metrics to the registered model and comparison identities. Local SOTA promotion additionally requires replicated independent evidence, three PASS gates and an actual ADVANCE release. A gate-only release does not itself promote SOTA. `validate` checks a fixed document/config inventory; it does not certify current official rules, measured performance, or truth of submitted metrics. Unit validation uses temporary fixtures and fake runners, not actual driving. No automatic historical scan, paper audit, result-table rebuild, or arbitrary command execution is supported.
