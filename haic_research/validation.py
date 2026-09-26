"""Structural checks of a fixed inventory; never inspect historical run data.

A passing result establishes document/config structure only. It does not certify
current competition compliance, gate evidence, or measured racing performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, load_config, validate_config


REQUIRED_PATHS = (
    "README.md", "AGENTS.md", "PROJECT_INFO.md", "RESTRICTIONS.md", "harness.config.json",
    "COMPETITION_INFO.md", "RESULTS.md", "SOTA.md", ".gitignore", "output/pdf/.gitkeep",
    "haic_research/__init__.py", "haic_research/config.py", "haic_research/models.py",
    "haic_research/state.py", "haic_research/policy.py", "haic_research/coordinator.py",
    "haic_research/records.py", "haic_research/commands.py", "haic_research/results.py",
    "haic_research/validation.py", "haic_research/cli.py", "scripts/harness/validate.ps1",
    "docs/experiments/INDEX.md", "docs/handoffs/REPORT_TEMPLATE.md", "docs/sources/INDEX.md",
    "docs/sources/legacy-path-map.md", "docs/sources/official-participants/README.md",
    "docs/sources/official-participants/LICENSE", "docs/strategy-history.md", "docs/report.md",
    "tests/test_haic_research_config.py", "tests/test_haic_research_state.py",
    "tests/test_haic_research_policy.py", "tests/test_haic_research_coordinator.py",
    "tests/test_haic_research_records.py", "tests/test_haic_research_commands.py",
    "tests/test_haic_research_results.py", "tests/test_haic_research_cli.py",
    "tests/test_haic_research_validation.py",
)
SITE = "https://ships-duo-ethical-saver.trycloudflare.com/"
PARTICIPANTS = "https://github.com/2026-HAIC/Participants"
PIN = "1c11db8afc2fbfcfb610672b7ee0ecd122c97741"
DOCUMENT_REFERENCES = {
    "AGENTS.md": (SITE, PARTICIPANTS, "docs/sources/official-participants/README.md",
                  "docs/sources/INDEX.md", "PROJECT_INFO.md", "RESTRICTIONS.md",
                  "central coordinator", "docs/handoffs/REPORT_TEMPLATE.md"),
    "PROJECT_INFO.md": (SITE, PARTICIPANTS, "AGENTS.md", "RESTRICTIONS.md",
                        "docs/sources/official-participants/README.md",
                        "docs/sources/official-participants/LICENSE", PIN,
                        "harness.config.json", "docs/experiments/INDEX.md", "docs/report.md"),
    "RESTRICTIONS.md": ("AGENTS.md", "confirmation", "blind", "plan-only", "rule_compliance"),
    "docs/sources/INDEX.md": (SITE, PARTICIPANTS, PIN, "official-participants/README.md",
                              "official-participants/LICENSE", "checked", "supported claim"),
    "docs/handoffs/REPORT_TEMPLATE.md": ("## fact", "## inference", "## unknown",
                                        "## recommendation", "## source_paths"),
}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def validate_project(root: Path) -> list[ValidationIssue]:
    """Validate explicit files and shared config policy without discovery or execution."""
    root = Path(root).resolve()
    issues: list[ValidationIssue] = []
    available: set[str] = set()
    for name in REQUIRED_PATHS:
        path = root / name
        try:
            # Refuse redirected inventory paths before any content is opened.
            if path.resolve() != path or not path.is_file():
                issues.append(ValidationIssue(name, "required regular file is missing or redirected"))
            else:
                available.add(name)
        except OSError as exc:
            issues.append(ValidationIssue(name, f"cannot inspect required file: {exc}"))

    if "harness.config.json" in available:
        try:
            config = load_config(root, validate=False)
            issues.extend(ValidationIssue("harness.config.json", f"{issue.path}: {issue.message}")
                          for issue in validate_config(config))
        except (ConfigError, OSError, TypeError, ValueError) as exc:
            issues.append(ValidationIssue("harness.config.json", str(exc)))

    for name, references in DOCUMENT_REFERENCES.items():
        if name not in available:
            continue
        try:
            text = (root / name).read_text(encoding="utf-8").lower()
        except (OSError, UnicodeError) as exc:
            issues.append(ValidationIssue(name, f"cannot read required document: {exc}"))
            continue
        missing = [reference for reference in references if reference.lower() not in text]
        if missing:
            issues.append(ValidationIssue(name, "missing governing references/roles: " + ", ".join(missing)))
    return issues
