from pathlib import Path
import sys

from research_ops import cli
from research_ops.cli import run_improvement


def test_run_improvement_is_plan_only_by_default(tmp_path: Path):
    for name in ("RULES.md", "SOTA.md", "COMPETITION_INFO.md", "RESTRICTIONS.md", "report.pdf"):
        (tmp_path / name).write_bytes(b"ready")

    result = run_improvement(tmp_path, "성능 개선해 줘")

    assert result["execution"]["requested"] is False
    assert result["submission"]["requested"] is False
    assert (tmp_path / "RESULTS.md").exists()
    assert Path(result["plan_path"]).exists()


def test_run_improvement_rejects_submit_without_confirmation(tmp_path: Path):
    for name in ("RULES.md", "SOTA.md", "RESULTS.md", "COMPETITION_INFO.md", "RESTRICTIONS.md", "report.pdf"):
        (tmp_path / name).write_bytes(b"ready")

    try:
        run_improvement(
            tmp_path,
            "성능 개선해 줘",
            submit=True,
            submission_command="python -c pass",
        )
    except ValueError as error:
        assert "confirm" in str(error)
    else:
        raise AssertionError("submission should require explicit confirmation")


def test_explicit_command_uses_configured_timeout(monkeypatch, tmp_path: Path):
    captured = {}

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(args, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        return Completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli._run_explicit_command("python -c pass", tmp_path, timeout_seconds=7_200)

    assert captured["timeout"] == 7_200


def test_explicit_command_removes_windows_argument_quotes(tmp_path: Path):
    command = (
        f'"{sys.executable}" -c "import sys; print(sys.argv[1])" '
        '"argument with spaces"'
    )

    result = cli._run_explicit_command(command, tmp_path)

    assert result["returncode"] == 0, result["stderr_tail"]
    assert result["stdout_tail"].strip() == "argument with spaces"
    assert result["stdout"].strip() == "argument with spaces"


def test_run_improvement_saves_full_command_logs(monkeypatch, tmp_path: Path):
    for name in ("RULES.md", "SOTA.md", "RESULTS.md", "COMPETITION_INFO.md", "RESTRICTIONS.md", "report.pdf"):
        (tmp_path / name).write_bytes(b"ready")

    monkeypatch.setattr(
        cli,
        "_run_explicit_command",
        lambda command, root, timeout_seconds: {
            "args": ["python", "-c", "pass"],
            "returncode": 0,
            "stdout": "full per-update result payload\n",
            "stderr": "diagnostic warning\n",
            "stdout_tail": "full per-update result payload\n",
            "stderr_tail": "diagnostic warning\n",
        },
    )

    result = run_improvement(
        tmp_path,
        "성능 개선해 줘",
        execute=True,
        command="python -c pass",
    )

    command_result = result["execution"]["command_result"]
    stdout_log = tmp_path / command_result["stdout_log_path"]
    stderr_log = tmp_path / command_result["stderr_log_path"]
    assert stdout_log.read_text(encoding="utf-8") == "full per-update result payload\n"
    assert stderr_log.read_text(encoding="utf-8") == "diagnostic warning\n"


def test_improve_cli_accepts_timeout_seconds(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["research_ops", "improve", "성능 개선해 줘", "--timeout-seconds", "7200"],
    )

    args = cli._parse_args()

    assert args.timeout_seconds == 7_200


def test_run_improvement_rejects_nonpositive_timeout(tmp_path: Path):
    try:
        run_improvement(tmp_path, "성능 개선해 줘", timeout_seconds=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive timeout should be rejected")
