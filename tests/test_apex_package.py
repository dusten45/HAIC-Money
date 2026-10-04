import zipfile

import pytest

from agents.apex_2026.package import build_package


def test_submission_is_one_exact_root_agent_and_preserves_existing_archive(tmp_path):
    source = tmp_path / "candidate.py"
    content = b"import numpy as np\nclass Agent:\n    def act(self, o): return np.zeros(3, np.float32)\n"
    source.write_bytes(content)
    target = tmp_path / "submission.zip"
    receipt = build_package(source, target)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ["agent.py"]
        assert archive.read("agent.py") == content
    assert receipt["source_sha256"] == receipt["member_sha256"]["agent.py"]
    previous = target.read_bytes()
    with pytest.raises(FileExistsError):
        build_package(source, target)
    assert target.read_bytes() == previous


@pytest.mark.parametrize("code", ("import os\n", "from pathlib import Path\n", "eval('1')\n"))
def test_prohibited_inference_is_not_packaged(tmp_path, code):
    source = tmp_path / "bad.py"
    source.write_text(code, encoding="utf-8")
    target = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="prohibited"):
        build_package(source, target)
    assert not target.exists()
