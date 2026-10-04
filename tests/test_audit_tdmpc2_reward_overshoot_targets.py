"""Synthetic replay-window and read-only RAW source binding checks."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import audit_tdmpc2_reward_overshoot_targets as audit
from scripts import diagnose_tdmpc2_h5_logged as bound
from tests.test_diagnose_tdmpc2_h5_logged import _bound_fixture


def _steps(tmp_path, lengths, endings, rewards=None):
    episodes, lines = [], []
    decision = 0
    for eid, (length, ending) in enumerate(zip(lengths, endings)):
        road = bound.ROADS[eid % 4]
        for offset in range(length):
            decision += 1
            last = offset == length - 1
            lines.append({"decision": decision, "episode": eid, "track_id": 1,
                          "geometry_seed": road,
                          "reward": float(rewards[eid][offset] if rewards else offset),
                          "terminated": bool(last and ending == "terminated"),
                          "truncated": bool(last and ending != "terminated"),
                          "terminal": bool(last and ending != "timeout")})
        episodes.append({"episode": eid, "length": length, "decisions": decision,
                         "track_id": 1, "geometry_seed": road, "finished": ending == "finished",
                         "terminated": ending == "terminated", "truncated": ending != "terminated",
                         "terminal": ending != "timeout"})
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in lines))
    return episodes, path


def _sealed_small_source(tmp_path, monkeypatch):
    run, _, _, producer = _bound_fixture(tmp_path, monkeypatch)
    (tmp_path / audit.SELF).write_bytes(Path(audit.__file__).read_bytes())
    ledger = run / "training.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    for row in rows:
        if row["event"] == "episode":
            row["finished"] = False
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(bound, "TRAIN_SHA", bound._digest(ledger))
    result = run / "result.json"
    data = json.loads(result.read_text())
    data["training_ledger_sha256"] = bound.TRAIN_SHA
    result.write_text(json.dumps(data))
    monkeypatch.setattr(bound, "RESULT_SHA", bound._digest(result))
    return producer


def test_exact_per_episode_starts_suffix_and_no_cross_episode_prefix(tmp_path):
    episodes, path = _steps(tmp_path, [5, 5, 3, 4],
                            ["terminated", "timeout", "finished", "terminated"],
                            [[0, 1, 2, 3, 4], [100, 101, 102, 103, 104],
                             [10, 11, 12], [20, 21, 22, 23]])
    result = audit._audit_steps(episodes, path)
    assert result["step_rows"] == 17
    assert result["episode_boundaries"] == 4
    assert result["boundary_types"] == {"terminated": 2, "finished_truncation": 1,
                                        "ordinary_timeout": 1}
    assert [r["h3_eligible_starts"] for r in result["roads"].values()] == [3, 3, 1, 2]
    assert [r["h5_extended_starts"] for r in result["roads"].values()] == [1, 1, 0, 0]
    assert result["roads"][str(bound.ROADS[0])]["suffix_min"] == pytest.approx(3 + .995 * 4)
    assert result["roads"][str(bound.ROADS[1])]["suffix_max"] == pytest.approx(103 + .995 * 104)
    assert audit._gate(result["roads"])["h5_extended_starts"] == 2


@pytest.mark.parametrize("fault", ["missing_row", "incomplete_line", "extra_row", "cross_boundary",
                                     "wrong_road", "nonfinite", "wrong_final", "wrong_finish"])
def test_incomplete_tampered_or_cross_episode_ledger_rejected(tmp_path, fault):
    episodes, path = _steps(tmp_path, [5, 5], ["terminated", "timeout"])
    lines = path.read_text().splitlines()
    if fault == "missing_row":
        lines.pop()
    elif fault == "incomplete_line":
        path.write_bytes(path.read_bytes()[:-1])
    elif fault == "extra_row":
        lines.append(lines[-1])
    elif fault == "wrong_finish":
        episodes[-1]["finished"] = True
    else:
        row = json.loads(lines[2 if fault != "wrong_final" else 4])
        if fault == "cross_boundary":
            row["terminated"] = row["terminal"] = True
        elif fault == "wrong_road":
            row["geometry_seed"] = bound.ROADS[1]
        elif fault == "nonfinite":
            row["reward"] = float("nan")
        else:
            row["terminal"] = False
        lines[2 if fault != "wrong_final" else 4] = json.dumps(row)
    if fault != "incomplete_line":
        path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        audit._audit_steps(episodes, path)


def test_95_percent_inclusive_and_three_roads_strict_range(tmp_path):
    lengths = [42] * 4  # 38 / 40 exactly .95 on every road.
    rewards = [[offset % 3 for offset in range(42)] for _ in range(3)] + [[0] * 42]
    episodes, path = _steps(tmp_path, lengths, ["terminated"] * 4, rewards)
    result = audit._gate(audit._audit_steps(episodes, path)["roads"])
    assert (result["h3_eligible_starts"], result["h5_extended_starts"]) == (160, 152)
    assert result["extension_fraction"] == .95
    assert result["roads_with_suffix_range_gt_0_1"] == 3
    assert result["gate"]["passed"]

    short, short_path = _steps(tmp_path, [41] * 4, ["terminated"] * 4,
                               [values[:41] for values in rewards])
    low_coverage = audit._gate(audit._audit_steps(short, short_path)["roads"])
    assert low_coverage["roads_with_suffix_range_gt_0_1"] == 3
    assert not low_coverage["gate"]["extension_coverage_passed"]
    assert not low_coverage["gate"]["passed"]

    rewards[2] = [0] * 42
    few, few_path = _steps(tmp_path, lengths, ["terminated"] * 4, rewards)
    low_diversity = audit._gate(audit._audit_steps(few, few_path)["roads"])
    assert low_diversity["gate"]["extension_coverage_passed"]
    assert low_diversity["roads_with_suffix_range_gt_0_1"] == 2
    assert not low_diversity["gate"]["passed"]
    strict = {str(seed): {"h3_eligible_starts": 40, "h5_extended_starts": 38,
                         "suffix_range": .1} for seed in bound.ROADS}
    assert not audit._gate(strict)["gate"]["road_range_passed"]


def test_source_tamper_is_refused_without_deserializing(tmp_path, monkeypatch):
    producer = _sealed_small_source(tmp_path, monkeypatch)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint deserialized"))
    producer.write_bytes(b"tampered producer source")
    with pytest.raises(ValueError, match="source hash mismatch"):
        audit.audit(root=tmp_path)


def test_mutation_between_before_after_hashes_is_refused(tmp_path, monkeypatch):
    _sealed_small_source(tmp_path, monkeypatch)
    original = audit._input_hashes
    calls = 0

    def changed_after(root):
        nonlocal calls
        hashes = original(root)
        calls += 1
        if calls == 2:
            hashes[f"{bound.RUN}/steps.jsonl"] = "0" * 64
        return hashes

    monkeypatch.setattr(audit, "_input_hashes", changed_after)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint deserialized"))
    with pytest.raises(ValueError, match="changed during audit"):
        audit.audit(root=tmp_path)
    assert calls == 2


def test_full_synthetic_cli_stdout_only_and_body_sha(tmp_path, monkeypatch, capsys):
    _sealed_small_source(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint deserialized"))
    audit.main()
    output = capsys.readouterr()
    assert not output.err and output.out.count("\n") == 1
    result = json.loads(output.out)
    body_sha = result.pop("body_sha256")
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == body_sha
    assert result["input_sha256_before"] == result["input_sha256_after"]
    assert result["source_protocol_sha256"] == bound.SOURCE_PROTOCOL_SHA
    assert (result["environment_resets"], result["torch_load_calls"]) == (0, 0)
    assert result["step_rows"] == 12 and result["episode_boundaries"] == 2
    assert result["boundary_types"] == {"terminated": 1, "finished_truncation": 0,
                                        "ordinary_timeout": 1}
    assert result["gate"]["passed"] is False
