import json

import numpy as np
import pytest

from scripts import resume_rlpd_coupled_recovery as r


def cases():
    return [{"policy_id": "rlpd-seed11", "geometry_seed": i, "stratum": "failure" if i < 10 else "finish-control"} for i in range(14)]


def rows():
    return [{"policy_id": "rlpd-seed11", "geometry_seed": i//3, "horizon": (0, 12, 25)[i%3], "episode_id": i} for i in range(28)]


def test_only_missing_fourteen_slots():
    missing = r.schedule(cases(), rows())
    assert len(missing) == 14
    assert [episode_id for _, _, episode_id in missing] == list(range(28, 42))
    assert missing[0][1] == 12


@pytest.mark.parametrize("change", ["duplicate", "renumber", "unexpected"])
def test_no_completed_selection_or_renumber(change):
    done = rows()
    if change == "duplicate":
        done.append(done[0])
    elif change == "renumber":
        done[0]["episode_id"] = 50
    else:
        done[0]["geometry_seed"] = 50
    with pytest.raises(r.parent.diagnosis.DiagnosisError):
        r.schedule(cases(), done)


def test_copy_exact_bytes_without_old_modification(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    source = old / "episode.npz"
    source.write_bytes(b"frozen episode bytes")
    row = {"path": source.name, "sha256": r.parent.branch._sha(source), "episode_id": 7}
    monkeypatch.setattr(r, "PARENT_OUTPUT", old)
    monkeypatch.setattr(r, "OUTPUT", new)
    r.copy_completed({"copied_episodes": [row]})
    assert source.read_bytes() == (new / source.name).read_bytes() == b"frozen episode bytes"
    assert json.loads((new / "episodes.jsonl").read_text())["episode_id"] == 7
    with pytest.raises(FileExistsError):
        r.copy_completed({"copied_episodes": [row]})


def test_signal_raises_for_partial_preservation():
    with pytest.raises(RuntimeError, match="partial trace retained"):
        r.interrupted(15, None)


def test_parent_preservation_real_preflight_without_reset(monkeypatch):
    monkeypatch.setattr(r.parent.diagnosis, "_make_env", lambda *a, **kw: pytest.fail("no reset allowed"))
    old = r.parent.preflight()
    done = r.parent.diagnosis._read_jsonl(r.PARENT_OUTPUT / "episodes.jsonl")
    assert len(done) == 28
    assert len(r.schedule(old["cases"], done)) == 14


def test_aggregate_full_census_duration_harm_and_unique_support(monkeypatch):
    done = []
    for episode_id in range(42):
        horizon = (0, 12, 25)[episode_id % 3]
        done.append({"policy_id": "rlpd-seed11", "geometry_seed": episode_id//3, "episode_id": episode_id,
                     "horizon": horizon, "stratum": "failure" if episode_id < 30 else "finish-control",
                     "steps": 100, "prefix_decisions": 5, "anchor_step": 5, "accepted_transitions": 1 if horizon else 0,
                     "local_recovery_qualified": True, "censored": False, "finished": episode_id % 3 == 1,
                     "current_anchor_curve_entry_overspeed": not horizon})
    # A truncated/unknown treatment cannot be counted as either rescue or harm.
    done[1]["censored"] = True
    done[1]["accepted_transitions"] = 0
    def arrays(row):
        return {"recovery_mask": np.array([bool(row["accepted_transitions"])]),
                "observation_sha256": np.array([str(row["geometry_seed"])]),
                "executed_action": np.array([[.1, -.2, .3]], np.float32)}
    monkeypatch.setattr(r, "arrays_at", arrays)
    monkeypatch.setattr(r.parent.branch, "_sha", lambda path: "hash")
    result = r.aggregate({"cases": cases(), "parent_protocol_sha256": "parent", "parent_interruption_receipt": {}}, done, 1.)
    assert len(result["pairs"]) == 28
    assert result["counts"]["episodes"] == 42
    assert result["counts"]["unknown_finish_pairs"] == 1
    assert result["counts"]["unique_accepted_transitions"] == 14
    assert result["counts"]["accepted_transitions"] == 27
    assert result["duration_stratum_outcomes"]["12"]["failure"]["rescued"] == 9
    assert result["duration_stratum_outcomes"]["25"]["failure"]["rescued"] == 0
    assert result["counts"]["new_decisions"] == 1400


def test_aggregate_rejects_missing_slot():
    with pytest.raises(r.parent.diagnosis.DiagnosisError, match="incomplete"):
        r.aggregate({"cases": cases()}, rows(), 0.)
