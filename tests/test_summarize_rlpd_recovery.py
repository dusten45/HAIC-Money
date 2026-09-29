import json

import numpy as np
import pytest

from scripts.summarize_rlpd_recovery import _sha, summarize


def dataset(tmp_path, *, censored=False):
    episodes = []
    for horizon in (0, 12, 25):
        n = 120
        mask = np.zeros(n, dtype=bool)
        mask[10:10 + horizon] = True
        role = np.full(n, "actor", dtype="U6")
        role[mask] = "oracle"
        path = tmp_path / f"h{horizon}.npz"
        np.savez(path, frames=np.zeros((n + 1, 84, 84), np.uint8), reward=np.zeros(n),
                 recovery_mask=mask, role=role, executed_action=np.zeros((n, 3), np.float32),
                 speed=np.arange(n + 1, dtype=float) - (1 if horizon else 0),
                 center_error=np.ones(n + 1), damage=np.zeros(n + 1),
                 progress=np.linspace(0, 1, n + 1))
        episodes.append({"path": path.name, "sha256": _sha(path), "steps": n,
                         "anchor_step": 10, "horizon": horizon, "stratum": "failure",
                         "policy_id": "rlpd-seed50", "geometry_seed": 4272000005,
                         "finished": bool(horizon), "censored": censored and horizon == 12,
                         "local_recovery_qualified": bool(horizon),
                         "accepted_transitions": horizon})
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "haic-rlpd-recovery-dataset-v1", "status": "complete", "episodes": episodes,
    }))
    return tmp_path


def test_same_time_telemetry_and_correlated_dedup(tmp_path):
    result = summarize(dataset(tmp_path))
    assert result["verified_episode_hashes"] == 3
    assert result["unique_accepted_transitions_by_stratum"]["failure"] == 1
    assert not result["data_gate"]["passed"]
    for table in result["finish_tables_by_horizon"].values():
        assert table["gained"] == 1 and table["lost"] == 0
        assert table["same_time_5s_followup_delta_medians"]["speed_m_s"] == -1
        assert table["same_time_5s_followup_delta_medians"]["progress"] == 0


def test_censored_pair_is_unknown_not_rescue(tmp_path):
    result = summarize(dataset(tmp_path, censored=True))
    table = result["finish_tables_by_horizon"]["12"]
    assert table["unknown"] == 1 and table["gained"] == 0


def test_changed_trace_rejected(tmp_path):
    root = dataset(tmp_path)
    with (root / "h0.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="trace"):
        summarize(root)
