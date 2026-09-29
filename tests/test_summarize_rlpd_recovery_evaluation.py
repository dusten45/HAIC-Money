import json

import numpy as np
import pytest

from scripts.evaluate_rlpd_recovery import SEEDS, sha, terminal_curve_association, transition_table
from scripts.summarize_rlpd_recovery_evaluation import summarize


def dataset(root):
    rows, hashes = [], {}
    for actor in ("v5", "control", "recovery"):
        for seed in SEEDS:
            trace = {"step": np.arange(2), "pre_speed": np.full(2, 15.),
                     "policy_or_oracle_action": np.tile([0, .7, 0], (2, 1)),
                     "oracle_action_at_state": np.tile([0, 0, .5], (2, 1)),
                     "curvature": np.full(2, .03), "center_error": np.full(2, 7.),
                     "damage": np.zeros(2)}
            path = root / f"{actor}-seed-{seed}.npz"
            np.savez(path, **trace)
            finish = seed == SEEDS[0]
            row = {"actor_id": actor, "geometry_seed": seed, "track_id": 1, "steps": 2,
                   "finished": finish, "censored": False, "terminated": not finish,
                   "truncated": finish, "reason": "finished" if finish else "off_track",
                   "progress": .95 if finish else .2, "damage": 0.,
                   "actor_sha256": actor, "trace_path": path.name, "trace_sha256": sha(path),
                   "road_centerline_sha256": str(seed)}
            row["terminal_curve_association"] = terminal_curve_association(trace, row)
            receipt = root / f"{actor}-seed-{seed}-receipt.json"
            receipt.write_text(json.dumps(row))
            rows.append(row)
            hashes[path.name], hashes[receipt.name] = sha(path), sha(receipt)
    pairs = {f"{a}->{b}": transition_table([r for r in rows if r["actor_id"] == a],
                                          [r for r in rows if r["actor_id"] == b])
             for a, b in (("v5", "control"), ("v5", "recovery"), ("control", "recovery"))}
    result = {"status": "complete", "episodes": rows, "contemporaneous_pairs": pairs,
              "per_actor": {name: {"finish_count": 1, "censored": 0,
                                   "curve_entry_associated_terminal_failures": 11}
                            for name in ("v5", "control", "recovery")}}
    (root / "result.json").write_text(json.dumps(result))
    hashes["result.json"] = sha(root / "result.json")
    (root / "manifest.json").write_text(json.dumps({"episodes_complete": 36, "files_sha256": hashes}))
    return root


def test_full_census_and_pairs(tmp_path):
    result = summarize(dataset(tmp_path))
    assert result["episodes"] == 36 and result["environment_resets"] == 0
    assert result["per_actor"]["v5"]["finished"] == 1
    assert result["pairs"]["v5->recovery"]["kept"] == 1


def test_changed_primary_trace_rejected(tmp_path):
    root = dataset(tmp_path)
    with (root / f"v5-seed-{SEEDS[0]}.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        summarize(root)


def test_wrong_claimed_aggregate_rejected(tmp_path):
    root = dataset(tmp_path)
    result = json.loads((root / "result.json").read_text())
    result["per_actor"]["v5"]["finish_count"] = 2
    (root / "result.json").write_text(json.dumps(result))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files_sha256"]["result.json"] = sha(root / "result.json")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="aggregate"):
        summarize(root)
