"""Synthetic paired-finish preparation, with no simulator or learner imports."""

import json

import numpy as np
import pytest

from scripts import prepare_rlpd_recovery_training_data as preparer


def episode(episode_id, group, horizon, stratum, *, finished=False, censored=False, local=True, overspeed=True):
    n, anchor = (2000 if censored else 100), 3
    seed = 4272000001 + group % 10
    frames = np.broadcast_to((np.arange(n + 1) % 256).astype(np.uint8)[:, None, None], (n + 1, 84, 84)).copy()
    frames[:, 0, 0] = group
    role = np.where(np.arange(n) < anchor, "prefix", np.where(np.arange(n) < anchor + horizon, "oracle", "actor"))
    applied = np.tile(np.array([.1, .5, 0], np.float32), (n, 1))
    oracle = np.tile(np.array([-.1, .2, .2], np.float32), (n + 1, 1))
    applied[role == "oracle"] = oracle[:-1][role == "oracle"]
    proposed = np.tile(np.array([.1, 0, -1], np.float32), (n, 1))
    executed = applied.copy()
    executed[:, 1:] = executed[:, 1:] * 2 - 1
    terminated = np.zeros(n, bool)
    terminated[-1] = not censored
    finished_flags = np.zeros(n, bool)
    finished_flags[-1] = finished
    arrays = {"frames": frames, "initial_stack": preparer.stack(frames, 0), "final_stack": preparer.stack(frames, n),
              "reward": np.linspace(-.1, 1, n), "step": np.arange(n), "role": role,
              "terminated": terminated, "truncated": np.zeros(n, bool), "episode_end": terminated.copy(),
              "terminal": terminated.copy(), "finished": finished_flags,
              "proposed_action": proposed, "executed_action": executed, "applied_action": applied,
              "oracle_action": oracle, "speed": np.full(n + 1, 15 if overspeed else 10.),
              "curvature": np.full(n + 1, .03), "center_error": np.full(n + 1, 0 if local else 7.),
              "progress": np.linspace(0, 1, n + 1), "damage": np.zeros(n + 1),
              "extra_unchanged_field": np.array([42, 99])}
    qualified = preparer.qualification(arrays, anchor, horizon, censored)
    arrays["recovery_mask"] = (role == "oracle") & qualified
    row = {"episode_id": episode_id, "steps": n, "anchor_step": anchor, "horizon": horizon,
           "finished": finished, "censored": censored, "local_recovery_qualified": qualified,
           "accepted_transitions": int(arrays["recovery_mask"].sum()),
           "current_anchor_curve_entry_overspeed": overspeed if horizon == 0 else None,
           "track_id": 1, "geometry_seed": seed, "policy_id": "rlpd-seed50" if stratum == "failure" else "rlpd-seed11",
           "stratum": stratum, "mode": "actor" if horizon == 0 else f"oracle-{horizon}",
           "actor_sha256": "a" * 64, "road_centerline_sha256": "b" * 64,
           "path": f"source-{episode_id}.npz"}
    return row, arrays


def make_source(tmp_path, overrides=None):
    (tmp_path / "runs").mkdir(exist_ok=True)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for group in range(14):
        stratum = "failure" if group < 10 else "finish-control"
        for horizon in (0, 12, 25):
            kwargs = {"finished": stratum == "finish-control" or (group == 0 and horizon > 0)}
            kwargs.update((overrides or {}).get((group, horizon), {}))
            row, arrays = episode(len(rows), group, horizon, stratum, **kwargs)
            np.savez_compressed(source / row["path"], **arrays)
            row["sha256"] = preparer.sha(source / row["path"])
            rows.append(row)
    manifest = {"format": preparer.FORMAT, "status": "complete", "episodes": rows,
                "counts": {"episodes": 42, "reset_intents": 43, "unknown_attempt_decisions": [0, 2000]}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    return source, manifest


def seal(source, manifest):
    (source / "manifest.json").write_text(json.dumps(manifest))
    return preparer.sha(source / "manifest.json")


def change_array(source, manifest, index, edit):
    row = manifest["episodes"][index]
    with np.load(source / row["path"], allow_pickle=False) as data:
        arrays = dict(data)
    edit(arrays)
    np.savez_compressed(source / row["path"], **arrays)
    row["sha256"] = preparer.sha(source / row["path"])


def test_retains_all42_only_mask_changes_and_preserves_costs(tmp_path):
    source, manifest = make_source(tmp_path)
    digest = seal(source, manifest)
    result = preparer.prepare(source, digest, "runs/prepared", root=tmp_path)
    assert result["eligibility_policy"] == preparer.POLICY
    assert result["source_manifest_sha256"] == digest
    assert result["source_cost_counts"] == manifest["counts"]
    assert result["environment_resets"] == result["learner_updates"] == 0
    assert len(result["episodes"]) == 42
    assert result["sufficiency_gate"]["passed"] is False
    for old, new in zip(manifest["episodes"], result["episodes"]):
        assert new["source_path"] == old["path"] and new["source_sha256"] == old["sha256"]
        output_path = tmp_path / "runs/prepared" / new["path"]
        assert preparer.sha(output_path) == new["sha256"]
        with np.load(source / old["path"]) as original, np.load(output_path) as derived:
            assert set(original.files) == set(derived.files)
            assert all(np.array_equal(original[k], derived[k]) for k in original.files if k != "recovery_mask")
            mask = derived["recovery_mask"]
            assert int(mask.sum()) == new["accepted_transitions"]
            if new["paired_finish_qualified"]:
                assert np.array_equal(np.flatnonzero(mask), np.arange(new["training_window_start"], new["training_window_end"]))
                assert (derived["role"][mask] == "actor").sum() == 63
                assert (derived["role"][mask] == "oracle").sum() == new["horizon"]
            else:
                assert not mask.any()
    assert preparer.sha(source / "manifest.json") == digest


@pytest.mark.parametrize("override", [
    {(0, 0): {"finished": True}},
    {(0, 0): {"overspeed": False}},
    {(0, 0): {"censored": True}},
    {(0, 12): {"finished": False}, (0, 25): {"finished": False}},
    {(0, 12): {"local": False}, (0, 25): {"local": False}},
    {(0, 12): {"censored": True, "finished": False}, (0, 25): {"censored": True, "finished": False}},
])
def test_failure_support_requires_actual_current_full_rescue(tmp_path, override):
    source, manifest = make_source(tmp_path, override)
    result = preparer.prepare(source, seal(source, manifest), "runs/prepared", root=tmp_path)
    assert result["counts"]["support_by_stratum"]["failure"]["accepted_transitions"] == 0
    assert result["sufficiency_gate"]["passed"] is False


def test_finishcontrol_requires_both_actual_finishes(tmp_path):
    source, manifest = make_source(tmp_path, {(10, 0): {"finished": False}, (11, 12): {"finished": False}})
    result = preparer.prepare(source, seal(source, manifest), "runs/prepared", root=tmp_path)
    assert not result["episodes"][31]["paired_finish_qualified"]
    assert not result["episodes"][32]["paired_finish_qualified"]
    assert not result["episodes"][34]["paired_finish_qualified"]
    assert result["episodes"][35]["paired_finish_qualified"]


def test_unique_failure_support_gate_and_menu_prefix_dedup(tmp_path):
    override = {(g, h): {"finished": True} for g in range(3) for h in (12, 25)}
    source, manifest = make_source(tmp_path, override)
    result = preparer.prepare(source, seal(source, manifest), "runs/prepared", root=tmp_path)
    support = result["counts"]["support_by_stratum"]["failure"]
    assert support["accepted_geometries"] == 3
    assert support["unique_accepted_transitions"] >= 128
    assert support["unique_accepted_transitions"] < support["accepted_transitions"]
    assert result["sufficiency_gate"]["passed"] is True


@pytest.mark.parametrize("corruption", ["unfinished", "finished", "local", "overspeed", "role", "action"])
def test_source_metadata_and_arrays_must_agree(tmp_path, corruption):
    source, manifest = make_source(tmp_path)
    if corruption == "unfinished":
        change_array(source, manifest, 0, lambda a: a["episode_end"].__setitem__(-1, False))
    elif corruption == "finished":
        manifest["episodes"][1]["finished"] = False
    elif corruption == "local":
        manifest["episodes"][1]["local_recovery_qualified"] = False
    elif corruption == "overspeed":
        manifest["episodes"][0]["current_anchor_curve_entry_overspeed"] = False
    elif corruption == "role":
        change_array(source, manifest, 1, lambda a: a["role"].__setitem__(4, "actor"))
    else:
        change_array(source, manifest, 1, lambda a: a["executed_action"].__setitem__((4, 0), .3))
    with pytest.raises(ValueError):
        preparer.prepare(source, seal(source, manifest), "runs/prepared", root=tmp_path)
    assert not (tmp_path / "runs/prepared").exists()


@pytest.mark.parametrize("corruption", ["partial", "missing", "duplicate", "hash", "manifesthash", "unsafe"])
def test_complete_source_hash_and_slot_gates(tmp_path, corruption):
    source, manifest = make_source(tmp_path)
    if corruption == "partial":
        manifest["status"] = "partial"
    elif corruption == "missing":
        manifest["episodes"].pop()
    elif corruption == "duplicate":
        manifest["episodes"][-1] = manifest["episodes"][0]
    elif corruption == "hash":
        manifest["episodes"][0]["sha256"] = "0" * 64
    elif corruption == "unsafe":
        manifest["episodes"][0]["path"] = "../outside.npz"
    digest = seal(source, manifest)
    if corruption == "manifesthash":
        digest = "0" * 64
    with pytest.raises(ValueError):
        preparer.prepare(source, digest, "runs/prepared", root=tmp_path)
    assert not (tmp_path / "runs/prepared").exists()


def test_new_output_only(tmp_path):
    source, manifest = make_source(tmp_path)
    (tmp_path / "runs/existing").mkdir()
    with pytest.raises(ValueError, match="new"):
        preparer.prepare(source, seal(source, manifest), "runs/existing", root=tmp_path)


def test_finish_with_truncation_is_real_terminal_not_censored(tmp_path):
    source, manifest = make_source(tmp_path)
    def truncated_finish(a):
        a["terminated"][-1] = False
        a["truncated"][-1] = True
    change_array(source, manifest, 1, truncated_finish)
    result = preparer.prepare(source, seal(source, manifest), "runs/prepared", root=tmp_path)
    assert result["episodes"][1]["paired_finish_qualified"] is True


def test_prepared_output_integrates_with_paired_policy_loader(tmp_path):
    from haic.algorithms.rlpd.recovery import load_recovery_replay

    overrides = {(g, h): {"finished": True} for g in range(3) for h in (12, 25)}
    source, manifest = make_source(tmp_path, overrides)
    result = preparer.prepare(source, seal(source, manifest), "runs/prepared", root=tmp_path)
    output = tmp_path / "runs/prepared"
    cells = [{"partition": "TRAIN", "obstacles": True, "track_id": 1, "geometry_seed": 4272000001 + i} for i in range(10)]
    replay, receipt = load_recovery_replay(output, manifest_sha256=preparer.sha(output / "manifest.json"),
                                           allowed_cells=cells, seed=0)
    assert replay.valid_count == result["counts"]["accepted_transitions"]
    assert receipt["unique_failure_transitions"] == result["counts"]["support_by_stratum"]["failure"]["unique_accepted_transitions"]
    assert receipt["failure_geometries"] == 3
    assert replay.immutable
