"""Generated fixtures only: no real checkpoints or environment interaction."""

import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_tdmpc2_head_fit_transfer as script


@pytest.fixture(autouse=True)
def no_environment_or_optimizer_actions(monkeypatch):
    import gymnasium
    from haic.algorithms.tdmpc2 import haic_env
    from core.vendor.car_racing import CarRacing

    def forbidden(*args, **kwargs):
        pytest.fail("environment/optimizer operation in frozen diagnostic")

    monkeypatch.setattr(gymnasium, "make", forbidden)
    monkeypatch.setattr(haic_env, "make_training_env", forbidden)
    monkeypatch.setattr(CarRacing, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_exact_split_and_seed_boundary():
    assert [script.phase(e, d) for e, d in [(0, 1), (11, 3844), (12, 3845), (28, 10000),
                                           (28, 10001), (43, 13773), (44, 13774), (306, 100354)]] == [
        None, None, "excluded_random", "excluded_random", "excluded_early_planned",
        "excluded_early_planned", "fit_planned", "fit_planned"]
    for e, d in [(307, 100355), (True, 1), (12, 0), (44, 10000)]:
        with pytest.raises(ValueError):
            script.phase(e, d)
    assert script.BATCH == 256 and script.SEED == 20260929
    assert sum(sum(pair) for pair in script.EXPECTED_COUNTS["fit_planned"]) == 86581
    assert sum(sum(pair) for pair in script.EXPECTED_COUNTS["excluded_random"]) == 6156
    assert sum(sum(pair) for pair in script.EXPECTED_COUNTS["excluded_early_planned"]) == 3773


def test_preflight_source_drift_before_load(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(script, "_sources", lambda root: (_ for _ in ()).throw(ValueError("source drift")))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("load before SHA checks"))
    with pytest.raises(ValueError, match="source drift"):
        script.preflight(tmp_path)
    assert not (tmp_path / script.OUTPUT).exists()


def test_load_rechecks_all_pins_before_deserialization(tmp_path, monkeypatch):
    spec = {"device": "cpu", "raw_source": {"checkpoint": {}},
            "overshoot_source": {"checkpoint": {}}, "adapted_checkpoint": {}}
    monkeypatch.setattr(script, "preflight", lambda *a, **k: (_ for _ in ()).throw(ValueError("changed SHA")))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("load before all pins"))
    with pytest.raises(ValueError, match="changed SHA"):
        script._load(tmp_path, {"schema": spec}, "a" * 64)


def test_open_checkpoint_sha_checked_before_load(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.pt"
    path.write_bytes(b"not pickle")
    ref = {"path": path.name, "sha256": "a" * 64}
    spec = {"device": "cpu", "raw_source": {"checkpoint": ref},
            "overshoot_source": {"checkpoint": ref}, "adapted_checkpoint": ref}
    monkeypatch.setattr(script, "preflight", lambda *a, **k: {"schema": spec})
    monkeypatch.setattr(script.head, "_pinned", lambda *a: path)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("load before stream SHA"))
    with pytest.raises(ValueError, match="SHA mismatch before"):
        script._load(tmp_path, {"schema": spec}, "b" * 64)


def test_all_pins_rechecked_before_second_checkpoint_load(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.pt"
    path.write_bytes(b"synthetic checkpoint bytes")
    ref = {"path": path.name, "sha256": script.head.raw._digest(path)}
    spec = {"device": "cpu", "raw_source": {"checkpoint": ref},
            "overshoot_source": {"checkpoint": ref}, "adapted_checkpoint": ref}
    checks, loads = [], []

    def checked(*args, **kwargs):
        checks.append(True)
        if len(checks) == 2:
            raise ValueError("dependency drift before second load")
        return {"schema": spec}

    monkeypatch.setattr(script, "preflight", checked)
    monkeypatch.setattr(script.head, "_pinned", lambda *a: path)
    monkeypatch.setattr(torch, "load", lambda *a, **k: loads.append(True) or {})
    with pytest.raises(ValueError, match="before second load"):
        script._load(tmp_path, {"schema": spec}, "b" * 64)
    assert len(checks) == 2 and len(loads) == 1


def test_no_load_schema_is_deterministic_and_frozen_schema_drift_rejected(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    for name in (script.SELF, script.TEST):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("synthetic dependency\n")
    groups = {p: {str(r): {s: [(44 if p == "fit_planned" else 12, si, 15000 if p == "fit_planned" else 9999, 1.)]
                           for si, s in enumerate(script.SIGNS)} for r in script.head.raw.ROADS} for p in script.PHASES}
    spec = {"raw_source": {"step_ledger": {"path": "synthetic.jsonl"}}, "overshoot_source": {}, "source_sha256": {}}
    pilot = {"journal_sha256": "a" * 64}
    monkeypatch.setattr(script, "_sources", lambda root: (pilot, spec, [], {}))
    monkeypatch.setattr(script, "groups_from_ledger", lambda *a: groups)
    monkeypatch.setattr(script, "EXPECTED_COUNTS", {p: ((1, 1),) * 4 for p in script.PHASES})
    monkeypatch.setattr(script.head, "_runtime", lambda: {"synthetic": True})
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("preflight deserialized"))
    first = script.preflight(tmp_path)
    assert script.preflight(tmp_path)["schema"] == first["schema"]
    assert first["torch_load_calls"] == first["optimizer_updates"] == first["environment_resets"] == 0
    assert first["schema"]["fit_gate"] == script.GATE
    assert first["schema"]["selection"]["batch_size"] == 256
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(first["schema"]))
    monkeypatch.setattr(script.head, "_pinned", lambda *a: frozen)
    assert script.preflight(tmp_path, "b" * 64)["schema"] == first["schema"]
    altered = copy.deepcopy(first["schema"])
    altered["selection"]["augmentation_seed"] += 1
    frozen.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="frozen diagnostic protocol"):
        script.preflight(tmp_path, "b" * 64)
    (tmp_path / script.OUTPUT).write_text("{}")
    with pytest.raises(ValueError, match="exclusive"):
        script.preflight(tmp_path)


def test_all_unique_ledger_transitions_no_parent309_mixing(monkeypatch):
    lengths = [327] * 306 + [292]
    episodes, rows, decision = [], [], 0
    for eid, length in enumerate(lengths):
        road = script.head.raw.ROADS[eid % 4]
        for step in range(length):
            decision += 1
            rows.append({"episode": eid, "decision": decision, "geometry_seed": road, "track_id": 1,
                         "reward": 3. if step % 2 else -.4})
        episodes.append({"episode": eid, "length": length, "decisions": decision, "geometry_seed": road, "track_id": 1})
    iterator = iter(rows)
    monkeypatch.setattr(script.head.raw, "_json", lambda line: next(iterator))
    monkeypatch.setattr(script.Path, "open", lambda *a, **k: io.BytesIO(b"{}\n" * decision))
    groups = script.groups_from_ledger(episodes, script.Path("synthetic"))
    entries = [x for p in groups.values() for road in p.values() for sign in road.values() for x in sign]
    assert len(entries) == len({(e, s) for e, s, _, _ in entries}) == 100354 - 12 * 327
    assert sum(len(s) for r in groups["excluded_random"].values() for s in r.values()) == 10000 - 12 * 327
    assert all(e >= 44 and d > 10000 for r in groups["fit_planned"].values() for s in r.values() for e, _, d, _ in s)
    with pytest.raises(ValueError, match="RAW307"):
        script.groups_from_ledger(episodes + [{}, {}], script.Path("synthetic"))


def test_metrics_natural_and_balanced_ce():
    pos = script.summarize([2, 4], [1, 3], [1, 3])
    neg = script.summarize([-.4] * 6, [0] * 6, [4] * 6)
    assert pos == {"count": 2, "ce": 2., "mae": 1., "signed_bias": -1., "prediction_mean": 2., "target_mean": 3.}
    roads = {str(r): {"positive": pos, "nonpositive": neg} for r in script.head.raw.ROADS}
    summary = script.ce_summary(roads)
    assert summary["count"] == 32
    assert summary["natural_ce"] == summary["balanced_ce_25_75_uniform_roads"] == 3.5
    roads[str(script.head.raw.ROADS[0])]["positive"] = {**pos, "count": 8}
    summary = script.ce_summary(roads)
    assert summary["natural_ce"] == pytest.approx(124 / 38)
    assert summary["balanced_ce_25_75_uniform_roads"] == 3.5


@pytest.mark.parametrize("which", [0, 1, 2])
def test_nonfinite_metrics_fail_closed(which):
    values = [[1.], [0.], [1.]]
    values[which] = [float("nan")]
    with pytest.raises(ValueError, match="nonfinite"):
        script.summarize(*values)


def _gate_metrics():
    old = {str(r): {"positive": {"count": 100, "mae": 1., "signed_bias": -1.},
                    "nonpositive": {"count": 200, "mae": .1, "signed_bias": .1}}
           for r in script.head.raw.ROADS}
    new = copy.deepcopy(old)
    for row in new.values():
        row["positive"].update(mae=.9, signed_bias=-.9)
        row["nonpositive"].update(mae=.15)
    return old, new


def test_gate_exact_borderlines_and_three_roads():
    old, new = _gate_metrics()
    report = script.fit_gate(old, new)
    assert report["status"] == "PASS" and report["qualifying_roads"] == 4
    assert report["policy_gate"] is False and report["policy_release"] is False
    new[str(script.head.raw.ROADS[0])]["positive"]["mae"] = .90000001
    assert script.fit_gate(old, new)["status"] == "PASS"
    new[str(script.head.raw.ROADS[1])]["positive"]["signed_bias"] = -.90000001
    report = script.fit_gate(old, new)
    assert report["status"] == "FAIL" and report["transfer_only_explanation_rejected"]
    assert report["next_mechanism"] == "objective_head_discrimination"


def test_gate_requires_nonpositive_control_on_every_road():
    old, new = _gate_metrics()
    new[str(script.head.raw.ROADS[-1])]["nonpositive"]["mae"] = .15000001
    assert script.fit_gate(old, new)["status"] == "FAIL"
    new[str(script.head.raw.ROADS[0])]["positive"]["count"] = 99
    with pytest.raises(ValueError, match="counts"):
        script.fit_gate(old, new)


def _adapted_fixture():
    parent = {"_reward.weight": torch.ones(1), "_encoder.weight": torch.tensor([0.])}
    source = {"overshoot_source": {}, "source_sha256": {}, "throughput_benchmark": {}}
    state = {"format": "haic-tdmpc2-head-only-model-v2", "protocol_sha256": script.PILOT_PROTOCOL_SHA,
             "source": {}, "source_sha256": {}, "throughput_benchmark": {}, "updates": 512,
             "seed": script.SEED, "nonhead_bitwise_parity": True,
             "model": {"_reward.weight": torch.zeros(1), "_encoder.weight": torch.tensor([0.])}}
    pilot = {"source": source, "nonhead_bitwise_parity": {
        "sha256_before": {"_encoder.weight": script._tensor_sha(parent["_encoder.weight"])},
        "changed_reward_keys": ["_reward.weight"]}}
    return parent, state, pilot


def test_adapted_nonhead_exact_bytes_including_signed_zero():
    parent, state, pilot = _adapted_fixture()
    assert script.validate_adapted(state, parent, pilot) is state["model"]
    state["model"]["_encoder.weight"] = torch.tensor([-0.])
    with pytest.raises(ValueError, match="nonhead byte parity"):
        script.validate_adapted(state, parent, pilot)


@pytest.mark.parametrize("defect", ["nan", "keys", "metadata", "no_change"])
def test_adapted_invalid_model_rejected(defect):
    parent, state, pilot = _adapted_fixture()
    if defect == "nan":
        state["model"]["_reward.weight"][:] = float("nan")
    elif defect == "keys":
        state["model"]["extra"] = torch.zeros(1)
    elif defect == "metadata":
        state["updates"] = 511
    else:
        state["model"]["_reward.weight"][:] = 1
    with pytest.raises(ValueError):
        script.validate_adapted(state, parent, pilot)


class FakeModel:
    cfg = SimpleNamespace(latent_dim=1, num_bins=101, vmin=-10., vmax=10., bin_size=.2)

    def __init__(self):
        self.draws = []

    def eval(self):
        return self

    def encode(self, obs, task=None):
        z = torch.rand(len(obs), 1, device=obs.device)
        self.draws.append(z.cpu().clone())
        return z

    def reward(self, z, action, task=None):
        return z.expand(-1, 101)

    def state_dict(self):
        return {"_reward.weight": torch.ones(1)}


def _score_fixture():
    groups = {p: {str(r): {s: [] for s in script.SIGNS} for r in script.head.raw.ROADS} for p in script.PHASES}
    episodes = []
    for p in script.PHASES:
        for r in script.head.raw.ROADS:
            for sign in script.SIGNS:
                eid = len(episodes)
                n = 257
                value = 3. if sign == "positive" else -.4
                episodes.append({"observations": np.zeros((n + 1, 4, 64, 64), np.uint8),
                                 "actions": np.zeros((n, 3), np.float32), "rewards": np.full(n, value, np.float32)})
                groups[p][str(r)][sign] = [(eid, i, i + 1, value) for i in range(n)]
    return {"episodes": episodes}, groups


def test_identical_augmentation_draws_batches_and_no_outer_rng_or_updates(monkeypatch):
    models = {name: FakeModel() for name in ("parent", "adapted")}
    replay, groups = _score_fixture()
    outer = torch.get_rng_state().clone()
    report = script.score_models(models, replay, groups)
    assert torch.equal(outer, torch.get_rng_state())
    assert report["parent"] == report["adapted"]
    assert len(models["parent"].draws) == 48
    assert all(torch.equal(a, b) for a, b in zip(models["parent"].draws, models["adapted"].draws))
    assert models["parent"].draws[0].shape == (256, 1) and models["parent"].draws[1].shape == (1, 1)
    assert not torch.equal(models["parent"].draws[0], models["parent"].draws[2])
    for p in script.PHASES:
        assert report["parent"]["ce_summaries"][p]["count"] == 2056


def test_nonfinite_reward_logits_and_label_mismatch_rejected():
    replay, groups = _score_fixture()
    models = {name: FakeModel() for name in ("parent", "adapted")}
    models["parent"].reward = lambda z, action, task=None: torch.full((len(z), 101), float("nan"))
    with pytest.raises(ValueError, match="reward logits"):
        script.score_models(models, replay, groups)
    replay["episodes"][0]["rewards"][0] = 4.
    with pytest.raises(ValueError, match="RAW pixel/action/reward/sign"):
        script.score_models({name: FakeModel() for name in ("parent", "adapted")}, replay, groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_scoring_preserves_initialized_cuda_generators():
    states = [torch.cuda.get_rng_state(i).clone() for i in range(torch.cuda.device_count())]
    replay, groups = _score_fixture()
    script.score_models({name: FakeModel() for name in ("parent", "adapted")}, replay, groups)
    assert all(torch.equal(state, torch.cuda.get_rng_state(i)) for i, state in enumerate(states))


def test_execute_preserves_failure_and_exclusive_output_without_load(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(script, "preflight", lambda *a, **k: {"schema": {}})
    monkeypatch.setattr(script, "_load", lambda *a: (_ for _ in ()).throw(ValueError("synthetic SHA drift")))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("real load"))
    with pytest.raises(ValueError, match="SHA drift"):
        script.execute(tmp_path, "a" * 64)
    result = json.loads((tmp_path / script.OUTPUT).read_text())
    assert result["status"] == "failed_no_resume"
    assert result["optimizer_updates"] == result["environment_resets"] == 0
    with pytest.raises(FileExistsError):
        script.execute(tmp_path, "a" * 64)


def test_synthetic_complete_execution_has_zero_actions_and_no_policy_release(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    replay, groups = _score_fixture()
    models = {name: FakeModel() for name in ("parent", "adapted")}
    pre = {"schema": {}, "_groups": groups}
    monkeypatch.setattr(script, "preflight", lambda *a, **k: pre)
    monkeypatch.setattr(script, "_load", lambda *a: (models, replay))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("real load"))
    report = script.execute(tmp_path, "a" * 64)
    assert report["environment_resets"] == report["optimizer_updates"] == 0
    assert report["frozen_model_bitwise_parity"] is True
    assert report["fit_gate"]["status"] == "FAIL" and report["fit_gate"]["policy_release"] is False
    assert json.loads((tmp_path / script.OUTPUT).read_text()) == report


def test_post_score_source_drift_does_not_write_complete_result(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    replay, groups = _score_fixture()
    models = {name: FakeModel() for name in ("parent", "adapted")}
    pre = {"schema": {}, "_groups": groups}

    def preflight(*args, **kwargs):
        if kwargs.get("allow_output"):
            raise ValueError("post-score source drift")
        return pre

    monkeypatch.setattr(script, "preflight", preflight)
    monkeypatch.setattr(script, "_load", lambda *a: (models, replay))
    with pytest.raises(ValueError, match="post-score source drift"):
        script.execute(tmp_path, "a" * 64)
    assert json.loads((tmp_path / script.OUTPUT).read_text())["status"] == "failed_no_resume"
