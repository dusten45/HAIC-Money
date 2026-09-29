import numpy as np
import pytest

from scripts import collect_rlpd_coupled_recovery as c


def fixture(n=100):
    return {"reward": np.zeros(n), "damage": np.zeros(n+1), "progress": np.linspace(0, .1, n+1),
            "center_error": np.zeros(n+1), "episode_end": np.zeros(n, bool), "finished": np.zeros(n, bool)}


def test_native_executed_coordinates():
    np.testing.assert_array_equal(c.native([.2, 0, 1]), np.array([.2, -1, 1], np.float32))
    with pytest.raises(c.diagnosis.DiagnosisError):
        c.native([0, -1, 0])


@pytest.mark.parametrize("info", [{"finished": True}, {"retire_reason": "off_track"}, {"retire_reason": "crash"}])
def test_finished_or_retired_even_if_truncated_is_terminal(info):
    assert c.terminal_flag(False, info)
    assert not c.terminal_flag(False, {})


def test_qualification_requires_full_followup():
    assert c.qualify(fixture(), 10, 12)["qualified"]
    assert not c.qualify(fixture(70), 10, 12)["qualified"]


@pytest.mark.parametrize("key,value", [("center_error", 6.01), ("damage", .2), ("progress", 0)])
def test_recovery_safety_gates(key, value):
    a = fixture()
    a[key][85] = value
    assert not c.qualify(a, 10, 12)["qualified"]


def test_early_failure_rejected():
    a = fixture()
    a["episode_end"][40] = True
    assert not c.qualify(a, 10, 12)["qualified"]


def test_current_prefix_pixel_parity():
    obs = np.zeros((4, 84, 84), np.float32)
    ref = {"observation_sha256": ["bad"]}
    with pytest.raises(c.diagnosis.DiagnosisError, match="pixel"):
        c.parity(obs, {}, ref, 0)


def test_frozen_selection():
    cases, _ = c.select_cases()
    assert len(cases) == 14
    assert [(x["policy_id"], x["geometry_seed"]) for x in cases[:10]] == c.branch.SAMPLES
    assert all(x["stratum"] == "finish-control" and x["replay_finished"] for x in cases[10:])


def test_closed_loop_full_actions_and_reconstructed_stacks(monkeypatch):
    from types import SimpleNamespace
    import haic.oracle_v1

    class Env:
        def __init__(self):
            self.t = 0
            self.obs = np.zeros((4, 84, 84), np.float32)
            self.actions = []
            self.action_space = SimpleNamespace(contains=lambda a: True)

        def step(self, action):
            self.actions.append(action.copy())
            self.t += 1
            self.obs = np.concatenate((self.obs[1:], np.full((1, 84, 84), self.t/255, np.float32)))
            return self.obs, 1., self.t == 90, False, {"finished": False}

        def close(self):
            pass

    envs = []
    def make(*args):
        env = Env()
        envs.append(env)
        return env, env, env.obs

    def capture(base, controller, env):
        return {key: (np.array([0., 0.]) if key == "position" else np.array([-.5, 0., 1.], np.float32)
                      if key == "oracle_action" else env.t/1000 if key == "progress" else 0.)
                for key in ("position", "speed", "heading", "center_error", "path_error", "heading_error",
                            "curvature", "arc_length", "progress", "damage", "off_track_counter", "oracle_action")}

    calls = []
    actor = SimpleNamespace(reset=lambda obs: None, act=lambda obs: calls.append(float(obs[-1, 0, 0])) or np.array([.8, 1., 0.], np.float32))
    monkeypatch.setattr(c.diagnosis, "_make_env", make)
    monkeypatch.setattr(c.branch, "_capture", capture)
    monkeypatch.setattr(c.diagnosis, "_track_hash", lambda base: "hash")
    monkeypatch.setattr(haic.oracle_v1, "OracleController", lambda *a, **k: None)
    case = {"geometry_seed": 4272000001, "anchor_step": 5}
    _, reference = c.collect(case, actor, 0)
    summary, data = c.collect(case, actor, 12, reference)
    assert len(calls) == 180
    assert summary["accepted_transitions"] == 12
    np.testing.assert_array_equal(data["applied_action"][5:17], np.tile([-.5, 0, 1], (12, 1)))
    np.testing.assert_array_equal(data["applied_action"][17:], reference["applied_action"][17:])
    assert data["frames"].shape == (91, 84, 84)
    np.testing.assert_array_equal(data["final_stack"], data["frames"][-4:])
    assert data["episode_end"][-1] and data["terminal"][-1]
    assert not data["recovery_mask"][:5].any() and not data["recovery_mask"][17:].any()


def test_prefix_state_mismatch_rejected():
    obs = np.zeros((4, 84, 84), np.float32)
    ref = {"observation_sha256": [c.diagnosis._obs_hash(obs)], "position": [np.array([1., 0.])]}
    with pytest.raises(c.diagnosis.DiagnosisError, match="state"):
        c.parity(obs, {"position": np.zeros(2)}, ref, 0)
