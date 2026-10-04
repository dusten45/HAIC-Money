"""Remove a legacy center confidence horizon while keeping actual body bounds."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.tests.test_predictive_control import legal_state

BASE = Path(__file__).resolve().parents[1]
V1 = BASE/'fast_predictive_agent.py'
V2 = BASE/'fast_predictive_v2_agent.py'
BUILDER = BASE/'research/speed_20261005/physics_build_predictive_v2.py'
FRAME = Path(__file__).parent/'fixtures/predictive-track3-frame40.npz'


def load(path):
    spec = importlib.util.spec_from_file_location('predictive_reference_v2_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate():
    return load(V2 if V2.exists() else V1)


def geometry(module, frame):
    agent = module.Agent()
    agent._road(frame)
    return agent, agent._predictive_geometry(frame, legal_state(100.))


def test_actual_saved_wide_road_keeps_strict_supported_prefix_beyond_31m_cap():
    frame = np.load(FRAME)['frame40']
    agent, new = geometry(candidate(), frame)
    _, previous = geometry(load(V1), frame)
    assert previous is not None and new is not None
    assert previous.path[-1, 1] == 31.
    assert new.path[-1, 1] > 34.6, 'legacy centerdepth5.8 still truncates the new body-checked reference'
    raw = candidate()._ClearRidgeReference._ridge(agent, frame)
    assert raw[-1, 1] > 36.8
    assert new.path[-1, 1] < raw[-1, 1], 'the observed unsupported image-boundary tail must remain excluded'
    np.testing.assert_array_equal(new.path[:-1], raw[:len(new.path)-1])
    assert np.min(agent._sample_distance(new.field, new.path)) >= 1.9


def test_saved_visibility_can_support_100mps_coast_and_firstblock_full_stop():
    module = candidate()
    frame = np.load(FRAME)['frame40']
    _, new = geometry(module, frame)
    planner = module.FixedControlPlanner(module.predict_step, new)
    action = np.array([0., 0., 0.], np.float32)
    rollout = planner.rollout(legal_state(100.), action)
    _, violation = planner._geometry(rollout['positions'], rollout['angles'], rollout['velocities'])
    assert violation == 0., 'the legacy center reference cap rejects an otherwise supported full-body trajectory'
    assert np.linalg.norm(rollout['velocities'][-1]) > 95.
    backup = planner._backup(rollout, action)
    assert backup['stopped'] and backup['violation'] == 0.


def test_grass_island_and_unknown_body_remain_rejected_in_raw_reference():
    module = candidate()
    frame = np.load(FRAME)['frame40']
    agent, _ = geometry(module, frame)
    island = frame.copy()
    island[44:48, 40:45] = .63
    field, unknown = agent._predictive_field(island, [])
    assert field[45, 42] == 0. and unknown[45, 42]
    agent._road(island)
    assert agent._predictive_geometry(island, legal_state(100.)) is None
    unsupported = frame.copy()
    unsupported[59:68, 37:48] = .63
    agent._road(unsupported)
    assert agent._predictive_geometry(unsupported, legal_state(100.)) is None


def test_v2_offline_builder_is_exact_and_frozen_v1_remains_unchanged():
    assert BUILDER.exists(), 'separate V2 builder has not been implemented'
    assert V2.exists()
    builder = load(BUILDER)
    assert builder.build_text(BASE.parents[1]) == V2.read_text()
    assert V2.read_bytes().startswith((BASE/'fast_rear_clear_agent.py').read_bytes())
    from hashlib import sha256
    assert sha256(V1.read_bytes()).hexdigest() == 'd2bd8a0855876f5cc4d281bd1183646ad29bd27e7f9231a4af6d917e901b844e'
