"""Fixed 216-row scalar/batch model profiles; no worlds, maps or laps."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_four_tire_model import predict_step
from agents.apex_2026.research.speed_20261005.predictive_batch_model import pack_states, predict_batch
from agents.apex_2026.research.speed_20261005.predictive_control import FixedControlPlanner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supplied_state(speed):
    center = np.array([0., -.0804591735470409])
    return {'angle': 0., 'velocity': np.array([0., float(speed)]), 'yaw': 0.,
            'mass': 7.3019199669361115, 'inertia': 19.170645977936854,
            'local_center': center, 'wheel_offsets': np.array(
                [[-1.1, 1.6], [1.1, 1.6], [-1.1, -1.64], [1.1, -1.64]])-center,
            'joint': np.zeros(4), 'omega': np.full(4, speed/.54),
            'gas': np.zeros(4), 'wheel_velocity_override': None}


def scalar_trace(initial, commands, steps, backup):
    states = [{name: value.copy() if isinstance(value, np.ndarray) else value
               for name, value in state.items()} for state in initial]
    commands = commands.copy()
    rows = []
    started = time.perf_counter()
    for tick in range(steps):
        if backup and tick == 4:
            commands[:, 0] = [np.mean(state['joint'][:2]) for state in states]
            commands[:, 1], commands[:, 2] = 0., 1.
        outputs = [predict_step(state, [float(value) for value in command])
                   for state, command in zip(states, commands)]
        states = [output[0] for output in outputs]
        rows.append((states, [output[1] for output in outputs]))
    return rows, 1000*(time.perf_counter()-started)


def batch_trace(initial, commands, steps, backup):
    started = time.perf_counter()
    state = pack_states(initial)
    pack_ms = 1000*(time.perf_counter()-started)
    commands = commands.copy()
    rows = []
    started = time.perf_counter()
    for tick in range(steps):
        if backup and tick == 4:
            commands[:, 0] = np.mean(state['joint'][:, :2], axis=1)
            commands[:, 1], commands[:, 2] = 0., 1.
        state, diagnostics = predict_batch(state, commands)
        rows.append((state, diagnostics))
    return rows, 1000*(time.perf_counter()-started), pack_ms


def main():
    output = Path('agents/apex_2026/results/speed-20261005/predictive-batch-v1-profile.json')
    assert not output.exists()
    sources = {'batch': Path(__file__).with_name('predictive_batch_model.py'),
               'scalar': Path(__file__).with_name('physics_four_tire_model.py'),
               'candidate_actions': Path(__file__).with_name('predictive_control.py'),
               'profile': Path(__file__)}
    before = {name: digest(path) for name, path in sources.items()}
    assert before['scalar'] == '84a91143f8d42db7cf1007586c5c13038c6e0e6d371624f262abcaba26e46064'
    # One fixed action set and three different supplied legal rolling states.
    # This is a kernel workload, not a policy/uncertainty parameter study.
    legal_actions = FixedControlPlanner(None, None).candidate_actions([0., 0., 0.])
    states = [supplied_state(speed) for _ in legal_actions for speed in (0., 70., 100.)]
    commands = np.repeat(np.asarray(legal_actions, float), 3, axis=0)
    commands[:, 0] *= -1.
    assert len(states) == len(commands) == 216
    cases = []
    for name, steps, backup in (('constant_control_16_ticks', 16, False),
                                ('first4_then_held_joint_locked_brake32', 36, True)):
        scalar, scalar_ms = scalar_trace(states, commands, steps, backup)
        batch, batch_ms, pack_ms = batch_trace(states, commands, steps, backup)
        max_state, max_diagnostic = 0., 0.
        for (state_rows, scalar_diagnostics), (packed, diagnostics) in zip(scalar, batch):
            expected = pack_states(state_rows)
            for field in expected:
                if field == 'wheel_velocity_override':
                    continue
                np.testing.assert_allclose(packed[field], expected[field], atol=1e-10, rtol=1e-12)
                max_state = max(max_state, float(np.max(abs(packed[field]-expected[field]))))
            for field, value in diagnostics.items():
                expected = np.asarray([row[field] for row in scalar_diagnostics])
                np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-12)
                max_diagnostic = max(max_diagnostic, float(np.max(abs(value-expected))))
        cases.append({'name': name, 'rows': len(states), 'raw_ticks': steps,
                      'equivalent_scalar_predictions': len(states)*steps,
                      'scalar_ms': scalar_ms, 'batch_ms': batch_ms,
                      'initial_batch_pack_ms': pack_ms, 'kernel_speedup': scalar_ms/batch_ms,
                      'max_state_abs_error': max_state, 'max_diagnostic_abs_error': max_diagnostic})
    assert before == {name: digest(path) for name, path in sources.items()}
    report = {'classification': 'fixed_uniform_asphalt_supplied_state_batch_kernel_profile',
              'source_sha256': before, 'all_sources_unchanged': True,
              'world_steps': 0, 'new_holdout_opened': False, 'cases': cases,
              'limits': ['Kernel speedup excludes camera geometry, observer and policy selection.',
                         'Uniform400N asphalt predictor only; no grass, damage or contact certification.',
                         'Non-None privileged wheel velocity override is outside supported scope.',
                         'Latency measures this host and fixed workload only, not official worst-case timing.',
                         'No planner or frozen standalone agent was modified or lap-tested.']}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
