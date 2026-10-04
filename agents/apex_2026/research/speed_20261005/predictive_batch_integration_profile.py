"""Three fixed existing-camera action profiles; no simulator or new lap."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


PARENT_SHA = 'b70af66e0ba39ea975d53909ec6d7a19df2a73005a036465d71dd7ac8caeff8a'
CANDIDATE_SHA = '93a6b4db3484a2621407a410af64fea6f88465cb49173f6c8234e94e5c240547'
HISTORY_SHA = 'fe51315481583212c889d12801e6b6b585574664df3217fbc8977a74ad0d0e78'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location('batch_standalone_profile', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(module, history, step):
    agent = module.Agent()
    for index in range(step):
        observation = history['observations'][index]
        module._BaseRearClearAgent.act(agent, observation)
        agent._observer.observe(agent._frame(observation))
        action = history['actions'][index].copy()
        agent._observer.advance(action)
        agent._predictive_previous_action = action.copy()
        agent.last_steer = float(action[0])
    return agent


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    return value


def equal(left, right):
    for key, value in right.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(left[key], value)
        elif isinstance(value, float):
            np.testing.assert_allclose(left[key], value, atol=1e-10, rtol=1e-12)
        else:
            assert left[key] == value, key


def main():
    base = Path('agents/apex_2026')
    output = base/'results/speed-20261005/predictive-batch-integration-v1-profile.json'
    assert not output.exists()
    sources = {'scalar_parent': base/'fast_predictive_v2_agent.py',
               'batch_candidate': base/'fast_predictive_batch_agent.py',
               'profile': Path(__file__)}
    before = {key: digest(path) for key, path in sources.items()}
    assert before['scalar_parent'] == PARENT_SHA
    assert before['batch_candidate'] == CANDIDATE_SHA
    history_path = base/'tests/fixtures/predictive-batch-history.npz'
    assert digest(history_path) == HISTORY_SHA
    history = np.load(history_path)
    modules = {key: load(sources[key]) for key in ('scalar_parent', 'batch_candidate')}
    cases = []
    for step in (20, 40, 60):
        agents = {key: prepare(module, history, step) for key, module in modules.items()}
        outputs, timings = {}, {}
        before_observation = history['observations'][step].copy()
        for key, agent in agents.items():
            observation = before_observation.copy()
            started = time.perf_counter()
            outputs[key] = agent.act(observation)
            timings[key] = time.perf_counter()-started
            np.testing.assert_array_equal(observation, before_observation)
            assert agent._observer.actions_seen == step+1
        scalar, batch = agents['scalar_parent'], agents['batch_candidate']
        np.testing.assert_array_equal(outputs['batch_candidate'], outputs['scalar_parent'])
        assert batch.predictive_status == scalar.predictive_status
        equal(batch.predictive_result, scalar.predictive_result)
        equal(batch.predictive_sensor, scalar.predictive_sensor)
        equal(batch._observer.state, scalar._observer.state)
        if batch.predictive_result:
            assert batch.predictive_model_calls == batch.predictive_result['batched_model_rows']
        cases.append({'step': step, 'status': batch.predictive_status,
                      'action': outputs['batch_candidate'].tolist(), 'elapsed_act_s': timings,
                      'act_speedup': timings['scalar_parent']/timings['batch_candidate'],
                      'scalar_model_row_predictions': scalar.predictive_model_calls,
                      'batch_model_row_predictions': batch.predictive_model_calls,
                      'scalar_plan': serial(scalar.predictive_result),
                      'batch_plan': serial(batch.predictive_result),
                      'sensor': serial(batch.predictive_sensor),
                      'every_scalar_plan_field_matched': True,
                      'observer_state_after_final_action_equal': True,
                      'past_legal_actions_seen_before_target': step})
    assert before == {key: digest(path) for key, path in sources.items()}
    result = {'classification': 'three_fixed_existing_camera_causal_standalone_profiles_not_lap',
              'source_sha256': before, 'history_sha256': HISTORY_SHA,
              'all_sources_unchanged': True, 'world_steps': 0, 'new_episodes': 0,
              'new_holdout_opened': False, 'truth_for_initialization': False,
              'old_candidate_logged_history_not_new_candidate_trajectory': True,
              'cases': cases,
              'limits': ['One fixed measurement per case on this shared host; no worst-case latency bound.',
                         'Timers include complete act: parent perception, field, ridge, scenarios, rollout/backup geometry and final scalar observer advance.',
                         'Warmup history and module loading are outside the action timers.',
                         'Finite output equivalence assumes successful planning within the fixed deadline.',
                         'Batch eager geometry has distinct error behavior for scalar-unvisited invalid rows.',
                         'No new driving, pace improvement, adoption or safety certificate is claimed.']}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
