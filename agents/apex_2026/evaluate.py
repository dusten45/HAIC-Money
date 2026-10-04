"""One official, unchanged simulator episode per process.

Run from repository root: python -m agents.apex_2026.evaluate --agent PATH
--track-id 1 --seed 42 --output RESULT.json [--config '{"key": 1}'].
Telemetry is evaluator-only and is never supplied to the policy. A started/error
receipt is an incomplete evaluation, never an observed DNF.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from threading import Thread
import time
import traceback
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAX_STEPS = 2000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_receipt(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_agent(path, config=None):
    """Use a fresh process for each call to avoid sibling-module cache collisions."""
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    name = '_apex_policy_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Agent(**(config or {}))


def safe_act(agent, observation, timeout_sec=5.0):
    """Match local_runner.safe_act without importing the root policy."""
    outcome = []
    def call():
        try:
            outcome.append(np.asarray(agent.act(observation), dtype=np.float32).reshape(-1))
        except Exception:
            outcome.append(None)
    thread = Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive() or not outcome or outcome[0] is None:
        return np.zeros(3, dtype=np.float32), False
    action = outcome[0]
    if action.shape != (3,) or not np.all(np.isfinite(action)):
        return np.zeros(3, dtype=np.float32), False
    return np.clip(action, [-1., 0., 0.], [1., 1., 1.]), True


def safe_reset(agent, observation, timeout_sec=5.0):
    if not hasattr(agent, 'reset'):
        return
    errors = []
    def call():
        try:
            agent.reset(observation)
        except Exception as error:
            errors.append(error)
    thread = Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError('agent.reset() timed out')
    if errors:
        raise errors[0]


def make_environment():
    from gymnasium.wrappers import TimeLimit
    from core.vendor.car_racing import CarRacing
    from env_wrapper import CarEnvironment
    return CarEnvironment(TimeLimit(CarRacing(continuous=True, render_mode=None),
                                    max_episode_steps=8200), skip_frames=4)


def provenance(agent_path):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for package in ('numpy', 'gymnasium', 'Box2D', 'opencv-python', 'pygame', 'torch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    official_files = sorted((ROOT / 'core').rglob('*.py'))
    official_files += [ROOT / 'env_wrapper.py', ROOT / 'damage.py', ROOT / 'local_runner.py']
    return {'git_commit': commit, 'python': platform.python_version(),
            'packages': versions, 'agent_path': str(agent_path),
            'agent_sha256': sha256(agent_path),
            'evaluator_sha256': sha256(__file__),
            'official_source_sha256': {str(p.relative_to(ROOT)): sha256(p)
                                       for p in official_files if p.is_file()}}


def runtime_hashes():
    """Hash loaded project/policy Python sources, including lazy imports."""
    hashes = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == '.py' and path.is_file() and 'site-packages' not in path.parts:
            hashes[str(path)] = sha256(path)
    return hashes


def diagnostics(env, info):
    raw = env.unwrapped
    row = {'sim_time_s': float(raw.t)}
    hull = getattr(getattr(raw, 'car', None), 'hull', None)
    if hull is not None:
        velocity = hull.linearVelocity
        row.update(position=[float(hull.position[0]), float(hull.position[1])],
                   speed_m_s=math.hypot(float(velocity[0]), float(velocity[1])),
                   heading_rad=float(hull.angle), angular_velocity=float(hull.angularVelocity))
    row['info'] = {key: value.item() if isinstance(value, np.generic) else value
                   for key, value in info.items()
                   if value is None or isinstance(value, (str, int, float, bool, np.generic))}
    return row


def save_frame(directory, step, observation):
    # The policy-visible final grayscale frame is sufficient for sparse diagnosis.
    import cv2
    directory.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.asarray(observation)[-1] * 255, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(directory / f'{step:04d}.png'), image):
        raise OSError('could not write diagnostic frame')


def run_episode(agent_path, track_id, seed, output, *, config=None, trace=None,
                frames=None, env_factory=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    agent_path = Path(agent_path).resolve()
    receipt = {'schema_version': 1, 'status': 'started', 'started_at': timestamp(),
               'track_id': track_id, 'seed': seed, 'config': config or {},
               'agent_path': str(agent_path), 'max_steps': MAX_STEPS,
               'skip_frames': 4, 'raw_frame_budget': 8200, 'steps': 0}
    # Exclusive initial creation preserves partial prior attempts.
    with output.open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    wall_start = time.perf_counter()
    env = None
    trace_handle = None
    inference = []
    try:
        receipt['provenance'] = provenance(agent_path)
        write_receipt(output, receipt)
        if trace:
            trace = Path(trace)
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace.open('x')
        agent = load_agent(agent_path, config)
        receipt['provenance']['runtime_source_sha256_before'] = runtime_hashes()
        write_receipt(output, receipt)
        env = (env_factory or make_environment)()
        observation, info = env.reset(seed=seed, options={'track_id': track_id})
        start_t = float(env.unwrapped.t)
        receipt['start_t'] = start_t
        safe_reset(agent, copy.deepcopy(observation))
        total_reward = 0.
        invalid_actions = invalid_streak = collisions = 0
        terminated = truncated = False
        retire_reason = None
        if frames:
            save_frame(Path(frames), 0, observation)
        for _ in range(MAX_STEPS):
            policy_observation = copy.deepcopy(observation)
            inference_start = time.perf_counter()
            action, valid = safe_act(agent, policy_observation)
            inference.append(time.perf_counter() - inference_start)
            invalid_actions += int(not valid)
            invalid_streak = 0 if valid else invalid_streak + 1
            if invalid_streak >= 10:
                retire_reason = 'invalid_action'
                break
            before = diagnostics(env, info) if trace_handle else None
            observation, reward, terminated, truncated, info = env.step(action)
            receipt['steps'] += 1
            total_reward += float(reward)
            collisions += int(bool(info.get('collision', False)))
            if trace_handle:
                row = {'step': receipt['steps'], 'action': action.tolist(), 'valid': valid,
                       'inference_s': inference[-1], 'reward': float(reward),
                       'before': before, 'after': diagnostics(env, info)}
                trace_handle.write(json.dumps(row, allow_nan=False) + '\n')
                trace_handle.flush()
            if frames and (receipt['steps'] % 100 == 0 or terminated or truncated):
                save_frame(Path(frames), receipt['steps'], observation)
            if terminated or truncated:
                break
        raw = env.unwrapped
        finish_time = raw.finish_time_s
        finished = finish_time is not None
        retire_reason = retire_reason or info.get('retire_reason')
        if not finished and retire_reason is None:
            retire_reason = 'off_track' if terminated else 'max_steps'
        receipt.update(status='completed', finished=finished,
                       finish_time_s=finish_time, finish_qualified=raw.finish_qualified_time_s is not None,
                       finish_qualified_time_s=raw.finish_qualified_time_s,
                       lapTimeMs=round((finish_time - start_t) * 1000) if finished else None,
                       progress=float(env._calculate_progress()), damage=float(info.get('damage', 0.)),
                       collision_steps=collisions, invalid_actions=invalid_actions,
                       retire_reason=retire_reason, terminated=bool(terminated), truncated=bool(truncated),
                       total_reward=total_reward, end_t=float(raw.t))
    except BaseException as error:
        receipt.update(status='error', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
        raise
    finally:
        if trace_handle:
            trace_handle.close()
        try:
            if env is not None:
                env.close()
        finally:
            receipt.update(ended_at=timestamp(), wall_time_s=time.perf_counter() - wall_start,
                           inference_calls=len(inference),
                           inference_mean_s=float(np.mean(inference)) if inference else None,
                           inference_max_s=max(inference) if inference else None)
            if 'provenance' in receipt:
                receipt['provenance']['runtime_source_sha256_after'] = runtime_hashes()
            write_receipt(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', type=Path, required=True)
    parser.add_argument('--config', default='{}', help='JSON object or path to JSON file')
    parser.add_argument('--track-id', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--frames', type=Path)
    args = parser.parse_args()
    if args.config.lstrip().startswith('{'):
        config = json.loads(args.config)
    else:
        config = json.loads(Path(args.config).read_text())
    if not isinstance(config, dict):
        parser.error('--config must contain a JSON object')
    result = run_episode(args.agent, args.track_id, args.seed, args.output,
                         config=config, trace=args.trace, frames=args.frames)
    print(json.dumps({key: result[key] for key in
                      ('status', 'finished', 'lapTimeMs', 'progress', 'damage', 'steps')}))


if __name__ == '__main__':
    main()
