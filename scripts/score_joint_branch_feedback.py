"""Actual A/B/C H4 costs under their own feedback, not a fixed-tail error label."""

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from haic.algorithms.joint_control.comparison import COST_WEIGHTS
from haic.algorithms.joint_control.interval_comparison import IntervalScene, score_trajectories
from scripts import run_joint_temporal_pilot as old
from scripts.calibrate_joint_paired_envelope import clean


def actual_h4(rows, raw, step=34) -> dict[str, Any] | None:
    chosen = [row for row in rows if step <= row['step'] < step + 4]
    states = [state for state in raw if step <= state['step'] < step + 4]
    if ([r['step'] for r in chosen] != list(range(step, step + 4))
            or [r['step'] for r in states] != [s for s in range(step, step + 4) for _ in range(4)]):
        return None
    start = chosen[0]['evaluation_only']['pre']
    if any(abs(s['t'] - start['t'] - .02 * (i + 1)) > 1e-7 for i, s in enumerate(states)):
        raise ValueError('H4 raw samples do not follow the sealed .02s clock')
    c, s = math.cos(start['yaw']), math.sin(start['yaw'])
    poses = [[0., 0., 0.]] + [[c * (p['x'] - start['x']) + s * (p['y'] - start['y']),
        -s * (p['x'] - start['x']) + c * (p['y'] - start['y']),
        (p['yaw'] - start['yaw'] + math.pi) % (2 * math.pi) - math.pi] for p in states]
    actions = np.asarray([r['action'] for r in chosen], np.float32)
    if actions.shape != (4, 3) or not np.isfinite(actions).all() or not np.isfinite(poses).all():
        raise ValueError('nonfinite/incomplete actual H4')
    return dict(start=start, poses=np.asarray(poses), controls=np.repeat(actions, 4, axis=0), actions=actions)


def score_feedback(raw_scene, windows) -> dict[str, Any]:
    if any(w is None for w in windows.values()):
        return dict(common_support=False, status='CENSORED_H4', delta_intervals=None)
    if any(w['start'] != windows['A']['start'] for w in windows.values()):
        raise ValueError('actual feedback branches do not share the same pre34 state')
    boolean = {'road', 'known', 'uncertain_boundary', 'unobserved'}
    scene_values: dict[str, Any] = {f.name: (raw_scene[f.name] if f.name in ('history_used', 'motion_provenance')
        else np.asarray(raw_scene[f.name], float).reshape(-1, 4) if f.name == 'boxes'
        else np.asarray(raw_scene[f.name], dtype=bool if f.name in boolean else float)) for f in fields(IntervalScene)}
    scene = IntervalScene(**scene_values)
    labels = list(windows)
    if labels != ['A', 'B', 'C']:
        raise ValueError('ordered A/B/C windows required')
    scored = score_trajectories(scene, np.stack([w['poses'] for w in windows.values()]),
                                np.stack([w['controls'] for w in windows.values()]))
    costs = scored['reference_costs']
    supported = bool(scored['common_support'] and costs.shape == (3, 5) and np.isfinite(costs).all())
    deltas = {}
    if supported:
        for name, i, j in [('B_minus_A', 1, 0), ('C_minus_A', 2, 0), ('C_minus_B', 2, 1)]:
            delta = costs[i] - costs[j]
            deltas[name] = dict(by_reference=delta, interval=[float(delta.min()), float(delta.max())],
                weighted_components_midpoint={k: float(COST_WEIGHTS[k] * (v[i] - v[j]))
                                              for k, v in scored['components'].items()})
    return dict(status='SUPPORTED' if supported else 'UNSUPPORTED', common_support=supported,
                reference_costs=costs, supported_ticks=scored['supported_ticks'],
                delta_intervals=deltas or None, actual_actions={k: w['actions'] for k, w in windows.items()})


def run(output, digest) -> dict[str, Any]:
    from scripts import run_joint_single_branch as branch
    protocol = branch.validate_frozen(output, digest)
    summary_path, report_path = output / 'summary.json', output / 'episode-report.json'
    summary, report = old.read_json(summary_path), old.read_json(report_path)
    if (summary.get('protocol_sha256') != digest or not summary.get('matched_ABC')
            or not summary.get('policy_valid') or summary.get('operator_error') is not None
            or report.get('protocol_sha256') != digest or report.get('operator_error') is not None
            or len(report.get('rows', [])) != 1):
        raise ValueError('matched policy-valid sealed ABC summary required before scoring')
    record = report['rows'][0]
    b_path = output / protocol['schedule'][0]['file']
    if (record.get('file') != b_path.name or record.get('status') != 'completed'
            or record.get('sha256') != old.sha(b_path)
            or report.get('artifacts_sha256', {}).get(b_path.name) != old.sha(b_path)):
        raise ValueError('B episode is not bound by its completed report')
    old.check_pins({str(old.local_path(output, name)): digest for name, digest in report['artifacts_sha256'].items()})
    gate = protocol['gate_binding']
    refs = gate['references']
    episodes = {'A': (Path(refs['A']['root']), old.read_json(refs['A']['episode_path'])),
                'B': (output, old.read_json(b_path)),
                'C': (Path(refs['C']['root']), old.read_json(refs['C']['episode_path']))}
    windows = {label: actual_h4(old.decisions(root, ep), old.raw_states(root, ep)) for label, (root, ep) in episodes.items()}
    anchor_path = Path(gate['fixed_h4']['anchor_path'])
    anchor = old.read_json(anchor_path)
    result = score_feedback(anchor['scene'], windows)
    result.update(schema='haic-joint-single-feedback-cost-v1',
        interpretation='Actual closed-feedback H4 costs in the same observed scene. NOT an error against fixed-tail predictions or a lap-time guarantee. No simulator/predictor execution, refit, support truncation or privileged-road substitution.',
        source_sha256={str(Path(__file__).resolve()): old.sha(Path(__file__))},
        input_sha256={str(output / 'protocol.json'): digest, str(anchor_path): old.sha(anchor_path),
                      **{str(path): old.sha(path) for path in (summary_path, report_path, b_path)}},
        fixed_tail_baseline_actions=gate['fixed_h4']['baseline_tail'],
        fixed_tail_alternative_actions=gate['fixed_h4']['alternative_tail'])
    result = cast(dict[str, Any], clean(result))
    old.save(output / 'feedback-cost.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), args.protocol_sha256), allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
