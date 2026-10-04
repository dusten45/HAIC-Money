"""Read recorded P1 traces only; no environment/policy import and no resets.

Run from repository root with .venv/bin/python <this file> --output PATH.
Budgets are descriptive proxies, not a proof that a faster legal lap exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(path):
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'completed' and receipt['finish_time_s'] is not None
    trace = path.with_suffix('.jsonl')
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    budget, opportunity, lateral_sum = {}, {}, {}
    metrics = {k: 0.0 for k in ['distance_m', 'braking_s', 'coasting_s',
        'braking_below_target_minus1_s', 'target_gap_gt10_low_gas_s',
        'visible_horizon_lt20m_s', 'target_ge99_9_s', 'target_le60_s',
        'recent_steer_reversal_s', 'target_drop_gt10_s', 'unknown_flow_s']}
    launch, previous_target, last_sign, last_reversal = True, None, 0, -1e9
    reversals, target_error, brake_entries = 0, 0., 0
    previous_brake = False
    for row in rows:
        b, a, d = row['before'], row['after'], row['policy_diagnostics']
        dt = max(0., min(a['sim_time_s'], receipt['finish_time_s']) - b['sim_time_s'])
        if not dt:
            continue
        p = np.asarray(d['path'], dtype=np.float32)
        distance = 63 - p[:, 1]
        kappas = np.zeros(len(p))
        for i in range(len(p)):
            lo, hi = max(0, i-3), min(len(p), i+4)
            if hi-lo >= 4:
                poly = np.polyfit(distance[lo:hi]/1.701, p[lo:hi, 0]/1.3608, 2)
                slope = 2*poly[0]*distance[i]/1.701 + poly[1]
                kappas[i] = abs(2*poly[0])/(1+slope*slope)**1.5
        local_cap = np.clip(np.sqrt(100/np.maximum(kappas, .001)), 24, 100)
        caps = {'max_speed': 100.,
                'path_braking_curvature': float(np.sqrt(local_cap**2+130*distance/1.701).min()),
                'tracking_correction': float(np.sqrt(100/max(abs(d['curvature']), .001))),
                'short_horizon': 32. if distance[-1] < 18 else 1e6}
        target = d['target_speed']
        target_error = max(target_error, abs(target-max(24, min(caps.values()))))
        binding = min(caps, key=caps.get)
        v = b['speed_m_s']
        if v >= 60:
            launch = False
        if launch:
            binding = 'launch_to_first60'
        ds = float(np.linalg.norm(np.asarray(a['position'])-b['position']))*dt/(a['sim_time_s']-b['sim_time_s'])
        budget[binding] = budget.get(binding, 0.) + dt
        opportunity[binding] = opportunity.get(binding, 0.) + max(0., dt-ds/100)
        lateral_sum[binding] = lateral_sum.get(binding, 0.) + abs(v*b['angular_velocity'])*dt
        steer, gas, brake = row['action']
        sign = int(np.sign(steer)) if abs(steer) >= .03 else 0
        if sign and last_sign and sign != last_sign:
            reversals += 1
            last_reversal = b['sim_time_s']
        if sign:
            last_sign = sign
        braking = brake > .01
        brake_entries += int(braking and not previous_brake)
        previous_brake = braking
        flags = {'braking_s': braking, 'coasting_s': gas < .01 and brake < .01,
            'braking_below_target_minus1_s': braking and v < target-1,
            'target_gap_gt10_low_gas_s': target-v > 10 and gas < .55 and brake < .01,
            'visible_horizon_lt20m_s': distance[-1]/1.701 < 20,
            'target_ge99_9_s': target >= 99.9, 'target_le60_s': target <= 60,
            'recent_steer_reversal_s': b['sim_time_s']-last_reversal < .24-1e-6,
            'target_drop_gt10_s': previous_target is not None and previous_target-target > 10,
            'unknown_flow_s': not d.get('flow_valid', False)}
        for key, flag in flags.items():
            metrics[key] += dt*flag
        metrics['distance_m'] += ds
        previous_target = target
    lap = receipt['lapTimeMs']/1000
    assert abs(sum(budget.values())-lap) < 1e-6
    assert target_error < .005, target_error
    instant = metrics['distance_m']/100
    return {'track_id': receipt['track_id'], 'seed': receipt['seed'],
        'receipt': str(path), 'receipt_sha256': sha(path), 'trace': str(trace),
        'trace_sha256': sha(trace), 'started_at': receipt['started_at'],
        'source_sha256': receipt['provenance']['agent_sha256'],
        'lap_s': lap, 'required_saving_to13_s': lap-13, 'required_saving_to10_s': lap-10,
        'binding_budget_s': budget,
        'binding_share': {k: v/lap for k,v in budget.items()},
        'optimistic_same_path_instant100_saving_s': opportunity,
        'mean_abs_speed_times_yaw_m_s2_by_binding': {k: lateral_sum[k]/v for k,v in budget.items()},
        'same_observed_path_instant100_time_s': instant,
        'remaining13s_budget_after_instant100_and_observed_launch_penalty_s': 13-instant-opportunity['launch_to_first60'],
        'overlapping_metrics': metrics, 'significant_steer_reversals': reversals,
        'brake_entries': brake_entries, 'max_reconstructed_target_error_m_s': target_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('/tmp/apex-final-frozen/required'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    episodes = [analyze(p) for p in sorted(args.input_dir.glob('t*.json'))]
    assert len(episodes) == 4
    probes = []
    for name in ['actuator', 'brake', 'force', 'predictive', 'shield', 'geodesic']:
        path = ROOT/'agents/apex_2026/v2/results'/f'{name}-experiments.json'
        if path.exists():
            obj = json.loads(path.read_text())
            probes.append({'ledger': str(path.relative_to(ROOT)), 'sha256': sha(path),
                'variant': obj.get('variant'), 'hypothesis': obj.get('hypothesis'),
                'required_summary': obj.get('required_summary'),
                'source_sha256': obj.get('source_sha256')})
    probe_receipts = []
    required_cells = {(e['track_id'], e['seed']) for e in episodes}
    for variant in ['actuator-r0', 'brake-r0', 'force-r0', 'predictive-r0', 'shield-r1', 'geodesic-r1']:
        for path in sorted(Path('/tmp/apex-v2-'+variant).glob('t*.json')):
            r = json.loads(path.read_text())
            if (r.get('track_id'), r.get('seed')) not in required_cells:
                continue
            item = {'variant': variant, 'receipt': str(path), 'sha256': sha(path),
                'track_id': r['track_id'], 'seed': r['seed'], 'status': r['status'],
                'lapTimeMs': r.get('lapTimeMs'), 'finished': r.get('finish_time_s') is not None,
                'source_sha256': r.get('provenance', {}).get('agent_sha256')}
            trace = path.with_suffix('.jsonl')
            if trace.exists() and r['status'] == 'completed':
                duration, braking, integral_v, lowtarget = 0., 0., 0., 0.
                for line in trace.read_text().splitlines():
                    row = json.loads(line)
                    b, a = row['before'], row['after']
                    end = r.get('finish_time_s') or a['sim_time_s']
                    dt = max(0., min(a['sim_time_s'], end)-b['sim_time_s'])
                    duration += dt
                    braking += dt*(row['action'][2] > .01)
                    integral_v += dt*b['speed_m_s']
                    lowtarget += dt*(row.get('policy_diagnostics', {}).get('target_speed', float('inf')) <= 60)
                item.update(trace_sha256=sha(trace), recorded_duration_s=duration,
                    braking_s=braking, mean_preaction_speed_m_s=integral_v/max(duration, 1e-9),
                    target_le60_s=lowtarget)
            probe_receipts.append(item)
    result = {'schema_version': 1, 'scope': 'read-only required P1 and existing v2 ledgers; zero resets',
        'analysis_source_sha256': sha(Path(__file__)), 'episodes': episodes, 'v2_required_receipt_snapshots': probe_receipts, 'v2_ledger_snapshots': probes,
        'definitions': {
            'binding_budget': 'Reconstruct exact P1 recorded-path speed caps; smallest cap wins, except initial interval before first true speed >=60m/s classified launch. These exclusive durations sum to official finish time.',
            '100_counterfactual': 'Observed position chord distance at instantaneous100m/s; maximum numerical speed is approximately100 from Box2D2m translation/20ms. No acceleration, corner, obstacle, or steering constraints. NOT an achievable lap or global optimal lower bound; a different route can have different distance.',
            'opportunity': 'Sum max(0,dt-distance/100), attributed by current binding category. An optimistic descriptive ceiling on same-path delay, NOT removable causal loss.',
            'oscillation_proxy': 'Sign reversals between nontrivial steering commands abs(steer)>=.03, with .24s following marked. Legitimate alternating bends also count; not proof of instability.',
            'overlap': 'Braking/coasting/short visible horizon/low gas/unknown flow metrics overlap cap categories and each other. Braking below target does not prove unnecessary braking under wheel or traction state constraints.',
            'lateral_proxy': 'abs(true preaction speed * true preaction hull angular velocity); privileged diagnostic, not exact tire force and never runtime input.',
            'v2_comparison': 'Ledger snapshots retain original hypotheses and required summaries. Failed/missing laps cannot be treated as fast successful evidence; no new experiments.'},
        'finding': 'Most time binds geometric path braking or tracking curvature, not maximum speed. Correction-limited intervals exhibit roughly100m/s2 lateral demand, so removing their cap is not justified. Recorded below-target braking is too brief to explain required savings. Same-route13s on track2 leaves about0.20s for all nonlaunch losses after an instantaneous100m/s idealization.',
        'single_mechanical_hypothesis': {
            'claim': 'A pose-anchored temporally persistent, curvature-continuous and steering-rate-feasible racing line can reduce transient tracking curvature and repeated bend entry/exit deceleration while preserving road and obstacle clearance; replacing row-wise paths alone is insufficient.',
            'controlled_test': 'Hold100m/s top speed,100m/s2 lateral budget,65m/s2 braking budget and pedal allocator fixed. Replace only path/trajectory planning with a time-parameterized curvature-continuous feasible line using pixel-derived pose/wheel state. Compare all consumed regression cells, not just two rescues.',
            'mechanical_prediction': 'At matched bend locations, reduced demanded and measured peak curvature, fewer short-interval steer reversals and brake reentries, and higher minimum bend/exit speed without additional offroad time, damage, or DNF. Require reductions in actual lap time and correction-limited delay, not only cap relabeling.',
            'falsification': 'If geometric/actual curvature and repeated braking do not fall, or any preserved completion is lost, reject claimed mechanism. Geodesic r1 already demonstrates that better topological recovery alone can be slower and lose other cells.',
            'limit': 'This is an untested architecture hypothesis, not evidence that10–13s is attainable. Existing same-route budgets make13s on track2 exceptionally demanding; pedal-only tweaks cannot plausibly explain7.68s.'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{'track': e['track_id'], 'lap_s': e['lap_s'], 'budget': e['binding_budget_s'], 'reversals': e['significant_steer_reversals'], 'brake_entries': e['brake_entries']} for e in episodes], indent=2))


if __name__ == '__main__':
    main()
