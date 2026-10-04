"""Compact matched development comparison; no environment or policy execution."""

import argparse
import json
from pathlib import Path
from typing import Any


def episode_metrics(episode) -> dict[str, Any]:
    physical = episode['physical']
    events = physical['road_reacquisition']
    returns = [event['first_contact_return_duration_s'] for event in events
               if event['first_contact_return_duration_s'] is not None]
    return dict(finished=episode['completed'], damage=episode['damage'],
                collision_decisions=episode['collisions'], hit_objects=physical['hit_objects'],
                off_track_streak=episode['off_track_count_max'],
                all_wheels_offroad_max_s=physical['no_wheel_road_contact_max_s'],
                any_wheel_offroad_max_s=physical['any_wheel_offroad_max_s'],
                min_clearance_m=physical['min_fixture_clearance'],
                road_departures=len(events), road_returns=len(returns),
                road_return_censored=len(events)-len(returns),
                max_observed_road_return_s=max(returns, default=None),
                postpass_road_return_censored=physical['road_reacquisition_censored'],
                lap_ms=episode['lapTimeMs'] if episode['completed'] else None,
                changed_decisions=episode['priority_changed_decisions'])


def analyze(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    report = json.loads((directory / 'episode-report.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    episodes = {}
    baseline_arm, candidate_arm = tuple(summary['arms'])
    for row in report['rows']:
        if row['status'] == 'completed':
            episodes[row['repeat'], row['track_id'], row['seed'], row['mode']] = json.loads((directory / row['file']).read_text())
    cells = []
    for pair in summary['pairs']:
        key = pair['repeat'], pair['track_id'], pair['seed']
        baseline = episode_metrics(episodes[*key, baseline_arm])
        candidate = episode_metrics(episodes[*key, candidate_arm])
        cells.append(dict(track=key[1], seed=key[2], repeat=key[0], baseline=baseline, candidate=candidate,
                          lost_finish=baseline['finished'] and not candidate['finished'],
                          gained_finish=candidate['finished'] and not baseline['finished'],
                          new_hits=sorted(set(candidate['hit_objects'])-set(baseline['hit_objects'])),
                          damage_harm=candidate['damage'] > baseline['damage'] + 1e-9,
                          matched=all(pair[k] for k in ('geometry_equal', 'initial_state_equal', 'initial_pixels_equal',
                                                       'prefix_observation_and_pre_state_equal'))
                                  and pair['first_changed_action_pre_state_equal'] is not False,
                          all_actions_equal=pair['all_actions_equal']))
    complete = (report['operator_error'] is None and len(cells) * 2 == len(report['rows'])
                and all(cell['matched'] for cell in cells))
    retained = not any(c['lost_finish'] or c['new_hits'] or c['damage_harm'] for c in cells)
    benefit = any(c['gained_finish'] or c['candidate']['damage'] < c['baseline']['damage'] - 1e-9
                  or c['candidate']['collision_decisions'] < c['baseline']['collision_decisions'] for c in cells)
    return dict(run=str(directory), scope=protocol['scope'], wrapper_sha256=protocol['wrapper_sha256'],
                comparator_sha256=protocol['comparator_zip_sha256'], complete=complete,
                verdict='REPEAT_CANDIDATE' if complete and retained and benefit else 'NOT_ADOPTED',
                finish_and_safety_retained=retained, finish_or_safety_benefit=benefit,
                cells=cells, aggregates=summary['arms'],
                matched_lap_pairs=summary['mutually_finished_pairs'],
                matched_lap_delta_ms=summary['mutually_finished_mean_delta_ms'],
                limitations=['Selected reused TRAIN/dev conditions, not generalization or official performance.',
                             'Road return is first ANY-wheel physical contact; max observed return is conditional, censors reported separately.',
                             'Reward streak and physical off-road duration are distinct; runtime does not observe exact reward streak.',
                             'Repeated deterministic execution is reproducibility, not additional independent roads.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    results = [analyze(run) for run in args.runs]
    with args.output.open('x') as stream:
        json.dump(dict(studies=results), stream, indent=2, allow_nan=False)
        stream.write('\n')
    for result in results:
        print(json.dumps({key: result[key] for key in ('run', 'verdict', 'aggregates', 'matched_lap_pairs', 'matched_lap_delta_ms')}))


if __name__ == '__main__':
    main()
