"""Research-only evidence and candidate fingerprint checks; never packaged."""
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median


def _signature(row):
    provenance = row.get('provenance', {})
    source = provenance.get('agent_sha256')
    if not source:
        raise ValueError('candidate source fingerprint is missing')
    before = provenance.get('runtime_source_sha256_before', {})
    after = provenance.get('runtime_source_sha256_after', {})
    if row.get('status') == 'completed' and (not before or not after):
        raise ValueError('candidate source inventories are missing')
    if row.get('status') == 'completed' and before.keys() - after.keys():
        raise ValueError('loaded candidate/runtime source changed or disappeared')
    for path in before.keys() & after.keys():
        if before[path] != after[path]:
            raise ValueError(f'loaded candidate/runtime source changed: {path}')
    # Include lazy imports and dependencies outside the entry point directory.
    # Paths intentionally bind a comparison to the same execution environment.
    dependencies = {**before, **after}
    if row.get('agent_path'):
        for inventory in (before, after):
            known = inventory.get(str(Path(row['agent_path']).resolve()))
            if row.get('status') == 'completed' and known is None:
                raise ValueError('candidate entry point is missing from source inventory')
            if known is not None and known != source:
                raise ValueError('candidate entry point changed during import/evaluation')
    elif row.get('status') == 'completed':
        raise ValueError('candidate entry point path is missing')
    return json.dumps([source, row.get('config', {}), dependencies], sort_keys=True)


def summarize(rows, expected_cells):
    expected = set(map(tuple, expected_cells))
    if len(expected) != len(expected_cells):
        raise ValueError('duplicate expected cells')
    seen = set()
    signatures = set()
    finished_times = []
    completed = target = eligible = 0
    for row in rows:
        cell = (row['track_id'], row['seed'])
        if cell in seen:
            raise ValueError(f'duplicate receipt: {cell}')
        if cell not in expected:
            raise ValueError(f'unexpected cell: {cell}')
        seen.add(cell)
        signatures.add(_signature(row))
        if row.get('status') != 'completed':
            continue
        completed += 1
        eligible += int(row.get('resource_eligible') is True)
        lap = row.get('lapTimeMs')
        start, finish, end = (row.get(k) for k in ('start_t', 'finish_time_s', 'end_t'))
        finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        if not finite(start) or not finite(end) or end < start:
            raise ValueError('invalid simulation start/end timestamp')
        if row.get('finished') is True:
            if not finite(finish) or not start < finish <= end or lap != round((finish-start)*1000):
                raise ValueError('finish timestamp and lapTimeMs disagree')
        elif finish is not None or lap is not None:
            raise ValueError('unfinished receipt has a finish timestamp or lapTimeMs')
        valid_time = (isinstance(lap, (int, float)) and not isinstance(lap, bool)
                      and math.isfinite(lap) and lap > 0)
        if row.get('finished') is True and valid_time:
            finished_times.append(lap)
            target += int(10000 <= lap <= 13000)
    if len(signatures) > 1:
        raise ValueError('mixed candidate sources, dependencies or configurations')
    complete = bool(expected) and seen == expected and completed == len(expected)
    driving_target_met = complete and target == len(expected)
    resource_eligible = complete and eligible == len(expected)
    return dict(expected=len(expected), observed=len(seen), completed=completed,
                complete=complete, finished=len(finished_times),
                finish_rate=len(finished_times)/len(expected) if expected else None,
                in_target_window=target,
                driving_target_met=driving_target_met,
                resource_eligible=resource_eligible,
                original_target_met=driving_target_met and resource_eligible,
                finished_median_ms=median(finished_times) if finished_times else None,
                finished_max_ms=max(finished_times) if finished_times else None)


def fingerprint_candidate(root, sources, config):
    root = Path(root).resolve()
    hashes = {}
    for source in sources:
        path = Path(source)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        relative = path.relative_to(root)
        hashes[str(relative)] = sha256(path.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError('candidate sources must not be empty')
    payload = dict(source_sha256=hashes, config=json.loads(json.dumps(config)))
    payload['candidate_id'] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def verify_candidate(root, manifest):
    root = Path(root).resolve()
    current = fingerprint_candidate(root, [root / p for p in manifest['source_sha256']],
                                    manifest['config'])
    if current != manifest:
        raise ValueError('frozen candidate changed; holdout evaluation is prohibited')
