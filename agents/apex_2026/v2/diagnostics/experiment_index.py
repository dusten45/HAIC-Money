"""Read-only receipt scoreboard: explicit allocations, never fastest-of-retries.

New folders require an explicit registry entry (or --registry JSON override).
Unknown receipt folders are reported, never assigned an inferred denominator.
No simulator or candidate imports; no resets.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[4]
BENCHMARK = ROOT/'agents/apex_2026/v2/benchmark.json'
spec = importlib.util.spec_from_file_location('apex_evidence_protocol', ROOT/'agents/apex_2026/protocol.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
REQUIRED = [(1,516237),(2,644062),(3,1007),(4,18800)]
OLD_FAILURES = [(2,4031370700),(3,4111953688)]
GEODESIC_LOSSES = [(1,3601050002),(3,3601050003)]
SCREEN6 = REQUIRED+OLD_FAILURES
SCREEN8 = SCREEN6+GEODESIC_LOSSES


def default_registry():
    registry = {}
    six = ['actuator-r0','brake-r0','exact-pedal-r0','force-r0','friction-envelope-r0',
        'predictive-r1','obstacle-shield-r0','obstacle-steering-r0','kinematic-r0','beam-r1','beam-r2','beam-r3','recovery-r0','recovery-r1','shield-r0','shield-r1','speed-r0','yaw-r0']
    four = ['geodesic-r0','geodesic-frenet-r5','predictive-r0']
    eight = ['fallback-r1','geodesic-r2','geodesic-frenet-r6','geodesic-footprint-r7','racing-r3','racing-r4']
    for names, cells, scope in [(six,SCREEN6,'screen6'),(four,REQUIRED,'required4'),(eight,SCREEN8,'screen8')]:
        for name in names:
            registry[name] = {'folders':['/tmp/apex-v2-'+name], 'expected_cells':cells, 'scope':scope}
    regression = [(c['track_id'],c['seed']) for c in json.loads(BENCHMARK.read_text())['consumed_regression']]
    registry['geodesic-r1'] = {'folders':['/tmp/apex-v2-geodesic-r1','/tmp/apex-v2-geodesic-r1-regression'],
        'expected_cells':REQUIRED+regression, 'scope':'required4+full_consumed_regression24'}
    registry['geodesic-frenet-r6'] = {'folders':['/tmp/apex-v2-geodesic-frenet-r6','/tmp/apex-v2-geodesic-frenet-r6-regression'],
        'expected_cells':REQUIRED+regression, 'scope':'required4+full_consumed_regression24'}
    registry['geodesic-footprint-r7'] = {'folders':['/tmp/apex-v2-geodesic-footprint-r7','/tmp/apex-v2-geodesic-footprint-r7-regression'],
        'expected_cells':REQUIRED+regression, 'scope':'required4+full_consumed_regression24'}
    registry['terrain-r8'] = {'folders':['/tmp/apex-v2-terrain-r8','/tmp/apex-v2-terrain-r8-regression'],
        'expected_cells':REQUIRED+regression, 'scope':'required4+full_consumed_regression24'}
    registry['curvature-memory-r0'] = {'folders':['/tmp/apex-v2-curvature-memory-r0'],
        'expected_cells':SCREEN8+[(1,3601050001)], 'scope':'screen9'}
    registry['terrain-grounded-r9'] = {'folders':['/tmp/apex-v2-terrain-grounded-r9','/tmp/apex-v2-terrain-grounded-r9-regression'],
        'expected_cells':REQUIRED+regression, 'scope':'required4+full_consumed_regression24'}
    registry['obstacle-steering-audit'] = {'folders':['/tmp/apex-v2-obstacle-steering-audit'],
        'expected_cells':[(3,4111953688)], 'scope':'diagnostic1_not_benchmark', 'benchmark_inclusion':False,
        'interpretation':'Image/state capture only; 326-action parity audit. Do not add to original screen6 or treat as independent validation.'}
    registry['curvature-memory-bugfix-untested'] = {'folders':[], 'expected_cells':[],
        'scope':'counterfactual_only_no_driving', 'source_archive':'agents/apex_2026/v2/results/curvature_memory_sources/b6c66310a58b.py',
        'source_sha256':hashlib.sha256((ROOT/'agents/apex_2026/v2/results/curvature_memory_sources/b6c66310a58b.py').read_bytes()).hexdigest(),
        'benchmark_inclusion':False, 'interpretation':'Bug-fixed source used for counterfactual analysis only; no driving receipt, not evaluated.'}
    registry['motion-registration-baseline-diagnostic'] = {'folders':['/tmp/apex-v2-motion-registration'],
        'expected_cells':[REQUIRED[0]], 'scope':'diagnostic1'}
    return registry


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def cell(row):
    return (row['track_id'],row['seed'])


def summarize_selection(rows, expected):
    expected = [tuple(c) for c in expected]
    assert len(expected)==len(set(expected))
    unique = {}
    for r in rows:
        if cell(r) in expected:
            key = (cell(r),r.get('started_at'),r.get('agent_path'),digest(r.get('config',{})),r.get('provenance',{}).get('agent_sha256'))
            if key in unique and digest(unique[key]) != digest(r):
                raise ValueError('Conflicting snapshots of one episode; do not silently choose a result')
            unique[key] = r
    selected = {}
    for r in sorted(unique.values(),key=lambda r:(r.get('started_at',''),digest(r))):
        selected.setdefault(cell(r),r)
    complete = [r for r in selected.values() if r.get('status')=='completed']
    warnings = []
    validated = []
    for r in complete:
        try:
            protocol.summarize([r],[cell(r)])
            validated.append(r)
        except (ValueError,KeyError,TypeError) as exc:
            warnings.append(f'{cell(r)}: {exc}')
    if validated:
        try:
            protocol.summarize(validated,[cell(r) for r in validated])
        except ValueError as exc:
            warnings.append(str(exc))
    good = [r for r in validated if r.get('finished') is True]
    return {'expected':len(expected),'expected_cells':[list(c) for c in expected],
        'observed':len(selected),'completed':len(complete),
        'incomplete':len(selected)-len(complete),'missing':len(expected)-len(selected),
        'unique_attempts':len(unique),'repeat_attempts':len(unique)-len(selected),
        'raw_finished':sum(r.get('finished') is True for r in complete),
        'finished':len(good) if not warnings else None,
        'dnf':sum(r.get('finished') is False for r in complete),
        'finished_median_ms':median([r['lapTimeMs'] for r in good]) if good and not warnings else None,
        'all_cells_finish':len(good)==len(expected) and not warnings,
        'target10_13_count':sum(10000<=r['lapTimeMs']<=13000 for r in good) if not warnings else None,
        'finish_qualified_completed':sum(r.get('finish_qualified') is True for r in complete),
        'progress_at_least95_completed':sum((r.get('progress') or 0)>=.95 for r in complete),
        'invalid_action_failure_count':sum(r.get('retire_reason')=='invalid_action' for r in complete),
        'resource_ineligible_completed':sum(r.get('resource_eligible') is False for r in complete),
        'resource_eligible_completed':sum(r.get('resource_eligible') is True for r in complete),
        'damage_sum_completed':sum(r.get('damage',0) or 0 for r in complete),
        'protocol_verified':not warnings,'warnings':warnings,
        'selected':[{'track_id':c[0],'seed':c[1],'started_at':r.get('started_at'),
                     'status':r['status'],'finished':r.get('finished'),'lapTimeMs':r.get('lapTimeMs')} for c,r in selected.items()]}


def read_primary(path):
    try:
        raw = path.read_bytes()
        row = json.loads(raw)
    except (OSError,ValueError):
        return None
    if not isinstance(row,dict) or not all(k in row for k in ['track_id','seed','status','agent_path']):
        return None
    return row,hashlib.sha256(raw).hexdigest()


def compact(path, row, sha):
    keys = ['track_id','seed','status','started_at','ended_at','finished','lapTimeMs','start_t','finish_time_s','end_t','progress','damage','resource_eligible','retire_reason','finish_qualified','finish_qualified_time_s','invalid_actions','act_timeout_count','error']
    return dict(receipt=str(path),receipt_sha256=sha,**{k:row.get(k) for k in keys})


def build(registry):
    all_known_paths = set()
    variants = []
    for label, allocation in registry.items():
        entries = {}
        for folder in allocation['folders']:
            p = Path(folder)
            for path in sorted(p.rglob('*.json')) if p.is_dir() else [p]:
                parsed = read_primary(path)
                if parsed:
                    all_known_paths.add(str(path.resolve()))
                    row, sha = parsed
                    entries[str(path.resolve())] = (row,sha)
        groups = {}
        for path,(row,sha) in entries.items():
            key = digest([row.get('provenance',{}).get('agent_sha256'),row.get('config',{})])
            groups.setdefault(key,[]).append((path,row,sha))
        if not groups:
            variants.append({'label':label,'allocation':allocation,'groups':[], 'execution_status':'not_executed_no_primary_receipts', 'warning':'No primary receipts yet; planned cells remain missing, no simulator run is inferred'})
            continue
        output_groups = []
        for key, items in groups.items():
            rows = [x[1] for x in items]
            first = rows[0]
            warnings = []
            for _,row,_ in items:
                src = Path(row['agent_path'])
                expected_sha = row.get('provenance',{}).get('agent_sha256')
                if not src.exists():
                    warnings.append('Current source unavailable: '+str(src))
                elif hashlib.sha256(src.read_bytes()).hexdigest()!=expected_sha:
                    warnings.append('Current source differs from historical receipt: '+str(src))
            unknown = sorted(set(cell(r) for r in rows)-set(map(tuple,allocation['expected_cells'])))
            if unknown:
                warnings.append('Receipts outside declared allocation: '+str(unknown))
            whole = summarize_selection(rows,allocation['expected_cells'])
            required = summarize_selection(rows,REQUIRED)
            consumed = [c for c in allocation['expected_cells'] if tuple(c) not in REQUIRED]
            signatures = {}
            for _,row,_ in items:
                p = row.get('provenance',{})
                inventory = {'before':p.get('runtime_source_sha256_before',{}),'after':p.get('runtime_source_sha256_after',{})}
                signatures[digest(inventory)] = {'before_sha256':digest(inventory['before']), 'after_sha256':digest(inventory['after']),
                    'before_count':len(inventory['before']), 'after_count':len(inventory['after']),
                    'changed_paths':[p for p in inventory['before'].keys() & inventory['after'].keys() if inventory['before'][p]!=inventory['after'][p]],
                    'note':'Full path-to-hash maps remain in SHA-bound primary receipts; protocol verifies them before aggregation.'}
            output_groups.append({'candidate_key':key,'source_sha256':first.get('provenance',{}).get('agent_sha256'),
                'config':first.get('config',{}),'agent_paths':sorted(set(r['agent_path'] for r in rows)),
                'runtime_inventories':signatures,'whole_declared_scope':whole,'required4_coverage':required,
                'consumed_declared_scope':summarize_selection(rows,consumed) if consumed else None,
                'warnings':sorted(set(warnings)), 'receipts':[compact(Path(p),r,sha) for p,r,sha in items]})
        variants.append({'label':label,'allocation':allocation,'groups':output_groups})
    unknown = []
    for folder in sorted(Path('/tmp').glob('apex-v2-*')):
        if folder.is_dir():
            for path in sorted(folder.rglob('*.json')):
                if str(path.resolve()) not in all_known_paths:
                    parsed = read_primary(path)
                    if parsed:
                        unknown.append(compact(path,*parsed))
    return variants, unknown


def baseline():
    scopes = {'historical_required4':(['/tmp/apex-final-frozen/required'],REQUIRED),
        'historical_consumed24':(['/tmp/apex-final-frozen/development','/tmp/apex-final-holdout/holdout'],
        [(c['track_id'],c['seed']) for c in json.loads(BENCHMARK.read_text())['consumed_regression']])}
    result = {}
    for label,(folders,expected) in scopes.items():
        entries=[]
        for folder in folders:
            for path in sorted(Path(folder).glob('*.json')):
                parsed=read_primary(path)
                if parsed:
                    entries.append((path,*parsed))
        result[label]={'summary':summarize_selection([r for _,r,_ in entries],expected),
            'receipts':[compact(p,r,s) for p,r,s in entries]}
    return result


def markdown(result):
    lines=['# V2 experiment scoreboard','',f"Receipt snapshot: {result['generated_at']}. No new resets.",'',
        'Explicit planned scopes retain missing and started cells. DNF means completed without a finish. Required4 coverage is reported even for smaller diagnostic allocations. Each source + exact config is separate; source/dependency verification failures invalidate aggregate finish claims.',
        '', '| Candidate | Required4 finishes | Required lap times (s; T1–4) | Declared scope finishes | Consumed subset | Pending / missing | Code / resource failures | Warnings |',
        '|---|---:|---|---|---|---:|---|---|']
    for variant in result['variants']:
        if not variant['groups']:
            total=len(variant['allocation']['expected_cells'])
            lines.append(f"| {variant['label']} | 0/4 | — / — / — / — | {variant['allocation']['scope']}: 0/{total} | — | 0 / {total} | 0 / 0 | awaiting receipts |")
        for g in variant['groups']:
            r=g['required4_coverage'];a=g['whole_declared_scope'];selected={(x['track_id'],x['seed']):x for x in r['selected']}
            times=[]
            for c in REQUIRED:
                x=selected.get(c)
                times.append('—' if x is None else 'pending' if x['status']!='completed' else 'DNF' if not x['finished'] else f"{x['lapTimeMs']/1000:.2f}")
            warn=len(g['warnings'])+len(a['warnings'])
            cs=g['consumed_declared_scope']
            consumed_text=f"{cs['finished']}/{cs['expected']}" if cs else '—'
            label=variant['label']+' '+g['candidate_key'][:7]
            lines.append(f"| {label} | {r['finished'] if r['finished'] is not None else 'INVALID'}/4 | {' / '.join(times)} | {variant['allocation']['scope']}: {a['finished'] if a['finished'] is not None else 'INVALID'}/{a['expected']} | {consumed_text} | {a['incomplete']} / {a['missing']} | {a['invalid_action_failure_count']} / {a['resource_ineligible_completed']} | {warn} |")
    lines += ['', 'V1 historical baseline (not new v2 validation):']
    for name,b in result['historical_v1_baseline'].items():
        s=b['summary'];lines.append(f"- {name}: {s['finished']}/{s['expected']} finishes; completed {s['completed']}; missing {s['missing']}.")
    lines += ['',f"Unregistered primary receipts: {len(result['unregistered_primary_receipts'])}. Add an explicit registry allocation before comparing them.",'',
        'The geodesic full24 regression includes its two earlier probes exactly once. Overlapping required/screen summaries must not be added together. Retry policy: deduplicate the same episode, then select the earliest started attempt per cell; preserve all attempts, never choose the fastest finish.',
        '', 'Rebuild from the repository root:', '', '```sh',
        '.venv/bin/python agents/apex_2026/v2/diagnostics/experiment_index.py', '```', '',
        'Use `--registry path.json` to replace the manual registry with `{label: {folders: [...], expected_cells: [[track,seed],...], scope: "..."}}`. Expected cells must be declared explicitly; new folders are not auto-promoted.',
        '', '[Primary receipt index](results/experiment-index.json) · [Generator](diagnostics/experiment_index.py)',
        '', 'Diagnostic partitions are excluded from benchmark totals and do not constitute independent validation. Counterfactual-only sources have no driving evidence.', '', 'Finished-only medians in JSON are descriptive and never replace the full declared denominator. No candidate is promoted by this table.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'agents/apex_2026/v2/results/experiment-index.json')
    p.add_argument('--markdown',type=Path,default=ROOT/'agents/apex_2026/v2/EXPERIMENTS.md')
    args=p.parse_args()
    registry=json.loads(args.registry.read_text()) if args.registry else default_registry()
    variants,unknown=build(registry)
    result={'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),
        'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'benchmark_sha256':hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        'registry':registry,'selection_policy':'same-episode deduplication; earliest started attempt per cell, never fastest',
        'variants':variants,'historical_v1_baseline':baseline(),'unregistered_primary_receipts':unknown}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    args.markdown.write_text(markdown(result))
    print(json.dumps({'variants':len(variants),'groups':sum(len(v['groups']) for v in variants),'unregistered':len(unknown),'output':str(args.output)}))


if __name__=='__main__':main()
