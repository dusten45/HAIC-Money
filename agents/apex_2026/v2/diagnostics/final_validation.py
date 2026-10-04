"""Read-only final evidence gates. Never selects, writes, allocates or resets.

Callers supply prospectively reviewed metadata inventories and an independently
verified push receipt. This module does not turn a local tracking ref into proof
of a remote push, or a scoped seed audit into global historical non-use.
"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from agents.apex_2026.protocol import summarize, verify_candidate, _signature

# Allocation metadata identified by existing cross-lane audit tooling, not
# private episode/outcome files. This necessary set is not a complete inventory.
KNOWN_PROTOCOL_SOURCES = frozenset({
    'experiments/drqv2-geometry-mix-v1-r6.json',
    'experiments/drqv2-teacher-replay-v1-r3.json',
    'experiments/pixel-rlpd-entropy-target-ablation-v5.json',
    'experiments/dreamerv3-b1-terminal-positive-weight-local-v9.json',
    'experiments/rlpd-g0-completion-v1.json',
    'experiments/drqv2-geometry-augmentation-v1.json',
    'experiments/drqv2-retention-r7.json',
})

def require(value, message):
    if not value: raise ValueError(message)

def digest(path): return sha256(Path(path).read_bytes()).hexdigest()

def timestamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None, 'Timestamp must include timezone')
    return result

def cells(rows): return [(row['track_id'],row['seed']) for row in rows]

def validate_fresh_evidence(benchmark, manifest, rows28, repeats4, *, pushed_at,
                            expected_evaluator_sha256, expected_official_sha256,
                            action_parity=None):
    """Pure receipt gate; file/remote authenticity must be verified separately."""
    required=cells(benchmark['required']);regression=cells(benchmark['consumed_regression'])
    require(len(required)==4 and len(regression)==24 and len(set(required+regression))==28,
            'Expected exactly4 required and24 distinct regression cells')
    require(set(manifest['source_sha256'])=={'agent.py'}, 'Expected one standalone frozen source')
    source=manifest['source_sha256']['agent.py'];boundary=timestamp(pushed_at)
    all_rows=list(rows28)+list(repeats4)
    require(len({_signature(row) for row in all_rows})==1, 'Mixed frozen candidate signatures')
    for row in all_rows:
        require(row.get('config')==manifest['config'], 'Frozen config mismatch')
        provenance=row['provenance']
        require(provenance.get('agent_sha256')==source, 'Frozen source mismatch')
        require(provenance.get('evaluator_sha256')==expected_evaluator_sha256, 'Evaluator mismatch')
        require(provenance.get('official_source_sha256')==expected_official_sha256, 'Official source mismatch')
        require(boundary<timestamp(row['started_at'])<=timestamp(row['ended_at']), 'Receipt is not fresh after pushed freeze')
    fresh=summarize(rows28,required+regression);repeat=summarize(repeats4,required)
    required_rows=[row for row in rows28 if (row['track_id'],row['seed']) in set(required)]
    required_summary=summarize(required_rows,required)
    laps=[row['lapTimeMs'] for row in required_rows if row.get('finished')]
    average=mean(laps) if len(laps)==4 else None
    gate=benchmark['matched_improvement_gate'];limit=gate['required_mean_lap_ms_at_most']
    speed_ok=average is not None and (average<limit if gate['required_mean_must_be_strictly_better'] else average<=limit)
    passed=(fresh['complete'] and fresh['finished']==28 and fresh['resource_eligible']
            and repeat['complete'] and repeat['resource_eligible'] and speed_ok)
    return dict(fresh28=fresh,required4=required_summary,repeat4=repeat,required_mean_lap_ms=average,
        evidence_gate_passed=bool(passed),repeat_action_parity=action_parity,
        repeat_policy='Repeat sources/resources/completeness required; exact action parity and repeat finishes reported, not added as new gates.',
        authenticity='Pure receipt validation only; verify actual frozen files and independently attested pushed commit separately.',
        new_holdout_allocated=False)

def verify_frozen_files(candidate_root, manifest):
    verify_candidate(candidate_root,manifest)
    return dict(candidate_id=manifest['candidate_id'],files_verified=True)

def audit_allocation_metadata(repo_root, inventory, benchmark, proposed_cells=()):
    """Check reviewed, SHA-pinned ID-only metadata without returning private IDs.

    Does not generate candidates. All unresolved schemas block. Inventory scope
    must be independently reviewed; required known paths are necessary only.
    """
    from scripts import audit_drq_training_seeds as existing
    root=Path(repo_root).resolve()
    require(inventory.get('format')=='apex-v2-reviewed-allocation-inventory-v1','Unknown inventory format')
    require(inventory.get('reviewed') is True and inventory.get('review_commit'),'Inventory review missing')
    require(inventory.get('unresolved_sources')==[],'Unresolved allocation sources')
    sources=inventory.get('protocol_sources',{})
    require(KNOWN_PROTOCOL_SOURCES<=set(sources),'Known reserved allocation metadata omitted')
    require(inventory.get('scope_limitations'),'Scoped audit limitations must be explicit')
    private={}
    evidence=[]
    try:
        declarations=existing._sources(root,sources,'protocol')
        for name,path,expected in declarations:
            raw=path.read_bytes();require(sha256(raw).hexdigest()==expected,'Allocation metadata hash mismatch')
            record=existing._parse_json(raw,name)
            local={};existing._protocol_ids(record,name,local)
            private.update(local)
            evidence.append(dict(path=name,sha256=expected,unique_seed_count=len(local)))
    except (existing.SeedAuditError,OSError,UnicodeError) as error:
        # Underlying parser messages can include private values: do not forward.
        raise ValueError('Allocation metadata unresolved; parser rejected a declared source') from None
    exposed={row['seed'] for row in benchmark['required']+benchmark['consumed_regression']}
    seeds=[]
    for row in proposed_cells:
        seed=row['seed'];track=row['track_id']
        require(type(seed) is int and 1<=seed<2**32 and type(track) is int and 1<=track<=4,'Invalid candidate cell')
        seeds.append(seed)
    require(len(seeds)==len(set(seeds)),'Duplicate candidate seeds')
    require(not(set(seeds)&(set(private)|exposed)),'Candidate intersects recorded exposure or reservation')
    require(all(digest(root/row['path'])==row['sha256'] for row in evidence),'Metadata changed during audit')
    return dict(status='no_known_overlap_in_reviewed_scope',sources=evidence,private_seed_count=len(private),
        consumed_seed_count=len(exposed),candidate_count=len(seeds),private_seed_values_exposed=False,
        parser_path='scripts/audit_drq_training_seeds.py',parser_sha256=digest(Path(existing.__file__)),
        inventory_sha256=sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),
        allocation_performed=False,global_freshness_certified=False,scope_limitations=inventory['scope_limitations'])
