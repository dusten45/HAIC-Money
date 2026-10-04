"""Evidence gates: incomplete/mixed runs cannot satisfy a frozen-candidate claim."""
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name('protocol.py')
    assert path.exists(), 'protocol evidence gates are not implemented'
    spec = importlib.util.spec_from_file_location('apex_protocol_test', path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


CELLS = [(1, 516237), (2, 644062), (3, 1007), (4, 18800)]


def episodes():
    return [dict(track_id=t, seed=s, status='completed', finished=True,
                 lapTimeMs=12000, start_t=1.02, finish_time_s=13.02,
                 end_t=13.10, resource_eligible=True,
                 agent_path='/project/agent.py',
                 provenance={'agent_sha256': 'one',
                             'runtime_source_sha256_before': {'/project/agent.py': 'one'},
                             'runtime_source_sha256_after': {'/project/agent.py': 'one'}}, config={})
            for t, s in CELLS]


def test_original_target_requires_every_requested_cell_and_both_time_bounds():
    m = module()
    assert m.summarize(episodes(), CELLS)['original_target_met']
    for bad in (9999, 13001):
        rows = episodes()
        rows[0]['lapTimeMs'] = bad
        rows[0]['finish_time_s'] = rows[0]['start_t'] + bad / 1000
        rows[0]['end_t'] = rows[0]['finish_time_s'] + .08
        assert not m.summarize(rows, CELLS)['original_target_met']
    rows = episodes()
    rows[0]['finished'] = False
    rows[0]['lapTimeMs'] = None
    rows[0]['finish_time_s'] = None
    result = m.summarize(rows, CELLS)
    assert not result['original_target_met']
    assert result['finished'] == 3 and result['expected'] == 4


def test_missing_error_and_duplicate_receipts_are_not_complete_evaluation():
    m = module()
    assert not m.summarize(episodes()[:3], CELLS)['complete']
    rows = episodes()
    rows[0]['status'] = 'error'
    assert not m.summarize(rows, CELLS)['complete']
    with pytest.raises(ValueError, match='duplicate'):
        m.summarize(episodes() + episodes()[:1], CELLS)


def test_mixed_policies_or_configs_cannot_pass_single_candidate_gate():
    m = module()
    for field, value in [('provenance', {'agent_sha256': 'two'}), ('config', {'gain': 2})]:
        rows = episodes()
        rows[0][field] = value
        with pytest.raises(ValueError, match='candidate'):
            m.summarize(rows, CELLS)


def test_freeze_source_change_blocks_holdout_verification(tmp_path):
    m = module()
    source = tmp_path / 'agent.py'
    source.write_text('class Agent: pass\n')
    manifest = m.fingerprint_candidate(tmp_path, [source], {'gain': 1})
    m.verify_candidate(tmp_path, manifest)
    source.write_text('class Agent: changed = True\n')
    with pytest.raises(ValueError, match='changed'):
        m.verify_candidate(tmp_path, manifest)


def test_fingerprint_refuses_files_outside_candidate_root(tmp_path):
    m = module()
    source = tmp_path / 'other.py'
    source.write_text('x=1\n')
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    with pytest.raises(ValueError):
        m.fingerprint_candidate(candidate, [source], {})


def test_timestamp_and_derived_lap_must_agree():
    m = module()
    for field, value in [('finish_time_s', None), ('start_t', float('nan')),
                         ('end_t', 12.0), ('lapTimeMs', 11000)]:
        rows = episodes()
        rows[0][field] = value
        with pytest.raises(ValueError, match='timestamp'):
            m.summarize(rows, CELLS)


def test_resource_eligibility_is_separate_from_raw_driving_outcome():
    rows = episodes()
    rows[0]['resource_eligible'] = False
    result = module().summarize(rows, CELLS)
    assert result['driving_target_met']
    assert result['finished'] == 4
    assert not result['original_target_met']


def test_changed_loaded_source_or_lazy_dependency_invalidates_candidate():
    m = module()
    rows = episodes()
    rows[0]['provenance'].update(runtime_source_sha256_before={'/project/helper.py': 'A'},
                                 runtime_source_sha256_after={'/project/helper.py': 'B'})
    with pytest.raises(ValueError, match='changed'):
        m.summarize(rows, CELLS)
    rows = episodes()
    for row in rows:
        row['provenance']['runtime_source_sha256_after']['/project/lazy.py'] = 'A'
    rows[0]['provenance']['runtime_source_sha256_after']['/project/lazy.py'] = 'B'
    with pytest.raises(ValueError, match='candidate'):
        m.summarize(rows, CELLS)


def test_relative_source_names_resolve_against_candidate_root(tmp_path):
    m = module()
    (tmp_path / 'agent.py').write_text('class Agent: pass\n')
    result = m.fingerprint_candidate(tmp_path, ['agent.py'], {})
    m.verify_candidate(tmp_path, result)


def test_missing_or_deleted_source_inventory_fails_closed():
    m = module()
    for key in ('runtime_source_sha256_before', 'runtime_source_sha256_after'):
        rows = episodes()
        rows[0]['provenance'].pop(key)
        with pytest.raises(ValueError, match='candidate'):
            m.summarize(rows, CELLS)
    rows = episodes()
    rows[0]['provenance']['runtime_source_sha256_before']['/project/deleted.py'] = 'x'
    with pytest.raises(ValueError, match='changed'):
        m.summarize(rows, CELLS)
