"""Reproduce two Beam R3 review counterexamples without environment resets.

Run from the repository root with:
  .venv/bin/python -m agents.apex_2026.v2.diagnostics.beam_r3_review

Only synthetic HUD rendering, static internal body geometry and existing three
completed REQUIRED traces are used. This diagnostic is never a runtime import.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from agents.apex_2026.v2.diagnostics.shadow_rpm import render
from agents.apex_2026.v2.shadow_physics import ShadowCar, decode_wheel_omega


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'agents/apex_2026/v2/results/beam_sources/aef90410f8ad.py'
SHADOW = ROOT / 'agents/apex_2026/v2/shadow_physics.py'
EXPECTED_SOURCE = 'aef90410f8ad7f3f022a7c5a07c9ba87b997304964a63e5ca0601d8e60e7e8fa'
EXPECTED_SHADOW = '8d5c21ce445bda2f192e4815739253ebf46d1099f635d966c12fe9e877990a56'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert digest(SOURCE) == EXPECTED_SOURCE
    assert digest(SHADOW) == EXPECTED_SHADOW
    spec = importlib.util.spec_from_file_location('review_frozen_beam_r3', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # No policy act, internal dynamics step, or official environment construction.
    policy = module.Agent.__new__(module.Agent)
    shadow = ShadowCar()
    free = np.ones((84, 84), np.uint8)
    free[64, 42] = 0
    point = (0., -1. / 1.701)
    count = policy._footprint_cost(free, 0., 0., 0.)
    inside_hull = any(fixture.TestPoint(point) for fixture in shadow.hull.fixtures)
    assert count == 0 and inside_hull

    rpm_examples = []
    for omega in (20., 100., 200., 300.):
        frame = render([omega] * 4)
        decoded = decode_wheel_omega(frame)
        bottom = float(frame[82:84, 14:24].max())
        ambiguous = bool(np.any(decoded > 340) or np.any(frame[82:84, 14:24] > .02))
        assert ambiguous and np.all(decoded < 340)
        rpm_examples.append(dict(true_omega_rad_s=omega, decoded=decoded.tolist(),
                                 bottom_pixel_max=bottom, ambiguous=ambiguous))

    evidence = []
    for name in ('t1-s516237', 't2-s644062', 't3-s1007'):
        receipt = Path('/tmp/apex-v2-beam-r3') / (name + '.json')
        trace = receipt.with_suffix('.jsonl')
        result = json.loads(receipt.read_text())
        assert result['status'] == 'completed' and result['finished']
        provenance = result['provenance']
        assert provenance['agent_sha256'] == EXPECTED_SOURCE
        for phase in ('before', 'after'):
            assert provenance['runtime_source_sha256_' + phase][str(SHADOW)] == EXPECTED_SHADOW
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        valid = [row['policy_diagnostics'] for row in rows
                 if row.get('policy_diagnostics', {}).get('valid')]
        evidence.append(dict(receipt=str(receipt), receipt_sha256=digest(receipt),
            trace=str(trace), trace_sha256=digest(trace), trace_rows=len(rows),
            valid_policy_rows=len(valid), flow_invalid_rows=sum(not d['flow_valid'] for d in valid),
            rpm_ambiguous_rows=sum(d['rpm_ambiguous'] for d in valid),
            lap_time_ms=result['lapTimeMs'], damage=result['damage'],
            inference_max_s=result['inference_max_s'], act_timeout_count=result['act_timeout_count'],
            resource_eligible=result['resource_eligible'], peak_memory_mib=result['peak_memory_mib']))

    report = dict(
        scope='Independent read-only review of immutable Beam R3. Zero environment resets, zero episodes, no candidate changes. This reproducer executes zero physics steps; the separately reported existing snapshot test advances only its internal shadow model.',
        source=dict(path=str(SOURCE), sha256=digest(SOURCE)),
        runtime_dependency=dict(path=str(SHADOW), sha256=digest(SHADOW)),
        reproducer=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))),
        footprint_counterexample=dict(blocked_pixel_yx=[64, 42], pose_xy_heading=[0., 0., 0.],
            sample_violations=count, blocked_pixel_metric_center=list(point),
            center_inside_actual_hull=inside_hull,
            limitation='Synthetic one-pixel mask hole establishes sampling incompleteness, not a reproduced official obstacle collision.'),
        rpm_ambiguity_counterexamples=rpm_examples,
        findings=[
            dict(location='beam_agent.py:183,240-248', severity='important',
                 finding='25 point samples can miss blocked pixels inside the hull. The rectangle also does not enclose fully steered front wheels.',
                 next_test='Rasterized hull/wheel polygon coverage at several headings; retain this interior-hole regression.'),
            dict(location='beam_agent.py:334', severity='diagnostic',
                 finding='The negative-RPM crop threshold fires on ordinary positive bar resampling bleed. This flag does not currently affect action selection.',
                 next_test='Separate synthetic positive, zero, negative and saturated bars before interpreting ambiguity frequency.'),
            dict(location='beam_agent.py:261-307', severity='limitation',
                 finding='Collision is a finite penalty; beam pruning and a 0.64-second horizon do not guarantee collision avoidance or stopping viability. Full-horizon completion fixes a ranking inconsistency but is still approximate.',
                 next_test='Distinguish selected collision count, existence of a collision-free candidate, and out-of-horizon stopping feasibility.'),
            dict(location='beam_agent.py:330-348; shadow_physics.py:263-284', severity='limitation',
                 finding='Invalid flow propagates predicted slip; all wheels have road contact and undamaged grip. Reverse/saturated RPM, per-wheel joint states and relative joint velocities remain partially unobserved.',
                 next_test='Bound reconstruction error across prolonged flow loss and surface/damage changes before claiming state-exact predictions.'),
            dict(location='beam_agent.py:290-307', severity='limitation',
                 finding='No internal time budget guarantees return before 5 seconds. The three completed receipts pass; other states and machine load remain unverified.',
                 next_test='Use actual act maxima and timeout counts from every completed receipt, not mean runtime.')],
        runtime_boundary='Inspected candidate and shadow imports contain no prohibited imports and no live simulator dependency. Package parents are namespace packages. Diagnostic renderer imports do not enter candidate runtime.',
        selected_existing_tests=dict(command='.venv/bin/python -m pytest -q agents/apex_2026/v2/test_beam_agent.py -k "snapshot or objective or invalid_observation or textured"', passed=5, deselected=3),
        evidence=evidence,
        interpretation='Three actual clean finishes and passing resource receipts are preserved. No promotion or 10-13-second target achievement follows from this review. Later batch cells are outside this frozen review evidence.')
    output = ROOT / 'agents/apex_2026/v2/results/beam-r3-review.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), footprint_counterexample=True,
                         positive_rpm_false_ambiguities=len(rpm_examples), completed_receipts=len(evidence))))


if __name__ == '__main__':
    main()
