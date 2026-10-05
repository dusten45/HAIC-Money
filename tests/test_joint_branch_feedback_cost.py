from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from haic.algorithms.joint_control.interval_comparison import extract_scene
from scripts.score_joint_branch_feedback import actual_h4, score_feedback, run, old, clean
from tests.test_joint_interval_comparison import fixture


def records():
    start = dict(t=3.66, x=10., y=20., yaw=0.)
    rows = [dict(step=i, action=[0., .1 * (i - 34), .2], evaluation_only=dict(pre=start)) for i in range(34, 38)]
    raw = [dict(step=34 + i // 4, t=start['t'] + .02 * (i + 1), x=10., y=20. + (i + 1) / 2, yaw=0.) for i in range(16)]
    return rows, raw


class FeedbackCostTests(unittest.TestCase):
    def test_actual_changing_controls_are_not_replaced_with_fixed_tail(self):
        rows, raw = records()
        w = actual_h4(rows, raw)
        assert w is not None
        np.testing.assert_array_equal(w['controls'], np.repeat(np.asarray([r['action'] for r in rows], np.float32), 4, axis=0))
        self.assertEqual(w['poses'].shape, (17, 3))
        self.assertEqual(w['poses'][-1, 1], 8.)

    def test_shortened_or_missing_hold_is_censored_not_truncated(self):
        rows, raw = records()
        self.assertIsNone(actual_h4(rows, raw[:-1]))
        self.assertIsNone(actual_h4(rows[:-1], raw))

    def test_bad_clock_fails(self):
        rows, raw = records()
        raw[3]['t'] += .01
        with self.assertRaises(ValueError):
            actual_h4(rows, raw)

    def test_same_paths_same_costs_and_full_support(self):
        frames, motions = fixture()
        w = actual_h4(*records())
        out = score_feedback(asdict(extract_scene(frames, motions=motions)), dict(A=w, B=w, C=w))
        self.assertTrue(out['common_support'])
        np.testing.assert_array_equal(out['supported_ticks'], [17] * 3)
        for delta in out['delta_intervals'].values():
            np.testing.assert_array_equal(delta['by_reference'], 0.)

    def test_missing_one_branch_does_not_create_favorable_cost(self):
        frames, motions = fixture()
        w = actual_h4(*records())
        out = score_feedback(asdict(extract_scene(frames, motions=motions)), dict(A=w, B=None, C=w))
        self.assertEqual(out['status'], 'CENSORED_H4')
        self.assertIsNone(out['delta_intervals'])

    def test_mismatched_start_fails(self):
        frames, motions = fixture()
        a = actual_h4(*records())
        b = actual_h4(*records())
        assert b is not None
        b['start'] = dict(b['start'], x=11.)
        with self.assertRaises(ValueError):
            score_feedback(asdict(extract_scene(frames, motions=motions)), dict(A=a, B=b, C=a))

    def test_operator_serializes_numpy_and_preserves_exclusive_output(self):
        frames, motions = fixture()
        rows, raw = records()
        rows = [dict(row, event='decision') for row in rows]
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            output = Path(directory)
            ds, rs = output / 'decisions.jsonl', output / 'raw.jsonl'
            ds.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            rs.write_text(''.join(json.dumps(r) + '\n' for r in raw))
            ep = dict(decision_stream_file=ds.name, raw_trace_file=rs.name,
                      decision_stream_sha256=old.sha(ds), raw_trace_sha256=old.sha(rs))
            episode = output / 'B.json'
            old.save(episode, ep)
            summary_path, report_path = output / 'summary.json', output / 'episode-report.json'
            summary = dict(protocol_sha256='bound-protocol-sha', matched_ABC=True, policy_valid=True, operator_error=None)
            report = dict(protocol_sha256='bound-protocol-sha', operator_error=None,
                          rows=[dict(file='B.json', status='completed', sha256=old.sha(episode))],
                          artifacts_sha256={p.name: old.sha(p) for p in (episode, ds, rs)})
            old.save(summary_path, summary)
            old.save(report_path, report)
            anchor = output / 'anchor.json'
            old.save(anchor, dict(scene=clean(asdict(extract_scene(frames, motions=motions)))))
            gate = dict(references={k: dict(root=str(output), episode_path=str(episode)) for k in ('A', 'C')},
                        fixed_h4=dict(anchor_path=str(anchor), baseline_tail=[], alternative_tail=[]))
            protocol = dict(gate_binding=gate, schedule=[dict(file='B.json')])
            with patch('scripts.run_joint_single_branch.validate_frozen', return_value=protocol):
                result = run(output, 'bound-protocol-sha')
                self.assertTrue(result['common_support'])
                self.assertEqual(result, old.read_json(output / 'feedback-cost.json'))
                self.assertEqual(result['input_sha256'][str(report_path)], old.sha(report_path))
                with self.assertRaises(FileExistsError):
                    run(output, 'bound-protocol-sha')
                summary_path.write_text(json.dumps(dict(summary, policy_valid=False)))
                with self.assertRaisesRegex(ValueError, 'matched policy-valid'):
                    run(output, 'bound-protocol-sha')
                summary_path.write_text(json.dumps(summary))
                episode.write_text(json.dumps(dict(ep, tampered=True)))
                with self.assertRaisesRegex(ValueError, 'not bound'):
                    run(output, 'bound-protocol-sha')


if __name__ == '__main__':
    unittest.main()
