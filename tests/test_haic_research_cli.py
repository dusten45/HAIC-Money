"""CLI integration uses temporary files and never invokes a real runner."""

import io
import json
import unittest
from unittest.mock import patch

from haic_research.cli import main
from tests.test_haic_research_commands import CommandFixture, GATES, metadata, research


class CliTests(CommandFixture):
    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        code = main(['--root', str(self.root), *args], runner=self.runner, stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    def test_cli_complete_lifecycle_preview_then_explicit_execution(self):
        for name, payload in (('manifest.json', metadata()), ('research.json', research()), ('arguments.json', self.args)):
            (self.root / name).write_text(json.dumps(payload), encoding='utf-8')
        code, text, _ = self.invoke('plan', '--manifest', str(self.root / 'manifest.json'), '--research', str(self.root / 'research.json'), '--arguments', str(self.root / 'arguments.json'), '--profile', 'train_policy')
        self.assertEqual(code, 0)
        plan = json.loads(text)
        self.assertEqual(len(plan['plan_hash']), 64)
        self.assertEqual(self.runner.calls, [])
        for stage in ('design', 'implementation', 'execution'):
            self.assertEqual(self.invoke('approve', 'run-001', stage, '--source-ref', 'explicit user authorization')[0], 0)
        code, preview, _ = self.invoke('run', 'run-001')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(preview)['mode'], 'plan-only')
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(self.invoke('run', 'run-001', '--execute')[0], 0)
        self.assertEqual(len(self.runner.calls), 1)
        report = {'gate_results': [{'name': name, 'status': 'PASS', 'rationale': 'observed', 'evidence_paths': ['evidence.json']} for name in GATES], 'evidence_paths': ['evidence.json']}
        (self.root / 'report.json').write_text(json.dumps(report), encoding='utf-8')
        self.assertEqual(self.invoke('report', 'run-001', '--outcome', 'ADVANCE', '--report', str(self.root / 'report.json'))[0], 0)
        code, status, _ = self.invoke('status', 'run-001')
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(status)['released'])

    def test_default_preview_does_not_require_execution_approval(self):
        self.plan()
        code, preview, _ = self.invoke('run', 'run-001')
        self.assertEqual(code, 0)
        self.assertIn('train_policy', preview)
        self.assertEqual(self.runner.calls, [])
        self.assertNotEqual(self.invoke('run', 'run-001', '--execute')[0], 0)

    def test_validation_reports_missing_structure_and_unavailable_validator(self):
        code, text, error = self.invoke('validate')
        self.assertEqual((code, error), (1, ''))
        response = json.loads(text)
        self.assertFalse(response['valid'])
        self.assertIn('AGENTS.md', [issue['path'] for issue in response['issues']])
        with patch('haic_research.cli.importlib.import_module', side_effect=ImportError('unavailable fixture')):
            code, text, error = self.invoke('validate')
        self.assertEqual(code, 2)
        self.assertIn('unavailable', error.lower())
        self.assertEqual(text, '')
        self.assertEqual(self.runner.calls, [])

    def test_external_operations_and_path_traversal_absent(self):
        for command in ('submit', 'upload', 'confirm-model'):
            self.assertNotEqual(self.invoke(command)[0], 0)
        self.assertNotEqual(self.invoke('status', '../old')[0], 0)


class CliResultReportingTests(unittest.TestCase):
    def fixture(self):
        from tests.test_haic_research_results import ResultReportingTests
        return ResultReportingTests('test_completion_first_rejects_lower_completion')

    def invoke(self, fixture, *args):
        out, err = io.StringIO(), io.StringIO()
        code = main(['--root', str(fixture.root), *args], runner=fixture.runner, stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    def inputs(self, fixture):
        from haic_research.records import _json_value
        for name, value in (('report.json', fixture.report), ('candidate.json', fixture.candidate), ('control.json', fixture.control)):
            (fixture.root / name).write_text(json.dumps(_json_value(value)), encoding='utf-8')

    def test_report_cli_records_and_promotes_exact_released_comparison(self):
        with self.fixture() as fixture:
            path = fixture.evaluation()
            self.inputs(fixture)
            code, output, error = self.invoke(fixture, 'report', path.name, '--outcome', 'ADVANCE', '--report', 'report.json', '--result', 'candidate.json', '--control', 'control.json', '--promote-sota')
            self.assertEqual((code, error), (0, ''))
            self.assertTrue(json.loads(output)['promotion']['eligible'])
            self.assertEqual(len(fixture.runner.calls), 1)
            before = [(fixture.root / name).read_bytes() for name in ('RESULTS.md', 'SOTA.md', 'docs/experiments/INDEX.md')]
            self.assertEqual(self.invoke(fixture, 'report', path.name, '--outcome', 'ADVANCE', '--report', 'report.json', '--result', 'candidate.json', '--control', 'control.json')[0], 2)
            self.assertEqual(before, [(fixture.root / name).read_bytes() for name in ('RESULTS.md', 'SOTA.md', 'docs/experiments/INDEX.md')])

    def test_report_cli_observation_without_control_and_invalid_options(self):
        with self.fixture() as fixture:
            path = fixture.evaluation()
            self.inputs(fixture)
            for extra in (('--control', 'control.json'), ('--promote-sota',), ('--result', 'candidate.json', '--promote-sota')):
                self.assertEqual(self.invoke(fixture, 'report', path.name, '--outcome', 'REJECT', '--report', 'report.json', *extra)[0], 2)
                self.assertFalse((path / 'result_evidence.json').exists())
            code, output, error = self.invoke(fixture, 'report', path.name, '--outcome', 'REJECT', '--report', 'report.json', '--result', 'candidate.json')
            self.assertEqual((code, error), (0, ''))
            self.assertFalse(json.loads(output)['promotion']['eligible'])
            self.assertTrue((fixture.root / 'docs/experiments/run-001.md').exists())

    def test_external_result_json_rejects_invalid_metrics_and_legacy_input(self):
        with self.fixture() as fixture:
            path = fixture.evaluation()
            self.inputs(fixture)
            target = fixture.root / 'candidate.json'
            payload = json.loads(target.read_text(encoding='utf-8'))
            payload['damage'] = float('nan')
            target.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(self.invoke(fixture, 'report', path.name, '--outcome', 'REJECT', '--report', 'report.json', '--result', 'candidate.json')[0], 2)
            self.assertEqual(self.invoke(fixture, 'report', path.name, '--outcome', 'REJECT', '--report', 'report.json', '--result', 'runs/old/result.json')[0], 2)
            self.assertFalse((path / 'result_evidence.json').exists())


if __name__ == '__main__':
    unittest.main()
