"""CLI integration uses temporary files and never invokes a real runner."""

import io
import json
import unittest

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

    def test_unavailable_validation_fails_clearly(self):
        code, text, error = self.invoke('validate')
        self.assertNotEqual(code, 0)
        self.assertIn('unavailable', error.lower())
        self.assertEqual(text, '')

    def test_external_operations_and_path_traversal_absent(self):
        for command in ('submit', 'upload', 'confirm-model'):
            self.assertNotEqual(self.invoke(command)[0], 0)
        self.assertNotEqual(self.invoke('status', '../old')[0], 0)


if __name__ == '__main__':
    unittest.main()
