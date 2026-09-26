"""Command lifecycle contracts: temporary v2 records and fake runners only."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from haic_research import records as record_module

from haic_research.commands import (
    CommandError, UnknownProfileError, approve_run, build_argv, execute_approved,
    load_run_plan, register_plan, replay_run_history, report_run,
)
from haic_research.config import load_config
from haic_research.models import GateResult, GateStatus, Hypothesis, IntegrationReport, WorkflowState
from haic_research.records import (
    RecordError, read_events, read_manifest, read_execution_plan,
    read_integration_report, run_transaction, write_execution_plan,
)
from haic_research.state import ApprovalError

ROOT = Path(__file__).resolve().parents[1]
GATES = ('rule_compliance', 'mechanism_activation', 'competitive_or_product_outcome')


class FakeRunner:
    def __init__(self, returncode=0, error=None):
        self.calls = []
        self.returncode = returncode
        self.error = error

    def __call__(self, argv, **kwargs):
        self.calls.append(dict(argv=argv, **kwargs))
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(argv, self.returncode, 'fake stdout', 'fake stderr')


def metadata(run_id='run-001'):
    return dict(run_id=run_id, cycle_id='cycle-' + run_id, purpose='registered experiment',
                hypothesis_hash='hypothesis', candidate_revision='candidate-rev',
                control_revision='control-rev', candidate_package_hash='candidate-package',
                control_package_hash='control-package', tool_versions={'python': '3'},
                runtime_versions={'engine': 'pinned'}, data_ids=['data-1'], map_ids=['map-1'],
                split_ids=['train'], resource_limits={'seconds': 60},
                permission_limits={'local_only': True}, source_hashes={'training': 'source-hash'})


def research():
    return {name: 'registered ' + name for name in Hypothesis.__dataclass_fields__ if name != 'source_paths'}


class CommandFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'harness.config.json').write_text((ROOT / 'harness.config.json').read_text(encoding='utf-8'), encoding='utf-8')
        self.config = load_config(self.root)
        for folder in ('training', 'haic_agent', 'core', 'core/vendor'):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            (self.root / folder / '__init__.py').write_text('# fake source fixture\n', encoding='utf-8')
        for module in ('train_policy', 'evaluate_closed_loop', 'package_submission', 'benchmark_corridor'):
            (self.root / 'training' / (module + '.py')).write_text('# fake registered module\n', encoding='utf-8')
        for filename in ('agent.py', 'env_wrapper.py', 'damage.py'):
            (self.root / filename).write_text('# fake source fixture\n', encoding='utf-8')
        self.input = self.root / 'training/maps/test.json'
        self.input.parent.mkdir(parents=True)
        for name in ('train-map.json', 'tune-map.json', 'held-map.json'):
            (self.input.parent / name).write_text('{}', encoding='utf-8')
        self.input.write_text(json.dumps({'schema_version': 1,
                              'train': [{'map': 'train-map.json', 'seeds': [1]}],
                              'tune': [{'map': 'tune-map.json', 'seeds': [2]}],
                              'held_out': [{'map': 'held-map.json', 'seeds': [3]}]}), encoding='utf-8')
        self.args = {'train-only-site-map-split': str(self.input), 'defer-tune': True, 'total-steps': 4}
        self.runner = FakeRunner()

    def plan(self, run_id='run-001'):
        return register_plan(self.config, metadata(run_id), 'train_policy', self.args, research=research())

    def approved(self):
        path = self.plan()
        for stage in ('design', 'implementation', 'execution'):
            approve_run(self.config, path, stage, source_ref='user authorization ' + stage)
        return path

    def execute(self, path, **kwargs):
        plan = load_run_plan(self.config, path)
        return execute_approved(self.config, path, plan['plan_hash'], plan['profile_id'], plan['arguments'], runner=kwargs.get('runner', self.runner))

    def alter_events(self, path, edit):
        target = path / 'events.jsonl'
        events = [json.loads(row) for row in target.read_text(encoding='utf-8').splitlines()]
        edit(events)
        target.write_text(''.join(json.dumps(row) + '\n' for row in events), encoding='utf-8')


class CommandProfileTests(CommandFixture):
    def test_unknown_profile_and_arguments_are_rejected(self):
        with self.assertRaises(UnknownProfileError):
            build_argv(self.config, 'arbitrary', {})
        with self.assertRaises(CommandError):
            build_argv(self.config, 'train_policy', {'invented': True})

    def test_typed_values_and_unsafe_paths_are_rejected(self):
        for change in ({'total-steps': 0}, {'total-steps': True}, {'total-steps': '4'},
                       {'resume': '../outside'}, {'resume': 'submissions/old.pt'},
                       {'resume': 'runs/old/checkpoint.pt'}, {'resume': '--oops'},
                       {'resume': 'x;echo'}, {'resume': 'missing.pt'}, {'learning-rate': float('nan')}):
            with self.subTest(change=change), self.assertRaises(CommandError):
                build_argv(self.config, 'train_policy', dict(self.args, **change))

    def test_output_is_forced_per_run_and_cannot_escape(self):
        path = self.plan()
        plan = load_run_plan(self.config, path)
        self.assertEqual(plan['arguments']['output'], str(self.config.artifact_root / 'run-001' / 'training'))
        with self.assertRaises(CommandError):
            register_plan(self.config, metadata('bad-output'), 'train_policy', dict(self.args, output=str(self.root / 'outside')), research=research())

    def test_required_checkpoint_inputs_and_file_outputs(self):
        for profile, filename in (('evaluate_closed_loop', 'evaluation'), ('package_submission', 'submission.zip')):
            with self.subTest(profile=profile):
                with self.assertRaises(CommandError):
                    register_plan(self.config, metadata(profile), profile, {}, research=research())
                path = register_plan(self.config, metadata(profile), profile,
                                     {'policy-checkpoint': str(self.input), 'dynamics-checkpoint': str(self.input)}, research=research())
                self.assertEqual(load_run_plan(self.config, path)['arguments']['output'], str(self.config.artifact_root / profile / filename))

    def test_diagnostic_repeated_real_flags_and_ineligibility(self):
        path = register_plan(self.config, metadata('diagnostic'), 'benchmark_corridor_diagnostic',
                             {'track': ['1:42', '2:101'], 'site-map': [str(self.input)], 'profile': ['safe', 'fast']}, research=research())
        plan = load_run_plan(self.config, path)
        argv = build_argv(self.config, plan['profile_id'], plan['arguments'])
        self.assertEqual(argv.count('--track'), 2)
        self.assertEqual(argv.count('--profile'), 2)
        self.assertFalse(plan['profile']['sota_eligible'])

    def test_missing_research_fields_are_rejected_before_record_creation(self):
        with self.assertRaises(CommandError):
            register_plan(self.config, metadata(), 'train_policy', self.args, research={'source_ref': 'user'})
        self.assertFalse(self.config.run_root.exists())


class CommandLifecycleTests(CommandFixture):
    def test_actual_module_shared_dependency_and_source_inventory_drift_refuse_runner(self):
        mutations = (
            ('training/train_policy.py', 'changed'),
            ('haic_agent/__init__.py', 'changed'),
            ('core/vendor/__init__.py', 'changed'),
            ('damage.py', 'changed'),
            ('training/new_local_helper.py', 'added'),
            ('haic_agent/__init__.py', None),
        )
        for index, (relative, value) in enumerate(mutations):
            with self.subTest(relative=relative):
                path = self.plan('source-' + str(index))
                for stage in ('design', 'implementation', 'execution'):
                    approve_run(self.config, path, stage, source_ref='user')
                if value is None:
                    (self.root / relative).unlink()
                else:
                    (self.root / relative).write_text('# ' + value + '\n', encoding='utf-8')
                manifest = read_manifest(path, config=self.config)
                with self.assertRaises(CommandError):
                    execute_approved(self.config, path, manifest.plan_hash, 'train_policy', self.args, runner=self.runner)
                self.assertFalse((path / '.execution.claim').exists())
        self.assertEqual(self.runner.calls, [])

    def test_explicit_package_source_root_is_bound_to_approval(self):
        package_root = self.root / 'selected-package'
        for relative in ('agent.py', 'haic_agent/__init__.py', 'haic_agent/observation.py',
                         'haic_agent/pixel_features.py', 'haic_agent/networks.py', 'haic_agent/dynamics.py',
                         'haic_agent/planner.py', 'haic_agent/runtime_config.py'):
            filename = package_root / relative
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_text('# selected package fixture\n', encoding='utf-8')
        args = {'policy-checkpoint': str(self.input), 'dynamics-checkpoint': str(self.input), 'source-root': str(package_root)}
        path = register_plan(self.config, metadata('package-source'), 'package_submission', args, research=research())
        for stage in ('design', 'implementation', 'execution'):
            approve_run(self.config, path, stage, source_ref='user')
        manifest = read_manifest(path, config=self.config)
        (package_root / 'agent.py').write_text('# changed selected source\n', encoding='utf-8')
        with self.assertRaises(CommandError):
            execute_approved(self.config, path, manifest.plan_hash, 'package_submission', args, runner=self.runner)
        self.assertFalse((path / '.execution.claim').exists())
        self.assertEqual(self.runner.calls, [])

    def test_interpreter_identity_drift_refuses_runner(self):
        path = self.approved()
        manifest = read_manifest(path, config=self.config)
        with patch('haic_research.commands.sys.version', 'different-runtime'):
            with self.assertRaises(CommandError):
                execute_approved(self.config, path, manifest.plan_hash, 'train_policy', self.args, runner=self.runner)
        self.assertEqual(self.runner.calls, [])

    def test_source_hardlink_is_refused_before_opening_linked_content(self):
        import os
        source = self.root / 'training/train_policy.py'
        old = self.root / 'runs/old/source.py'
        old.parent.mkdir(parents=True)
        old.write_text('# forbidden historical source\n', encoding='utf-8')
        source.unlink()
        os.link(old, source)
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('unguarded source read')):
            with self.assertRaises((CommandError, RecordError)):
                self.plan()

    def test_split_references_reject_legacy_targets_without_opening_them(self):
        split = self.root / 'split.json'
        for index, forbidden in enumerate(('runs/old/map.json', 'artifacts/haic/map.json', 'submissions/map.json')):
            split.write_text(json.dumps({'schema_version': 1, 'train': [{'map': forbidden, 'seeds': [1]}]}), encoding='utf-8')
            with self.subTest(forbidden=forbidden), patch('haic_research.records._read_text', wraps=record_module._read_text) as reader:
                with self.assertRaises(CommandError):
                    register_plan(self.config, metadata('legacy-' + str(index)), 'train_policy', dict(self.args, **{'train-only-site-map-split': str(split)}), research=research())
                self.assertTrue(all(str(call.args[0]) != str(self.root / forbidden) for call in reader.call_args_list))
        self.assertEqual(self.runner.calls, [])

    def test_split_mutation_after_approval_refuses_before_execution_claim(self):
        # Put the split at repository root so the module's relative confinement
        # would otherwise allow its historical target.
        split = self.root / 'split.json'
        split.write_text(json.dumps({'schema_version': 1, 'train': [{'map': 'training/maps/train-map.json', 'seeds': [1]}]}), encoding='utf-8')
        args = dict(self.args, **{'train-only-site-map-split': str(split)})
        path = register_plan(self.config, metadata('changed-split'), 'train_policy', args, research=research())
        for stage in ('design', 'implementation', 'execution'):
            approve_run(self.config, path, stage, source_ref='user')
        manifest = read_manifest(path, config=self.config)
        split.write_text(json.dumps({'schema_version': 1, 'train': [{'map': 'runs/old/map.json', 'seeds': [1]}]}), encoding='utf-8')
        with self.assertRaises(CommandError):
            execute_approved(self.config, path, manifest.plan_hash, 'train_policy', args, runner=self.runner)
        self.assertFalse((path / '.execution.claim').exists())
        self.assertEqual(self.runner.calls, [])

    def test_split_map_hardlink_is_refused_without_opening_target(self):
        import os
        safe_name = self.root / 'training/maps/linked-map.json'
        forbidden = self.root / 'runs/old/map.json'
        forbidden.parent.mkdir(parents=True)
        forbidden.write_text('{}', encoding='utf-8')
        os.link(forbidden, safe_name)
        self.input.write_text(json.dumps({'schema_version': 1, 'train': [{'map': 'linked-map.json', 'seeds': [1]}]}), encoding='utf-8')
        with patch('haic_research.records._read_text', wraps=record_module._read_text) as reader:
            with self.assertRaises(CommandError):
                self.plan()
            self.assertTrue(all(call.args[0] not in {safe_name, forbidden} for call in reader.call_args_list))

    def test_normal_split_loaders_check_all_groups_and_train_only_ignores_unloaded_groups(self):
        split = self.root / 'split.json'
        split.write_text(json.dumps({'schema_version': 1,
                                    'train': [{'map': 'training/maps/train-map.json', 'seeds': [1]}],
                                    'tune': [{'map': 'runs/old/tune-map.json', 'seeds': [2]}],
                                    'held_out': [{'map': 'runs/old/held-map.json', 'seeds': [3]}]}), encoding='utf-8')
        register_plan(self.config, metadata('train-only'), 'train_policy', dict(self.args, **{'train-only-site-map-split': str(split)}), research=research())
        for profile in ('evaluate_closed_loop', 'benchmark_corridor_diagnostic'):
            with self.subTest(profile=profile), self.assertRaises(CommandError):
                args = {'site-map-split': str(split)}
                if profile == 'evaluate_closed_loop':
                    args.update({'policy-checkpoint': str(self.input), 'dynamics-checkpoint': str(self.input)})
                register_plan(self.config, metadata(profile), profile, args, research=research())

    def test_new_persistence_helpers_confine_and_preserve_immutable_plan(self):
        path = self.plan()
        payload = read_execution_plan(path, config=self.config)
        with self.assertRaises(RecordError):
            write_execution_plan(path, payload, config=self.config)
        self.assertEqual(read_execution_plan(path, config=self.config), payload)
        for location in (self.root, self.root / 'runs/old', path / 'nested'):
            with self.assertRaises(RecordError):
                write_execution_plan(location, payload, config=self.config)
        with run_transaction(path, config=self.config) as run:
            pass
        with self.assertRaises(RecordError):
            run.write_plan(payload)
        with self.assertRaises(RecordError):
            run.append(replace(read_events(path, config=self.config)[0], event_id='after-lock'))
        with self.assertRaises(RecordError):
            run.write_report(IntegrationReport(tuple(GateResult(name, GateStatus.PASS) for name in GATES), ()))

    def test_execution_claim_must_identify_the_exact_started_event_and_plan(self):
        path = self.approved()
        self.execute(path)
        claim = path / '.execution.claim'
        original = json.loads(claim.read_text(encoding='utf-8'))
        for edit in ({'event_id': 'absent'}, {'plan_hash': 'other-run'}):
            claim.write_text(json.dumps(dict(original, **edit)), encoding='utf-8')
            with self.assertRaises(CommandError):
                replay_run_history(self.config, path)

    def test_complete_valid_lifecycle_preserves_immutable_plan(self):
        path = self.approved()
        before = (path / 'execution_plan.json').read_bytes()
        result = self.execute(path)
        self.assertEqual(result.returncode, 0)
        call = self.runner.calls[0]
        self.assertEqual(call['argv'][:3], [sys.executable, '-m', 'training.train_policy'])
        self.assertEqual(call['cwd'], self.root)
        self.assertFalse(call['shell'])
        self.assertGreater(call['timeout'], 0)
        self.assertEqual(replay_run_history(self.config, path).state, WorkflowState.EVALUATE)
        report_run(self.config, path, 'ADVANCE', IntegrationReport(tuple(GateResult(name, GateStatus.PASS, 'observed', ('evidence.json',)) for name in GATES), ('evidence.json',)))
        history = replay_run_history(self.config, path)
        self.assertTrue(history.released)
        self.assertEqual(history.state, WorkflowState.STOPPED)
        self.assertEqual((path / 'execution_plan.json').read_bytes(), before)
        self.assertEqual(read_manifest(path, config=self.config).approval_hash, '')

    def test_missing_skipped_and_duplicate_approval_stages_are_rejected(self):
        path = self.plan()
        with self.assertRaises(ApprovalError):
            self.execute(path)
        with self.assertRaises(ApprovalError):
            approve_run(self.config, path, 'execution', source_ref='user')
        approve_run(self.config, path, 'design', source_ref='user')
        with self.assertRaises(ApprovalError):
            approve_run(self.config, path, 'design', source_ref='user')
        self.assertEqual(self.runner.calls, [])

    def test_approval_needs_authorization_reference(self):
        with self.assertRaises(ApprovalError):
            approve_run(self.config, self.plan(), 'design', source_ref='')

    def test_changed_arguments_hash_profile_or_metadata_are_rejected(self):
        path = self.approved()
        plan = load_run_plan(self.config, path)
        for hash_value, profile, args in ((plan['plan_hash'], 'train_policy', dict(plan['arguments'], **{'total-steps': 8})),
                                         ('stale', 'train_policy', plan['arguments']),
                                         (plan['plan_hash'], 'package_submission', plan['arguments'])):
            with self.subTest(profile=profile, hash=hash_value), self.assertRaises(CommandError):
                execute_approved(self.config, path, hash_value, profile, args, runner=self.runner)
        target = path / 'run_manifest.json'
        data = json.loads(target.read_text(encoding='utf-8'))
        data['candidate_revision'] = 'changed-source'
        target.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaises(CommandError):
            self.execute(path)
        self.assertEqual(self.runner.calls, [])

    def test_current_profile_drift_invalidates_approved_plan(self):
        path = self.approved()
        profiles = json.loads(json.dumps(self.config.command_profiles))
        profiles['train_policy']['timeout_seconds'] += 1
        changed = replace(self.config, command_profiles=profiles)
        with self.assertRaises(CommandError):
            load_run_plan(changed, path)

    def test_cross_run_approval_log_is_rejected(self):
        approved = self.approved()
        other = self.plan('run-002')
        (other / 'events.jsonl').write_bytes((approved / 'events.jsonl').read_bytes())
        with self.assertRaises((CommandError, RecordError)):
            self.execute(other)

    def test_malformed_duplicate_wrong_hash_order_and_reference_events_fail_closed(self):
        changes = [lambda rows: rows.append(rows[-1]),
                   lambda rows: rows[-1].update(timestamp='2020-01-01T00:00:00Z'),
                   lambda rows: rows[-1].update(timestamp='2026-09-26T00:00:00'),
                   lambda rows: rows[-1].update(approved_plan_hash='stale'),
                   lambda rows: rows[-1].update(approval_stage='design'),
                   lambda rows: rows[5].update(approval_ref='absent'),
                   lambda rows: rows[5].update(approval_ref=rows[0]['event_id'])]
        for index, edit in enumerate(changes):
            with self.subTest(index=index):
                path = self.plan('bad-' + str(index))
                for stage in ('design', 'implementation', 'execution'):
                    approve_run(self.config, path, stage, source_ref='user')
                self.alter_events(path, edit)
                with self.assertRaises((CommandError, RecordError, ApprovalError)):
                    self.execute(path)
        malformed = self.plan('malformed')
        with (malformed / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('{\n')
        with self.assertRaises((CommandError, RecordError)):
            self.execute(malformed)
        self.assertEqual(self.runner.calls, [])

    def test_repeated_failed_and_timed_out_execution_are_consumed(self):
        for index, runner in enumerate((FakeRunner(2), FakeRunner(error=subprocess.TimeoutExpired(['fake'], 1)), FakeRunner(error=OSError('fake failure')))):
            with self.subTest(index=index):
                path = self.plan('failure-' + str(index))
                for stage in ('design', 'implementation', 'execution'):
                    approve_run(self.config, path, stage, source_ref='user')
                result = self.execute(path, runner=runner)
                self.assertFalse(result.succeeded)
                self.assertEqual(read_events(path, config=self.config)[-1].kind, 'EXECUTION_FAILED')
                with self.assertRaises(CommandError):
                    self.execute(path, runner=runner)
                self.assertEqual(len(runner.calls), 1)

    def test_concurrent_invocation_reserves_once(self):
        path = self.approved()
        entered, finish = threading.Event(), threading.Event()
        results = []
        runner = FakeRunner()
        def blocked(argv, **kwargs):
            entered.set()
            finish.wait(5)
            return runner(argv, **kwargs)
        thread = threading.Thread(target=lambda: results.append(self.execute(path, runner=blocked)))
        thread.start()
        self.assertTrue(entered.wait(5))
        try:
            with self.assertRaises((CommandError, RecordError)):
                self.execute(path)
        finally:
            finish.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertEqual(len(runner.calls), 1)

    def test_non_advance_and_non_passing_gates_never_release(self):
        for index, (outcome, status) in enumerate((('REVISE', GateStatus.PASS), ('PIVOT', GateStatus.PASS), ('REJECT', GateStatus.PASS), ('ADVANCE', GateStatus.UNKNOWN))):
            path = self.plan('report-' + str(index))
            for stage in ('design', 'implementation', 'execution'):
                approve_run(self.config, path, stage, source_ref='user')
            self.execute(path)
            report_run(self.config, path, outcome, IntegrationReport(tuple(GateResult(name, status) for name in GATES), ()))
            history = replay_run_history(self.config, path)
            self.assertFalse(history.released)
            self.assertEqual(history.outcome, WorkflowState(outcome))
            self.assertEqual(history.state, WorkflowState.STOPPED)

    def test_fake_state_labels_cannot_replace_execution_start_and_finish(self):
        path = self.approved()
        self.execute(path)
        self.alter_events(path, lambda rows: rows[-1].update(kind='STATE'))
        with self.assertRaises(CommandError):
            replay_run_history(self.config, path)

    def test_successful_execution_requires_zero_exit_evidence(self):
        path = self.approved()
        self.execute(path)
        self.alter_events(path, lambda rows: rows[-1].update(resource_usage={'returncode': 4}))
        with self.assertRaises(CommandError):
            replay_run_history(self.config, path)


if __name__ == '__main__':
    unittest.main()
