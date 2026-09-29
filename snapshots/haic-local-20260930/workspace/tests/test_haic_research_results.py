"""Reporting uses temporary registered runs and FakeRunner only."""
from dataclasses import replace

from haic_research.commands import approve_run, register_plan, report_run
from haic_research.models import GateResult, IntegrationReport
from haic_research.records import RecordError, read_integration_report, run_transaction
from haic_research.results import append_experiment_summary, persist_result_evidence, promote_sota, report_experiment
from tests.test_haic_research_commands import CommandFixture, GATES, metadata, research
from tests.test_haic_research_policy import result


class ResultReportingTests(CommandFixture):
    def setUp(self):
        super().setUp()
        self.before = b"# Results\r\n\r\n<!-- BEGIN GENERATED RESULTS -->\r\nold rows\r\n<!-- END GENERATED RESULTS -->\r\n"
        (self.root / 'RESULTS.md').write_bytes(self.before)
        (self.root / 'SOTA.md').write_bytes(b'# Historical SOTA\r\nold pointer\r\n')
        (self.root / 'docs/experiments').mkdir(parents=True)
        (self.root / 'docs/experiments/INDEX.md').write_text('# New experiments\n', encoding='utf-8')
        self.candidate = result('candidate-rev', seed_ids=('s1', 's2'))
        self.control = result('control-rev', seed_ids=('s1', 's2'), completion_count=7)
        self.report = IntegrationReport(tuple(GateResult(name, 'PASS') for name in GATES), ())

    def evaluation(self, *, profile='evaluate_closed_loop', split='held_out', preregister=True):
        meta = metadata()
        meta.update(split_ids=[split], map_ids=['m1'])
        if preregister:
            meta.update(comparison_id='c1', seed_ids=['s1', 's2'], comparison_episode_count=10)
        args = {'policy-checkpoint': str(self.input), 'dynamics-checkpoint': str(self.input)}
        if profile == 'train_policy':
            args = self.args
        if profile == 'benchmark_corridor_diagnostic':
            args = {'track': ['1:42'], 'profile': ['safe']}
        path = register_plan(self.config, meta, profile, args, research=research())
        for stage in ('design', 'execution'):
            approve_run(self.config, path, stage, source_ref='fixture authorization')
        self.execute(path)
        return path

    def record(self, path, outcome='ADVANCE', report=None):
        persist_result_evidence(self.config, path, self.candidate, self.control)
        report_run(self.config, path, outcome, report or replace(self.report, evidence_paths=((path / 'result_evidence.json').relative_to(self.root).as_posix(), (path / 'comparison_evidence.json').relative_to(self.root).as_posix())))

    def promote(self, path, candidate=None, control=None):
        return promote_sota(self.root / 'SOTA.md', candidate or self.candidate, control or self.control, path / 'integration_report.json', config=self.config, run_dir=path)

    def test_append_preserves_generated_bytes_and_duplicate_refuses(self):
        path = self.evaluation()
        self.record(path)
        detail = self.root / 'docs/experiments/run-001.md'
        append_experiment_summary(self.root / 'RESULTS.md', detail, self.candidate, config=self.config, run_dir=path)
        after = (self.root / 'RESULTS.md').read_bytes()
        marker = b'<!-- BEGIN GENERATED RESULTS -->'
        self.assertEqual(after[after.index(marker):], self.before[self.before.index(marker):])
        self.assertIn('integration_report.json', detail.read_text(encoding='utf-8'))
        snapshot = detail.read_bytes()
        with self.assertRaises(RecordError):
            append_experiment_summary(self.root / 'RESULTS.md', detail, self.candidate, config=self.config, run_dir=path)
        self.assertEqual(detail.read_bytes(), snapshot)

    def test_promotes_released_matched_candidate_preserving_history(self):
        path = self.evaluation()
        self.record(path)
        before = (self.root / 'SOTA.md').read_bytes()
        self.assertTrue(self.promote(path).eligible)
        self.assertTrue((self.root / 'SOTA.md').read_bytes().startswith(before))
        with self.assertRaises(RecordError):
            self.promote(path)

    def test_completion_first_rejects_lower_completion(self):
        path = self.evaluation()
        self.candidate = replace(self.candidate, completion_count=6, median_finished_lap_ms=1)
        self.record(path)
        self.assertFalse(self.promote(path).eligible)

    def test_nonadvance_reports_cannot_promote(self):
        for outcome in ('REVISE', 'PIVOT', 'REJECT'):
            with self.subTest(outcome=outcome):
                with ResultReportingTests('test_completion_first_rejects_lower_completion') as fixture:
                    path = fixture.evaluation()
                    fixture.record(path, outcome)
                    self.assertFalse(fixture.promote(path).eligible)

    def __enter__(self):
        self.setUp()
        return self

    def __exit__(self, *args):
        self.doCleanups()

    def test_altered_payload_and_unrelated_report_refused(self):
        path = self.evaluation()
        self.record(path)
        with self.assertRaises(RecordError):
            self.promote(path, replace(self.candidate, damage=0))
        with self.assertRaises(RecordError):
            promote_sota(self.root / 'SOTA.md', self.candidate, self.control, self.root / 'unrelated.json', config=self.config, run_dir=path)

    def test_evidence_is_immutable_and_must_precede_gate_report(self):
        path = self.evaluation()
        self.record(path)
        with self.assertRaises(RecordError):
            persist_result_evidence(self.config, path, self.candidate, self.control)
        with run_transaction(path, config=self.config) as run:
            transaction = run
        with self.assertRaises(RecordError):
            transaction.write_result_evidence({'bad': True})

    def test_gates_independent_split_profile_replication_and_protocol_required(self):
        cases = [('train_policy', 'held_out', True), ('benchmark_corridor_diagnostic', 'held_out', True),
                 ('evaluate_closed_loop', 'train', True), ('evaluate_closed_loop', 'tune', True),
                 ('evaluate_closed_loop', 'held_out', False)]
        for profile, split, preregister in cases:
            with self.subTest(profile=profile, split=split), ResultReportingTests('test_completion_first_rejects_lower_completion') as fixture:
                fixture.candidate = replace(fixture.candidate, split_id=split)
                fixture.control = replace(fixture.control, split_id=split)
                path = fixture.evaluation(profile=profile, split=split, preregister=preregister)
                fixture.record(path)
                self.assertFalse(fixture.promote(path).eligible)
        for change in ({'seed_ids': ('s1',)}, {'rule_compliance': 'UNKNOWN'}, {'eligibility': 'diagnostic'}, {'comparison_id': 'unregistered'}):
            with self.subTest(change=change), ResultReportingTests('test_completion_first_rejects_lower_completion') as fixture:
                fixture.candidate = replace(fixture.candidate, **change)
                fixture.control = replace(fixture.control, **{key: value for key, value in change.items() if key in ('seed_ids', 'comparison_id')})
                path = fixture.evaluation()
                fixture.record(path)
                self.assertFalse(fixture.promote(path).eligible)
        for status in ('FAIL', 'UNKNOWN', 'NOT_APPLICABLE'):
            with self.subTest(status=status), ResultReportingTests('test_completion_first_rejects_lower_completion') as fixture:
                path = fixture.evaluation()
                fixture.record(path, report=replace(fixture.report, gate_results=(GateResult(GATES[0], status), *fixture.report.gate_results[1:])))
                self.assertFalse(fixture.promote(path).eligible)

    def test_unmatched_observation_records_without_promotion(self):
        path = self.evaluation()
        self.control = replace(self.control, map_ids=('other-map',))
        decision = report_experiment(self.config, path, 'REJECT', self.report, self.candidate, self.control, promote=True)
        self.assertFalse(decision.eligible)
        self.assertTrue((self.root / 'docs/experiments/run-001.md').exists())

    def test_paths_are_canonical_and_legacy_rejected_before_read(self):
        path = self.evaluation()
        self.record(path)
        for results, detail in ((self.root / 'runs/old/RESULTS.md', self.root / 'docs/experiments/run-001.md'), (self.root / 'RESULTS.md', self.root / 'submissions/old.md')):
            with self.assertRaises(RecordError):
                append_experiment_summary(results, detail, self.candidate, config=self.config, run_dir=path)
        with self.assertRaises(RecordError):
            promote_sota(self.root / 'elsewhere.md', self.candidate, self.control, path / 'integration_report.json', config=self.config, run_dir=path)

    def test_report_experiment_binds_evidence_before_report_and_reaches_docs(self):
        path = self.evaluation()
        decision = report_experiment(self.config, path, 'ADVANCE', self.report, self.candidate, self.control, promote=True)
        self.assertTrue(decision.eligible)
        report = read_integration_report(path, config=self.config)
        self.assertIn('runs/haic-research-v2/run-001/comparison_evidence.json', report.evidence_paths)
        self.assertTrue((path / 'result_evidence.json').exists())

    def test_report_without_exact_evidence_links_cannot_authorize_results(self):
        path = self.evaluation()
        persist_result_evidence(self.config, path, self.candidate, self.control)
        report_run(self.config, path, 'ADVANCE', self.report)
        self.assertFalse(self.promote(path).eligible)
        self.assertEqual((self.root / 'SOTA.md').read_bytes(), b'# Historical SOTA\r\nold pointer\r\n')

    def test_untrusted_candidate_id_never_changes_document_destinations(self):
        path = self.evaluation()
        self.candidate = replace(self.candidate, candidate_id='../../outside')
        self.record(path)
        self.assertFalse(self.promote(path).eligible)
        append_experiment_summary(self.root / 'RESULTS.md', self.root / 'docs/experiments/run-001.md', self.candidate, config=self.config, run_dir=path)
        self.assertFalse((self.root.parent / 'outside.md').exists())
        with self.assertRaises(RecordError):
            append_experiment_summary(self.root / 'RESULTS.md', self.root / 'docs/experiments/../../outside.md', self.candidate, config=self.config, run_dir=path)
        with self.assertRaises(RecordError):
            promote_sota(self.root / 'SOTA.md', self.candidate, self.control, path / 'integration_report.json', config=self.config, run_dir=self.config.run_root / '../old')

    def test_duplicate_index_or_detail_preflights_without_partial_changes(self):
        for duplicate in ('index', 'detail'):
            with self.subTest(duplicate=duplicate), ResultReportingTests('test_completion_first_rejects_lower_completion') as fixture:
                path = fixture.evaluation()
                fixture.record(path)
                detail = fixture.root / 'docs/experiments/run-001.md'
                index = fixture.root / 'docs/experiments/INDEX.md'
                if duplicate == 'index':
                    index.write_text('<!-- HAIC V2 RUN run-001 -->', encoding='utf-8')
                else:
                    detail.write_text('prior immutable experiment', encoding='utf-8')
                before = (fixture.root / 'RESULTS.md').read_bytes(), index.read_bytes(), detail.read_bytes() if detail.exists() else None
                with self.assertRaises(RecordError):
                    append_experiment_summary(fixture.root / 'RESULTS.md', detail, fixture.candidate, config=fixture.config, run_dir=path)
                after = (fixture.root / 'RESULTS.md').read_bytes(), index.read_bytes(), detail.read_bytes() if detail.exists() else None
                self.assertEqual(before, after)

    def test_result_evidence_redirection_and_hardlink_are_refused(self):
        path = self.evaluation()
        self.record(path)
        evidence = path / 'comparison_evidence.json'
        alias = self.root / 'alias.json'
        import os
        os.link(evidence, alias)
        with self.assertRaises(RecordError):
            self.promote(path)

    def test_missing_evidence_and_mismatched_manifest_ids_cannot_promote(self):
        path = self.evaluation()
        self.candidate = replace(self.candidate, map_ids=('unregistered',))
        self.control = replace(self.control, map_ids=('unregistered',))
        self.record(path)
        self.assertFalse(self.promote(path).eligible)
        (path / 'comparison_evidence.json').unlink()
        with self.assertRaises(RecordError):
            self.promote(path)

    def test_unregistered_denominator_cannot_promote(self):
        path = self.evaluation()
        self.candidate = replace(self.candidate, episode_count=11)
        self.control = replace(self.control, episode_count=11)
        self.record(path)
        self.assertFalse(self.promote(path).eligible)

    def test_invalid_report_does_not_freeze_result_evidence(self):
        path = self.evaluation()
        for outcome, report in (('STOPPED', self.report), ('REJECT', IntegrationReport((), ()))):
            with self.assertRaises((RecordError, ValueError)):
                report_experiment(self.config, path, outcome, report, self.candidate, self.control)
            self.assertFalse((path / 'result_evidence.json').exists())
