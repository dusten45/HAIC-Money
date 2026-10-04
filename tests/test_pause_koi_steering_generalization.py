"""Synthetic files/processes/flocks only; no real operator, Agent, or environment."""

import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

from scripts import pause_koi_steering_generalization as pause


def save(path, data):
    path.write_text(json.dumps(data) + "\n")


class SyntheticRun:
    def __init__(self, root):
        self.root = root
        self.run = root / "runs" / pause.STUDY
        self.run.mkdir(parents=True)
        (root / "experiments/train-seed-claims").mkdir(parents=True)
        self.review = dict(path=str(root / "experiments/review.json"), sha256="r" * 64)
        save(self.run / "execution-review.json", self.review)
        self.protocol: dict[str, Any] = dict(schedule=[dict(mode=arm, track_id=1, seed=3184000001, status="unrun")
                                      for arm in ("crossing_projection", "steering_release_v1")])
        self.slot = self.protocol["schedule"][0]
        self.output = self.run / "1-3184000001-crossing_projection.json"
        raw = self.output.with_suffix(".raw.jsonl")
        raw.write_text('{"step":1,"t":1.02}\n')
        decisions = self.output.with_suffix(".decisions.jsonl")
        decisions.write_text('{"status":"act_intent"}\n{"status":"act_returned"}\n')
        episode = dict(mode=self.slot["mode"], track_id=1, seed=3184000001, error=None,
                       invalid_actions=0, retire_reason="finished", raw_trace_file=raw.name,
                       raw_trace_sha256=pause.sha(raw), partial_decisions_file=decisions.name,
                       partial_decisions_sha256=pause.sha(decisions))
        save(self.output, episode)
        original = self.output.with_suffix(".process.json")
        save(original, dict(status="completed", error=None, wall_time_s=1.0))
        bound = self.output.with_suffix(".bound-process.json")
        save(bound, dict(status="completed", error=None, slot_id=pause.slot_id(self.slot),
                         operator_sha256=pause.OPERATOR_SHA, original_process_file=original.name,
                         original_process_sha256=pause.sha(original)))
        self.output.with_suffix(".stdout.txt").write_text("")
        self.output.with_suffix(".stderr.txt").write_text("")
        self.row = dict(self.slot, status="completed", file=self.output.name, sha256=pause.sha(self.output),
                        process_file=bound.name, process_sha256=pause.sha(bound), slot_id=pause.slot_id(self.slot),
                        error=None, invalid_actions=0)
        self.intent = dict(status="reset_intent", arm=self.slot["mode"], track=1, seed=3184000001,
                           slot_id=pause.slot_id(self.slot), time=1.0)
        self.write_ledger([self.intent, self.row])

    def write_ledger(self, rows):
        (self.run / "reset-ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def terminal(self):
        rows = [self.row, dict(self.protocol["schedule"][1], status="operator_error",
                               error="KeyboardInterrupt: ", partial_artifacts=[])]
        save(self.run / "episode-report.json", dict(protocol_sha256=pause.PROTOCOL_SHA, rows=rows,
             complete=False, operator_error="KeyboardInterrupt: ",
             reset_ledger_sha256=pause.sha(self.run / "reset-ledger.jsonl"), review_receipt=self.review))
        save(self.run / "operator-failure.json", dict(error="KeyboardInterrupt: "))

    def sources(self):
        scripts = self.root / 'scripts'
        scripts.mkdir()
        operator = scripts / 'evaluate_koi_steering_generalization.py'
        operator.write_text('# synthetic frozen source only\n')
        finalizer = scripts / 'finalize_koi_steering_generalization.py'
        finalizer.write_text('# synthetic frozen finalizer only\n')
        for name in ('candidate.zip', 'crossing-projection-source-reconstruction.zip', 'candidate.manifest.json'):
            (self.run/name).write_text('synthetic bytes only')
        guard = self.root/'experiments'/f'{pause.STUDY}-evidence-guard.json'
        audit = self.root/'experiments/audit.json'
        save(guard, {'synthetic': True})
        save(audit, {'synthetic': True})
        copied = self.run/'copy.py'
        copied.write_bytes(operator.read_bytes())
        schedule = []
        for seed in range(3184000001,3184000025):
            for track in (1,2,3):
                arms = ['crossing_projection','steering_release_v1']
                if not (track+seed)%2:
                    arms.reverse()
                schedule.extend(dict(mode=a,track_id=track,seed=seed,status='unrun') for a in arms)
        protocol = dict(study=pause.STUDY, episodes=144, schedule=schedule,
                        operator_sha256=pause.sha(operator), analyzer_sha256='a'*64,
                        helper_source_sha256={str(operator):pause.sha(operator)},
                        environment_source_sha256={}, root_agent_source_sha256={},
                        reference_evidence={'source_sha256':{}},
                        source_copies=[dict(file=copied.name,sha256=pause.sha(copied))],
                        model_hashes={'crossing_projection':pause.sha(self.run/'crossing-projection-source-reconstruction.zip'),
                                      'steering_release_v1':pause.sha(self.run/'candidate.zip')},
                        candidate_manifest_sha256=pause.sha(self.run/'candidate.manifest.json'),
                        freshness={'audit_receipt':{'path':'experiments/audit.json','sha256':pause.sha(audit)}})
        save(self.root/'experiments'/f'{pause.STUDY}.json',protocol)
        save(self.run/'protocol.json',protocol)
        protocol_sha = pause.sha(self.run/'protocol.json')
        review_path = Path(self.review['path'])
        save(review_path, dict(status='passed',environment_resets=0,protocol_sha256=protocol_sha,
                              operator_sha256=pause.sha(operator),analyzer_sha256='a'*64,
                              audit_sha256=pause.sha(audit),finalizer_sha256=pause.sha(finalizer),
                              evidence_guard_sha256=pause.sha(guard)))
        self.review['sha256'] = pause.sha(review_path)
        save(self.run/'execution-review.json',self.review)
        return protocol_sha, pause.sha(operator)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pause-koi-test-", dir="/tmp/kilo")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = SyntheticRun(Path(self.temporary.name))

    def test_completed_prefix_rehashes_every_episode_artifact(self):
        f = self.fixture
        completed, pins, digest = pause.completed_evidence(f.run, f.protocol)
        self.assertEqual(completed, [f.row])
        self.assertEqual(len(pins), 7)
        self.assertEqual(digest, pause.sha(f.run / "reset-ledger.jsonl"))

    def test_each_completed_artifact_mutation_is_rejected(self):
        f = self.fixture
        for suffix in pause.SUFFIXES[:5]:
            with self.subTest(suffix=suffix):
                path = f.output.with_suffix(suffix)
                raw = path.read_bytes()
                path.write_bytes(raw + b" ")
                with self.assertRaises(ValueError):
                    pause.completed_evidence(f.run, f.protocol)
                path.write_bytes(raw)

    def test_pending_intent_without_raw_is_never_zero_reset(self):
        f = self.fixture
        f.write_ledger([f.intent, f.row, dict(status="reset_intent")])
        with self.assertRaisesRegex(ValueError, "completed-episode boundary"):
            pause.completed_evidence(f.run, f.protocol)

    def test_torn_ledger_is_rejected(self):
        path = self.fixture.run / "reset-ledger.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        with self.assertRaisesRegex(ValueError, "torn"):
            pause.ledger_rows(self.fixture.run)

    def test_unexecuted_empty_raw_is_ambiguous(self):
        f = self.fixture
        (f.run / "1-3184000001-steering_release_v1.raw.jsonl").touch()
        with self.assertRaisesRegex(ValueError, "already has artifacts"):
            pause.completed_evidence(f.run, f.protocol)

    def test_rehashed_wrong_identity_and_nonzero_exit_are_rejected(self):
        f = self.fixture
        bound = f.output.with_suffix(".bound-process.json")
        original = bound.read_bytes()
        for key, value in (("slot_id", "wrong"), ("operator_sha256", "bad"),
                           ("status", "operator_error"), ("error", "child_exit_-2")):
            with self.subTest(key=key):
                data = json.loads(original)
                data[key] = value
                save(bound, data)
                f.row["process_sha256"] = pause.sha(bound)
                f.write_ledger([f.intent, f.row])
                with self.assertRaises(ValueError):
                    pause.completed_evidence(f.run, f.protocol)
        bound.write_bytes(original)

    def test_terminal_requires_keyboardinterrupt_and_unchanged_ledger(self):
        f = self.fixture
        f.terminal()
        completed, _, digest = pause.completed_evidence(f.run, f.protocol)
        terminal = pause.terminal_evidence(f.run, f.protocol, completed, digest, f.review)
        self.assertTrue(terminal["process_exited"])
        report = pause.read_json(f.run / "episode-report.json")
        report["reset_ledger_sha256"] = "bad"
        save(f.run / "episode-report.json", report)
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            pause.terminal_evidence(f.run, f.protocol, completed, digest, f.review)

    def test_terminal_rejects_preintent_partial_exposure(self):
        f = self.fixture
        f.terminal()
        report = pause.read_json(f.run / "episode-report.json")
        report["rows"][1]["partial_artifacts"] = [dict(file="untrusted")]
        save(f.run / "episode-report.json", report)
        with self.assertRaisesRegex(ValueError, "partial child evidence"):
            pause.terminal_evidence(f.run, f.protocol, [f.row],
                                    pause.sha(f.run / "reset-ledger.jsonl"), f.review)

    def test_receipt_verifier_rejects_rehashed_false_proof(self):
        f = self.fixture
        receipt = f.run / "boundary-pause.json"
        for data in (dict(format=pause.FORMAT, status="refused"),
                     dict(format=pause.FORMAT, status="paused", coordinator_sha256="bad")):
            save(receipt, data)
            with self.assertRaises(ValueError):
                pause.verify_boundary(receipt, pause.sha(receipt), root=f.root)

    def test_exact_frozen_source_required_without_imports(self):
        with self.assertRaisesRegex(ValueError, "exact original frozen protocol"):
            pause.source_evidence(self.fixture.root, "0" * 64)

    def test_source_closure_rehashes_protocol_models_review_and_copies(self):
        f = self.fixture
        protocol_sha, operator_sha = f.sources()
        with patch.object(pause,'PROTOCOL_SHA',protocol_sha), patch.object(pause,'OPERATOR_SHA',operator_sha):
            protocol, pins = pause.source_evidence(f.root,protocol_sha)
            self.assertEqual(len(protocol['schedule']),144)
            self.assertEqual(len(pins),12)
            for name in ('candidate.zip','copy.py','candidate.manifest.json'):
                with self.subTest(name=name):
                    path = f.run/name
                    raw = path.read_bytes()
                    path.write_bytes(raw+b'changed')
                    with self.assertRaisesRegex(ValueError,'frozen evidence changed'):
                        pause.source_evidence(f.root,protocol_sha)
                    path.write_bytes(raw)

    def test_parent_start_tick_mismatch_refused(self):
        f = self.fixture
        identity = dict(cmdline=['python','-B','-m','scripts.evaluate_koi_steering_generalization','--run',
                                '--protocol-sha256',pause.PROTOCOL_SHA,'--review',f.review['path'],
                                '--review-sha256',f.review['sha256']],
                        start_ticks=123,cwd=str(f.root))
        with self.assertRaisesRegex(ValueError,'start time'):
            pause.verify_parent(identity,124,f.root,{'operator_sha256':pause.OPERATOR_SHA})

    def test_missing_public_pidfd_support_fails_before_source_or_lock(self):
        with patch.object(pause,'source_evidence') as sources, \
             patch.object(pause,'os') as mocked_os:
            del mocked_os.pidfd_open
            with self.assertRaisesRegex(ValueError,'public pidfd APIs'):
                pause.pause(1,1,self.fixture.run/'boundary-pause.json',root=self.fixture.root)
            sources.assert_not_called()

    def test_invalid_receipt_location_rejected_before_lock_or_signal(self):
        with patch.object(pause.signal, "pidfd_send_signal", create=True) as send:
            with self.assertRaisesRegex(ValueError, "canonical run"):
                pause.pause(os.getpid(), 0, self.fixture.run / "protocol.json", root=self.fixture.root)
            send.assert_not_called()

    def test_wrapper_pid_refused(self):
        identity = dict(cmdline=["/bin/bash", "-c", "python -m ..."])
        with self.assertRaisesRegex(ValueError, "do not target Bash"):
            pause.verify_parent(identity, 0, self.fixture.root, {})

    def test_direct_kernel_owner_waiter_proof_never_reads_protected_syscall(self):
        registry = self.fixture.root/'experiments/train-seed-claims'
        descriptor = os.open(registry,os.O_RDONLY|os.O_DIRECTORY)
        self.addCleanup(os.close,descriptor)
        info = os.fstat(descriptor)
        token = f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'
        owner = f'7: FLOCK ADVISORY WRITE {os.getpid()} {token} 0 EOF'
        waiter = f'7: -> FLOCK ADVISORY WRITE 123456 {token} 0 EOF'
        reads = []

        def read(path,*args,**kwargs):
            reads.append(str(path))
            self.assertEqual(str(path),'/proc/locks')
            return owner+'\n'+waiter+'\n'

        with patch.object(Path,'read_text',read), patch.object(pause,'children',return_value=[]):
            proof = pause.blocked_flock(123456,descriptor,registry)
        self.assertEqual(reads,['/proc/locks'])
        self.assertEqual(proof['proof_mode'],pause.PROOF_MODE)
        self.assertNotIn('blocked_parent_syscall',proof)
        self.assertNotIn('parent_fd',proof)

    def test_kernel_owner_waiter_rejects_wrong_identity_kind_and_file(self):
        registry = self.fixture.root/'experiments/train-seed-claims'
        descriptor = os.open(registry,os.O_RDONLY|os.O_DIRECTORY)
        self.addCleanup(os.close,descriptor)
        info = os.fstat(descriptor)
        token = f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'
        owner = ['7:','FLOCK','ADVISORY','WRITE',str(os.getpid()),token,'0','EOF']
        waiter = ['7:','->','FLOCK','ADVISORY','WRITE','123456',token,'0','EOF']
        cases = [(0,4,'999999'),(1,5,'999999'),(0,1,'POSIX'),(1,2,'OFDLCK'),
                 (0,3,'READ'),(1,4,'READ'),(0,5,'00:00:0'),(1,6,'00:00:0'),
                 (0,6,'1'),(1,8,'1'),(1,0,'8:')]
        for row,index,value in cases:
            with self.subTest(row=row,index=index,value=value):
                lines = [owner.copy(),waiter.copy()]
                lines[row][index] = value
                with patch.object(Path,'read_text',return_value='\n'.join(' '.join(r) for r in lines)), \
                     patch.object(pause,'children',return_value=[]):
                    with self.assertRaises(ValueError):
                        pause.blocked_flock(123456,descriptor,registry)
        for lines in ('',' '.join(owner),' '.join(waiter)):
            with self.subTest(missing_entry=lines):
                with patch.object(Path,'read_text',return_value=lines), \
                     patch.object(pause,'children',return_value=[]):
                    with self.assertRaises(ValueError):
                        pause.blocked_flock(123456,descriptor,registry)

    def test_unreadable_kernel_locks_has_no_alternative_proof(self):
        registry = self.fixture.root/'experiments/train-seed-claims'
        descriptor = os.open(registry,os.O_RDONLY|os.O_DIRECTORY)
        self.addCleanup(os.close,descriptor)
        with patch.object(Path,'read_text',side_effect=PermissionError('denied')) as read:
            with self.assertRaises(PermissionError):
                pause.blocked_flock(123456,descriptor,registry)
        self.assertEqual(read.call_count,1)

    def test_kernel_proof_rejects_registry_inode_replacement(self):
        registry = self.fixture.root/'experiments/train-seed-claims'
        descriptor = os.open(registry,os.O_RDONLY|os.O_DIRECTORY)
        self.addCleanup(os.close,descriptor)
        registry.rename(registry.with_name('old-registry'))
        registry.mkdir()
        with self.assertRaisesRegex(ValueError,'inode changed'):
            pause.blocked_flock(123456,descriptor,registry)


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open") and
                     hasattr(signal, "pidfd_send_signal"), "requires Linux pidfds")
class KernelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pause-koi-kernel-", dir="/tmp/kilo")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def launch(self, code, *args):
        process = subprocess.Popen([sys.executable, "-u", "-c", code, *map(str, args)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.cleanup_process, process)
        assert process.stdout is not None and process.stderr is not None
        return process

    @staticmethod
    def cleanup_process(process):
        if process.poll() is None:
            process.kill()  # Only test-owned synthetic processes; never real operator PIDs.
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    def test_kernel_proof_and_parent_only_sigint_at_blocked_flock(self):
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        process = self.launch("""
import fcntl,os,sys
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_DIRECTORY)
print('before flock',flush=True)
try:
    fcntl.flock(fd,fcntl.LOCK_EX)
    raise RuntimeError('unexpected reset permission')
except KeyboardInterrupt:
    print('KeyboardInterrupt boundary',flush=True)
finally:
    os.close(fd)
""", self.directory)
        assert process.stdout is not None
        self.assertEqual(process.stdout.readline().strip(), "before flock")
        pidfd = os.pidfd_open(process.pid)
        self.addCleanup(os.close, pidfd)
        # One bounded pidfd wait lets the synthetic process enter the syscall.
        self.assertEqual(select.select([pidfd], [], [], .1)[0], [])
        proof = pause.blocked_flock(process.pid, descriptor, self.directory)
        self.assertEqual(len(proof["kernel_lock_lines"]), 2)
        self.assertEqual(proof["proof_mode"],pause.PROOF_MODE)
        signal.pidfd_send_signal(pidfd, signal.SIGINT)
        pause.wait_exit(pidfd, timeout=5)
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertEqual(process.stdout.readline().strip(), "KeyboardInterrupt boundary")
        self.assertFalse((self.directory / "reset").exists())

    def test_wrong_directory_lock_refuses_signal(self):
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        process = self.launch("print('ready',flush=True)")
        process.wait(timeout=5)
        with self.assertRaises(ValueError):
            pause.blocked_flock(process.pid, descriptor, self.directory)

    def test_natural_child_exit_while_reset_lock_held(self):
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        process = subprocess.Popen([sys.executable, "-c", "import os,sys;os.read(int(sys.argv[1]),1)",
                                    str(read_fd)], pass_fds=(read_fd,))
        self.addCleanup(lambda: process.wait(timeout=5))
        pidfd = os.pidfd_open(process.pid)
        self.addCleanup(os.close, pidfd)
        os.write(write_fd, b"x")
        pause.wait_exit(pidfd, timeout=5)
        self.assertEqual(process.wait(timeout=5), 0)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_pidfd_wait_timeout_sends_nothing(self):
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        process = subprocess.Popen([sys.executable, "-c", "import os,sys;os.read(int(sys.argv[1]),1)",
                                    str(read_fd)], pass_fds=(read_fd,))
        pidfd = os.pidfd_open(process.pid)
        self.addCleanup(os.close, pidfd)
        try:
            with patch.object(pause.signal, "pidfd_send_signal") as send:
                with self.assertRaisesRegex(ValueError, "process exit not observed"):
                    pause.wait_exit(pidfd, timeout=.01)
                send.assert_not_called()
        finally:
            os.write(write_fd, b"x")
            process.wait(timeout=5)

    def test_complete_manager_synthetic_parent_and_receipt_verification(self):
        fixture = SyntheticRun(self.directory)
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        code = """
import fcntl,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); run=root/'runs'/sys.argv[2]
os.read(int(sys.argv[3]),1)
fd=os.open(root/'experiments/train-seed-claims',os.O_RDONLY|os.O_DIRECTORY)
try:
    fcntl.flock(fd,fcntl.LOCK_EX)
    raise RuntimeError('new reset intent must not execute')
except KeyboardInterrupt:
    (run/'operator-failure.json').write_text(json.dumps({'error':'KeyboardInterrupt: '}))
    (run/'episode-report.json').write_text(sys.argv[4])
finally:
    os.close(fd)
"""
        report = dict(protocol_sha256=pause.PROTOCOL_SHA, rows=[fixture.row,
                      dict(fixture.protocol['schedule'][1], status='operator_error',
                           error='KeyboardInterrupt: ', partial_artifacts=[])],
                      complete=False, operator_error='KeyboardInterrupt: ',
                      reset_ledger_sha256=pause.sha(fixture.run/'reset-ledger.jsonl'),
                      review_receipt=fixture.review)
        process = subprocess.Popen([sys.executable, '-c', code, str(self.directory), pause.STUDY,
                                    str(read_fd), json.dumps(report)], pass_fds=(read_fd,))
        self.addCleanup(lambda: process.wait(timeout=5))
        identity = pause.process_identity(process.pid)
        real_flock = fcntl.flock

        def acquired(fd, operation):
            real_flock(fd, operation)
            os.write(write_fd, b'x')

        receipt_path = fixture.run/'boundary-pause.json'
        sources = {str(fixture.run/'execution-review.json'): pause.sha(fixture.run/'execution-review.json')}
        with patch.object(pause, 'source_evidence', return_value=(fixture.protocol, sources)) as source_checks, \
             patch.object(pause, 'verify_parent'), \
             patch.object(pause, 'completed_evidence', wraps=pause.completed_evidence) as completed_checks, \
             patch.object(pause, 'blocked_flock', wraps=pause.blocked_flock) as kernel_checks, \
             patch.object(pause.fcntl, 'flock', side_effect=acquired), \
             patch.object(pause.signal, 'pidfd_send_signal', wraps=signal.pidfd_send_signal) as send:
            receipt = pause.pause(process.pid, identity['start_ticks'], receipt_path,
                                  root=self.directory, settle_seconds=.2, wait_seconds=5)
            self.assertEqual(receipt['status'], 'paused')
            self.assertEqual(receipt['boundary']['completed_slots'], 1)
            self.assertEqual(receipt['boundary']['pending_reset_intents'], 0)
            self.assertFalse(receipt['boundary']['child_interrupted'])
            self.assertEqual(receipt['format'],'koi-steering-generalization-pause-v2')
            self.assertEqual(receipt['lock']['proof_mode'],pause.PROOF_MODE)
            self.assertEqual(len(receipt['lock']['snapshots']),2)
            self.assertGreater(receipt['terminal']['wall_time_s'], 0)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(send.call_args.args[1], signal.SIGINT)
            self.assertEqual(kernel_checks.call_count,2)
            self.assertEqual(completed_checks.call_count,2)
            self.assertEqual(source_checks.call_count,3)
            verified = pause.verify_boundary(receipt_path, pause.sha(receipt_path), root=self.directory)
            self.assertEqual(verified, receipt)
            original = receipt_path.read_bytes()
            for field, value in [('child_interrupted', True), ('pending_reset_intents', 1),
                                 ('no_live_children', False), ('no_new_reset', False)]:
                with self.subTest(field=field):
                    changed = json.loads(original)
                    changed['boundary'][field] = value
                    save(receipt_path, changed)
                    with self.assertRaises(ValueError):
                        pause.verify_boundary(receipt_path, pause.sha(receipt_path), root=self.directory)
            receipt_path.write_bytes(original)
            for snapshot_index in (0,1):
                for field,value in [('proof_mode','old-proof'),('manager_pid',-1),('parent_pid',-1),
                                    ('source_state_sha256','bad'),('artifact_state_sha256','bad'),
                                    ('reset_ledger_sha256','bad'),('no_live_children',False),
                                    ('completed_slots',99),('kernel_lock_lines',[])]:
                    with self.subTest(snapshot=snapshot_index,field=field):
                        changed = json.loads(original)
                        changed['lock']['snapshots'][snapshot_index][field] = value
                        save(receipt_path,changed)
                        with self.assertRaises(ValueError):
                            pause.verify_boundary(receipt_path,pause.sha(receipt_path),root=self.directory)
            receipt_path.write_bytes(original)
        self.assertEqual(process.wait(timeout=5), 0)

    def exercise_child_drain(self, post_reset):
        fixture = SyntheticRun(self.directory)
        fixture.protocol['python'] = sys.executable
        raw = fixture.output.with_suffix('.raw.jsonl')
        original_raw = raw.read_text()
        if not post_reset:
            raw.write_text('')
        child_read, child_write = os.pipe()
        parent_read, parent_write = os.pipe()
        for fd in (child_read,child_write,parent_read,parent_write):
            self.addCleanup(os.close,fd)
        scripts = self.directory/'scripts'
        scripts.mkdir()
        (scripts/'__init__.py').write_text('')
        # This temporary module is a pipe-controlled synthetic worker, not an
        # Agent or environment. Production source checks are isolated mocks.
        (scripts/'evaluate_koi_steering_generalization.py').write_text('''
def worker(arm,track,seed,output):
    import json,os,pathlib
    path=pathlib.Path(output); control=json.loads((path.parent/'child-control.json').read_text())
    print('child ready',flush=True)
    os.read(control['fd'],1)
    path.with_suffix('.raw.jsonl').write_text(control['raw'])
    (path.parent/'child-natural-exit.txt').write_text('natural, no signal')
''')
        save(fixture.run/'child-control.json',dict(fd=child_read,raw=original_raw))
        intent = fixture.intent
        command = (f"import sys;sys.path.insert(0,{str(self.directory)!r});from scripts.evaluate_koi_steering_generalization "
                   f"import worker;worker({intent['arm']!r},{intent['track']},{intent['seed']},{str(fixture.output)!r})")
        report = dict(protocol_sha256=pause.PROTOCOL_SHA,rows=[fixture.row,
                      dict(fixture.protocol['schedule'][1],status='operator_error',
                           error='KeyboardInterrupt: ',partial_artifacts=[])],
                      complete=False,operator_error='KeyboardInterrupt: ',
                      reset_ledger_sha256=pause.sha(fixture.run/'reset-ledger.jsonl'),
                      review_receipt=fixture.review)
        parent_code = '''
import fcntl,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); run=root/'runs'/sys.argv[2]
child=subprocess.Popen([sys.executable,'-I','-c',sys.argv[3]],pass_fds=(int(sys.argv[4]),),stdout=subprocess.PIPE,text=True)
assert child.stdout.readline().strip()=='child ready'
print('parent ready',flush=True)
assert child.wait()==0
print('child reaped',flush=True)
os.read(int(sys.argv[5]),1)
fd=os.open(root/'experiments/train-seed-claims',os.O_RDONLY|os.O_DIRECTORY)
try:
    fcntl.flock(fd,fcntl.LOCK_EX)
    raise RuntimeError('no new reset intent is allowed')
except KeyboardInterrupt:
    (run/'operator-failure.json').write_text(json.dumps({'error':'KeyboardInterrupt: '}))
    (run/'episode-report.json').write_text(sys.argv[6])
finally:
    os.close(fd)
'''
        process = subprocess.Popen([sys.executable,'-u','-c',parent_code,str(self.directory),pause.STUDY,
                                    command,str(child_read),str(parent_read),json.dumps(report)],
                                   pass_fds=(child_read,parent_read),stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,text=True)
        self.addCleanup(self.cleanup_process,process)
        assert process.stdout is not None
        self.assertEqual(process.stdout.readline().strip(),'parent ready')
        identity = pause.process_identity(process.pid)
        real_flock, real_wait, real_ledger = fcntl.flock,pause.wait_exit,pause.ledger_rows
        acquisitions, observations = [], []

        def acquire(fd,operation):
            real_flock(fd,operation)
            acquisitions.append(fd)
            if post_reset or len(acquisitions)==2:
                os.write(parent_write,b'x')

        def drain(fd,parent_pidfd=None,timeout=180):
            if parent_pidfd is not None:
                probe = os.open(self.directory/'experiments/train-seed-claims',os.O_RDONLY|os.O_DIRECTORY)
                try:
                    try:
                        real_flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        held = False
                    except BlockingIOError:
                        held = True
                    self.assertEqual(held,post_reset)
                    observations.append(held)
                finally:
                    os.close(probe)
                os.write(child_write,b'x')
            real_wait(fd,parent_pidfd,timeout)
            if parent_pidfd is not None:
                assert process.stdout is not None
                self.assertEqual(process.stdout.readline().strip(),'child reaped')

        first_ledger = True

        def ledger(run):
            nonlocal first_ledger
            if first_ledger:
                first_ledger = False
                return [fixture.intent]
            return real_ledger(run)

        sources = {str(fixture.run/'execution-review.json'):pause.sha(fixture.run/'execution-review.json')}
        receipt_path = fixture.run/'boundary-pause.json'
        with patch.object(pause,'source_evidence',return_value=(fixture.protocol,sources)), \
             patch.object(pause,'verify_parent'), \
             patch.object(pause.fcntl,'flock',side_effect=acquire), \
             patch.object(pause,'wait_exit',side_effect=drain), \
             patch.object(pause,'ledger_rows',side_effect=ledger), \
             patch.object(pause.signal,'pidfd_send_signal',wraps=signal.pidfd_send_signal) as send:
            receipt = pause.pause(process.pid,identity['start_ticks'],receipt_path,
                                  root=self.directory,settle_seconds=.2,wait_seconds=5)
            self.assertEqual(receipt['status'],'paused')
            self.assertEqual(send.call_count,1)
        self.assertEqual(observations,[post_reset])
        self.assertEqual(len(acquisitions),1 if post_reset else 2)
        self.assertEqual(receipt['observations'][0]['post_reset_witness'],post_reset)
        self.assertEqual(receipt['observations'][0]['lock_held_during_wait'],post_reset)
        self.assertEqual((fixture.run/'child-natural-exit.txt').read_text(),'natural, no signal')
        self.assertEqual(process.wait(timeout=5),0)

    def test_active_postreset_child_finishes_with_lock_held(self):
        self.exercise_child_drain(True)

    def test_ambiguous_prereset_child_drains_with_lock_released(self):
        self.exercise_child_drain(False)

    def test_blocked_parent_with_live_child_is_not_a_boundary(self):
        descriptor = os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY)
        self.addCleanup(os.close,descriptor)
        fcntl.flock(descriptor,fcntl.LOCK_EX)
        read_fd,write_fd = os.pipe()
        self.addCleanup(os.close,read_fd)
        self.addCleanup(os.close,write_fd)
        code = '''
import fcntl,os,subprocess,sys
child=subprocess.Popen([sys.executable,'-c','import os,sys;os.read(int(sys.argv[1]),1)',sys.argv[2]],pass_fds=(int(sys.argv[2]),))
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_DIRECTORY)
print('ready',flush=True)
fcntl.flock(fd,fcntl.LOCK_EX)
assert child.wait()==0
os.close(fd)
'''
        process = subprocess.Popen([sys.executable,'-u','-c',code,str(self.directory),str(read_fd)],
                                   pass_fds=(read_fd,),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.addCleanup(self.cleanup_process,process)
        assert process.stdout is not None
        self.assertEqual(process.stdout.readline().strip(),'ready')
        pidfd = os.pidfd_open(process.pid)
        self.addCleanup(os.close,pidfd)
        self.assertEqual(select.select([pidfd],[],[],.1)[0],[])
        try:
            with patch.object(pause.signal,'pidfd_send_signal') as send:
                with self.assertRaisesRegex(ValueError,'still has a live child'):
                    pause.blocked_flock(process.pid,descriptor,self.directory)
                send.assert_not_called()
        finally:
            os.write(write_fd,b'x')
            fcntl.flock(descriptor,fcntl.LOCK_UN)
        self.assertEqual(process.wait(timeout=5),0)


if __name__ == "__main__":
    unittest.main()
