"""Independent safety review: mock signals/flocks, never touch live run processes."""

from contextlib import ExitStack
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

from scripts import audit_koi_steering_generalization as audit
from scripts import pause_koi_steering_generalization as manager
from tests.test_pause_koi_steering_generalization import SyntheticRun, save


class IndependentPauseSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="boundary-independent-", dir="/tmp/kilo")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = SyntheticRun(self.root)

    def test_normal_run_operational_proof_is_not_allocation_metadata(self):
        # Exercise the actual frozen discovery function, not a broad skip wrapper.
        proof = self.fixture.run / "boundary-pause.json"
        save(proof, {"format": manager.FORMAT, "status": "paused", "existing_slot": "1:3184000001:crossing_projection"})
        save(self.fixture.run / "protocol.json", {"synthetic": True})
        allocated = self.fixture.run / "reset-intents.jsonl"
        allocated.write_text('{"seed":3184000001}\n')
        paths = set(audit._discover(self.root))
        self.assertNotIn(proof.relative_to(self.root).as_posix(), paths)
        self.assertIn(allocated.relative_to(self.root).as_posix(), paths)
        self.assertIn((self.fixture.run / "protocol.json").relative_to(self.root).as_posix(), paths)
        # The KOI auditor independently discovers every nonprotected reset-ledger.
        self.assertIn(self.fixture.run / "reset-ledger.jsonl", list(self.root.glob("runs/**/reset-ledger.jsonl")))
        root_descriptor = self.root / "experiments/boundary-pause.json"
        save(root_descriptor, {"seed": 3184000001})
        self.assertIn(root_descriptor.relative_to(self.root).as_posix(), audit._discover(self.root))

    def test_canonical_run_receipt_is_accepted_before_any_lock_or_signal(self):
        receipt = self.fixture.run / "boundary-pause.json"
        with patch.object(manager, "source_evidence", side_effect=ValueError("source sentinel")), \
             patch.object(manager.os, "pidfd_open", create=True), \
             patch.object(manager.fcntl, "flock") as flock, \
             patch.object(manager.signal, "pidfd_send_signal", create=True) as send:
            with self.assertRaisesRegex(ValueError, "source sentinel"):
                manager.pause(777, 321, receipt, root=self.root)
            flock.assert_not_called()
            send.assert_not_called()

    def test_unsupported_interpreter_refuses_before_sources_lock_or_signal(self):
        receipt = self.fixture.run / "boundary-pause.json"
        with patch.object(manager, "os", new=SimpleNamespace()), \
             patch.object(manager, "source_evidence") as sources, \
             patch.object(manager.fcntl, "flock") as flock, \
             patch.object(manager.signal, "pidfd_send_signal", create=True) as send:
            with self.assertRaisesRegex(ValueError, "public pidfd APIs"):
                manager.pause(777, 321, receipt, root=self.root)
            sources.assert_not_called()
            flock.assert_not_called()
            send.assert_not_called()

    def test_root_descriptor_or_alternate_run_filename_is_rejected_before_sources(self):
        for receipt in (self.root / "experiments/pause.json", self.fixture.run / "other-pause.json"):
            with self.subTest(receipt=receipt), patch.object(manager, "source_evidence") as sources, \
                 patch.object(manager.signal, "pidfd_send_signal", create=True) as send:
                with self.assertRaises(ValueError):
                    manager.pause(777, 321, receipt, root=self.root)
                sources.assert_not_called()
                send.assert_not_called()

    def test_direct_kernel_proof_requires_linked_owner_waiter_full_range_inode_and_no_children(self):
        registry = self.root / "experiments/train-seed-claims"
        info = registry.stat()
        device, inode = info.st_dev, info.st_ino
        token = f"{os.major(device):02x}:{os.minor(device):02x}:{inode}"
        owner = f"3: FLOCK ADVISORY WRITE 900 {token} 0 EOF"
        waiter = f"3: -> FLOCK ADVISORY WRITE 777 {token} 0 EOF"
        actual_read_text = Path.read_text
        for defect in (None, "owner", "waiter", "link", "range", "nonexclusive", "inode", "livechild", "platform", "unreadable"):
            locks = owner + "\n" + waiter + "\n"
            target_inode = inode
            if defect == "owner":
                locks = locks.replace("WRITE 900", "WRITE 901")
            elif defect == "waiter":
                locks = locks.replace("WRITE 777", "WRITE 778")
            elif defect == "link":
                locks = owner + "\n" + waiter.replace("3:", "4:", 1) + "\n"
            elif defect == "range":
                locks = locks.replace(" 0 EOF", " 1 EOF")
            elif defect == "nonexclusive":
                locks = locks.replace(" WRITE ", " READ ")
            elif defect == "inode":
                target_inode += 1

            def read_text(path, *args, **kwargs):
                if str(path) == "/proc/locks":
                    if defect == "unreadable":
                        raise PermissionError("public kernel locks unavailable")
                    return locks
                if str(path).startswith("/proc/777/"):
                    raise AssertionError("protected target syscall/fd/wchan proof read forbidden")
                return actual_read_text(path, *args, **kwargs)

            with self.subTest(defect=defect), \
                 patch.object(manager.platform, "system", return_value="Darwin" if defect == "platform" else "Linux"), \
                 patch.object(manager.os, "getpid", return_value=900), \
                 patch.object(manager.os, "fstat", return_value=SimpleNamespace(st_dev=device, st_ino=target_inode, st_mode=stat.S_IFDIR)), \
                 patch.object(Path, "read_text", new=read_text), \
                 patch.object(manager, "children", return_value=[778] if defect == "livechild" else []), \
                 patch.object(manager.signal, "pidfd_send_signal", create=True) as send:
                if defect:
                    with self.assertRaises(PermissionError if defect == "unreadable" else ValueError):
                        manager.blocked_flock(777, 222, registry)
                else:
                    proof = manager.blocked_flock(777, 222, registry)
                    self.assertEqual(proof["proof_mode"], manager.PROOF_MODE)
                    self.assertEqual(proof["kernel_lock_lines"], [owner, waiter])
                send.assert_not_called()

    def test_manager_refuses_all_unsafe_states_and_releases_lock_without_signal(self):
        for defect in ("multiple_children", "parent_exited", "first_proof", "second_proof", "proc_permission", "source_drift", "identity_drift", "pending_intent", "unreset_artifact"):
            with self.subTest(defect=defect):
                self.exercise_manager(defect)

    def exercise_manager(self, defect=None):
        fixture = self.fixture
        receipt_path = fixture.run / "boundary-pause.json"
        if receipt_path.exists():
            receipt_path.unlink()
        fixture.write_ledger([fixture.intent, fixture.row])
        extra_raw = fixture.run / "1-3184000001-steering_release_v1.raw.jsonl"
        if extra_raw.exists():
            extra_raw.unlink()
        if defect == "pending_intent":
            fixture.write_ledger([fixture.intent, fixture.row, {"status": "reset_intent"}])
        if defect == "unreset_artifact":
            extra_raw.write_text("")
        identity = {"pid": 777, "start_ticks": 321}
        inode = SimpleNamespace(st_dev=49, st_ino=7654, st_mode=stat.S_IFDIR)
        lock_proof = dict(proof_mode=manager.PROOF_MODE, path=str(self.root / "experiments/train-seed-claims"),
                          device=49, inode=7654, manager_pid=900, parent_pid=777,
                          kernel_lock_lines=["3: FLOCK ADVISORY WRITE 900 00:31:7654 0 EOF",
                                             "3: -> FLOCK ADVISORY WRITE 777 00:31:7654 0 EOF"])
        real_stat = os.stat

        def registry_stat(path, *args, **kwargs):
            if str(path) == str(self.root / "experiments/train-seed-claims"):
                return inode
            return real_stat(path, *args, **kwargs)

        with ExitStack() as stack:
            def mocked(target, attribute, **kwargs):
                return stack.enter_context(patch.object(target, attribute, **kwargs))

            sources = mocked(manager, "source_evidence", return_value=(fixture.protocol, {"source": "exact"}))
            if defect == "source_drift":
                sources.side_effect = [(fixture.protocol, {"source": "exact"}), (fixture.protocol, {"source": "changed"})]
            mocked(manager, "verify_parent")
            identity_mock = mocked(manager, "process_identity", return_value=identity)
            if defect == "identity_drift":
                identity_mock.side_effect = [identity, identity, {"pid": 777, "start_ticks": 322}]
            mocked(manager.os, "pidfd_open", return_value=111, create=True)
            mocked(manager.os, "open", return_value=222)
            close = mocked(manager.os, "close")
            mocked(manager.os, "stat", side_effect=registry_stat)
            mocked(manager.os, "fstat", return_value=inode)
            mocked(manager.os, "fsync")
            flock = mocked(manager.fcntl, "flock")
            mocked(manager, "children", return_value=[778, 779] if defect == "multiple_children" else [])
            mocked(manager.select, "select", return_value=([111], [], []) if defect == "parent_exited" else ([], [], []))
            proof = mocked(manager, "blocked_flock", return_value=deepcopy(lock_proof))
            if defect == "first_proof":
                proof.side_effect = ValueError("unsupported kernel proof")
            elif defect == "second_proof":
                proof.side_effect = [deepcopy(lock_proof), ValueError("proof changed before signal")]
            elif defect == "proc_permission":
                proof.side_effect = PermissionError("public kernel lock visibility denied")
            send = mocked(manager.signal, "pidfd_send_signal", create=True)
            mocked(manager, "wait_exit")
            mocked(manager, "terminal_evidence", return_value={"process_exited": True})
            if defect:
                with self.assertRaises(PermissionError if defect == "proc_permission" else ValueError):
                    manager.pause(777, 321, receipt_path, root=self.root, settle_seconds=.01, wait_seconds=1)
                send.assert_not_called()
                receipt = manager.read_json(receipt_path)
                self.assertEqual(receipt["status"], "refused")
                self.assertIsNone(receipt["signal"])
            else:
                receipt = manager.pause(777, 321, receipt_path, root=self.root, settle_seconds=.01, wait_seconds=1)
                self.assertEqual(receipt["status"], "paused")
                send.assert_called_once_with(111, signal.SIGINT)
                self.assertEqual(proof.call_count, 2)
                self.assertEqual(receipt["boundary"]["pending_reset_intents"], 0)
                self.assertFalse(receipt["boundary"]["child_interrupted"])
            flock.assert_called_once_with(222, fcntl.LOCK_EX)
            self.assertIn(call(222), close.call_args_list)
            self.assertIn(call(111), close.call_args_list)

    def test_only_parent_pidfd_receives_one_mock_signal_after_two_kernel_proofs(self):
        self.exercise_manager()


if __name__ == "__main__":
    unittest.main()
