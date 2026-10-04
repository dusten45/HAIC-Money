import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import package_koi_adaptive_avoidance as package


class TestKoiAdaptivePackage(unittest.TestCase):
    def test_baseline_sources_are_verified_and_use_explicit_crossing_mode(self):
        files = package.frozen_baseline_files()
        self.assertEqual(len(files), 10)
        self.assertIn(b"ContactContinuityAgent('crossing_projection')", files["agent.py"])
        self.assertEqual(
            package.digest(files["haic_agent/contact_continuity_runtime.py"]),
            package.CONTACT_HASH,
        )

    def test_tampered_contact_source_is_rejected(self):
        original_run = subprocess.run

        def changed(command, **kwargs):
            if command[-1].endswith("/haic_agent/contact_continuity_runtime.py"):
                return subprocess.CompletedProcess(command, 0, stdout=b"tampered")
            return original_run(command, **kwargs)

        with patch.object(package.subprocess, "run", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "contact module hash"):
                package.frozen_baseline_files()

    def test_package_smoke_layout_hashes_and_baseline_preservation(self):
        baseline = package.frozen_baseline_files()
        with tempfile.TemporaryDirectory(dir="/tmp/kilo") as temporary:
            output = Path(temporary) / "candidate.zip"
            manifest = package.build_package(output)
            self.assertEqual(len(manifest["files"]), 11)
            self.assertFalse(manifest["driving_evaluation_performed"])
            self.assertFalse(manifest["official_submission"])
            self.assertEqual(manifest["reference_crossing_zip_sha256"], package.CROSSING_HASH)
            self.assertIn("not original crossing ZIP", manifest["baseline_restoration"])
            smoke = manifest["smoke"]
            self.assertTrue(smoke["prefix_equal"])
            self.assertTrue(smoke["no_obstacle_equal"])
            self.assertTrue(smoke["far_only_equal"])
            self.assertTrue(smoke["moving_crossing_equal"])
            self.assertEqual(smoke["environment_resets"], 0)
            self.assertGreater(smoke["obstacle"]["pass_speed"], 44.0)
            self.assertLess(
                smoke["obstacle"]["candidate_action"][2],
                smoke["obstacle"]["baseline_action"][2],
            )
            self.assertEqual(manifest["candidate_zip_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text()), manifest)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {row["path"] for row in manifest["files"]})
                for row in manifest["files"]:
                    content = archive.read(row["path"])
                    self.assertEqual(len(content), row["bytes"])
                    self.assertEqual(package.digest(content), row["sha256"])
                for name, content in baseline.items():
                    if name != "agent.py":
                        self.assertEqual(archive.read(name), content)
                self.assertIn(b"AdaptiveAvoidanceAgent", archive.read("agent.py"))
                self.assertFalse(any(name.endswith(".pt") for name in archive.namelist()))

    def test_archive_is_deterministic(self):
        with tempfile.TemporaryDirectory(dir="/tmp/kilo") as temporary:
            first = package.build_package(Path(temporary) / "first.zip")
            second = package.build_package(Path(temporary) / "second.zip")
            self.assertEqual(first["candidate_zip_sha256"], second["candidate_zip_sha256"])

    def test_existing_output_or_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir="/tmp/kilo") as temporary:
            output = Path(temporary) / "candidate.zip"
            output.write_bytes(b"existing candidate")
            with self.assertRaises(FileExistsError):
                package.build_package(output)
            self.assertEqual(output.read_bytes(), b"existing candidate")
            other = Path(temporary) / "other.zip"
            receipt = other.with_suffix(".manifest.json")
            receipt.write_text("existing receipt")
            with self.assertRaises(FileExistsError):
                package.build_package(other)
            self.assertFalse(other.exists())
            self.assertEqual(receipt.read_text(), "existing receipt")

    def test_invalid_output_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory(dir="/tmp/kilo") as temporary:
            for output in (Path(temporary) / "candidate.txt", Path(temporary) / "missing/candidate.zip"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    package.build_package(output)
                self.assertFalse(output.exists())

    def test_receipt_failure_rolls_back_new_zip(self):
        original_open = Path.open
        with tempfile.TemporaryDirectory(dir="/tmp/kilo") as temporary:
            output = Path(temporary) / "candidate.zip"
            receipt = output.with_suffix(".manifest.json")

            def failed_open(path, mode="r", *args, **kwargs):
                if path == receipt and mode == "xb":
                    raise PermissionError("synthetic receipt write failure")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", new=failed_open):
                with self.assertRaisesRegex(PermissionError, "receipt write failure"):
                    package.build_package(output)
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())

    def test_concurrent_receipt_is_preserved_while_our_zip_is_rolled_back(self):
        original_open = Path.open
        with tempfile.TemporaryDirectory(dir="/tmp/kilo") as temporary:
            output = Path(temporary) / "candidate.zip"
            receipt = output.with_suffix(".manifest.json")

            def raced_open(path, mode="r", *args, **kwargs):
                if path == receipt and mode == "xb":
                    receipt.write_text("external receipt")
                    raise FileExistsError("synthetic concurrent receipt")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", new=raced_open):
                with self.assertRaisesRegex(FileExistsError, "concurrent receipt"):
                    package.build_package(output)
            self.assertFalse(output.exists())
            self.assertEqual(receipt.read_text(), "external receipt")


if __name__ == "__main__":
    unittest.main()
