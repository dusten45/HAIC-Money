import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import package_koi_minimum_clearance as package


class PackageTests(unittest.TestCase):
    def test_real_source_smoke_member_hashes_and_immutable_baseline(self):
        baseline = package.frozen_baseline_files()
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            output = Path(directory) / 'candidate.zip'
            manifest = package.build_package(output)
            self.assertEqual(manifest['candidate_zip_sha256'], package.digest(output.read_bytes()))
            self.assertEqual(json.loads(output.with_suffix('.manifest.json').read_text()), manifest)
            self.assertEqual(len(manifest['files']), 11)
            self.assertTrue(manifest['steering_only'])
            self.assertTrue(manifest['baseline_speed_target_and_pedal_computation_unchanged'])
            self.assertFalse(manifest['driving_evaluation_performed'])
            self.assertFalse(manifest['official_submission'])
            self.assertEqual(manifest['smoke']['environment_resets'], 0)
            for key in ('prefix_equal', 'no_obstacle_equal', 'far_only_equal', 'same_observation_pedals_equal',
                        'minimum_displacement_active', 'projection_off_return_active'):
                self.assertTrue(manifest['smoke'][key])
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {row['path'] for row in manifest['files']})
                for row in manifest['files']:
                    data = archive.read(row['path'])
                    self.assertEqual(len(data), row['bytes'])
                    self.assertEqual(package.digest(data), row['sha256'])
                for name, data in baseline.items():
                    if name != 'agent.py':
                        self.assertEqual(archive.read(name), data)
                self.assertNotIn('haic_agent/adaptive_avoidance_runtime.py', archive.namelist())
                self.assertIn(b'MinimumClearanceAgent', archive.read('agent.py'))

    def test_deterministic_rebuild(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            one = package.build_package(Path(directory) / 'one.zip')
            two = package.build_package(Path(directory) / 'two.zip')
            self.assertEqual(one['candidate_zip_sha256'], two['candidate_zip_sha256'])

    def test_separate_hold_variant_packages_explicit_configuration_and_real_smoke(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            output = Path(directory) / 'hold.zip'
            manifest = package.build_package(output, command_hold_seconds=.08)
            self.assertEqual(manifest['candidate'], 'koi-minimum-clearance-v2')
            self.assertEqual(manifest['command_hold_seconds'], .08)
            self.assertTrue(manifest['smoke']['actual_hold_command_verified'])
            self.assertFalse(manifest['smoke']['full_command_fallback_verified'])
            self.assertTrue(manifest['smoke']['same_observation_pedals_equal'])
            with zipfile.ZipFile(output) as archive:
                self.assertIn(b'command_hold_seconds=0.08', archive.read('agent.py'))

    def test_projection_variant_has_distinct_identity_and_real_guard_smoke(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            output = Path(directory) / 'projection.zip'
            manifest = package.build_package(output, command_hold_seconds=.08, require_projection_clearance=True)
            self.assertEqual(manifest['candidate'], 'koi-minimum-clearance-v3')
            self.assertTrue(manifest['require_projection_clearance'])
            self.assertTrue(manifest['smoke']['projection_footprint_guard_verified'])
            with zipfile.ZipFile(output) as archive:
                self.assertIn(b'require_projection_clearance=True', archive.read('agent.py'))

    def test_refuses_existing_zip_or_manifest_and_invalid_output(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            output = Path(directory) / 'candidate.zip'
            output.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                package.build_package(output)
            self.assertEqual(output.read_bytes(), b'existing')
            output = Path(directory) / 'other.zip'
            output.with_suffix('.manifest.json').write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                package.build_package(output)
            self.assertFalse(output.exists())
            for output in (Path(directory) / 'bad.txt', Path(directory) / 'missing/bad.zip'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    package.build_package(output)

    def test_receipt_publication_failure_rolls_back_only_owned_zip(self):
        original_open = Path.open
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            output = Path(directory) / 'candidate.zip'
            receipt = output.with_suffix('.manifest.json')
            def failed(path, mode='r', *args, **kwargs):
                if path == receipt and mode == 'xb':
                    raise PermissionError('synthetic receipt error')
                return original_open(path, mode, *args, **kwargs)
            with patch.object(Path, 'open', failed), self.assertRaises(PermissionError):
                package.build_package(output)
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())

    def test_concurrently_created_manifest_is_preserved(self):
        original_open = Path.open
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            output = Path(directory) / 'candidate.zip'
            receipt = output.with_suffix('.manifest.json')
            def raced(path, mode='r', *args, **kwargs):
                if path == receipt and mode == 'xb':
                    receipt.write_bytes(b'concurrent')
                    raise FileExistsError('synthetic race')
                return original_open(path, mode, *args, **kwargs)
            with patch.object(Path, 'open', raced), self.assertRaises(FileExistsError):
                package.build_package(output)
            self.assertFalse(output.exists())
            self.assertEqual(receipt.read_bytes(), b'concurrent')


if __name__ == '__main__':
    unittest.main()
