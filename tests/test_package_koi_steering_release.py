from pathlib import Path
import tempfile
import zipfile

import pytest

from scripts import package_koi_steering_release as package


def test_real_frozen_source_smoke_and_twelve_members():
    with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
        output = Path(directory) / 'candidate.zip'
        manifest = package.build_package(output)
        assert manifest['smoke']['safe_release_active']
        assert manifest['smoke']['environment_resets'] == 0
        assert not manifest['speed_target_changed'] and not manifest['margin_reduction']
        with zipfile.ZipFile(output) as archive:
            assert len(archive.namelist()) == 12
            for receipt in manifest['files']:
                name = receipt['path']
                data = archive.read(name)
                assert package.digest(data) == receipt['sha256']
                assert len(data) == receipt['bytes']
        assert package.digest(output.read_bytes()) == manifest['candidate_zip_sha256']
        baseline = package.frozen_baseline_files(package.ROOT)
        assert manifest['baseline_source_sha256'] == {name: package.digest(data) for name, data in baseline.items()}
        assert len(manifest['files']) == len({receipt['path'] for receipt in manifest['files']})
        with zipfile.ZipFile(output) as archive:
            for name, data in baseline.items():
                if name != 'agent.py':
                    assert archive.read(name) == data


def test_frozen_outputs_not_overwritten_and_variant_receipt_explicit():
    with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
        output = Path(directory) / 'candidate.zip'
        manifest = package.build_package(output, candidate_name='koi-steering-release-v2', stabilize_ambiguous_flank=True)
        assert manifest['candidate'] == 'koi-steering-release-v2'
        assert manifest['stabilize_ambiguous_flank']
        with zipfile.ZipFile(output) as archive:
            assert b'stabilize_ambiguous_flank=True' in archive.read('agent.py')
        before = output.read_bytes()
        with pytest.raises(FileExistsError):
            package.build_package(output)
        assert output.read_bytes() == before


def test_deterministic_package_bytes_and_study_identity_guard():
    with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
        a = package.build_package(Path(directory) / 'a.zip')
        b = package.build_package(Path(directory) / 'b.zip')
        assert a['candidate_zip_sha256'] == b['candidate_zip_sha256']
        with pytest.raises(ValueError):
            package.build_package(Path(directory) / 'bad.zip', candidate_name='minimum-clearance-v4')
