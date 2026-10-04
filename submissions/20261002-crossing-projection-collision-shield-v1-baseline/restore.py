"""Preserve/restore the user-designated submission baseline without running Agent."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any
import zipfile
import zlib


ZIP_SHA = 'c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801'
SHIELD_SHA = 'ad772bde9a9c3f4596fdfc742e7cac33f1b605ff52a2b3f76fbd4b75d6361d96'
RESULT_SHA = '25819a469f5e44dec9ae216feb67e38ca06cbc6ab71ee9440a7772e97a048ea5'
BASELINE_ID = 'crossing-projection-collision-shield-v1-baseline-20261002'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def contained(bundle, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError(f'unsafe member path: {relative}')
    target = bundle / path
    if not target.resolve().is_relative_to(bundle.resolve()):
        raise ValueError(f'path escapes bundle: {relative}')
    return target


def verify(bundle, source_only=False):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['zip_sha256'] != ZIP_SHA or manifest['baseline_id'] != BASELINE_ID:
        raise ValueError('incorrect baseline identity')
    checksums = {}
    for line in (bundle / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        checksums[name] = expected
    required = dict(manifest['files_sha256'], **{'manifest.json': digest((bundle / 'manifest.json').read_bytes())})
    if checksums != required:
        raise ValueError('checksum inventory differs from manifest')
    for name, expected in checksums.items():
        if source_only and name == 'submission.zip':
            continue
        if digest(contained(bundle, name).read_bytes()) != expected:
            raise ValueError(f'checksum mismatch: {name}')
    if not source_only and digest((bundle / 'submission.zip').read_bytes()) != ZIP_SHA:
        raise ValueError('submission bytes differ from designated baseline')
    return manifest


def rebuild(bundle, manifest):
    # Match the original packager's ZipInfo-based serialization, without imports
    # from the project, /tmp environments, or the original worktree.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(manifest['submission_members']):
            data = contained(bundle, 'source/' + name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    data = buffer.getvalue()
    if digest(data) != ZIP_SHA:
        raise ValueError('rebuild is not byte-identical; use the preserved ZIP (Python/zlib serialization may differ)')
    return data


def freeze(bundle, root):
    if bundle.exists():
        raise FileExistsError('refusing to overwrite a baseline bundle')
    if not bundle.parent.is_dir():
        raise FileNotFoundError(bundle.parent)
    original = root / 'submissions/koi-collision-shield-v1-experimental-submission.zip'
    package = original.read_bytes()
    receipt_path = original.with_suffix('.receipt.json')
    receipt = json.loads(receipt_path.read_text())
    if digest(package) != ZIP_SHA or receipt['zip_sha256'] != ZIP_SHA:
        raise ValueError('submitted artifact differs from designated ZIP')
    result_path = root / 'experiments/koi-collision-shield-v1-result.json'
    result_bytes = result_path.read_bytes()
    if digest(result_bytes) != RESULT_SHA or json.loads(result_bytes)['verdict'] != 'NOT_ADOPTED':
        raise ValueError('original research verdict/artifact changed')
    preserved_paths = set((root / 'submissions').rglob('*.zip')) | {
        receipt_path, result_path, root / 'agent.py',
        root / 'haic/algorithms/koi/collision_shield.py',
        root / 'haic/algorithms/koi/steering_release.py',
        root / 'runs/koi-steering-generalization-v1/crossing-projection-source-reconstruction.zip',
    }
    preserved = {str(path.relative_to(root)): digest(path.read_bytes()) for path in sorted(preserved_paths)}
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        members = {item.filename: archive.read(item) for item in archive.infolist()}
        if archive.testzip() is not None or len(members) != len(archive.infolist()):
            raise ValueError('archive CRC/duplicate member failure')
    if {name: digest(data) for name, data in members.items()} != {
            item['path']: item['sha256'] for item in receipt['files']}:
        raise ValueError('package source differs from original receipt')
    shield_source = members['haic_agent/collision_shield_runtime.py']
    if digest(shield_source) != SHIELD_SHA:
        raise ValueError('shield policy or parameters changed')
    parameter_expressions = []
    source_text = shield_source.decode()
    for node in ast.parse(source_text).body:
        if isinstance(node, ast.Assign):
            names = [item.id for target in node.targets for item in ast.walk(target)
                     if isinstance(item, ast.Name) and item.id.isupper()]
            if names:
                parameter_expressions.append(dict(names=names, expression=ast.get_source_segment(source_text, node.value)))
    payload = {'submission.zip': package, 'restore.py': Path(__file__).read_bytes(),
               'provenance/package.receipt.json': receipt_path.read_bytes(),
               'provenance/research-result.json': result_bytes,
               'provenance/research-protocol.json': (root / 'runs/koi-collision-shield-v1/protocol.json').read_bytes(),
               'provenance/package_operator.py': (root / 'scripts/package_koi_collision_shield.py').read_bytes(),
               'runtime-requirements.txt': b'numpy==1.26.0\nopencv-python==4.8.1.78\n'}
    payload.update({'source/' + name: data for name, data in members.items()})
    for name, data in payload.items():
        write_new(contained(bundle, name), data)
    manifest: dict[str, Any] = dict(
        baseline_id=BASELINE_ID, status='FROZEN_SUBMISSION_BASELINE_BY_USER_DESIGNATION',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        original_package=str(original.relative_to(root)), zip_sha256=ZIP_SHA,
        policy_source_sha256=SHIELD_SHA, source_authority='exact submitted ZIP, not the current worktree',
        official_result=dict(evidence='USER_REPORTED_OFFICIAL_RESULT',
            reported_at_utc='2026-10-02T01:36:47Z', track_id=4, finished=True,
            lap_seconds=18.4, overall_rank_at_report=6, submission_id=None,
            server_receipt=None, independently_verified=False,
            package_binding='user identifies this previously packaged v1 as the successful submission',
            site_confirmation='NOT_REPORTED'),
        research_verdict='NOT_ADOPTED', research_result_sha256=RESULT_SHA,
        research_gate_note='Original +20ms efficiency failure unchanged; submission baseline designation is separate.',
        purpose='imminent obstacle-collision DNF prevention; no additional overavoidance tuning',
        runtime=dict(python='3.11', platform='Linux', device='CPU',
                     direct_packages={'numpy': '1.26.0', 'opencv-python': '4.8.1.78'},
                     extra_requirements_in_zip=False, external_weights=False,
                     parameter_authority='All policy constants and inline parameters are in exact source/ member bytes.',
                     shield_top_level_parameter_expressions=parameter_expressions),
        submission_members=sorted(members),
        files_sha256={name: digest(data) for name, data in sorted(payload.items())},
        preserved_external_artifacts_sha256=preserved,
        restore_commands=dict(verify='python restore.py verify --bundle .',
                              exact_copy='python restore.py restore --bundle . --output /existing/path/submission.zip',
                              from_sources='python restore.py rebuild --bundle . --output /existing/path/rebuilt.zip'),
        restoration_note='Prefer exact-copy restore. Rebuild verifies the original ZIP hash and fails closed on codec drift. Outputs must be new paths outside the bundle.',
        build_runtime=dict(python=sys.version.split()[0], zlib=zlib.ZLIB_RUNTIME_VERSION),
        new_agent_execution=False, new_simulator_evaluation=False, external_action=False)
    rebuilt = rebuild(bundle, manifest)
    if rebuilt != package:
        raise ValueError('source rebuild parity failed')
    if any(digest((root / name).read_bytes()) != expected for name, expected in preserved.items()):
        raise ValueError('existing protected artifact changed during freeze')
    manifest['validation'] = dict(member_hashes_match_receipt=True, original_artifacts_unchanged=True,
                                  source_rebuild_byte_identical=True, environment_resets=0)
    manifest_bytes = json.dumps(manifest, indent=2, allow_nan=False).encode() + b'\n'
    write_new(bundle / 'manifest.json', manifest_bytes)
    checksums = dict(manifest['files_sha256'], **{'manifest.json': digest(manifest_bytes)})
    write_new(bundle / 'SHA256SUMS', ''.join(f'{value}  {name}\n' for name, value in sorted(checksums.items())).encode())
    verify(bundle)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'verify', 'restore', 'rebuild'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.command == 'freeze':
        manifest = freeze(bundle, args.repo_root.resolve())
    else:
        manifest = verify(bundle, source_only=args.command == 'rebuild')
        if args.command in ('restore', 'rebuild'):
            if args.output is None or not args.output.parent.is_dir():
                raise ValueError('an output in an existing directory is required')
            if args.output.resolve().is_relative_to(bundle):
                raise ValueError('restoration must not modify the frozen bundle')
            data = rebuild(bundle, manifest) if args.command == 'rebuild' else (bundle / 'submission.zip').read_bytes()
            with args.output.open('xb') as stream:
                stream.write(data)
            if digest(args.output.read_bytes()) != ZIP_SHA:
                raise ValueError('restored on-disk ZIP mismatch')
    print(json.dumps(dict(command=args.command, baseline_id=manifest['baseline_id'],
                          bundle=str(bundle), zip_sha256=ZIP_SHA,
                          source_members=len(manifest['submission_members']), verified=True), indent=2))


if __name__ == '__main__':
    main()
