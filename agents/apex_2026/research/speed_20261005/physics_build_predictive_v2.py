"""Offline V2 assembly: replace legacy5.8 ridge cap with strict1.9 prefix.

Frozen V1 and its builder are unchanged. All body, circle, stop-tail, scenario,
observer, baseline and time-budget code is reused exactly; no world execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

from agents.apex_2026.research.speed_20261005.physics_build_predictive import build_text as build_v1

V1_SHA = 'd2bd8a0855876f5cc4d281bd1183646ad29bd27e7f9231a4af6d917e901b844e'
V1_BUILDER_SHA = 'f60e8218e636a421a39f62db7749e5f9fd05d776182edc5ba58c7fe739e37631'
PREFIX_METHOD = '''    def _predictive_supported_prefix(self, path, field):
        # Keep original nodes, including their minimum-five-node semantics.
        # Dense samples only locate the first unknown/unsupported boundary;
        # a later supported road branch can never rejoin after a grass gap.
        if self._sample_distance(field, path[:1])[0] < 1.9:
            return None
        prefix = [path[0].copy()]
        for left, right in zip(path[:-1], path[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(right-left)/.25))+1)
            dense = np.linspace(left, right, count)
            depth = self._sample_distance(field, dense)
            unsafe = np.flatnonzero(depth < 1.9)
            if len(unsafe):
                if len(prefix) < 5:
                    return None
                index = int(unsafe[0])
                if index > 0 and np.linalg.norm(dense[index-1]-prefix[-1]) > 1e-8:
                    prefix.append(dense[index-1].copy())
                return np.asarray(prefix)
            prefix.append(right.copy())
        return np.asarray(prefix) if len(prefix) >= 5 else None

'''


def build_text(root):
    root = Path(root)
    source = root/'agents/apex_2026/fast_predictive_agent.py'
    builder = root/'agents/apex_2026/research/speed_20261005/physics_build_predictive.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == V1_SHA
    assert hashlib.sha256(builder.read_bytes()).hexdigest() == V1_BUILDER_SHA
    text = build_v1(root)
    assert text == source.read_text()
    head, child = text.split('class Agent(_BaseRearClearAgent):', 1)
    old = '        path = self._ridge(frame)\n'
    assert child.count(old) == 1
    child = child.replace(old, '        path = _ClearRidgeReference._ridge(self, frame)\n'
                          '        if path is not None:\n'
                          '            path = self._predictive_supported_prefix(path, field)\n', 1)
    marker = '    def _predictive_geometry(self, frame, state):\n'
    assert child.count(marker) == 1
    child = child.replace(marker, PREFIX_METHOD+marker, 1)
    result = head+'class Agent(_BaseRearClearAgent):'+child
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    text = build_text(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
