"""Synthetic files/signatures only; no real G0 trace, simulator or actor load."""

import dataclasses
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from haic.algorithms.rlpd.prefix_parity import INITIAL_HASH, ParityError, Prefix, Signature
from haic.algorithms.rlpd.recorded_prefix import (
    _FrozenSource, _bind_frozen_source, bind_original_g0_prefix, verify_captured_prefix,
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def frozen_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha(payload)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def fake_trace():
    stack = np.stack([np.full((84, 84), i, dtype=np.uint8) for i in (1, 2, 3, 4)])
    actions = np.array([[0.125, 0.5, 0.0], [-0.25, 0.25, 0.5], [0.75, 0.0, 0.0]],
                       dtype=np.float32)
    zeros = np.zeros(3, dtype=np.float64)
    trace = {
        "initial_stack": stack,
        "observation_frame": np.stack([np.full((84, 84), i, dtype=np.uint8) for i in (4, 5, 6)]),
        "next_frame": np.stack([np.full((84, 84), i, dtype=np.uint8) for i in (5, 6, 7)]),
        "proposed_native_action": actions.copy(),
        "executed_native_action": actions.copy(),
        "commanded_official_action": actions.copy(),
        "raw_official_action": actions.astype(np.float64),
        "summed_reward": np.array([0.25, -0.1, 1.0], dtype=np.float64),
        "new_tiles": np.array([1, 0, 1], dtype=np.int64),
        "directed_delta": zeros.copy(),
        "centerline_distance_m": zeros.copy(),
        "centerline_fraction": zeros.copy(),
        "speed_m_s": zeros.copy(),
        "heading_error_rad": zeros.copy(),
        "x_m": zeros.copy(),
        "y_m": zeros.copy(),
        "progress": zeros.copy(),
        "damage": zeros.copy(),
        "contact": np.zeros(3, dtype=np.bool_),
        "off_track_counter": np.zeros(3, dtype=np.int64),
        "finish_qualified": np.zeros(3, dtype=np.bool_),
        "finished": np.zeros(3, dtype=np.bool_),
        "terminated": np.array([False, False, True]),
        "truncated": np.zeros(3, dtype=np.bool_),
        "finish_phase": np.array(["unqualified"] * 3),
        "raw_frames": np.array([4, 4, 3], dtype=np.int64),
        "raw_finish_phase_bits": np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 255]],
                                           dtype=np.uint8),
    }
    return trace


def fixture(root, *, change_trace=None, change_protocol=None, change_manifest=None,
            change_ledger=None):
    """Write a complete fake two-actor/two-road SHA chain, with one selected NPZ."""
    run = root / "runs/fake-g0"
    actor_payload = b"synthetic actor bytes (never loaded)"
    actor_sha = frozen_file(run / "actor.pt", actor_payload)
    protocol = {
        "format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen", "partition": "TRAIN",
        "frame_skip": 4, "max_steps": 2000, "reward_shaping": False,
        "collision_penalty": 0.0, "interventions": False, "learner_updates": 0,
        "geometry_audit_sha256": sha(b"fake audited roads"),
        "cells": [{"partition": "TRAIN", "obstacles": True, "track_id": 1,
                   "geometry_seed": seed} for seed in (123, 124)],
        "actors": [{"id": name, "sha256": actor_sha, "path": "runs/fake-g0/actor.pt",
                    "source_sha256": sha(b"synthetic actor source"),
                    "export_protocol_sha256": sha(b"synthetic export protocol"),
                    "action_mode": "exported_tanh_mean"} for name in ("alpha", "beta")],
    }
    if change_protocol is not None:
        change_protocol(protocol)
    protocol_path = "experiments/fake-g0.json"
    protocol_sha = frozen_file(root / protocol_path, json_bytes(protocol))
    trace = fake_trace()
    if change_trace is not None:
        change_trace(trace)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **trace)
    trace_sha = frozen_file(run / "traces/seed-123-track-1-alpha.npz", buffer.getvalue())
    rows = []
    for seed in (123, 124):
        for name in ("alpha", "beta"):
            rows.append({
                "partition": "TRAIN", "track_id": 1, "geometry_seed": seed,
                "actor_id": name, "actor_sha256": actor_sha,
                "road_centerline_sha256": sha(f"synthetic road {seed}".encode()),
                "trace_path": f"traces/seed-{seed}-track-1-{name}.npz",
                "trace_sha256": trace_sha if (seed, name) == (123, "alpha") else sha(b"unused trace"),
                "steps": 3, "driven_raw_frames": 11,
                "raw_reward_sum": math.fsum(float(x) for x in trace["summed_reward"]),
            })
    if change_ledger is not None:
        change_ledger(rows)
    ledger_path = "runs/fake-g0/cells.jsonl"
    ledger_sha = frozen_file(root / ledger_path, b"".join(json_bytes(row) for row in rows))
    manifest = {
        "format": "haic-rlpd-g0-diagnostic-result-v1",
        "role": "TRAIN-only-failure-diagnostic", "ranked": False,
        "protocol_path": protocol_path, "protocol_sha256": protocol_sha,
        "cells_sha256": ledger_sha, "geometry_audit_sha256": protocol["geometry_audit_sha256"],
        "cell_count": 4, "geometry_count": 2,
    }
    if change_manifest is not None:
        change_manifest(manifest)
    manifest_path = "runs/fake-g0/manifest.json"
    manifest_sha = frozen_file(root / manifest_path, json_bytes(manifest))
    source = _FrozenSource(protocol_path, protocol_sha, manifest_path, manifest_sha,
                           ledger_path, ledger_sha)
    return source, trace


def captured_from(recorded):
    """Only a synthetic Prefix: no env or accessible Box2D state was captured."""
    full_track_sha = sha(b"full (N,4) track, not G0 centerline XY")
    empty = sha(b"")
    state = (("road_sha256", full_track_sha),
             ("road_centerline_sha256", recorded.road_centerline_sha256),
             ("outer.max_episode_steps", 2000), ("inner.max_episode_steps", 8200))
    initial = Signature(empty, INITIAL_HASH, empty, 0, recorded.initial_observation_sha256,
                        empty, None, None, None, (), state)
    history = INITIAL_HASH
    signatures = []
    for step in recorded.steps:
        history = sha(bytes.fromhex(history) + step.official_action)
        signatures.append(Signature(
            sha(step.official_action), history, sha(step.raw_action * step.raw_frame_count),
            step.raw_frame_count, step.next_observation_sha256, empty,
            step.reward, step.terminated, step.truncated, (), state,
        ))
    return Prefix(recorded.track_id, recorded.geometry_seed, full_track_sha, initial,
                  recorded.actions, tuple(signatures))


class OriginalG0SourceTests(unittest.TestCase):
    def bind(self, root, source, **overrides):
        params = {"track_id": 1, "geometry_seed": 123, "actor_id": "alpha", "anchor_decisions": 2}
        params.update(overrides)
        return _bind_frozen_source(root, source, **params)

    def test_correct_fake_frozen_source_and_pure_captured_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, trace = fixture(root)
            recorded = self.bind(root, source)
            captured = captured_from(recorded)
            self.assertIsNone(verify_captured_prefix(recorded, captured))
            self.assertEqual(recorded.actions,
                             tuple(a.tobytes() for a in trace["commanded_official_action"][:2]))
            self.assertEqual(recorded.steps[0].raw_action,
                             trace["commanded_official_action"][0].astype(np.float64).tobytes())
            self.assertEqual(recorded.steps[1].reward, float(trace["summed_reward"][1]))
            self.assertEqual(recorded.trace_sha256,
                             sha((root / "runs/fake-g0/traces/seed-123-track-1-alpha.npz").read_bytes()))
            self.assertEqual(recorded.protocol_sha256, source.protocol_sha256)
            self.assertEqual(recorded.manifest_sha256, source.manifest_sha256)
            self.assertEqual(recorded.ledger_sha256, source.ledger_sha256)
            self.assertEqual(recorded.actor_sha256, sha(b"synthetic actor bytes (never loaded)"))
            self.assertNotEqual(recorded.road_centerline_sha256, captured.road_sha256)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                recorded.actor_id = "beta"

    def test_public_entry_point_rejects_self_pinned_fake_source_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            with self.assertRaisesRegex(ParityError, "not the original frozen source"):
                bind_original_g0_prefix(
                    root, expected_protocol_sha256=source.protocol_sha256,
                    expected_manifest_sha256=source.manifest_sha256,
                    expected_ledger_sha256=source.ledger_sha256, track_id=1,
                    geometry_seed=123, actor_id="alpha", anchor_decisions=2,
                )
            # Even the real declared hashes cannot validate an unrelated directory.
            with self.assertRaisesRegex(ParityError, "missing frozen source"):
                bind_original_g0_prefix(
                    root,
                    expected_protocol_sha256="6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b",
                    expected_manifest_sha256="52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358",
                    expected_ledger_sha256="924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559",
                    track_id=1, geometry_seed=123, actor_id="alpha", anchor_decisions=2,
                )

    def test_tampered_protocol_manifest_ledger_trace_or_actor_bytes_rejected(self):
        paths = ("experiments/fake-g0.json", "runs/fake-g0/manifest.json",
                 "runs/fake-g0/cells.jsonl", "runs/fake-g0/traces/seed-123-track-1-alpha.npz",
                 "runs/fake-g0/actor.pt")
        for relative in paths:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _ = fixture(root)
                path = root / relative
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(ParityError, "frozen SHA mismatch"):
                    self.bind(root, source)

    def test_self_consistent_fabricated_captured_action_prefix_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            recorded = self.bind(root, source)
            captured = captured_from(recorded)
            fabricated = np.array([0.5, 0.25, 0.0], dtype=np.float32).tobytes()
            history = sha(bytes.fromhex(INITIAL_HASH) + fabricated)
            first = dataclasses.replace(captured.steps[0], action_sha256=sha(fabricated),
                                        prefix_sha256=history,
                                        raw_action_sha256=sha(np.frombuffer(fabricated, dtype=np.float32)
                                                              .astype(np.float64).tobytes() * 4))
            second = dataclasses.replace(captured.steps[1],
                                         prefix_sha256=sha(bytes.fromhex(history) + captured.actions[1]))
            invented = dataclasses.replace(captured, actions=(fabricated, captured.actions[1]),
                                           steps=(first, second))
            with self.assertRaisesRegex(ParityError, "official action bytes"):
                verify_captured_prefix(recorded, invented)

    def test_float64_raw_command_must_be_exact_widening_even_if_numerically_close(self):
        def change(trace):
            trace["raw_official_action"][0, 0] = np.nextafter(
                trace["raw_official_action"][0, 0], np.inf)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_trace=change)
            with self.assertRaisesRegex(ParityError, "exact widened official action"):
                self.bind(root, source)

    def test_float64_signed_zero_raw_command_must_preserve_official_bits(self):
        def change(trace):
            trace["raw_official_action"][0, 2] = -0.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_trace=change)
            with self.assertRaisesRegex(ParityError, "exact widened official action"):
                self.bind(root, source)

    def test_wrong_official_dtype_or_missing_pixel_column_fails_after_trace_pinning(self):
        for change in (
            lambda trace: trace.update(commanded_official_action=trace["commanded_official_action"].astype(np.float64)),
            lambda trace: trace.pop("next_frame"),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _ = fixture(root, change_trace=change)
                with self.assertRaises(ParityError):
                    self.bind(root, source)

    def test_off_by_one_pixel_step_fails_even_with_recomputed_trace_and_ledger_hashes(self):
        def change(trace):
            trace["observation_frame"][1] = trace["next_frame"][1]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_trace=change)
            with self.assertRaisesRegex(ParityError, "off by one at step 1"):
                self.bind(root, source)

    def test_wrong_actor_or_cell_and_mismatched_paired_road_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            for params in ({"actor_id": "other"}, {"geometry_seed": 999}, {"track_id": 2}):
                with self.subTest(params=params), self.assertRaisesRegex(ParityError, "not a frozen G0 ledger row"):
                    self.bind(root, source, **params)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_ledger=lambda rows: rows[1].update(
                road_centerline_sha256=sha(b"a different road")))
            with self.assertRaisesRegex(ParityError, "different road centerline"):
                self.bind(root, source)

    def test_captured_road_must_match_original_xy_centerline_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            recorded = self.bind(root, source)
            captured = captured_from(recorded)
            wrong_state = (("road_sha256", captured.road_sha256),
                           ("road_centerline_sha256", sha(b"other centerline")))
            fabricated = dataclasses.replace(
                captured,
                initial=dataclasses.replace(captured.initial, state=wrong_state),
                steps=tuple(dataclasses.replace(step, state=wrong_state) for step in captured.steps),
            )
            with self.assertRaisesRegex(ParityError, "centerline SHA differs"):
                verify_captured_prefix(recorded, fabricated)

    def test_shorten_both_consistent_clocks_still_rejects_original_g0_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            recorded = self.bind(root, source)
            captured = captured_from(recorded)
            shortened_state = tuple(
                (name, 3 if name == "outer.max_episode_steps" else
                 212 if name == "inner.max_episode_steps" else value)
                for name, value in captured.initial.state
            )
            shortened = dataclasses.replace(
                captured,
                initial=dataclasses.replace(captured.initial, state=shortened_state),
                steps=tuple(dataclasses.replace(step, state=shortened_state)
                            for step in captured.steps),
            )
            with self.assertRaisesRegex(ParityError, "deadline differs"):
                verify_captured_prefix(recorded, shortened)

    def test_wrong_protocol_actor_identity_rejected_despite_rehashed_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_ledger=lambda rows: rows[0].update(
                actor_sha256=sha(b"fabricated source actor")))
            with self.assertRaisesRegex(ParityError, "ledger row differs"):
                self.bind(root, source)

    def test_manifest_must_link_externally_pinned_protocol_and_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_manifest=lambda manifest: manifest.update(
                cells_sha256=sha(b"alternate ledger")))
            with self.assertRaisesRegex(ParityError, "manifest does not bind"):
                self.bind(root, source)

    def test_terminal_anchor_or_earlier_terminal_transition_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            with self.assertRaisesRegex(ParityError, "terminal decision"):
                self.bind(root, source, anchor_decisions=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_trace=lambda trace: trace["terminated"].__setitem__(1, True))
            with self.assertRaisesRegex(ParityError, "continues after terminal"):
                self.bind(root, source)

    def test_captured_observation_reward_and_flags_must_match_recorded_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            recorded = self.bind(root, source)
            captured = captured_from(recorded)
            for changed in (
                dataclasses.replace(captured, initial=dataclasses.replace(
                    captured.initial, observation_sha256=sha(b"wrong initial pixels"))),
                dataclasses.replace(captured, steps=(dataclasses.replace(
                    captured.steps[0], observation_sha256=sha(b"off by one pixels")),
                    captured.steps[1])),
                dataclasses.replace(captured, steps=(dataclasses.replace(
                    captured.steps[0], reward=0.26), captured.steps[1])),
                dataclasses.replace(captured, steps=(dataclasses.replace(
                    captured.steps[0], terminated=True), captured.steps[1])),
                dataclasses.replace(captured, steps=(dataclasses.replace(
                    captured.steps[0], raw_frame_count=3), captured.steps[1])),
            ):
                with self.subTest(changed=changed), self.assertRaises(ParityError):
                    verify_captured_prefix(recorded, changed)

    def test_npz_object_arrays_cannot_be_loaded(self):
        def change(trace):
            trace["finished"] = np.array([False, False, True], dtype=object)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root, change_trace=change)
            with self.assertRaisesRegex(ValueError, "Object arrays cannot be loaded"):
                self.bind(root, source)


if __name__ == "__main__":
    unittest.main()
