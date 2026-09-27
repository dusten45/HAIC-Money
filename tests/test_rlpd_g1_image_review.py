"""Synthetic-only G1 packet/label/provenance checks; no environment or real traces."""

import base64
from dataclasses import replace
import hashlib
import json
import unittest

import numpy as np

from haic.algorithms.rlpd.g1_coverage import CoverageRules
from haic.algorithms.rlpd.g1_image_review import (
    ReviewLabel, ReviewPlan, bind_trace_hashes, build_review_packets,
    join_authenticated_outcomes, seal_review, serialize_review_packets,
)


class TestG1ImageReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = CoverageRules(
            cells=tuple((1, 510000 + i) for i in range(24)),
            primary_actor_id="entropy-v5-author-seed50",
            comparator_actor_id="long-horizon-seed11",
            image_rubric_sha256="a" * 64,
        )
        cls.plan = ReviewPlan(
            cls.rules, "b" * 64, b"synthetic-private-salt-v1", 1, 2, 0, 0, 1,
        )
        initial = np.broadcast_to(np.arange(4, dtype=np.uint8)[:, None, None],
                                  (4, 84, 84)).copy()
        frames = np.broadcast_to(np.array([4, 5, 5, 5, 5, 5, 5, 5], np.uint8)[:, None, None],
                                 (8, 84, 84)).copy()
        cls.images = tuple(None if i == 3 else (initial.copy(), frames[:2].copy()) if i == 5
                           else (initial.copy(), frames.copy()) for i in range(48))
        cls.packets, raw = build_review_packets(cls.plan, cls.images)
        cls.entries = bind_trace_hashes(raw, tuple(
            hashlib.sha256(f"synthetic-original-trace-{i}".encode()).hexdigest()
            if image is not None else None for i, image in enumerate(cls.images)
        ))
        cls.labels = tuple(
            ReviewLabel(packet.packet_id, "unknown", "incomplete_images")
            if any(frame is None for frame in packet.ordinary + packet.first_flag)
            else ReviewLabel(packet.packet_id, "yes", "visible_stall")
            if packet.packet_id in (cls.plan.packet_id(0), cls.plan.packet_id(1))
            else ReviewLabel(packet.packet_id, "no", "no_visible_stall")
            for packet in cls.packets
        )
        cls.seal = seal_review(cls.plan, cls.packets, cls.labels, cls.entries,
                               sealed_at_utc="2026-09-26T16:11:00Z")
        cls.rows = tuple({
            "slot_index": entry.slot_index, "packet_id": entry.packet_id,
            "protocol_sha256": cls.plan.protocol_sha256,
            "original_pixel_sha256": entry.original_pixel_sha256,
            "trace_sha256": entry.trace_sha256,
            "track_id": entry.track_id, "geometry_seed": entry.geometry_seed,
            "actor_id": entry.actor_id, "actor_sha256": entry.actor_sha256,
            "road_centerline_sha256": hashlib.sha256(
                f"synthetic-road-{entry.geometry_seed}".encode()).hexdigest()
            if entry.original_pixel_sha256 is not None else None,
            "status": "unrun" if entry.original_pixel_sha256 is None
            else "collection_censored" if entry.slot_index == 5 else "complete",
            "outcome": None if entry.original_pixel_sha256 is None
            else "unknown" if entry.slot_index == 5 else "off_track",
        } for entry in cls.entries)

    def test_exact_48_slot_sorted_opaque_ids_and_pixel_only_first_flag(self):
        self.assertEqual(len(self.packets), 48)
        self.assertEqual(len(set(packet.packet_id for packet in self.packets)), 48)
        self.assertEqual([packet.packet_id for packet in self.packets],
                         sorted(packet.packet_id for packet in self.packets))
        self.assertEqual(self.entries[0].first_flag_decision, 5)
        self.assertIsNone(self.entries[3].first_flag_decision)
        self.assertEqual(self.entries[0].actor_id, self.rules.primary_actor_id)
        self.assertEqual(self.entries[1].actor_id, self.rules.comparator_actor_id)
        self.assertEqual(self.entries[2].geometry_seed, self.entries[3].geometry_seed)
        self.assertNotEqual(self.plan.packet_id(0), replace(
            self.plan, salt=b"other-private-salt-v1").packet_id(0))

    def test_pre_action_stack_excludes_current_action_and_never_reanchors_window(self):
        packet = next(packet for packet in self.packets if packet.packet_id == self.plan.packet_id(0))
        ordinary = np.frombuffer(packet.ordinary[0], np.uint8).reshape(4, 84, 84)
        first = np.frombuffer(packet.first_flag[0], np.uint8).reshape(4, 84, 84)
        self.assertEqual(ordinary[:, 0, 0].tolist(), [1, 2, 3, 4])
        self.assertEqual(first[:, 0, 0].tolist(), [5, 5, 5, 5])
        changed_frames = self.images[0][1].copy()
        changed_frames[5] = 87  # t=5 after-action frame, not part of pre-action stack t=5
        changed = list(self.images)
        changed[0] = (self.images[0][0], changed_frames)
        packets, entries = build_review_packets(self.plan, tuple(changed))
        changed_packet = next(p for p in packets if p.packet_id == packet.packet_id)
        self.assertEqual(changed_packet.first_flag[0], packet.first_flag[0])
        self.assertNotEqual(changed_packet.first_flag[1], packet.first_flag[1])
        self.assertEqual(entries[0].first_flag_decision, 5)
        self.assertNotEqual(entries[0].original_pixel_sha256, self.entries[0].original_pixel_sha256)

    def test_review_serialization_is_bounded_and_has_no_identity_or_telemetry(self):
        rendered = serialize_review_packets(self.plan, self.packets)
        parsed = json.loads(rendered)
        self.assertEqual(len(parsed), 48)
        self.assertEqual(set(parsed[0]), {"packet_id", "image_rubric_sha256", "ordinary", "first_flag"})
        for key in (b"geometry_seed", b"track_id", b"actor", b"reward", b"speed",
                    b"contact", b"progress", b"terminal", b"outcome", b"finished",
                    b"training_source", b"trace_path", b"trace_sha256", b"slot_index",
                    self.plan.salt, b"510000", b"off_track"):
            with self.subTest(key=key):
                self.assertNotIn(key, rendered)
        self.assertNotIn(self.entries[0].original_pixel_sha256.encode(), rendered)
        self.assertNotIn(self.entries[0].trace_sha256.encode(), rendered)
        packet = next(p for p in parsed if p["packet_id"] == self.plan.packet_id(0))
        self.assertEqual(len(base64.b64decode(packet["ordinary"][0])), 4 * 84 * 84)
        self.assertNotIn(b"sealed_at_utc", rendered)

    def test_telemetry_and_outcome_changes_cannot_change_packet_or_receipt(self):
        before = serialize_review_packets(self.plan, self.packets)
        before_sha = hashlib.sha256(before).hexdigest()
        rows = [dict(row) for row in self.rows]
        rows[0].update(outcome="finished", reward=987654321,
                       telemetry={"speed_m_s": 9999, "contact": True, "progress": 1.0})
        rows[1].update(outcome="crash", reward=-12345,
                       telemetry={"speed_m_s": 0, "contact": False})
        packets, raw = build_review_packets(self.plan, self.images)
        restricted = bind_trace_hashes(raw, [entry.trace_sha256 for entry in self.entries])
        again = seal_review(self.plan, packets, self.labels, restricted,
                            sealed_at_utc=self.seal.sealed_at_utc)
        self.assertEqual(before_sha, again.packets_sha256)
        self.assertEqual(again, self.seal)
        joined = join_authenticated_outcomes(self.plan, packets, self.labels, restricted,
                                              self.seal, rows)
        self.assertEqual((joined[0]["image_flag"], joined[1]["image_flag"]), ("yes", "yes"))
        self.assertEqual((joined[0]["outcome"], joined[1]["outcome"]), ("finished", "crash"))
        self.assertNotIn("telemetry", joined[0])
        self.assertNotIn("reward", joined[0])

    def test_missing_duplicate_reordered_and_edited_packets_rejected(self):
        for altered in (
            self.packets[:-1], self.packets[::-1], self.packets[:-1] + (self.packets[0],),
            (replace(self.packets[0], ordinary=(b"bad",) + self.packets[0].ordinary[1:]),)
            + self.packets[1:],
        ):
            with self.subTest(length=len(altered)), self.assertRaises(ValueError):
                serialize_review_packets(self.plan, altered)
            with self.assertRaises(ValueError):
                seal_review(self.plan, altered, self.labels, self.entries,
                            sealed_at_utc=self.seal.sealed_at_utc)
        # Even a valid-length pixel swap is rejected against the original slot map.
        position = next(i for i, packet in enumerate(self.packets)
                        if packet.packet_id == self.plan.packet_id(0))
        packet = self.packets[position]
        swapped = list(self.packets)
        swapped[position] = replace(packet, ordinary=(b"x" * len(packet.ordinary[0]),)
                                    + packet.ordinary[1:])
        with self.assertRaisesRegex(ValueError, "restricted slot mapping"):
            seal_review(self.plan, swapped, self.labels, self.entries,
                        sealed_at_utc=self.seal.sealed_at_utc)

    def test_labels_require_all_identities_reasons_and_immutable_sealed_digest(self):
        for altered in (
            self.labels[:-1], self.labels[::-1], self.labels[:-1] + (self.labels[0],),
            (replace(self.labels[0], reason="speed=0"),) + self.labels[1:],
        ):
            with self.subTest(length=len(altered)), self.assertRaises(ValueError):
                seal_review(self.plan, self.packets, altered, self.entries,
                            sealed_at_utc=self.seal.sealed_at_utc)
        labels = list(self.labels)
        labels[0] = replace(labels[0], flag="unknown", reason="ambiguous_pixels")
        with self.assertRaisesRegex(ValueError, "sealed receipt"):
            join_authenticated_outcomes(self.plan, self.packets, labels, self.entries,
                                        self.seal, self.rows)
        with self.assertRaisesRegex(ValueError, "sealed receipt"):
            join_authenticated_outcomes(self.plan, self.packets, self.labels, self.entries,
                                        replace(self.seal, labels_sha256="c" * 64), self.rows)

    def test_unrun_and_incomplete_image_slots_remain_unknown_after_join(self):
        joined = join_authenticated_outcomes(self.plan, self.packets, self.labels,
                                              self.entries, self.seal, self.rows)
        self.assertEqual(len(joined), 48)
        for index in (3, 5):
            with self.subTest(index=index):
                self.assertEqual(joined[index]["image_flag"], "unknown")
                self.assertEqual(joined[index]["image_reason"], "incomplete_images")
                self.assertFalse(joined[index]["image_review_complete"])
        for index in (3, 5):
            changed = list(self.labels)
            position = next(i for i, label in enumerate(changed)
                            if label.packet_id == self.plan.packet_id(index))
            changed[position] = ReviewLabel(self.plan.packet_id(index), "yes", "visible_stall")
            with self.assertRaisesRegex(ValueError, "must remain unknown"):
                seal_review(self.plan, self.packets, changed, self.entries,
                            sealed_at_utc=self.seal.sealed_at_utc)
        altered = [dict(row) for row in self.rows]
        altered[5]["outcome"] = "off_track"
        with self.assertRaisesRegex(ValueError, "censoring"):
            join_authenticated_outcomes(self.plan, self.packets, self.labels,
                                        self.entries, self.seal, altered)

    def test_join_refuses_changed_pixel_trace_actor_schedule_or_road_provenance(self):
        changes = (
            (0, "original_pixel_sha256", "c" * 64),
            (0, "trace_sha256", "d" * 64),
            (0, "protocol_sha256", "e" * 64),
            (0, "actor_sha256", "f" * 64),
            (0, "actor_id", self.rules.comparator_actor_id),
            (0, "geometry_seed", 123),
            (0, "packet_id", self.plan.packet_id(1)),
            (0, "slot_index", 1),
            (1, "road_centerline_sha256", "0" * 64),
            (3, "outcome", "finished"),
        )
        for index, key, value in changes:
            with self.subTest(key=key):
                rows = [dict(row) for row in self.rows]
                rows[index][key] = value
                with self.assertRaises(ValueError):
                    join_authenticated_outcomes(self.plan, self.packets, self.labels,
                                                self.entries, self.seal, rows)
        rows = [dict(row) for row in self.rows]
        rows[0]["image_flag"] = "yes"
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            join_authenticated_outcomes(self.plan, self.packets, self.labels,
                                        self.entries, self.seal, rows)
        with self.assertRaisesRegex(ValueError, "every scheduled"):
            join_authenticated_outcomes(self.plan, self.packets, self.labels,
                                        self.entries, self.seal, self.rows[:-1])
        rows = [dict(row) for row in self.rows]
        del rows[3]["outcome"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            join_authenticated_outcomes(self.plan, self.packets, self.labels,
                                        self.entries, self.seal, rows)

    def test_restricted_mapping_mutation_after_seal_rejected(self):
        entries = list(self.entries)
        entries[0] = replace(entries[0], trace_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "sealed receipt"):
            join_authenticated_outcomes(self.plan, self.packets, self.labels,
                                        entries, self.seal, self.rows)
        entries = list(self.entries)
        entries[0] = replace(entries[0], actor_id=self.rules.comparator_actor_id)
        with self.assertRaisesRegex(ValueError, "restricted identity"):
            seal_review(self.plan, self.packets, self.labels, entries,
                        sealed_at_utc=self.seal.sealed_at_utc)

    def test_invalid_pixel_dtype_shape_extra_columns_and_short_schedule_fail(self):
        with self.assertRaisesRegex(ValueError, "48-slot"):
            build_review_packets(self.plan, self.images[:-1])
        for wrong in (np.zeros((3, 84, 84), np.uint8),
                      np.zeros((4, 84, 84), np.float32),
                      np.zeros((4, 84, 84, 1), np.uint8)):
            with self.subTest(shape=wrong.shape, dtype=wrong.dtype):
                images = list(self.images)
                images[0] = (wrong, self.images[0][1])
                with self.assertRaisesRegex(ValueError, "uint8"):
                    build_review_packets(self.plan, images)
        images = list(self.images)
        images[0] = (self.images[0][0], np.zeros((2001, 84, 84), np.uint8))
        with self.assertRaisesRegex(ValueError, "bounded uint8"):
            build_review_packets(self.plan, images)
        with self.assertRaises(TypeError):
            build_review_packets(self.plan, self.images, reward=1)

    def test_flag_rule_and_window_schedule_are_validated_before_images(self):
        for override in ({"ordinary_anchor": -1}, {"first_flag_min_decision": 2000},
                         {"pixel_difference_ceiling": -1}, {"window_after": 33},
                         {"salt": b"short"}, {"ordinary_anchor": True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                replace(self.plan, **override)
        changed_plan = replace(self.plan, window_after=2)
        changed_packets, _ = build_review_packets(changed_plan, self.images)
        self.assertNotEqual(serialize_review_packets(changed_plan, changed_packets),
                            serialize_review_packets(self.plan, self.packets))
        with self.assertRaisesRegex(ValueError, "UTC"):
            seal_review(self.plan, self.packets, self.labels, self.entries,
                        sealed_at_utc="tomorrow")
        with self.assertRaisesRegex(ValueError, "invalid local"):
            seal_review(self.plan, self.packets, self.labels, self.entries,
                        sealed_at_utc="2026-99-99T99:99:99Z")
        # A threshold change can yield identical windows, but not the same seal.
        altered = replace(self.plan, pixel_difference_ceiling=1)
        self.assertEqual(serialize_review_packets(altered, self.packets),
                         serialize_review_packets(self.plan, self.packets))
        with self.assertRaisesRegex(ValueError, "sealed receipt"):
            join_authenticated_outcomes(altered, self.packets, self.labels,
                                        self.entries, self.seal, self.rows)

    def test_trace_binding_requires_original_hash_for_each_observed_slot(self):
        raw_packets, raw_entries = build_review_packets(self.plan, self.images)
        self.assertEqual(raw_packets, self.packets)
        trace_hashes = [entry.trace_sha256 for entry in self.entries]
        for index, wrong in ((0, None), (3, "a" * 64), (0, "malformed")):
            with self.subTest(index=index, wrong=wrong):
                changed = trace_hashes.copy()
                changed[index] = wrong
                with self.assertRaisesRegex(ValueError, "trace SHAs"):
                    bind_trace_hashes(raw_entries, changed)
        with self.assertRaisesRegex(ValueError, "restricted identity"):
            seal_review(self.plan, self.packets, self.labels, raw_entries,
                        sealed_at_utc=self.seal.sealed_at_utc)


if __name__ == "__main__":
    unittest.main()
