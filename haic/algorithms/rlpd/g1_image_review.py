"""Pure G1 TRAIN image-review preparation; never an actor or policy input.

Only serialize packets with ``serialize_review_packets``. Restricted entries are
for an independently authenticated collector, not for the image reviewer. A
local timestamp and SHA receipt record process order; neither proves that a
human was blind to outcomes or that the collector/trace was authentic.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Sequence

import numpy as np

from haic.algorithms.rlpd.g1_coverage import ACTOR_HASHES, CoverageRules


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_STACK_BYTES = 4 * 84 * 84
_PIXEL_DIFFERENCES = 3 * 84 * 84
_REASONS = {
    "yes": frozenset({"visible_stall"}),
    "no": frozenset({"no_visible_stall"}),
    "unknown": frozenset({"ambiguous_pixels", "incomplete_images", "not_reviewed"}),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


@dataclass(frozen=True)
class ReviewPlan:
    """Freeze the pixel-only candidate anchor and windows before collection.

    ``pixel_difference_ceiling`` is an integer sum over three adjacent frame
    pairs; it chooses an image to review, NOT a pixel-stall classification.
    ``salt`` must remain outside the reviewer packet. There is no default flag
    threshold or outcome-dependent window adjustment.
    """

    rules: CoverageRules
    protocol_sha256: str
    salt: bytes
    ordinary_anchor: int
    first_flag_min_decision: int
    pixel_difference_ceiling: int
    window_before: int
    window_after: int

    def __post_init__(self) -> None:
        if (not isinstance(self.rules, CoverageRules)
                or not _valid_sha(self.protocol_sha256)
                or type(self.salt) is not bytes or len(self.salt) < 16
                or any(type(value) is not int for value in (
                    self.ordinary_anchor, self.first_flag_min_decision,
                    self.pixel_difference_ceiling, self.window_before,
                    self.window_after))
                or not 0 <= self.ordinary_anchor < 2000
                or not 0 <= self.first_flag_min_decision < 2000
                or not 0 <= self.pixel_difference_ceiling <= 255 * _PIXEL_DIFFERENCES
                or not 0 <= self.window_before <= 32
                or not 0 <= self.window_after <= 32):
            raise ValueError("G1 pixel anchors, windows, protocol SHA and salt must be fixed")

    def packet_id(self, slot_index: int) -> str:
        if type(slot_index) is not int or not 0 <= slot_index < 48:
            raise ValueError("G1 packet slot index must be 0..47")
        return _sha(b"haic-g1-image-packet-v1\0" + bytes.fromhex(self.protocol_sha256)
                    + self.salt + slot_index.to_bytes(2, "big"))


@dataclass(frozen=True)
class ReviewPacket:
    packet_id: str
    image_rubric_sha256: str
    ordinary: tuple[bytes | None, ...]
    first_flag: tuple[bytes | None, ...]


@dataclass(frozen=True)
class RestrictedEntry:
    slot_index: int
    packet_id: str
    track_id: int
    geometry_seed: int
    actor_id: str
    actor_sha256: str
    review_packet_sha256: str
    original_pixel_sha256: str | None
    trace_sha256: str | None
    first_flag_decision: int | None


@dataclass(frozen=True)
class ReviewLabel:
    packet_id: str
    flag: str
    reason: str


@dataclass(frozen=True)
class ReviewSeal:
    plan_sha256: str
    packets_sha256: str
    labels_sha256: str
    restricted_sha256: str
    sealed_at_utc: str
    receipt_sha256: str


def _stack_at(initial: np.ndarray, frames: np.ndarray, decision: int) -> np.ndarray:
    if decision == 0:
        return initial
    if decision < 4:
        return np.concatenate((initial[decision:], frames[:decision]), axis=0)
    return frames[decision - 4:decision]


def _window(initial: np.ndarray, frames: np.ndarray, anchor: int | None,
            plan: ReviewPlan) -> tuple[bytes | None, ...]:
    return tuple(
        _stack_at(initial, frames, decision).tobytes(order="C")
        if anchor is not None and 0 <= (decision := anchor + offset) < len(frames)
        else None
        for offset in range(-plan.window_before, plan.window_after + 1)
    )


def _packet_data(packet: ReviewPacket) -> dict[str, Any]:
    return {
        "packet_id": packet.packet_id,
        "image_rubric_sha256": packet.image_rubric_sha256,
        "ordinary": [base64.b64encode(frame).decode("ascii") if frame is not None else None
                     for frame in packet.ordinary],
        "first_flag": [base64.b64encode(frame).decode("ascii") if frame is not None else None
                       for frame in packet.first_flag],
    }


def _plan_sha(plan: ReviewPlan) -> str:
    return _sha(_json_bytes({
        "protocol_sha256": plan.protocol_sha256,
        "image_rubric_sha256": plan.rules.image_rubric_sha256,
        "salt_sha256": _sha(plan.salt),
        "ordinary_anchor": plan.ordinary_anchor,
        "first_flag_min_decision": plan.first_flag_min_decision,
        "pixel_difference_ceiling": plan.pixel_difference_ceiling,
        "window_before": plan.window_before,
        "window_after": plan.window_after,
    }))


def build_review_packets(
    plan: ReviewPlan,
    images: Sequence[tuple[np.ndarray, np.ndarray] | None],
) -> tuple[tuple[ReviewPacket, ...], tuple[RestrictedEntry, ...]]:
    """Build all 48 packets from original uint8 pixels (None means unrun).

    The only episode data accepted is initial_stack(4,84,84) and next_frame
    (N,84,84). Decision t's stack contains initial frames and next_frame[:t]:
    it never contains the result of the current action. None reserves a missing
    slot with unknown imagery; no shorter or replacement schedule is accepted.
    """
    if not isinstance(plan, ReviewPlan) or len(images) != 48:
        raise ValueError("G1 image review requires the exact 48-slot schedule")
    packets = []
    restricted = []
    for index, image_pair in enumerate(images):
        packet_id = plan.packet_id(index)
        track_id, seed = plan.rules.cells[index // 2]
        actor_id = (plan.rules.primary_actor_id, plan.rules.comparator_actor_id)[index % 2]
        original_sha = None
        flag_decision = None
        empty = (None,) * (plan.window_before + plan.window_after + 1)
        ordinary = first = empty
        if image_pair is not None:
            if not isinstance(image_pair, tuple) or len(image_pair) != 2:
                raise ValueError("observed slots require only an initial stack and next frames")
            initial, frames = image_pair
            if (not isinstance(initial, np.ndarray) or initial.dtype != np.uint8
                    or initial.shape != (4, 84, 84)
                    or not isinstance(frames, np.ndarray) or frames.dtype != np.uint8
                    or frames.ndim != 3 or frames.shape[1:] != (84, 84)
                    or len(frames) > plan.rules.max_decisions_per_episode):
                raise ValueError("G1 review accepts only bounded uint8 four-frame pixels")
            initial_bytes = initial.tobytes(order="C")
            frames_bytes = frames.tobytes(order="C")
            original_sha = _sha(b"haic-g1-original-pixels-v1\0" + len(frames).to_bytes(4, "big")
                                + initial_bytes + frames_bytes)
            ordinary = _window(initial, frames, plan.ordinary_anchor, plan)
            for decision in range(plan.first_flag_min_decision, len(frames)):
                stack = _stack_at(initial, frames, decision)
                score = np.abs(np.diff(stack.astype(np.int16), axis=0)).sum(dtype=np.int64)
                if score <= plan.pixel_difference_ceiling:
                    flag_decision = decision
                    break
            first = _window(initial, frames, flag_decision, plan)
        packet = ReviewPacket(packet_id, plan.rules.image_rubric_sha256, ordinary, first)
        packets.append(packet)
        restricted.append(RestrictedEntry(
            index, packet_id, track_id, seed, actor_id, ACTOR_HASHES[actor_id],
            _sha(_json_bytes(_packet_data(packet))), original_sha, None, flag_decision,
        ))
    # Sorting by salted opaque ID removes actor/road presentation order.
    return tuple(sorted(packets, key=lambda item: item.packet_id)), tuple(restricted)


def bind_trace_hashes(
    entries: Sequence[RestrictedEntry], trace_hashes: Sequence[str | None],
) -> tuple[RestrictedEntry, ...]:
    """Bind independently verified original trace SHAs in the restricted map only."""
    if len(entries) != 48 or len(trace_hashes) != 48:
        raise ValueError("trace SHA binding must cover all 48 slots")
    bound = []
    for index, (entry, trace_sha) in enumerate(zip(entries, trace_hashes)):
        if (not isinstance(entry, RestrictedEntry) or entry.slot_index != index
                or ((entry.original_pixel_sha256 is None) != (trace_sha is None))
                or (trace_sha is not None and not _valid_sha(trace_sha))):
            raise ValueError("trace SHAs must match observed and missing slots")
        bound.append(replace(entry, trace_sha256=trace_sha))
    return tuple(bound)


def _check_packets(plan: ReviewPlan, packets: Sequence[ReviewPacket]) -> None:
    if len(packets) != 48 or any(not isinstance(packet, ReviewPacket) for packet in packets):
        raise ValueError("missing, duplicate or reordered G1 review packets")
    if tuple(packet.packet_id for packet in packets) != tuple(
        sorted(plan.packet_id(index) for index in range(48))
    ):
        raise ValueError("missing, duplicate or reordered G1 review packets")
    length = plan.window_before + plan.window_after + 1
    for packet in packets:
        if (not isinstance(packet, ReviewPacket)
                or packet.image_rubric_sha256 != plan.rules.image_rubric_sha256
                or any(type(window) is not tuple or len(window) != length
                       or any(frame is not None and (type(frame) is not bytes
                                                    or len(frame) != _STACK_BYTES)
                              for frame in window)
                       for window in (packet.ordinary, packet.first_flag))):
            raise ValueError("invalid G1 review packet or image window")


def serialize_review_packets(plan: ReviewPlan, packets: Sequence[ReviewPacket]) -> bytes:
    """The ONLY reviewer-facing serialization: no restricted mapping or outcomes."""
    _check_packets(plan, packets)
    return _json_bytes([_packet_data(packet) for packet in packets])


def _label_bytes(packets: Sequence[ReviewPacket], labels: Sequence[ReviewLabel]) -> bytes:
    if len(labels) != 48:
        raise ValueError("all 48 image labels must be indexed")
    for packet, label in zip(packets, labels):
        if (not isinstance(label, ReviewLabel) or label.packet_id != packet.packet_id
                or type(label.flag) is not str or type(label.reason) is not str
                or label.flag not in _REASONS or label.reason not in _REASONS[label.flag]):
            raise ValueError("missing, mutable, misindexed or invalid image labels")
        complete = all(frame is not None for frame in packet.ordinary + packet.first_flag)
        if not complete and (label.flag != "unknown" or label.reason != "incomplete_images"):
            raise ValueError("incomplete or absent images must remain unknown")
        if complete and label.reason == "incomplete_images":
            raise ValueError("complete images cannot claim missing-window reason")
    return _json_bytes([{"packet_id": label.packet_id, "flag": label.flag,
                         "reason": label.reason} for label in labels])


def _restricted_bytes(plan: ReviewPlan, entries: Sequence[RestrictedEntry]) -> bytes:
    if len(entries) != 48:
        raise ValueError("restricted mapping must retain every slot")
    for index, entry in enumerate(entries):
        actor_id = (plan.rules.primary_actor_id, plan.rules.comparator_actor_id)[index % 2]
        if (not isinstance(entry, RestrictedEntry) or entry.slot_index != index
                or entry.packet_id != plan.packet_id(index)
                or (entry.track_id, entry.geometry_seed) != plan.rules.cells[index // 2]
                or entry.actor_id != actor_id or entry.actor_sha256 != ACTOR_HASHES[actor_id]
                or not _valid_sha(entry.review_packet_sha256)
                or ((entry.original_pixel_sha256 is None) != (entry.trace_sha256 is None))
                or (entry.original_pixel_sha256 is not None
                    and (not _valid_sha(entry.original_pixel_sha256)
                         or not _valid_sha(entry.trace_sha256)))
                or (entry.original_pixel_sha256 is None and entry.first_flag_decision is not None)
                or (entry.first_flag_decision is not None
                    and (type(entry.first_flag_decision) is not int
                         or not 0 <= entry.first_flag_decision < 2000))):
            raise ValueError("restricted identity or original pixel/trace provenance drift")
    return _json_bytes([entry.__dict__ for entry in entries])


def _receipt_sha(plan: ReviewPlan, seal: ReviewSeal) -> str:
    return _sha(_json_bytes({
        "format": "haic-g1-image-review-seal-v1",
        "plan_sha256": seal.plan_sha256,
        "packets_sha256": seal.packets_sha256,
        "labels_sha256": seal.labels_sha256,
        "restricted_sha256": seal.restricted_sha256,
        "sealed_at_utc": seal.sealed_at_utc,
    }))


def seal_review(
    plan: ReviewPlan, packets: Sequence[ReviewPacket], labels: Sequence[ReviewLabel],
    entries: Sequence[RestrictedEntry], *, sealed_at_utc: str,
) -> ReviewSeal:
    """Seal exclusive packet/label/mapping hashes before any outcome join."""
    if (not isinstance(sealed_at_utc, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", sealed_at_utc
    )):
        raise ValueError("seal needs a UTC seconds-resolution local receipt timestamp")
    try:
        datetime.strptime(sealed_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("invalid local receipt timestamp") from exc
    _check_packets(plan, packets)
    by_id = {packet.packet_id: packet for packet in packets}
    if len(entries) != 48 or any(
        not isinstance(entry, RestrictedEntry)
        or entry.packet_id not in by_id
        or entry.review_packet_sha256 != _sha(_json_bytes(_packet_data(by_id[entry.packet_id])))
        for entry in entries
    ):
        raise ValueError("review packet windows differ from restricted slot mapping")
    result = ReviewSeal(
        _plan_sha(plan),
        _sha(serialize_review_packets(plan, packets)),
        _sha(_label_bytes(packets, labels)),
        _sha(_restricted_bytes(plan, entries)),
        sealed_at_utc, "",
    )
    return replace(result, receipt_sha256=_receipt_sha(plan, result))


def join_authenticated_outcomes(
    plan: ReviewPlan, packets: Sequence[ReviewPacket], labels: Sequence[ReviewLabel],
    entries: Sequence[RestrictedEntry], seal: ReviewSeal,
    outcomes: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Join externally authenticated rows by sealed SHA and exact restricted ID.

    Caller must independently verify collector/trace bytes and receipt chronology.
    Neither telemetry nor outcome is allowed to define/upgrade an image label.
    """
    if not isinstance(seal, ReviewSeal) or seal != seal_review(
        plan, packets, labels, entries, sealed_at_utc=seal.sealed_at_utc
    ):
        raise ValueError("packet, label or restricted SHA differs from sealed receipt")
    if len(outcomes) != 48:
        raise ValueError("authenticated outcome join requires every scheduled slot")
    labels_by_id = {label.packet_id: label for label in labels}
    packets_by_id = {packet.packet_id: packet for packet in packets}
    joined = []
    road_hashes: dict[int, str] = {}
    for index, (entry, row) in enumerate(zip(entries, outcomes)):
        if not isinstance(row, dict) or "image_flag" in row or "image_review_blinded" in row:
            raise ValueError("outcomes cannot supply an image review flag")
        expected = {
            "slot_index": index, "packet_id": entry.packet_id,
            "protocol_sha256": plan.protocol_sha256,
            "original_pixel_sha256": entry.original_pixel_sha256,
            "trace_sha256": entry.trace_sha256,
            "track_id": entry.track_id, "geometry_seed": entry.geometry_seed,
            "actor_id": entry.actor_id, "actor_sha256": entry.actor_sha256,
        }
        if (any(key not in row or type(row[key]) is not type(value) or row[key] != value
                for key, value in expected.items())
                or not {"status", "outcome", "road_centerline_sha256"} <= row.keys()):
            raise ValueError("outcome identity, actor or pixel/trace provenance mismatch")
        status, outcome = row.get("status"), row.get("outcome")
        road_sha = row.get("road_centerline_sha256")
        if entry.original_pixel_sha256 is None:
            if status != "unrun" or outcome is not None or road_sha is not None:
                raise ValueError("unrun image slot cannot claim an episode outcome")
        else:
            if (not _valid_sha(road_sha)
                    or (status == "collection_censored" and outcome != "unknown")
                    or (status == "complete" and outcome not in (
                        "finished", "off_track", "crash", "out_of_bounds",
                        "task_timeout", "unknown"))
                    or status not in ("complete", "collection_censored")):
                raise ValueError("outcome status, road or censoring is invalid")
            prior = road_hashes.setdefault(index // 2, road_sha)
            if prior != road_sha:
                raise ValueError("paired actors have differing original road hashes")
        label = labels_by_id[entry.packet_id]
        packet = packets_by_id[entry.packet_id]
        complete = all(frame is not None for frame in packet.ordinary + packet.first_flag)
        joined.append({
            "slot_index": index, "packet_id": entry.packet_id,
            "image_flag": label.flag, "image_reason": label.reason,
            "image_review_complete": complete,
            "status": status, "outcome": outcome,
        })
    return tuple(joined)
