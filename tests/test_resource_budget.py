"""Synthetic-only resource forecasts; no collector, trainer or environment import."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict, replace

from haic.resource_budget import ResourceForecast, ResourceSnapshot, assess


GIB = 1024**3


def snapshot(**changes: object) -> ResourceSnapshot:
    base = ResourceSnapshot(
        host_available_bytes=40 * GIB, disk_available_bytes=20 * GIB,
        cgroup_leaf_limit_bytes=64 * GIB, cgroup_leaf_current_bytes=50 * GIB,
        verified_min_cgroup_raw_headroom_bytes=14 * GIB,
        cgroup_ancestry_verified=True,
        cgroup_ancestry_evidence="synthetic leaf and ancestor paired readings",
        inactive_file_bytes=0, file_dirty_bytes=0, file_writeback_bytes=0,
        temp_on_output_filesystem=True, oom_events=0, previous_oom_events=0,
    )
    return replace(base, **changes)


def forecast(**changes: object) -> ResourceForecast:
    base = ResourceForecast(
        phase="admission", measurement_reference="synthetic phase measurement", uses_gpu=False,
        incremental_ram_bytes=3 * GIB, remaining_peak_incremental_ram_bytes=3 * GIB,
        competing_ram_growth_bytes=GIB,
        ram_reserve_bytes=GIB, next_write_bytes=GIB,
        remaining_output_bytes=3 * GIB, peak_temp_bytes=GIB,
        competing_disk_growth_bytes=GIB, disk_reserve_bytes=GIB,
    )
    return replace(base, **changes)


class ResourceBudgetTests(unittest.TestCase):
    def test_old_pilot_raw_14_2_gib_with_clean_cache_meets_small_measured_demand(self) -> None:
        current = snapshot(cgroup_leaf_current_bytes=64 * GIB - int(14.2 * GIB),
                           verified_min_cgroup_raw_headroom_bytes=int(14.2 * GIB),
                           inactive_file_bytes=25 * GIB)
        result = assess(current, forecast())
        self.assertEqual(result.status, "allow")
        self.assertEqual(result.raw_cgroup_headroom_bytes, int(14.2 * GIB))
        self.assertEqual(result.required_ram_bytes, 5 * GIB)
        self.assertEqual(result.possible_ram_capacity_bytes, int(14.2 * GIB) + 25 * GIB)
        self.assertNotIn("ram_reclaim_dependent", result.reasons)

    def test_raw_14_gib_without_cache_blocks_large_16_gib_next_growth(self) -> None:
        result = assess(snapshot(), forecast(incremental_ram_bytes=16 * GIB,
                                             remaining_peak_incremental_ram_bytes=16 * GIB))
        self.assertEqual(result.raw_cgroup_headroom_bytes, 14 * GIB)
        self.assertEqual(result.possible_ram_capacity_bytes, 14 * GIB)
        self.assertIn("ram_capacity_shortfall", result.reasons)
        self.assertIn("ram_next_capacity_shortfall", result.reasons)

    def test_midrun_remaining_growth_not_full_initial_peak(self) -> None:
        current = snapshot(cgroup_leaf_current_bytes=60 * GIB,
                           verified_min_cgroup_raw_headroom_bytes=4 * GIB)
        remaining = forecast(phase="continuation", incremental_ram_bytes=GIB,
                             remaining_peak_incremental_ram_bytes=GIB,
                             remaining_output_bytes=2 * GIB)
        result = assess(current, remaining)
        self.assertEqual(result.status, "allow")
        self.assertEqual(result.raw_cgroup_headroom_bytes, 4 * GIB)
        self.assertEqual(result.required_ram_bytes, 3 * GIB)
        self.assertEqual(result.forecast.phase, "continuation")

    def test_reclaim_is_only_possible_and_host_bounded(self) -> None:
        current = snapshot(cgroup_leaf_current_bytes=62 * GIB,
                           verified_min_cgroup_raw_headroom_bytes=2 * GIB,
                           inactive_file_bytes=25 * GIB,
                           host_available_bytes=8 * GIB)
        result = assess(current, forecast())
        self.assertEqual(result.status, "block")
        self.assertEqual(result.conservative_ram_capacity_bytes, 2 * GIB)
        self.assertEqual(result.possible_ram_capacity_bytes, 8 * GIB)
        self.assertIn("ram_conservative_capacity_shortfall", result.reasons)
        self.assertIn("ram_reclaim_dependent", result.reasons)
        too_little_host = assess(replace(current, host_available_bytes=4 * GIB), forecast())
        self.assertEqual(too_little_host.status, "block")
        self.assertEqual(too_little_host.possible_ram_capacity_bytes, 4 * GIB)
        self.assertIn("ram_capacity_shortfall", too_little_host.reasons)
        # After observed reclaim, a NEW snapshot's raw gap can pass the same demand.
        remeasured = assess(replace(current, cgroup_leaf_current_bytes=58 * GIB,
                                  verified_min_cgroup_raw_headroom_bytes=6 * GIB), forecast())
        self.assertEqual(remeasured.status, "allow")
        self.assertEqual(remeasured.raw_cgroup_headroom_bytes, 6 * GIB)

    def test_dirty_writeback_cache_never_counted_as_clean(self) -> None:
        current = snapshot(cgroup_leaf_current_bytes=62 * GIB,
                           verified_min_cgroup_raw_headroom_bytes=2 * GIB,
                           inactive_file_bytes=25 * GIB,
                           file_dirty_bytes=20 * GIB, file_writeback_bytes=5 * GIB)
        result = assess(current, forecast())
        self.assertEqual(result.possible_clean_inactive_bytes, 0)
        self.assertEqual(result.possible_ram_capacity_bytes, 2 * GIB)
        self.assertIn("ram_capacity_shortfall", result.reasons)

    def test_unavailable_cache_stats_do_not_justify_reclaim(self) -> None:
        current = snapshot(cgroup_leaf_current_bytes=62 * GIB,
                           verified_min_cgroup_raw_headroom_bytes=2 * GIB,
                           inactive_file_bytes=25 * GIB,
                           file_dirty_bytes=None)
        result = assess(current, forecast())
        self.assertEqual(result.status, "block")
        self.assertIn("cache_telemetry_missing_for_reclaim", result.reasons)

    def test_next_write_fits_but_remaining_study_disk_does_not(self) -> None:
        current = snapshot(disk_available_bytes=5 * GIB)
        result = assess(current, forecast(remaining_output_bytes=8 * GIB))
        self.assertEqual(result.required_next_disk_bytes, 4 * GIB)
        self.assertEqual(result.required_remaining_disk_bytes, 11 * GIB)
        self.assertEqual(result.status, "block")
        self.assertIn("disk_remaining_shortfall", result.reasons)
        self.assertNotIn("disk_next_write_shortfall", result.reasons)

    def test_next_write_and_forecast_consistency_fail_closed(self) -> None:
        current = snapshot(disk_available_bytes=3 * GIB)
        result = assess(current, forecast(next_write_bytes=4 * GIB,
                                          remaining_output_bytes=GIB))
        self.assertIn("disk_next_write_shortfall", result.reasons)
        self.assertIn("next_write_exceeds_remaining_output_forecast", result.reasons)

    def test_separate_or_unknown_temp_filesystem_cannot_use_output_free_space(self) -> None:
        for same_filesystem in (False, None):
            with self.subTest(same_filesystem=same_filesystem):
                current = snapshot(temp_on_output_filesystem=same_filesystem)
                result = assess(current, forecast())
                self.assertEqual(result.status, "block")
                self.assertIn("temp_filesystem_not_verified_same_as_output", result.reasons)
        self.assertEqual(assess(snapshot(temp_on_output_filesystem=None),
                                forecast(peak_temp_bytes=0)).status, "allow")

    def test_admission_budgets_later_checkpoint_peak_and_inconsistent_forecast(self) -> None:
        current = snapshot(cgroup_leaf_current_bytes=60 * GIB,
                           verified_min_cgroup_raw_headroom_bytes=4 * GIB)
        later_peak = forecast(incremental_ram_bytes=GIB,
                              remaining_peak_incremental_ram_bytes=4 * GIB)
        result = assess(current, later_peak)
        self.assertEqual(result.required_next_ram_bytes, 3 * GIB)
        self.assertEqual(result.required_ram_bytes, 6 * GIB)
        self.assertEqual(result.status, "block")
        self.assertIn("ram_capacity_shortfall", result.reasons)
        self.assertNotIn("ram_next_capacity_shortfall", result.reasons)
        self.assertIn("remaining_ram_peak_below_next_demand", assess(
            snapshot(), forecast(remaining_peak_incremental_ram_bytes=2 * GIB)).reasons)

    def test_shared_gpu_passes_when_reserved_and_peer_growth_fit(self) -> None:
        future = forecast(uses_gpu=True, gpu_incremental_reserved_bytes=2 * GIB,
                          gpu_measurement_reference="synthetic selected-device reserved/context peak",
                          competing_gpu_growth_bytes=GIB, gpu_reserve_bytes=GIB)
        result = assess(snapshot(gpu_free_bytes=5 * GIB), future)
        self.assertEqual(result.status, "allow")
        self.assertEqual(result.required_gpu_bytes, 4 * GIB)
        self.assertIn("gpu_capacity_shortfall",
                      assess(snapshot(gpu_free_bytes=3 * GIB), future).reasons)
        self.assertIn("gpu_telemetry_missing", assess(snapshot(), future).reasons)
        self.assertEqual(assess(snapshot(gpu_free_bytes=None), forecast()).status, "allow")

    def test_gpu_use_cannot_omit_free_or_measured_next_reservation(self) -> None:
        omitted = forecast(uses_gpu=True)
        result = assess(snapshot(), omitted)
        self.assertEqual(result.status, "block")
        self.assertIn("gpu_telemetry_missing", result.reasons)
        self.assertIn("gpu_forecast_unmeasured", result.reasons)
        self.assertIn("gpu_forecast_unmeasured", assess(
            snapshot(gpu_free_bytes=2 * GIB), omitted).reasons)
        measured_zero = forecast(uses_gpu=True, gpu_incremental_reserved_bytes=0,
                                 gpu_measurement_reference="synthetic measured steady-state reservation")
        self.assertEqual(assess(snapshot(gpu_free_bytes=2 * GIB), measured_zero).status, "allow")
        self.assertIn("gpu_forecast_without_gpu_use", assess(
            snapshot(), forecast(gpu_incremental_reserved_bytes=GIB)).reasons)

    def test_verified_unlimited_fallback_but_unknown_or_hidden_ancestor_blocks(self) -> None:
        leaf_max = snapshot(cgroup_leaf_limit_bytes="max", cgroup_leaf_current_bytes=None,
                            verified_min_cgroup_raw_headroom_bytes=None,
                            host_available_bytes=8 * GIB)
        result = assess(leaf_max, forecast())
        self.assertEqual(result.status, "block")
        self.assertIn("cgroup_min_raw_headroom_missing", result.reasons)
        verified = assess(replace(leaf_max, verified_effective_unlimited=True), forecast())
        self.assertEqual(verified.status, "warn")
        self.assertIsNone(verified.raw_cgroup_headroom_bytes)
        self.assertEqual(verified.possible_ram_capacity_bytes, 8 * GIB)
        self.assertIn("verified_unlimited_host_only_ram", verified.reasons)
        stale_cache = assess(
            replace(leaf_max, verified_effective_unlimited=True, inactive_file_bytes=GIB),
            forecast())
        self.assertEqual(stale_cache.status, "warn")
        self.assertEqual(stale_cache.possible_ram_capacity_bytes, 8 * GIB)
        self.assertIsNone(stale_cache.possible_clean_inactive_bytes)
        self.assertIn("cache_stats_ignored_without_leaf_current", stale_cache.reasons)
        # The leaf is max, but an observed ancestor is bounded and nearly full.
        parent = replace(leaf_max, verified_min_cgroup_raw_headroom_bytes=GIB)
        bounded = assess(parent, forecast())
        self.assertEqual(bounded.status, "block")
        self.assertEqual(bounded.raw_cgroup_headroom_bytes, GIB)
        self.assertIn("ram_capacity_shortfall", bounded.reasons)
        self.assertIn("ram_capacity_shortfall", assess(
            replace(leaf_max, verified_effective_unlimited=True,
                     host_available_bytes=4 * GIB), forecast()).reasons)

    def test_tight_parent_with_larger_limit_than_leaf_blocks(self) -> None:
        # Leaf 8-1=7 GiB; parent 64-63.9 ~= 0.1 GiB, including sibling charges.
        parent_gap = 64 * GIB - int(63.9 * GIB)
        current = snapshot(cgroup_leaf_limit_bytes=8 * GIB, cgroup_leaf_current_bytes=GIB,
                           verified_min_cgroup_raw_headroom_bytes=parent_gap,
                           cgroup_ancestry_evidence="synthetic leaf 8/1, parent 64/63.9 paired readings")
        result = assess(current, forecast())
        self.assertEqual(result.status, "block")
        self.assertEqual(result.raw_cgroup_headroom_bytes, parent_gap)
        self.assertIn("ram_next_capacity_shortfall", result.reasons)

    def test_tight_grandparent_despite_parent_minimum_absolute_limit_blocks(self) -> None:
        # Leaf max; parent 64-10=54 GiB; grandparent 72-71=1 GiB.
        current = snapshot(cgroup_leaf_limit_bytes="max", cgroup_leaf_current_bytes=None,
                           verified_min_cgroup_raw_headroom_bytes=GIB,
                           cgroup_ancestry_evidence="synthetic leaf max, parent 64/10, grandparent 72/71")
        result = assess(current, forecast())
        self.assertEqual(result.status, "block")
        self.assertEqual(result.raw_cgroup_headroom_bytes, GIB)
        self.assertIn("ram_next_capacity_shortfall", result.reasons)

    def test_unknown_leaf_missing_effective_usage_and_inconsistent_cache_block(self) -> None:
        cases = (
            (snapshot(cgroup_leaf_limit_bytes=None, verified_min_cgroup_raw_headroom_bytes=None),
             "cgroup_leaf_telemetry_missing"),
            (snapshot(verified_min_cgroup_raw_headroom_bytes=None),
             "cgroup_min_raw_headroom_missing"),
            (snapshot(cgroup_leaf_current_bytes=None),
             "cgroup_leaf_current_missing"),
            (snapshot(cgroup_leaf_current_bytes=10 * GIB,
                      inactive_file_bytes=11 * GIB),
             "inactive_file_exceeds_cgroup_current"),
            (snapshot(cgroup_leaf_limit_bytes=4 * GIB, cgroup_leaf_current_bytes=GIB),
             "cgroup_min_raw_headroom_exceeds_leaf"),
            (snapshot(verified_effective_unlimited=True),
             "conflicting_effective_cgroup_bound"),
            (snapshot(cgroup_leaf_limit_bytes="max",
                      verified_effective_unlimited=True),
             "conflicting_effective_cgroup_bound"),
            (snapshot(cgroup_ancestry_verified=False), "cgroup_ancestry_unverified"),
            (snapshot(cgroup_ancestry_evidence=None), "cgroup_ancestry_unverified"),
        )
        for current, reason in cases:
            with self.subTest(reason=reason):
                result = assess(current, forecast())
                self.assertEqual(result.status, "block")
                self.assertIn(reason, result.reasons)

    def test_missing_essential_telemetry_blocks(self) -> None:
        cases = (
            (snapshot(host_available_bytes=None), "host_memory_telemetry_missing"),
            (snapshot(disk_available_bytes=None), "disk_telemetry_missing"),
            (snapshot(cgroup_leaf_current_bytes=None), "cgroup_leaf_current_missing"),
        )
        for current, reason in cases:
            with self.subTest(reason=reason):
                result = assess(current, forecast())
                self.assertEqual(result.status, "block")
                self.assertIn(reason, result.reasons)

    def test_oom_delta_blocks_and_snapshot_survives_for_partial_receipt(self) -> None:
        current = snapshot(oom_events=3, previous_oom_events=2)
        result = assess(current, forecast(phase="continuation"))
        self.assertEqual(result.status, "block")
        self.assertIn("oom_event_increased", result.reasons)
        self.assertEqual(result.snapshot, current)
        self.assertEqual(result.forecast.phase, "continuation")
        self.assertEqual(json.loads(json.dumps(asdict(result)))["snapshot"]["oom_events"], 3)
        self.assertIn("oom_counter_reset_or_wrong_cgroup", assess(
            snapshot(oom_events=1, previous_oom_events=2), forecast()).reasons)
        missing = assess(snapshot(oom_events=None, previous_oom_events=2), forecast())
        self.assertEqual(missing.status, "warn")
        self.assertIn("oom_telemetry_missing", missing.reasons)
        self.assertIn("oom_telemetry_missing", assess(
            snapshot(oom_events=None, previous_oom_events=None), forecast()).reasons)
        self.assertEqual(assess(snapshot(oom_events=2, previous_oom_events=2),
                                forecast()).status, "allow")
        self.assertIn("oom_trend_unavailable", assess(
            snapshot(oom_events=2, previous_oom_events=None), forecast()).reasons)

    def test_cpu_contention_warns_but_explicit_wall_budget_can_block(self) -> None:
        current = snapshot(cpu_contended=True)
        result = assess(current, forecast(estimated_remaining_wall_seconds=5000))
        self.assertEqual(result.status, "warn")
        self.assertIn("cpu_contention_wall_risk", result.reasons)
        limited = assess(current, forecast(estimated_remaining_wall_seconds=5000,
                                           remaining_wall_budget_seconds=4000))
        self.assertEqual(limited.status, "block")
        self.assertIn("wall_budget_shortfall", limited.reasons)
        self.assertIn("wall_forecast_missing", assess(snapshot(), forecast(
            remaining_wall_budget_seconds=4000)).reasons)
        self.assertEqual(assess(snapshot(), forecast(
            estimated_remaining_wall_seconds=10**400)).status, "allow")

    def test_malformed_or_unmeasured_essential_fields_fail_closed(self) -> None:
        cases = (
            (snapshot(host_available_bytes=-1), forecast(), "invalid_snapshot.host_available_bytes"),
            (snapshot(file_dirty_bytes=-1), forecast(), "invalid_snapshot.file_dirty_bytes"),
            (snapshot(cgroup_leaf_limit_bytes="bad"), forecast(),
             "invalid_snapshot.cgroup_leaf_limit_bytes"),
            (snapshot(verified_min_cgroup_raw_headroom_bytes=-1), forecast(),
             "invalid_snapshot.verified_min_cgroup_raw_headroom_bytes"),
            (snapshot(cgroup_ancestry_verified=1), forecast(),
             "invalid_snapshot.cgroup_ancestry_verified"),
            (snapshot(cgroup_ancestry_evidence=" "), forecast(),
             "invalid_snapshot.cgroup_ancestry_evidence"),
            (snapshot(verified_effective_unlimited=1), forecast(),
             "invalid_snapshot.verified_effective_unlimited"),
            (snapshot(temp_on_output_filesystem="yes"), forecast(),
             "invalid_snapshot.temp_on_output_filesystem"),
            (snapshot(gpu_free_bytes=-1), forecast(), "invalid_snapshot.gpu_free_bytes"),
            (snapshot(), forecast(incremental_ram_bytes=True), "invalid_forecast.incremental_ram_bytes"),
            (snapshot(), forecast(remaining_peak_incremental_ram_bytes=-1),
             "invalid_forecast.remaining_peak_incremental_ram_bytes"),
            (snapshot(), forecast(uses_gpu=None), "invalid_forecast.uses_gpu"),
            (snapshot(), forecast(uses_gpu=True, gpu_incremental_reserved_bytes=-1),
             "invalid_forecast.gpu_incremental_reserved_bytes"),
            (snapshot(), forecast(uses_gpu=True, gpu_measurement_reference=" "),
             "invalid_forecast.gpu_measurement_reference"),
            (snapshot(), forecast(next_write_bytes=-1), "invalid_forecast.next_write_bytes"),
            (snapshot(), forecast(estimated_remaining_wall_seconds=float("nan")),
             "invalid_forecast.estimated_remaining_wall_seconds"),
            (snapshot(), forecast(remaining_wall_budget_seconds=float("inf")),
             "invalid_forecast.remaining_wall_budget_seconds"),
            (snapshot(), forecast(measurement_reference=" "),
             "invalid_forecast.measurement_reference"),
        )
        for current, future, reason in cases:
            with self.subTest(reason=reason):
                result = assess(current, future)
                self.assertEqual(result.status, "block")
                self.assertIn(reason, result.reasons)
