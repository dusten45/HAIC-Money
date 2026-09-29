"""Pure, future-only local TRAIN resource assessment; no trainer imports this module.

The caller supplies contemporaneous measurements and a per-study measured forecast
for the *remaining* phase, not a process peak or a universal free-space floor.
``assess`` does no I/O, allocation, admission, or receipt writing. A future
protocol must bind the forecast evidence, verify the cgroup ancestry and
filesystem layout, and remeasure before costly operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


Phase = Literal["admission", "continuation"]
Status = Literal["allow", "warn", "block"]


@dataclass(frozen=True)
class ResourceSnapshot:
    host_available_bytes: int | None  # /proc/meminfo MemAvailable
    disk_available_bytes: int | None  # statvfs(output parent), f_bavail * f_frsize
    cgroup_leaf_limit_bytes: int | Literal["max"] | None = None  # leaf alone is not a bound
    cgroup_leaf_current_bytes: int | None = None  # for leaf cache sanity, not headroom
    verified_min_cgroup_raw_headroom_bytes: int | None = None  # min(max-current) per finite scope
    cgroup_ancestry_verified: bool = False  # evidence includes leaf and every ancestor
    cgroup_ancestry_evidence: str | None = None  # identities and same-scope limit/current samples
    verified_effective_unlimited: bool = False  # leaf and all ancestors verified 'max'
    inactive_file_bytes: int | None = None  # memory.stat at the leaf cgroup
    file_dirty_bytes: int | None = None
    file_writeback_bytes: int | None = None
    temp_on_output_filesystem: bool | None = None  # verified same filesystem for writes
    gpu_free_bytes: int | None = None  # measured for the selected device, if used
    oom_events: int | None = None  # memory.events oom at the monitored cgroup
    previous_oom_events: int | None = None  # same cgroup, earlier observation
    cpu_contended: bool | None = None


@dataclass(frozen=True)
class ResourceForecast:
    phase: Phase
    measurement_reference: str  # study/phase-specific measurement receipt or protocol
    uses_gpu: bool  # explicitly declare CUDA/device use even when next demand is zero
    incremental_ram_bytes: int  # additional next-phase demand, not already-allocated RAM
    remaining_peak_incremental_ram_bytes: int  # additional peak through remaining study
    competing_ram_growth_bytes: int
    ram_reserve_bytes: int
    next_write_bytes: int  # permanent bytes for the next write
    remaining_output_bytes: int  # all remaining immutable checkpoints/logs incl. next
    peak_temp_bytes: int  # extra space while a write/serialization is in progress
    competing_disk_growth_bytes: int
    disk_reserve_bytes: int
    gpu_incremental_reserved_bytes: int | None = None  # measured next context/reserved growth
    gpu_measurement_reference: str | None = None
    competing_gpu_growth_bytes: int = 0
    gpu_reserve_bytes: int = 0
    estimated_remaining_wall_seconds: float | None = None
    remaining_wall_budget_seconds: float | None = None


@dataclass(frozen=True)
class ResourceAssessment:
    status: Status
    reasons: tuple[str, ...]
    snapshot: ResourceSnapshot
    forecast: ResourceForecast
    raw_cgroup_headroom_bytes: int | None = None
    conservative_ram_capacity_bytes: int | None = None
    possible_ram_capacity_bytes: int | None = None
    possible_clean_inactive_bytes: int | None = None
    required_next_ram_bytes: int | None = None
    required_ram_bytes: int | None = None
    required_next_disk_bytes: int | None = None
    required_remaining_disk_bytes: int | None = None
    required_gpu_bytes: int | None = None


def assess(snapshot: ResourceSnapshot, forecast: ResourceForecast) -> ResourceAssessment:
    """Return a structured decision, never assuming reclaim or an idle device.

    ``possible_ram_capacity_bytes`` is an upper *possibility*, not a guarantee:
    clean inactive cgroup file pages might be reclaimed, subject to host available
    memory. Dirty/writeback pages are never counted. A cache-dependent assessment
    BLOCKS until reclaim and remeasurement establish sufficient raw capacity.
    A caller must supply the verified minimum of (memory.max - memory.current)
    over *each finite* cgroup in the leaf-to-root chain; the smallest limit with
    its own current is not enough when another ancestor has higher sibling use.
    Missing ancestry evidence blocks; only a verified all-unlimited chain falls
    back to host availability (with a warning). Cache counters and optional leaf
    current must share a scope; temp and output must share a verified filesystem.
    Every forecast byte is incremental/remaining at this phase; callers must not
    feed an already allocated process peak or already written checkpoints again.
    """
    invalid: list[str] = []
    for name in (
        "host_available_bytes", "disk_available_bytes", "cgroup_leaf_current_bytes",
        "verified_min_cgroup_raw_headroom_bytes",
        "inactive_file_bytes", "file_dirty_bytes", "file_writeback_bytes",
        "gpu_free_bytes", "oom_events", "previous_oom_events",
    ):
        value = getattr(snapshot, name)
        if value is not None and (type(value) is not int or value < 0):
            invalid.append(f"invalid_snapshot.{name}")
    leaf = snapshot.cgroup_leaf_limit_bytes
    if leaf is not None and leaf != "max" and (type(leaf) is not int or leaf < 0):
        invalid.append("invalid_snapshot.cgroup_leaf_limit_bytes")
    if type(snapshot.cgroup_ancestry_verified) is not bool:
        invalid.append("invalid_snapshot.cgroup_ancestry_verified")
    evidence = snapshot.cgroup_ancestry_evidence
    if evidence is not None and (not isinstance(evidence, str) or not evidence.strip()):
        invalid.append("invalid_snapshot.cgroup_ancestry_evidence")
    if type(snapshot.verified_effective_unlimited) is not bool:
        invalid.append("invalid_snapshot.verified_effective_unlimited")
    if (snapshot.temp_on_output_filesystem is not None
            and type(snapshot.temp_on_output_filesystem) is not bool):
        invalid.append("invalid_snapshot.temp_on_output_filesystem")
    if snapshot.cpu_contended is not None and type(snapshot.cpu_contended) is not bool:
        invalid.append("invalid_snapshot.cpu_contended")
    if forecast.phase not in ("admission", "continuation"):
        invalid.append("invalid_forecast.phase")
    if not isinstance(forecast.measurement_reference, str) or not forecast.measurement_reference.strip():
        invalid.append("invalid_forecast.measurement_reference")
    if type(forecast.uses_gpu) is not bool:
        invalid.append("invalid_forecast.uses_gpu")
    for name in (
        "incremental_ram_bytes", "remaining_peak_incremental_ram_bytes",
        "competing_ram_growth_bytes", "ram_reserve_bytes",
        "next_write_bytes", "remaining_output_bytes", "peak_temp_bytes",
        "competing_disk_growth_bytes", "disk_reserve_bytes",
        "competing_gpu_growth_bytes", "gpu_reserve_bytes",
    ):
        value = getattr(forecast, name)
        if type(value) is not int or value < 0:
            invalid.append(f"invalid_forecast.{name}")
    gpu_growth = forecast.gpu_incremental_reserved_bytes
    if gpu_growth is not None and (type(gpu_growth) is not int or gpu_growth < 0):
        invalid.append("invalid_forecast.gpu_incremental_reserved_bytes")
    gpu_reference = forecast.gpu_measurement_reference
    if gpu_reference is not None and (not isinstance(gpu_reference, str) or not gpu_reference.strip()):
        invalid.append("invalid_forecast.gpu_measurement_reference")
    for name in ("estimated_remaining_wall_seconds", "remaining_wall_budget_seconds"):
        value = getattr(forecast, name)
        if value is not None and (type(value) not in (int, float) or value < 0
                                  or (type(value) is float and not math.isfinite(value))):
            invalid.append(f"invalid_forecast.{name}")
    if invalid:
        return ResourceAssessment("block", tuple(invalid), snapshot, forecast)

    blocks: list[str] = []
    warnings: list[str] = []
    if snapshot.host_available_bytes is None:
        blocks.append("host_memory_telemetry_missing")
    if snapshot.disk_available_bytes is None:
        blocks.append("disk_telemetry_missing")

    raw: int | None = None
    current = snapshot.cgroup_leaf_current_bytes
    measured_min = snapshot.verified_min_cgroup_raw_headroom_bytes
    ancestry_proven = snapshot.cgroup_ancestry_verified and evidence is not None
    if not ancestry_proven:
        blocks.append("cgroup_ancestry_unverified")
    if leaf is None:
        blocks.append("cgroup_leaf_telemetry_missing")
    if measured_min is None:
        if not snapshot.verified_effective_unlimited or leaf != "max":
            blocks.append("cgroup_min_raw_headroom_missing")
        elif ancestry_proven:
            warnings.append("verified_unlimited_host_only_ram")
    else:
        if snapshot.verified_effective_unlimited:
            blocks.append("conflicting_effective_cgroup_bound")
        if ancestry_proven and leaf is not None:
            raw = measured_min
        if type(leaf) is int:
            if current is None:
                blocks.append("cgroup_leaf_current_missing")
            elif current > leaf:
                blocks.append("cgroup_leaf_current_exceeds_limit")
            elif measured_min > leaf - current:
                blocks.append("cgroup_min_raw_headroom_exceeds_leaf")

    if snapshot.inactive_file_bytes is not None:
        if current is None and snapshot.inactive_file_bytes > 0:
            warnings.append("cache_stats_ignored_without_leaf_current")
        elif current is not None and snapshot.inactive_file_bytes > current:
            blocks.append("inactive_file_exceeds_cgroup_current")

    clean: int | None = None
    inactive, dirty, writeback = (snapshot.inactive_file_bytes, snapshot.file_dirty_bytes,
                                  snapshot.file_writeback_bytes)
    if current is not None and inactive is not None and dirty is not None and writeback is not None:
        clean = max(0, inactive - dirty - writeback)

    conservative: int | None = None
    possible: int | None = None
    if snapshot.host_available_bytes is not None and (raw is not None or
            (ancestry_proven and snapshot.verified_effective_unlimited
             and leaf == "max" and measured_min is None)):
        conservative = (min(raw, snapshot.host_available_bytes)
                        if raw is not None else snapshot.host_available_bytes)
        possible = (min(raw + (clean or 0), snapshot.host_available_bytes)
                    if raw is not None else snapshot.host_available_bytes)

    next_ram_required = (forecast.incremental_ram_bytes + forecast.competing_ram_growth_bytes
                         + forecast.ram_reserve_bytes)
    ram_required = (forecast.remaining_peak_incremental_ram_bytes
                    + forecast.competing_ram_growth_bytes + forecast.ram_reserve_bytes)
    next_disk_required = (forecast.next_write_bytes + forecast.peak_temp_bytes
                          + forecast.competing_disk_growth_bytes + forecast.disk_reserve_bytes)
    remaining_disk_required = (forecast.remaining_output_bytes + forecast.peak_temp_bytes
                               + forecast.competing_disk_growth_bytes + forecast.disk_reserve_bytes)
    gpu_required = ((gpu_growth + forecast.competing_gpu_growth_bytes + forecast.gpu_reserve_bytes)
                    if gpu_growth is not None else None)

    if forecast.remaining_peak_incremental_ram_bytes < forecast.incremental_ram_bytes:
        blocks.append("remaining_ram_peak_below_next_demand")
    if possible is not None and conservative is not None:
        if next_ram_required > conservative:
            blocks.append("ram_next_capacity_shortfall")
        if ram_required > possible:
            blocks.append("ram_capacity_shortfall")
            if raw is not None and clean is None and ram_required > conservative:
                blocks.append("cache_telemetry_missing_for_reclaim")
        if ram_required > conservative:
            blocks.append("ram_conservative_capacity_shortfall")
        if raw is not None and clean is not None and ram_required <= possible and ram_required > conservative:
            warnings.append("ram_reclaim_dependent")
    if forecast.peak_temp_bytes and snapshot.temp_on_output_filesystem is not True:
        blocks.append("temp_filesystem_not_verified_same_as_output")
    if forecast.next_write_bytes > forecast.remaining_output_bytes:
        blocks.append("next_write_exceeds_remaining_output_forecast")
    if snapshot.disk_available_bytes is not None:
        if next_disk_required > snapshot.disk_available_bytes:
            blocks.append("disk_next_write_shortfall")
        if remaining_disk_required > snapshot.disk_available_bytes:
            blocks.append("disk_remaining_shortfall")
    if forecast.uses_gpu:
        if gpu_growth is None or gpu_reference is None:
            blocks.append("gpu_forecast_unmeasured")
        if snapshot.gpu_free_bytes is None:
            blocks.append("gpu_telemetry_missing")
        elif gpu_required is not None and gpu_required > snapshot.gpu_free_bytes:
            blocks.append("gpu_capacity_shortfall")
    elif gpu_growth is not None or gpu_reference is not None or forecast.competing_gpu_growth_bytes or forecast.gpu_reserve_bytes:
        blocks.append("gpu_forecast_without_gpu_use")
    if snapshot.previous_oom_events is not None:
        if snapshot.oom_events is None:
            warnings.append("oom_telemetry_missing")
        elif snapshot.oom_events > snapshot.previous_oom_events:
            blocks.append("oom_event_increased")
        elif snapshot.oom_events < snapshot.previous_oom_events:
            blocks.append("oom_counter_reset_or_wrong_cgroup")
    elif snapshot.oom_events is not None:
        warnings.append("oom_trend_unavailable")
    else:
        warnings.append("oom_telemetry_missing")
    if snapshot.cpu_contended:
        warnings.append("cpu_contention_wall_risk")
    if forecast.remaining_wall_budget_seconds is not None:
        if forecast.estimated_remaining_wall_seconds is None:
            blocks.append("wall_forecast_missing")
        elif forecast.estimated_remaining_wall_seconds > forecast.remaining_wall_budget_seconds:
            blocks.append("wall_budget_shortfall")

    status: Status = "block" if blocks else "warn" if warnings else "allow"
    return ResourceAssessment(
        status=status, reasons=tuple(blocks + warnings), snapshot=snapshot, forecast=forecast,
        raw_cgroup_headroom_bytes=raw, conservative_ram_capacity_bytes=conservative,
        possible_ram_capacity_bytes=possible, possible_clean_inactive_bytes=clean,
        required_next_ram_bytes=next_ram_required, required_ram_bytes=ram_required,
        required_next_disk_bytes=next_disk_required,
        required_remaining_disk_bytes=remaining_disk_required, required_gpu_bytes=gpu_required,
    )
