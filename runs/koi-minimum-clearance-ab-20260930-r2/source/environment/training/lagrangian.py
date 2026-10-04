"""TRAIN-only collision-cost accounting and Lagrange multiplier updates.

This module is intentionally independent of policy rollout and inference code.
Callers must pass the environment's real end flags separately from a collector
cutoff, and must submit only completed TRAIN episodes to the multiplier.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import math

from damage import DAMAGE_PER_COLLISION, MAX_DAMAGE


TRAIN_SPLIT = "train"
REFERENCE_LAMBDA = 30.0
DAMAGE_STOCK_NORMALIZER = 3000.0
OFF_TRACK_COST = 1.0 / 30.0
EPISODE_FAILURE_COST = 0.2
DEFAULT_COST_BUDGET = 0.05
DEFAULT_DUAL_STEP_SIZE = 5.0
DEFAULT_WINDOW_SIZE = 8
DEFAULT_MIN_MULTIPLIER = 0.0
DEFAULT_MAX_MULTIPLIER = 60.0


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _damage_value(name: str, value: float) -> float:
    result = _finite_nonnegative(name, value)
    if result > MAX_DAMAGE:
        raise ValueError(f"{name} must not exceed the environment damage cap")
    return result


@dataclass(frozen=True)
class EpisodeBoundary:
    """Separate environment episode ends from rollout collector cutoffs.

    A Gymnasium finish may report ``environment_truncated=True`` with
    ``finished=True``. The trainer's registered per-episode decision cap is
    represented by ``episode_time_limit``. A PPO batch/collector cutoff sets
    only ``collector_truncated`` and must not complete an episode or incur
    the terminal-failure cost.
    """

    terminated: bool = False
    environment_truncated: bool = False
    episode_time_limit: bool = False
    collector_truncated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "terminated",
            "environment_truncated",
            "episode_time_limit",
            "collector_truncated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    @property
    def episode_done(self) -> bool:
        return (
            self.terminated
            or self.environment_truncated
            or self.episode_time_limit
        )

    @property
    def collector_only(self) -> bool:
        return self.collector_truncated and not self.episode_done


@dataclass(frozen=True)
class CostComponents:
    """Non-negative costs for one transition or their episode-wise sums."""

    contact_cost: float = 0.0
    damage_stock_cost: float = 0.0
    off_track_cost: float = 0.0
    failure_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "contact_cost",
            "damage_stock_cost",
            "off_track_cost",
            "failure_cost",
        ):
            _finite_nonnegative(name, getattr(self, name))

    @property
    def total_cost(self) -> float:
        return math.fsum(
            (
                self.contact_cost,
                self.damage_stock_cost,
                self.off_track_cost,
                self.failure_cost,
            )
        )

    def __add__(self, other: object) -> "CostComponents":
        if not isinstance(other, CostComponents):
            return NotImplemented
        return CostComponents(
            contact_cost=self.contact_cost + other.contact_cost,
            damage_stock_cost=self.damage_stock_cost + other.damage_stock_cost,
            off_track_cost=self.off_track_cost + other.off_track_cost,
            failure_cost=self.failure_cost + other.failure_cost,
        )


def compute_transition_cost(
    *,
    damage_before: float,
    damage_after: float,
    collision: bool,
    off_track: bool,
    finished: bool,
    boundary: EpisodeBoundary,
) -> CostComponents:
    """Convert TRAIN transition labels into the frozen scalar-cost components.

    The contact term uses the larger of positive damage increase and the
    per-collision-decision floor. Thus one colliding decision that also adds
    0.2 damage is counted once, while a collision at the damage cap still
    receives the collision cost. Collision onsets are a separate metric.
    """
    if not isinstance(boundary, EpisodeBoundary):
        raise TypeError("boundary must be an EpisodeBoundary")
    for name, value in (
        ("collision", collision),
        ("off_track", off_track),
        ("finished", finished),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")

    previous_damage = _damage_value("damage_before", damage_before)
    next_damage = _damage_value("damage_after", damage_after)
    damage_increase = max(0.0, next_damage - previous_damage)
    collision_floor = DAMAGE_PER_COLLISION if collision else 0.0
    failure = (
        EPISODE_FAILURE_COST
        if boundary.episode_done and not finished
        else 0.0
    )
    return CostComponents(
        contact_cost=max(damage_increase, collision_floor),
        damage_stock_cost=next_damage / DAMAGE_STOCK_NORMALIZER,
        off_track_cost=OFF_TRACK_COST if off_track else 0.0,
        failure_cost=failure,
    )


@dataclass(frozen=True)
class CompletedEpisodeCost:
    """Cost record eligible for a TRAIN multiplier window."""

    components: CostComponents
    steps: int
    split: str = TRAIN_SPLIT

    def __post_init__(self) -> None:
        if not isinstance(self.components, CostComponents):
            raise TypeError("components must be CostComponents")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.split != TRAIN_SPLIT:
            raise ValueError("only completed TRAIN episodes can enter cost windows")

    @property
    def total_cost(self) -> float:
        return self.components.total_cost


@dataclass(frozen=True)
class IncompleteEpisodeCost:
    """Diagnostic snapshot of a fragment discarded before environment end."""

    components: CostComponents
    steps: int
    split: str = TRAIN_SPLIT

    @property
    def total_cost(self) -> float:
        return self.components.total_cost


class EpisodeCostAccumulator:
    """Accumulate transitions and emit a record only at a real env episode end."""

    def __init__(self) -> None:
        self._components = CostComponents()
        self._steps = 0

    @property
    def pending_cost(self) -> float:
        return self._components.total_cost

    @property
    def pending_steps(self) -> int:
        return self._steps

    def add_transition(
        self,
        cost: CostComponents,
        boundary: EpisodeBoundary,
        *,
        split: str,
    ) -> CompletedEpisodeCost | None:
        """Add a TRAIN transition; collector-only cuts leave it incomplete.

        If a collector fragment is resumed, keep using this accumulator. If
        the caller resets the environment instead, call ``discard_partial``;
        that diagnostic record cannot be submitted to the multiplier.
        """
        if split != TRAIN_SPLIT:
            raise ValueError("Lagrangian safety costs accept TRAIN data only")
        if not isinstance(cost, CostComponents):
            raise TypeError("cost must be CostComponents")
        if not isinstance(boundary, EpisodeBoundary):
            raise TypeError("boundary must be an EpisodeBoundary")

        self._components = self._components + cost
        self._steps += 1
        if not boundary.episode_done:
            return None

        completed = CompletedEpisodeCost(
            components=self._components,
            steps=self._steps,
            split=TRAIN_SPLIT,
        )
        self._clear()
        return completed

    def discard_partial(self) -> IncompleteEpisodeCost | None:
        """Clear a fragment when its environment is reset before true episode end."""
        if self._steps == 0:
            return None
        incomplete = IncompleteEpisodeCost(
            components=self._components,
            steps=self._steps,
            split=TRAIN_SPLIT,
        )
        self._clear()
        return incomplete

    def _clear(self) -> None:
        self._components = CostComponents()
        self._steps = 0


@dataclass(frozen=True)
class MultiplierWindow:
    """One common eight-episode report window for either multiplier arm."""

    mode: str
    episode_count: int
    mean_cost: float
    multiplier_before: float
    multiplier_after: float


class LagrangeMultiplier:
    """Fixed or projected-dual λ updated from completed TRAIN episodes only."""

    def __init__(
        self,
        *,
        mode: str = "fixed",
        initial_value: float = REFERENCE_LAMBDA,
        cost_budget: float = DEFAULT_COST_BUDGET,
        dual_step_size: float = DEFAULT_DUAL_STEP_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        minimum: float = DEFAULT_MIN_MULTIPLIER,
        maximum: float = DEFAULT_MAX_MULTIPLIER,
    ) -> None:
        if mode not in ("fixed", "adaptive"):
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        self._minimum = _finite_nonnegative("minimum", minimum)
        self._maximum = _finite_nonnegative("maximum", maximum)
        if self._minimum > self._maximum:
            raise ValueError("minimum must not exceed maximum")
        self._value = _finite_nonnegative("initial_value", initial_value)
        if not self._minimum <= self._value <= self._maximum:
            raise ValueError("initial_value must be within multiplier bounds")
        self._cost_budget = _finite_nonnegative("cost_budget", cost_budget)
        self._dual_step_size = _finite_nonnegative("dual_step_size", dual_step_size)
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        self._mode = mode
        self._window_size = window_size
        self._window_costs: list[float] = []
        self._completed_episode_count = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def window_episode_count(self) -> int:
        return len(self._window_costs)

    @property
    def completed_episode_count(self) -> int:
        return self._completed_episode_count

    def state_dict(self) -> dict[str, object]:
        """Return resumable multiplier state, including a partial λ window."""
        return {
            "version": 1,
            "mode": self._mode,
            "value": self._value,
            "cost_budget": self._cost_budget,
            "dual_step_size": self._dual_step_size,
            "window_size": self._window_size,
            "minimum": self._minimum,
            "maximum": self._maximum,
            "window_costs": list(self._window_costs),
            "completed_episode_count": self._completed_episode_count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore λ only when its dual-update configuration still matches."""
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("unsupported Lagrange multiplier state")
        if state.get("mode") != self._mode:
            raise ValueError("saved Lagrange mode does not match the requested mode")
        if state.get("window_size") != self._window_size:
            raise ValueError("saved Lagrange window_size does not match the requested value")
        for name, expected in (
            ("cost_budget", self._cost_budget),
            ("dual_step_size", self._dual_step_size),
            ("minimum", self._minimum),
            ("maximum", self._maximum),
        ):
            saved = _finite_nonnegative(f"saved {name}", state.get(name))
            if saved != expected:
                raise ValueError(f"saved Lagrange {name} does not match the requested value")

        value = _finite_nonnegative("saved multiplier value", state.get("value"))
        if not self._minimum <= value <= self._maximum:
            raise ValueError("saved multiplier value is outside the configured bounds")
        raw_window_costs = state.get("window_costs")
        if not isinstance(raw_window_costs, (list, tuple)):
            raise ValueError("saved window_costs must be a sequence")
        window_costs = [
            _finite_nonnegative("saved episode cost", item) for item in raw_window_costs
        ]
        if len(window_costs) >= self._window_size:
            raise ValueError("saved multiplier window must be incomplete")
        completed_count = state.get("completed_episode_count")
        if (
            isinstance(completed_count, bool)
            or not isinstance(completed_count, int)
            or completed_count < len(window_costs)
            or completed_count % self._window_size != len(window_costs)
        ):
            raise ValueError("saved completed episode count is inconsistent with its window")

        self._value = value
        self._window_costs = window_costs
        self._completed_episode_count = completed_count

    def observe_completed_episode(
        self, episode: CompletedEpisodeCost
    ) -> MultiplierWindow | None:
        """Observe one fully ended TRAIN episode, updating every full window."""
        if not isinstance(episode, CompletedEpisodeCost):
            raise TypeError("only CompletedEpisodeCost records may update λ")
        if episode.split != TRAIN_SPLIT:
            raise ValueError("only TRAIN episode costs may update λ")
        self._window_costs.append(_finite_nonnegative("episode cost", episode.total_cost))
        self._completed_episode_count += 1
        if len(self._window_costs) < self._window_size:
            return None

        mean_cost = math.fsum(self._window_costs) / self._window_size
        before = self._value
        if self._mode == "adaptive":
            self._value = min(
                self._maximum,
                max(
                    self._minimum,
                    self._value + self._dual_step_size * (mean_cost - self._cost_budget),
                ),
            )
        window = MultiplierWindow(
            mode=self._mode,
            episode_count=self._completed_episode_count,
            mean_cost=mean_cost,
            multiplier_before=before,
            multiplier_after=self._value,
        )
        self._window_costs.clear()
        return window


__all__ = [
    "CompletedEpisodeCost",
    "CostComponents",
    "DEFAULT_COST_BUDGET",
    "DEFAULT_DUAL_STEP_SIZE",
    "DEFAULT_MAX_MULTIPLIER",
    "DEFAULT_MIN_MULTIPLIER",
    "DEFAULT_WINDOW_SIZE",
    "EPISODE_FAILURE_COST",
    "EpisodeBoundary",
    "EpisodeCostAccumulator",
    "IncompleteEpisodeCost",
    "LagrangeMultiplier",
    "MultiplierWindow",
    "OFF_TRACK_COST",
    "REFERENCE_LAMBDA",
    "TRAIN_SPLIT",
    "compute_transition_cost",
]
