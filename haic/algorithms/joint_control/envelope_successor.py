"""Performance-envelope adapter; the pinned successor's physical guards stay fixed."""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from time import perf_counter, process_time
from types import FunctionType, MethodType
from typing import Any

import numpy as np

from . import paired_residual
from .successor import TemporalSuccessor


PERFORMANCE_METHOD = "shared_paired_envelope_excess"
PERFORMANCE_TARGET = "per-reference actual paired delta outside shared-state envelope"


class EnvelopeSuccessor(TemporalSuccessor):
    """Replace only the performance comparator in an isolated copy of old act.

    ``calibration`` is the complete paired-envelope JSON mapping. Its unchanged
    ``absolute_calibration`` drives all inherited observer/mapping/physical gates.
    Construction rejects incomplete calibration, not substitutes zero residuals.
    Inputs are copied, including tuple-valued residual arrays; inference never
    fits or modifies them. The caller owns artifact/source provenance checks.
    The .05 residual floor is a new performance-calibration choice; the separate
    material-gain threshold remains unchanged.

    Candidate direction, H4, physics, eligibility, forty-intervention budget,
    query/commit lifecycle and actual-action feedback are inherited verbatim.
    Champion ownership remains exclusive and calls serial. No frozen source or
    module-global binding is edited, even temporarily.
    """

    def __init__(self, champion, calibration, *, never_intervene=False):
        if (not isinstance(calibration, Mapping)
                or calibration.get("schema") != paired_residual.SCHEMA
                or calibration.get("calibration_complete") is not True):
            raise ValueError("complete paired-envelope calibration required")
        self.envelope_calibration: dict[str, Any] = deepcopy(dict(calibration))
        performance = self.envelope_calibration.get("performance")
        if not isinstance(performance, Mapping):
            raise ValueError("performance residual calibration required")
        values: tuple[Any, ...] = tuple(performance.get(key) for key in ("lower", "upper", "floor"))
        if (any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) for value in values)
                or not np.isfinite(values).all()
                or min(values[:2]) < paired_residual.RESIDUAL_FLOOR
                or values[2] != paired_residual.RESIDUAL_FLOOR
                or performance.get("target") != PERFORMANCE_TARGET):
            raise ValueError("finite envelope residuals with the declared .05 floor and target required")
        absolute = self.envelope_calibration.get("absolute_calibration")
        if not isinstance(absolute, Mapping):
            raise ValueError("complete original absolute calibration required")
        super().__init__(champion, absolute, never_intervene=never_intervene)
        _, reason = self._calibration()
        if reason is not None:
            raise ValueError("invalid original absolute calibration: " + reason)
        # Share the one private absolute copy between inherited gates and scorer.
        self.envelope_calibration["absolute_calibration"] = self.calibration

        def compare(scene, hypotheses, actions, *, observer_valid, **_old_residuals):
            result = paired_residual.compare_candidates(
                scene, hypotheses, actions, observer_valid=observer_valid,
                calibration=self.envelope_calibration)
            # Reconstruct the old once-expanded interval from the SAME prediction,
            # only for diagnosis. Do not call the predictor or scorer a second time.
            delta = np.asarray(result["reference_delta"], dtype=float)
            supported = np.asarray(result["cost_supported"], dtype=bool) & np.isfinite(delta).all(axis=(1, 2))
            old_interval = np.full((len(delta), 2), np.nan)
            old_interval[supported, 0] = delta[supported].min(axis=(1, 2))
            old_interval[supported, 1] = delta[supported].max(axis=(1, 2))
            q = float(self.calibration["paired_cost_residual"])
            old_interval[1:] += [-q, q]
            return dict(result, old_delta_interval=old_interval,
                        old_paired_cost_residual=q, performance_method=PERFORMANCE_METHOD)

        original = TemporalSuccessor.act
        namespace = dict(original.__globals__, compare_candidates=compare)
        function = FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        self._envelope_act = MethodType(function, self)

    def act(self, observation):
        cpu_start, wall_start = process_time(), perf_counter()
        try:
            return self._envelope_act(observation)
        finally:
            performance = self.envelope_calibration["performance"]
            self.last_diagnostics.update(
                performance_method=PERFORMANCE_METHOD, performance_target=performance["target"],
                performance_residual_lower=float(performance["lower"]),
                performance_residual_upper=float(performance["upper"]))
            # Supersede the inner timer: include wrapper and provenance work too.
            # The runner's external full-call timer remains authoritative.
            self.last_diagnostics["act_cpu_seconds"] = process_time() - cpu_start
            self.last_diagnostics["act_wall_seconds"] = perf_counter() - wall_start
