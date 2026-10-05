"""Empirical error outside a shared paired-cost envelope, not every point in it.

Physical trajectory/mapping uncertainty is a separate veto and is NOT reduced.
Finite CAL maxima are not probabilistic coverage or closed-loop safety proofs.
"""

from copy import deepcopy
from typing import Any

import numpy as np

from . import comparison
from . import interval_comparison


SCHEMA = "haic-joint-temporal-paired-envelope-calibration-v1"
COST_FLOOR = comparison.COST_FLOOR
# New conservative residual floor, numerically equal to the unchanged material
# benefit threshold. The old every-scenario calibration had NO residual floor.
RESIDUAL_FLOOR = .05


def envelope_excess(predicted_delta, actual_delta) -> dict[str, Any]:
    """Keep each road realization paired; reduce only the shared state axis."""
    predicted, actual = np.asarray(predicted_delta, float), np.asarray(actual_delta, float)
    if (predicted.ndim != 2 or not all(predicted.shape) or actual.shape != predicted.shape[1:]
            or not np.isfinite(predicted).all() or not np.isfinite(actual).all()):
        raise ValueError("finite [state,reference] prediction and same-reference actual deltas required")
    lower, upper = predicted.min(axis=0), predicted.max(axis=0)
    return dict(lower=float(np.maximum(lower - actual, 0.).max()),
                upper=float(np.maximum(actual - upper, 0.).max()),
                predicted_lower=lower.tolist(), predicted_upper=upper.tolist(),
                actual=actual.tolist())


def fit_calibration(rows, absolute_calibration) -> dict[str, Any]:
    """Rows carry explicit CAL roles; held-out values cannot alter the fit.

    Unsupported CAL rows remain reported, never become a zero residual. Repeats
    do not add independent samples or alter a maximum. Empty support fails closed.
    """
    absolute = deepcopy(absolute_calibration)
    if absolute.get("calibration_complete") is not True:
        raise ValueError("original trajectory/cost calibration is incomplete")
    cal_seeds = set(absolute["CAL"])
    held_seeds = set(absolute["HELD_OUT_FROM_FIT"])
    if cal_seeds & held_seeds:
        raise ValueError("CAL and HELD_OUT geometries overlap")
    records, maxima = [], []
    for row in rows:
        seed = int(row["seed"])
        expected = "CAL" if seed in cal_seeds else "HELD_OUT_FROM_FIT" if seed in held_seeds else None
        if expected is None or row["role"] != expected:
            raise ValueError("row role differs from frozen geometry split")
        if expected != "CAL":
            continue
        entry = dict(anchor_id=row["anchor_id"], seed=seed, supported=bool(row["supported"]))
        if entry["supported"]:
            excess = envelope_excess(row["predicted_delta"], row["actual_delta"])
            entry.update(excess)
            maxima.append((excess["lower"], excess["upper"]))
        records.append(entry)
    if not maxima:
        raise ValueError("no supported CAL paired labels")
    raw = np.max(maxima, axis=0)
    return dict(schema=SCHEMA, calibration_complete=True, absolute_calibration=absolute,
                performance=dict(lower=max(RESIDUAL_FLOOR, float(raw[0])), upper=max(RESIDUAL_FLOOR, float(raw[1])),
                                 raw_lower=float(raw[0]), raw_upper=float(raw[1]), floor=RESIDUAL_FLOOR,
                                 floor_provenance="NEW conservative residual floor; original material benefit threshold remains .05",
                                 target="per-reference actual paired delta outside shared-state envelope",
                                 cal_rows=records,
                                 supported_starts=len(maxima),
                                 supported_geometries=len({r["seed"] for r in records if r["supported"]})),
                qualification="Empirical development calibration; neither all-scenario truth nor a safety guarantee.")


def expanded_interval(reference_delta, *, lower, upper):
    """Expand once, leaving the identical baseline comparison exactly zero."""
    delta = np.asarray(reference_delta, float)
    if delta.ndim != 3 or delta.shape[0] < 2 or not all(delta.shape):
        raise ValueError("paired delta shape must be [candidate,state,reference]")
    if not np.isfinite([lower, upper]).all() or min(lower, upper) < RESIDUAL_FLOOR:
        raise ValueError("finite residuals at least the newly declared .05 residual floor required")
    support = np.isfinite(delta).all(axis=(1, 2))
    interval = np.full((len(delta), 2), np.nan)
    interval[support, 0] = delta[support].min(axis=(1, 2))
    interval[support, 1] = delta[support].max(axis=(1, 2))
    interval[1:] += [-lower, upper]
    return interval


def compare_candidates(scene, hypotheses, actions, *, observer_valid, calibration):
    if calibration.get("schema") != SCHEMA or calibration.get("calibration_complete") is not True:
        raise ValueError("fitted paired-envelope calibration required")
    absolute, residual = calibration["absolute_calibration"], calibration["performance"]
    # Do not add the old every-scenario residual too. Safety arrays and all cost
    # components are exactly the frozen calculation, with performance expansion0.
    result = interval_comparison.compare_candidates(
        scene, hypotheses, actions, observer_valid=observer_valid,
        position_residual=absolute["position_residual"], yaw_residual=absolute["yaw_residual"],
        paired_cost_residual=0.)
    interval = expanded_interval(result["reference_delta"], lower=residual["lower"], upper=residual["upper"])
    support = result["cost_supported"] & np.isfinite(interval).all(axis=1)
    interval[~support] = np.nan
    sign = np.zeros(len(actions), dtype=np.int8)
    sign[support & (interval[:, 0] > COST_FLOOR)] = 1
    sign[support & (interval[:, 1] < -COST_FLOOR)] = -1
    reasons = []
    for i, old_reasons in enumerate(result["abstain_reasons"]):
        flags = [r for r in old_reasons if r != "empirical_order_uncertain"]
        if sign[i] == 0:
            flags.append("empirical_order_uncertain")
        reasons.append(tuple(flags))
    result.update(delta_interval=interval, robust_sign=sign,
                  abstain=~support | result["veto"].any(axis=1) | (sign == 0),
                  abstain_reasons=tuple(reasons),
                  performance_residual=dict(lower=residual["lower"], upper=residual["upper"], target=residual["target"]))
    return result
