"""Bounded TRAIN-only successor around a supplied, immutable frozen champion.

The caller loads submissions/20261002-crossing-projection-collision-shield-v1-
baseline/source/agent.py and verifies its provenance. No source is edited here.
The graph must be Agent.driver=shield, shield.driver=crossing nominal. This is
not champion adoption, a tuned controller, or a shield-derived safety guarantee.
"""

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from time import perf_counter, process_time
from types import FunctionType, MethodType, ModuleType
from typing import Any

import numpy as np

from .interval_comparison import compare_candidates, extract_scene
from .observer import TemporalObserver


CALIBRATION_SCHEMA = "haic-joint-temporal-interval-calibration-v1"
MAX_INTERVENTIONS = 40


def _jsonable(value: Any) -> Any:
    """Unknown/nonfinite diagnostic values are null, never fictitious zeroes."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class TemporalSuccessor:
    """One champion query and one actual-action commit per .08s endpoint.

    Only baseline plus ONE joint candidate is considered: steer towards zero by
    .04 (exact zero goes +.04), gas +.05, brake -.05, with official clipping.
    Prediction uses that first hold plus three common baseline holds, H4/.32s,
    the same nineteen states and reference stresses, and original physics1/1/1.
    Forty interventions is an episode action budget, never an episode truncation.

    ``calibration`` is the analyzer's JSON mapping, not a filename. Missing or
    unsupported calibration abstains. Mapping residuals widen detached snapshots;
    observer state, signs, validity and candidate-independent histories stay intact.
    ``reset(obs)`` consumes obs once; the next act uses that endpoint, not observe.
    ``reset()`` clears the champion and initializes the observer at the next act.
    Construction does not reset the supplied champion. Calls must be serial and
    the caller must give this wrapper exclusive ownership of that instance.
    """

    def __init__(self, champion, calibration, *, never_intervene=False):
        self.champion = champion
        self.shield = champion.driver
        self.nominal = self.shield.driver
        if getattr(self.nominal, "contact_mode", None) != "crossing_projection":
            raise ValueError("requires the frozen crossing-projection champion graph")
        original = self.shield.act.__func__
        if not all(name in original.__globals__ for name in ("advance_boxes", "project_paths")):
            raise ValueError("requires the frozen collision shield proposal implementation")
        # Clone only the function's namespace, not its code or any champion state.
        # This hook cannot intercept another agent using the original module.
        self._shield_module = ModuleType("_temporal_successor_shield_proposal")
        self._shield_module.__dict__.update(original.__globals__)
        function = FunctionType(original.__code__, self._shield_module.__dict__,
                                original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        self._proposal = MethodType(function, self.shield)
        self.calibration = deepcopy(calibration) if isinstance(calibration, Mapping) else {}
        self.never_intervene = bool(never_intervene)
        self._clear()

    def _clear(self):
        self.observer = TemporalObserver(decision_dt=.08)
        self.frames = deque(maxlen=4)
        self.snapshots = deque(maxlen=4)
        self.issued_actions = deque(maxlen=4)
        self.intervention_count = 0
        self.last_diagnostics: dict[str, Any] = {}
        self._started = False
        self._pending = False

    def reset(self, observation=None):
        self.champion.reset(observation)
        self._clear()
        if observation is not None:
            self._record(observation, self.observer.reset(observation))

    def _record(self, observation, snapshot):
        image = np.asarray(observation)
        self.frames.append((image[-1] if image.ndim == 3 else image).copy())
        self.snapshots.append(deepcopy(snapshot))
        self._started = True

    def _calibration(self) -> tuple[dict[str, Any] | None, str | None]:
        c = self.calibration
        if c.get("schema") != CALIBRATION_SCHEMA or c.get("calibration_complete") is not True:
            return None, "calibration_missing_or_incomplete"
        try:
            position = np.asarray(c["position_residual"], dtype=float)
            yaw = np.asarray(c["yaw_residual"], dtype=float)
            paired = float(c["paired_cost_residual"])
            if (position.shape != (17,) or yaw.shape != (17,)
                    or not np.isfinite(position).all() or not np.isfinite(yaw).all()
                    or position[0] != 0 or yaw[0] != 0
                    or np.any(position[1:] < .5) or np.any(yaw[1:] < .05)
                    or not np.isfinite(paired) or paired < 0):
                raise ValueError("invalid empirical residual envelope")
            if not isinstance(c["mapping"]["by_provenance"], Mapping):
                raise ValueError("missing mapping calibration")
        except (KeyError, TypeError, ValueError):
            return None, "calibration_invalid"
        return dict(position_residual=position.copy(), yaw_residual=yaw.copy(),
                    paired_cost_residual=paired), None

    def _motions(self):
        calibrated, latest_reasons = [], []
        support = self.calibration.get("mapping", {}).get("by_provenance", {})
        for snapshot in list(self.snapshots)[1:]:
            motion = deepcopy(snapshot["mapping_motion"])
            reasons = []
            provenance = motion.get("provenance", "unsupported")
            bound = support.get(provenance, {})
            if not motion.get("valid", False):
                reasons.append("mapping_invalid")
            try:
                if not isinstance(bound, Mapping):
                    raise ValueError("missing mapping calibration")
                residual = np.array([bound["position_residual"], bound["yaw_residual"]], dtype=float)
                existing = np.array([motion["position_uncertainty"], motion["yaw_uncertainty"]], dtype=float)
                if (bound.get("supported") is not True or bound.get("n_cal", 0) <= 0
                        or residual.shape != (2,) or existing.shape != (2,)
                        or not np.isfinite(residual).all() or np.any(residual < 0)
                        or not np.isfinite(existing).all() or np.any(existing < 0)):
                    raise ValueError("unsupported mapping provenance")
                motion["position_uncertainty"], motion["yaw_uncertainty"] = np.maximum(existing, residual)
            except (KeyError, TypeError, ValueError):
                motion["valid"] = False
                reasons.append("mapping_calibration_unsupported:" + str(provenance))
            motion["calibration_reasons"] = reasons
            calibrated.append(motion)
            latest_reasons = reasons
        # A bad older link stops mapping past that link, not a newer supported
        # suffix. extract_scene enforces that causal break without filling pixels.
        if not calibrated:
            latest_reasons = ["mapping_history_missing"]
        return calibrated, list(dict.fromkeys(latest_reasons))

    def _query(self, observation):
        shield, module = self.shield, self._shield_module
        captured = []
        original_advance = module.advance_boxes
        had_act = "act" in vars(shield)
        original_act = vars(shield).get("act")

        def capture(boxes, pose):
            captured.append((boxes.copy(), pose.copy()))
            return original_advance(boxes, pose)

        try:
            module.__dict__["advance_boxes"] = capture
            shield.act = self._proposal
            proposal = self.champion.act(observation)
        finally:
            module.__dict__["advance_boxes"] = original_advance
            if had_act:
                shield.act = original_act
            else:
                del shield.act
        proposal = np.asarray(proposal)
        if (proposal.shape != (3,) or proposal.dtype != np.float32
                or not np.isfinite(proposal).all() or abs(proposal[0]) > 1
                or np.any((proposal[1:] < 0) | (proposal[1:] > 1))):
            raise ValueError("frozen champion must issue a valid float32 action")
        return proposal.copy(), captured

    def act(self, observation):
        cpu_start, wall_start = process_time(), perf_counter()
        self.last_diagnostics = dict(eligible=False, intervention=False, reasons=[], comparison=None,
                                     forecast=None, action_history_check=None)
        try:
            if not self._started:
                self._record(observation, self.observer.reset(observation))
            elif self._pending:
                self._record(observation, self.observer.observe(observation))
            self._pending = False
            snapshot = self.snapshots[-1]
            pre_wheel = float(self.shield.wheel_angle)
            pre_recovery = any(getattr(self.nominal, key, 0) > 0
                               for key in ("impact_left", "recovery_left", "contact_left"))
            proposal, captured = self._query(observation)
            info = self.shield.last_step_diagnostics()
            shield_info = self.shield.last_shield
            reasons = []
            if self.never_intervene:
                reasons.append("never_intervene")
            if self.intervention_count >= MAX_INTERVENTIONS:
                reasons.append("intervention_budget")
            if not snapshot["valid"]:
                reasons.append("observer_invalid")
            if not float(snapshot["state"].forward_speed) > 5:
                reasons.append("not_supported_forward_motion")
            speed = float(info.get("pixel_speed", np.nan))
            if not 40 <= speed <= 80:
                reasons.append("hud_speed_outside_pilot_envelope")
            steer, gas, brake = (float(value) for value in proposal)
            lower = np.array([-.35, 0., .02], dtype=np.float32)
            upper = np.array([.35, .1, .5], dtype=np.float32)
            if not (np.all(proposal >= lower) and np.all(proposal <= upper)):
                reasons.append("action_outside_pilot_envelope")
            if (shield_info.get("active", True) or shield_info.get("baseline_threat", True)
                    or shield_info.get("rearm_blocked", True) or self.shield.encounter_open
                    or np.asarray(self.shield.box_threatened).any()):
                reasons.append("shield_active_threat_or_blocked")
            if shield_info.get("reason") not in ("baseline_clear", "no_visible_obstacle"):
                reasons.append("shield_projection_unsupported")
            if (pre_recovery or any(info.get(key, False) for key in (
                    "recovery_changed", "recovery_steps_remaining", "impact_proxy_trigger",
                    "impact_steps_remaining", "contact_proxy", "contact_remaining", "contact_active"))):
                reasons.append("nominal_recovery_or_contact")
            if (len(captured) != 1 or not self.nominal.brake_history
                    or not np.isfinite(pre_wheel)
                    or shield_info.get("projection_speed") is None):
                reasons.append("feedback_capture_unavailable")
            residuals, calibration_reason = self._calibration()
            if calibration_reason:
                reasons.append(calibration_reason)
                motions = []
            else:
                motions, mapping_reasons = self._motions()
                reasons.extend(mapping_reasons)
            eligible = not reasons
            issued, selected, comparison = proposal.copy(), 0, None
            if eligible and residuals is not None:
                candidate = proposal.copy()
                candidate[0] = np.sign(steer) * max(0., abs(steer) - .04) if steer else .04
                candidate[1:] = [min(1., gas + .05), max(0., brake - .05)]
                actions = np.stack((proposal, candidate))
                try:
                    scene = extract_scene(np.stack(self.frames), motions=motions)
                    comparison = compare_candidates(scene, self.observer.shared_hypotheses(), actions,
                                                    observer_valid=True, **residuals)
                    interval = np.asarray(comparison["delta_interval"], dtype=float)
                    full_cost = (bool(comparison["common_support"])
                                 and np.asarray(comparison["cost_supported"]).all())
                    absolute = np.asarray(comparison["absolute_supported"])[1].all()
                    margin_veto = np.asarray(comparison["veto"])[1].any()
                    beneficial = (interval.shape == (2, 2) and np.isfinite(interval).all()
                                  and interval[1, 1] < -.05)
                    if full_cost and absolute and not margin_veto and beneficial:
                        issued, selected = candidate, 1
                    else:
                        reasons.extend(comparison["abstain_reasons"][1])
                        if not reasons:
                            reasons.append("no_robust_supported_improvement")
                except (ValueError, TypeError, KeyError, FloatingPointError) as error:
                    reasons.append("comparison_error")
                    self.last_diagnostics["comparison_error"] = str(error)
                    comparison = None
            intervention = selected == 1
            wheels = boxes = None
            if intervention:
                # Shield coordinates are action-positive RIGHT, not observer CCW.
                # Recompute from the merged UNADVANCED boxes, never propagate twice.
                paths, wheels, hold = self._shield_module.project_paths(
                    [float(issued[0])], float(issued[0]), float(shield_info["projection_speed"]),
                    pre_wheel, actions=1)
                boxes = self._shield_module.advance_boxes(captured[0][0], paths[0, hold])
                self.nominal.brake_history[-1] = float(issued[2])
                self.shield.wheel_angle = float(wheels[0])
                self.shield.boxes = boxes
                self.intervention_count += 1
            self.observer.commit_action(issued)
            self._pending = True
            self.issued_actions.append(issued.copy())
            history = np.asarray(self.nominal.brake_history[-len(self.issued_actions):])
            actual = np.asarray(self.issued_actions)[:, 2]
            history_check = dict(observer_pending_matches=bool(np.array_equal(np.asarray(self.observer._pending), issued)),
                                 nominal_brake_history_matches=bool(np.array_equal(history, actual)),
                                 issued_action=issued.copy(), nominal_brake_history=history.copy())
            if intervention and wheels is not None and boxes is not None:
                history_check.update(shield_wheel_matches=self.shield.wheel_angle == float(wheels[0]),
                                     shield_boxes_match=bool(np.array_equal(self.shield.boxes, boxes)))
            forecast = None
            if comparison is not None and residuals is not None:
                forecast = dict(poses=comparison["poses"][selected],
                                actions=np.stack((issued, proposal, proposal, proposal)),
                                position_residual=residuals["position_residual"],
                                yaw_residual=residuals["yaw_residual"])
            self.last_diagnostics.update(
                decision_index=snapshot["decision_index"], eligible=eligible, intervention=intervention,
                reasons=list(dict.fromkeys(reasons)), proposal_action=proposal, issued_action=issued,
                intervention_count=self.intervention_count, observer_valid=bool(snapshot["valid"]),
                observer_invalid_reasons=snapshot["invalid_reasons"], observer_state=snapshot["state"],
                mapping_provenance=[m.get("provenance") for m in motions], mapping_snapshots=motions,
                comparison=comparison, forecast=forecast, action_history_check=history_check,
                champion_proposal_diagnostics=info, pre_shield_wheel=pre_wheel,
                executed_shield_wheel=self.shield.wheel_angle,
                horizon_seconds=.32, hold_seconds=.08, candidate_count=2 if eligible else 0)
            self.last_diagnostics = _jsonable(self.last_diagnostics)
            if not all(history_check[key] for key in (
                    "observer_pending_matches", "nominal_brake_history_matches")):
                raise RuntimeError("actual issued-action history mismatch")
            return issued
        except Exception as error:
            self.last_diagnostics = _jsonable(dict(self.last_diagnostics, error=type(error).__name__ + ": " + str(error)))
            raise
        finally:
            # Includes proposal, observer, geometry, scorer, feedback, commit AND
            # diagnostic materialization; the runner also times the full call.
            self.last_diagnostics["act_cpu_seconds"] = process_time() - cpu_start
            self.last_diagnostics["act_wall_seconds"] = perf_counter() - wall_start

    def last_step_diagnostics(self):
        return deepcopy(self.last_diagnostics)
