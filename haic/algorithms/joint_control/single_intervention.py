"""Fixed B-branch: reproduce C34 once, then query the champion on new observations."""

from numbers import Integral
from time import perf_counter, process_time

import numpy as np

from .envelope_successor import EnvelopeSuccessor


INTERVENTION_STEP = 34
BRANCH_MODE = "single_intervention_then_champion"


class SingleInterventionSuccessor(EnvelopeSuccessor):
    """Use the unchanged envelope decision only at 1-based decision 34.

    The runner binds ``expected_action`` to the recorded C34 float32 action and
    checks A/C prefix provenance. This class never replays future observations or
    actions. Inherited code performs the sole query, comparison, feedback and
    observer commit; its forty-action budget is not edited.

    A mismatch raises BEFORE returning an action for Env.step, including an
    unexpected scheduled abstention. An inherited proposal may already have
    advanced internal state when a check fails; that state is retained for
    diagnosis and further act calls fail until reset. No rollback or retry is
    performed here. ``one_shot_executed`` means the verified action was returned,
    not an assertion that the caller subsequently stepped the environment.
    """

    def __init__(self, champion, calibration, *, expected_action, intervention_step=INTERVENTION_STEP):
        if (not isinstance(intervention_step, Integral) or isinstance(intervention_step, bool)
                or intervention_step != INTERVENTION_STEP):
            raise ValueError("the only permitted intervention step is integer 34")
        expected = np.asarray(expected_action, dtype=float)
        if (expected.shape != (3,) or not np.isfinite(expected).all() or abs(expected[0]) > 1
                or np.any((expected[1:] < 0) | (expected[1:] > 1))):
            raise ValueError("expected_action must be a finite legal three-component action")
        self.expected_action = expected.astype(np.float32)
        self.expected_action.setflags(write=False)
        self.intervention_step = INTERVENTION_STEP
        super().__init__(champion, calibration, never_intervene=True)

    def _clear(self):
        super()._clear()
        self.steps = 0
        self.one_shot_executed = False
        self._one_shot_failed = False
        self.never_intervene = True

    def act(self, observation):
        cpu_start, wall_start = process_time(), perf_counter()
        step = self.steps + 1
        scheduled = step == self.intervention_step
        returned = False
        checks = {}
        self.last_diagnostics = dict(eligible=False, intervention=False, reasons=[], comparison=None,
                                     forecast=None, action_history_check=None)
        try:
            if self._one_shot_failed:
                raise RuntimeError("single-intervention failure is latched; reset required")
            checks["nominal_prefix_steps_match"] = bool(
                isinstance(self.nominal.steps, Integral) and not isinstance(self.nominal.steps, bool)
                and self.nominal.steps == self.steps)
            checks["prior_intervention_count_matches"] = self.intervention_count == int(self.one_shot_executed)
            checks["scheduled_state_matches"] = self.one_shot_executed == (step > self.intervention_step)
            if not all(checks.values()):
                raise RuntimeError("single-intervention pre-action index/state mismatch")
            self.never_intervene = not scheduled
            action = super().act(observation)
            info = self.last_diagnostics
            checks["nominal_steps_match"] = bool(self.nominal.steps == step)
            checks["observer_index_matches"] = info.get("decision_index") == step - 1
            checks["intervention_flag_matches"] = info.get("intervention") is scheduled
            checks["intervention_count_matches"] = self.intervention_count == int(step >= self.intervention_step)
            checks["float32_action"] = action.shape == (3,) and action.dtype == np.float32
            required_action = (self.expected_action if scheduled
                               else np.asarray(info["proposal_action"], dtype=np.float32))
            checks["required_action_matches"] = action.tobytes() == required_action.tobytes()
            required_feedback = ["observer_pending_matches", "nominal_brake_history_matches"]
            if scheduled:
                required_feedback += ["shield_wheel_matches", "shield_boxes_match"]
            history = info.get("action_history_check") or {}
            checks["actual_feedback_matches"] = all(history.get(key) is True for key in required_feedback)
            if not all(checks.values()):
                raise RuntimeError("scheduled C34 intervention not reproduced" if scheduled
                                   else "single-intervention champion continuation mismatch")
            self.steps = step
            self.one_shot_executed = step >= self.intervention_step
            returned = True
            return action
        except Exception as error:
            self._one_shot_failed = True
            self.last_diagnostics["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            self.never_intervene = True
            self.last_diagnostics.update(
                branch_mode=BRANCH_MODE, planned_step=self.intervention_step, decision_step=step,
                one_shot_scheduled=scheduled, one_shot_executed=self.one_shot_executed,
                expected_action=self.expected_action.tolist(), one_shot_checks=checks,
                action_returned=returned, one_shot_failed=self._one_shot_failed)
            # Includes inherited work, verification and branch metadata. The
            # runner's external whole-call timing remains authoritative.
            self.last_diagnostics["act_cpu_seconds"] = process_time() - cpu_start
            self.last_diagnostics["act_wall_seconds"] = perf_counter() - wall_start
