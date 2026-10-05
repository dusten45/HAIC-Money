"""Synthetic observer tests only: no environment, Agent or new driving data."""

from dataclasses import asdict
import builtins
import unittest
from unittest.mock import patch

import numpy as np

from haic.algorithms.joint_control.observer import TemporalObserver
from haic.algorithms.joint_control.physics import predict
from tests.test_joint_control_hud import hud_frame


def measurements(speed=20., yaw=0., angle=0., omega=None, valid=None):
    return dict(speed=speed, yaw_rate=yaw, wheel_angle=angle,
                wheel_omega=np.full(4, speed / .54) if omega is None else np.asarray(omega),
                valid=np.ones(7, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
                confidence=np.ones(7))


def motion(vf=20., vl=0., yaw=0., dt=.08, valid=True):
    return dict(valid=valid, forward_speed=vf, lateral_speed=vl, yaw_rate=yaw,
                forward=vf * dt, right=vl * dt, yaw_delta=yaw * dt,
                residual_p90=0., tracks=20, inliers=20, inlier_fraction=1.)


class TemporalObserverTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((84, 84), dtype=np.uint8)
        self.hud = patch("haic.algorithms.joint_control.observer.decode_hud", return_value=measurements()).start()
        self.flow = patch("haic.algorithms.joint_control.observer.estimate_body_motion", return_value=motion()).start()
        self.addCleanup(patch.stopall)

    def assert_state_equal(self, left, right):
        for key, value in asdict(left).items():
            np.testing.assert_array_equal(value, getattr(right, key))

    def advance(self, observer, action=(0., 0., 0.), **kwargs):
        observer.commit_action(action)
        return observer.observe(self.frame, **kwargs)

    def test_startup_speed_norm_does_not_imply_forward_or_zero_lateral(self):
        observer = TemporalObserver()
        result = observer.reset(self.frame)
        self.assertFalse(result["valid"])
        self.assertEqual(result["state"].forward_speed, 0.)
        self.assertLess(result["lower"].forward_speed, -20)
        self.assertGreater(result["upper"].lateral_speed, 20)
        self.assertIsNone(result["measurement_age"]["ground_motion"])
        self.flow.assert_not_called()

    def test_commit_observe_lifecycle_rejects_double_advance_and_commit(self):
        observer = TemporalObserver()
        for call in (observer.snapshot, observer.shared_hypotheses,
                     lambda: observer.observe(self.frame), lambda: observer.commit_action([0, 0, 0])):
            with self.assertRaises(RuntimeError):
                call()
        observer.reset(self.frame)
        with self.assertRaises(RuntimeError):
            observer.observe(self.frame)
        before = observer.snapshot()
        observer.commit_action([.1, .6, .2])
        self.assert_state_equal(before["state"], observer.snapshot()["state"])
        with self.assertRaises(RuntimeError):
            observer.commit_action([0, 0, 0])
        with self.assertRaises(RuntimeError):
            observer.shared_hypotheses()
        observer.observe(self.frame)
        self.assertEqual(observer.snapshot()["decision_index"], 1)
        with self.assertRaises(RuntimeError):
            observer.observe(self.frame)

    def test_prediction_uses_copied_actual_action_and_original_parameters(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        action = np.array([.2, .7, .1])
        observer.commit_action(action)
        action[:] = 0
        with patch("haic.algorithms.joint_control.observer.predict", wraps=predict) as model:
            observer.observe(self.frame)
        np.testing.assert_array_equal(model.call_args.args[1], [[[.2, .7, .1]]])
        self.assertEqual(asdict(model.call_args.kwargs["parameters"]),
                         dict(mass_scale=1., yaw_inertia_scale=1., tire_stiffness_scale=1.))

    def test_no_hud_and_no_flow_are_exact_prediction_not_silent_zero(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        self.advance(observer)
        before = observer.snapshot()
        self.hud.return_value = measurements(speed=np.nan, valid=[False] * 7)
        self.flow.return_value = motion(valid=False)
        action = np.array([.3, .4, .1])
        predicted = predict(before["state"], action[None, None])["final_state"]
        result = self.advance(observer, action)
        for name in ("forward_speed", "lateral_speed", "yaw_rate", "wheel_angle", "gas_state"):
            self.assertAlmostEqual(getattr(result["state"], name), float(np.asarray(getattr(predicted, name))[0]))
        np.testing.assert_allclose(result["state"].wheel_omega, np.asarray(predicted.wheel_omega)[0])
        self.assertGreater(result["uncertainty"]["forward_speed"], before["uncertainty"]["forward_speed"])

    def test_signed_reversal_and_lateral_motion_are_retained(self):
        self.hud.return_value = measurements(speed=np.hypot(20, 2), omega=[-37.] * 4)
        self.flow.return_value = motion(vf=-20., vl=2.)
        observer = TemporalObserver()
        observer.reset(self.frame)
        result = self.advance(observer)
        self.assertTrue(result["motion"]["accepted_for_correction"])
        self.assertLess(result["state"].forward_speed, -15.)
        self.assertGreater(result["state"].lateral_speed, 1.)
        self.flow.return_value = motion(valid=False)
        result = self.advance(observer)
        self.assertLess(result["state"].forward_speed, 0.)

    def test_interval_flow_is_not_mistaken_for_endpoint_velocity(self):
        observer = TemporalObserver()
        self.hud.return_value = measurements(speed=0., omega=[0.] * 4)
        observer.reset(self.frame)
        state = observer.snapshot()["state"]
        action = np.array([0., 1., 0.])
        forecast = predict(state, action[None, None])
        endpoint = float(forecast["forward_speed"][0, 0])
        average = float(forecast["relative_y"][0, 0]) / .08
        self.hud.return_value = measurements(speed=endpoint, omega=forecast["wheel_omega"][0, 0])
        self.flow.return_value = motion(vf=average)
        result = self.advance(observer, action)
        self.assertNotAlmostEqual(endpoint, average, places=3)
        self.assertAlmostEqual(result["state"].forward_speed, endpoint, places=10)

    def test_short_omega_abstention_preserves_predictor_state(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        self.advance(observer)
        before = observer.snapshot()["state"]
        expected = predict(before, np.array([[[0, 0, 0]]]))["wheel_omega"][0, 0]
        self.hud.return_value = measurements(omega=[np.nan] * 4, valid=[True] * 3 + [False] * 4)
        self.flow.return_value = motion(valid=False)
        result = self.advance(observer)
        np.testing.assert_array_equal(result["state"].wheel_omega, expected)
        self.assertTrue(np.any(result["state"].wheel_omega != 0))
        self.assertTrue(np.all(result["uncertainty"]["wheel_omega"] > 12))

    def test_gas_history_depends_on_every_issued_hold(self):
        self.flow.return_value = motion(valid=False)
        observer = TemporalObserver()
        observer.reset(self.frame)
        self.assertAlmostEqual(self.advance(observer, (0, 1, 0))["state"].gas_state, .4)
        self.assertAlmostEqual(self.advance(observer, (0, 1, 0))["state"].gas_state, .8)
        self.assertAlmostEqual(self.advance(observer, (0, .2, 0))["state"].gas_state, .2)
        self.assertEqual(observer.snapshot()["uncertainty"]["gas_state"], 0.)

    def test_repeated_images_do_not_erase_action_history(self):
        self.hud.return_value = measurements(speed=np.nan, valid=[False] * 7)
        self.flow.return_value = motion(valid=False)
        a, b = TemporalObserver(), TemporalObserver()
        a.reset(self.frame)
        b.reset(self.frame)
        self.advance(a, (.2, 1, 0))
        self.advance(b, (-.2, 0, .5))
        self.assertLess(a.snapshot()["state"].wheel_angle, 0)
        self.assertGreater(b.snapshot()["state"].wheel_angle, 0)
        self.assertNotEqual(a.snapshot()["state"].forward_speed, b.snapshot()["state"].forward_speed)

    def test_motion_not_required_every_frame_but_age_eventually_invalidates(self):
        observer = TemporalObserver()
        result = observer.reset(self.frame)
        for _ in range(3):
            result = self.advance(observer)
        self.assertTrue(result["valid"], result["invalid_reasons"])
        self.flow.return_value = motion(valid=False)
        result = self.advance(observer)
        self.assertTrue(result["valid"], result["invalid_reasons"])
        self.assertAlmostEqual(result["measurement_age"]["ground_motion"], .08)
        for _ in range(11):
            result = self.advance(observer)
        self.assertFalse(result["valid"])
        self.assertIn("ground_motion_stale_or_unknown", result["invalid_reasons"])

    def test_hud_dropout_increases_uncertainty_and_invalidates(self):
        observer = TemporalObserver()
        result = observer.reset(self.frame)
        for _ in range(3):
            result = self.advance(observer)
        previous = result["uncertainty"]["yaw_rate"]
        self.hud.return_value = measurements(speed=np.nan, valid=[False] * 7)
        self.flow.return_value = motion(valid=False)
        for _ in range(4):
            result = self.advance(observer)
        self.assertFalse(result["valid"])
        self.assertIn("speed_stale", result["invalid_reasons"])
        self.assertGreater(result["uncertainty"]["yaw_rate"], previous)

    def test_innovation_shock_invalidates_even_with_valid_image_channels(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        for _ in range(3):
            self.advance(observer)
        self.hud.return_value = measurements(speed=7., yaw=2.)
        self.flow.return_value = motion(valid=False)
        result = self.advance(observer)
        self.assertFalse(result["valid"])
        self.assertIn("recent_innovation_shock", result["invalid_reasons"])
        self.assertIn("speed", result["innovations"]["shock_channels"])
        self.assertIn("yaw_rate", result["innovations"]["shock_channels"])
        self.assertGreaterEqual(result["uncertainty"]["lateral_speed"], 5.)

    def test_motion_hud_disagreement_does_not_refresh_ground_age(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        self.advance(observer)
        self.flow.return_value = motion(vf=80., yaw=3.)
        result = self.advance(observer)
        self.assertTrue(result["motion"]["valid"])
        self.assertFalse(result["motion"]["accepted_for_correction"])
        self.assertAlmostEqual(result["measurement_age"]["ground_motion"], .08)

    def test_snapshots_and_hypotheses_are_pure_detached_and_shared(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        self.advance(observer)
        before = observer.snapshot()
        first, second = observer.shared_hypotheses(), observer.shared_hypotheses()
        self.assert_state_equal(first, second)
        self.assertEqual(np.asarray(first.wheel_omega).shape, (19, 4))
        self.assertAlmostEqual(np.asarray(first.forward_speed)[1], before["lower"].forward_speed)
        self.assertAlmostEqual(np.asarray(first.forward_speed)[2], before["upper"].forward_speed)
        actions = np.tile(np.array([[[0., 0., 0.]]]), (19, 1, 1))
        predict(first, actions)
        predict(first, actions + [0, .5, 0])
        self.assert_state_equal(first, second)
        np.asarray(first.wheel_omega)[:] = -999
        before["state"].wheel_omega[:] = 999
        before["uncertainty"]["wheel_omega"][:] = 999
        self.assertLess(observer.snapshot()["state"].wheel_omega.max(), 100)
        self.assertEqual(observer.snapshot()["decision_index"], 1)

    def test_only_latest_frame_is_used_and_input_history_is_copied(self):
        stack = np.zeros((4, 84, 84), dtype=np.uint8)
        stack[-1] = 11
        observer = TemporalObserver()
        observer.reset(stack)
        stack[:] = 255
        next_stack = np.full((4, 84, 84), 77, dtype=np.uint8)
        next_stack[-1] = 22
        observer.commit_action([0, 0, 0])
        observer.observe(next_stack)
        np.testing.assert_array_equal(self.flow.call_args.args[0], 11)
        np.testing.assert_array_equal(self.flow.call_args.args[1], 22)
        np.testing.assert_array_equal(self.hud.call_args.args[0], 22)

    def test_invalid_input_does_not_consume_commit(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        for action in ([0, 2, 0], [np.nan, 0, 0], [0, 0], [2, 0, 0]):
            with self.assertRaises(ValueError):
                observer.commit_action(action)
        observer.commit_action([0, 0, 0])
        for bad in (np.zeros((83, 84)), np.full((84, 84), np.nan), np.ones((84, 84), dtype=int)):
            with self.assertRaises(ValueError):
                observer.observe(bad)
        for duration in (.03, .16, -1, np.nan):
            with self.assertRaises(ValueError):
                observer.observe(self.frame, elapsed_dt=duration)
        self.assertEqual(observer.snapshot()["decision_index"], 0)
        observer.observe(self.frame)
        self.assertEqual(observer.snapshot()["decision_index"], 1)

    def test_actual_shortened_interval_and_reset_clear_history(self):
        observer = TemporalObserver()
        observer.reset(self.frame)
        result = self.advance(observer, (0, 1, 0), elapsed_dt=.02)
        self.assertAlmostEqual(result["state"].gas_state, .1)
        self.assertEqual(result["interval_seconds"], .02)
        self.assertEqual(self.flow.call_args.args[2], .02)
        observer.commit_action([0, 1, 0])
        reset = observer.reset(self.frame)
        self.assertEqual(reset["decision_index"], 0)
        self.assertEqual(reset["state"].gas_state, 0)
        self.assertFalse(reset["awaiting_observation"])

    def test_mapping_uses_measured_transform_then_uncertain_issued_history(self):
        observer = TemporalObserver()
        result = observer.reset(self.frame)
        self.assertFalse(result["mapping_motion"]["valid"])
        for _ in range(3):
            result = self.advance(observer)
        mapping = result["mapping_motion"]
        self.assertTrue(mapping["valid"])
        self.assertEqual(mapping["provenance"], "measured_image")
        self.assertAlmostEqual(mapping["forward"], 1.6)
        self.flow.return_value = motion(valid=False)
        result = self.advance(observer)
        mapping = result["mapping_motion"]
        self.assertTrue(mapping["valid"])
        self.assertFalse(result["motion"]["valid"])
        self.assertEqual(mapping["provenance"], "predicted_corrected_issued_action")
        self.assertGreater(mapping["forward"], 1.)
        self.assertGreater(mapping["position_uncertainty"], .05)
        self.assertGreater(mapping["yaw_uncertainty"], .005)
        self.assertEqual(mapping["residual_p90"], 0.)
        self.assertEqual(mapping["model_allowance"], dict(position=.05, yaw=.005))
        mapping["valid"] = False
        self.assertTrue(observer.snapshot()["mapping_motion"]["valid"])

    def test_mapping_refuses_startup_stale_direction_and_collision_innovation(self):
        observer = TemporalObserver()
        result = observer.reset(self.frame)
        self.flow.return_value = motion(valid=False)
        self.assertFalse(self.advance(observer)["mapping_motion"]["valid"])
        self.flow.return_value = motion()
        for _ in range(4):
            result = self.advance(observer)
        self.assertTrue(result["valid"])
        self.flow.return_value = motion(valid=False)
        self.hud.return_value = measurements(speed=5.)
        result = self.advance(observer)
        self.assertFalse(result["mapping_motion"]["valid"])
        self.assertIn("recent_innovation_shock", result["invalid_reasons"])

    def test_no_future_continuation_can_change_saved_prefix(self):
        a, b = TemporalObserver(), TemporalObserver()
        a.reset(self.frame)
        b.reset(self.frame)
        prefix_a, prefix_b = [], []
        for action in ((.1, .2, 0), (-.1, .5, 0), (0, 0, .1)):
            prefix_a.append(self.advance(a, action))
            prefix_b.append(self.advance(b, action))
        self.hud.return_value = measurements(speed=60., angle=.3, yaw=1.)
        self.advance(a, (.4, 1., 0))
        self.hud.return_value = measurements(speed=0., angle=-.3, yaw=-1.)
        self.advance(b, (-.4, 0, 1.))
        for left, right in zip(prefix_a, prefix_b):
            self.assert_state_equal(left["state"], right["state"])
            self.assertEqual(left["valid"], right["valid"])
            self.assertEqual(left["mapping_motion"], right["mapping_motion"])

    def test_near_rest_is_bounded_not_exact_and_unknown_spin_is_not_zero(self):
        self.hud.return_value = measurements(speed=1., omega=[np.nan] * 4,
                                             valid=[True] * 3 + [False] * 4)
        observer = TemporalObserver()
        result = observer.reset(self.frame)
        self.assertFalse(result["valid"])
        self.assertEqual(result["measurement_age"]["ground_motion"], 0.)
        self.assertLess(result["lower"].forward_speed, -1.)
        self.assertGreater(result["upper"].lateral_speed, 1.)
        self.assertTrue((result["lower"].wheel_omega < 0).all())
        self.assertTrue((result["upper"].wheel_omega > 0).all())

    def test_runtime_has_no_file_label_environment_or_agent_dependency(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"core", "Box2D", "gym", "gymnasium", "agent", "env_wrapper"}:
                raise AssertionError("forbidden simulator/controller import: " + name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded), patch(
            "builtins.open", side_effect=AssertionError("runtime file/label access")
        ):
            observer = TemporalObserver()
            observer.reset(self.frame)
            self.advance(observer)
            self.advance(observer)
            observer.shared_hypotheses()
            observer.snapshot()

    def test_bad_duration_constructor_fails_explicitly(self):
        for value in (False, "0.08", [0.08], complex(.08), np.nan, .03, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TemporalObserver(value)  # type: ignore[arg-type]

    def test_real_hud_short_bars_and_physical_signs(self):
        patch.stopall()
        frame = hud_frame(20., -.2, -.5, (20., 20., 20., 20.))
        observer = TemporalObserver()
        result = observer.reset(frame)
        self.assertLess(result["state"].wheel_angle, 0)
        self.assertLess(result["state"].yaw_rate, 0)
        self.assertTrue((result["uncertainty"]["wheel_omega"] >= 60).all())
        self.assertTrue((result["lower"].wheel_omega < 0).all())
        self.assertTrue((result["upper"].wheel_omega > 0).all())


def evaluate_old_probe():
    """Opt-in DEVELOPMENT replay, never part of unit-test discovery.

    Only the nine already-consumed old probe arms are admitted. All three seeds
    are development, not fresh/held out. Labels are used after each observer call
    solely for errors/deduplication; no label reaches reset/observe/commit.
    Reproduce with python -B -m tests.test_joint_temporal_observer --evaluate-old-probe.
    Returns evidence to the caller/stdout without creating or changing any file.
    """
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path
    import time

    import cv2
    from haic.algorithms.joint_control.hud import HudCalibration, decode_hud
    from haic.algorithms.joint_control.validation import (
        gas_history, jsonable, motion_features, runtime_states,
    )

    cv2.setNumThreads(1)
    root = Path("runs/joint-prediction-probe-v1")
    paths = [Path(__file__), *(Path("haic/algorithms/joint_control") / name
                             for name in ("observer.py", "physics.py", "hud.py", "motion.py", "validation.py"))]
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    calibration = HudCalibration(gain=(1.0095651493835058, 1, 1, 1, 1, 1, 1),
                                 bias=(.014490804482105879, 0, 0, 0, 0, 0, 0))
    names = ("forward_speed", "lateral_speed", "yaw_rate", "wheel_angle",
             "wheel_omega_0", "wheel_omega_1", "wheel_omega_2", "wheel_omega_3", "gas_state")
    rows, all_rows, anchors, durations, seen, inputs = [], [], [], [], set(), {}

    def vector(state):
        return np.array([state.forward_speed, state.lateral_speed, state.yaw_rate,
                         state.wheel_angle, *state.wheel_omega, state.gas_state])

    for seed in (3184000002, 3184000013, 3184000015):
        for arm in ("baseline", "repeat", "alternative"):
            folder = root / f"1-{seed}-{arm}"
            receipt_path = folder / "receipt.json"
            inputs[str(receipt_path)] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            receipt = json.loads(receipt_path.read_text())
            if receipt["status"] != "COMPLETE":
                raise ValueError("old probe arm is not complete")
            for name in ("data.npz", "state.json"):
                digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
                if digest != receipt["artifacts_sha256"][name]:
                    raise ValueError("old probe input hash mismatch")
                inputs[str(folder / name)] = digest
            endpoints = json.loads((folder / "state.json").read_text())["endpoints"]
            with np.load(folder / "data.npz", allow_pickle=False) as saved:
                observations, actions, times = (saved[key] for key in ("observations", "actions", "endpoint_time"))
            if observations.shape != (15, 4, 84, 84) or actions.shape != (14, 3):
                raise ValueError("only the old anchor10/H4 probe schema is admitted")
            ticks = np.rint(np.diff(times) / .02).astype(int)
            if not np.all(ticks == 4) or not np.allclose(np.diff(times), .08, atol=1e-12, rtol=0):
                raise ValueError("old probe full hold timing mismatch")
            hud = decode_hud(observations[:, -1], calibration)
            old = runtime_states(hud, motion_features(observations[:, -1], np.diff(times)),
                                 gas_history(actions, ticks), actions, ticks)
            observer, prefix = TemporalObserver(), hashlib.sha256()
            for index, observation in enumerate(observations):
                started = time.perf_counter()
                if index == 0:
                    snapshot = observer.reset(observation)
                else:
                    observer.commit_action(actions[index - 1])
                    snapshot = observer.observe(observation, elapsed_dt=float(times[index] - times[index - 1]))
                    prefix.update(actions[index - 1].tobytes() + ticks[index - 1].tobytes())
                durations.append(1000 * (time.perf_counter() - started))
                physical = endpoints[index]["state"]
                hull, wheels = physical["hull"], physical["wheels"]
                vx, vy = hull["linearVelocity"]
                c, s = np.cos(hull["angle"]), np.sin(hull["angle"])
                truth = np.array([-s * vx + c * vy, c * vx + s * vy, hull["angularVelocity"],
                                  wheels[0]["joint"]["angle"], *[wheel["omega"] for wheel in wheels],
                                  .5 * (wheels[2]["gas"] + wheels[3]["gas"])])
                old_state = np.array([old[name][index] for name in names[:4]]
                                     + old["wheel_omega"][index].tolist() + [old["gas_state"][index]])
                key = (seed, prefix.hexdigest(), hashlib.sha256(observation.tobytes() + truth.tobytes()).hexdigest())
                row = dict(seed=seed, arm=arm, index=index, valid=snapshot["valid"],
                           invalid_reasons=snapshot["invalid_reasons"], state=vector(snapshot["state"]),
                           truth=truth, error=vector(snapshot["state"]) - truth, old_error=old_state - truth,
                           lower=vector(snapshot["lower"]), upper=vector(snapshot["upper"]),
                           measurement_age=snapshot["measurement_age"], motion=snapshot["motion"],
                           mapping_motion=snapshot["mapping_motion"], innovations=snapshot["innovations"])
                all_rows.append(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if index == 10:
                    anchors.append(row)

    def metrics(subset):
        if not subset:
            return dict(n=0, valid=0)
        error = np.abs(np.stack([row["error"] for row in subset]))
        old_error = np.abs(np.stack([row["old_error"] for row in subset]))
        coverage = np.stack([(row["truth"] >= row["lower"]) & (row["truth"] <= row["upper"]) for row in subset])
        summarize = lambda values: {name: dict(p50=float(np.median(values[:, i])),
                                              p95=float(np.quantile(values[:, i], .95)),
                                              max=float(values[:, i].max())) for i, name in enumerate(names)}
        return dict(n=len(subset), valid=sum(row["valid"] for row in subset),
                    temporal=summarize(error), previous_initializer=summarize(old_error),
                    inside_heuristic_interval=dict(zip(names, coverage.sum(axis=0).tolist())),
                    invalid_reasons=dict(Counter(reason for row in subset for reason in row["invalid_reasons"])))

    if sources != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}:
        raise ValueError("source changed during development evaluation")
    return jsonable(dict(
        scope="Consumed DEVELOPMENT old probe only; no new data, environment, Agent or fitting",
        design_trials="One point-estimator/validity design; mapping-only addition leaves state updates unchanged; no label-fitted correction",
        source_sha256=sources, input_sha256=inputs, columns=names,
        all_stored=metrics(all_rows), all_unique=metrics(rows),
        runtime_valid=metrics([row for row in rows if row["valid"]]),
        per_geometry={str(seed): metrics([row for row in rows if row["seed"] == seed])
                      for seed in (3184000002, 3184000013, 3184000015)},
        anchors=anchors, rows_unique=rows,
        call_timing_ms=dict(n=len(durations), p50=float(np.median(durations)),
                            p95=float(np.quantile(durations, .95)), max=float(max(durations))),
        caveat="Three similar early acceleration starts, not independent broad regimes; intervals are heuristic not calibrated coverage guarantees",
    ))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--evaluate-old-probe"]:
        import json
        print(json.dumps(evaluate_old_probe(), indent=2, allow_nan=False))
    else:
        unittest.main()
