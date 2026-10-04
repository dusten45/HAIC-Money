"""Offline AST assembly of a standalone camera predictive research agent.

No simulator/world execution. Frozen public model, camera observer, geometry
and supplied-state planner are extracted without their diagnostic imports.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

SOURCES = {
    'agents/apex_2026/fast_rear_clear_agent.py': '093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc',
    'agents/apex_2026/research/speed_20261005/physics_four_tire_model.py': '84a91143f8d42db7cf1007586c5c13038c6e0e6d371624f262abcaba26e46064',
    'agents/apex_2026/research/speed_20261005/physics_hud_joint_body.py': 'df8ee78e05d702a39395c4076129c551074d96fdd842cc08db7d40e2aa41edd1',
    'agents/apex_2026/research/speed_20261005/physics_camera_observer.py': '80d079c5f46e064e1d1c5af2f3844911d8d3c7dfab125dd40db84bcc214dd008',
    'agents/apex_2026/research/speed_20261005/camera_mpc_reference.py': '565d398e4e55c1e8bb5dad5d38e6ffa2c1a42eb5a8675832c2103f1197741920',
    'agents/apex_2026/research/speed_20261005/predictive_control.py': '341c61575651774cd5b582c644e47da021fc27737f43cb606b1a57fa561caf99',
}

ADAPTER = r'''
def _fixed_camera_calibration():
    return {
        'body_joint': {
            'body_slope': .08573317526988666,
            'body_intercept': .3065159749445597,
            'joint_positive': [53.32976668292053, .3237707931261848],
            'joint_negative': [53.33307758282946, .11603323404605852],
        },
        'rear_intercept': np.array([.020195027723669583, .00796898243261536]),
        'rear_inverse': np.array([[282.49406468720383, -6.042857175259085e-14],
                                 [2.3915441163734697e-13, 309.2412452796282]]),
        'yaw_slope': 2.1639217019081123,
        'yaw_intercept': .151470661163329,
    }


class _HullCameraGeometry(CameraGeometry):
    def evaluate(self, positions, angles, velocities, dt=.02):
        positions = np.asarray(positions, float)
        hull_velocity = np.asarray(velocities, float).copy()
        if len(positions) > 1:
            # Projection positions are hull origins, not compound centers.
            hull_velocity[1:] = np.diff(positions, axis=0)/dt
        return super().evaluate(positions, angles, hull_velocity, dt)


class _PlanningBudgetExceeded(RuntimeError):
    pass


class Agent(_BaseRearClearAgent):
    """Coupled camera/model control with a real-action fallback and stop tail.

    Three representative state scenarios are not a complete uncertainty box
    or a safety certificate. Unknown camera support retains the parent action.
    """

    def __init__(self, *args, planning_budget_s=3., **kwargs):
        if not np.isfinite(planning_budget_s) or planning_budget_s < 0.:
            raise ValueError('planning budget must be finite and nonnegative')
        self._planning_budget_s = min(float(planning_budget_s), 3.)
        self._observer = CameraObserver(_fixed_camera_calibration())
        super().__init__(*args, **kwargs)

    def reset(self, observation=None):
        super().reset(observation)
        self._observer.reset()
        self._predictive_previous_action = np.zeros(3, np.float32)
        self._predictive_deadline = 0.
        self.predictive_status = 'reset'
        self.predictive_result = {}
        self.predictive_sensor = {}
        self.predictive_latency_s = 0.
        self.predictive_model_calls = 0

    def _predictive_field(self, frame, circles):
        image = frame[:73]
        road = (image >= .32) & (image <= .51)
        x = (np.arange(84)-self.CAR_X)/self.PX_X
        y = (self.CAR_Y-np.arange(73))/self.PX_Y
        # Fill only rendered dark car pixels inside the known vehicle bounds;
        # a bright enclosed grass component is never converted to asphalt.
        car = ((abs(x)[None, :] <= 1.6) & (y[:, None] >= -2.4) &
               (y[:, None] <= 2.6) & (image < .32))
        road |= car
        for object_x, object_y in circles:
            core = ((x[None, :]-object_x)**2+(y[:, None]-object_y)**2 <= 1.2**2)
            road |= core & (image >= .655) & (image <= .705)
        # The image boundary is a visibility boundary; no exterior road is
        # inferred from an asphalt run touching it.
        road[[0, -1], :] = False
        road[:, [0, -1]] = False
        unknown = ~road
        distance = np.where(road, 1000., 0.)
        neighbors = [(dy, dx, np.hypot(dx/self.PX_X, dy/self.PX_Y))
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        for _ in range(100):
            updated = distance.copy()
            for dy, dx, weight in neighbors:
                y0, y1 = max(0, dy), min(73, 73+dy)
                x0, x1 = max(0, dx), min(84, 84+dx)
                np.minimum(updated[y0:y1, x0:x1],
                           distance[y0-dy:y1-dy, x0-dx:x1-dx]+weight,
                           out=updated[y0:y1, x0:x1])
            if np.max(abs(updated-distance)) < 1e-8:
                break
            distance = updated
        return distance, unknown

    def _predictive_scenarios(self, state):
        scenarios = [copy.deepcopy(state)]
        speed = float(np.linalg.norm(state['velocity']))
        # Two fixed correlated alternatives, not all independent sensor-error
        # combinations. Axle rolling speed covaries with body-speed magnitude.
        for sign in (-1., 1.):
            other = copy.deepcopy(state)
            other_speed = float(np.clip(speed+sign*1.6, 0., 100.))
            side = float(np.clip(state['velocity'][0]+sign*.35, -.8*other_speed, .8*other_speed))
            other['velocity'] = np.array([side, np.sqrt(max(0., other_speed**2-side**2))])
            other['yaw'] = float(state['yaw']+sign*.15)
            other['joint'][:2] = np.clip(state['joint'][:2]+sign*.006, -.4, .4)
            other['omega'] = np.maximum(0., state['omega']+(other_speed-speed)/.54)
            scenarios.append(other)
        return scenarios

    def _predictive_step(self, state, command):
        if time.perf_counter() >= self._predictive_deadline:
            raise _PlanningBudgetExceeded('camera planning time budget reached')
        self.predictive_model_calls += 1
        return predict_step(state, command)

    def _predictive_geometry(self, frame, state):
        road = self.corridor_road
        if road is None or self.lost_frames:
            return None
        circles = [(float(x), float(y)) for y, x in self._circles(frame, road)]
        # Parent act already transported the active pass once. Keep that
        # remembered actual circle even on a current detection miss.
        if (self.pass_side and self.pass_y is not None and self.pass_missing <= 4 and
                np.isfinite([self.pass_x, self.pass_y]).all()):
            remembered = np.array([self.pass_x, self.pass_y], float)
            if all(np.linalg.norm(remembered-np.asarray(circle)) > .5 for circle in circles):
                circles.append(tuple(remembered))
        field, unknown = self._predictive_field(frame, circles)
        path = self._ridge(frame)
        if path is None or len(path) < 5:
            return None
        dense = np.concatenate([np.linspace(left, right, 10) for left, right in zip(path[:-1], path[1:])])
        if np.min(self._sample_distance(field, dense)) < 1.9:
            return None
        geometry = _HullCameraGeometry(path, field, circles, unknown)
        initial = geometry.evaluate(np.zeros((1, 2)), np.zeros(1),
                                    np.asarray(state['velocity'])[None], .02)
        return geometry if not initial['violation'].any() else None

    def act(self, observation):
        started = time.perf_counter()
        self._predictive_deadline = started+self._planning_budget_s
        previous = self._predictive_previous_action.copy()
        # Exactly one parent call updates all inherited perception/pass memory.
        fallback = np.asarray(super().act(observation), np.float32)
        emitted = fallback
        self.predictive_status = 'fallback_invalid_camera'
        self.predictive_result = {}
        self.predictive_sensor = {}
        self.predictive_model_calls = 0
        frame = self._frame(observation)
        if frame is not None:
            state, sensor = self._observer.observe(frame)
            self.predictive_sensor = sensor
            if sensor['speed_mps'] < 20.:
                self.predictive_status = 'fallback_low_speed'
            elif sensor['dynamics_innovation_suspect'] or sensor['rear_bar_top_nonzero']:
                self.predictive_status = 'fallback_innovation'
            elif sensor['yaw_beyond_calibration']:
                self.predictive_status = 'fallback_yaw'
            else:
                geometry = self._predictive_geometry(frame, state)
                self.predictive_status = 'fallback_unsupported'
                if geometry is not None:
                    planner = FixedControlPlanner(self._predictive_step, geometry)
                    result = planner.plan(self._predictive_scenarios(state), previous)
                    self.predictive_result = result
                    if result.get('feasible') and result.get('action') is not None:
                        action = np.asarray(result['action'], np.float32)
                        if (action.shape == (3,) and np.isfinite(action).all() and
                                np.all(action >= [-1., 0., 0.]) and np.all(action <= 1.)):
                            emitted = action
                            self.predictive_status = 'predictive'
                    else:
                        self.predictive_status = 'fallback_timeout' if '_PlanningBudgetExceeded' in result.get('error', '') else 'fallback_no_feasible'
        emitted = np.asarray(emitted, np.float32)
        # Every actual action, including fallback, advances history exactly once.
        self._observer.advance(emitted)
        self._predictive_previous_action = emitted.copy()
        self.last_steer = float(emitted[0])
        self.predictive_latency_s = time.perf_counter()-started
        return emitted
'''


def extract(root, relative, names):
    tree = ast.parse((root/relative).read_text())
    nodes = []
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            nodes.append(node)
    assert len(nodes) == len(names), (relative, names)
    return '\n\n'.join(ast.unparse(node) for node in nodes)


def build_text(root):
    root = Path(root)
    for relative, sha in SOURCES.items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == sha, relative
    baseline = (root/'agents/apex_2026/fast_rear_clear_agent.py').read_text()
    sections = [baseline, '\n_BaseRearClearAgent = Agent\n\nimport math\nimport copy\nimport time\n\nDT = .02\n']
    folder = 'agents/apex_2026/research/speed_20261005/'
    for filename, names in (
        ('physics_four_tire_model.py', {'rotate', 'cross', 'tire_inputs', 'tire_forces', 'predict_step'}),
        ('physics_hud_joint_body.py', {'joint_measurement', 'body_measurement', 'decode'}),
        ('physics_camera_observer.py', {'PUBLIC_GEOMETRY', 'SENSOR_BOUNDS', 'CameraObserver'}),
        ('camera_mpc_reference.py', {'arclength', 'project_one', 'project_progress', 'body_points', 'sample_field',
                                     'road_slack', 'obstacle_slack', 'terminal_speed_limit', 'CameraGeometry'}),
        ('predictive_control.py', {'FixedControlPlanner'}),
    ):
        sections.append(extract(root, folder+filename, names))
    sections.append(ADAPTER.strip())
    result = '\n\n'.join(sections).rstrip()+'\n'
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    text = build_text(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
