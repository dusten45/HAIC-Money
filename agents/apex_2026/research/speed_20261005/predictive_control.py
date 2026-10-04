"""NumPy-only supplied-state joint control research; no simulator imports.

The caller supplies camera-derived state scenarios, a pure four-tire predictor
and frozen camera geometry. This helper neither observes privileged state nor
certifies the supplied uncertainty scenarios. Its finite emergency stop tail
is a uniform-asphalt model assumption, not a vehicle safety certificate.
"""
import numpy as np


class FixedControlPlanner:
    DT, STEPS, EXECUTED_STEPS, BACKUP_STEPS = .02, 16, 4, 32
    STEER_OFFSETS = np.asarray([-.24, -.12, -.06, -.03, 0., .03, .06, .12, .24])
    PEDALS = ((0., 0.), (.16, 0.), (.3, 0.), (.6, 0.), (1., 0.),
              (0., .25), (0., .55), (0., 1.))
    ARRAY_SHAPES = {'velocity': (2,), 'local_center': (2,), 'wheel_offsets': (4, 2),
                    'joint': (4,), 'omega': (4,), 'gas': (4,)}
    SCALARS = ('angle', 'yaw', 'mass', 'inertia')

    def __init__(self, predict_step, geometry):
        self.predict_step = predict_step
        self.geometry = geometry
        self.model_steps = 0

    @classmethod
    def _copy_state(cls, state):
        expected = set(cls.ARRAY_SHAPES) | set(cls.SCALARS) | {'wheel_velocity_override'}
        if set(state) != expected or state['wheel_velocity_override'] is not None:
            raise ValueError('expected supplied camera model state without wheel velocity override')
        result = {'wheel_velocity_override': None}
        for name in cls.SCALARS:
            result[name] = float(state[name])
            if not np.isfinite(result[name]):
                raise ValueError('nonfinite model scalar')
        if result['mass'] <= 0. or result['inertia'] <= 0.:
            raise ValueError('invalid mechanical constants')
        for name, shape in cls.ARRAY_SHAPES.items():
            value = np.asarray(state[name], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid model array')
            result[name] = value.copy()
        if np.any(abs(result['joint']) > .400001) or np.any(result['gas'] < 0.) or np.any(result['gas'] > 1.):
            raise ValueError('model joint or gas outside mechanical bounds')
        return result

    @staticmethod
    def _rotate(vector, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.asarray([c*vector[0]-s*vector[1], s*vector[0]+c*vector[1]])

    def candidate_actions(self, previous_action):
        previous = np.asarray(previous_action, dtype=float)
        if previous.shape != (3,) or not np.isfinite(previous).all():
            raise ValueError('expected finite previous legal action')
        if np.any(previous < [-1., 0., 0.]) or np.any(previous > 1.):
            raise ValueError('previous action outside legal bounds')
        commands = np.unique(np.clip(previous[0]+self.STEER_OFFSETS, -.4, .4))
        commands = commands[abs(commands-previous[0]) <= .2400001]
        return np.asarray([[steer, gas, brake] for steer in commands
                           for gas, brake in self.PEDALS], dtype=np.float32).reshape(-1, 3)

    def _advance(self, state, cm, action):
        self.model_steps += 1
        # The official legal steer sign is opposite the physical joint.
        state, diagnostics = self.predict_step(state, [-float(action[0]), float(action[1]), float(action[2])])
        state = self._copy_state(state)
        cm = cm+self.DT*state['velocity']
        hull = cm-self._rotate(state['local_center'], state['angle'])
        forces = np.asarray(diagnostics['tire_force_local_N'], dtype=float)
        ratios = np.asarray(diagnostics['unsaturated_tire_demand_ratio'], dtype=float)
        forward = np.asarray(diagnostics['wheel_forward_mps'], dtype=float)
        if (forces.shape != (4, 2) or ratios.shape != (4,) or forward.shape != (4,)
                or not np.isfinite(forces).all() or not np.isfinite(ratios).all()
                or not np.isfinite(forward).all()
                or np.max(np.linalg.norm(forces, axis=1)) > 400.00001):
            raise ValueError('invalid bounded four-tire diagnostics')
        return state, cm, hull, diagnostics

    def rollout(self, supplied_state, action):
        state = self._copy_state(supplied_state)
        action = np.asarray(action, dtype=np.float32)
        if (action.shape != (3,) or not np.isfinite(action).all()
                or abs(action[0]) > .400001 or np.any(action[1:] < 0.)
                or np.any(action[1:] > 1.) or action[1]*action[2] != 0.):
            raise ValueError('invalid separate-pedal constant action')
        cm = self._rotate(state['local_center'], state['angle'])
        states, centers, positions = [state], [cm.copy()], [np.zeros(2)]
        angles, velocities, diagnostics = [state['angle']], [state['velocity'].copy()], []
        for _ in range(self.STEPS):
            state, cm, hull, row = self._advance(state, cm, action)
            states.append(state)
            centers.append(cm.copy())
            positions.append(hull)
            angles.append(state['angle'])
            velocities.append(state['velocity'].copy())
            diagnostics.append(row)
        return {'states': states, 'cm_positions': np.asarray(centers),
                'positions': np.asarray(positions), 'angles': np.asarray(angles),
                'velocities': np.asarray(velocities), 'diagnostics': diagnostics}

    def _geometry(self, positions, angles, velocities):
        result = self.geometry.evaluate(positions, angles, velocities, self.DT)
        count = len(positions)
        progress = np.asarray(result['progress_m'], dtype=float)
        road = np.asarray(result['road_slack_m'], dtype=float)
        obstacle = np.asarray(result['obstacle_slack_m'], dtype=float)
        front = np.asarray(result['reference_front_slack_m'], dtype=float)
        if (any(value.shape != (count,) for value in (progress, road, obstacle, front))
                or not np.isfinite(progress).all() or not np.isfinite(road).all()
                or not np.isfinite(front).all() or np.isnan(obstacle).any()):
            raise ValueError('invalid camera geometry result')
        violation = float(np.max(np.maximum(0., -np.minimum(np.minimum(road, obstacle), front))))
        return progress, violation

    @staticmethod
    def _stopped(state):
        return float(np.linalg.norm(state['velocity'])) <= .5 and abs(state['yaw']) <= .1

    def _backup(self, rollout, action):
        index = self.EXECUTED_STEPS
        state = self._copy_state(rollout['states'][index])
        cm = rollout['cm_positions'][index].copy()
        positions = list(rollout['positions'][:index+1].copy())
        angles = list(rollout['angles'][:index+1].copy())
        velocities = list(rollout['velocities'][:index+1].copy())
        # Hold the actual joint rather than extending the candidate target.
        # Limit the backup command change too; the model still applies the
        # actual 3rad/s motor and per-raw-step locking brake dynamics.
        steer = float(np.clip(-np.mean(state['joint'][:2]), -.4, .4))
        steer = float(np.clip(steer, float(action[0])-.24, float(action[0])+.24))
        # 1.0 is exact in float32. float32(.9) becomes .899999976 when
        # converted to Python float, missing the predictor's >=.9 lock.
        command = np.asarray([steer, 0., 1.], dtype=np.float32)
        ticks = 0
        while not self._stopped(state) and ticks < self.BACKUP_STEPS:
            state, cm, hull, _ = self._advance(state, cm, command)
            positions.append(hull)
            angles.append(state['angle'])
            velocities.append(state['velocity'].copy())
            ticks += 1
        progress, violation = self._geometry(np.asarray(positions), np.asarray(angles), np.asarray(velocities))
        return {'stopped': self._stopped(state), 'violation': violation,
                'progress_m': float(progress[-1]), 'duration_s': ticks*self.DT}

    def plan(self, supplied_states, previous_action):
        """Return a finite feasible first action, or None for caller fallback.

        Hard constraints hold for every supplied scenario; they do not cover
        unprovided observation errors. Emergency feasibility starts after
        the only executed .08s block, not after the unexecuted .32s horizon.
        """
        self.model_steps = 0
        try:
            states = [self._copy_state(state) for state in supplied_states]
            if not 1 <= len(states) <= 7:
                raise ValueError('expected one to seven supplied camera state scenarios')
            actions = self.candidate_actions(previous_action)
            best, feasible_count, worst_violation = None, 0, 0.
            for action in actions:
                summaries = []
                for state in states:
                    rollout = self.rollout(state, action)
                    progress, violation = self._geometry(rollout['positions'], rollout['angles'], rollout['velocities'])
                    worst_violation = max(worst_violation, violation)
                    if violation > 0.:
                        break
                    backup = self._backup(rollout, action)
                    worst_violation = max(worst_violation, backup['violation'])
                    if not backup['stopped'] or backup['violation'] > 0.:
                        break
                    ratio_cost, slip_cost = [], []
                    for predicted, row in zip(rollout['states'][1:], rollout['diagnostics']):
                        ratio = np.asarray(row['unsaturated_tire_demand_ratio'])
                        slip = .54*predicted['omega'][2:]-np.asarray(row['wheel_forward_mps'])[2:]
                        ratio_cost.append(float(np.mean(np.maximum(0., ratio-1.)**2)))
                        slip_cost.append(float(np.mean(slip**2)))
                    summaries.append({'progress': float(progress[-1]),
                                      'model_cost': .10*float(np.mean(ratio_cost))+.002*float(np.mean(slip_cost)),
                                      'terminal_speed': float(np.linalg.norm(rollout['velocities'][-1])),
                                      'backup_progress': backup['progress_m'], 'backup_duration': backup['duration_s']})
                if len(summaries) != len(states):
                    continue
                feasible_count += 1
                progress = min(row['progress'] for row in summaries)
                # Meter-scale progress dominates. Tire oversupply and abrupt
                # command changes are soft costs, without a center/pursuit
                # steering proposal or a nominal reference-speed ceiling.
                score = -progress+max(row['model_cost'] for row in summaries)
                score += .5*(float(action[0])-float(previous_action[0]))**2+.02*float(action[2])
                if best is None or score < best['score']:
                    best = {'feasible': True, 'action': action.copy(), 'reason': 'feasible',
                            'score': float(score), 'worst_progress_m': progress,
                            'predicted_terminal_speed_mps': min(row['terminal_speed'] for row in summaries),
                            'backup_start_s': self.EXECUTED_STEPS*self.DT,
                            'backup_end_progress_m': max(row['backup_progress'] for row in summaries),
                            'backup_duration_s': max(row['backup_duration'] for row in summaries)}
            if best is None:
                best = {'feasible': False, 'action': None, 'reason': 'no_feasible_candidate'}
            best.update(candidate_count=len(actions), feasible_count=feasible_count,
                        scenario_count=len(states), model_steps=self.model_steps,
                        rejected_max_violation_m=worst_violation)
            return best
        except Exception as error:
            return {'feasible': False, 'action': None, 'reason': 'planning_error',
                    'error': type(error).__name__+': '+str(error)[:160], 'model_steps': self.model_steps}
