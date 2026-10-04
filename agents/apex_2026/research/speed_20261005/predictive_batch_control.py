"""NumPy-only supplied-state joint control research; no simulator imports.

The caller supplies camera-derived state scenarios, a pure four-tire predictor
and frozen camera geometry. This helper neither observes privileged state nor
certifies the supplied uncertainty scenarios. Its finite emergency stop tail
is a uniform-asphalt model assumption, not a vehicle safety certificate.
"""
import numpy as np


class _ScalarPlannerReference:
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


class FixedBatchControlPlanner(_ScalarPlannerReference):
    """Batch the model while retaining the frozen scalar control policy.

    The supplied batch callback has the same physical target/gas/brake model
    contract. Geometry remains an independent pure per-trajectory callback.
    First-stop masks retain exactly the unpadded scalar emergency trajectory.
    """

    def __init__(self, predict_batch, geometry):
        super().__init__(None, geometry)
        self.predict_batch = predict_batch
        self.batch_calls = 0
        self.batched_model_rows = 0

    def _pack_many(self, states):
        result = {'wheel_velocity_override': None}
        for key in self.SCALARS:
            result[key] = np.asarray([state[key] for state in states], float)
        for key in self.ARRAY_SHAPES:
            result[key] = np.stack([state[key] for state in states])
        return result

    @staticmethod
    def _copy_packed(state):
        return {name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in state.items()}

    @staticmethod
    def _rotate_many(vector, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.stack((c*vector[:, 0]-s*vector[:, 1],
                         s*vector[:, 0]+c*vector[:, 1]), axis=1)

    def _call_batch(self, state, commands):
        count = len(commands)
        self.batch_calls += 1
        self.batched_model_rows += count
        result, diagnostics = self.predict_batch(state, commands)
        expected = set(self.SCALARS) | set(self.ARRAY_SHAPES) | {'wheel_velocity_override'}
        if set(result) != expected or result['wheel_velocity_override'] is not None:
            raise ValueError('invalid batch prediction state keys')
        for key in self.SCALARS:
            value = np.asarray(result[key], float)
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError('invalid batch scalar prediction')
        for key, shape in self.ARRAY_SHAPES.items():
            value = np.asarray(result[key], float)
            if value.shape != (count, *shape) or not np.isfinite(value).all():
                raise ValueError('invalid batch array prediction')
        if (np.any(result['mass'] <= 0.) or np.any(result['inertia'] <= 0.)
                or np.any(abs(result['joint']) > .400001)
                or np.any(result['gas'] < 0.) or np.any(result['gas'] > 1.)):
            raise ValueError('invalid batch mechanical bounds')
        for key, shape in (('tire_force_local_N', (count, 4, 2)),
                           ('unsaturated_tire_demand_ratio', (count, 4)),
                           ('wheel_forward_mps', (count, 4))):
            value = np.asarray(diagnostics[key], float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid batch tire diagnostics')
        if np.max(np.linalg.norm(diagnostics['tire_force_local_N'], axis=-1)) > 400.00001:
            raise ValueError('batch tire force outside model bound')
        return self._copy_packed(result), {key: np.asarray(value).copy() for key, value in diagnostics.items()}

    def _advance(self, state, cm, action):
        # Preserve the inherited public single-trajectory rollout interface.
        self.model_steps += 1
        packed, diagnostics = self._call_batch(self._pack_many([self._copy_state(state)]),
                                              [[-float(action[0]), float(action[1]), float(action[2])]])
        row = {'wheel_velocity_override': None}
        for key in self.SCALARS:
            row[key] = float(packed[key][0])
        for key in self.ARRAY_SHAPES:
            row[key] = packed[key][0].copy()
        cm = cm+self.DT*row['velocity']
        hull = cm-self._rotate(row['local_center'], row['angle'])
        return row, cm, hull, {key: value[0] for key, value in diagnostics.items()}

    def _rollout_matrix(self, states, actions):
        scenario_count, action_count = len(states), len(actions)
        state = self._pack_many([item for _ in actions for item in states])
        count = action_count*scenario_count
        commands = np.repeat(np.asarray(actions, float), scenario_count, axis=0)
        commands[:, 0] *= -1.
        cm = self._rotate_many(state['local_center'], state['angle'])
        positions = np.zeros((count, self.STEPS+1, 2))
        centers = np.empty_like(positions)
        angles = np.empty((count, self.STEPS+1))
        velocities = np.empty_like(positions)
        centers[:, 0], angles[:, 0], velocities[:, 0] = cm, state['angle'], state['velocity']
        omega, ratios, forward = [], [], []
        for tick in range(1, self.STEPS+1):
            state, diagnostics = self._call_batch(state, commands)
            cm = cm+self.DT*state['velocity']
            centers[:, tick] = cm
            positions[:, tick] = cm-self._rotate_many(state['local_center'], state['angle'])
            angles[:, tick], velocities[:, tick] = state['angle'], state['velocity']
            omega.append(state['omega'].copy())
            ratios.append(diagnostics['unsaturated_tire_demand_ratio'])
            forward.append(diagnostics['wheel_forward_mps'])
            if tick == self.EXECUTED_STEPS:
                first_state = self._copy_packed(state)
        return {'positions': positions, 'angles': angles, 'velocities': velocities,
                'centers': centers, 'first_state': first_state,
                'omega': np.stack(omega, axis=1), 'ratios': np.stack(ratios, axis=1),
                'forward': np.stack(forward, axis=1)}

    @staticmethod
    def _stopped_many(state):
        return (np.linalg.norm(state['velocity'], axis=1) <= .5) & (abs(state['yaw']) <= .1)

    def _backup_matrix(self, rows, actions, eligible):
        count = len(eligible)
        scenarios = count//len(actions)
        state = self._copy_packed(rows['first_state'])
        index = self.EXECUTED_STEPS
        cm = rows['centers'][:, index].copy()
        size = index+self.BACKUP_STEPS+1
        positions = np.empty((count, size, 2))
        angles = np.empty((count, size))
        velocities = np.empty_like(positions)
        positions[:, :index+1] = rows['positions'][:, :index+1]
        angles[:, :index+1] = rows['angles'][:, :index+1]
        velocities[:, :index+1] = rows['velocities'][:, :index+1]
        selected = np.repeat(np.asarray(actions, float), scenarios, axis=0)
        steer = np.clip(-np.mean(state['joint'][:, :2], axis=1), -.4, .4)
        steer = np.clip(steer, selected[:, 0]-.24, selected[:, 0]+.24)
        # Round the held legal command exactly as the scalar backup does,
        # then promote normalized values and invert to physical targets.
        commands = np.asarray(np.stack((steer, np.zeros(count), np.ones(count)), axis=1), np.float32).astype(float)
        commands[:, 0] *= -1.
        stopped = self._stopped_many(state)
        ticks = np.where(stopped, 0, self.BACKUP_STEPS)
        for tick in range(1, self.BACKUP_STEPS+1):
            active = np.flatnonzero(eligible & ~stopped)
            if not len(active):
                break
            subset = {key: None if value is None else value[active].copy() for key, value in state.items()}
            predicted, _ = self._call_batch(subset, commands[active])
            for key, value in predicted.items():
                if value is not None:
                    state[key][active] = value
            cm[active] += self.DT*predicted['velocity']
            positions[active, index+tick] = cm[active]-self._rotate_many(predicted['local_center'], predicted['angle'])
            angles[active, index+tick], velocities[active, index+tick] = predicted['angle'], predicted['velocity']
            newly_stopped = active[self._stopped_many(predicted)]
            ticks[newly_stopped], stopped[newly_stopped] = tick, True
        progress = np.zeros(count)
        violations = np.zeros(count)
        for row in np.flatnonzero(eligible):
            stop = index+int(ticks[row])+1
            values, violation = self._geometry(positions[row, :stop], angles[row, :stop], velocities[row, :stop])
            progress[row], violations[row] = values[-1], violation
        return {'progress_m': progress, 'violation': violations, 'stopped': stopped,
                'duration_s': ticks*self.DT, 'pose_counts': index+ticks+1}

    def plan(self, supplied_states, previous_action):
        self.model_steps = self.batch_calls = self.batched_model_rows = 0
        try:
            states = [self._copy_state(state) for state in supplied_states]
            if not 1 <= len(states) <= 7:
                raise ValueError('expected one to seven supplied camera state scenarios')
            actions = self.candidate_actions(previous_action)
            if not len(actions):
                return {'feasible': False, 'action': None, 'reason': 'no_feasible_candidate',
                        'candidate_count': 0, 'feasible_count': 0, 'scenario_count': len(states),
                        'model_steps': 0, 'rejected_max_violation_m': 0., 'batch_calls': 0, 'batched_model_rows': 0}
            rows = self._rollout_matrix(states, actions)
            count = len(rows['positions'])
            progress, violations = np.empty(count), np.empty(count)
            for row in range(count):
                values, violations[row] = self._geometry(rows['positions'][row], rows['angles'][row], rows['velocities'][row])
                progress[row] = values[-1]
            backup = self._backup_matrix(rows, actions, violations <= 0.)
            ratio_cost = np.mean(np.mean(np.maximum(0., rows['ratios']-1.)**2, axis=2), axis=1)
            slip = .54*rows['omega'][:, :, 2:]-rows['forward'][:, :, 2:]
            model_cost = .10*ratio_cost+.002*np.mean(np.mean(slip**2, axis=2), axis=1)
            terminal = np.linalg.norm(rows['velocities'][:, -1], axis=1)
            best, feasible_count, worst_violation = None, 0, 0.
            for candidate, action in enumerate(actions):
                summaries = []
                for scenario in range(len(states)):
                    row = candidate*len(states)+scenario
                    # Logical telemetry preserves scalar early rejection;
                    # actual simultaneous work is reported separately.
                    self.model_steps += self.STEPS
                    worst_violation = max(worst_violation, float(violations[row]))
                    if violations[row] > 0.:
                        break
                    self.model_steps += int(round(backup['duration_s'][row]/self.DT))
                    worst_violation = max(worst_violation, float(backup['violation'][row]))
                    if not backup['stopped'][row] or backup['violation'][row] > 0.:
                        break
                    summaries.append({'progress': float(progress[row]), 'model_cost': float(model_cost[row]),
                                      'terminal_speed': float(terminal[row]),
                                      'backup_progress': float(backup['progress_m'][row]),
                                      'backup_duration': float(backup['duration_s'][row])})
                if len(summaries) != len(states):
                    continue
                feasible_count += 1
                worst_progress = min(row['progress'] for row in summaries)
                score = -worst_progress+max(row['model_cost'] for row in summaries)
                score += .5*(float(action[0])-float(previous_action[0]))**2+.02*float(action[2])
                if best is None or score < best['score']:
                    best = {'feasible': True, 'action': action.copy(), 'reason': 'feasible',
                            'score': float(score), 'worst_progress_m': worst_progress,
                            'predicted_terminal_speed_mps': min(row['terminal_speed'] for row in summaries),
                            'backup_start_s': self.EXECUTED_STEPS*self.DT,
                            'backup_end_progress_m': max(row['backup_progress'] for row in summaries),
                            'backup_duration_s': max(row['backup_duration'] for row in summaries)}
            if best is None:
                best = {'feasible': False, 'action': None, 'reason': 'no_feasible_candidate'}
            best.update(candidate_count=len(actions), feasible_count=feasible_count,
                        scenario_count=len(states), model_steps=self.model_steps,
                        rejected_max_violation_m=worst_violation,
                        batch_calls=self.batch_calls, batched_model_rows=self.batched_model_rows)
            return best
        except Exception as error:
            return {'feasible': False, 'action': None, 'reason': 'planning_error',
                    'error': type(error).__name__+': '+str(error)[:160], 'model_steps': self.model_steps,
                    'batch_calls': self.batch_calls, 'batched_model_rows': self.batched_model_rows}
