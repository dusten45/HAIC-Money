"""Independent pixel-only handoff/path mechanisms, retaining frozen dual reference."""
import numpy as np
from haic_agent.connected_guarded_preview_runtime import ConnectedGuardedPreviewAgent
from haic_agent.fast_completion_coordination import observed_center

MODES = ('control', 'dual_reference', 'handoff_decay', 'early_gap', 'heading_continuation', 'straight_acceleration')

class HandoffPathAgent(ConnectedGuardedPreviewAgent):
    def __init__(self, mechanism='control'):
        if mechanism not in MODES:
            raise ValueError('unknown mechanism')
        self.handoff_mode = mechanism
        super().__init__('control' if mechanism == 'control' else 'dual_preview')

    def reset(self, observation=None):
        super().reset(observation)
        self.previous_residual = 0.
        self.reliable_steer = 0.
        self.heading_lost = 0
        self.gap_side = 0.
        self.gap_missing = 0

    def act(self, observation):
        impact_before = self.impact_left
        action = super().act(observation)
        parent = action.copy()
        baseline = np.asarray(self.last['baseline_action'])
        obstacle = self.base._last_obstacle
        centers = self.last['road_centers']
        activated = False
        if obstacle is not None:
            self.gap_missing = 0
            if self.gap_side == 0.:
                self.gap_side = float(self.base._obstacle_side)
        else:
            self.gap_missing += 1
            if self.gap_missing > 3:
                self.gap_side = 0.
        if self.steps > 10:
            if self.handoff_mode == 'handoff_decay':
                residual = float(action[0]-baseline[0])
                if not self.last['preview_authorized']:
                    residual = float(np.sign(self.previous_residual)*max(0., abs(self.previous_residual)-.04))
                    action[0] = np.clip(baseline[0]+residual, -.7, .7)
                    activated = abs(residual) > 1e-6
                self.previous_residual = residual
            elif self.handoff_mode == 'early_gap':
                if obstacle is not None and 26. <= obstacle[0] <= 52. and impact_before == 0 and not self.last['impact_proxy_trigger']:
                    y, x, _ = obstacle
                    gap = x+self.gap_side*9.
                    center = observed_center(centers, y)
                    if center is not None:
                        residual = np.clip(.018*(gap-center), -.12, .12)*float(np.clip((y-26.)/14., 0., 1.))
                        action[0] = np.clip(action[0]+residual, -.7, .7)
                        activated = abs(residual) > 1e-6
            elif self.handoff_mode == 'heading_continuation':
                if 42 in centers and 54 in centers:
                    self.reliable_steer = float(action[0])
                    self.heading_lost = 0
                else:
                    self.heading_lost += 1
                    near, middle = observed_center(centers, 54), observed_center(centers, 42)
                    if near is not None and middle is not None and obstacle is None:
                        action[0] = np.clip(.022*(middle-42.)+.018*(middle-near)+self.last['correction'], -.7, .7)
                        self.reliable_steer = float(action[0])
                        activated = True
                    elif len(centers) < 2 and self.heading_lost <= 6 and obstacle is None:
                        action[0] = self.reliable_steer
                        activated = True
            elif self.handoff_mode == 'straight_acceleration':
                target = self.last['connected_target']
                if (target is not None and self.last['connected_reach'] > 28.
                        and abs(centers.get(54, 100.)-42.) < 2. and abs(target[0]) < 2.
                        and obstacle is None and self.last['preview_authorized'] and self.last['pixel_speed'] < 68.):
                    action[1], action[2] = max(float(action[1]), .5), 0.
                    activated = True
        self.brake_history[-1] = float(action[2])
        self.last.update(handoff_mode=self.handoff_mode, mechanism_active=bool(activated),
                         mechanism_changed=bool(np.max(np.abs(action-parent)) > 1e-6),
                         parent_action=parent.tolist(), residual_memory=self.previous_residual,
                         heading_lost=self.heading_lost, observed_obstacle=None if obstacle is None else list(obstacle),
                         observed_obstacle_side=float(self.base._obstacle_side),
                         completion_changed=bool(np.max(np.abs(action-baseline)) > 1e-6),
                         final_gas=float(action[1]), final_brake=float(action[2]))
        return action
