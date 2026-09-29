"""Keep stronger acceleration; add threat-entry brake authority."""
import numpy as np
from haic_agent.acceleration_envelope_runtime import AccelerationEnvelopeAgent
class AccelerationBrakingAgent(AccelerationEnvelopeAgent):
    def act(self,observation):
        action=super().act(observation)
        accelerated=action.copy()
        centers=self.last['road_centers']
        sweep=max(centers.values())-min(centers.values()) if len(centers)>1 else 99.
        threat=(self.base._last_obstacle is not None or 42 not in centers or 54 not in centers
                or abs(centers.get(54,100.)-42.)>3. or sweep>4.)
        excess=self.last['pixel_speed']-self.last['target_speed']
        applied=False
        if self.steps>10 and self.accel_mode!='control' and threat and excess>2.:
            action[1]=0.
            action[2]=max(float(action[2]),float(np.clip(.08*excess,.2,1.)))
            applied=bool(np.max(np.abs(action-accelerated))>1e-6)
        self.brake_history[-1]=float(action[2])
        parent=np.asarray(self.last['parent_action'])
        self.last.update(brake_takeover=applied,brake_threat=bool(threat),brake_excess=float(excess),
                         pre_brake_action=accelerated.tolist(),
                         mechanism_changed=bool(np.max(np.abs(action-parent))>1e-6),
                         completion_changed=bool(np.max(np.abs(action-parent))>1e-6),
                         final_gas=float(action[1]),final_brake=float(action[2]))
        return action
