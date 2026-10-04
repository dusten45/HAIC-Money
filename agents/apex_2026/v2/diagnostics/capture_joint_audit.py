import json,pathlib,hashlib,numpy as np,cv2
from agents.apex_2026 import evaluate as ev
root=pathlib.Path('/tmp/apex-v2-obstacle-steering-audit');original=ev.safe_act;step=0

def observed(agent, observation, timeout_sec=5., details=None):
 global step
 step+=1
 copied=np.array(observation,copy=True)
 prior=dict(throttle=float(agent.internal_throttle),model_slip=float(agent.model_slip))
 result=original(agent,observation,timeout_sec,details)
 d=agent.diagnostics
 if d.get('obstacle_shield',{}).get('active'):
  state=dict(speed=d['obstacle_shield_speed'],yaw=float(d['yaw_rate']),wheel=d['obstacle_shield_wheel'],slip=float(agent.observed_slip if agent.observed_flow else prior['model_slip']),throttle=prior['throttle'])
  np.save(root/f'{step:04d}-observation.npy',copied)
  cv2.imwrite(str(root/f'{step:04d}-frame.png'),np.clip(copied[-1]*255,0,255).astype(np.uint8))
  (root/f'{step:04d}-state.json').write_text(json.dumps(dict(step=step,state=state,action=result[0].tolist(),diagnostics=d),indent=2))
 return result

ev.safe_act=observed
result=ev.run_episode(track_id=3,seed=4111953688,agent_path='agents/apex_2026/v2/obstacle_steering_agent.py',output=root/'t3-s4111953688.json',trace=root/'t3-s4111953688.jsonl',config={})
print(result)
