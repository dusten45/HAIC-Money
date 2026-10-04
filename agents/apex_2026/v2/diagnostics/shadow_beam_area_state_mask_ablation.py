exec(open('/tmp/apex-v2-beam-r4/shifted_plan.py').read().split('out=[]')[0])
# Current138 reconstructedstate initial exists in138 ego coordinate frame.
current_initial=initial
old=Agent();oldstack=obs[17];oldd=rows[136]['policy_diagnostics'];_,oldslip,_=old.road._motion(oldstack[-2],oldstack[-1]);old.shadow.reset(oldd['speed']*math.cos(oldslip),-oldd['yaw_rate'],float(np.clip(-oldd['wheel_angle'],-.4,.4)),lateral_speed=oldd['speed']*math.sin(oldslip),throttle=.699999988079071,omegas=decode_wheel_omega(oldstack[-1]));old.shadow.step(sequence[0],4);pred_initial=old.snapshot(old.shadow);oldfree,_,_=old.road._free_space(oldstack[-1]);oldfree[59:68,40:45]=1
r=rows[136];h=r['before']['heading_rad'];delta=np.array(r['after']['position'])-r['before']['position'];translation=np.array([math.cos(h)*delta[0]+math.sin(h)*delta[1],-math.sin(h)*delta[0]+math.cos(h)*delta[1]]);rotation=r['after']['heading_rad']-h

def transform(state,theta,offset):
 c,s=math.cos(theta),math.sin(theta);rot=np.array([[c,-s],[s,c]]);bodies=[]
 for pos,angle,vel,yaw in state[0]:bodies.append((tuple(rot@pos+offset),angle+theta,tuple(rot@vel),yaw))
 return bodies,state[1]
rot=np.array([[math.cos(-rotation),-math.sin(-rotation)],[math.sin(-rotation),math.cos(-rotation)]])
variants=[('predictedstate_oldmask',pred_initial,oldfree),('observedstate_oldmask',transform(current_initial,rotation,translation),oldfree),('predictedstate_newmask',transform(pred_initial,-rotation,-rot@translation),free),('observedstate_newmask',current_initial,free)]
results=[]
for label,state,mask in variants:
 model=Agent();model.restore(model.shadow,state);costs=[]
 for action in sequence[1:]:
  for _ in range(4):
   x,y,h,_,_=model.shadow.step(action);costs.append(model._footprint_cost(mask,x,y,h))
 results.append({'label':label,'firstblock':costs[:4],'all_ticks':costs,'total':sum(costs)});print(label,costs[:4],sum(costs))
Path('/tmp/apex-v2-beam-r4/state_mask_ablation.json').write_text(json.dumps({'coordinate_transform':{'translation':translation.tolist(),'rotation':rotation},'results':results,'scope':'Offline two-by-two predicted/current pixelstate versus137/138 observedraster; exactrigid coordinate transform. No environmentreset.'},indent=2))
