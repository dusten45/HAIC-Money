"""Exact combined-controller replay and offline road/constraint attribution."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_arc_memory_agent import Agent,_PathReference,_ArcGuardReference,_EarlyCircleReference
from agents.apex_2026.research.speed_20261005.graph_geometry import local,distance_to_path
from local_simulator.environment import create_environment,reset_environment
from local_simulator.schema import MapSpec


def bounds_probe(prior,frame):
    a=copy.deepcopy(prior);road=a._road(frame)
    if road is None:return dict(road_available=False)
    ahead,center=road[:2];circles=a._circles(frame,road);speed,yaw=a._speed(frame),a._yaw(frame);original,imminent,distance=_PathReference._route(a,ahead,center,circles,speed,yaw)
    result=dict(road_available=True,road_extent_m=float(ahead[-1]),camera_circles=circles,imminent=imminent,pass_side_after=a.pass_side,pass_missing_after=a.pass_missing,pass_y_after=a.pass_y)
    if ahead[-1]<12. or imminent:result['qp_status']='early fallback';return result
    y=np.linspace(0.,float(ahead[-1]),36);slope=(np.interp(y+2.,ahead,center)-np.interp(y-2.,ahead,center))/4.;inset=1.9*np.sqrt(1+slope*slope);lower=(np.interp(y,ahead,road[3])-42)/1.3608+inset;upper=(np.interp(y,ahead,road[4])-42)/1.3608-inset
    solver_circles=list(circles);remembered=a.pass_side and a.pass_missing>0 and a.pass_y is not None
    if remembered:solver_circles.append((a.pass_y,a.pass_x))
    result['remembered_constraint_added']=bool(remembered);result['solver_circles']=solver_circles
    for oy,ox in solver_circles:
        if oy<-4 or oy>y[-1]+4:continue
        side=a.pass_side if distance is not None and abs(oy-distance)<1 else(-1 if ox>=np.interp(oy,ahead,original) else 1)
        extent=3.8*np.sqrt(np.maximum(0.,1-((y-oy)/4.5)**2));active=abs(y-oy)<4.5
        if side<0:upper[active]=np.minimum(upper[active],ox-extent[active])
        else:lower[active]=np.maximum(lower[active],ox+extent[active])
    result.update(qp_status='infeasible fallback' if np.any(lower>upper) or np.any(lower[:2]>0.) or np.any(upper[:2]<0.) else 'solved',bound_conflict_max_m=float(np.max(lower-upper)),ego_lower_m=lower[:2].tolist(),ego_upper_m=upper[:2].tolist())
    return result


def main():
    root=Path('.haic-artifacts/apex-speed-20261005');frames=np.load(root/'arc-memory-v1-probe-track4/frames.npz')['frames'];trace=json.loads((root/'arc-memory-v1-traces/track-4-seed-18800.json').read_text());parent=json.loads((root/'arc-clear-v2-traces/track-4-seed-18800.json').read_text());spec=MapSpec(4,18800,'official',(),700,4);env,raw=create_environment(spec,None)
    try:reset_environment(env,spec);track=np.asarray(raw.track)[:,2:4];obstacles=np.asarray([list(b.position) for b in raw.obstacles]);initial=dict(position=list(raw.car.hull.position),angle=float(raw.car.hull.angle))
    finally:env.close()
    agent=Agent();rows=[];fixtures=[];focus=set(range(23,34))|set(range(90,112))|{120,125,135,140}
    for i,frame in enumerate(frames):
        before=copy.deepcopy(agent)
        if i in (94,95):fixtures.append(dict(step=i,state={name:getattr(before,name) for name in ('last_steer','last_target','last_speed','last_yaw','last_center','lost_frames','pass_side','pass_missing','pass_x','pass_y','route_distance','spin_frames','circle_cooldown','mode','arc_guard_used','arc_guard_curve')}))
        diagnostic=bounds_probe(before,frame) if i in focus else None
        if i in (94,95,96):
            probe=copy.deepcopy(before);r=probe._road(frame)
            if r is not None and r[0][-1]>=6.:
                speed,yaw=probe._speed(frame),probe._yaw(frame);path,_,_=probe._route(r[0],r[1],probe._circles(frame,r),speed,yaw)
                original=_EarlyCircleReference._steering(probe,r[0],path,speed,yaw);original_emitted=probe.last_steer+float(np.clip(original-probe.last_steer,-.24,.24));field=probe._distance_field(frame);original_depth=probe._arc_guard_depth(field,original_emitted)
                probe.arc_guard_used=False;selected=_ArcGuardReference._steering(probe,r[0],path,speed,yaw);selected_emitted=probe.last_steer+float(np.clip(selected-probe.last_steer,-.24,.24))
                diagnostic['ungated_arc_guard_counterfactual']=dict(original_emitted_steer=original_emitted,original_depth_m=original_depth,selected_emitted_steer=selected_emitted,selected_depth_m=probe._arc_guard_depth(field,selected_emitted),selected_used=probe.arc_guard_used,opposite_turn=original_emitted*selected_emitted<0.)
        action=agent.act(np.stack([frame]*4));np.testing.assert_array_equal(action,np.asarray(trace[i]['action'],np.float32))
        if diagnostic is None:continue
        state=trace[i-1] if i else initial;road_truth=local(track,state['position'],state['angle']);circle_truth=local(obstacles,state['position'],state['angle']);nearest=int(np.argmin(np.linalg.norm(circle_truth,axis=1)));ego_gap=float(distance_to_path(np.zeros((1,2)),np.vstack((road_truth,road_truth[0])))[0]);near_circle=circle_truth[nearest];diagnostic.update(actual_route_invoked=not agent.lost_frames,step=i,mode=agent.mode,lost_frames=agent.lost_frames,arc_guard_used=agent.arc_guard_used,corridor_used=agent.corridor_used,action=action.tolist(),speed_hud=agent.last_speed,target=agent.last_target,pass_side=agent.pass_side,pass_missing=agent.pass_missing,pass_y=agent.pass_y,ego_center_gap_m=ego_gap,true_nearest_circle_local_m=near_circle.tolist(),true_nearest_circle_index=nearest,progress=trace[i]['progress']);rows.append(diagnostic)
    np.savez_compressed('agents/apex_2026/results/speed-20261005/arc-memory-probe-fixtures.npz',frames=frames[[94,95]])
    Path('agents/apex_2026/results/speed-20261005/arc-memory-probe-fixtures.json').write_text(json.dumps(fixtures,indent=2)+'\n')
    result=dict(trace_sha256=hashlib.sha256((root/'arc-memory-v1-traces/track-4-seed-18800.json').read_bytes()).hexdigest(),frames_sha256=hashlib.sha256((root/'arc-memory-v1-probe-track4/frames.npz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path('agents/apex_2026/fast_arc_memory_agent.py').read_bytes()).hexdigest(),exact_replayed_actions=len(frames),new_driving_episodes=0,first_action_divergence=next(i for i,(a,b) in enumerate(zip(parent,trace)) if a['action']!=b['action']),new_contact_steps=[r['step'] for r in trace if r['collision']],parent_contact_steps=[r['step'] for r in parent if r['collision']],rows=rows);Path('agents/apex_2026/results/speed-20261005/arc-memory-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([dict(step=r['step'],counterfactual=r.get('ungated_arc_guard_counterfactual')) for r in rows if r['step'] in (94,95,96)]))
if __name__=='__main__':main()
