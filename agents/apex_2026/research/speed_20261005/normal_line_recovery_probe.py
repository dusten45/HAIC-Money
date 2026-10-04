"""Exact saved camera attribution of Normal V1's T2 recovery regression."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_normal_line_agent import Agent, _RearClearReference, _ConfidenceReference
from agents.apex_2026.research.speed_20261005.graph_geometry import local,distance_to_path
from local_simulator.environment import create_environment,reset_environment
from local_simulator.schema import MapSpec


class Inspector(Agent):
    def _ridge(self,frame):
        base=_RearClearReference._ridge(self,frame)
        result=super()._ridge(frame)
        self.probe_selected=(base is not None and result is not None and not np.array_equal(base,result))
        self.probe_path_depth=self._line_body_depth(self._distance_field(frame),result) if self.probe_selected else None
        return result


def main():
    root=Path('.haic-artifacts/apex-speed-20261005')
    source=Path('agents/apex_2026/fast_normal_line_agent.py')
    framespath=root/'normal-line-v1-probe-track2/frames.npz'
    tracepath=root/'normal-line-v1-traces/track-2-seed-644062.json'
    frames=np.load(framespath)['frames'];trace=json.loads(tracepath.read_text())
    spec=MapSpec(2,644062,'official',(),700,4);env,raw=create_environment(spec,render_mode=None)
    try:
        reset_environment(env,spec);track=np.asarray(raw.track)[:,2:4]
        initial=dict(position=list(raw.car.hull.position),angle=float(raw.car.hull.angle))
    finally:env.close()
    agent=Inspector();rows=[]
    for i,frame in enumerate(frames):
        agent.probe_selected=False;agent.probe_path_depth=None
        previous_steer=agent.last_steer
        confidence=copy.deepcopy(agent)
        confidence.arc_guard_used=False;confidence.arc_guard_curve=0.
        confidence_action=_ConfidenceReference.act(confidence,frame)
        action=agent.act(frame)
        np.testing.assert_array_equal(action,np.asarray(trace[i]['action'],np.float32))
        state=trace[i-1] if i else initial
        road=local(track,state['position'],state['angle'])
        center_gap=float(distance_to_path(np.zeros((1,2)),np.vstack((road,road[0])))[0])
        if i>0:
            previous=trace[i-2] if i>1 else initial
            velocity=np.asarray(state['position'])-previous['position']
            vlocal=local(velocity[None],[0.,0.],state['angle'])[0]
            slip=float(np.degrees(np.arctan2(vlocal[0],vlocal[1])))
        else:slip=0.
        field=agent._distance_field(frame)
        depth=agent._arc_guard_depth(field,float(action[0]))
        row=dict(step=i,mode=agent.mode,optimized_selected=agent.probe_selected,
                 optimized_path_body_depth_m=agent.probe_path_depth,
                 emitted_8m_arc_body_depth_m=depth,speed=agent.last_speed,
                 yaw=agent.last_yaw,reference_curvature=agent.reference_curvature,
                 emitted_kinematic_curvature=float(np.tan(action[0])/agent.WHEELBASE),
                 emitted_lateral_budget=float(agent.last_speed**2*abs(np.tan(action[0]))/agent.WHEELBASE),
                 target=agent.last_target,lost_frames=agent.lost_frames,
                 road_available=agent.corridor_road is not None,road_extent_m=float(agent.corridor_road[0][-1]) if agent.corridor_road is not None else None,
                 ego_center_gap_m=center_gap,offline_velocity_slip_deg=slip,
                 previous_steer=previous_steer,action=action.tolist())
        row.update(confidence_action=confidence_action.tolist(),
                   confidence_arc_guard_used=confidence.arc_guard_used,
                   confidence_target=confidence.last_target,
                   confidence_emitted_8m_depth_m=confidence._arc_guard_depth(field,float(confidence_action[0])))
        rows.append(row)
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frames_sha256=hashlib.sha256(framespath.read_bytes()).hexdigest(),
                trace_sha256=hashlib.sha256(tracepath.read_bytes()).hexdigest(),
                exact_actions=len(rows),new_diagnostic_replays=1,
                selected_rows=[r for r in rows if r['optimized_selected']],onset_rows=rows[140:173],
                limitation='Arc assumes constant emitted steer over8m. Body path and arc checks are camera heuristics; label pose/velocity/track enter offline reporting only.')
    path=Path('agents/apex_2026/results/speed-20261005/normal-line-recovery-probe.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(selected=result['selected_rows'],onset=[r for r in rows[145:160]])))


if __name__=='__main__':main()
