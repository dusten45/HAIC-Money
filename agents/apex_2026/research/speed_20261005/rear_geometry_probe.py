"""Frozen RearClear camera-only route attribution; map labels stay offline."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_rear_clear_agent import Agent
from agents.apex_2026.research.speed_20261005.graph_geometry import local, distance_to_path
from agents.apex_2026.research.speed_20261005.graph_curvature import arc, fit, closest_s
from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


class Inspector(Agent):
    def _route(self, ahead, center, circles, speed, yaw):
        result = super()._route(ahead, center, circles, speed, yaw)
        self.probe_route = np.column_stack((result[0], ahead))
        return result

    def _ridge(self, frame):
        result = super()._ridge(frame)
        self.probe_ridge = result
        return result

    def _target(self, ahead, path, speed):
        target = super()._target(ahead, path, speed)
        self.probe_path_target = target
        return target

    def _ridge_target(self, path, speed):
        target = super()._ridge_target(path, speed)
        self.probe_path_target = target
        return target


def dense(path):
    s = arc(path)
    at = np.arange(0., s[-1], .5)
    return np.column_stack([np.interp(at, s, path[:, j]) for j in (0, 1)])


def main():
    root = Path('.haic-artifacts/apex-speed-20261005')
    source = Path('agents/apex_2026/fast_rear_clear_agent.py')
    frames_path = root/'rear-geometry-track2/frames.npz'
    trace_path = root/'rear-clear-v1-traces/track-2-seed-644062.json'
    frames = np.load(frames_path)['frames']
    trace = json.loads(trace_path.read_text())
    ref_path = Path('agents/apex_2026/results/speed-20261005/physics-racingline-refine-v2.json')
    ref = json.loads(ref_path.read_text())['rows'][1]
    model_route = np.asarray(ref['route'])
    spec = MapSpec(2, 644062, 'official', (), 700, 4)
    env, raw = create_environment(spec, render_mode=None)
    try:
        reset_environment(env, spec)
        center = np.asarray(raw.track)[:, 2:4]
        initial = dict(position=list(raw.car.hull.position), angle=float(raw.car.hull.angle))
    finally:
        env.close()
    agent = Inspector(); rows = []; overlays = []
    for i, frame in enumerate(frames):
        agent.probe_route = agent.probe_ridge = None
        agent.probe_path_target = None
        action = agent.act(frame)
        np.testing.assert_array_equal(action, np.asarray(trace[i]['action'], np.float32))
        state = trace[i-1] if i else initial
        true = local(center, state['position'], state['angle'])
        optimal = local(model_route, state['position'], state['angle'])
        def neighborhood(path):
            nearest = int(np.argmin(np.linalg.norm(path, axis=1)))
            return dense(path[np.arange(nearest-5, nearest+22)%len(path)])
        true, optimal = neighborhood(true), neighborhood(optimal)
        road = agent.corridor_road
        circles = agent._circles(frame, road) if road is not None else []
        path = agent.probe_ridge if agent.mode == 'ridge' else agent.probe_route
        row = dict(step=i, mode=agent.mode, speed=agent.last_speed,
                   target=agent.last_target, path_target=agent.probe_path_target,
                   curvature=agent.reference_curvature, yaw=agent.last_yaw,
                   pass_side=agent.pass_side, pass_y=agent.pass_y,
                   cooldown=agent.circle_cooldown, circles=circles,
                   corridor=agent.corridor_used, arc_guard=agent.arc_guard_used,
                   action=action.tolist())
        if path is not None:
            path = path[path[:, 1]>=0.] if agent.mode != 'ridge' else path
            points = dense(path)
            row['path_center_error_m'] = float(np.mean(distance_to_path(points, true)))
            field = agent._distance_field(frame)
            row['camera_depth_min_m'] = float(np.min(agent._sample_distance(field, points)))
            row['fits'] = []
            for at in (4., 10., 18., 26.):
                p = fit(path, at)
                if p is None or at > arc(path)[-1]-2.: continue
                point = np.array([np.interp(at, arc(path), path[:,j]) for j in (0,1)])
                t = fit(true, closest_s(true, point)[0])
                o = fit(optimal, closest_s(optimal, point)[0])
                if t is None or o is None: continue
                cap = lambda k: float(min(100., np.sqrt(190./max(abs(k), .0005))))
                row['fits'].append(dict(at=at, path_k=p[0], center_k=t[0], model_k=o[0],
                                        path_cap=cap(p[0]), center_cap=cap(t[0]), model_cap=cap(o[0])))
            overlays.append((row, frame, path, true, optimal))
        rows.append(row)
    full = root/'rear-geometry-full.json'
    full.write_text(json.dumps(rows, indent=2)+'\n')
    selected = [r for r in rows if r['target']<70 and not r['circles'] and not r['pass_side'] and r.get('fits')]
    for r in selected:
        r['largest_excess_cap_mps'] = max(f['center_cap']-f['path_cap'] for f in r['fits'])
    selected.sort(key=lambda r:r['largest_excess_cap_mps'], reverse=True)
    result = dict(classification='privileged offline diagnostic; no candidate or new benchmark',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frames_sha256=hashlib.sha256(frames_path.read_bytes()).hexdigest(),
                  trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                  full_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
                  exact_actions=len(rows), new_driving_replay_episodes=1,
                  clear_low_target_frames=len(selected), worst_clear_low_target=selected[:16],
                  mode_counts={m:sum(r['mode']==m for r in rows) for m in ('ridge','confidence')},
                  limitation='Center curvature is a smoothed polyline label. Full-track model is an optimistic point-car local optimum, not a legal trajectory, actual lap, or lower bound.')
    out=Path('agents/apex_2026/results/speed-20261005/rear-geometry-probe.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r['step'] for r in selected[:8]]
    fig, axes=plt.subplots(2,4,figsize=(14,8))
    for ax, i in zip(axes.flat,chosen):
        row,frame,path,true,optimal=next(v for v in overlays if v[0]['step']==i)
        ax.imshow(frame,cmap='gray',vmin=0,vmax=1)
        for p,c,label in ((path,'r','controller'),(true,'g','road center'),(optimal,'c','offline point model')):
            ax.plot(42+p[:,0]*1.3608,63-p[:,1]*1.701,c,label=label)
        ax.set(xlim=(0,83),ylim=(73,0),title=f"{i} {row['mode']} v{row['speed']:.0f} target{row['target']:.0f}")
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out.with_suffix('.png'),dpi=130)
    print(json.dumps(dict(exact_actions=len(rows),modes=result['mode_counts'],worst=[dict(step=r['step'],mode=r['mode'],target=r['target'],excess=r['largest_excess_cap_mps'],fits=r['fits']) for r in selected[:8]])))


if __name__ == '__main__': main()
