"""Privileged offline attribution of camera ridge tangent/curvature errors.

Saved images determine every inferred route. Track and pre-action hull pose
only label diagnostics. No driving episodes or submission edits are made.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_ridge_agent import Agent
from agents.apex_2026.fast_confidence_agent import Agent as ConfidenceAgent
from agents.apex_2026.research.speed_20261005.graph_geometry import local
from local_simulator.environment import create_environment,reset_environment
from local_simulator.schema import MapSpec


def arc(path):return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]


def fit(path,at,window=7.):
    s=arc(path);selected=abs(s-at)<=window
    if selected.sum()<5:return None
    x=s[selected]-at;w=np.exp(-.5*(x/5.)**2)
    cx=np.polyfit(x,path[selected,0],2,w=w);cy=np.polyfit(x,path[selected,1],2,w=w)
    derivative=np.array([cx[1],cy[1]])
    k=(cx[1]*2*cy[0]-cy[1]*2*cx[0])/max(np.linalg.norm(derivative)**3,.001)
    theta=np.arctan2(cx[1],cy[1])
    return float(k),float(theta)


def closest_s(path,point):
    links=np.diff(path,axis=0);s=arc(path);fraction=np.clip(np.sum((point-path[:-1])*links,axis=1)/np.maximum(np.sum(links*links,axis=1),1e-8),0.,1.);distance=np.linalg.norm(point-path[:-1]-fraction[:,None]*links,axis=1);i=int(np.argmin(distance))
    return float(s[i]+fraction[i]*np.linalg.norm(links[i])),float(distance[i])


def smooth(path,window=5.):
    """Camera-only local quadratic coordinate smoothing; retains endpoints."""
    s=arc(path);result=path.copy()
    for i in range(1,len(path)-1):
        select=abs(s-s[i])<=window
        if select.sum()<5:continue
        x=s[select]-s[i];w=np.exp(-.5*(x/(.7*window))**2)
        result[i]=[np.polyfit(x,path[select,j],2,w=w)[-1] for j in (0,1)]
    return result


def supported_prefix(agent, frame, path, minimum_depth=5.8):
    depth=agent._sample_distance(agent._distance_field(frame),path)
    unsupported=np.flatnonzero(depth<minimum_depth)
    result=path[:int(unsupported[0])] if len(unsupported) else path
    return result if len(result)>=5 else None


def residual_uncertainty(path, at):
    s=arc(path);selected=abs(s-at)<=7.
    if selected.sum()<5:return None
    x=s[selected]-at;weights=np.exp(-.5*(x/5.)**2)
    design=np.column_stack((x*x,x,np.ones_like(x)))
    weighted=design*weights[:,None]
    coefficient=np.linalg.lstsq(weighted,path[selected]*weights[:,None],rcond=None)[0]
    residual=path[selected]-design@coefficient
    variance=np.sum((residual*weights[:,None])**2,axis=0)/max(len(x)-3,1)
    inverse=np.linalg.inv(weighted.T@weighted)
    ax,ay=coefficient[0];vx,vy=coefficient[1]
    scale=max((vx*vx+vy*vy)**1.5,.001)
    # Approximate only second-derivative variance, holding local tangent
    # fixed. This independent residual model omits systematic image bias.
    sigma=2.*np.sqrt(max(inverse[0,0]*(vx*vx*variance[1]+vy*vy*variance[0]),0.))/scale
    return dict(fit_residual_rms_m=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),curvature_sigma_per_m=float(sigma))


def summary(rows,key):
    valid=[r for r in rows if key in r];error=np.asarray([r[key]['curvature']-r['truth']['curvature'] for r in valid]);angle=np.asarray([r[key]['heading_error_deg'] for r in valid]);over=np.asarray([r[key]['speed_cap_mps']-r['truth']['speed_cap_mps'] for r in valid]);return dict(samples=len(valid),curvature_mae_per_m=float(np.mean(abs(error))),curvature_bias_per_m=float(np.mean(error)),heading_mae_deg=float(np.mean(abs(angle))),speed_cap_mean_difference_mps=float(np.mean(over)),speed_cap_p10_difference_mps=float(np.percentile(over,10)),speed_cap_p90_difference_mps=float(np.percentile(over,90)),excess_braking_samples=int(np.sum(over<-5)),unsafe_extra_speed_samples=int(np.sum(over>5)))


def main():
    agent=Agent();confidence=ConfidenceAgent();root=Path('.haic-artifacts/apex-speed-20261005');rows=[];excluded=[]
    for track,seed,maxstep in [(2,644062,170),(4,18800,150)]:
        spec=MapSpec(track,seed,'official',(),700,4);env,raw=create_environment(spec,render_mode=None)
        try:reset_environment(env,spec);road=np.asarray(raw.track)[:,2:4];initial=dict(position=list(raw.car.hull.position),angle=float(raw.car.hull.angle))
        finally:env.close()
        recent_circle_age=999
        frames=np.load(root/f'corridor-v1-probe-track{track}/frames.npz')['frames'];trace=json.loads((root/'corridor-v1-traces'/f'track-{track}-seed-{seed}.json').read_text())
        for i,frame in enumerate(frames[:maxstep]):
            confidence_road=confidence._road(frame); circles=confidence._circles(frame,confidence_road) if confidence_road is not None else []
            recent_circle_age=0 if circles else recent_circle_age+1
            state=trace[i-1] if i else initial;nearest=int(np.argmin(np.linalg.norm(road-state['position'],axis=1)));truth=local(road[np.arange(nearest-5,nearest+25)%len(road)],state['position'],state['angle']);truth_arc=arc(truth); dense_arc=np.arange(0.,truth_arc[-1],1.); truth=np.column_stack([np.interp(dense_arc,truth_arc,truth[:,j]) for j in (0,1)]); _,ego_gap=closest_s(truth,np.zeros(2));ridge=agent._ridge(frame)
            if ego_gap>4.8 or ridge is None:excluded.append(dict(track=track,step=i,ego_center_gap_m=ego_gap,ridge_available=ridge is not None));continue
            routes={'raw':ridge,'smooth5':smooth(ridge,5.),'smooth7':smooth(ridge,7.)}; supported=supported_prefix(agent,frame,ridge)
            if supported is not None:routes['supported58']=supported
            s=arc(ridge)
            for at in (4.,10.,18.,26.):
                if at>s[-1]-2.:continue
                point=np.array([np.interp(at,s,ridge[:,j]) for j in (0,1)]);truth_at,gap=closest_s(truth,point);expected=fit(truth,truth_at)
                if expected is None:continue
                true_k,true_theta=expected;row=dict(track=track,step=i,at_m=at,road_confident=confidence_road is not None,camera_circle_count=len(circles),recent_circle_age=recent_circle_age,ego_center_gap_m=ego_gap,ridge_point_gap_m=gap,truth=dict(curvature=true_k,speed_cap_mps=float(min(100.,np.sqrt(190./max(abs(true_k),.0005))))))
                for name,path in routes.items():
                    inferred=fit(path,at) if at<=arc(path)[-1]-2. else None
                    if inferred is None:continue
                    k,theta=inferred;heading_error=np.arctan2(np.sin(theta-true_theta),np.cos(theta-true_theta));row[name]=dict(curvature=k,heading_error_deg=float(np.degrees(heading_error)),speed_cap_mps=float(min(100.,np.sqrt(190./max(abs(k),.0005)))),**residual_uncertainty(path,at))
                rows.append(row)
    result=dict(source_sha256=hashlib.sha256(Path('agents/apex_2026/fast_ridge_agent.py').read_bytes()).hexdigest(),new_driving_episodes=0,selection='Camera prefixes track2 frames0..169 / track4 frames0..149; retain ego track-center distance <=4.8m and available ridge',summary={key:summary(rows,key) for key in ('raw','smooth5','smooth7','supported58')},subsets={label:{key:summary(selected,key) for key in ('raw','smooth5','smooth7','supported58')} for label,selected in [('clear_confident',[r for r in rows if r['road_confident'] and r['recent_circle_age']>12]),('actual_turns',[r for r in rows if abs(r['truth']['curvature'])>.012])]},excluded=excluded,rows=rows)
    result['residual_uncertainty_diagnostic']={key:dict(samples=len(valid),within_three_sigma=int(sum(abs(r[key]['curvature']-r['truth']['curvature'])<=3*r[key]['curvature_sigma_per_m'] for r in valid)),curvature_error_residual_correlation=float(np.corrcoef([abs(r[key]['curvature']-r['truth']['curvature']) for r in valid],[r[key]['fit_residual_rms_m'] for r in valid])[0,1])) for key in ('raw','smooth7') for valid in [[r for r in rows if key in r and abs(r['truth']['curvature'])>.012]]}
    result['truth_method']='Official center polyline resampled at1m arclength; same weighted quadratic +/-7m curvature fit, local nearest-route projection. This is a smoothed reference, not exact tire-required curvature.'
    result['method_parameters']=dict(curvature_fit_halfwindow_m=7.,fit_weight_sigma_m=5.,smoothing_halfwindows_m=[5.,7.],depth_support_min_m=5.8,lateral_accel_mps2=190.,cruise_cap_mps=100.,truth_polyline_resample_m=1.,clear_circle_age_min_frames=12)
    result['helper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    full_path=root/'ridge-curvature-full.json';full_path.write_text(json.dumps(result,indent=2)+'\n')
    compact={key:value for key,value in result.items() if key not in ('rows','excluded')};compact['excluded_frames']=len(excluded);compact['total_point_windows']=len(rows);compact['full_data_sha256']=hashlib.sha256(full_path.read_bytes()).hexdigest();compact['representative_worst_rows']=sorted([r for r in rows if 'raw' in r],key=lambda r:abs(r['raw']['curvature']-r['truth']['curvature']),reverse=True)[:12]
    Path('agents/apex_2026/results/speed-20261005/ridge-curvature.json').write_text(json.dumps(compact,indent=2)+'\n');print(json.dumps(compact['summary']))
if __name__=='__main__':main()
