"""Research-only HUD-yaw-constrained image registration and measurement audit.

estimate_motion uses two public grayscale frames and their pixel-decoded yaw only.
Unknown translation/slip remains None. Diagnostic capture never feeds truth to policy.
"""
from __future__ import annotations
import cv2
import numpy as np

PIXELS_X, PIXELS_Y = 1.3608, 1.701


def estimate_motion(previous, current, previous_yaw, current_yaw, dt=.08, speed_hint=None):
    """Estimate average displacement velocity in the current camera body frame.

    Yaw is right-positive rad/s. This is interval-average motion, not a claim
    of instantaneous velocity or tire slip at the final physics tick.
    """
    result=dict(valid=False,reason='invalid_input',translation_pixels=None,
                lateral_m_s=None,forward_m_s=None,slip_rad=None,
                correlation=None,gradient_min_eigenvalue=None)
    a,b=np.asarray(previous),np.asarray(current)
    if (a.shape!=(84,84) or b.shape!=(84,84) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite([previous_yaw,current_yaw,dt]).all()
            or dt<=0):
        return result
    a,b=a.astype(np.float32),b.astype(np.float32)
    # Box2D advances angle with each new angular velocity, four right-endpoint
    # samples per action. Linear yaw interpolation gives weights3/8 and5/8.
    ticks=max(1,int(round(dt/.02)))
    current_weight=(ticks+1)/(2*ticks)
    theta=-((1-current_weight)*previous_yaw+current_weight*current_yaw)*dt
    c,s=np.cos(theta),np.sin(theta)
    linear=np.array([[c,-s*PIXELS_X/PIXELS_Y],[s*PIXELS_Y/PIXELS_X,c]])
    center=np.array([42.,63.])
    matrix=np.column_stack((linear,center-linear@center)).astype(np.float32)
    rotated=cv2.warpAffine(a,matrix,(84,84))
    template=rotated[12:52,16:68]
    gx=cv2.Sobel(template,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(template,cv2.CV_32F,0,1,ksize=3)/8
    gradients=np.column_stack((gx.ravel(),gy.ravel()))
    eigen=float(np.linalg.eigvalsh(gradients.T@gradients/len(gradients))[0])
    result['gradient_min_eigenvalue']=eigen
    if eigen<1e-5:
        result['reason']='translation_unobservable'
        return result
    matches=cv2.matchTemplate(b[:74],template,cv2.TM_CCOEFF_NORMED)
    _,peak,_,location=cv2.minMaxLoc(matches)
    if peak<.5:
        result.update(reason='weak_coarse_match',correlation=float(peak))
        return result
    transform=np.array([[1.,0.,location[0]-16],[0.,1.,location[1]-12]],np.float32)
    mask=np.zeros((84,84),np.uint8);mask[5:72,5:79]=255;mask[54:72,31:53]=0
    try:
        corr,transform=cv2.findTransformECC(rotated,b,transform,cv2.MOTION_TRANSLATION,
                        (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-5),mask,3)
    except cv2.error:
        result['reason']='refinement_failed'
        return result
    dx,dy=map(float,transform[:,2]);lateral=-dx/PIXELS_X/dt;forward=dy/PIXELS_Y/dt
    result.update(correlation=float(corr))
    if corr<.75 or abs(dx)>16 or dy<-.5 or dy>20 or np.hypot(lateral,forward)>140:
        result['reason']='confidence_or_motion_bound'
        return result
    if speed_hint is not None and (not np.isfinite(speed_hint) or abs(np.hypot(lateral,forward)-speed_hint)>15.):
        result['reason']='pixel_speed_inconsistent'
        return result
    result.update(valid=True,reason='ok',translation_pixels=[dx,dy],lateral_m_s=lateral,
                  forward_m_s=forward,slip_rad=float(np.arctan2(lateral,max(.1,forward))))
    return result


def interval_truth(before, after):
    """Privileged diagnostic truth only; never called by a runtime candidate."""
    dt=after['t']-before['t'];c,s=np.cos(after['angle']),np.sin(after['angle'])
    displacement=(np.array(after['position'])-before['position'])/dt
    lateral=float(displacement@[c,s]);forward=float(displacement@[-s,c])
    velocity=np.array(after['velocity'])
    return dict(lateral_m_s=lateral,forward_m_s=forward,
                slip_rad=float(np.arctan2(lateral,forward)),
                endpoint_slip_rad=float(np.arctan2(velocity@[c,s],velocity@[-s,c])),
                endpoint_speed_m_s=float(np.linalg.norm(velocity)))


def capture(output_dir):
    """Exactly one fresh REQUIRED t1/516237 unchanged frozen-P1 episode."""
    import hashlib
    import json
    from pathlib import Path
    from agents.apex_2026.evaluate import make_environment, run_episode
    root=Path(__file__).resolve().parents[4]
    source=root/'agents/apex_2026/candidate/agent.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='b27142578d98eb090da8ac858eb784d48053b026142cf894b072da2c6440ac97'
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    class RecordingEnvironment:
        def __init__(self):
            self.environment=make_environment();self.observations=[];self.truth=[]
        def __getattr__(self,name):return getattr(self.environment,name)
        def record(self,observation):
            raw=self.environment.unwrapped;h=raw.car.hull
            self.observations.append(np.array(observation,copy=True))
            self.truth.append(dict(t=float(raw.t),position=list(map(float,h.position)),
                velocity=list(map(float,h.linearVelocity)),angle=float(h.angle),
                angular_velocity=float(h.angularVelocity)))
        def reset(self,**kwargs):
            observation,info=self.environment.reset(**kwargs);self.record(observation)
            return observation,info
        def step(self,action):
            result=self.environment.step(action);self.record(result[0]);return result
        def close(self):
            self.environment.close()
            np.savez_compressed(output_dir/'observations.npz',observations=np.array(self.observations))
            (output_dir/'truth.json').write_text(json.dumps(self.truth,indent=2)+'\n')
    return run_episode(source,1,516237,output_dir/'receipt.json',trace=output_dir/'trace.jsonl',
                       env_factory=RecordingEnvironment)


def analyze(capture_dir,output):
    import hashlib
    import importlib.util
    import json
    import time
    from datetime import datetime,timezone
    from pathlib import Path
    root=Path(__file__).resolve().parents[4];folder=Path(capture_dir)
    candidate=root/'agents/apex_2026/candidate/agent.py'
    spec=importlib.util.spec_from_file_location('unchanged_motion_baseline',candidate)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);agent=m.Agent()
    observations=np.load(folder/'observations.npz')['observations']
    truth=json.loads((folder/'truth.json').read_text());receipt=json.loads((folder/'receipt.json').read_text())
    assert (receipt['track_id'],receipt['seed'])==(1,516237)
    assert len(observations)==len(truth)==receipt['steps']+1
    rows=[]
    for i in range(1,len(truth)):
        previous,current=observations[i-1,-1],observations[i,-1]
        old_yaw=agent._hud_dynamics(previous)[0];new_yaw=agent._hud_dynamics(current)[0]
        dt=truth[i]['t']-truth[i-1]['t'];target=interval_truth(truth[i-1],truth[i])
        pixel_speed=float(np.clip((.5*(previous[77:83,10:13].sum()+current[77:83,10:13].sum())-.27)/.085,0,100))
        start=time.perf_counter();dense=estimate_motion(previous,current,old_yaw,new_yaw,dt,speed_hint=pixel_speed);dense['latency_s']=time.perf_counter()-start
        start=time.perf_counter();lk_yaw,lk_slip,lk_valid=agent._motion(previous,current);lk=dict(valid=lk_valid,slip_rad=lk_slip if lk_valid else None,yaw_rad_s=lk_yaw if lk_valid else None,latency_s=time.perf_counter()-start)
        fallback=dict(lk if lk['valid'] else dense)
        fallback['selected_method']='lk' if lk['valid'] else ('dense' if dense['valid'] else None)
        fallback['latency_s']=lk['latency_s']+(0. if lk['valid'] else dense['latency_s'])
        rows.append(dict(step=i,truth=target,hud_yaw_rad_s=[old_yaw,new_yaw],dense=dense,lk=lk,lk_then_dense=fallback))
    def stats(part,method):
        valid=[r for r in part if r[method]['valid']]
        errors=[abs(r[method]['slip_rad']-r['truth']['slip_rad']) for r in valid]
        endpoint=[abs(r[method]['slip_rad']-r['truth']['endpoint_slip_rad']) for r in valid]
        result=dict(eligible=len(part),valid=len(valid),availability=len(valid)/len(part) if part else None,
                    slip_mae_rad=float(np.mean(errors)) if errors else None,
                    slip_p90_abs_error_rad=float(np.percentile(errors,90)) if errors else None,
                    slip_max_abs_error_rad=max(errors) if errors else None,
                    endpoint_slip_mae_rad=float(np.mean(endpoint)) if endpoint else None,
                    mean_latency_s=float(np.mean([r[method]['latency_s'] for r in part])) if part else None)
        if method=='dense' and valid:
            result['lateral_velocity_mae_m_s']=float(np.mean([abs(r['dense']['lateral_m_s']-r['truth']['lateral_m_s']) for r in valid]))
            result['forward_velocity_mae_m_s']=float(np.mean([abs(r['dense']['forward_m_s']-r['truth']['forward_m_s']) for r in valid]))
        return result
    domain=[r for r in rows if 20<=r['truth']['endpoint_speed_m_s']<=90]
    matched=[r for r in domain if r['dense']['valid'] and r['lk']['valid']]
    report=dict(schema='apex-v2-motion-registration-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
                scope='One newly authorized REQUIRED t1/516237 diagnostic replay of unchanged frozen baseline. No holdout/new seed; capture truth never passed to policy.',
                capture=dict(track_id=1,seed=516237,finished=receipt['finished'],lapTimeMs=receipt['lapTimeMs'],steps=receipt['steps'],simulator_resets=1),
                algorithm='Discrete four-tick endpoint-weighted pixel HUD yaw fixes anisotropic image rotation; normalized correlation initializes translation; ECC refines translation only. Gradient minimum eigenvalue and photometric correlation gate availability; unknown returns None.',
                confidence=dict(min_gradient_eigenvalue=1e-5,min_coarse_correlation=.5,min_ecc_correlation=.75,max_pixel_speed_difference_m_s=15.),
                metrics=dict(all={k:stats(rows,k) for k in ['dense','lk','lk_then_dense']},speed20to90={k:stats(domain,k) for k in ['dense','lk','lk_then_dense']},matched_valid_speed20to90={k:stats(matched,k) for k in ['dense','lk']}),
                limitations=['One consumed REQUIRED road under one unchanged policy, not generalization or controller-performance evidence.',
                             'Truth displacement is expressed in final body axes and averaged over the frame interval; endpoint slip is reported separately.',
                             'Four-tick endpoint-weighted HUD yaw approximates integrated yaw under a linear change assumption; nonlinear angular changes can violate this.',
                             'High correlation can be misleading on repeating road texture; confidence is heuristic, not a calibrated error probability.',
                             'Image registration assumes a rigid static world and fixed post-warmup zoom; car and HUD are excluded where practical.',
                             'OpenCV timings reflect this concurrent research process, not an official evaluator resource claim.'],
                rows=rows,evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),candidate,folder/'receipt.json',folder/'trace.jsonl',folder/'observations.npz',folder/'truth.json']})
    history=[]
    for name in ('analysis-initial.json','analysis-discrete-yaw.json'):
        historical=folder/name
        if historical.exists():
            prior=json.loads(historical.read_text())
            history.append(dict(artifact=str(historical),sha256=hashlib.sha256(historical.read_bytes()).hexdigest(),metrics=prior['metrics']))
    report['same_capture_research_history']=history
    report['selection_disclosure']='Discrete quadrature and pixel-speed confidence were investigated after inspecting this one capture; all final accuracy is in-sample diagnostic evidence, not independent validation.'
    report['recommendation']='Retain LK when valid; dense registration can supply explicit-confidence interval-motion fallback, pending independent geometry and closed-loop evaluation. Do not treat unknown slip as zero.'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report['metrics'],indent=2))
    return report


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('capture');c.add_argument('--output-dir',required=True)
    a=sub.add_parser('analyze');a.add_argument('--capture-dir',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='capture':capture(args.output_dir)
    else:analyze(args.capture_dir,args.output)


if __name__=='__main__':main()
