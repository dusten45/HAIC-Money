"""Offline fixed-action pedal residuals: HUD wheel plus bounded steering motor.

Reads only existing P1 REQUIRED traces. No environment or physics simulator is
constructed. Privileged speed is used only in a separately labeled diagnostic
condition, never in a runtime agent.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

A=2735.0620191784787
B=303.8957799087198
D=.00031570330627447987
MODES=('command','clipped_command','held_hud_wheel','motor_before_force','motor_after_force')


def motor_tick(wheel,command):
    """Right-positive convention for both command and decoded HUD wheel.

    Kinematic approximation to the official motor target; Box2D joint inertia,
    force limits and damage multipliers are deliberately not simulated.
    """
    return float(np.clip(wheel+.02*np.clip(50*(command-wheel),-3,3),-.4,.4))


def predict(speed,command,gas,brake,throttle,wheel,mode):
    v=float(speed);g=float(throttle);delta=float(wheel)
    for _ in range(4):
        g+=min(float(gas)-g,.1)
        if mode=='command':angle=command
        elif mode=='clipped_command':angle=float(np.clip(command,-.4,.4))
        elif mode=='held_hud_wheel':angle=wheel
        elif mode=='motor_before_force':
            delta=motor_tick(delta,command);angle=delta
        elif mode=='motor_after_force':angle=delta
        else:raise ValueError(mode)
        v=max(0.,v+.02*(A*g/(v+2.7)-B*brake-D*np.tan(angle)**2*v**3))
        if mode=='motor_after_force':delta=motor_tick(delta,command)
    return float(v),float(delta)


def describe(values):
    x=np.asarray(values,float)
    if not len(x):return dict(n=0)
    return dict(n=len(x),bias=float(x.mean()),mae=float(np.abs(x).mean()),
                rmse=float(np.sqrt(np.mean(x*x))),p90_abs=float(np.percentile(abs(x),90)),
                max_abs=float(np.max(abs(x))))


def self_checks(parent_source):
    assert abs(motor_tick(0.,.8)-.06)<1e-12
    assert motor_tick(.39,.8)==.4
    assert abs(motor_tick(.1,.12)-.12)<1e-12
    assert abs(motor_tick(.3,-.3)-.24)<1e-12
    for delta,command in [(0.,.8),(.1,.12),(.3,-.3),(.39,.8)]:
        assert abs(motor_tick(delta,command)+motor_tick(-delta,-command))<1e-12
    spec=importlib.util.spec_from_file_location('actuator_residual_parent',parent_source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for v,c,g,b,t in [(50,.1,.3,0,0),(70,-.8,0,.2,.8),(35,.03,.5,0,.4)]:
        expected=m.Agent._predict_pedals(v,c,g,b,t)[0]
        assert abs(predict(v,c,g,b,t,0,'command')[0]-expected)<1e-12
    return dict(motor_direction_rate_limit_and_symmetry=True,parent_predictor_exact_numeric_parity=True)


def analyze(trace_dir,output,capture_dir=None):
    root=Path(__file__).resolve().parents[4];parent=root/'agents/apex_2026/results/pedal_sources/aab71b842b07.py'
    checks=self_checks(parent);folder=Path(trace_dir);rows=[];evidence={};source_hash=hashlib.sha256(parent.read_bytes()).hexdigest()
    assert source_hash=='aab71b842b07d69d7fa01556e40626c606b2ac2d70823987e3dba733e65ff410'
    for tid,seed in [(1,516237),(2,644062),(3,1007),(4,18800)]:
        p=folder/f't{tid}.jsonl';rp=p.with_suffix('.json');receipt=json.loads(rp.read_text())
        assert (receipt['track_id'],receipt['seed'])==(tid,seed) and receipt['provenance']['agent_sha256']==source_hash
        records=[json.loads(s) for s in p.read_text().splitlines()];throttle=0.
        for i,r in enumerate(records):
            q=r['policy_diagnostics'];command,gas,brake=map(float,r['action']);initial=throttle
            for _ in range(4):throttle+=min(gas-throttle,.1)
            if not q.get('valid'):continue
            assert abs(throttle-q['internal_throttle'])<1e-6
            if not np.isclose(r['after']['sim_time_s']-r['before']['sim_time_s'],.08):continue
            wheel=q.get('wheel_angle')
            if wheel is None:continue
            actual=float(r['after']['speed_m_s']);truth_speed=float(r['before']['speed_m_s'])
            predictions={};errors={}
            for condition,speed in [('pixel_speed',q['speed']),('true_initial_speed_diagnostic_only',truth_speed)]:
                predictions[condition]={mode:predict(speed,command,gas,brake,initial,wheel,mode)[0] for mode in MODES}
                errors[condition]={mode:value-actual for mode,value in predictions[condition].items()}
            predicted_wheel=wheel
            for _ in range(4):predicted_wheel=motor_tick(predicted_wheel,command)
            next_wheel=(records[i+1]['policy_diagnostics'].get('wheel_angle') if i+1<len(records) and records[i+1]['policy_diagnostics'].get('valid') else None)
            wheel_errors=None if next_wheel is None else {"motor":predicted_wheel-next_wheel,"command":command-next_wheel,"held":wheel-next_wheel}
            rows.append(dict(track_id=tid,seed=seed,step=r['step'],true_initial_speed=truth_speed,pixel_initial_speed=q['speed'],actual_next_speed=actual,command=command,hud_wheel=wheel,gas=gas,brake=brake,initial_throttle=initial,flow_valid=q['flow_valid'],pixel_slip=q['slip'],allocator_mode=q['allocator_mode'],damage=r['before']['info'].get('damage'),prediction_errors_m_s=errors,predicted_speeds_m_s=predictions,wheel_prediction_error_rad=wheel_errors))
        for path in [p,rp]:evidence[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    eligible=[r for r in rows if 20<=r['true_initial_speed']<=90]
    groups={'all_20to90':eligible,
            'command_outside_sampled_range':[r for r in eligible if abs(r['command'])>.2],
            'command_inside_sampled_range':[r for r in eligible if abs(r['command'])<=.2],
            'wheel_command_gap_gt_06':[r for r in eligible if abs(r['command']-r['hud_wheel'])>.06],
            'flow_valid_low_slip':[r for r in eligible if r['flow_valid'] and abs(r['pixel_slip'])<=np.deg2rad(5)],
            'flow_unknown':[r for r in eligible if not r['flow_valid']]}
    for tid in range(1,5):groups[f'track_{tid}']=[r for r in eligible if r['track_id']==tid]
    metrics={}
    for name,subset in groups.items():
        conditions={}
        for condition in ['pixel_speed','true_initial_speed_diagnostic_only']:
            conditions[condition]={mode:describe([r['prediction_errors_m_s'][condition][mode] for r in subset]) for mode in MODES}
            for mode in MODES[1:]:
                differences=[abs(r['prediction_errors_m_s'][condition][mode])-abs(r['prediction_errors_m_s'][condition]['command']) for r in subset]
                conditions[condition][mode]['paired_abs_error_change_mean']=float(np.mean(differences)) if differences else None
                conditions[condition][mode]['paired_improved_rows']=sum(d < -1e-9 for d in differences)
                conditions[condition][mode]['paired_worsened_rows']=sum(d > 1e-9 for d in differences)
        metrics[name]=dict(n=len(subset),speed_residual=conditions,wheel_residual={mode:describe([r['wheel_prediction_error_rad'][mode] for r in subset if r['wheel_prediction_error_rad'] is not None]) for mode in ['motor','command','held']})
    supplemental=None
    if capture_dir is not None:
        capture_path=Path(capture_dir)
        captured=np.load(capture_path/'observations.npz')['observations']
        capture_rows=[json.loads(x) for x in (capture_path/'trace.jsonl').read_text().splitlines()]
        capture_receipt=json.loads((capture_path/'receipt.json').read_text())
        assert (capture_receipt['track_id'],capture_receipt['seed'])==(1,516237)
        # The baked frozen agent must reproduce the same archived P1 actions.
        parent_rows=[json.loads(x) for x in (folder/'t1.jsonl').read_text().splitlines()]
        assert len(capture_rows)==len(parent_rows)
        assert all(a['action']==b['action'] for a,b in zip(capture_rows,parent_rows))
        residuals={};input_errors={};internal=0.
        for r in capture_rows:
            q=r['policy_diagnostics'];command,gas,brake=r['action'];initial=internal
            for _ in range(4):internal+=min(gas-internal,.1)
            if not q.get('valid') or not 20<=r['before']['speed_m_s']<=90:continue
            if not np.isclose(r['after']['sim_time_s']-r['before']['sim_time_s'],.08):continue
            frame=captured[r['step']-1,-1]
            current=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0,100))
            for condition,v in [('two_frame_mean',q['speed']),('current_frame',current),('true_initial_diagnostic_only',r['before']['speed_m_s'])]:
                input_errors.setdefault(condition,[]).append(v-r['before']['speed_m_s'])
                for mode in ['command','motor_after_force']:
                    error=predict(v,command,gas,brake,initial,q['wheel_angle'],mode)[0]-r['after']['speed_m_s']
                    residuals.setdefault(condition+'_'+mode,[]).append(error)
        supplemental=dict(scope='Previously authorized one-episode motion-registration capture, reused offline; zero additional resets. Its actions exactly match P1 required track1 archive.',
                          speed_input_errors_m_s={k:describe(v) for k,v in input_errors.items()},
                          speed_prediction_errors_m_s={k:describe(v) for k,v in residuals.items()})
        for p in [capture_path/'observations.npz',capture_path/'trace.jsonl',capture_path/'receipt.json']:
            evidence[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__),parent,root/'core/vendor/car_dynamics.py',root/'core/vendor/car_racing.py',root/'env_wrapper.py']:
        evidence[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(schema='apex-v2-actuator-residuals-v1',recorded_at=datetime.now(timezone.utc).isoformat(),scope='Offline one-step predictions on recorded P1 REQUIRED4 actions. No new environment, reset, candidate edit, parameter fitting or action selection.',self_checks=checks,action_count=len(rows),eligible_count=len(eligible),source_parent_sha256=source_hash,
        steering_semantics='Environment passes -action[0] to Car.steer; HUD decoder already negates front joint angle. Thus command and decoded wheel in this calculation are both right-positive. Motor approximation delta+=.02*clip(50*(command-delta),-3,3), bounded to±.4.',
        variants={'command':'Original P1: same commanded angle used for all four longitudinal drag calculations, no actuator lag/limit.',
                  'clipped_command':'Same command approximation, bounded±.4; isolates joint-limit extrapolation.',
                  'held_hud_wheel':'Current pixel wheel angle held across the whole80ms; simple state-only reference.',
                  'motor_before_force':'Update wheel approximation before each drag calculation.',
                  'motor_after_force':'Use current wheel for force, then advance wheel. Closer to Car.step/world.Step ordering, still not actual Box2D.'},
        metrics=metrics,previous_capture_speed_input_comparison=supplemental,recommendation='Do not promote the actuator motor model on these residuals: wheel-state prediction improves but pixel-input speed prediction does not. Diagnose speed-decoder bias and temporal alignment before adding actuator complexity. Any controller benefit requires separately authorized matched driving evidence.',worst_pixel_speed_examples=sorted(eligible,key=lambda r:abs(r['prediction_errors_m_s']['pixel_speed']['command']),reverse=True)[:12],largest_actuator_effect_examples=sorted(eligible,key=lambda r:abs(r['predicted_speeds_m_s']['pixel_speed']['motor_after_force']-r['predicted_speeds_m_s']['pixel_speed']['command']),reverse=True)[:12],limitations=['All four traces are consumed successful P1 REQUIRED laps; no new geometry or closed-loop performance claim.',
            'The motor target speed is not guaranteed realized joint speed: motor torque, body/wheel angular motion, tire contacts and joint constraints can change it.',
            'Steady rolling drag was calibrated after conditioning a constant command; substituting a transient wheel angle does not validate that drag law under yaw/slip transients.',
            'HUD wheel is one front joint measurement, not full front axle pose. Its quantization and timing remain measurement errors.',
            'True-initial-speed condition is diagnostic only and removes much of pixel-speed lag; it is unavailable to runtime policy.',
            'Both tick-order variants are disclosed; choosing the better on these same traces would be in-sample selection.',
            'Full80ms rows only; terminal partial-action rows are excluded. Target20–90m/s is selected using diagnostic true initial speed.'],evidence_sha256=evidence)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    for group in ['all_20to90','command_outside_sampled_range','wheel_command_gap_gt_06']:
        print(group,'n',metrics[group]['n'])
        for condition,x in metrics[group]['speed_residual'].items():print(condition,{k:round(v['mae'],5) for k,v in x.items() if v['n']})
    print('wheel residual',metrics['all_20to90']['wheel_residual'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trace-dir',default='/tmp/apex-pedal-p1');parser.add_argument('--output',required=True);parser.add_argument('--capture-dir');args=parser.parse_args();analyze(args.trace_dir,args.output,args.capture_dir)
