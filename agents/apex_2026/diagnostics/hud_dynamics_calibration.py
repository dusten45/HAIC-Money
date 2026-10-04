"""Public-renderer-only HUD calibration: no simulator, reset, or hidden state input.

The pure decode_hud function reads an ordinary 84x84 grayscale observation.
Synthetic privileged values are used solely to measure the public renderer.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
YAW_COEFFICIENTS = (.4580150260806972, .024360133436811916)
STEER_COEFFICIENTS = (.018757917750222175, .0019537197954649153)


def signals(frame):
    yaw = np.asarray(frame)[74:82, 52:75]
    steer = np.asarray(frame)[74:82, 31:52]
    return (float(yaw[:, :11].sum() - yaw[:, 11:].sum()), float(yaw.sum()),
            float(steer[:, :11].sum() - steer[:, 11:].sum()), float(steer.sum()))


def decode_hud(frame):
    """Signed hull yaw rad/s and actual front-left joint angle rad, current frame.

    Calibration is valid for yaw [-6,6], angle [-.4,.4]. Positive rendered values
    extend LEFT. Use the last frame, not a temporal mean, for current dynamics.
    Zero total mass means zero; nonzero tiny bars have quantization ambiguity.
    """
    yaw, yaw_total, steer, steer_total = signals(frame)
    return {'yaw_rate': YAW_COEFFICIENTS[0]*yaw+YAW_COEFFICIENTS[1] if yaw_total else 0.,
            'wheel_angle': STEER_COEFFICIENTS[0]*steer+STEER_COEFFICIENTS[1] if steer_total else 0.}


def render_hud(yaw, steer, speed=0., rpm=0.):
    """Exact public rendering and preprocessing; never instantiate CarRacing."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import pygame
    from core.vendor.car_racing import CarRacing, WINDOW_W, WINDOW_H, STATE_W, STATE_H
    from env_wrapper import image_preprocessing
    fake = SimpleNamespace(
        surf=pygame.Surface((WINDOW_W, WINDOW_H)),
        car=SimpleNamespace(hull=SimpleNamespace(linearVelocity=(float(speed),0.),
                                                angularVelocity=float(yaw)),
                            wheels=[SimpleNamespace(omega=float(rpm),
                                    joint=SimpleNamespace(angle=float(steer))) for _ in range(4)]))
    CarRacing._render_indicators(fake, WINDOW_W, WINDOW_H)
    rgb = CarRacing._create_image_array(fake, fake.surf, (STATE_W, STATE_H))
    return image_preprocessing(rgb)


def metrics(rows):
    result = {}
    for name in ('yaw_rate', 'wheel_angle'):
        errors=np.array([r['predicted'][name]-r['true'][name] for r in rows])
        truth=np.array([r['true'][name] for r in rows])
        result[name]={'mae':float(np.abs(errors).mean()), 'max_abs_error':float(np.abs(errors).max()),
                      'bias':float(errors.mean()),
                      'positive_bias':float(errors[truth>0].mean()) if np.any(truth>0) else None,
                      'negative_bias':float(errors[truth<0].mean()) if np.any(truth<0) else None}
    return result


def calibrate():
    calibration={}
    # Stagger paired signs off integer renderer endpoints to avoid fitting the
    # native raster's floor bias at an unusually aligned calibration grid.
    yaw_grid=(np.arange(240)+.37)*6/240
    steer_grid=(np.arange(160)+.37)*.4/160
    for key, values, index in [('yaw_rate',np.r_[-yaw_grid,yaw_grid],0),
                               ('wheel_angle',np.r_[-steer_grid,steer_grid],2)]:
        rows=[]
        for value in values:
            if abs(value)<1e-5:
                continue
            yaw,steer=(value,0.) if key=='yaw_rate' else (0.,value)
            signal=signals(render_hud(yaw,steer))[index]
            rows.append({'true':float(value),'signed_mass':signal})
        x=np.array([[r['signed_mass'],1.] for r in rows]);y=np.array([r['true'] for r in rows])
        coeff=np.linalg.lstsq(x,y,rcond=None)[0]
        calibration[key]={'coefficients':coeff.tolist(),'sample_count':len(rows),'samples':rows}
    rng=np.random.default_rng(20261004)
    rows=[]
    for yaw,steer,speed,rpm in zip(rng.uniform(-6,6,512),rng.uniform(-.4,.4,512),
                                  rng.uniform(0,100,512),rng.uniform(-200,400,512)):
        rows.append({'true':{'yaw_rate':float(yaw),'wheel_angle':float(steer)},
                     'predicted':decode_hud(render_hud(yaw,steer,speed,rpm))})
    nuisance=[]; invariance_max=0.
    for yaw in (-6,-3,-1,-.1,0,.1,1,3,6):
        for steer in (-.4,-.2,0,.2,.4):
            baseline=decode_hud(render_hud(yaw,steer))
            for speed in (0,50,100):
                for rpm in (-200,0,100,400):
                    decoded=decode_hud(render_hud(yaw,steer,speed,rpm))
                    invariance_max=max(invariance_max,max(abs(decoded[k]-baseline[k]) for k in decoded))
                    nuisance.append({'true':{'yaw_rate':yaw,'wheel_angle':steer},'predicted':decoded})
    mirror={}
    for key,values in [('yaw_rate',np.linspace(.01,6,240)),('wheel_angle',np.linspace(.001,.4,160))]:
        sums=[]
        for v in values:
            plus=(v,0) if key=='yaw_rate' else (0,v)
            minus=(-v,0) if key=='yaw_rate' else (0,-v)
            sums.append(decode_hud(render_hud(*plus))[key]+decode_hud(render_hud(*minus))[key])
        mirror[key]={'mean_signed_sum':float(np.mean(sums)),
                     'max_abs_sum':float(np.max(np.abs(sums))), 'pairs':len(sums)}
    examples=[]
    for yaw,steer in [(0,0),(.01,.001),(-.01,-.001),(.1,.01),(-.1,-.01),(1,.1),(-1,-.1),(6,.4),(-6,-.4)]:
        examples.append({'true':{'yaw_rate':yaw,'wheel_angle':steer},
                         'predicted':decode_hud(render_hud(yaw,steer,100,400))})
    return {'calibration':calibration,'random_validation':{'count':len(rows),'seed':20261004,
                'metrics':metrics(rows),'worst_examples':sorted(rows,key=lambda r:abs(r['predicted']['yaw_rate']-r['true']['yaw_rate']),reverse=True)[:5]},
            'nuisance_validation':{'count':len(nuisance),'speeds':[0,50,100],
                'wheel_omega_values':[-200,0,100,400],'max_decoder_change':invariance_max,
                'metrics':metrics(nuisance)},'mirror_validation':mirror,'examples':examples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Started receipt guards exclusive outputs before any diagnostic work.
    result={'status':'started','started_at':datetime.now(timezone.utc).isoformat(),
            'simulator_resets':0,'driving_episodes':0,'scope':'synthetic public HUD renderer only'}
    with args.output.open('x') as f:
        json.dump(result,f)
    try:
        result.update(calibrate())
        result.update(status='completed',decoder={'yaw_coefficients':YAW_COEFFICIENTS,
            'steer_coefficients':STEER_COEFFICIENTS,'rows':[74,82],
            'yaw_left_right_columns':[[52,63],[63,75]],
            'steer_left_right_columns':[[31,42],[42,52]],
            'zero_rule':'zero total crop intensity => exactly 0',
            'units':['yaw rad/s','front-left joint radians'],
            'input':'current 84x84 grayscale frame only'},
            provenance={'python':platform.python_version(),
                'packages':{p:importlib.metadata.version(p) for p in ['numpy','pygame','opencv-python']},
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__),ROOT/'core/vendor/car_racing.py',ROOT/'env_wrapper.py']}})
    except BaseException as error:
        result.update(status='error',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['ended_at']=datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['random_validation','nuisance_validation','mirror_validation']},indent=2))


if __name__=='__main__':
    main()
