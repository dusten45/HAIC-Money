"""Calibrate four independent wheel omega HUD bars using public renderer only."""
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pygame
from core.vendor.car_racing import CarRacing, WINDOW_W, WINDOW_H, STATE_W, STATE_H
from env_wrapper import image_preprocessing


def render(omegas, speed=40., yaw=0., steer=0.):
    fake=SimpleNamespace(surf=pygame.Surface((WINDOW_W,WINDOW_H)),car=SimpleNamespace(
        hull=SimpleNamespace(linearVelocity=(float(speed),0.),angularVelocity=float(yaw)),
        wheels=[SimpleNamespace(omega=float(v),joint=SimpleNamespace(angle=float(steer))) for v in omegas]))
    CarRacing._render_indicators(fake, WINDOW_W, WINDOW_H)
    rgb=CarRacing._create_image_array(fake,fake.surf,(STATE_W,STATE_H))
    return image_preprocessing(rgb)


def features(frame):
    return np.r_[frame[74:82,14:25].sum(axis=0),1.]


def calibrate():
    rng=np.random.default_rng(842026)
    truth=rng.uniform(1,350,(1200,4))
    X=np.array([features(render(v)) for v in truth])
    coef=np.linalg.lstsq(X[:800],truth[:800],rcond=1e-6)[0]
    pred=X[800:]@coef
    errors=np.abs(pred-truth[800:])
    out=dict(scope='Synthetic public renderer; positive RPM [1,350] rad/s. Zero handled separately; reverse RPM not identified.',
             coefficients=coef.tolist(),train_count=800,test_count=400,
             mae=errors.mean(axis=0).tolist(),p95=np.quantile(errors,.95,axis=0).tolist(),max=errors.max(axis=0).tolist())
    Path('agents/apex_2026/v2/results/shadow-rpm.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


if __name__=='__main__':
    print(json.dumps(calibrate(),indent=2))
