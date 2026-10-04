"""Synthetic public-renderer speed calibration; no driving resets."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pygame
from core.vendor.car_racing import CarRacing, WINDOW_W, WINDOW_H, STATE_W, STATE_H
from env_wrapper import image_preprocessing


def render(speed, background=0):
    surface = pygame.Surface((WINDOW_W, WINDOW_H))
    surface.fill((background, background, background))
    fake = SimpleNamespace(surf=surface, car=SimpleNamespace(
        hull=SimpleNamespace(linearVelocity=(float(speed), 0.), angularVelocity=0.),
        wheels=[SimpleNamespace(omega=80., joint=SimpleNamespace(angle=0.)) for _ in range(4)]))
    CarRacing._render_indicators(fake, WINDOW_W, WINDOW_H)
    rgb = CarRacing._create_image_array(fake, fake.surf, (STATE_W, STATE_H))
    return image_preprocessing(rgb)


def calibrate():
    train = np.linspace(0., 140., 1401)
    mass = np.array([render(v)[74:83, 10:13].sum() for v in train])
    coef = np.polyfit(mass, train, 1)
    rng = np.random.default_rng(56182026)
    truth = rng.uniform(0., 140., 1000)
    frames = [render(v, int(bg)) for v, bg in zip(truth, rng.integers(0, 256, len(truth)))]
    measured = np.array([f[74:83, 10:13].sum() for f in frames])
    new = np.polyval(coef, measured)
    old = np.array([(f[77:83, 10:13].sum()-.27)/.085 for f in frames])
    def errors(pred):
        e = pred-truth
        return dict(mae=float(np.abs(e).mean()), bias=float(e.mean()),
                    rmse=float(np.sqrt(np.mean(e*e))), maximum=float(np.abs(e).max()))
    result = dict(scope='Synthetic public HUD, zero environment resets; independently sampled test speeds and scene backgrounds.',
                  train_count=len(train), test_count=len(truth), speed_domain=[0,140],
                  crop=[74,83,10,13], coefficients=coef.tolist(), old=errors(old), calibrated=errors(new))
    Path('agents/apex_2026/v2/results/speed-hud.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    calibrate()
