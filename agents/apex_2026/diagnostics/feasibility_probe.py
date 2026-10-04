"""Replay four REQUIRED geometry resets and synthetic HUD calibration.

Adapted from original measurement script probe.py; see results/feasibility.json
for original executed source and hash. This replay adaptation was not used for
the recorded measurements. Existing output files are never overwritten.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--release-dir', type=Path, required=True, help='Existing t1..t4.json and .jsonl release evidence')
    args = parser.parse_args()
    import sys,json,math,hashlib
    from datetime import datetime,timezone
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import numpy as np
    from agents.apex_2026.evaluate import make_environment
    from env_wrapper import image_preprocessing
    from core.vendor.car_racing import CarRacing, WINDOW_W, WINDOW_H,STATE_W,STATE_H
    import pygame
    from types import SimpleNamespace
    out=args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for name in ['summary.json', 'hud-calibration.json'] + [f'geometry-t{i}.json' for i in range(1,5)]:
     if (out/name).exists(): raise FileExistsError(out/name)
    for tid in range(1,5):
     for suffix in ('json','jsonl'):
      if not (args.release_dir/f't{tid}.{suffix}').is_file(): raise FileNotFoundError(args.release_dir/f't{tid}.{suffix}')
    summary=[]
    for tid,seed in [(1,516237),(2,644062),(3,1007),(4,18800)]:
     p=out/f'geometry-t{tid}.json'
     r=dict(status='started',track_id=tid,seed=seed,started_at=datetime.now(timezone.utc).isoformat(),agent_actions=0,purpose='geometry-only unchanged official reset')
     with p.open('x') as f: json.dump(r,f)
     env=make_environment()
     try:
      obs,info=env.reset(seed=seed,options={'track_id':tid})
      raw=env.unwrapped
      xy=np.array(raw.track)[:,2:4]; ds=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
      mass=float(raw.car.hull.mass+sum(w.mass for w in raw.car.wheels))
      r.update(status='completed',track=raw.track,tiles=len(xy),centerline_length=float(ds.sum()),car_mass=mass,start_t=raw.t,road_friction=sorted(set(float(t.road_friction) for t in raw.road)),wheelbase=3.24)
      p.write_text(json.dumps(r,indent=2))
      records=[json.loads(s) for s in (args.release_dir / f't{tid}.jsonl').read_text().splitlines()]
      pos=np.array([records[0]['before']['position']]+[q['after']['position'] for q in records]); speed=np.array([q['after']['speed_m_s'] for q in records]); result=json.loads((args.release_dir / f't{tid}.json').read_text())
      summary.append(dict(track_id=tid,tiles=len(xy),centerline_length=float(ds.sum()),car_mass=mass,force_accel_upper=1600/mass,path_length=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum()),speed_mean=float(speed.mean()),speed_max=float(speed.max()),speed_p10_p50_p90=np.percentile(speed,[10,50,90]).tolist(),lapTimeMs=result['lapTimeMs'],damage=result['damage'],progress=result['progress'],centerline_average_for13=float(ds.sum()/13),centerline_average_for10=float(ds.sum()/10)))
     finally: env.close()
    # Synthetic indicator-only rendering: no world reset, no physics mutations.
    cal=[]
    for v in np.arange(0,151,5):
     wheels=[SimpleNamespace(omega=0,joint=SimpleNamespace(angle=0)) for _ in range(4)]
     fake=SimpleNamespace(surf=pygame.Surface((WINDOW_W,WINDOW_H)),car=SimpleNamespace(hull=SimpleNamespace(linearVelocity=(float(v),0),angularVelocity=0),wheels=wheels))
     CarRacing._render_indicators(fake,WINDOW_W,WINDOW_H)
     img=CarRacing._create_image_array(fake,fake.surf,(STATE_W,STATE_H)); frame=image_preprocessing(img)
     mass=float(frame[77:83,10:13].sum())
     cal.append(dict(speed=float(v),mass=mass,line_estimate=float(np.clip((mass-.27)/.085,0,100)),rollout_estimate=float(np.clip(frame[74:81,11:12].mean(axis=1).sum()/.042,0,100))))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); (out/'hud-calibration.json').write_text(json.dumps(cal,indent=2))
    print(json.dumps(summary,indent=2)); print(json.dumps(cal[::2],indent=2))


if __name__ == "__main__":
    main()
