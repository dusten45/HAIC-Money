"""Replay synthetic car dynamics and fixed-centerline relaxation (no resets).

Adapted from original measurement script limits.py; see results/feasibility.json
for original executed source and hash. This replay adaptation was not used for
the recorded measurements. Existing output files are never overwritten.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--geometry-dir', type=Path, required=True, help='Existing geometry-t1..t4.json receipts')
    args = parser.parse_args()
    import sys,json,math
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import Box2D,numpy as np
    from core.vendor.car_dynamics import Car
    out=args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for name in ['synthetic-straight.json','centerline-relaxation.json']:
     if (out/name).exists(): raise FileExistsError(out/name)
    for tid in range(1,5):
     if not (args.geometry_dir/f'geometry-t{tid}.json').is_file(): raise FileNotFoundError(args.geometry_dir/f'geometry-t{tid}.json')
    # Synthetic isolated dynamics diagnostic. No CarRacing reset, no track cells.
    w=Box2D.b2World(gravity=(0,0)); car=Car(w,0,0,0)
    class Road: road_friction=1.
    road=Road()
    for wheel in car.wheels: wheel.tiles.add(road)
    rows=[]
    for i in range(1000):
     car.gas(1); car.brake(0); car.steer(0); car.step(.02); w.Step(.02,180,60)
     rows.append(dict(t=(i+1)*.02,speed=float(car.hull.linearVelocity.length),y=float(car.hull.position.y)))
    (out/'synthetic-straight.json').write_text(json.dumps(dict(kind='synthetic uniform road, isolated official Car dynamics, NOT an episode',samples=rows),indent=2))
    print('straight',[(r['t'],r['speed'],r['y']) for r in rows if round(r['t']*100)%100==0][:13])
    print('peak accel',max((rows[i]['speed']-rows[i-1]['speed'])/.02 for i in range(1,len(rows))))
    results=[]
    for tid in range(1,5):
     g=json.loads((args.geometry_dir/f'geometry-t{tid}.json').read_text()); xy=np.array(g['track'])[:,2:4]
     delta=np.roll(xy,-1,axis=0)-xy; ds=np.linalg.norm(delta,axis=1); angles=np.arctan2(delta[:,1],delta[:,0]); da=np.abs(np.angle(np.exp(1j*(angles-np.roll(angles,1))))); k=da/((ds+np.roll(ds,1))/2)
     a=1600/g['car_mass']; vmax=np.minimum(100,np.sqrt(a/np.maximum(k,1e-9)))
     results.append(dict(track_id=tid,centerline_constant100_time=ds.sum()/100,centerline_pointmass_optimistic_time=float(np.sum(ds/vmax)),min_pointmass_curve_cap=float(vmax.min()),max_curvature=float(k.max())))
    print(json.dumps(results,indent=2)); (out/'centerline-relaxation.json').write_text(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
