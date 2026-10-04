"""Compare a frozen motion model on cached cameras; no simulator is run."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','cache','truth','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    source_hash=sha(args.source)
    spec=importlib.util.spec_from_file_location('cached_motion',args.source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observer=module.Agent()
    observations=np.load(args.cache)['observations']
    truth=json.loads(args.truth.read_text())
    assert sha(args.cache)==truth['camera_cache_sha256']
    assert len(observations)==len(truth['rows'])
    rows=[]
    for observation,original in zip(observations,truth['rows']):
        began=time.perf_counter()
        estimate=observer._estimate_motion(observation)
        elapsed=1000*(time.perf_counter()-began)
        # Truth is used only after inference returned its camera-only estimate.
        error=np.arctan2(np.sin(estimate['heading']-original['true_heading']),
                         np.cos(estimate['heading']-original['true_heading']))
        rows.append({**original,**estimate,'heading_error':float(error),'latency_ms':elapsed})
    selected=[r for r in rows if r['confidence']>=.7 and r['true_speed']>=20.]
    large=[r for r in rows if r['true_speed']>=20. and abs(r['true_heading'])>.20]
    false=[r for r in selected if abs(r['heading'])>.20 and abs(r['true_heading'])<=.20]
    summary={'samples':len(rows),'confident_high_speed_samples':len(selected),
             'true_large_slip_samples':len(large),
             'large_slip_confident_samples':sum(r['confidence']>=.7 for r in large),
             'false_confident_large_slip_samples':len(false),
             'false_confident_large_slip_steps':[r['step'] for r in false],
             'motion_latency_max_ms':max(r['latency_ms'] for r in rows)}
    if selected:
        heading=np.abs([r['heading_error'] for r in selected])*180/np.pi
        lateral=np.abs([r['lateral_velocity']-r['true_lateral_velocity'] for r in selected])
        summary.update({'heading_mae_deg':float(np.mean(heading)),
                        'heading_p90_deg':float(np.quantile(heading,.9)),
                        'lateral_velocity_mae_mps':float(np.mean(lateral)),
                        'lateral_velocity_p90_mps':float(np.quantile(lateral,.9))})
    assert sha(args.source)==source_hash
    report={'type':'offline camera cache comparison; no new simulation episode',
            'source_sha256':source_hash,'camera_cache_sha256':sha(args.cache),
            'truth_receipt_sha256':sha(args.truth),'truth_enters_inference':False,
            'summary':summary,'rows':rows}
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
