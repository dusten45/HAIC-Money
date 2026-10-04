"""Read-only matched-progress Beam R3 versus frozen P1 speed decomposition.

Three completed required geometries only. Telemetry is diagnostic truth, never
an agent input. Progress-bin interpolation is approximate because tile progress
is quantized; these comparisons do not identify controller-change causality.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('/tmp/apex-final-frozen/required')
BEAM=Path('/tmp/apex-v2-beam-r3')
CELLS=[(1,516237),(2,644062),(3,1007)]
N=20


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory,track,seed):
    receipt=directory/f't{track}-s{seed}.json';trace=receipt.with_suffix('.jsonl')
    r=json.loads(receipt.read_text());rows=[json.loads(s) for s in trace.read_text().splitlines()]
    assert r['finished'] and r['damage']==0
    # columns: time, speed*time, gas*time, brake*time, |steer|*time,
    # integrated heading change, sampled distance, positive delta-v, negative dv.
    bins=np.zeros((N,9));previous_progress=0.;totals=np.zeros(9)
    weighted_acceleration=[];high_speed_time=0.;brake_time=0.;coast_time=0.
    speed_integral=0.;distance=0.;maxspeed=0.
    for row in rows:
        before,after=row['before'],row['after']
        full_dt=after['sim_time_s']-before['sim_time_s']
        dt=max(0.,min(after['sim_time_s'],r['finish_time_s'])-before['sim_time_s'])
        if dt<=0:continue
        fraction=dt/full_dt
        dv=fraction*(after['speed_m_s']-before['speed_m_s'])
        speed=before['speed_m_s']+.5*dv
        steer,gas,brake=row['action'];heading=fraction*abs(after['heading_rad']-before['heading_rad'])
        ds=fraction*float(np.linalg.norm(np.array(after['position'])-before['position']))
        end=float(after['info'].get('progress',previous_progress));start=previous_progress
        assert end+1e-8>=start
        weights=np.zeros(N)
        if end<=start+1e-12:weights[min(N-1,int(start*N))]=1.
        else:
            for j in range(N):weights[j]=max(0.,min(end,(j+1)/N)-max(start,j/N))/(end-start)
        values=np.array([dt,speed*dt,gas*dt,brake*dt,abs(steer)*dt,heading,ds,max(0.,dv),max(0.,-dv)])
        bins+=weights[:,None]*values;totals+=values
        high_speed_time+=dt*(speed>=70.);brake_time+=dt*(brake>.03)
        coast_time+=dt*(gas<.01 and brake<.01)
        speed_integral+=speed*dt;distance+=ds;maxspeed=max(maxspeed,before['speed_m_s'],after['speed_m_s'])
        if gas>.3 and brake==0 and abs(steer)<.07 and 30<=speed<=70:
            weighted_acceleration.append((dv/dt,dt))
        previous_progress=end
    assert abs(totals[0]-r['lapTimeMs']/1000)<1e-6
    acceleration=np.average([x[0] for x in weighted_acceleration],weights=[x[1] for x in weighted_acceleration]) if weighted_acceleration else None
    summary=dict(lap_s=r['lapTimeMs']/1000,finished=r['finished'],damage=r['damage'],
        terminal_trace_time_excluded_s=rows[-1]['after']['sim_time_s']-r['finish_time_s'],
        geometric_chord_distance_m=distance,speed_integral_distance_m=speed_integral,
        time_mean_speed=speed_integral/totals[0],max_speed=maxspeed,
        high_speed_ge70_fraction=high_speed_time/totals[0],braking_fraction=brake_time/totals[0],
        coasting_fraction=coast_time/totals[0],mean_gas=totals[2]/totals[0],mean_brake=totals[3]/totals[0],
        mean_absolute_steer=totals[4]/totals[0],total_positive_speed_change=totals[7],total_negative_speed_change=totals[8],
        straight_acceleration_gas_gt03_speed30_70=acceleration,straight_acceleration_selected_time_s=sum(x[1] for x in weighted_acceleration),
        receipt=str(receipt),receipt_sha256=sha(receipt),trace=str(trace),trace_sha256=sha(trace),source_sha256=r['provenance']['agent_sha256'])
    return summary,bins


def main():
    outcomes=[]
    for track,seed in CELLS:
        baseline,p=analyze(BASE,track,seed);beam,b=analyze(BEAM,track,seed)
        turn=p[:,5]/np.maximum(p[:,6],1e-6)
        order=np.argsort(turn);groups={'lower_turn_third':order[:7],'middle_turn_third':order[7:13],'higher_turn_third':order[13:]}
        group_rows={}
        for label,indices in groups.items():
            pp=p[indices].sum(axis=0);bb=b[indices].sum(axis=0)
            group_rows[label]=dict(bins=indices.tolist(),baseline_time_s=float(pp[0]),beam_time_s=float(bb[0]),time_saved_s=float(pp[0]-bb[0]),baseline_speed=float(pp[1]/pp[0]),beam_speed=float(bb[1]/bb[0]),baseline_brake=float(pp[3]/pp[0]),beam_brake=float(bb[3]/bb[0]))
        common=[]
        for i in range(N):
            common.append(dict(progress_start=i/N,progress_end=(i+1)/N,baseline_turn_per_m=float(turn[i]),
                baseline_time_s=float(p[i,0]),beam_time_s=float(b[i,0]),time_saved_s=float(p[i,0]-b[i,0]),
                baseline_speed=float(p[i,1]/p[i,0]),beam_speed=float(b[i,1]/b[i,0]),
                baseline_gas=float(p[i,2]/p[i,0]),beam_gas=float(b[i,2]/b[i,0]),
                baseline_brake=float(p[i,3]/p[i,0]),beam_brake=float(b[i,3]/b[i,0]),
                baseline_absolute_steer=float(p[i,4]/p[i,0]),beam_absolute_steer=float(b[i,4]/b[i,0])))
        # Symmetric multiplicative decomposition: T = distance / mean speed.
        # Attribution is algebraic, not a causal intervention.
        ratio_distance=beam['speed_integral_distance_m']/baseline['speed_integral_distance_m']
        ratio_speed=beam['time_mean_speed']/baseline['time_mean_speed']
        log_gain=np.log(baseline['lap_s']/beam['lap_s'])
        outcomes.append(dict(track_id=track,seed=seed,baseline=baseline,beam=beam,
            time_saved_s=baseline['lap_s']-beam['lap_s'],lap_reduction_fraction=1-beam['lap_s']/baseline['lap_s'],
            traveled_distance_ratio=ratio_distance,mean_speed_ratio=ratio_speed,
            algebraic_log_gain_share_distance=float(-np.log(ratio_distance)/log_gain),
            algebraic_log_gain_share_speed=float(np.log(ratio_speed)/log_gain),
            turn_groups=group_rows,progress_bins=common))
    output=dict(scope='Three matched completed required cells, tracks1–3; no new episodes. Track4 and consumed failures excluded prospectively.',
        analysis_sha256=sha(Path(__file__)),binning='Twenty fixed normalized tile-progress bins; split each decision interval linearly across crossed bins. Time/controls are duration weighted. Turning groups are per-track baseline observed body-heading change per traveled distance tertiles, not privileged road curvature.',
        limitations=['Tile-progress quantization limits fine temporal alignment;5% bins reduce but do not remove it.',
          'Terminal action intervals are trimmed to recorded actual finish timestamp; partial-interval distance/speed are linearly interpolated.',
          'Geometric distance sums80ms chords; speed-integral distance uses trapezoidal true speed samples.',
          'Body-yaw turning groups may include slip and obstacle maneuvers; they are not ground-truth road-shape labels.',
          'Same cells establish observed matched gains, not which controller component caused them. Three successes do not establish full-suite performance.'],
        cells=outcomes)
    saved=sum(c['time_saved_s'] for c in outcomes)
    by_turn={name:sum(c['turn_groups'][name]['time_saved_s'] for c in outcomes)
             for name in ('lower_turn_third','middle_turn_third','higher_turn_third')}
    output['aggregate']=dict(matched_cells=3,baseline_lap_sum_s=sum(c['baseline']['lap_s'] for c in outcomes),
        beam_lap_sum_s=sum(c['beam']['lap_s'] for c in outcomes),total_saved_s=saved,
        time_saved_by_turn_group_s=by_turn,
        higher_turn_saved_fraction=by_turn['higher_turn_third']/saved,
        middle_and_higher_turn_saved_fraction=(by_turn['middle_turn_third']+by_turn['higher_turn_third'])/saved)
    output['observations']=[
        'All three gains accompany higher mean physical speed and moderately shorter traveled distance.',
        'Most saved time occurs in middle/high baseline body-turning progress bins, with less braking and more coasting.',
        'Total speed reductions decrease on every track; mean gas is similar or modestly higher.',
        'Selected straight-line acceleration is not consistently higher, so stronger acceleration is not supported as the primary explanation.']
    output['causal_hypothesis']='Joint short-sequence planning preserves momentum through turns and avoids repeated braking/reacceleration; shorter routes contribute secondarily. Component ablations would be required to establish this mechanism causally.'
    for track,seed in CELLS:
        left=json.loads((BASE/f't{track}-s{seed}.json').read_text())
        right=json.loads((BEAM/f't{track}-s{seed}.json').read_text())
        assert left['provenance']['official_source_sha256']==right['provenance']['official_source_sha256']
    output['official_source_inventories_match']=True
    result=Path('agents/apex_2026/v2/results/beam-speed-gain.json');result.write_text(json.dumps(output,indent=2)+'\n')
    for c in outcomes:
        print('track',c['track_id'],'saved',round(c['time_saved_s'],2),'distance%',round(100*(c['traveled_distance_ratio']-1),2),'speed%',round(100*(c['mean_speed_ratio']-1),2),'speed share',round(c['algebraic_log_gain_share_speed'],3))
        print('turn groups',[(k,round(v['time_saved_s'],2),round(v['baseline_speed'],1),round(v['beam_speed'],1)) for k,v in c['turn_groups'].items()])
        print('pedals',[(k,round(c['baseline'][k],3),round(c['beam'][k],3)) for k in ['mean_gas','mean_brake','braking_fraction','coasting_fraction','high_speed_ge70_fraction','total_negative_speed_change','straight_acceleration_gas_gt03_speed30_70']])


if __name__=='__main__':main()
