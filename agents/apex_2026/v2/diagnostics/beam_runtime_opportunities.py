"""Read existing R6 receipts/profile only; no act calls, benchmarks or resets."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=Path('/tmp/apex-v2-beam-refined-r6');episodes=[];inputs={}
    for track,seed in [(1,516237),(3,4111953688)]:
        receipt=base/f't{track}-s{seed}.json';trace=receipt.with_suffix('.jsonl')
        r=json.loads(receipt.read_text());rows=[json.loads(x) for x in trace.read_text().splitlines()]
        assert r['status']=='completed' and r['provenance']['agent_sha256']=='cc0543c2bebbec0f2c1917a60b4e506cdd3da5bd562131e7090ae2bb55d65425'
        durations=np.array([x['inference_s'] for x in rows]);counts=np.array([x['policy_diagnostics']['collision_checked_raw_ticks'] for x in rows]);worst=rows[int(np.argmax(durations))]
        episodes.append(dict(track_id=track,seed=seed,actions=len(rows),lapTimeMs=r['lapTimeMs'],
            time_percentiles_s={str(p):float(np.percentile(durations,p)) for p in [50,90,95,99,100]},
            actions_over4s=int(np.sum(durations>4)),timeout_count=r['act_timeout_count'],
            remaining_margin_to5s=5-float(durations.max()),maximum_step=worst['step'],
            ticks_min=int(counts.min()),ticks_max=int(counts.max()),
            ticks_at_maximum_latency=worst['policy_diagnostics']['collision_checked_raw_ticks']))
        inputs[str(receipt)]=sha(receipt);inputs[str(trace)]=sha(trace)
    profile_path=Path('agents/apex_2026/v2/results/beam-profile.json');profile=json.loads(profile_path.read_text());inputs[str(profile_path)]=sha(profile_path)
    gains_path=Path('agents/apex_2026/v2/results/beam-speed-gain.json');gains=json.loads(gains_path.read_text());inputs[str(gains_path)]=sha(gains_path)
    target_context=[]
    for cell in gains['cells']:
        b=cell['beam'];distance=b['speed_integral_distance_m']
        target_context.append(dict(track_id=cell['track_id'],source='historical R3, not current R6',observed_lap_s=b['lap_s'],
            observed_distance_m=distance,observed_mean_speed=b['time_mean_speed'],observed_max_speed=b['max_speed'],
            mean_speed_needed_for13s_same_distance=distance/13,
            mean_speed_needed_for10s_same_distance=distance/10,
            constant100m_s_distance_only_lower_bound_s=distance/100))
    result=dict(scope='Read-only existing initial2 R6 episodes and older R3profile; no new model calls, heavy benchmarks, resets or runtime edits.',
        analysis_sha256=sha(Path(__file__)),input_sha256=inputs,episodes=episodes,
        worst_case_prediction_tick_accounting=dict(beam_expansion=2052,feedback_completions=6300,temporal_refinement=224,total=8576,
            assumption='Every pursuit lattice has3 distinct steering choices; depth8,width8,3pedals. Clipping can reduce actual count.',refinement_fraction=224/8576),
        profile_scope='One older R3 saved frame: footprint was25point sampling, not current full-area+1pixel SAT; its fractions cannot be asserted for R6.',
        existing_profile_seconds=dict(total=3.001,completion_cumulative=2.238,physics_step_cumulative=1.679,
            old_sampled_footprint_cumulative=.533,projection_cumulative=.266),
        existing_projection_cache_evidence=profile['projection_cache_probe'],
        parity_limit='The2000case result covers a local projection function only; no whole-agent exact-behavior claim or currentR6 speedup is established.',
        exact_preserving_opportunities=[
            dict(priority=1,change='Cache per-observation projection geometry: segment starts/deltas, squared denominators, arclength increments and math.atan2 headings.',
                 reason='These arrays are unchanged across thousands of projection calls; existing2000-case static probe has bitwise parity and1.527x local speedup.',
                 bounded_expectation='Oldprofile projection was8.9% oftotal; applying its measured local ratio implies only about3.1% end-to-end saving there. CurrentR6 benefit unmeasured.',
                 validation='Preserve numerical operation ordering/math.atan2; bitwise projection outputs and all saved action/full-state parity before a new source freeze.'),
            dict(priority=2,change='Keep cheap current circular broadphase; when it finds possible unsafe cells, reuse exact transformed fixture polygons for a tighter global pixel AABB and integral-image rejection before SAT.',
                 reason='Circle bounds overestimate long hull width, sending safe near-edge poses into raster-cell SAT. Cells outside all exact polygon AABBs cannot contribute to current count.',
                 bounded_expectation='Potentially helps expensive edge/obstacle states; no measured speedup or implementation yet.',
                 validation='Use identical polygon transforms, include boundary-touch cells and outside-image cells, preserve union counting; exact unsafe-count parity and whole-act/state parity on saved boundary/rotation/wheel cases.'),
            dict(priority=3,change='Only after evaluation finishes, profile one saved slow R6 input under controlled single-thread resources to attribute walltime between current SAT, physics and completion work.',
                 reason='Current latency tails cannot be explained using older sampled-footprint profile; avoid optimizing from stale fractions.',
                 bounded_expectation='Measurement only, no new driving reset. Full-state transition memoization is not approved without hidden solver-state audit and observed cache hits.')],
        structural_target_context=target_context,
        structural_followups=[
            'For10–13s, improving wall-clock inference alone does not lower simulated lap time. It supplies resource headroom while preserving controls.',
            'Historical R3 higher/middle-turn bins carry87% of its speed gain. Remaining gains need verified corner-entry/exit trajectories and less route distance, not merely more straight throttle.',
            'On the observed R3track2 route,13s requires87.57m/s mean, almost its88.12m/s peak;10s would require113.85m/s mean, above the public100m/s translation ceiling. This is conditional on that route, not a global impossibility claim.',
            'A bounded future structural study can separate target-envelope limits from search limits by offline model-feasible path/speed plans, retaining RPM/state uncertainty and collision checks. Do not raise lateral/braking gains blindly.',
            'Lowest-positive HUD speed/RPM bins remain ambiguous; propagate an interval/confidence at near-rest instead of repeatedly injecting estimated wheel energy. Keep this separate from unchanged R6 until its frozen evaluation finishes.'],
        conclusion='Observed max4.385s leaves0.615s(12.3%) below5s, so there is limited but positive measured headroom, not a worst-case guarantee. Prefer pure geometry caching/exact collision broadphase over reducing search width/depth, skipping collision ticks or approximate dynamics.')
    Path('agents/apex_2026/v2/results/beam-r6-runtime-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=episodes,worst_case_prediction_tick_accounting=result['worst_case_prediction_tick_accounting'],structural_target_context=target_context),indent=2))


if __name__=='__main__':main()
