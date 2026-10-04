exec(open('/tmp/apex-v2-beam-r4/preflight.py').read().split('for step in [')[0])
results=[]
for step in [134,135,136,137,138]:
 a=Agent();gas=0.
 for prev in rows[:step-1]:
  for _ in range(4):gas+=min(float(prev['action'][1])-gas,.1)
 a.throttle=gas;a.last_steer=rows[step-2]['action'][0];a.slip=rows[step-2]['policy_diagnostics']['predicted_slip'];saved={};original=a._search
 def capture(reference,arc,targets,free,initial):
  saved.update(reference=reference,arc=arc,targets=targets,free=free,initial=initial)
  return original(reference,arc,targets,free,initial)
 a._search=capture;act=a.act(obs[step-120]);node=dict(cost=0.,progress=0.,state=saved['initial'],sequence=[],last=a.last_steer);pursuit,_=a._pursuit(node,saved['reference'],saved['arc']);costs=[]
 for steer in np.unique(np.clip(pursuit+np.array([-.14,0,.14]),-.4,.4)):
  for gas,brake in ((.7,0.),(0.,0.),(0.,.65)):
   a.restore(a.shadow,saved['initial']);cs=[]
   for _ in range(4):
    x,y,h,_,_=a.shadow.step((steer,gas,brake));cs.append(a._footprint_cost(saved['free'],x,y,h))
   costs.append({'action':[float(steer),gas,brake],'unsafe_cells_by_tick':cs})
 for tag,sequence in [('old',rows[step-1]['policy_diagnostics']['beam_sequence']),('new',a.diagnostics['beam_sequence'])]:
  a.restore(a.shadow,saved['initial']);cs=[]
  for ac in sequence:
   for _ in range(4):
    x,y,h,_,_=a.shadow.step(ac);cs.append(a._footprint_cost(saved['free'],x,y,h))
  saved[tag+'_sequence_unsafe']=sum(cs)
 result={'step':step,'selected_action':act.tolist(),'first_action_candidates':costs,'old_sequence_unsafe':saved['old_sequence_unsafe'],'new_sequence_unsafe':saved['new_sequence_unsafe']};results.append(result);print(step,act.tolist(),saved['old_sequence_unsafe'],saved['new_sequence_unsafe'],[sum(c['unsafe_cells_by_tick']) for c in costs],flush=True)
Path('/tmp/apex-v2-beam-r4/candidates.json').write_text(json.dumps(results,indent=2))
