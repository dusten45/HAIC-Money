"""Tiny synthetic-only protocol tests; no real allocation metadata or episodes."""
import copy,json,tempfile,unittest
from pathlib import Path
from agents.apex_2026.v2.diagnostics import final_validation as v

class FinalValidationTests(unittest.TestCase):
 def setUp(self):
  required=[dict(track_id=t,seed=t) for t in range(1,5)]
  regression=[dict(track_id=t,seed=100+t*10+i) for t in range(1,5) for i in range(6)]
  self.benchmark=dict(required=required,consumed_regression=regression,matched_improvement_gate=dict(required_mean_lap_ms_at_most=18530,required_mean_must_be_strictly_better=True))
  self.manifest=dict(source_sha256={'agent.py':'a'*64},config={})
  self.rows=[self.row(cell) for cell in required+regression];self.repeats=[self.row(cell) for cell in required]
 def row(self,cell):
  return dict(**cell,status='completed',finished=True,lapTimeMs=12000,start_t=1.02,finish_time_s=13.02,end_t=13.10,resource_eligible=True,
    agent_path='/frozen/agent.py',config={},started_at='2026-10-04T20:01:00+00:00',ended_at='2026-10-04T20:02:00+00:00',
    provenance=dict(agent_sha256='a'*64,evaluator_sha256='e'*64,official_source_sha256={'core/x':'b'*64},
      runtime_source_sha256_before={'/frozen/agent.py':'a'*64},runtime_source_sha256_after={'/frozen/agent.py':'a'*64}))
 def check(self):
  return v.validate_fresh_evidence(self.benchmark,self.manifest,self.rows,self.repeats,pushed_at='2026-10-04T20:00:00+00:00',expected_evaluator_sha256='e'*64,expected_official_sha256={'core/x':'b'*64},action_parity={'exact':False})
 def test_complete_gate_and_parity_is_report_only(self):
  result=self.check();self.assertTrue(result['evidence_gate_passed']);self.assertFalse(result['repeat_action_parity']['exact'])
 def test_stale_and_changed_sources_fail_closed(self):
  for mutate in (lambda row:row.update(started_at='2026-10-04T19:00:00+00:00'),lambda row:row['provenance'].update(evaluator_sha256='other'),lambda row:row['provenance'].update(official_source_sha256={}),lambda row:row['provenance']['runtime_source_sha256_after'].update({'/frozen/agent.py':'changed'})):
   before=copy.deepcopy(self.rows);mutate(self.rows[0])
   with self.assertRaises(ValueError):self.check()
   self.rows=before
 def test_missing_duplicate_failure_and_resource_receipts(self):
  original=copy.deepcopy(self.rows);self.rows.pop();self.assertFalse(self.check()['evidence_gate_passed'])
  self.rows=copy.deepcopy(original)+[copy.deepcopy(original[0])]
  with self.assertRaises(ValueError):self.check()
  self.rows=original;self.rows[0].update(finished=False,lapTimeMs=None,finish_time_s=None)
  self.assertFalse(self.check()['evidence_gate_passed'])
  self.rows[0]=self.row(self.benchmark['required'][0]);self.repeats[0]['resource_eligible']=False
  self.assertFalse(self.check()['evidence_gate_passed'])
 def test_mean_threshold_is_strict(self):
  for row in self.rows[:4]:row.update(lapTimeMs=18530,finish_time_s=19.55,end_t=19.6)
  self.assertFalse(self.check()['evidence_gate_passed'])
 def test_frozen_file_change_detected(self):
  from agents.apex_2026.protocol import fingerprint_candidate
  with tempfile.TemporaryDirectory(dir='/tmp') as directory:
   path=Path(directory)/'agent.py';path.write_text('x=1\n');manifest=fingerprint_candidate(directory,['agent.py'],{})
   self.assertTrue(v.verify_frozen_files(directory,manifest)['files_verified']);path.write_text('x=2\n')
   with self.assertRaises(ValueError):v.verify_frozen_files(directory,manifest)
 def inventory(self,root):
  sources={}
  for name in v.KNOWN_PROTOCOL_SOURCES:
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'reserved_training_seeds':[4001,4002]}));sources[name]=v.digest(path)
  return dict(format='apex-v2-reviewed-allocation-inventory-v1',reviewed=True,review_commit='synthetic',unresolved_sources=[],protocol_sources=sources,scope_limitations=['Synthetic metadata only'])
 def test_metadata_is_id_only_and_no_seed_values_returned(self):
  with tempfile.TemporaryDirectory(dir='/tmp') as directory:
   root=Path(directory);inventory=self.inventory(root)
   result=v.audit_allocation_metadata(root,inventory,self.benchmark,[{'track_id':1,'seed':5001}])
   self.assertEqual(result['private_seed_count'],2)
   self.assertNotIn('4001',json.dumps(result));self.assertFalse(result['allocation_performed'])
 def test_collisions_unknown_schema_and_inventory_omissions_block(self):
  with tempfile.TemporaryDirectory(dir='/tmp') as directory:
   root=Path(directory);inventory=self.inventory(root)
   for seed in (4001,1):
    with self.assertRaises(ValueError):v.audit_allocation_metadata(root,inventory,self.benchmark,[{'track_id':1,'seed':seed}])
   name=next(iter(inventory['protocol_sources']));path=root/name
   path.write_text(json.dumps({'unknown_geometry_seed_scheme':[9001]}));inventory['protocol_sources'][name]=v.digest(path)
   with self.assertRaisesRegex(ValueError,'unresolved'):v.audit_allocation_metadata(root,inventory,self.benchmark)
   inventory['protocol_sources'].pop(name)
   with self.assertRaisesRegex(ValueError,'omitted'):v.audit_allocation_metadata(root,inventory,self.benchmark)
 def test_metadata_hash_change_and_unresolved_review_block(self):
  with tempfile.TemporaryDirectory(dir='/tmp') as directory:
   root=Path(directory);inventory=self.inventory(root);name=next(iter(inventory['protocol_sources']))
   (root/name).write_text('{}')
   with self.assertRaises(ValueError):v.audit_allocation_metadata(root,inventory,self.benchmark)
   inventory['unresolved_sources']=['pending schema']
   with self.assertRaisesRegex(ValueError,'Unresolved'):v.audit_allocation_metadata(root,inventory,self.benchmark)

if __name__=='__main__':unittest.main()
