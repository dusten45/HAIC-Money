import ast,hashlib,pathlib,tempfile,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
BEAM=ROOT/'results/beam_sources/aef90410f8ad.py'
SHADOW=ROOT/'results/shadow-sources/r2/shadow_physics.py'

class BeamPackageTests(unittest.TestCase):
 def builder(self):
  from agents.apex_2026.v2.diagnostics import build_beam_package
  return build_beam_package
 def test_hash_mismatch_fails_before_output(self):
  b=self.builder()
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   t=pathlib.Path(t);bad=t/'bad.py';bad.write_bytes(BEAM.read_bytes()+b'\n# changed\n')
   with self.assertRaises(ValueError):b.build(bad,SHADOW,t/'out')
   self.assertFalse((t/'out').exists())
 def test_shadow_hash_mismatch_fails_closed(self):
  b=self.builder()
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   t=pathlib.Path(t);bad=t/'bad.py';bad.write_bytes(SHADOW.read_bytes()+b'\n# changed\n')
   with self.assertRaises(ValueError):b.build(BEAM,bad,t/'out')
   self.assertFalse((t/'out').exists())
 def test_deterministic_and_no_project_import(self):
  b=self.builder()
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   t=pathlib.Path(t);b.build(BEAM,SHADOW,t/'a');b.build(BEAM,SHADOW,t/'b')
   self.assertEqual((t/'a/agent.py').read_bytes(),(t/'b/agent.py').read_bytes())
   self.assertEqual((t/'a/manifest.json').read_bytes(),(t/'b/manifest.json').read_bytes())
   source=(t/'a/agent.py').read_text();tree=ast.parse(source)
   names=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]
   self.assertEqual(names.count('_Road'),1);self.assertEqual(names.count('_ShadowRoad'),1)
   self.assertNotIn('from agents.',source)
   self.assertIn('MIT License',source);self.assertIn('Copyright (c) 2016 OpenAI',source)
 def test_binding_rename_preserves_shadow_ast_except_names(self):
  b=self.builder();original=SHADOW.read_text();renamed=b.rename_shadow_road(original)
  tree=ast.parse(renamed)
  for n in ast.walk(tree):
   if isinstance(n,ast.ClassDef) and n.name=='_ShadowRoad':n.name='_Road'
   if isinstance(n,ast.Name) and n.id=='_ShadowRoad':n.id='_Road'
  self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(original),include_attributes=False))
 def test_non_tmp_output_rejected(self):
  b=self.builder()
  with self.assertRaises(ValueError):b.build(BEAM,SHADOW,ROOT/'candidate')
 def test_output_symlink_rejected(self):
  b=self.builder()
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   t=pathlib.Path(t);b.build(BEAM,SHADOW,t/'a');(t/'b').mkdir()
   (t/'b/agent.py').symlink_to(t/'a/agent.py')
   with self.assertRaises(ValueError):b.build(BEAM,SHADOW,t/'b')

class BeamR6PackageTests(unittest.TestCase):
 def test_r6_selector_builds_pinned_source(self):
  from agents.apex_2026.v2.diagnostics import build_beam_package as b
  source=ROOT/'results/beam_refined_sources/cc0543c2bebb.py'
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   result=b.build(source,SHADOW,pathlib.Path(t)/'out',variant='r6')
   self.assertEqual(result['beam_sha256'],'cc0543c2bebbec0f2c1917a60b4e506cdd3da5bd562131e7090ae2bb55d65425')
 def test_selector_mismatch_and_unknown_fail_closed(self):
  from agents.apex_2026.v2.diagnostics import build_beam_package as b
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   output=pathlib.Path(t)/'out'
   for variant in ('r6','unapproved'):
    with self.assertRaises(ValueError):b.build(BEAM,SHADOW,output,variant=variant)
    self.assertFalse(output.exists())
 def test_changed_r6_hash_fails_closed(self):
  from agents.apex_2026.v2.diagnostics import build_beam_package as b
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   t=pathlib.Path(t);bad=t/'bad.py'
   bad.write_bytes((ROOT/'results/beam_refined_sources/cc0543c2bebb.py').read_bytes()+b'\n')
   with self.assertRaises(ValueError):b.build(bad,SHADOW,t/'out',variant='r6')
   self.assertFalse((t/'out').exists())
 def test_r3_default_artifact_is_unchanged(self):
  from agents.apex_2026.v2.diagnostics import build_beam_package as b
  with tempfile.TemporaryDirectory(dir='/tmp') as t:
   output=pathlib.Path(t)/'out';b.build(BEAM,SHADOW,output)
   self.assertEqual(hashlib.sha256((output/'agent.py').read_bytes()).hexdigest(),
       '311ca7521444cb56378525f33c8d72870117eeac586a7e24a430b587d2a192f5')

if __name__=='__main__':unittest.main()
