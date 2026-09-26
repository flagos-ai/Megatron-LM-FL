#!/usr/bin/env python3
import ast,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parent; spec=importlib.util.spec_from_file_location('audit',ROOT/'audit_plugin_overrides.py'); audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
class Tests(unittest.TestCase):
 def test_signature_tracks_kinds_defaults_and_async(self):
  node=ast.parse('async def f(a, /, b: int = 2, *args, c=None, **kwargs) -> bool: pass').body[0]; s=audit.signature(node)
  self.assertEqual(s['kind'],'async-function'); self.assertEqual([x['kind'] for x in s['parameters']],['posonly','positional','vararg','kwonly','kwarg']); self.assertEqual(s['parameters'][1]['default'],'2')
 def test_control_flow_is_transparent_to_lexical_lookup(self):
  tree=ast.parse('if enabled:\n    class A:\n        def f(self): pass\n')
  self.assertIsNotNone(audit.find_qual(tree,['A','f']))
 def test_decorator_does_not_change_api_contract(self):
  a=audit.signature(ast.parse('@overridable\ndef f(x=1): pass').body[0]); b=audit.signature(ast.parse('def f(x=1): pass').body[0])
  self.assertEqual(audit.signature_contract(a),audit.signature_contract(b))
 def test_subclass_without_init_inherits_constructor_contract(self):
  node=ast.parse('class Child(pkg.Base):\n    pass').body[0]; sig=audit.signature(node)
  self.assertTrue(sig['inherits_constructor']); self.assertEqual(sig['bases'],['pkg.Base'])
 def test_overridable_site_keeps_lexical_class(self):
  original=audit.git; audit.git=lambda repo,*args,**kwargs:'class A:\n    @overridable\n    def f(self): pass\n'
  try:
   sites=audit.extract_overridable_sites(None,'r',[{'path':'pkg/mod.py','decorators':['overridable']}])
   self.assertEqual(sites[0]['target'],'pkg.mod.A.f'); self.assertEqual(sites[0]['method_key'],'A.f')
  finally: audit.git=original
 def test_method_key_matches_runtime_rule(self): self.assertEqual(audit.method_key('megatron.core.foo.Bar.run'),'Bar.run')
 def test_resolve_nested_method_from_git_cache(self):
  class Repo:
   def resolve(self): return self
  original=audit.git; audit.git=lambda repo,*args,**kwargs:'class A:\n    def f(self, x=1): pass\n'
  try:
   x=audit.resolve(Repo(),'r','pkg.mod.A.f',{}); self.assertEqual(x['status'],'found'); self.assertEqual(x['signature']['parameters'][1]['default'],'1')
  finally: audit.git=original
 def test_decisions_are_composed_without_changing_identity(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); audit_doc={'summary':{'registrations':1},'rows':[{'identity':'A.f@default','findings':['x'],'disposition':'manual','owner':None,'tests':[]}]}; decision={'decisions':[{'identity':'A.f@default','disposition':'approved-exception','owner':'team','tests':['test_f'],'reason':'adapter','reviewer':'r'}]}
   (root/'a.json').write_text(json.dumps(audit_doc)); (root/'d.json').write_text(json.dumps(decision))
   proc=subprocess.run(['python',str(ROOT/'apply_override_decisions.py'),'--audit',str(root/'a.json'),'--decisions',str(root/'d.json'),'--output',str(root/'o.json')],capture_output=True,text=True)
   self.assertEqual(proc.returncode,0,proc.stderr); self.assertEqual(json.loads((root/'o.json').read_text())['rows'][0]['identity'],'A.f@default')
if __name__=='__main__': unittest.main()
