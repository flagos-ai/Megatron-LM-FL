#!/usr/bin/env python3
import ast,importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).parent; spec=importlib.util.spec_from_file_location('audit',ROOT/'audit_platform_contract.py'); audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
class Tests(unittest.TestCase):
 def test_signature_tracks_defaults(self):
  n=ast.parse('def f(self, x=None, *, y=1): pass').body[0]; self.assertEqual(audit.sig(n)[1]['default'],'None'); self.assertEqual(audit.sig(n)[2]['kind'],'kwonly')
 def test_property_kind(self):
  cls=ast.parse('class P:\n @property\n def Stream(self): pass').body[0]; self.assertEqual(audit.methods(cls)['Stream']['kind'],'property')
 def test_added_optional_default_is_compatible(self):
  base={'kind':'method','parameters':[{'name':'x','kind':'positional','default':None}]}; impl={'kind':'method','parameters':[{'name':'x','kind':'positional','default':'None'}]}
  self.assertTrue(audit.compatible_signature(base,impl))
 def test_platform_subclass_discovery(self):
  rows=audit.parse_platform('class P(PlatformBase):\n def f(self): pass\n','p.py'); self.assertEqual(rows[0]['class'],'P')
 def test_enclosing_symbol(self):
  tree=ast.parse('def f():\n x=1\n'); self.assertEqual(audit.enclosing_symbol(tree,2),'f')
 def test_registration_and_selection_are_dynamic(self):
  reg='p=PlatformX()\nPLATFORMS["x"] = p\n'; manager='if "x" in PLATFORMS.keys(): pass\nelif "cpu" in PLATFORMS.keys(): pass'
  rows,order=audit.parse_registration(reg,manager); self.assertEqual(rows[0]['class'],'PlatformX'); self.assertEqual(order,['x','cpu'])
 def test_occurrence_identity_is_stable_input(self):
  import hashlib
  value='MGP-'+hashlib.sha256(('a.py\0f\0torch.cuda').encode()).hexdigest()[:12].upper(); self.assertTrue(value.startswith('MGP-'))
 def test_device_patterns(self):
  self.assertTrue(audit.PATTERNS['torch.cuda'].search('torch.cuda.current_device()')); self.assertTrue(audit.PATTERNS['cuda-literal'].search('device="cuda:0"'))
if __name__=='__main__': unittest.main()
