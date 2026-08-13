#!/usr/bin/env python3
import ast,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parent; spec=importlib.util.spec_from_file_location('build',ROOT/'build_runtime_feature_ledger.py'); build=importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
class Tests(unittest.TestCase):
 def test_stage_classification(self):
  self.assertEqual(build.stage('megatron/training/arguments.py','training'),'configuration'); self.assertEqual(build.stage('tests/unit/test_x.py','tests-common'),'observing-test'); self.assertEqual(build.stage('x/checkpoint.py','invasive-runtime'),'output-state')
 def test_nested_symbols(self):
  rows,status=build.symbols('class A:\n def f(self): pass\n'); self.assertEqual(status,'parsed'); self.assertEqual([x['qualname'] for x in rows],['A','A.f'])
 def test_import_and_call_relations(self):
  r=build.relations('from pkg import f\ndef g(): return f()')
  self.assertEqual(r['imports'],['pkg:f']); self.assertEqual(r['calls'],['f'])
 def test_syntax_error_is_visible(self): self.assertEqual(build.symbols('def x(')[1],'syntax-error')
 def test_split_merge_decisions_preserve_coverage(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); candidate={'refs':{},'delta_facts':[{'delta_id':'D1','action':'REPLAY_FL'},{'delta_id':'D2','action':'REPLAY_FL'}]}; decision={'features':[{'feature_id':'F','feature_name':'f','delta_ids':['D2','D1'],'invariant':'i','target_change':'t','strategy':'REPLAY_FL_NARROW','owner':'o','stage_rationale':{'execution':'e'},'dependencies':[],'observing_tests':['test'],'reviewer':'r','approval_status':'approved'}]}; (root/'c').write_text(json.dumps(candidate)); (root/'d').write_text(json.dumps(decision)); p=subprocess.run(['python',str(ROOT/'apply_feature_decisions.py'),'--candidate',str(root/'c'),'--decisions',str(root/'d'),'--output',str(root/'o')],capture_output=True,text=True); self.assertEqual(p.returncode,0,p.stderr); self.assertEqual(len(json.loads((root/'o').read_text())['features'][0]['members']),2)
 def test_unassigned_delta_fails(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); (root/'c').write_text(json.dumps({'refs':{},'delta_facts':[{'delta_id':'D','action':'REPLAY_FL'}]})); (root/'d').write_text(json.dumps({'features':[]})); p=subprocess.run(['python',str(ROOT/'apply_feature_decisions.py'),'--candidate',str(root/'c'),'--decisions',str(root/'d'),'--output',str(root/'o')],capture_output=True,text=True); self.assertNotEqual(p.returncode,0)
if __name__=='__main__': unittest.main()
