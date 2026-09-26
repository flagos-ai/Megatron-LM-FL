import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]
def mod(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py'); x=importlib.util.module_from_spec(spec); spec.loader.exec_module(x); return x
audit=mod('audit_training_integration')
def test_ast_extracts_arguments_fields_calls_and_markers():
 text='# FlagScale Begin\nclass C:\n x:int=1\ndef f(p):\n p.add_argument("--foo")\n train()\n'; x=audit.facts(text)
 assert x['arguments']==['--foo'] and x['fields']==['x'] and 'train' in x['calls'] and x['markers']==1
def test_missing_and_parse_error_are_distinct():
 assert audit.facts(None)['exists'] is False; assert audit.facts('def broken(')['parse_error']
def test_lifecycle_stages_do_not_treat_any_import_as_public_api():
 f=audit.facts('import os\ndef train():\n save_checkpoint()\n'); assert audit.stages('megatron/training/training.py',f)==['checkpoint-output','execution']
def test_public_init_and_observing_test_stages():
 assert 'public-api' in audit.stages('megatron/__init__.py',audit.facts('from .x import y'))
 assert 'observing-test' in audit.stages('tests/unit_tests/test_x.py',audit.facts('def test_x(): pass'))
def test_validator_rejects_missing_review(tmp_path):
 row={'id':'x','fork':{'parse_error':None},'target':{'parse_error':None},'owner':None,'invariant':None,'target_relationship':None,'strategy':None,'observing_tests':[],'external_gate':None}
 p=tmp_path/'a'; p.write_text(json.dumps({'schema_version':'1.0','rows':[row]})); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_training_audit.py'),'--audit',str(p)],capture_output=True,text=True)
 assert r.returncode==2 and 'missing invariant x' in r.stdout
def test_validator_accepts_complete_external_gate(tmp_path):
 row={'id':'x','fork':{'parse_error':None},'target':{'parse_error':None},'owner':'o','invariant':'i','target_relationship':'changed','strategy':'adapt','observing_tests':[],'external_gate':{'owner':'hw'}}
 p=tmp_path/'a'; p.write_text(json.dumps({'schema_version':'1.0','rows':[row]})); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_training_audit.py'),'--audit',str(p)],capture_output=True,text=True)
 assert r.returncode==0
def test_decisions_cannot_change_facts(tmp_path):
 m={'refs':{'fork':'a'},'rows':[{'id':'x','path':'p'}]}; d={'refs':m['refs'],'decisions':{'x':{'path':'q'}}}; mp=tmp_path/'m'; dp=tmp_path/'d'; out=tmp_path/'o'; mp.write_text(json.dumps(m)); dp.write_text(json.dumps(d))
 r=subprocess.run([sys.executable,str(ROOT/'scripts'/'apply_training_decisions.py'),'--audit',str(mp),'--decisions',str(dp),'--output',str(out)],capture_output=True,text=True)
 assert r.returncode!=0 and 'unsupported fields' in r.stderr
