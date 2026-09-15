import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]
def mod(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py'); x=importlib.util.module_from_spec(spec); spec.loader.exec_module(x); return x
build=mod('build_test_matrix')
def test_row_id_is_stable_and_identity_sensitive():
 assert build.rid('a')==build.rid('a') and build.rid('a')!=build.rid('b')
def test_add_preserves_sorted_members_routes_and_not_run():
 rows=[]; build.add(rows,'unit','x','cpu','focused',['b','a','a'],'pytest x',routes=['r2','r1'])
 assert rows[0]['members']==['a','b'] and rows[0]['route_ids']==['r1','r2'] and rows[0]['execution']['status']=='not-run'
def test_validator_rejects_unreviewed_and_unproven_pass(tmp_path):
 data={'schema_version':'1.0','discovery':{'covered_routes':1,'test_hardware_routes':1,'uncovered_routes':0},'rows':[{'id':'x','disposition':'authorized','owner':'o','command':'c','execution':{'status':'pass','exit_code':0,'log':None}}]}
 p=tmp_path/'m.json'; p.write_text(json.dumps(data)); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_test_matrix.py'),'--matrix',str(p)],capture_output=True,text=True)
 assert r.returncode==2 and 'unproven pass x' in r.stdout
def test_validator_accepts_owned_external_gate(tmp_path):
 data={'schema_version':'1.0','discovery':{'covered_routes':1,'test_hardware_routes':1,'uncovered_routes':0},'rows':[{'id':'x','disposition':'external-gate','owner':'hw','command_template':'pytest','reason':'host unavailable','execution':{'status':'blocked','exit_code':None,'log':None}}]}
 p=tmp_path/'m.json'; p.write_text(json.dumps(data)); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_test_matrix.py'),'--matrix',str(p)],capture_output=True,text=True)
 assert r.returncode==0
def test_decision_composition_rejects_ref_mismatch(tmp_path):
 m={'refs':{'fork':'a'},'rows':[]}; d={'refs':{'fork':'b'},'decisions':{}}; mp=tmp_path/'m'; dp=tmp_path/'d'; out=tmp_path/'o'; mp.write_text(json.dumps(m)); dp.write_text(json.dumps(d))
 r=subprocess.run([sys.executable,str(ROOT/'scripts'/'apply_test_decisions.py'),'--matrix',str(mp),'--decisions',str(dp),'--output',str(out)],capture_output=True,text=True)
 assert r.returncode!=0 and 'artifact refs mismatch' in r.stderr
def test_decisions_cannot_change_identity(tmp_path):
 m={'refs':{'fork':'a'},'rows':[{'id':'x','identity':'immutable'}]}; d={'refs':m['refs'],'decisions':{'x':{'identity':'changed'}}}; mp=tmp_path/'m'; dp=tmp_path/'d'; out=tmp_path/'o'; mp.write_text(json.dumps(m)); dp.write_text(json.dumps(d))
 r=subprocess.run([sys.executable,str(ROOT/'scripts'/'apply_test_decisions.py'),'--matrix',str(mp),'--decisions',str(dp),'--output',str(out)],capture_output=True,text=True)
 assert r.returncode!=0 and 'unsupported decision fields' in r.stderr
