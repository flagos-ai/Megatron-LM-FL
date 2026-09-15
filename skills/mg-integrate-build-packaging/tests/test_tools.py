import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]
def mod(n):
 s=importlib.util.spec_from_file_location(n,ROOT/'scripts'/f'{n}.py'); x=importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
a=mod('audit_build_packaging')
def test_pyproject_facts():
 x=a.facts('pyproject.toml','[build-system]\nrequires=["setuptools"]\n[project]\nname="x"\ndependencies=["torch"]\n'); assert x['build_system']['requires']==['setuptools'] and x['project']['dependencies']==['torch']
def test_manifest_and_docker_facts():
 assert a.facts('MANIFEST.in','include A\n# no\n')['manifest_rules']==['include A']
 x=a.facts('docker/Dockerfile.x','ARG BASE\nFROM ubuntu AS build\n'); assert x['docker_from']==['ubuntu'] and x['docker_stages']==['build'] and x['docker_args']==['BASE']
def test_shell_and_python_syntax():
 assert a.facts('x.sh','if then')['shell_valid'] is False; assert a.facts('x.py','def x(')['python_error']
def test_validator_rejects_unreviewed(tmp_path):
 row={'id':'x','fork':{},'owner':None,'requirement':None,'target_relationship':None,'strategy':None,'validation_gates':[],'external_gate':None}; p=tmp_path/'a'; p.write_text(json.dumps({'schema_version':'1.0','rows':[row]})); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_build_audit.py'),'--audit',str(p)],capture_output=True,text=True); assert r.returncode==2
def test_validator_accepts_complete(tmp_path):
 row={'id':'x','fork':{},'owner':'o','requirement':'r','target_relationship':'changed','strategy':'adapt','validation_gates':['wheel']}; p=tmp_path/'a'; p.write_text(json.dumps({'schema_version':'1.0','rows':[row]})); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_build_audit.py'),'--audit',str(p)],capture_output=True,text=True); assert r.returncode==0
def test_decision_cannot_change_path(tmp_path):
 m={'refs':{'fork':'a'},'rows':[{'id':'x','path':'p'}]}; d={'refs':m['refs'],'decisions':{'x':{'path':'q'}}}; mp=tmp_path/'m'; dp=tmp_path/'d'; o=tmp_path/'o'; mp.write_text(json.dumps(m)); dp.write_text(json.dumps(d)); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'apply_build_decisions.py'),'--audit',str(mp),'--decisions',str(dp),'--output',str(o)],capture_output=True,text=True); assert r.returncode!=0
