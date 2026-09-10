import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]
s=importlib.util.spec_from_file_location('a',ROOT/'scripts'/'audit_repository_support.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a)
def test_markdown_gitignore_and_json_facts(tmp_path):
 repo=tmp_path; subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
 assert a.facts(repo,'HEAD','x')['exists'] is False
def test_validator_rejects_unreviewed(tmp_path):
 row={'id':'x','fork':{},'owner':None,'purpose':None,'target_relationship':None,'strategy':None,'downstream_consumers':[],'validation':[]}; p=tmp_path/'a'; p.write_text(json.dumps({'rows':[row]})); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_repository_support.py'),'--audit',str(p)],capture_output=True,text=True); assert r.returncode==2
def test_validator_accepts_complete(tmp_path):
 row={'id':'x','fork':{},'owner':'o','purpose':'p','target_relationship':'changed','strategy':'adapt','downstream_consumers':['ci'],'validation':['syntax']}; p=tmp_path/'a'; p.write_text(json.dumps({'rows':[row]})); r=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_repository_support.py'),'--audit',str(p)],capture_output=True,text=True); assert r.returncode==0
def test_skill_links_contract():
 text=(ROOT/'SKILL.md').read_text(); assert 'repository-support.md' in text and 'audit_repository_support.py' in text
