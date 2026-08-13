import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
audit=module('audit_cicd')
def test_yaml_on_mapping_is_preserved():
 doc,error=audit.parse_yaml('on:\n  push:\n  pull_request:\njobs: {}\n'); assert error is None and audit.workflow_events(doc)==['pull_request','push']
def test_yaml_on_scalar_and_list():
 assert audit.workflow_events({'on':'push'})==['push']; assert audit.workflow_events({'on':['push','workflow_dispatch']})==['push','workflow_dispatch']
def test_yaml_parse_error():
 doc,error=audit.parse_yaml('jobs: ['); assert doc=={} and error
def test_validator_rejects_unreviewed_findings(tmp_path):
 data={'schema_version':'1.0','cicd_route_ids':[],'workflows':[],'references':[{'source':'x','line':1,'reference':'missing','dynamic':False,'exists':False,'owner':None,'disposition':'manual'}],'backend_coverage':[],'failure_masks':[],'shell_checks':[]}
 src=tmp_path/'audit.json'; src.write_text(json.dumps(data)); p=subprocess.run([sys.executable,str(ROOT/'scripts'/'validate_cicd_audit.py'),'--audit',str(src)],capture_output=True,text=True)
 assert p.returncode==2 and 'unresolved reference x:1' in p.stdout
def test_decisions_preserve_identity_and_close_reference(tmp_path):
 audit_data={'schema_version':'1.0','refs':{'fork':'a','target':'b'},'references':[{'source':'x','line':1,'reference':'missing','dynamic':False,'exists':False,'owner':None,'disposition':'manual'}]}
 decisions={'refs':audit_data['refs'],'decisions':{'references':{'x:1:missing':{'owner':'ci','disposition':'approved-external','reason':'fixture'}}}}
 src=tmp_path/'audit.json'; dec=tmp_path/'decisions.json'; out=tmp_path/'out.json'; src.write_text(json.dumps(audit_data)); dec.write_text(json.dumps(decisions))
 subprocess.run([sys.executable,str(ROOT/'scripts'/'apply_cicd_decisions.py'),'--audit',str(src),'--decisions',str(dec),'--output',str(out)],check=True)
 row=json.loads(out.read_text())['references'][0]; assert row['source']=='x' and row['reference']=='missing' and row['owner']=='ci'
def test_decisions_reject_stale_key(tmp_path):
 audit_data={'refs':{'fork':'a','target':'b'},'references':[]}; decisions={'refs':audit_data['refs'],'decisions':{'references':{'stale':{'owner':'ci'}}}}
 src=tmp_path/'audit.json'; dec=tmp_path/'decisions.json'; out=tmp_path/'out.json'; src.write_text(json.dumps(audit_data)); dec.write_text(json.dumps(decisions))
 p=subprocess.run([sys.executable,str(ROOT/'scripts'/'apply_cicd_decisions.py'),'--audit',str(src),'--decisions',str(dec),'--output',str(out)],capture_output=True,text=True)
 assert p.returncode!=0 and 'unknown references decision key' in p.stderr

def test_local_reference_extraction_ignores_external_and_comments():
 assert audit.local_references('uses: ./.github/actions/local')==['.github/actions/local']
 assert audit.local_references('uses: NVIDIA/x/.github/workflows/a.yml@v1')==[]
 assert audit.local_references('# tests/missing.py')==[]
 assert audit.local_references('run: pytest tests/unit_tests/test_x.py')==['tests/unit_tests/test_x.py']
def test_glob_reference_is_discovered_for_manual_resolution():
 assert audit.local_references('run: pytest tests/unit_tests/test_*.py')==['tests/unit_tests/test_*.py']
