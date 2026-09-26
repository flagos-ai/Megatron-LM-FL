#!/usr/bin/env python3
import argparse,csv,json,re,subprocess
from pathlib import Path
import yaml
LOCAL=re.compile(r'(?<![A-Za-z0-9_./@-])(?:\./)?((?:\.github|tests|tools|scripts|examples)/[A-Za-z0-9_./*?{}$\[\]-]+|Dockerfile(?:\.[A-Za-z0-9_.-]+)?)')
def git(repo,*args,check=True,input_text=None):
 p=subprocess.run(['git','-C',str(repo),*args],input=input_text,text=True,encoding='utf-8',errors='replace',capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip())
 return p
def load(p):
 x=json.loads(p.read_text());
 if not isinstance(x,dict): raise SystemExit('invalid JSON object')
 return x
def exists(repo,ref,path): return git(repo,'cat-file','-e',f'{ref}:{path}',check=False).returncode==0
def parse_yaml(text):
 try:
  x=yaml.safe_load(text); return x if isinstance(x,dict) else {},None
 except yaml.YAMLError as e: return {},str(e)
def workflow_events(doc):
 value=doc.get('on',doc.get(True,{}))
 if isinstance(value,dict): return sorted(map(str,value))
 if isinstance(value,list): return sorted(map(str,value))
 return [str(value)] if value is not None else []
def local_references(line):
 stripped=line.lstrip()
 if not stripped or stripped.startswith('#'): return []
 return [m.group(1).rstrip('),]}"\x27') for m in LOCAL.finditer(line)]
def generated_references(text):
 found=set()
 for line in text.splitlines():
  if line.lstrip().startswith('#'): continue
  for ref in local_references(line):
   if re.search(r'>\s*(?:\./)?'+re.escape(ref)+r'(?:\s|$)',line): found.add(ref)
 return found
def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--routing',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); inv=load(a.inventory); routing=load(a.routing); refs=inv['resolved_refs']; fork=refs['fork']; target=refs['target']; repo=a.repo.resolve()
 if routing.get('refs')!=refs: raise SystemExit('artifact refs mismatch')
 paths=git(repo,'ls-tree','-r','--name-only',fork,'--','.github').stdout.splitlines(); yaml_paths=sorted(x for x in paths if x.endswith(('.yml','.yaml'))); workflows=[]; references=[]; masks=[]
 for path in yaml_paths:
  text=git(repo,'show',f'{fork}:{path}').stdout; doc,error=parse_yaml(text); generated=generated_references(text); jobs=doc.get('jobs',{}) if isinstance(doc.get('jobs',{}),dict) else {}; row={'path':path,'kind':'workflow' if '/workflows/' in path else 'config','parse_error':error,'fork_only':not exists(repo,target,path),'events':workflow_events(doc),'permissions':doc.get('permissions'),'jobs':[]}
  for name,job in sorted(jobs.items()):
   if not isinstance(job,dict): continue
   row['jobs'].append({'name':name,'runs_on':job.get('runs-on'),'uses':job.get('uses'),'needs':job.get('needs'),'if':job.get('if'),'continue_on_error':job.get('continue-on-error'),'matrix':job.get('strategy',{}).get('matrix') if isinstance(job.get('strategy',{}),dict) else None})
   if job.get('continue-on-error'): masks.append({'path':path,'job':name,'kind':'continue-on-error','owner':None,'disposition':'manual'})
  for number,line in enumerate(text.splitlines(),1):
   if ('|| true' in line and '$'+'{{' not in line) or re.search(r'pytest.*(?:--ignore| -k |deselect)',line): masks.append({'path':path,'line':number,'kind':'shell-or-test-mask','text':line.strip(),'owner':None,'disposition':'manual'})
   for ref in local_references(line):
    dynamic=any(c in ref for c in '*?{}$['); present=(not dynamic and exists(repo,fork,ref)) or ref in generated; references.append({'source':path,'line':number,'reference':ref,'dynamic':dynamic,'generated':ref in generated,'exists':present,'owner':None,'disposition':'manual' if dynamic or not present else ('generated' if ref in generated else 'resolved')})
  workflows.append(row)
 configs=sorted(Path(x).stem for x in yaml_paths if x.startswith('.github/configs/'))
 platform_text=git(repo,'show',f'{fork}:megatron/plugin/platform/platform_register.py',check=False).stdout; platforms=sorted(set(re.findall(r'PLATFORMS\s*\[\s*["\x27]([^"\x27]+)',platform_text))); backend_coverage=[{'backend':x,'has_config':x in configs,'owner':None,'disposition':'covered' if x in configs else 'manual'} for x in platforms]
 shell_paths=sorted(x for x in paths if x.endswith('.sh')); shell=[]
 for path in shell_paths:
  text=git(repo,'show',f'{fork}:{path}').stdout; proc=subprocess.run(['bash','-n'],input=text,text=True,capture_output=True); shell.append({'path':path,'valid':proc.returncode==0,'error':proc.stderr.strip(),'owner':None,'disposition':'valid' if proc.returncode==0 else 'manual'})
 changes={x['id']:x for x in inv.get('fork_changes',[])}; routes=[r for r in routing.get('routes',[]) if r.get('primary_domain')=='cicd' or 'cicd' in r.get('secondary_domains',[])]; route_paths=sorted({changes[r['delta_id']]['path'] for r in routes if r['delta_id'] in changes})
 out={'schema_version':'1.0','refs':refs,'workflows':workflows,'references':sorted(references,key=lambda x:(x['source'],x['line'],x['reference'])),'failure_masks':sorted(masks,key=lambda x:(x['path'],x.get('line',0),x.get('job',''))),'configs':configs,'platforms':platforms,'backend_coverage':backend_coverage,'shell_checks':shell,'cicd_route_ids':sorted(r['delta_id'] for r in routes),'cicd_route_paths':route_paths,'summary':{'yaml_files':len(workflows),'parse_errors':sum(bool(x['parse_error']) for x in workflows),'fork_only_yaml':sum(x['fork_only'] for x in workflows),'jobs':sum(len(x['jobs']) for x in workflows),'references':len(references),'missing_references':sum(not x['exists'] and not x['dynamic'] for x in references),'dynamic_references':sum(x['dynamic'] for x in references),'failure_masks':len(masks),'platforms':len(platforms),'configs':len(configs),'backend_gaps':sum(not x['has_config'] for x in backend_coverage),'shell_files':len(shell),'shell_errors':sum(not x['valid'] for x in shell),'cicd_routes':len(routes)}}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'cicd-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with (a.output/'workflow-matrix.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['path','kind','parse_error','fork_only','jobs']); [w.writerow([x['path'],x['kind'],bool(x['parse_error']),x['fork_only'],len(x['jobs'])]) for x in workflows]
 with (a.output/'missing-references.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['source','line','reference','dynamic','exists']); [w.writerow([x[k] for k in ['source','line','reference','dynamic','exists']]) for x in references if x['dynamic'] or not x['exists']]
 print(json.dumps(out['summary'],indent=2,sort_keys=True))
if __name__=='__main__': main()
