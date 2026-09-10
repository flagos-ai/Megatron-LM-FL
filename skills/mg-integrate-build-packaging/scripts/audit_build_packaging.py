#!/usr/bin/env python3
import argparse,ast,csv,hashlib,json,re,subprocess,tomllib
from pathlib import Path
def git(repo,*a,check=True,input=None):
 p=subprocess.run(['git','-C',str(repo),*a],input=input,text=True,encoding='utf-8',errors='replace',capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip())
 return p
def load(p): return json.loads(p.read_text())
def blob(repo,ref,path):
 p=git(repo,'show',f'{ref}:{path}',check=False); return p.stdout if p.returncode==0 else None
def facts(path,text):
 x={'exists':text is not None,'kind':Path(path).suffix or Path(path).name,'sha256':hashlib.sha256(text.encode()).hexdigest() if text is not None else None}
 if text is None:return x
 if path.endswith('pyproject.toml'):
  try:
   d=tomllib.loads(text); x['toml_error']=None; x['build_system']=d.get('build-system',{}); x['project']={k:d.get('project',{}).get(k) for k in ['name','version','requires-python','dependencies','optional-dependencies']}; x['tool_sections']=sorted(d.get('tool',{}))
  except tomllib.TOMLDecodeError as e:x['toml_error']=str(e)
 if Path(path).name=='MANIFEST.in': x['manifest_rules']=[l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith('#')]
 if 'Dockerfile' in Path(path).name:
  x['docker_from']=re.findall(r'^FROM\s+([^\s]+)',text,re.M|re.I); x['docker_stages']=re.findall(r'^FROM\s+\S+\s+AS\s+(\S+)',text,re.M|re.I); x['docker_args']=re.findall(r'^ARG\s+([^=\s]+)',text,re.M)
 if path.endswith('.sh'):
  p=subprocess.run(['bash','-n'],input=text,text=True,capture_output=True); x['shell_valid']=p.returncode==0; x['shell_error']=p.stderr.strip()
 if path.endswith('.py'):
  try:
   t=ast.parse(text); x['python_error']=None; x['imports']=sorted({a.name for n in ast.walk(t) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names})
  except SyntaxError as e:x['python_error']=str(e)
 return x
def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--routing',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); inv=load(a.inventory); routing=load(a.routing)
 if inv['resolved_refs']!=routing.get('refs'):raise SystemExit('artifact refs mismatch')
 refs=inv['resolved_refs']; changes={x['id']:x for x in inv['fork_changes']}; routes=[x for x in routing['routes'] if x['primary_domain']=='build-packaging' or 'build-packaging' in x.get('secondary_domains',[])]; rows=[]
 for r in routes:
  c=changes[r['delta_id']]; f=blob(a.repo,refs['fork'],c['path']); t=blob(a.repo,refs['target'],c['path']); rows.append({'id':'MGBP-'+hashlib.sha256((r['delta_id']+'\0'+c['path']).encode()).hexdigest()[:12].upper(),'delta_id':r['delta_id'],'path':c['path'],'status':c['status'],'primary_domain':r['primary_domain'],'secondary_domains':r.get('secondary_domains',[]),'fork':facts(c['path'],f),'target':facts(c['path'],t),'owner':None,'requirement':None,'target_relationship':None,'strategy':None,'validation_gates':[],'external_gate':None,'reason':None,'evidence':[]})
 rows.sort(key=lambda x:x['delta_id']); out={'schema_version':'1.0','refs':refs,'rows':rows,'summary':{'routes':len(rows),'fork_only':sum(x['fork']['exists'] and not x['target']['exists'] for x in rows),'toml_errors':sum(bool(x['fork'].get('toml_error') or x['target'].get('toml_error')) for x in rows),'shell_errors':sum(x['fork'].get('shell_valid') is False for x in rows),'python_errors':sum(bool(x['fork'].get('python_error') or x['target'].get('python_error')) for x in rows),'manual':len(rows)}}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'build-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with (a.output/'build-routes.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['id','delta_id','path','fork_exists','target_exists','kind']); [w.writerow([x['id'],x['delta_id'],x['path'],x['fork']['exists'],x['target']['exists'],x['fork']['kind']]) for x in rows]
 print(json.dumps(out['summary'],sort_keys=True))
if __name__=='__main__':main()
