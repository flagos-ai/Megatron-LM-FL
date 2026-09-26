#!/usr/bin/env python3
import argparse,csv,hashlib,json,re,subprocess
from pathlib import Path
import yaml
def git(repo,*a,check=True):
 p=subprocess.run(['git','-C',str(repo),*a],text=True,encoding='utf-8',errors='replace',capture_output=True)
 if check and p.returncode:raise SystemExit(p.stderr.strip())
 return p
def blob(repo,ref,path):
 p=git(repo,'show',f'{ref}:{path}',check=False); return p.stdout if p.returncode==0 else None
def mode(repo,ref,path):
 p=git(repo,'ls-tree',ref,'--',path,check=False).stdout.split(); return p[0] if p else None
def facts(repo,ref,path):
 text=blob(repo,ref,path); x={'exists':text is not None,'mode':mode(repo,ref,path),'sha256':hashlib.sha256(text.encode()).hexdigest() if text is not None else None}
 if text is None:return x
 if path.endswith('.md'):x['markdown_links']=sorted(set(re.findall(r'\[[^]]*\]\(([^)]+)\)',text))); x['fence_balanced']=text.count('```')%2==0
 if Path(path).name=='.gitignore':x['rules']=[l for l in text.splitlines() if l and not l.startswith('#')]
 if path.endswith(('.yml','.yaml')):
  try:yaml.safe_load(text); x['parse_error']=None
  except yaml.YAMLError as e:x['parse_error']=str(e)
 if path.endswith('.json'):
  try:json.loads(text); x['parse_error']=None
  except json.JSONDecodeError as e:x['parse_error']=str(e)
 if path.endswith('.sh'):
  p=subprocess.run(['bash','-n'],input=text,text=True,capture_output=True); x['shell_valid']=p.returncode==0; x['shell_error']=p.stderr.strip()
 return x
def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--routing',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); inv=json.loads(a.inventory.read_text()); routing=json.loads(a.routing.read_text())
 if inv['resolved_refs']!=routing.get('refs'):raise SystemExit('artifact refs mismatch')
 refs=inv['resolved_refs']; changes={x['id']:x for x in inv['fork_changes']}; routes=[x for x in routing['routes'] if x['primary_domain']=='repository-support' or 'repository-support' in x.get('secondary_domains',[])]; rows=[]
 for r in routes:
  c=changes[r['delta_id']]; rows.append({'id':'MGRS-'+hashlib.sha256((r['delta_id']+'\0'+c['path']).encode()).hexdigest()[:12].upper(),'delta_id':r['delta_id'],'path':c['path'],'status':c['status'],'primary_domain':r['primary_domain'],'secondary_domains':r.get('secondary_domains',[]),'fork':facts(a.repo,refs['fork'],c['path']),'target':facts(a.repo,refs['target'],c['path']),'owner':None,'purpose':None,'target_relationship':None,'strategy':None,'downstream_consumers':[],'validation':[],'reason':None})
 rows.sort(key=lambda x:x['delta_id']); out={'schema_version':'1.0','refs':refs,'rows':rows,'summary':{'routes':len(rows),'primary':sum(x['primary_domain']=='repository-support' for x in rows),'fork_only':sum(x['fork']['exists'] and not x['target']['exists'] for x in rows),'parse_errors':sum(bool(x['fork'].get('parse_error')) for x in rows),'shell_errors':sum(x['fork'].get('shell_valid') is False for x in rows)}}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'repository-support-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with (a.output/'repository-support-routes.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['id','delta_id','path','primary','fork_exists','target_exists']); [w.writerow([x['id'],x['delta_id'],x['path'],x['primary_domain']=='repository-support',x['fork']['exists'],x['target']['exists']]) for x in rows]
 print(json.dumps(out['summary'],sort_keys=True))
if __name__=='__main__':main()
