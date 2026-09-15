#!/usr/bin/env python3
import argparse,ast,csv,hashlib,json,re,subprocess
from pathlib import Path
def git(repo,*args,check=True):
 p=subprocess.run(['git','-C',str(repo),*args],text=True,encoding='utf-8',errors='replace',capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip())
 return p
def load(p): return json.loads(p.read_text())
def show(repo,ref,path):
 p=git(repo,'show',f'{ref}:{path}',check=False); return p.stdout if p.returncode==0 else None
def facts(text):
 if text is None:return {'exists':False,'parse_error':None,'symbols':[],'imports':[],'arguments':[],'fields':[],'calls':[],'markers':0}
 out={'exists':True,'parse_error':None,'symbols':[],'imports':[],'arguments':[],'fields':[],'calls':[],'markers':len(re.findall(r'FlagScale\s+(?:Begin|End|Add)',text,re.I))}
 try: tree=ast.parse(text)
 except SyntaxError as e: out['parse_error']=str(e); return out
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): out['symbols'].append(n.name)
  elif isinstance(n,(ast.Import,ast.ImportFrom)):
   out['imports'].extend(a.name for a in n.names)
  elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name): out['fields'].append(n.target.id)
  elif isinstance(n,ast.Call):
   f=n.func
   name=f.id if isinstance(f,ast.Name) else f.attr if isinstance(f,ast.Attribute) else None
   if name: out['calls'].append(name)
   if name=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str): out['arguments'].append(n.args[0].value)
 for k in ['symbols','imports','arguments','fields','calls']: out[k]=sorted(set(out[k]))
 return out
def stages(path,f):
 s=set()
 if path.endswith('__init__.py'): s.add('public-api')
 if f['arguments'] or 'argument' in path: s|={'argument-declaration','validation'}
 if f['fields'] or 'config' in path: s.add('config-propagation')
 calls=set(f['calls'])
 if calls&{'build_model','setup_model_and_optimizer','get_model','get_optimizer','get_learning_rate_scheduler'}: s.add('construction')
 if 'training.py' in path or calls&{'train','evaluate','train_step','forward_backward_func'}: s.add('execution')
 if calls&{'save_checkpoint','load_checkpoint','write_args_to_tensorboard','log_metrics'}: s.add('checkpoint-output')
 if path.startswith('tests/'): s.add('observing-test')
 return sorted(s or {'training-support'})
def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--routing',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 inv=load(a.inventory); routing=load(a.routing)
 if inv['resolved_refs']!=routing.get('refs'): raise SystemExit('artifact refs mismatch')
 refs=inv['resolved_refs']; changes={x['id']:x for x in inv['fork_changes']}; routes=[x for x in routing['routes'] if x['primary_domain']=='training-integration' or 'training-integration' in x.get('secondary_domains',[])]
 rows=[]
 for route in routes:
  ch=changes[route['delta_id']]; ff=facts(show(a.repo,refs['fork'],ch['path'])); tf=facts(show(a.repo,refs['target'],ch['path']))
  rows.append({'id':'MGTI-'+hashlib.sha256((route['delta_id']+'\0'+ch['path']).encode()).hexdigest()[:12].upper(),'delta_id':route['delta_id'],'path':ch['path'],'status':ch['status'],'primary_domain':route['primary_domain'],'secondary_domains':route.get('secondary_domains',[]),'stages':sorted(set(stages(ch['path'],ff)+stages(ch['path'],tf))),'fork':ff,'target':tf,'owner':None,'invariant':None,'target_relationship':None,'strategy':None,'observing_tests':[],'external_gate':None,'reason':None,'evidence':[]})
 rows.sort(key=lambda x:x['delta_id']); out={'schema_version':'1.0','refs':refs,'rows':rows,'summary':{'routes':len(routes),'parse_errors':sum(bool(x['fork']['parse_error'] or x['target']['parse_error']) for x in rows),'fork_only':sum(x['fork']['exists'] and not x['target']['exists'] for x in rows),'target_missing_symbols':sum(bool(set(x['fork']['symbols'])-set(x['target']['symbols'])) for x in rows),'manual':len(rows)}}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'training-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with (a.output/'training-routes.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['id','delta_id','path','stages','fork_exists','target_exists','fork_symbols','target_symbols']); [w.writerow([x['id'],x['delta_id'],x['path'],','.join(x['stages']),x['fork']['exists'],x['target']['exists'],len(x['fork']['symbols']),len(x['target']['symbols'])]) for x in rows]
 print(json.dumps(out['summary'],sort_keys=True))
if __name__=='__main__':main()
