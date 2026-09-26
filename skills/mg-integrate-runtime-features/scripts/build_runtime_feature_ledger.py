#!/usr/bin/env python3
import argparse,ast,csv,json,subprocess
from collections import Counter,defaultdict
from pathlib import Path
def git(repo,*args,check=True):
 p=subprocess.run(['git','-C',str(repo),*args],text=True,encoding='utf-8',errors='replace',capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip())
 return p.stdout
def load(p):
 x=json.loads(p.read_text());
 if not isinstance(x,dict): raise SystemExit(f'invalid JSON object: {p}')
 return x
def symbols(text):
 try: tree=ast.parse(text)
 except SyntaxError: return [],'syntax-error'
 out=[]; parents=[]
 def walk(n):
  if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
   parents.append(n.name); out.append({'qualname':'.'.join(parents),'kind':'class' if isinstance(n,ast.ClassDef) else 'async-function' if isinstance(n,ast.AsyncFunctionDef) else 'function','line':n.lineno})
  for c in ast.iter_child_nodes(n): walk(c)
  if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)): parents.pop()
 walk(tree); return sorted(out,key=lambda x:(x['line'],x['qualname'])),'parsed'

def relations(text):
 try: tree=ast.parse(text)
 except SyntaxError: return {"imports":[],"calls":[]}
 imports=[]; calls=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import): imports.extend(alias.name for alias in node.names)
  elif isinstance(node,ast.ImportFrom): imports.append((node.module or "")+":"+",".join(alias.name for alias in node.names))
  elif isinstance(node,ast.Call):
   try: calls.append(ast.unparse(node.func))
   except Exception: pass
 return {"imports":sorted(set(imports)),"calls":sorted(set(calls))}

def stage(path,category):
 p=path.lower()
 if category.startswith('tests-') or '/test' in p or p.startswith('tests/'): return 'observing-test'
 if 'checkpoint' in p or 'state_dict' in p or 'metric' in p: return 'output-state'
 if any(x in p for x in ['argument','config','yaml']): return 'configuration'
 if any(x in p for x in ['schedule','router','dispatch','pipeline']): return 'dispatch'
 if any(x in p for x in ['__init__','builder','factory']): return 'construction'
 return 'execution'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--decisions',type=Path,required=True); p.add_argument('--routing',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); inv=load(a.inventory); dec=load(a.decisions); routing=load(a.routing); refs=inv['resolved_refs']
 if dec.get('refs')!=refs or routing.get('refs')!=refs: raise SystemExit('artifact refs mismatch')
 changes={x['id']:x for x in inv.get('fork_changes',[])}; decisions={x['id']:x for x in dec.get('decisions',[])}; marker_count=Counter(x['path'] for x in inv.get('flagscale_blocks',[])); routes=[x for x in routing.get('routes',[]) if x.get('primary_domain')=='runtime-feature' or 'runtime-feature' in x.get('secondary_domains',[])]; facts=[]
 for r in sorted(routes,key=lambda x:x['delta_id']):
  d=r['delta_id']; c=changes.get(d); q=decisions.get(d)
  if not c or not q: raise SystemExit(f'missing delta evidence {d}')
  text=git(a.repo.resolve(),'show',f"{refs['fork']}:{c['path']}",check=False); syms,status=symbols(text) if c['path'].endswith('.py') and text else ([],'non-python-or-missing'); rel=relations(text) if c['path'].endswith('.py') and text else {'imports':[],'calls':[]}
  facts.append({'delta_id':d,'source_group_id':r['feature_group'],'path':c['path'],'category':c['category'],'priority':c['priority'],'action':q['effective_action'],'both_changed':bool(c.get('both_changed')),'primary_domain':r['primary_domain'],'secondary_domains':r.get('secondary_domains',[]),'secondary_handlers':r.get('secondary_handlers',{}),'candidate_stage':stage(c['path'],c['category']),'marker_block_count':marker_count[c['path']],'symbols':syms,'imports':rel['imports'],'calls':rel['calls'],'parse_status':status})
 groups=defaultdict(list)
 for x in facts: groups[x['source_group_id']].append(x)
 candidates=[]
 for gid in sorted(groups):
  members=groups[gid]; definitions={sym['qualname'].split('.')[-1]:m['delta_id'] for m in members for sym in m['symbols']}; edges=[]
  for m in members:
   for call in m['calls']:
    name=call.split('.')[-1]
    if name in definitions and definitions[name]!=m['delta_id']: edges.append({'from_delta_id':m['delta_id'],'to_delta_id':definitions[name],'kind':'candidate-call','symbol':name})
  unique_edges=sorted({(e['from_delta_id'],e['to_delta_id'],e['kind'],e['symbol']) for e in edges}); stages=Counter(x['candidate_stage'] for x in members); candidates.append({'source_group_id':gid,'delta_ids':sorted(x['delta_id'] for x in members),'paths':sorted({x['path'] for x in members}),'actions':dict(sorted(Counter(x['action'] for x in members).items())),'candidate_stages':dict(sorted(stages.items())),'candidate_edges':[{'from_delta_id':e[0],'to_delta_id':e[1],'kind':e[2],'symbol':e[3]} for e in unique_edges],'redesign':any(x['action']=='REDESIGN' for x in members),'both_changed':sum(x['both_changed'] for x in members),'review_status':'needs-review','feature_name':None,'invariant':None,'target_change':None,'strategy':None,'owner':None,'dependencies':[],'observing_tests':[],'reviewer':None,'approval_status':'unapproved','skill_gap_ids':[]})
 out={'schema_version':'1.0','refs':refs,'delta_facts':facts,'candidate_features':candidates,'summary':{'runtime_routes':len(facts),'candidate_features':len(candidates),'redesign_features':sum(x['redesign'] for x in candidates),'features_without_candidate_test':sum(not x['candidate_stages'].get('observing-test') for x in candidates),'marker_blocks':sum(x['marker_block_count'] for x in facts),'candidate_edges':sum(len(x['candidate_edges']) for x in candidates)}}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'candidate-runtime-features.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with (a.output/'runtime-feature-matrix.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['source_group_id','delta_count','path_count','actions','stages','redesign','both_changed']); [w.writerow([x['source_group_id'],len(x['delta_ids']),len(x['paths']),json.dumps(x['actions'],sort_keys=True),json.dumps(x['candidate_stages'],sort_keys=True),x['redesign'],x['both_changed']]) for x in candidates]
 print(json.dumps(out['summary'],indent=2,sort_keys=True))
if __name__=='__main__': main()
