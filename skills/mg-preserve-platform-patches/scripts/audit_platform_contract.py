#!/usr/bin/env python3
import argparse,ast,csv,hashlib,json,re,subprocess
from pathlib import Path
PATTERNS={'torch.cuda':re.compile(r'\btorch\.cuda\b'),'cuda-literal':re.compile(r'["\x27]cuda(?::[^"\x27]*)?["\x27]',re.I),'nccl':re.compile(r'\bnccl\b',re.I),'nvtx':re.compile(r'\bnvtx\b',re.I),'cuda-symbol':re.compile(r'\bCUDA[A-Za-z0-9_]*\b|\bcuda_[A-Za-z0-9_]+\b')}
def git(repo,*args,check=True):
 p=subprocess.run(['git','-C',str(repo),*args],text=True,encoding='utf-8',errors='replace',capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip())
 return p.stdout
def load(p):
 x=json.loads(p.read_text());
 if not isinstance(x,dict): raise SystemExit(f'invalid JSON object: {p}')
 return x
def sig(n):
 a=n.args; pos=a.posonlyargs+a.args; start=len(pos)-len(a.defaults); out=[]
 for i,x in enumerate(pos): out.append({'name':x.arg,'kind':'posonly' if i<len(a.posonlyargs) else 'positional','default':ast.unparse(a.defaults[i-start]) if i>=start else None})
 if a.vararg: out.append({'name':a.vararg.arg,'kind':'vararg','default':None})
 for x,d in zip(a.kwonlyargs,a.kw_defaults): out.append({'name':x.arg,'kind':'kwonly','default':ast.unparse(d) if d else None})
 if a.kwarg: out.append({'name':a.kwarg.arg,'kind':'kwarg','default':None})
 return out
def methods(cls):
 out={}
 for n in cls.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
   dec={ast.unparse(x) for x in n.decorator_list}; out[n.name]={'kind':'property' if 'property' in dec else 'async' if isinstance(n,ast.AsyncFunctionDef) else 'method','parameters':sig(n),'abstract':any('abstractmethod' in x for x in dec)}
 return out
def compatible_signature(base, impl):
 if base['kind'] != impl['kind'] or len(base['parameters']) != len(impl['parameters']): return False
 for left,right in zip(base['parameters'],impl['parameters']):
  if (left['name'],left['kind']) != (right['name'],right['kind']): return False
  if left['default'] is not None and left['default'] != right['default']: return False
 return True

def parse_platform(text,path):
 tree=ast.parse(text); rows=[]
 for n in tree.body:
  if isinstance(n,ast.ClassDef) and any(ast.unparse(b).split('.')[-1].startswith('Platform') for b in n.bases): rows.append({'class':n.name,'path':path,'bases':[ast.unparse(b) for b in n.bases],'methods':methods(n)})
 return rows
def enclosing_symbol(tree,line):
 best=None
 for n in ast.walk(tree):
  if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and getattr(n,'lineno',0)<=line<=getattr(n,'end_lineno',-1):
   if best is None or n.lineno>=best.lineno: best=n
 return best.name if best else '<module>'

def parse_registration(register_text, manager_text):
 assignments = {}
 try:
  tree=ast.parse(register_text)
  for node in ast.walk(tree):
   if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name): assignments[node.targets[0].id]=node.value.func.id
 except SyntaxError: pass
 registrations=[]
 for match in re.finditer(r'PLATFORMS\s*\[\s*["\']([^"\']+)["\']\s*\]\s*=\s*([A-Za-z_][A-Za-z0-9_]*)',register_text): registrations.append({'key':match.group(1),'instance_variable':match.group(2),'class':assignments.get(match.group(2))})
 selection=[m.group(1) for m in re.finditer(r'["\']([^"\']+)["\']\s+in\s+PLATFORMS\.keys\(\)',manager_text)]
 return sorted(registrations,key=lambda x:x['key']),selection

def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--routing',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); inv=load(a.inventory); routing=load(a.routing); refs=inv['resolved_refs']
 if routing.get('refs')!=refs: raise SystemExit('artifact refs mismatch')
 manifest=inv.get('platform_manifest',{}); files=manifest.get('platform_files',[]); fork=refs['fork']; repo=a.repo.resolve(); base_methods={}; platforms=[]
 register_text=git(repo,'show',f'{fork}:megatron/plugin/platform/platform_register.py',check=False); manager_text=git(repo,'show',f'{fork}:megatron/plugin/platform/platform_manager.py',check=False); registrations,selection_order=parse_registration(register_text,manager_text)
 for path in sorted(files):
  text=git(repo,'show',f'{fork}:{path}',check=False)
  if not text: continue
  try: parsed=parse_platform(text,path)
  except SyntaxError: continue
  if path.endswith('platform_base.py'):
   tree=ast.parse(text); cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PlatformBase'),None); base_methods=methods(cls) if cls else {}
  platforms.extend(parsed)
 for row in platforms:
  inherited=any(b.split('.')[-1] != 'PlatformBase' for b in row['bases']); row['missing_base_apis']=sorted(set(base_methods)-set(row['methods'])) if not inherited else []; row['signature_mismatches']=sorted(k for k in set(base_methods)&set(row['methods']) if not compatible_signature(base_methods[k],row['methods'][k])); row['inherited_contract']=inherited; row.update({'owner':None,'disposition':'manual' if row['missing_base_apis'] or row['signature_mismatches'] or inherited else 'compatible','tests':[]})
 route_rows=[r for r in routing.get('routes',[]) if r.get('primary_domain')=='platform' or 'platform' in r.get('secondary_domains',[])]; route_by_path={}
 changes={r['id']:r for r in inv.get('fork_changes',[])}
 for r in route_rows:
  c=changes.get(r['delta_id']);
  if c: route_by_path.setdefault(c['path'],[]).append(r['delta_id'])
 occurrences=[]
 for path in sorted(route_by_path):
  for ref_name in ['fork','target']:
   text=git(repo,'show',f'{refs[ref_name]}:{path}',check=False)
   if not text: continue
   try: tree=ast.parse(text) if path.endswith('.py') else None
   except SyntaxError: tree=None
   for number,line in enumerate(text.splitlines(),1):
    for family,pattern in PATTERNS.items():
     if pattern.search(line): occurrences.append({'path':path,'line':number,'symbol':enclosing_symbol(tree,number) if tree else '<non-python>','family':family,'ref':ref_name,'route_delta_ids':sorted(route_by_path[path]),'text':line.strip()[:240],'classification':'manual','owner':None,'reason':None,'tests':[]})
 grouped={}
 for item in occurrences:
  key=(item['path'],item['symbol'],item['family'])
  row=grouped.setdefault(key,{'path':item['path'],'symbol':item['symbol'],'family':item['family'],'route_delta_ids':item['route_delta_ids'],'fork_count':0,'target_count':0,'sample_lines':{'fork':[],'target':[]},'classification':'manual','owner':None,'reason':None,'tests':[]})
  row[item['ref']+'_count']+=1
  if len(row['sample_lines'][item['ref']])<3: row['sample_lines'][item['ref']].append({'line':item['line'],'text':item['text']})
 occurrences=list(grouped.values())
 for row in occurrences:
  row['occurrence_id']='MGP-'+hashlib.sha256(('\0'.join([row['path'],row['symbol'],row['family']])).encode()).hexdigest()[:12].upper()
  row['change_scope']='target-new-or-expanded' if row['target_count']>row['fork_count'] else 'fork-only-or-reduced' if row['fork_count']>row['target_count'] else 'shared-count'
 occurrences.sort(key=lambda x:(x['path'],x['symbol'],x['family']))
 reg_by_class={r['class']:r['key'] for r in registrations if r.get('class')}
 for row in platforms: row['registration_key']=reg_by_class.get(row['class']); row['selected']=row['registration_key'] in selection_order
 platforms.sort(key=lambda x:(x['class'],x['path'])); out={'schema_version':'1.0','refs':refs,'base_api_count':len(base_methods),'registrations':registrations,'selection_order':selection_order,'platforms':platforms,'platform_route_ids':sorted(r['delta_id'] for r in route_rows),'raw_device_occurrences':occurrences,'summary':{'platforms':len(platforms),'base_apis':len(base_methods),'platform_routes':len(route_rows),'contract_reviews':sum(x['disposition']=='manual' for x in platforms),'raw_occurrences':len(occurrences)}}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'platform-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with (a.output/'capability-matrix.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['class','path','bases','missing_base_apis','signature_mismatches','disposition']); [w.writerow([x['class'],x['path'],','.join(x['bases']),','.join(x['missing_base_apis']),','.join(x['signature_mismatches']),x['disposition']]) for x in platforms]
 with (a.output/'raw-device-assumptions.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['path','symbol','family','fork_count','target_count','change_scope','classification']); [w.writerow([x[k] for k in ['path','symbol','family','fork_count','target_count','change_scope','classification']]) for x in occurrences]
 print(json.dumps(out['summary'],indent=2,sort_keys=True))
if __name__=='__main__': main()
