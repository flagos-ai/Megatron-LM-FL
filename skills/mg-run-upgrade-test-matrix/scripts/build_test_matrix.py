#!/usr/bin/env python3
import argparse,csv,hashlib,json,re,subprocess
from collections import defaultdict
from pathlib import Path
import yaml
def git(repo,*args,check=True):
 p=subprocess.run(['git','-C',str(repo),*args],text=True,encoding='utf-8',errors='replace',capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip())
 return p
def load(p):
 x=json.loads(p.read_text())
 if not isinstance(x,dict): raise SystemExit('invalid JSON object')
 return x
def rid(*parts): return 'MGT-'+hashlib.sha256('\0'.join(parts).encode()).hexdigest()[:12].upper()
def add(rows,kind,cap,hardware,tier,members,command,resources=None,routes=None):
 members=sorted(set(members)); identity=f'{kind}:{cap}:{hardware}'
 rows.append({'id':rid(identity),'identity':identity,'kind':kind,'capability':cap,'hardware':hardware,'tier':tier,'member_count':len(members),'members':members,'command_template':command,'required_resources':sorted(set(resources or [])),'route_ids':sorted(set(routes or [])),'owner':None,'disposition':'manual','environment':None,'command':None,'reason':None,'execution':{'status':'not-run','exit_code':None,'log':None}})
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--inventory',type=Path,required=True); ap.add_argument('--routing',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 inv=load(a.inventory); routing=load(a.routing)
 if inv['resolved_refs']!=routing.get('refs'): raise SystemExit('artifact refs mismatch')
 repo=a.repo.resolve(); ref=inv['resolved_refs']['fork']; paths=git(repo,'ls-tree','-r','--name-only',ref,'--','tests','.github/configs','.github/workflows','megatron/plugin/platform').stdout.splitlines()
 route_rows=[x for x in routing.get('routes',[]) if x.get('primary_domain')=='test-hardware' or 'test-hardware' in x.get('secondary_domains',[])]
 changes={x['id']:x for x in inv.get('fork_changes',[])}; test_routes=defaultdict(list)
 for r in route_rows:
  if r['delta_id'] in changes: test_routes[changes[r['delta_id']]['path']].append(r['delta_id'])
 rows=[]; unit=defaultdict(list)
 for p in paths:
  if p.startswith('tests/unit_tests/') and p.endswith(('.py','.sh')):
   rest=p[len('tests/unit_tests/'):]; group=rest.split('/')[0] if '/' in rest else 'root'; unit[group].append(p)
 for group,members in sorted(unit.items()):
  routes=sum((test_routes.get(p,[]) for p in members),[])
  add(rows,'unit-suite',group,'runtime-detected','focused-unit',members,f'pytest -q tests/unit_tests/{group}' if group!='root' else 'pytest -q tests/unit_tests --ignore-glob=tests/unit_tests/*/*',routes=routes)
 functional=defaultdict(list); golden=defaultdict(set)
 prefix='tests/functional_tests/test_cases/'
 for p in paths:
  if p.startswith(prefix):
   rest=p[len(prefix):]; family=rest.split('/')[0]; functional[family].append(p)
   m=re.search(r'golden_values_([^/]+)\.json$',p)
   if m: golden[family].add(m.group(1))
 for family,members in sorted(functional.items()):
  resources=['dataset','tokenizer','checkpoint-or-golden'] if any(x.endswith('model_config.yaml') for x in members) else []
  add(rows,'functional-family',family,'accelerator-matrix','functional-golden',members,f'bash tests/functional_tests/shell_test_utils/run_ci_test.sh <{family}-case>',resources,routes=sum((test_routes.get(p,[]) for p in members),[]))
 configs=[]
 for p in sorted(x for x in paths if x.startswith('.github/configs/') and x.endswith(('.yml','.yaml'))):
  doc=yaml.safe_load(git(repo,'show',f'{ref}:{p}').stdout) or {}; hw=str(doc.get('hardware_name') or Path(p).stem)
  configs.append(hw); add(rows,'ci-hardware-config',hw,hw,'single-or-distributed',[p],str(doc.get('setup_script') or '<configured-workflow>'),['authorized-runner'],test_routes.get(p,[]))
 platform_text=git(repo,'show',f'{ref}:megatron/plugin/platform/platform_register.py',check=False).stdout
 platforms=sorted(set(re.findall(r'PLATFORMS\s*\[\s*["\x27]([^"\x27]+)',platform_text)))
 for platform in platforms:
  add(rows,'platform-gate',platform,platform,'single-accelerator',[],f'pytest -q tests/unit_tests -k {platform}',['matching-accelerator'],[])
 flag=[p for p in paths if 'flagscale' in p.lower()]
 if flag: add(rows,'flagscale-e2e','flagscale-integration','accelerator-matrix','flagscale-e2e',flag,'<FlagScale authorized smoke command>',['FlagScale checkout','dataset','authorized-runner'],sum((test_routes.get(p,[]) for p in flag),[]))
 covered=set(sum((x['route_ids'] for x in rows),[])); uncovered=sorted(r['delta_id'] for r in route_rows if r['delta_id'] not in covered)
 if uncovered: add(rows,'route-coverage','test-hardware-deltas','review','static',[changes[x]['path'] for x in uncovered if x in changes],'<review and assign observing tests>',routes=uncovered)
 rows=sorted(rows,key=lambda x:(x['tier'],x['identity']))
 out={'schema_version':'1.0','refs':inv['resolved_refs'],'rows':rows,'discovery':{'unit_groups':len(unit),'functional_families':len(functional),'golden_environments':{k:sorted(v) for k,v in golden.items()},'ci_hardware_configs':sorted(configs),'platforms':platforms,'test_hardware_routes':len(route_rows),'covered_routes':len(set(sum((x['route_ids'] for x in rows),[]))),'fallback_routes':len(uncovered),'uncovered_routes':0},'summary':{'rows':len(rows),'manual':sum(x['disposition']=='manual' for x in rows),'not_run':len(rows)}}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'test-matrix.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with (a.output/'test-matrix.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['id','kind','capability','hardware','tier','members','routes','disposition','status']); [w.writerow([x['id'],x['kind'],x['capability'],x['hardware'],x['tier'],x['member_count'],len(x['route_ids']),x['disposition'],x['execution']['status']]) for x in rows]
 print(json.dumps({**out['summary'],**out['discovery']},sort_keys=True))
if __name__=='__main__': main()
