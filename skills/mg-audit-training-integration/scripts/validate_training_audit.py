#!/usr/bin/env python3
import argparse,json
from pathlib import Path
STRATEGIES={'preserve','adapt','upstream','redesign','drop'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.audit.read_text()); errors=[]; ids=[]
 if x.get('schema_version')!='1.0': errors.append('unsupported schema')
 for r in x.get('rows',[]):
  ids.append(r.get('id'))
  if r['fork'].get('parse_error') or r['target'].get('parse_error'): errors.append('parse error '+r['id'])
  for field in ['owner','invariant','target_relationship','strategy']:
   if not r.get(field): errors.append(f'missing {field} {r["id"]}')
  if r.get('strategy') and r['strategy'] not in STRATEGIES: errors.append('invalid strategy '+r['id'])
  if not r.get('observing_tests') and not r.get('external_gate'): errors.append('missing observing test/gate '+r['id'])
 if len(ids)!=len(set(ids)): errors.append('duplicate ids')
 result={'valid':not errors,'errors':errors}; text=json.dumps(result,indent=2,sort_keys=True)+'\n'; print(text,end='')
 if a.output:a.output.write_text(text)
 if errors:raise SystemExit(2)
if __name__=='__main__':main()
