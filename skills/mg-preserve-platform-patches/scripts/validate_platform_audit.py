#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.audit.read_text()); errors=[]
 if x.get('schema_version')!='1.0': errors.append('unsupported schema')
 if len(x.get('platform_route_ids',[]))!=len(set(x.get('platform_route_ids',[]))): errors.append('duplicate platform route IDs')
 for r in x.get('platforms',[]):
  if r.get('disposition') not in {'compatible','approved-exception'} or not r.get('owner') or not r.get('tests'): errors.append(f"unresolved platform {r.get('class')}")
 for r in x.get('raw_device_occurrences',[]):
  if r.get('classification') not in {'portable-abstraction','intentional-backend','build-test-only','approved-exception'} or not r.get('owner') or not r.get('reason') or not r.get('tests'): errors.append(f"unresolved occurrence {r.get('occurrence_id')}")
 result={'valid':not errors,'errors':errors}; text=json.dumps(result,indent=2,sort_keys=True)+'\n'; print(text,end='');
 if a.output: a.output.write_text(text)
 if errors: raise SystemExit(2)
if __name__=='__main__': main()
