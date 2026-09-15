#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.audit.read_text()); errors=[]; rows=x.get('rows',[]); ids=[r.get('identity') for r in rows]
 if x.get('schema_version')!='1.0': errors.append('unsupported schema_version')
 if len(ids)!=len(set(ids)): errors.append('duplicate registry identities')
 if len(rows)!=x.get('summary',{}).get('registrations'): errors.append('registration count mismatch')
 sites=x.get('overridable_sites',[]); site_ids=[(r.get('method_key'),r.get('target')) for r in sites]
 if len(site_ids)!=len(set(site_ids)): errors.append('duplicate overridable sites')
 if len(sites)!=x.get('summary',{}).get('overridable_sites'): errors.append('overridable site count mismatch')
 for r in rows:
  ident=r.get('identity')
  if r.get('findings') and r.get('disposition')!='approved-exception': errors.append(f'{ident} has unresolved findings')
  if not r.get('owner'): errors.append(f'{ident} missing owner')
  if not r.get('tests'): errors.append(f'{ident} missing focused tests')
  if r.get('disposition') not in {'compatible','approved-exception','obsolete'}: errors.append(f'{ident} unresolved disposition')
 result={'valid':not errors,'row_count':len(rows),'errors':errors}; text=json.dumps(result,indent=2,sort_keys=True)+'\n'; print(text,end='')
 if a.output: a.output.write_text(text)
 if errors: raise SystemExit(2)
if __name__=='__main__': main()
