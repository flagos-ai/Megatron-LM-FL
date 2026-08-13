#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ALLOWED={'compatible','approved-exception','obsolete'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--decisions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); audit=json.loads(a.audit.read_text()); decisions=json.loads(a.decisions.read_text()).get('decisions',[]); by={}
 for d in decisions:
  ident=d.get('identity')
  if not ident or ident in by: raise SystemExit('missing or duplicate decision identity')
  if d.get('disposition') not in ALLOWED or not d.get('owner') or not d.get('tests') or not d.get('reason') or not d.get('reviewer'): raise SystemExit(f'{ident} has incomplete decision')
  by[ident]=d
 ids={r['identity'] for r in audit.get('rows',[])}; unknown=sorted(set(by)-ids)
 if unknown: raise SystemExit('unknown decision identities: '+', '.join(unknown))
 out=dict(audit); rows=[]
 for source in audit.get('rows',[]):
  row=dict(source); d=by.get(row['identity'])
  if d:
   for key in ['disposition','owner','tests','reason','reviewer','blocked_test_owner']: row[key]=d.get(key)
  rows.append(row)
 out['rows']=rows; out['summary']=dict(out.get('summary',{})); out['summary']['unresolved']=sum(r.get('disposition') not in ALLOWED or not r.get('owner') or not r.get('tests') for r in rows)
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out['summary'],indent=2,sort_keys=True))
if __name__=='__main__': main()
