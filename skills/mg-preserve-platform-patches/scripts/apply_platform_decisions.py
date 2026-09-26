#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ALLOWED_PLATFORM={'compatible','approved-exception'}; ALLOWED_OCC={'portable-abstraction','intentional-backend','build-test-only','approved-exception'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--decisions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); x=json.loads(a.audit.read_text()); d=json.loads(a.decisions.read_text()); platforms={r['class']:r for r in d.get('platforms',[])}; occ={r['occurrence_id']:r for r in d.get('occurrences',[])}
 if len(platforms)!=len(d.get('platforms',[])) or len(occ)!=len(d.get('occurrences',[])): raise SystemExit('duplicate decision identity')
 out=dict(x); out['platforms']=[]
 for source in x.get('platforms',[]):
  row=dict(source); dec=platforms.get(row['class']);
  if dec:
   if dec.get('disposition') not in ALLOWED_PLATFORM or not dec.get('owner') or not dec.get('tests') or not dec.get('reason'): raise SystemExit(f"incomplete platform decision {row['class']}")
   row.update({k:dec[k] for k in ['disposition','owner','tests','reason']})
  out['platforms'].append(row)
 out['raw_device_occurrences']=[]
 for source in x.get('raw_device_occurrences',[]):
  row=dict(source); dec=occ.get(row['occurrence_id'])
  if dec:
   if dec.get('classification') not in ALLOWED_OCC or not dec.get('owner') or not dec.get('tests') or not dec.get('reason'): raise SystemExit(f"incomplete occurrence decision {row['occurrence_id']}")
   row.update({k:dec[k] for k in ['classification','owner','tests','reason']})
  out['raw_device_occurrences'].append(row)
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
