#!/usr/bin/env python3
import argparse,json
from pathlib import Path
DISP={'authorized','external-gate','not-applicable','superseded'}
STATUS={'not-run','pass','fail','blocked','skipped'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--matrix',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.matrix.read_text()); errors=[]; ids=[]
 if x.get('schema_version')!='1.0': errors.append('unsupported schema')
 for r in x.get('rows',[]):
  ids.append(r.get('id')); d=r.get('disposition'); e=r.get('execution',{}); status=e.get('status')
  if d not in DISP: errors.append(f"unreviewed row {r.get('id')}")
  if d in {'authorized','external-gate'} and (not r.get('owner') or not (r.get('command') or r.get('command_template'))): errors.append(f"missing owner/command {r.get('id')}")
  if status not in STATUS: errors.append(f"invalid status {r.get('id')}")
  if status=='pass' and (e.get('exit_code')!=0 or not e.get('log')): errors.append(f"unproven pass {r.get('id')}")
  if status in {'blocked','skipped'} and not r.get('reason'): errors.append(f"unexplained non-pass {r.get('id')}")
 if len(ids)!=len(set(ids)): errors.append('duplicate row ids')
 d=x.get('discovery',{})
 if d.get('covered_routes')!=d.get('test_hardware_routes') or d.get('uncovered_routes')!=0: errors.append('test-hardware route coverage incomplete')
 result={'valid':not errors,'errors':errors}; text=json.dumps(result,indent=2,sort_keys=True)+'\n'; print(text,end='')
 if a.output: a.output.write_text(text)
 if errors: raise SystemExit(2)
if __name__=='__main__': main()
