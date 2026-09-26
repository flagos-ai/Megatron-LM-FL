#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ALLOWED={'owner','disposition','environment','command','reason','evidence'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--matrix',type=Path,required=True); p.add_argument('--decisions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 matrix=json.loads(a.matrix.read_text()); dec=json.loads(a.decisions.read_text())
 if matrix.get('refs')!=dec.get('refs'): raise SystemExit('artifact refs mismatch')
 rows={x['id']:x for x in matrix['rows']}
 for key,value in dec.get('decisions',{}).items():
  if key not in rows: raise SystemExit('unknown decision row: '+key)
  bad=set(value)-ALLOWED
  if bad: raise SystemExit('unsupported decision fields: '+','.join(sorted(bad)))
  rows[key].update(value)
 matrix['decision_summary']={'applied':len(dec.get('decisions',{})),'source':str(a.decisions)}
 a.output.write_text(json.dumps(matrix,indent=2,sort_keys=True)+'\n')
 print(json.dumps(matrix['decision_summary'],sort_keys=True))
if __name__=='__main__': main()
