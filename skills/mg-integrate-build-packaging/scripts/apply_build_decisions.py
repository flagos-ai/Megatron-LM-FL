#!/usr/bin/env python3
import argparse,json
from pathlib import Path
FIELDS={'owner','requirement','target_relationship','strategy','validation_gates','external_gate','reason','evidence'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--decisions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); x=json.loads(a.audit.read_text()); d=json.loads(a.decisions.read_text())
 if x.get('refs')!=d.get('refs'):raise SystemExit('artifact refs mismatch')
 rows={r['id']:r for r in x['rows']}
 for k,v in d.get('decisions',{}).items():
  if k not in rows:raise SystemExit('unknown row: '+k)
  if set(v)-FIELDS:raise SystemExit('unsupported fields')
  rows[k].update(v)
 x['decision_summary']={'applied':len(d.get('decisions',{}))}; a.output.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
