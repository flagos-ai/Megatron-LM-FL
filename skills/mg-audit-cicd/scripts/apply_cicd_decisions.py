#!/usr/bin/env python3
import argparse,json
from pathlib import Path
FIELDS={'owner','disposition','reason','evidence','tests'}
GROUPS={'references':lambda x:f"{x['source']}:{x['line']}:{x['reference']}",'failure_masks':lambda x:f"{x['path']}:{x.get('line',x.get('job',''))}:{x['kind']}",'backend_coverage':lambda x:x['backend'],'shell_checks':lambda x:x['path']}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--decisions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 audit=json.loads(a.audit.read_text()); decisions=json.loads(a.decisions.read_text())
 if decisions.get('refs')!=audit.get('refs'): raise SystemExit('artifact refs mismatch')
 supplied=decisions.get('decisions',{}); unknown=set(supplied)-set(GROUPS)
 if unknown: raise SystemExit('unknown decision groups: '+','.join(sorted(unknown)))
 applied=0
 for group,key_fn in GROUPS.items():
  rows={key_fn(row):row for row in audit.get(group,[])}
  for key,decision in supplied.get(group,{}).items():
   if key not in rows: raise SystemExit(f'unknown {group} decision key: {key}')
   bad=set(decision)-FIELDS
   if bad: raise SystemExit(f'unsupported decision fields for {key}: '+','.join(sorted(bad)))
   rows[key].update(decision); applied+=1
 audit['decision_summary']={'applied':applied,'source':str(a.decisions)}
 a.output.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n'); print(json.dumps(audit['decision_summary'],sort_keys=True))
if __name__=='__main__': main()
