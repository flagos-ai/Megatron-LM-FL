#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.audit.read_text()); errors=[]
 if x.get('schema_version')!='1.0': errors.append('unsupported schema')
 if len(x.get('cicd_route_ids',[]))!=len(set(x.get('cicd_route_ids',[]))): errors.append('duplicate CI routes')
 for w in x.get('workflows',[]):
  if w.get('parse_error'): errors.append(f"YAML parse error {w['path']}")
 for r in x.get('references',[]):
  if (r.get('dynamic') or not r.get('exists')) and (r.get('disposition') not in {'approved-dynamic','approved-external','removed'} or not r.get('owner')): errors.append(f"unresolved reference {r['source']}:{r['line']}")
 for r in x.get('backend_coverage',[]):
  if not r.get('has_config') and (r.get('disposition')!='approved-exclusion' or not r.get('owner')): errors.append(f"backend gap {r['backend']}")
 for r in x.get('failure_masks',[]):
  if r.get('disposition') not in {'approved','removed'} or not r.get('owner'): errors.append(f"unreviewed failure mask {r['path']}")
 for r in x.get('shell_checks',[]):
  if not r.get('valid') and (r.get('disposition')!='approved-blocker' or not r.get('owner')): errors.append(f"shell syntax error {r['path']}")
 result={'valid':not errors,'errors':errors}; text=json.dumps(result,indent=2,sort_keys=True)+'\n'; print(text,end='');
 if a.output: a.output.write_text(text)
 if errors: raise SystemExit(2)
if __name__=='__main__': main()
