#!/usr/bin/env python3
import argparse,json
from pathlib import Path
S={'preserve','adapt','upstream','redesign','drop'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.audit.read_text()); e=[]; ids=[]
 if x.get('schema_version')!='1.0':e.append('unsupported schema')
 for r in x.get('rows',[]):
  ids.append(r.get('id'))
  for f in ['owner','requirement','target_relationship','strategy']:
   if not r.get(f):e.append(f'missing {f} {r["id"]}')
  if r.get('strategy') and r['strategy'] not in S:e.append('invalid strategy '+r['id'])
  if not r.get('validation_gates') and not r.get('external_gate'):e.append('missing validation gate '+r['id'])
  if r['fork'].get('toml_error') or r['fork'].get('shell_valid') is False or r['fork'].get('python_error'):e.append('source syntax error '+r['id'])
 if len(ids)!=len(set(ids)):e.append('duplicate ids')
 out={'valid':not e,'errors':e}; text=json.dumps(out,indent=2,sort_keys=True)+'\n'; print(text,end='')
 if a.output:a.output.write_text(text)
 if e:raise SystemExit(2)
if __name__=='__main__':main()
