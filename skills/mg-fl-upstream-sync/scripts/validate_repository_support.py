#!/usr/bin/env python3
import argparse,json
from pathlib import Path
S={'preserve','adapt','upstream','drop'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.audit.read_text()); e=[]
 for r in x.get('rows',[]):
  for f in ['owner','purpose','target_relationship','strategy']:
   if not r.get(f):e.append(f'missing {f} {r["id"]}')
  if r.get('strategy') and r['strategy'] not in S:e.append('invalid strategy '+r['id'])
  if not r.get('downstream_consumers') or not r.get('validation'):e.append('missing consumer/validation '+r['id'])
  if r['fork'].get('parse_error') or r['fork'].get('shell_valid') is False or r['fork'].get('fence_balanced') is False:e.append('format error '+r['id'])
 out={'valid':not e,'errors':e}; text=json.dumps(out,indent=2,sort_keys=True)+'\n'; print(text,end='')
 if a.output:a.output.write_text(text)
 if e:raise SystemExit(2)
if __name__=='__main__':main()
