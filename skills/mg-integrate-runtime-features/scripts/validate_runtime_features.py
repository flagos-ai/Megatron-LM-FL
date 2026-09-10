#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ALLOWED={'TARGET_COVERS','TARGET_PLUS_FL_DELTA','REPLAY_FL_NARROW','REDESIGN_APPROVED'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--features',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args(); x=json.loads(a.features.read_text()); errors=[]; seen=[]
 if x.get('schema_version')!='1.0': errors.append('unsupported schema')
 for f in x.get('features',[]):
  ids=[m.get('delta_id') for m in f.get('members',[])]; seen+=ids
  if set(ids)!=set(f.get('delta_ids',[])): errors.append(f"member mismatch {f.get('feature_id')}")
  for key in ['feature_name','invariant','target_change','owner','reviewer']: 
   if not f.get(key): errors.append(f"{f.get('feature_id')} missing {key}")
  if f.get('strategy') not in ALLOWED: errors.append(f"{f.get('feature_id')} invalid strategy")
  if not f.get('observing_tests'): errors.append(f"{f.get('feature_id')} missing observing tests")
  if not f.get('stage_rationale',{}).get('execution'): errors.append(f"{f.get('feature_id')} missing execution semantics")
  if f.get('contains_redesign') and (f.get('strategy')!='REDESIGN_APPROVED' or f.get('approval_status')!='approved'): errors.append(f"{f.get('feature_id')} redesign is not approved")
  for t in f.get('observing_tests',[]):
   if isinstance(t,dict) and t.get('status')=='blocked' and not t.get('external_owner'): errors.append(f"{f.get('feature_id')} blocked test lacks owner")
 if len(seen)!=len(set(seen)) or len(seen)!=x.get('summary',{}).get('runtime_routes'): errors.append('runtime delta coverage mismatch')
 result={'valid':not errors,'errors':errors}; text=json.dumps(result,indent=2,sort_keys=True)+'\n'; print(text,end='');
 if a.output: a.output.write_text(text)
 if errors: raise SystemExit(2)
if __name__=='__main__': main()
