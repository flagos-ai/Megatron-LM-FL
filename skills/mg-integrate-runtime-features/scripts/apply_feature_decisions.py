#!/usr/bin/env python3
import argparse,json
from pathlib import Path
REQUIRED=['feature_id','feature_name','delta_ids','invariant','target_change','strategy','owner','stage_rationale','dependencies','observing_tests','reviewer','approval_status']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--decisions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); c=json.loads(a.candidate.read_text()); ds=json.loads(a.decisions.read_text()).get('features',[]); facts={x['delta_id']:x for x in c.get('delta_facts',[])}; seen=set(); features=[]
 for d in ds:
  missing=[k for k in REQUIRED if k not in d or d.get(k) is None or (k not in {'dependencies'} and not d.get(k))]
  if missing: raise SystemExit(f"incomplete feature decision: {','.join(missing)}")
  ids=d['delta_ids']; unknown=set(ids)-set(facts); overlap=set(ids)&seen
  if unknown or overlap or len(ids)!=len(set(ids)): raise SystemExit('unknown, duplicate, or overlapping feature members')
  seen.update(ids); row=dict(d); row['members']=[facts[x] for x in sorted(ids)]; row['contains_redesign']=any(facts[x]['action']=='REDESIGN' for x in ids); features.append(row)
 missing=sorted(set(facts)-seen)
 if missing: raise SystemExit('unassigned runtime deltas: '+', '.join(missing))
 out={'schema_version':'1.0','refs':c['refs'],'features':sorted(features,key=lambda x:x['feature_id']),'summary':{'runtime_routes':len(facts),'reviewed_features':len(features)}}; a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
