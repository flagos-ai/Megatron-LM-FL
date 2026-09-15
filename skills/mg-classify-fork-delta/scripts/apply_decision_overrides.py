#!/usr/bin/env python3
"""Compose immutable generic candidates with evidence-backed reviewer overrides."""
import argparse,json
from collections import Counter
from pathlib import Path
ALLOWED={"UPSTREAM_COVERS","UPSTREAM_PLUS_FL_DELTA","REPLAY_FL","REDESIGN","MANUAL"}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--candidates",type=Path,required=True); ap.add_argument("--overrides",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
 data=json.loads(a.candidates.read_text()); raw=json.loads(a.overrides.read_text()); overrides=raw.get("overrides",[]); by_id={x["id"]:x for x in data["decisions"]}; seen=set(); errors=[]
 for o in overrides:
  ident=o.get("id"); required=("from_action","to_action","evidence_type","evidence_refs","reviewer","reason")
  if ident in seen: errors.append(f"duplicate override: {ident}"); continue
  seen.add(ident)
  if ident not in by_id: errors.append(f"unknown delta id: {ident}"); continue
  missing=[k for k in required if not o.get(k)];
  if missing: errors.append(f"{ident}: missing {','.join(missing)}"); continue
  row=by_id[ident]
  if o["from_action"]!=row["candidate_action"]: errors.append(f"{ident}: stale from_action"); continue
  if o["to_action"] not in ALLOWED: errors.append(f"{ident}: invalid to_action"); continue
  row["effective_action"]=o["to_action"]; row["review_override"]=o
 for row in by_id.values(): row.setdefault("effective_action",row["candidate_action"])
 if errors: print(json.dumps({"valid":False,"errors":errors},indent=2)); raise SystemExit(2)
 rows=[by_id[k] for k in sorted(by_id)]; counts=Counter(x["effective_action"] for x in rows); result={"schema_version":"1.0","refs":data["refs"],"counts":dict(counts),"override_count":len(overrides),"manual_count":counts.get("MANUAL",0),"decisions":rows}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"valid":True,"counts":dict(counts),"overrides":len(overrides)},sort_keys=True))
 if counts.get("MANUAL"): raise SystemExit(2)
if __name__=="__main__": main()
