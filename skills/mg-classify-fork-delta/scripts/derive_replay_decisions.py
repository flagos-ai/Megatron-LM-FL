#!/usr/bin/env python3
"""Derive conservative, generic replay candidates from classifier artifacts."""
import argparse,json,subprocess,hashlib
from collections import Counter,defaultdict
from pathlib import Path
SUPPORT={"cicd-common","cicd-vendor","tests-common","tests-vendor","build-packaging","examples-tools-docs","repository-metadata"}
RUNTIME={"invasive-runtime","training"}
def git(repo,*a,check=True):
 p=subprocess.run(["git","-C",str(repo),*a],text=True,capture_output=True)
 if check and p.returncode: raise SystemExit(p.stderr.strip())
 return p.stdout.rstrip("\n")
def oid(repo,ref,path): return git(repo,"rev-parse",f"{ref}:{path}",check=False).strip() or None
def commits(repo,base,fork,path): return git(repo,"log","--format=%H",f"{base}..{fork}","--",path).splitlines()
def decide(x,repo,refs,by_commit):
 p=x["path"]; base,fork,target=(refs[k] for k in ("sync_tree_base","fork","target")); bo,fo,to=(oid(repo,r,p) for r in (base,fork,target)); cs=commits(repo,base,fork,p)
 prov=[e for c in cs for e in by_commit.get(c,[])]; covered=[e for e in prov if e.get("target_candidates")]; uncovered=[e for e in prov if not e.get("target_candidates")]; independent=[c for c in cs if c not in by_commit]
 evidence={"base_oid":bo,"fork_oid":fo,"target_oid":to,"touching_commits":cs,"covered_provenance":covered,"uncovered_provenance":uncovered,"independent_fork_commits":independent}
 if fo and to and fo==to: return "UPSTREAM_COVERS","high","identical fork/target blob",evidence
 if not fo and not to: return "UPSTREAM_COVERS","high","absent from fork and target",evidence
 if x["category"].startswith("plugin-") and fo and not to: return "REPLAY_FL","high","fork plugin path absent from target",evidence
 if not x["both_changed"]: return "REPLAY_FL","medium","only fork changed path",evidence
 if uncovered and x["category"] in RUNTIME: return "REDESIGN","high","upstream-derived fork work absent from target on changed runtime path",evidence
 if covered: return "UPSTREAM_PLUS_FL_DELTA","medium","target has provenance candidate; verify and replay only independent FL delta",evidence
 if x["category"] in SUPPORT: return "UPSTREAM_PLUS_FL_DELTA","medium","both sides changed support/build/test surface",evidence
 if x["category"] in RUNTIME: return "REDESIGN","medium","both sides changed runtime semantics without equivalence proof",evidence
 return "MANUAL","low","no safe generic rule",evidence
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--inventory",type=Path,required=True); ap.add_argument("--provenance",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
 inv=json.loads(a.inventory.read_text()); raw=json.loads(a.provenance.read_text()); entries=raw.get("entries",[]); by=defaultdict(list)
 for e in entries: by[e.get("fork_commit")].append(e)
 rows=[]
 for x in inv["fork_changes"]:
  action,confidence,reason,evidence=decide(x,a.repo.resolve(),inv["resolved_refs"],by); group_source=sorted(evidence["touching_commits"]); group="commits-"+hashlib.sha256("|".join(group_source or [x["path"]]).encode()).hexdigest()[:10]
  rows.append({**x,"candidate_action":action,"confidence":confidence,"reason":reason,"evidence":evidence,"group_id":group})
 counts=Counter(x["candidate_action"] for x in rows); result={"schema_version":"1.0","refs":inv["resolved_refs"],"counts":dict(counts),"manual_count":counts.get("MANUAL",0),"decisions":rows}; a.output.mkdir(parents=True,exist_ok=True)
 (a.output/"candidate-decisions.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 md=["# Generic Candidate Replay Decisions","","Evidence-based candidates; not approval.","","| Action | Count |","|---|---:|",*[f"| {k} | {v} |" for k,v in sorted(counts.items())],"","| ID | Group | Action | Confidence | Path |","|---|---|---|---|---|",*[f"| {x['id']} | {x['group_id']} | {x['candidate_action']} | {x['confidence']} | `{x['path']}` |" for x in rows]]
 (a.output/"candidate-decisions.md").write_text("\n".join(md)+"\n"); print(json.dumps({"counts":dict(counts),"manual":counts.get("MANUAL",0),"output":str(a.output)},sort_keys=True))
 if counts.get("MANUAL"): raise SystemExit(2)
if __name__=="__main__": main()
