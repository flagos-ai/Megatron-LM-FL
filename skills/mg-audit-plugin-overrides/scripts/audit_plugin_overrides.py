#!/usr/bin/env python3
"""Statically audit Megatron plugin override registrations across Git refs."""

import argparse, ast, csv, io, json, subprocess
from pathlib import Path


def git(repo: Path, *args: str, check=True) -> str:
    p=subprocess.run(["git","-C",str(repo),*args],text=True,encoding="utf-8",errors="replace",capture_output=True)
    if check and p.returncode: raise SystemExit(p.stderr.strip() or f"git {' '.join(args)} failed")
    return p.stdout


def load(path: Path) -> dict:
    x=json.loads(path.read_text())
    if not isinstance(x,dict): raise SystemExit(f"invalid JSON object: {path}")
    return x


def literal(node, default=None):
    try: return ast.literal_eval(node)
    except Exception: return default


def dotted_name(node):
    parts=[]
    while isinstance(node,ast.Attribute): parts.append(node.attr); node=node.value
    if isinstance(node,ast.Name): parts.append(node.id)
    return ".".join(reversed(parts))


def signature(node):
    if isinstance(node,ast.ClassDef):
        init=next((x for x in node.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=="__init__"),None)
        return signature(init) if init else {"kind":"class","parameters":[],"inherits_constructor":True,"bases":[ast.unparse(x) for x in node.bases]}
    if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)): return None
    a=node.args; positional=a.posonlyargs+a.args; default_start=len(positional)-len(a.defaults); params=[]
    for i,arg in enumerate(positional):
        params.append({"name":arg.arg,"kind":"posonly" if i<len(a.posonlyargs) else "positional","annotation":ast.unparse(arg.annotation) if arg.annotation else None,"default":ast.unparse(a.defaults[i-default_start]) if i>=default_start else None})
    if a.vararg: params.append({"name":a.vararg.arg,"kind":"vararg","annotation":ast.unparse(a.vararg.annotation) if a.vararg.annotation else None,"default":None})
    for arg,default in zip(a.kwonlyargs,a.kw_defaults): params.append({"name":arg.arg,"kind":"kwonly","annotation":ast.unparse(arg.annotation) if arg.annotation else None,"default":ast.unparse(default) if default else None})
    if a.kwarg: params.append({"name":a.kwarg.arg,"kind":"kwarg","annotation":ast.unparse(a.kwarg.annotation) if a.kwarg.annotation else None,"default":None})
    return {"kind":"async-function" if isinstance(node,ast.AsyncFunctionDef) else "function","parameters":params,"returns":ast.unparse(node.returns) if node.returns else None,"decorators":[dotted_name(d.func) if isinstance(d,ast.Call) else dotted_name(d) for d in node.decorator_list]}


def scope_definitions(body):
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While, ast.Match)):
            for field in ("body", "orelse", "finalbody"):
                yield from scope_definitions(getattr(node, field, []))
            for handler in getattr(node, "handlers", []):
                yield from scope_definitions(handler.body)
            for case in getattr(node, "cases", []):
                yield from scope_definitions(case.body)


def find_qual(tree, parts):
    body=tree.body; node=None
    for part in parts:
        node=next((x for x in scope_definitions(body) if x.name==part),None)
        if node is None: return None
        body=getattr(node,"body",[])
    return node


def signature_contract(value):
    if not value: return value
    return {key: value.get(key) for key in ("kind", "parameters", "returns")}


def resolve(repo,ref,dotted,cache):
    parts=dotted.split(".")
    for i in range(len(parts)-1,0,-1):
        module="/".join(parts[:i]); candidates=[module+".py",module+"/__init__.py"]
        for path in candidates:
            key=(ref,path)
            if key not in cache:
                text=git(repo,"show",f"{ref}:{path}",check=False); cache[key]=text if text else None
            text=cache[key]
            if text is None: continue
            try: tree=ast.parse(text)
            except SyntaxError: return {"path":path,"status":"syntax-error","kind":None,"signature":None}
            node=find_qual(tree,parts[i:])
            if node:
                kind="class" if isinstance(node,ast.ClassDef) else "async-function" if isinstance(node,ast.AsyncFunctionDef) else "function"
                return {"path":path,"status":"found","kind":kind,"signature":signature(node),"line":node.lineno,"bases":[ast.unparse(x) for x in node.bases] if isinstance(node,ast.ClassDef) else []}
    return {"path":None,"status":"missing","kind":None,"signature":None}


def extract_registrations(repo,ref):
    paths=git(repo,"grep","-l","-e","register(","-e","@override(",ref,"--","megatron/plugin/*.py","megatron/plugin/**/*.py",check=False).splitlines()
    rows=[]
    for entry in sorted(set(paths)):
        path=entry.split(":",1)[1] if entry.startswith(ref+":") else entry
        if "/tests/" in path: continue
        text=git(repo,"show",f"{ref}:{path}",check=False)
        if not text: continue
        try: tree=ast.parse(text)
        except SyntaxError: continue
        module=path[:-3].replace("/",".") if path.endswith(".py") else path.replace("/",".")
        parents=[]
        def walk(node):
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)): parents.append(node.name)
            if isinstance(node,ast.Call) and dotted_name(node.func).split(".")[-1]=="register":
                kw={x.arg:x.value for x in node.keywords if x.arg}
                target=literal(kw.get("target")); impl=literal(kw.get("impl")); vendor=literal(kw.get("vendor"),"default")
                rows.append({"style":"centralized","target":target,"implementation":impl,"vendor":str(vendor).lower() if vendor is not None else None,"source":path,"line":node.lineno,"dynamic":not isinstance(target,str) or not isinstance(impl,str)})
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if isinstance(dec,ast.Call) and dotted_name(dec.func).split(".")[-1]=="override":
                        vals=[literal(x) for x in dec.args]; kw={x.arg:literal(x.value) for x in dec.keywords if x.arg}
                        key=f"{vals[0]}.{vals[1]}" if len(vals)>=2 and all(isinstance(x,str) for x in vals[:2]) else None
                        rows.append({"style":"eager-decorator","target":None,"method_key":key,"implementation":module+"."+".".join(parents),"vendor":str(kw.get("vendor","default")).lower(),"source":path,"line":node.lineno,"dynamic":key is None})
            for child in ast.iter_child_nodes(node): walk(child)
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)): parents.pop()
        walk(tree)
    return rows


def method_key(target):
    parts=target.rsplit(".",2); return ".".join(parts[-2:]) if len(parts)>=2 else target



def extract_overridable_sites(repo, ref, manifest):
    paths = sorted({row.get("path") for row in manifest if "overridable" in row.get("decorators", []) and row.get("path")})
    sites = []
    for path in paths:
        text = git(repo, "show", f"{ref}:{path}", check=False)
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        module = path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")
        parents = []
        def walk(node):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                parents.append(node.name)
                decorators = [dotted_name(d.func) if isinstance(d, ast.Call) else dotted_name(d) for d in node.decorator_list]
                if "overridable" in {name.split(".")[-1] for name in decorators}:
                    full = module + "." + ".".join(parents)
                    sites.append({"target": full, "method_key": method_key(full), "path": path, "line": node.lineno, "kind": "class" if isinstance(node, ast.ClassDef) else "async-function" if isinstance(node, ast.AsyncFunctionDef) else "function", "decorators": decorators})
            for child in ast.iter_child_nodes(node):
                walk(child)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                parents.pop()
        walk(tree)
    return sorted(sites, key=lambda x: (x["method_key"], x["target"]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--inventory",type=Path,required=True); ap.add_argument("--routing",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    inv=load(a.inventory); routing=load(a.routing); refs=inv.get("resolved_refs",{})
    if routing.get("refs")!=refs: raise SystemExit("inventory/routing refs mismatch")
    for name in ["sync_tree_base","fork","target"]:
        if not refs.get(name): raise SystemExit(f"missing {name}")
    regs=extract_registrations(a.repo.resolve(),refs["fork"]); sites=extract_overridable_sites(a.repo.resolve(),refs["fork"],inv.get("override_manifest",[])); cache={}; identities={}; rows=[]
    for reg in regs:
        target=reg.get("target"); key=reg.get("method_key") or (method_key(target) if isinstance(target,str) else None); identity=(key,reg.get("vendor")); duplicate=identity in identities; identities[identity]=identities.get(identity,0)+1
        target_states={name:resolve(a.repo.resolve(),refs[name],target,cache) if isinstance(target,str) else None for name in ["sync_tree_base","fork","target"]}
        impl=resolve(a.repo.resolve(),refs["fork"],reg.get("implementation"),cache) if isinstance(reg.get("implementation"),str) else None
        reasons=[]
        if reg.get("dynamic"): reasons.append("unsupported-dynamic-registration")
        if duplicate: reasons.append("duplicate-registration")
        if target_states["fork"] and target_states["fork"]["status"]!="found": reasons.append("missing-fork-target")
        if target_states["target"] and target_states["target"]["status"]!="found": reasons.append("missing-target-release-target")
        if impl and impl["status"]!="found": reasons.append("missing-implementation")
        fork_sig=target_states["fork"]["signature"] if target_states["fork"] else None; target_sig=target_states["target"]["signature"] if target_states["target"] else None; impl_sig=impl["signature"] if impl else None
        if fork_sig and target_sig and signature_contract(fork_sig)!=signature_contract(target_sig): reasons.append("target-signature-drift")
        if target_sig and impl_sig:
            inherited = impl.get("kind") == "class" and target_states["target"].get("kind") == "class" and any(base.split(".")[-1] == target.rsplit(".", 1)[-1] for base in impl.get("bases", [])) and impl_sig.get("inherits_constructor")
            if not inherited and signature_contract(target_sig) != signature_contract(impl_sig): reasons.append("implementation-signature-mismatch")
        rows.append({**reg,"method_key":key,"identity":f"{key}@{reg.get('vendor')}","target_states":target_states,"implementation_state":impl,"findings":sorted(set(reasons)),"disposition":"compatible" if not reasons else "manual","owner":None,"tests":[]})
    rows.sort(key=lambda x:(str(x.get("method_key")),str(x.get("vendor")),str(x.get("target")),str(x.get("implementation"))))
    registered_keys={row["method_key"] for row in rows}; site_keys={site["method_key"] for site in sites}
    for site in sites: site["registration_status"]="registered" if site["method_key"] in registered_keys else "fallback-only"
    vendors=sorted({x.get("vendor") for x in rows if x.get("vendor")})
    out={"schema_version":"1.0","refs":refs,"rows":rows,"overridable_sites":sites,"vendors":vendors,"summary":{"overridable_sites":len(sites),"registered_sites":sum(site["registration_status"]=="registered" for site in sites),"fallback_only_sites":sum(site["registration_status"]=="fallback-only" for site in sites),"registry_keys_without_site":len(registered_keys-site_keys),"registrations":len(rows),"vendors":len(vendors),"duplicates":sum("duplicate-registration" in x["findings"] for x in rows),"missing_targets":sum(any(y in x["findings"] for y in ["missing-fork-target","missing-target-release-target"]) for x in rows),"missing_implementations":sum("missing-implementation" in x["findings"] for x in rows),"signature_reviews":sum(any("signature" in y for y in x["findings"]) for x in rows),"manual":sum(x["disposition"]=="manual" for x in rows)}}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/"override-audit.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    with (a.output/"override-matrix.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(["identity","target","implementation","style","findings","disposition"])
        for x in rows: w.writerow([x["identity"],x.get("target"),x.get("implementation"),x["style"],",".join(x["findings"]),x["disposition"]])
    print(json.dumps(out["summary"],indent=2,sort_keys=True))


if __name__=="__main__": main()
