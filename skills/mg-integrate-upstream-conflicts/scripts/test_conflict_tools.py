#!/usr/bin/env python3

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("builder", ROOT / "build_conflict_ledger.py")
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


class ConflictToolsTests(unittest.TestCase):
    def test_parse_textual_and_auto_merge(self):
        text = """changed in both
  base   100644 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa x.py
  our    100644 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb x.py
  their  100644 cccccccccccccccccccccccccccccccccccccccc x.py
@@
+<<<<<<< .our
changed in both
  base   100644 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa y.py
  our    100644 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb y.py
  their  100644 cccccccccccccccccccccccccccccccccccccccc y.py
"""
        parsed = builder.parse_merge_tree(text)
        self.assertEqual(parsed["x.py"]["textual_status"], "textual-conflict")
        self.assertEqual(parsed["y.py"]["textual_status"], "auto-merged-semantic-risk")

    def test_parser_tolerates_replacement_characters(self):
        text = "changed in both\n  base   100644 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa data.bin\n  our    100644 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb data.bin\n  their  100644 cccccccccccccccccccccccccccccccccccccccc data.bin\n\ufffd\n"
        parsed = builder.parse_merge_tree(text)
        self.assertEqual(parsed["data.bin"]["textual_status"], "auto-merged-semantic-risk")

    def test_resolution_composition_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = {"schema_version":"1.0","refs":{},"summary":{"both_changed":1},"rows":[{
                "delta_id":"MGD-1","path":"x.py","priority":"P0","action":"UPSTREAM_PLUS_FL_DELTA","status":"unresolved","approval_status":"unapproved",
                "owner":None,"fork_invariant":None,"upstream_change":None,"affected_symbols":[],"resolution_strategy":None,"acceptance_tests":[],"reviewer":None
            }]}
            override = {"resolutions":[{
                "delta_id":"MGD-1","owner":"team","fork_invariant":"behavior","upstream_change":"refactor",
                "affected_symbols":["f"],"resolution_strategy":"TARGET_PLUS_FL_DELTA",
                "acceptance_tests":[{"name":"test_f","status":"planned"}],"reviewer":"reviewer",
                "approval_status":"approved","reason":"preserve both"
            }]}
            (root/"candidate.json").write_text(json.dumps(candidate)); (root/"overrides.json").write_text(json.dumps(override))
            compose=subprocess.run(["python",str(ROOT/"apply_conflict_resolutions.py"),"--candidate",str(root/"candidate.json"),"--overrides",str(root/"overrides.json"),"--output",str(root/"effective.json")],capture_output=True,text=True)
            self.assertEqual(compose.returncode,0,compose.stderr)
            validate=subprocess.run(["python",str(ROOT/"validate_conflict_ledger.py"),"--ledger",str(root/"effective.json")],capture_output=True,text=True)
            self.assertEqual(validate.returncode,0,validate.stdout+validate.stderr)

    def test_unresolved_ledger_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"ledger.json"
            path.write_text(json.dumps({"schema_version":"1.0","summary":{"both_changed":1},"rows":[{"delta_id":"MGD-1","priority":"P0","status":"unresolved"}]}))
            proc=subprocess.run(["python",str(ROOT/"validate_conflict_ledger.py"),"--ledger",str(path)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,2)
            self.assertIn("missing or invalid strategy",proc.stdout)


if __name__ == "__main__": unittest.main()
