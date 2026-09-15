#!/usr/bin/env python3
import importlib.util, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/file); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
classifier=load("classifier","classify_fork_delta.py")
decisions=load("decisions","derive_replay_decisions.py")
class ClassifierTests(unittest.TestCase):
 def test_dynamic_vendor_category(self):
  self.assertEqual(classifier.category("megatron/plugin/acme/x.py",{"acme"},{"chipx"}),"plugin-vendor")
  self.assertEqual(classifier.category("megatron/plugin/acme/x.py",set(),set()),"plugin-contract")
 def test_marker_anomaly_is_inventory_data(self):
  blocks, anomalies=classifier.marker_blocks("x.py","# FlagScale Begin\nx=1\n")
  self.assertEqual(blocks,[]); self.assertEqual(len(anomalies),1)
 def test_identical_blob_decision_contract(self):
  self.assertIn("UPSTREAM_COVERS",decisions.decide.__code__.co_consts)
 def test_no_upgrade_specific_tokens(self):
  text="\n".join((ROOT/name).read_text() for name in ("classify_fork_delta.py","derive_replay_decisions.py"))
  for token in ("core_"+"v0.18.2","fa49"+"08745","471"+"737e","5fbc"+"23a","MGD-"+"0030"):
   self.assertNotIn(token,text)
if __name__=="__main__": unittest.main()
