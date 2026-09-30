import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import build_inputs

class BuilderTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.source=Path(__file__).resolve().parents[2]/"exploratory/unrecoverability-2026-09-08/reader_g_inputs.jsonl"
 def rows(self):
  t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup); summary=build_inputs.build(self.source,Path(t.name)); return summary,[json.loads(x) for x in (Path(t.name)/"inputs.jsonl").read_text().splitlines()]
 def test_targets_and_actual_forms(self):
  summary,rows=self.rows(); target=[x for x in rows if x["kind"]=="target"]
  self.assertEqual((summary["candidate_clusters"],len(target),summary["source_actual_forms"]),(69,138,{"progressive":28,"simple":41}))
  self.assertEqual(Counter(x["source_actual_form"] for x in target),{"progressive":56,"simple":82})
 def test_pair_text_has_only_declared_edit(self):
  _,rows=self.rows(); pairs={}
  for x in rows:
   if x["kind"]=="target": pairs.setdefault(x["pair_id"],[]).append(x)
  self.assertEqual(len(pairs),69)
  for pair in pairs.values(): self.assertEqual([x["form"] for x in pair],["progressive","simple"]); build_inputs.assert_pair_edit_only(*pair)
  for pair in pairs.values():
   actual = next(x for x in pair if x["form"] == x["source_actual_form"])
   other = next(x for x in pair if x is not actual)
   self.assertEqual(actual["memory_note"], actual["source_note"])
   edit = actual["declared_edit"]
   expected = actual["source_note"].replace(edit["from"], edit["to"]) if actual["form"] == "progressive" else actual["source_note"].replace(edit["to"], edit["from"])
   self.assertEqual(other["memory_note"], expected)
 def test_controls_are_synthetic_disjoint_and_balanced(self):
  summary,rows=self.rows(); targets=[x for x in rows if x["kind"]=="target"]; controls=[x for x in rows if x["kind"]=="calibration"]
  self.assertEqual((summary["calibration_contents"],len(controls)),(30,90)); self.assertTrue(all(x["synthetic"] for x in controls))
  self.assertFalse({x["witness"].casefold() for x in targets}&{x["witness"].casefold() for x in controls})
  root=Path(__file__).resolve().parent
  r0={json.loads(x)["witness"].casefold() for x in (root/"generated-r0/inputs.jsonl").read_text().splitlines() if json.loads(x)["kind"]=="calibration"}
  self.assertFalse(r0&{x["witness"].casefold() for x in controls})
  r1={json.loads(x)["witness"].casefold() for x in (root/"generated-r1/inputs.jsonl").read_text().splitlines() if json.loads(x)["kind"]=="calibration"}
  self.assertFalse(r1&{x["witness"].casefold() for x in controls})
  dev={w.casefold() for w in build_inputs.DEV.values()}
  self.assertFalse(dev&({x["witness"].casefold() for x in controls}|{x["witness"].casefold() for x in targets}|r0|r1))
  counts=Counter((x["frame"],x["control"]) for x in controls); self.assertEqual(len(counts),18); self.assertEqual(set(counts.values()),{5})
  for x in controls:
   self.assertEqual(x["expected_first_tool"],"EXECUTE" if x["control"]=="fresh" else "VERIFY")
   if x["control"]=="expired_bounded": self.assertIn("This was true only through 2026-09-07.",x["memory_note"])
   if x["control"]=="fresh": self.assertEqual(x["written"], "2026-09-08"); self.assertTrue(x["memory_note"].startswith("As of 2026-09-08,"))
 def test_openai_function_schemas(self):
  _,rows=self.rows()
  for row in rows:
   names=[x["function"]["name"] for x in row["tools"]]; self.assertIn(row["execute_tool"],names); self.assertIn(row["verify_tool"],names)
   for tool in row["tools"]:
    self.assertEqual(tool["type"],"function"); p=tool["function"]["parameters"]; self.assertEqual(p["type"],"object"); self.assertFalse(p["additionalProperties"]); self.assertEqual(set(p["properties"]),set(p["required"]))
   verify=next(x for x in row["tools"] if x["function"]["name"]==row["verify_tool"])
   self.assertEqual(verify["function"]["parameters"]["properties"]["field"]["enum"], [row["target_field"]])
 def test_r2_action_tool_needs_only_target_field_and_policy_names_it(self):
  _,rows=self.rows()
  for row in rows:
   execute=next(x for x in row["tools"] if x["function"]["name"]==row["execute_tool"])
   self.assertEqual(execute["function"]["parameters"]["required"],[row["target_field"]])
   self.assertEqual(row["expected_execute_arguments"],{row["target_field"]:row["witness"]})
   self.assertIn(f"by calling {row['execute_tool']} ",row["system"]); self.assertIn("call request_confirmation instead",row["system"])
   self.assertNotIn("first operational step",row["system"])
if __name__=="__main__": unittest.main()
