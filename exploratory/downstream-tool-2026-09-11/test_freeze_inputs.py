import json
import tempfile
import unittest
from pathlib import Path

import build_inputs
import freeze_inputs


def config():
    readers = [{"reader_config_id": f"reader-{n}", "model": f"model-{n}", "provider": "pinned-provider",
                "served_model_expectation": f"model-{n}", "temperature": 0, "max_output_tokens": 24} for n in range(1, 4)]
    readers[0]["reasoning"] = "low"
    return {"reader_configs": readers, "thresholds": {"fresh_execute": 0.7, "verify": 0.7, "parse": 0.95},
            "success_rule": "predeclared", "saturation_rule": "predeclared", "retry_rule": "one logged retry",
            "api_settings": {"endpoint": "freeze-time", "tool_choice": "required"},
            "price_snapshot": {"catalog_timestamp": "2026-09-12T00:00:00Z", "prices": {f"model-{n}": {"input_per_million": 1, "output_per_million": 2} for n in range(1, 4)}},
            "hard_cost_cap_usd": 10, "user_authorization": {"authorized": True, "date": "2026-09-12"}, "no_target_before_gate": True}


class FreezeCompilerTest(unittest.TestCase):
    def test_cross_product_manifest_hashes_and_determinism(self):
        root = Path(__file__).resolve().parents[2]
        source = root / "exploratory/unrecoverability-2026-09-08/reader_g_inputs.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); generated = tmp / "generated"; build_inputs.build(source, generated)
            cfg = tmp / "config.json"; cfg.write_text(json.dumps(config()))
            first = freeze_inputs.compile_freeze(generated / "inputs.jsonl", cfg, tmp / "one")
            second = freeze_inputs.compile_freeze(generated / "inputs.jsonl", cfg, tmp / "two")
            controls = freeze_inputs.load_jsonl(tmp / "one/control.jsonl")
            targets = freeze_inputs.load_jsonl(tmp / "one/target.jsonl")
            self.assertEqual((len(controls), len(targets)), (270, 414))
            self.assertEqual({x["stage"] for x in controls}, {"control"})
            self.assertEqual({x["stage"] for x in targets}, {"target"})
            self.assertTrue(all(x["reasoning"] == "low" for x in controls + targets if x["reader_config_id"] == "reader-1"))
            self.assertTrue(all("reasoning" not in x for x in controls + targets if x["reader_config_id"] != "reader-1"))
            self.assertEqual(first["hashes"]["frozen_control"], second["hashes"]["frozen_control"])
            self.assertEqual(first["hashes"]["frozen_target"], second["hashes"]["frozen_target"])
            self.assertEqual(first["hashes"]["runner_prices"], second["hashes"]["runner_prices"])
            self.assertEqual(json.loads((tmp / "one/prices.json").read_text()), config()["price_snapshot"]["prices"])
            self.assertEqual(sorted(x["frozen_call_order"] for x in controls + targets), list(range(1, 685)))
            self.assertNotIn("freeze_manifest", first["hashes"])

    def test_rejects_non_three_readers_and_cost_over_cap(self):
        bad = config(); bad["reader_configs"] = bad["reader_configs"][:2]
        with self.assertRaisesRegex(ValueError, "exactly three"):
            freeze_inputs.validate_config(bad)
        bad = config(); bad["hard_cost_cap_usd"] = 0.0000001
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); generated = tmp / "generated"; build_inputs.build(root / "exploratory/unrecoverability-2026-09-08/reader_g_inputs.jsonl", generated)
            cfg = tmp / "config.json"; cfg.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "exceeds hard cap"):
                freeze_inputs.compile_freeze(generated / "inputs.jsonl", cfg, tmp / "frozen")


if __name__ == "__main__":
    unittest.main()
