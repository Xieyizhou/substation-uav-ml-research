import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import unittest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/research/ml_training_recovery_v1/hard-negative-isolated-v2"

class HardNegativeV2Tests(unittest.TestCase):
    def test_matrix_is_paired_and_not_admitted(self):
        matrix=json.loads((BASE/"matrix.json").read_text());ledger=json.loads((BASE/"intake-ledger.json").read_text())
        self.assertEqual(matrix["counts"],{"poses":24,"lights_per_pose":2,"frames":48})
        self.assertEqual(len(ledger["entries"]),48)
        self.assertEqual(set(Counter(r["derivation_group"] for r in ledger["entries"]).values()),{2})
        self.assertFalse(matrix["training_admitted"]);self.assertFalse(ledger["training_admitted"])

    def test_every_world_physically_excludes_target_models(self):
        matrix=json.loads((BASE/"matrix.json").read_text())
        for run in matrix["runs"]:
            plan=json.loads(Path(run["plan_path"]).read_text());names={m.get("name") for m in ET.parse(Path(run["plan_path"]).parent/"world.sdf").getroot().find("world").findall("model")}
            self.assertTrue(set(plan["removed_target_models"]).isdisjoint(names))
            self.assertEqual(plan["annotation_mode"],"full_2d")
            self.assertTrue(plan["diagnostic_require_no_targets"])

if __name__=="__main__":unittest.main()
